"""
DNA Embedding Inversion Attack - Training Entry Point
"""

from __future__ import annotations

import os

# Disable tokenizer parallelism to avoid deadlocks when using multiple workers in DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil

from omegaconf import OmegaConf, DictConfig
import torch
from torch.utils.data import DataLoader
import hydra
import logging

from src.utils import (
    set_determinism,
    maybe_init_wandb,
    log_factory,
    save_json,
    dynamic_import_class,
    variable_length_collate,
)
from src.data import (
    load_split_embeddings,
    load_multi_split_embeddings,
    create_dataset,
    create_multi_dataset,
    max_target_length,
)
from src.train import fit, evaluate
from src.tokenizers import CharacterTokenizer, HuggingFaceTokenizer


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: D401
    """Execute the DNA embedding inversion pipeline.

    This function orchestrates the entire workflow, from configuration loading
    and determinism setup to data loading, model training, and artifact
    persistence. It adheres to a strict, fail-fast philosophy.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra configuration object, composed from YAML files and command-line
        overrides. It contains all settings for the run.

    """
    logger = logging.getLogger(__name__)
    logger.info("Loaded config:\n" + OmegaConf.to_yaml(cfg))

    # 1. Determinism and environment setup
    set_determinism(cfg.train.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.train.device)

    # 2. Validate and get mode from model config
    mode = cfg.model.mode
    assert mode in [
        "per_token",
        "mean",
    ], f"Invalid mode: {mode}. Must be 'per_token' or 'mean'"
    logger.info(f"Training mode: {mode}")

    # 3. Check for existing run
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if os.path.exists(os.path.join(output_dir, "model.pt")):
        logger.info(f"Model already exists in {output_dir}. Skipping training.")
        return

    # 3b. Setup Tokenizer
    tokenizer_cfg = cfg.data.tokenizer

    if tokenizer_cfg.type == "char":
        tokenizer = CharacterTokenizer()
    elif tokenizer_cfg.type == "huggingface":
        tokenizer = HuggingFaceTokenizer(tokenizer_cfg.model_name)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_cfg.type}")
    logger.info(f"Initialized tokenizer: {tokenizer_cfg.type}, vocab_size={tokenizer.vocab_size}")

    # 4. Dynamically import model class from configured model file
    model_file_path = os.path.join(
        os.path.dirname(__file__),
        "src",
        cfg.model.model_file_name,
    )

    # Determine model class name based on model type
    model_type = cfg.model.model_type
    if model_type == "encoder":
        model_class_name = "EncoderReconstructor"
    elif model_type == "decoder":
        model_class_name = "DecoderReconstructor"
    elif model_type == "knn":
        model_class_name = "KNNReconstructor"
    elif model_type == "resnet":
        model_class_name = "ResNetReconstructor"
    elif model_type == "query_decoder":
        model_class_name = "QueryDecoderReconstructor"
    elif model_type == "mask_predict":
        model_class_name = "MaskPredictReconstructor"

    elif mode == "mean":
        model_class_name = "SequenceMeanReconstructor"
    else:
        model_class_name = "SequenceReconstructor"

    ModelClass = dynamic_import_class(model_file_path, model_class_name)

    # 5. Load data with HDF5 for true lazy loading
    is_multi = cfg.data.get("multi", False)
    if is_multi:
        assert mode == "mean", (
            "Multi-length training is currently only supported in 'mean' mode "
            "(model input is a fixed-size mean embedding regardless of sequence length)."
        )
        data_dict, counts_dict, train_stats = load_multi_split_embeddings(cfg.data)
        logger.info(
            f"Multi-length mode: {len(cfg.data.seq_lengths)} sequence lengths "
            f"({list(cfg.data.seq_lengths)})"
        )
    else:
        data_dict, counts_dict, train_stats = load_split_embeddings(cfg.data)
    logger.info(f"Loaded - train: {counts_dict['train']}, val: {counts_dict['val']}")

    data_is_mean = cfg.data.get("mean", False)

    # Log training set normalization statistics (used for all splits to avoid data leakage)
    if train_stats:
        logger.info(
            f"Training set embedding stats (applied to all splits) - "
            f"min: {train_stats['min']:.4f}, max: {train_stats['max']:.4f}, "
            f"mean: {train_stats['mean']:.4f}, std: {train_stats['std']:.4f}"
        )

    # 6. Build DataLoaders with lazy-loaded datasets
    # All datasets use training set statistics for normalization to prevent data leakage
    loaders = {}
    for split in ["train", "val"]:
        max_samples = None
        if split == "val":
            max_samples = cfg.train.max_val_samples

        if is_multi:
            # Train shards may add extra per-length data (train_seq_lengths);
            # val keeps one file per canonical length (seq_lengths).
            split_seq_lengths = (
                list(cfg.data.train_seq_lengths)
                if split == "train"
                else list(cfg.data.seq_lengths)
            )
            dataset = create_multi_dataset(
                data_files=data_dict[split],
                seq_lengths=split_seq_lengths,
                mode=mode,
                tokenizer=tokenizer,
                embedding_dim=cfg.data.embedding_dim,
                normalization_stats=train_stats if train_stats else None,
                normalization_method=cfg.data.normalization_method,
                data_is_mean=data_is_mean,
                subset_fraction=cfg.data.subset_fraction,
                max_samples=max_samples,
            )
        else:
            dataset = create_dataset(
                data_dict[split],
                mode,
                tokenizer,
                cfg.data.embedding_dim,
                cfg.data.seq_length,
                normalization_stats=train_stats if train_stats else None,
                normalization_method=cfg.data.normalization_method,
                data_is_mean=data_is_mean,
                subset_fraction=cfg.data.subset_fraction,
                max_samples=max_samples,
            )

        use_workers = cfg.optim.num_workers > 0
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.optim.batch_size,
            shuffle=(split == "train" and cfg.data.shuffle),
            num_workers=cfg.optim.num_workers,
            pin_memory=True,
            persistent_workers=use_workers,
            collate_fn=variable_length_collate,
        )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    logger.info(f"DataLoaders created - train: {len(train_loader)}, val: {len(val_loader)} batches")

    # 7. Instantiate model, optimizer, and loss function
    # Calculate effective sequence length (number of tokens). One extra slot is
    # reserved for the EOS token that the dataset appends to every target.
    if tokenizer_cfg.type == "huggingface":
        # Size the decoder from the data it is actually trained and validated on.
        # A probe sequence cannot do this job: the previous "N" * seq_length probe
        # was the one string the pipeline never contains (prepare_hg38.py drops
        # every N-containing chunk), and since neither DNA tokenizer has merges
        # for 'N' it shattered into one token per nucleotide and over-sized the
        # decoder ~5x -- 123 slots where 30 (DNABERT-2) and 21 (NTv2) suffice.
        # val is scanned alongside train because fit() computes the same loss on it.
        if is_multi:
            scan_splits = [
                (data_dict["train"], list(cfg.data.train_seq_lengths)),
                (data_dict["val"], list(cfg.data.seq_lengths)),
            ]
        else:
            scan_splits = [
                ([data_dict[split]], [cfg.data.seq_length]) for split in ["train", "val"]
            ]
        effective_seq_length = max(
            max_target_length(files, seq_lengths, tokenizer)
            for files, seq_lengths in scan_splits
        )
        logger.info(
            f"Sized effective_seq_length from the data: {cfg.data.seq_length} nt -> "
            f"{effective_seq_length} tokens (longest observed target, includes EOS)"
        )
    else:
        # The character tokenizer emits exactly one token per nucleotide, so the
        # longest target is known without scanning: seq_length tokens + EOS.
        effective_seq_length = cfg.data.seq_length + 1

    # Build model kwargs based on model type and mode
    if model_type == "encoder" or model_type == "decoder":
        # Encoder/Decoder calculates output_dim internally from seq_length * output_dim
        model_kwargs = {
            "input_dim": cfg.data.embedding_dim,
            "hidden_dims": cfg.model.hidden_dims,
            "mode": mode,
            "seq_length": effective_seq_length,
            "output_dim": tokenizer.vocab_size,
            "d_model": cfg.model.d_model,
            "nhead": cfg.model.nhead,
            "num_layers": cfg.model.num_layers,
            "dim_feedforward": cfg.model.dim_feedforward,
            "dropout": cfg.model.dropout,
        }
    elif model_type == "knn":
        model_kwargs = {
            "input_dim": cfg.data.embedding_dim,
            "output_dim": tokenizer.vocab_size,
            "k": cfg.model.k,
        }

    elif model_type == "resnet":
        model_kwargs = {
            "input_dim": cfg.data.embedding_dim,
            "mode": mode,
            "seq_length": effective_seq_length,
            "output_dim": tokenizer.vocab_size,
            "d_model": cfg.model.d_model,
            "n_blocks": cfg.model.n_blocks,
            "kernel_size": cfg.model.kernel_size,
            "dropout": cfg.model.dropout,
        }

    elif model_type == "query_decoder":
        model_kwargs = {
            "input_dim": cfg.data.embedding_dim,
            "mode": mode,
            "seq_length": effective_seq_length,
            "output_dim": tokenizer.vocab_size,
            "d_model": cfg.model.d_model,
            "nhead": cfg.model.nhead,
            "num_layers": cfg.model.num_layers,
            "dim_feedforward": cfg.model.dim_feedforward,
            "dropout": cfg.model.dropout,
            "n_context_tokens": cfg.model.n_context_tokens,
            "aux_length_loss_weight": cfg.model.aux_length_loss_weight,
        }

    elif model_type == "mask_predict":
        model_kwargs = {
            "input_dim": cfg.data.embedding_dim,
            "mode": mode,
            "seq_length": effective_seq_length,
            "output_dim": tokenizer.vocab_size,
            "d_model": cfg.model.d_model,
            "nhead": cfg.model.nhead,
            "num_layers": cfg.model.num_layers,
            "dim_feedforward": cfg.model.dim_feedforward,
            "dropout": cfg.model.dropout,
            "n_context_tokens": cfg.model.n_context_tokens,
            "aux_length_loss_weight": cfg.model.aux_length_loss_weight,
            "num_iterations": cfg.model.num_iterations,
        }

    elif mode == "mean":
        # Mean mode MLP calculates output_dim from seq_length * output_dim
        model_kwargs = {
            "input_dim": cfg.data.embedding_dim,
            "hidden_dims": cfg.model.hidden_dims,
            "seq_length": effective_seq_length,
            "output_dim": tokenizer.vocab_size,
            "dropout": cfg.model.dropout,
        }
    else:
        # Per-nucleotide mode uses output_dim as output_dim
        model_kwargs = {
            "input_dim": cfg.data.embedding_dim,
            "hidden_dims": cfg.model.hidden_dims,
            "output_dim": tokenizer.vocab_size,
            "dropout": cfg.model.dropout,
        }

    model = ModelClass(**model_kwargs).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    if model_type == "knn":
        logger.info(f"Fitting {model_type} model with training data...")
        model.fit(train_loader)

        logger.info(f"Evaluating {model_type} model...")
        # Note: evaluate needs update to handle tokenizer if it uses decode_onehot_to_sequence
        val_metrics = evaluate(model, val_loader, loss_fn, device, tokenizer=tokenizer)
        results = {
            "best_val_loss": val_metrics["loss"],
            "stopped_early": False,
            "final_epoch": 0,
        }

        # Initialize wandb for KNN/Chroma if needed (for consistency)
        wandb_run = maybe_init_wandb(cfg)
    else:

        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
            eps=cfg.optim.eps,
        )

        # OneCycleLR is a fixed-horizon schedule: total_steps is sized to the
        # full epoch budget and the LR only reaches its convergence-friendly
        # minimum at the final step. Early stopping would cut it off mid-anneal
        # (near peak LR), so the two are mutually exclusive. Fail fast on the
        # contradiction rather than silently training an un-annealed model.
        assert not (cfg.optim.use_scheduler and cfg.train.early_stopping.enabled), (
            "optim.use_scheduler=true (OneCycleLR) requires train.early_stopping.enabled=false; "
            "early stopping interrupts the LR anneal. Disable one of them."
        )

        # Create learning rate scheduler if enabled
        scheduler = None
        if cfg.optim.use_scheduler:
            total_steps = len(train_loader) * cfg.optim.epochs
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.optim.scheduler.max_lr,
                total_steps=total_steps,
                pct_start=cfg.optim.scheduler.pct_start,
                anneal_strategy=cfg.optim.scheduler.anneal_strategy,
                div_factor=cfg.optim.scheduler.div_factor,
                final_div_factor=cfg.optim.scheduler.final_div_factor,
            )
            logger.info(
                f"OneCycleLR scheduler enabled - max_lr: {cfg.optim.scheduler.max_lr}, "
                f"total_steps: {total_steps}, pct_start: {cfg.optim.scheduler.pct_start}"
            )

        # 8. Set up logging and execute training
        wandb_run = maybe_init_wandb(cfg)
        log_fn = log_factory(logger, wandb_run)

        results = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=cfg.optim.epochs,
            device=device,
            log_fn=log_fn,
            tokenizer=tokenizer,
            interval=cfg.train.log_interval,
            scheduler=scheduler,
            early_stopping_enabled=cfg.train.early_stopping.enabled,
            early_stopping_patience=cfg.train.early_stopping.patience,
            early_stopping_min_delta=cfg.train.early_stopping.min_delta,
            checkpoint_path=os.path.join(output_dir, "checkpoint.pt"),
            grad_clip_norm=cfg.optim.grad_clip_norm,
            skip_frac_threshold=cfg.train.nan_backoff.skip_frac_threshold,
            lr_backoff=cfg.train.nan_backoff.lr_backoff,
        )

    logger.info(
        "Final results: " + ", ".join(f"{k}={v:.4f}" for k, v in results.items())
    )  # 9. Persist artifacts

    # Copy model file to output directory for reproducibility
    model_source = os.path.join(os.path.dirname(__file__), "src", cfg.model.model_file_name)
    model_dest = os.path.join(output_dir, "model.py")
    shutil.copy2(model_source, model_dest)
    logger.info(f"Copied {cfg.model.model_file_name} to {model_dest}")

    model_path = os.path.join(output_dir, "model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "results": results,
            "input_dim": cfg.data.embedding_dim,
            "output_dim": tokenizer.vocab_size,
            "mode": mode,
            "effective_seq_length": effective_seq_length,
            "tokenizer_type": tokenizer_cfg.type,
            "tokenizer_model": tokenizer_cfg.get("model_name", None),
        },
        model_path,
    )
    logger.info(f"Saved model checkpoint to {model_path}")

    # Training finished and the final model is persisted; the resumable per-epoch
    # checkpoint is now redundant, so drop it to reclaim disk.
    resume_ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    if os.path.exists(resume_ckpt_path):
        os.remove(resume_ckpt_path)
        logger.info(f"Removed resume checkpoint {resume_ckpt_path}")

    # Also store results separately for easy inspection
    save_json(results, os.path.join(output_dir, "results.json"))
    logger.info(f"Saved results to {output_dir}")

    if wandb_run:
        wandb_run.summary.update(results)
        wandb_run.finish()


if __name__ == "__main__":  # pragma: no cover
    main()  # type: ignore
