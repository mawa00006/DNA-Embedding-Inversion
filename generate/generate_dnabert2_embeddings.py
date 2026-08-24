"""Generate DNA embeddings from a DNABERT-2 model using sequences from a file.

This script extracts embeddings from DNABERT-2 model for sequences provided in a file.
The output is HDF5 files for train, val, and test with datasets 'sequences' and 'embeddings'.

If mean=false (default): embeddings contains per-nucleotide embedding matrices
of shape [seq_length x embedding_dim] for each sequence.

If mean=true: embeddings contains mean-pooled embeddings of shape [embedding_dim]
for each sequence.

Example usage:
    python generate_dnabert2_embeddings.py input_path=data.csv num_sequences=1000 seq_length=50
    python generate_dnabert2_embeddings.py input_path=data.csv num_sequences=1000 seq_length=50 mean=true
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import h5py
from omegaconf import DictConfig
import hydra
import hydra.utils as hy_utils
from transformers import AutoTokenizer, AutoModel, BertConfig

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    file_sha256,
    set_determinism,
    load_sequences_from_file,
    load_identification_csv,
    update_yaml_keys,
)
from src.generation_io import run_generation


def _write_identification_metadata(
    h5_file: h5py.File,
    split_indices: np.ndarray,
    individual_ids: List[str],
    locus_ids: List[int],
    haplotypes: List[int],
) -> None:
    """Write per-row identification labels into an H5 file (aligned with sequences/embeddings)."""
    dt_str = h5py.string_dtype(encoding="utf-8")
    n = len(split_indices)
    ind_ds = h5_file.create_dataset("individual_ids", (n,), dtype=dt_str)
    loc_ds = h5_file.create_dataset("locus_ids", (n,), dtype=np.int32)
    hap_ds = h5_file.create_dataset("haplotypes", (n,), dtype=np.int8)
    for i, src_idx in enumerate(split_indices):
        ind_ds[i] = individual_ids[src_idx]
        loc_ds[i] = locus_ids[src_idx]
        hap_ds[i] = haplotypes[src_idx]
    h5_file.attrs["identification_mode"] = True


def _patch_triton_flash_attn():
    """Patch DNABERT-2's cached flash_attn_triton.py for Triton ≥ 3.x compatibility.

    The bundled file uses deprecated ``tl.dot(..., trans_a/trans_b=True)`` kwargs
    that were removed in newer Triton releases.  This replaces them with the
    equivalent ``tl.dot(tl.trans(x), y)`` / ``tl.dot(x, tl.trans(y))`` calls.
    """
    for name, mod in list(sys.modules.items()):
        if "flash_attn_triton" not in name or not hasattr(mod, "__file__") or mod.__file__ is None:
            continue
        fpath = Path(mod.__file__)
        if not fpath.exists():
            continue
        src = fpath.read_text()
        if "trans_b=True" not in src and "trans_a=True" not in src:
            continue
        patched = src
        patched = patched.replace("tl.dot(q, k, trans_b=True)", "tl.dot(q, tl.trans(k))")
        patched = patched.replace("tl.dot(do, v, trans_b=True)", "tl.dot(do, tl.trans(v))")
        patched = patched.replace(
            "tl.dot(p.to(do.dtype), do, trans_a=True)",
            "tl.dot(tl.trans(p.to(do.dtype)), do)",
        )
        patched = patched.replace("tl.dot(ds, q, trans_a=True)", "tl.dot(tl.trans(ds), q)")
        fpath.write_text(patched)
        logging.getLogger(__name__).info("Patched %s for Triton 3.x compatibility", fpath)


def embed_sequence_dnabert(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    sequence: str,
    device: str,
) -> np.ndarray:
    """Extract embeddings for a DNA sequence from DNABERT-2 model.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        DNABERT-2 tokenizer.
    model : AutoModel
        DNABERT-2 model instance.
    sequence : str
        DNA sequence string.
    device : str
        Device to place input tensors on.

    Returns
    -------
    np.ndarray
        Per-nucleotide embedding of shape (num_tokens, embedding_dim).
    """
    # Tokenize without special tokens so embeddings align 1:1 with sequence tokens
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        device
    )

    with torch.no_grad():
        hidden_states = model(inputs)[0]  # [1, num_tokens, embedding_dim]

    token_embs = hidden_states[0].detach().float().cpu().numpy()  # [num_tokens, embedding_dim]

    return token_embs


def _load_dnabert(checkpoint: str, device: str):
    """Load the DNABERT-2 tokenizer + model (with the Triton flash-attn patch)."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading DNABERT-2 tokenizer and model from: {checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    config = BertConfig.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint, config=config, trust_remote_code=True)

    # Patch & reload the Triton flash-attn module so the first forward pass
    # compiles the fixed kernels (DNABERT-2 ships trans_a/trans_b that Triton ≥ 3 removed).
    _patch_triton_flash_attn()
    import importlib
    for mod_name in list(sys.modules):
        if "flash_attn_triton" in mod_name:
            importlib.reload(sys.modules[mod_name])

    model.to(device)
    model.eval()
    logger.info(f"DNABERT-2 model {checkpoint} loaded successfully")
    return tokenizer, model


def _make_embed_chunk_dnabert(checkpoint: str, device: str):
    """Load the model and return ``embed_chunk``: sequences -> mean-pooled ``[D]`` arrays.

    Reuses ``embed_sequence_dnabert``; the model is loaded once.
    """
    tokenizer, model = _load_dnabert(checkpoint, device)

    def embed_chunk(seqs: List[str]) -> List[np.ndarray]:
        return [embed_sequence_dnabert(tokenizer, model, s, device).mean(axis=0) for s in seqs]

    return embed_chunk


def generate_embeddings_dnabert(
    sequences: List[str], checkpoint: str, device: str, mean_pool: bool = False
) -> List[np.ndarray]:
    """Generate DNABERT-2 embeddings for multiple sequences.

    When ``mean_pool=True``, each per-sequence ``[num_tokens, D]`` embedding is
    reduced to ``[D]`` before being appended to the result list. This bounds the
    host RAM at ``N*D`` instead of ``N*num_tokens*D``.
    """
    logger = logging.getLogger(__name__)
    tokenizer, model = _load_dnabert(checkpoint, device)
    logger.info(f"mean_pool={mean_pool}")

    embeddings = []
    for i, seq in enumerate(sequences):
        emb = embed_sequence_dnabert(tokenizer, model, seq, device)
        if mean_pool:
            emb = emb.mean(axis=0)
        embeddings.append(emb)

        if (i + 1) % 100 == 0:
            logger.info(f"Generated embeddings for {i + 1}/{len(sequences)} sequences")

    return embeddings


@hydra.main(config_path="../conf", config_name="generate/dnabert2", version_base=None)
def main(cfg: DictConfig) -> None:
    """Generate DNA sequences and DNABERT-2 embeddings using Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing data generation parameters.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating DNA sequences and DNABERT-2 embeddings...")

    set_determinism(int(cfg.seed))

    # Load sequences from file or generate random ones
    assert cfg.input_path is not None, "input_path must be provided in the configuration."
    input_path = cfg.input_path

    input_path = hy_utils.to_absolute_path(input_path)
    logger.info(f"Loading sequences from file: {input_path}")
    identification_mode = cfg.identification_mode

    if identification_mode:
        logger.info(f"Loading identification CSV (preserving order, no dedup): {input_path}")
        id_data = load_identification_csv(input_path)
        sequences = id_data["sequences"]
        individual_ids = id_data["individual_ids"]
        locus_ids = id_data["locus_ids"]
        haplotypes = id_data["haplotypes"]
        assert all(len(s) == cfg.seq_length for s in sequences), (
            "Identification CSV contains sequences not matching cfg.seq_length"
        )
        logger.info(
            f"Loaded {len(sequences)} identification rows "
            f"({len(set(individual_ids))} individuals, {len(set(locus_ids))} loci)"
        )
    else:
        logger.info(f"Max sequences (num_sequences): {cfg.num_sequences}")
        # Note: seq_length here acts as a filter/truncator for loaded sequences
        logger.info(f"Max sequence length (seq_length): {cfg.seq_length}")
        sequences = load_sequences_from_file(
            input_path, max_length=cfg.seq_length, max_sequences=cfg.num_sequences
        )
        individual_ids = None
        locus_ids = None
        haplotypes = None

        if bool(cfg.get("deduplicate_sequences", True)):
            original_count = len(sequences)
            sequences = list(dict.fromkeys(sequences))
            duplicates_removed = original_count - len(sequences)
            if duplicates_removed > 0:
                logger.warning(f"Removed {duplicates_removed} duplicate sequences")

        logger.info(
            f"Loaded {len(sequences)} sequence windows from file "
            f"(limited to {cfg.num_sequences}, target length {cfg.seq_length})"
        )

    # Extract embeddings
    checkpoint = cfg.checkpoint
    device = cfg.device

    logger.info(f"\n")
    logger.info(f"Using checkpoint: {checkpoint}")
    logger.info(f"Using device: {device}")
    logger.info(f"{'=' * 80}")

    # Resumable, incremental generation for the mean-split and identification
    # paths (handles fast-exit on complete output and resume on partial). Returns
    # False only for the legacy all-at-once per-token (mean=false) case.
    if run_generation(
        cfg,
        sequences=sequences,
        individual_ids=individual_ids,
        locus_ids=locus_ids,
        haplotypes=haplotypes,
        make_embed_chunk=lambda: _make_embed_chunk_dnabert(checkpoint, device),
        logger=logger,
    ):
        return

    use_mean = bool(cfg.mean)
    embeddings = generate_embeddings_dnabert(sequences, checkpoint, device, mean_pool=use_mean)

    # Split data into train, val, test. The mean and identification paths are
    # handled resumably in run_generation above, so this legacy all-at-once path is
    # only ever reached for the per-token (mean=false) track -- always a full split.
    train_split = cfg.train_split
    val_split = cfg.val_split
    test_split = 1.0 - train_split - val_split
    assert (
        train_split + val_split + test_split > 0.99
    ), "Split ratios must sum to approximately 1.0"

    n = len(sequences)
    train_n = int(n * train_split)
    val_n = int(n * val_split)

    # Create indices and split
    indices = np.arange(n)
    if not identification_mode:
        np.random.shuffle(indices)

    train_idx = indices[:train_n]
    val_idx = indices[train_n : train_n + val_n]
    test_idx = indices[train_n + val_n :]

    logger.info(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Save each split to HDF5
    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    # Note: when use_mean=True, each `embeddings[i]` is already shape [D] (pooled
    # inside the generation loop to bound host RAM); otherwise it is [num_tokens, D].

    # Compute global statistics from TRAINING SET only
    train_sequences = [sequences[i] for i in splits["train"]]
    train_embeddings = [embeddings[i] for i in splits["train"]]

    if use_mean:
        logger.info("Computing global statistics from training set (mean-pooled embeddings)...")
        all_train_values = np.concatenate([emb.ravel() for emb in train_embeddings])
        embedding_dim = train_embeddings[0].shape[-1]
    else:
        logger.info(
            "Computing global statistics from training set (per-nucleotide embeddings)..."
        )
        all_train_values = np.concatenate([emb.flatten() for emb in train_embeddings])
        embedding_dim = train_embeddings[0].shape[1]

    # Compute global scalar statistics (not per-dimension)
    global_min = float(np.min(all_train_values))
    global_max = float(np.max(all_train_values))
    global_mean = float(np.mean(all_train_values))
    global_std = float(np.std(all_train_values))

    logger.info("Global statistics from training set:")
    logger.info(f"  min:  {global_min:.6f}")
    logger.info(f"  max:  {global_max:.6f}")
    logger.info(f"  mean: {global_mean:.6f}")
    logger.info(f"  std:  {global_std:.6f}")

    hashes = {}

    for split_name, split_indices in splits.items():
        split_sequences = [sequences[i] for i in split_indices]
        split_embeddings = [embeddings[i] for i in split_indices]

        output_path = hy_utils.to_absolute_path(cfg[f"{split_name}_output_path"])

        if use_mean:
            logger.info(f"Saving mean-pooled embeddings for {split_name} split")
            # Embeddings were mean-pooled inside the generation loop; each is [D].
            mean_embeddings = split_embeddings

            with h5py.File(output_path, "w") as f:
                dt_str = h5py.string_dtype(encoding="utf-8")
                seq_dataset = f.create_dataset("sequences", (len(split_sequences),), dtype=dt_str)
                for i, seq in enumerate(split_sequences):
                    seq_dataset[i] = seq

                # Create fixed-size float dataset for mean embeddings
                emb_dataset = f.create_dataset(
                    "embeddings", (len(mean_embeddings), embedding_dim), dtype=np.float32
                )
                for i, emb_mean in enumerate(mean_embeddings):
                    emb_dataset[i] = emb_mean

                if identification_mode:
                    _write_identification_metadata(
                        f, split_indices, individual_ids, locus_ids, haplotypes
                    )

                # Store metadata as attributes
                f.attrs["embedding_dim"] = embedding_dim
                f.attrs["checkpoint"] = checkpoint

                # Store GLOBAL (training) statistics So loader uses correct normalization
                f.attrs["emb_min"] = global_min
                f.attrs["emb_max"] = global_max
                f.attrs["emb_mean"] = global_mean
                f.attrs["emb_std"] = global_std

            logger.info(
                f"Saved {len(split_sequences)} {split_name} sequences with mean embeddings to {output_path}"
            )
        else:
            logger.info(f"Saving per-nucleotide embeddings for {split_name} split")

            with h5py.File(output_path, "w") as f:
                # Create variable-length string dataset for sequences
                dt_str = h5py.string_dtype(encoding="utf-8")
                seq_dataset = f.create_dataset("sequences", (len(split_sequences),), dtype=dt_str)
                for i, seq in enumerate(split_sequences):
                    seq_dataset[i] = seq

                # Create variable-length float dataset for embeddings
                dt_vlen = h5py.vlen_dtype(np.float32)
                emb_dataset = f.create_dataset(
                    "embeddings", (len(split_embeddings),), dtype=dt_vlen
                )
                for i, emb in enumerate(split_embeddings):
                    emb_dataset[i] = emb.flatten()

                if identification_mode:
                    _write_identification_metadata(
                        f, split_indices, individual_ids, locus_ids, haplotypes
                    )

                # Store shape metadata as attributes
                f.attrs["embedding_dim"] = embedding_dim
                f.attrs["checkpoint"] = checkpoint

                # Store GLOBAL (training) statistics
                f.attrs["emb_min"] = global_min
                f.attrs["emb_max"] = global_max
                f.attrs["emb_mean"] = global_mean
                f.attrs["emb_std"] = global_std

            logger.info(f"Saved {len(split_sequences)} {split_name} sequences to {output_path}")
            logger.info(f"Number of embeddings: {len(split_embeddings)}")

        # Compute SHA256
        sha256_hash = file_sha256(output_path)
        hashes[split_name] = sha256_hash
        logger.info(f"SHA256 hash of {output_path}: {sha256_hash}")

    if "update_config" in cfg and cfg.update_config:
        config_path = hy_utils.to_absolute_path(cfg.update_config)
        logger.info(f"Updating config file {config_path}...")
        updates = {
            "test_sha256": hashes["test"],
            "skip_sha256_check": False,
            "test_csv": cfg.test_output_path,
            "embedding_dim": embedding_dim,
            "seq_length": cfg.seq_length,
        }
        updates["train_sha256"] = hashes["train"]
        updates["val_sha256"] = hashes["val"]
        updates["train_csv"] = cfg.train_output_path
        updates["val_csv"] = cfg.val_output_path

        update_yaml_keys(str(config_path), updates)
        logger.info("Config file updated.")
    else:
        print(f"Update conf/config.yaml data section:")
        print(f"train_sha256: {hashes['train']}")
        print(f"val_sha256: {hashes['val']}")
        print(f"test_sha256: {hashes['test']}")


if __name__ == "__main__":  # pragma: no cover
    main()  # type: ignore
