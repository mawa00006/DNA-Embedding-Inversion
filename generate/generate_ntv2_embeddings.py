"""Generate DNA embeddings from an NTV2 model using sequences from a file.

This script extracts embeddings from NTV2 model for sequences provided in a file.
The output is HDF5 files for train, val, and test with datasets 'sequences' and 'embeddings'.

If mean=false (default): embeddings contains per-nucleotide embedding matrices
of shape [seq_length x embedding_dim] for each sequence.

If mean=true: embeddings contains mean-pooled embeddings of shape [embedding_dim]
for each sequence.

Example usage:
    python generate_ntv2_embeddings.py input_path=data.csv checkpoint=InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
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
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    NUCLEOTIDES,
    file_sha256,
    set_determinism,
    load_sequences_from_file,
    load_identification_csv,
    update_yaml_keys,
)
from src.generation_io import mean_pool_in_batches, run_generation


def _write_identification_metadata(
    h5_file: h5py.File,
    split_indices: np.ndarray,
    individual_ids: List[str],
    locus_ids: List[int],
    haplotypes: List[int],
) -> None:
    """Write per-row identification labels (individual_id/locus_id/haplotype) into an H5 file.

    Datasets are aligned 1:1 with ``sequences`` and ``embeddings``.
    """
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


def embed_batch_ntv2(
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    sequences: List[str],
    seq_length: int,
    device: str,
) -> List[np.ndarray]:
    """Tokenize a batch of DNA sequences with the NTV2 tokenizer and extract per-token embeddings.

    Tokenizes without special tokens so embeddings align 1:1 with sequence tokens.
    Returns a list of NumPy arrays, each of shape (num_tokens_i, embedding_dim).
    """
    assert sequences, "Empty batch"
    enc = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        add_special_tokens=False,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )

    # Last hidden state: [batch, seq_len, dim]
    last = outputs.hidden_states[-1]

    mask_np = attention_mask.bool().cpu().numpy()
    embs_np = last.detach().float().cpu().numpy()

    results = []
    for i in range(len(sequences)):
        seq_emb = embs_np[i][mask_np[i]]
        results.append(seq_emb)

    return results


def _make_embed_chunk_ntv2(checkpoint: str, device: str, seq_length: int, batch_size: int):
    """Load the model and return ``embed_chunk``: sequences -> mean-pooled ``[D]`` arrays.

    Sub-batches by ``batch_size`` and reuses ``embed_batch_ntv2``; the model is loaded once.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading NTV2 tokenizer/model from: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint, trust_remote_code=True)
    model_max = tokenizer.model_max_length
    assert seq_length <= model_max, (
        f"Requested seq_length={seq_length} > tokenizer.model_max_length={model_max}"
    )
    model.to(device)
    model.eval()

    def embed_chunk(seqs: List[str]) -> List[np.ndarray]:
        return mean_pool_in_batches(
            seqs, batch_size, lambda b: embed_batch_ntv2(tokenizer, model, b, seq_length, device)
        )

    return embed_chunk


def generate_embeddings_ntv2(
    sequences: List[str], checkpoint: str, seq_length: int, device: str, batch_size: int = 32, mean_pool: bool = False
) -> List[np.ndarray]:
    """Load tokenizer/model from `checkpoint` and compute embeddings for sequences.

    When ``mean_pool=True``, each per-sequence ``[num_tokens, D]`` embedding is
    reduced to ``[D]`` before being appended, bounding host RAM at ``N*D`` instead
    of ``N*num_tokens*D``.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading NTV2 tokenizer/model from: {checkpoint}")
    logger.info(f"Using batch_size={batch_size}, mean_pool={mean_pool}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint, trust_remote_code=True)

    # Sanity: ensure seq_length fits tokenizer max length
    model_max = tokenizer.model_max_length
    assert seq_length <= model_max, f"Requested seq_length={seq_length} > tokenizer.model_max_length={model_max}"

    model.to(device)
    model.eval()

    embeddings = []

    n = len(sequences)
    log_every_batches = max(1, 100 // batch_size)

    for batch_idx, start in enumerate(range(0, n, batch_size)):
        batch = sequences[start : start + batch_size]
        batch_embs = embed_batch_ntv2(tokenizer, model, batch, seq_length, device)
        if mean_pool:
            batch_embs = [emb.mean(axis=0) for emb in batch_embs]
        embeddings.extend(batch_embs)

        done = min(start + batch_size, n)
        if (batch_idx + 1) % log_every_batches == 0 or done == n:
            logger.info(f"Generated embeddings for {done}/{n} sequences")

    return embeddings


@hydra.main(config_path="../conf", config_name="generate/ntv2", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint: generate sequences and extract NTV2 embeddings."""
    logger = logging.getLogger(__name__)
    logger.info("Generating DNA sequences and NTV2 embeddings...")

    set_determinism(int(cfg.seed))

    # Load sequences from file or generate random ones
    input_path = hy_utils.to_absolute_path(cfg.input_path)
    identification_mode = cfg.identification_mode

    if identification_mode:
        logger.info(f"Loading identification CSV (preserving order, no dedup): {input_path}")
        id_data = load_identification_csv(input_path)
        sequences = id_data["sequences"]
        individual_ids = id_data["individual_ids"]
        locus_ids = id_data["locus_ids"]
        haplotypes = id_data["haplotypes"]
        # Validate sequence length consistency.
        assert all(len(s) == cfg.seq_length for s in sequences), (
            "Identification CSV contains sequences not matching cfg.seq_length"
        )
        logger.info(
            f"Loaded {len(sequences)} identification rows "
            f"({len(set(individual_ids))} individuals, {len(set(locus_ids))} loci)"
        )
    else:
        logger.info(f"Loading sequences from file: {input_path}")
        logger.info(f"Max sequences (num_sequences): {cfg.num_sequences}")
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

    checkpoint = cfg.checkpoint
    device = cfg.device

    logger.info(f"Using checkpoint: {checkpoint}")
    logger.info(f"Using device: {device}")

    batch_size = cfg.batch_size

    # Resumable, incremental generation for the mean-split and identification
    # paths (fast-exit on complete, resume on partial). Returns False only for the
    # legacy all-at-once per-token (mean=false) case.
    if run_generation(
        cfg,
        sequences=sequences,
        individual_ids=individual_ids,
        locus_ids=locus_ids,
        haplotypes=haplotypes,
        make_embed_chunk=lambda: _make_embed_chunk_ntv2(
            checkpoint, device, int(cfg.seq_length), int(batch_size)
        ),
        logger=logger,
    ):
        return

    use_mean = bool(cfg.mean)
    embeddings = generate_embeddings_ntv2(
        sequences, checkpoint, int(cfg.seq_length), device, batch_size=batch_size, mean_pool=use_mean
    )

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

    indices = np.arange(n)
    if not identification_mode:
        np.random.shuffle(indices)

    train_idx = indices[:train_n]
    val_idx = indices[train_n : train_n + val_n]
    test_idx = indices[train_n + val_n :]

    logger.info(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

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

                emb_dataset = f.create_dataset(
                    "embeddings", (len(mean_embeddings), embedding_dim), dtype=np.float32
                )
                for i, emb_mean in enumerate(mean_embeddings):
                    emb_dataset[i] = emb_mean

                if identification_mode:
                    _write_identification_metadata(
                        f, split_indices, individual_ids, locus_ids, haplotypes
                    )

                f.attrs["embedding_dim"] = embedding_dim
                f.attrs["checkpoint"] = checkpoint

                # STORE GLOBAL STATS
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
                dt_str = h5py.string_dtype(encoding="utf-8")
                seq_dataset = f.create_dataset("sequences", (len(split_sequences),), dtype=dt_str)
                for i, seq in enumerate(split_sequences):
                    seq_dataset[i] = seq

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

                f.attrs["embedding_dim"] = embedding_dim
                f.attrs["checkpoint"] = checkpoint

                # STORE GLOBAL STATS
                f.attrs["emb_min"] = global_min
                f.attrs["emb_max"] = global_max
                f.attrs["emb_mean"] = global_mean
                f.attrs["emb_std"] = global_std

        logger.info(f"Saved {len(split_sequences)} {split_name} sequences to {output_path}")
        logger.info(f"Number of embeddings: {len(split_embeddings)}")

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
