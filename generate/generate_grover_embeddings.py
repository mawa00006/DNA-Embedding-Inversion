"""Generate DNA embeddings from GROVER using sequences from a file.

This script extracts embeddings from a GROVER checkpoint (a standard BERT
encoder pretrained on human genome with a learned byte-pair-style vocabulary)
for sequences provided in a file. The output is HDF5 files for train, val, and
test with datasets 'sequences' and 'embeddings'.

Only mean=true is supported: embeddings contains mean-pooled embeddings of shape
[embedding_dim] for each sequence.

Architecture notes:

* GROVER's config.json carries no `auto_map`, so `AutoModel` resolves through
  the library's own registered `model_type: bert` mapping to a bare
  `transformers.BertModel` -- the masked-LM checkpoint minus its MLM head.
  `trust_remote_code` is passed for parity with the other generators but has
  no effect here; no remote python file is fetched.
* The tokenizer is a fast tokenizer (`tokenizer.json`, learned genome
  vocabulary of 609 tokens) that takes a raw ACGT string directly -- no
  k-mer or other preprocessing, unlike DNABERT-1.
* forward() takes `attention_mask` normally and the tokenizer returns one; the
  default `padding_side` is right. Neither needs the workarounds Caduceus
  needs.
* hidden_states has n_layer + 1 entries (embedding output + one per encoder
  layer); the observed length is asserted against config.num_hidden_layers,
  and hidden_states[-1] is asserted equal to last_hidden_state so the selected
  layer is pinned regardless of internal implementation details.
* AutoModel loads the bare backbone against masked-LM weights; missing_keys is
  asserted empty so a mismatch fails loudly instead of silently random-
  initialising part of the model.
* tokenizer_config.json's own `model_max_length` is a real value (512, matching
  config.max_position_embeddings) -- unlike DNABERT-1, no sentinel fallback is
  needed here, but the same defensive helper is used for consistency.

Example usage:
    python generate_grover_embeddings.py input_path=data.csv \
        checkpoint=PoetschLab/GROVER
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import h5py
from omegaconf import DictConfig
import hydra
import hydra.utils as hy_utils
from transformers import AutoConfig, AutoTokenizer, AutoModel

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


# Index into outputs.hidden_states. -1 is the last hidden state, matching the
# other HF generators.
HIDDEN_STATES_OFFSET = 1


def expected_num_hidden_states(config) -> int:
    """Number of entries ``outputs.hidden_states`` will have for this architecture."""
    return int(config.num_hidden_layers) + HIDDEN_STATES_OFFSET


def resolve_layer_index(num_hidden_states: int, layer_index: int) -> int:
    """Normalize ``layer_index`` (which may be negative) to a positive index.

    Kept separate from the forward pass so the resolved value can be logged and
    stamped into the output files without loading the model twice.
    """
    idx = int(layer_index)
    assert -num_hidden_states <= idx < num_hidden_states, (
        f"layer_index={idx} out of range for {num_hidden_states} hidden states"
    )
    return idx % num_hidden_states


def _resolve_model_max(tokenizer, config) -> int:
    """Positional-embedding capacity for this checkpoint.

    ``tokenizer.model_max_length`` is trustworthy for GROVER (its
    tokenizer_config.json sets it explicitly to 512), but is treated
    defensively anyway: if it were an unset sentinel (as for DNABERT-1),
    ``config.max_position_embeddings`` -- the real architecture-level bound --
    is used instead.
    """
    tok_max = int(tokenizer.model_max_length)
    cfg_max = int(config.max_position_embeddings)
    return cfg_max if tok_max > 1_000_000 else min(tok_max, cfg_max)


def embed_batch_grover(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    sequences: List[str],
    max_length: int,
    device: str,
    layer_index: int,
) -> List[np.ndarray]:
    """Tokenize a batch of DNA sequences with the GROVER tokenizer and extract per-token embeddings.

    Tokenizes without special tokens so embeddings align 1:1 with sequence tokens.
    Returns a list of NumPy arrays, each of shape (num_tokens_i, embedding_dim).
    """
    assert sequences, "Empty batch"
    enc = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )

    # Selected hidden state: [batch, seq_len, dim]. ``layer_index`` is already
    # resolved to a positive index by the caller.
    last = outputs.hidden_states[layer_index]

    mask_np = attention_mask.bool().cpu().numpy()
    embs_np = last.detach().float().cpu().numpy()

    results = []
    for i in range(len(sequences)):
        seq_emb = embs_np[i][mask_np[i]]
        results.append(seq_emb)

    return results


def _load_grover(checkpoint: str, revision: Optional[str], device: str):
    """Load the GROVER tokenizer/model and return ``(tokenizer, model, model_max)``."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading GROVER tokenizer/model from: {checkpoint} (revision={revision})")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, revision=revision, trust_remote_code=True
    )
    # AutoModel loads the bare backbone against masked-LM weights, so missing
    # weights would otherwise be silently random-initialised and only logged.
    model, loading_info = AutoModel.from_pretrained(
        checkpoint, revision=revision, trust_remote_code=True, output_loading_info=True
    )
    assert not loading_info["missing_keys"], (
        f"Missing keys loading bare backbone from masked-LM checkpoint: "
        f"{loading_info['missing_keys']}"
    )

    assert tokenizer.pad_token is not None, "GROVER tokenizer has no pad token"
    # BERT's forward masks padding correctly regardless of side, but the
    # generators standardize on right padding; assert rather than assume since
    # the checkpoint's own default is what actually ships.
    assert tokenizer.padding_side == "right", (
        f"Expected right padding, tokenizer defaults to {tokenizer.padding_side!r}"
    )

    model_max = _resolve_model_max(tokenizer, model.config)
    model.to(device)
    model.eval()
    return tokenizer, model, model_max


def _make_embed_chunk_grover(
    checkpoint: str,
    device: str,
    seq_length: int,
    batch_size: int,
    layer_index: int,
    revision: Optional[str] = None,
):
    """Load the model and return ``embed_chunk``: sequences -> mean-pooled ``[D]`` arrays.

    Sub-batches by ``batch_size`` and reuses ``embed_batch_grover``; the model
    is loaded once. Note that ``run_generation`` feeds this closure slices of
    ``gen_chunk_size`` samples, so the effective batch is
    ``min(batch_size, gen_chunk_size)``.
    """
    logger = logging.getLogger(__name__)
    tokenizer, model, model_max = _load_grover(checkpoint, revision, device)

    assert seq_length <= model_max, (
        f"Requested seq_length={seq_length} > model max positions={model_max}"
    )

    # One short forward to check the hidden-state layout against this checkpoint.
    with torch.no_grad():
        probe_enc = tokenizer(["ACGTACGT"], return_tensors="pt", add_special_tokens=False)
        probe = model(
            input_ids=probe_enc["input_ids"].to(device),
            attention_mask=probe_enc["attention_mask"].to(device),
            output_hidden_states=True,
        )
    num_hidden_states = len(probe.hidden_states)
    expected = expected_num_hidden_states(model.config)
    assert num_hidden_states == expected, (
        f"hidden_states has {num_hidden_states} entries, expected {expected} "
        f"(num_hidden_layers={model.config.num_hidden_layers} + {HIDDEN_STATES_OFFSET}). "
        "The layer indexing assumption in this generator no longer holds for this checkpoint."
    )
    resolved_index = resolve_layer_index(num_hidden_states, layer_index)

    assert torch.equal(probe.hidden_states[-1], probe.last_hidden_state), (
        "hidden_states[-1] does not match last_hidden_state for this checkpoint; "
        "the layer selected by layer_index=-1 would silently differ from the "
        "documented last hidden state."
    )

    embedding_dim = int(probe.hidden_states[resolved_index].shape[-1])
    assert embedding_dim == int(model.config.hidden_size), (
        f"Hidden state width {embedding_dim} does not match config.hidden_size="
        f"{model.config.hidden_size}"
    )
    logger.info(
        "[layer] num_hidden_layers=%d num_hidden_states=%d layer_index=%s -> "
        "resolved_layer_index=%d embedding_dim=%d",
        int(model.config.num_hidden_layers),
        num_hidden_states,
        layer_index,
        resolved_index,
        embedding_dim,
    )

    def embed_chunk(seqs: List[str]) -> List[np.ndarray]:
        return mean_pool_in_batches(
            seqs,
            batch_size,
            lambda b: embed_batch_grover(
                tokenizer, model, b, model_max, device, resolved_index
            ),
        )

    return embed_chunk


def _stamp_layer_provenance(
    output_paths: List[str],
    *,
    layer_index: int,
    resolved_layer_index: int,
    n_layer: int,
    num_hidden_states: int,
    checkpoint: str,
    revision: Optional[str],
    logger: logging.Logger,
) -> None:
    """Write layer provenance into the generated HDF5 files.

    The shared resumable engine in ``src/generation_io.py`` writes no layer
    provenance for any generator, so this stamps the attrs after
    ``run_generation`` returns, making "which layer was this file?" answerable
    from the data alone.
    """
    for path in output_paths:
        if not os.path.exists(path):
            logger.warning(f"[layer] cannot stamp provenance, missing file: {path}")
            continue
        with h5py.File(path, "a") as f:
            f.attrs["layer_index"] = int(layer_index)
            f.attrs["resolved_layer_index"] = int(resolved_layer_index)
            f.attrs["n_layer"] = int(n_layer)
            f.attrs["num_hidden_states"] = int(num_hidden_states)
            f.attrs["checkpoint"] = checkpoint
            f.attrs["revision"] = revision or ""
        logger.info(
            f"[layer] stamped resolved_layer_index={resolved_layer_index} "
            f"n_layer={n_layer} into {path}"
        )


@hydra.main(config_path="../conf", config_name="generate/grover", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint: generate sequences and extract GROVER embeddings."""
    logger = logging.getLogger(__name__)
    logger.info("Generating DNA sequences and GROVER embeddings...")

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
    revision = cfg.get("revision", None)
    layer_index = int(cfg.get("layer_index", -1))

    logger.info(f"Using checkpoint: {checkpoint}")
    logger.info(f"Using revision: {revision}")
    logger.info(f"Using device: {device}")

    # Resolve from the config alone, so the index is known on the fast-exit path
    # where no model is built. The closure re-derives it and asserts they agree.
    model_config = AutoConfig.from_pretrained(
        checkpoint, revision=revision, trust_remote_code=True
    )
    n_layer = int(model_config.num_hidden_layers)
    num_hidden_states = expected_num_hidden_states(model_config)
    resolved_layer_index = resolve_layer_index(num_hidden_states, layer_index)
    logger.info(
        f"Extracting embeddings from hidden_states[{layer_index}] -> "
        f"resolved index {resolved_layer_index} of {num_hidden_states} "
        f"(num_hidden_layers={n_layer}, hidden_size={model_config.hidden_size})"
    )

    batch_size = cfg.batch_size

    # Resumable generation: fast-exit on complete, resume on partial. Returns
    # False only for the legacy per-token path.
    if run_generation(
        cfg,
        sequences=sequences,
        individual_ids=individual_ids,
        locus_ids=locus_ids,
        haplotypes=haplotypes,
        make_embed_chunk=lambda: _make_embed_chunk_grover(
            checkpoint,
            device,
            int(cfg.seq_length),
            int(batch_size),
            resolved_layer_index,
            revision=revision,
        ),
        logger=logger,
    ):
        # run_generation does not record the layer, so stamp it here. Idempotent,
        # so re-stamping on the fast-exit path is harmless.
        stamped = (
            [hy_utils.to_absolute_path(cfg.test_output_path)]
            if identification_mode
            else [
                hy_utils.to_absolute_path(cfg[f"{sp}_output_path"])
                for sp in ("train", "val", "test")
            ]
        )
        _stamp_layer_provenance(
            stamped,
            layer_index=layer_index,
            resolved_layer_index=resolved_layer_index,
            n_layer=n_layer,
            num_hidden_states=num_hidden_states,
            checkpoint=checkpoint,
            revision=revision,
            logger=logger,
        )
        return

    # Only reachable for mean=false, which this pipeline never uses.
    raise NotImplementedError(
        "mean=false (per-token embeddings) is not supported. "
        "The multilen benchmark always runs mean=true."
    )


if __name__ == "__main__":  # pragma: no cover
    main()  # type: ignore
