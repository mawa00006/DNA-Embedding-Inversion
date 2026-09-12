"""Generate DNA embeddings from Caduceus using sequences from a file.

This script extracts embeddings from a Caduceus checkpoint (RC-equivariant
Mamba) for sequences provided in a file. The output is HDF5 files for train,
val, and test with datasets 'sequences' and 'embeddings'.

Only mean=true is supported: embeddings contains mean-pooled embeddings of shape
[embedding_dim] for each sequence.

Architecture notes, since a straight copy of another HF generator breaks on all
of these:

* config.rcps is FALSE for this checkpoint (the -ph variant), so the width is
  d_model=256. The -ps variant sets rcps=True, where RCPSEmbedding concatenates the
  forward and reverse-complement channels: hidden state width is 2*d_model, not
  d_model. This generator keeps that raw concatenated vector as emitted -- no
  folding or averaging of the two strands.
* The config exposes width/depth as d_model / n_layer; there is no hidden_size.
* Caduceus.forward has no attention_mask argument. Passing one raises
  TypeError. The mask is computed and used to drop padding before pooling, but
  never passed to the model.
* The tokenizer sets model_input_names = ["input_ids"], so the mask is only
  returned if return_attention_mask=True is passed.
* The tokenizer defaults to padding_side="left", but neither side is safe: the
  mixer is bidirectional with no masking, so pad states leak from either end.
  Every corpus run here is fixed-length, so right padding is forced AND a
  ragged batch is asserted against rather than silently tolerated.
* hidden_states has n_layer + 1 entries, but the final normed state is only
  appended on the fused_add_norm=True branch. The observed length is asserted
  against n_layer, and hidden_states[-1] is asserted equal to last_hidden_state
  so the selected layer is pinned regardless of that branch.
* AutoModel loads the bare backbone against masked-LM weights; missing_keys is
  asserted empty so a mismatch fails loudly instead of silently random-
  initialising part of the model.
* mamba_ssm is a hard module-scope import and is CUDA-only, so this model
  cannot be loaded on a CPU-only machine.

Example usage:
    python generate_caduceus_embeddings.py input_path=data.csv \
        checkpoint=kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16
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
#
# The final normed state is appended only on the fused_add_norm=True branch, so
# the observed length is asserted against n_layer, and hidden_states[-1] is
# separately asserted equal to last_hidden_state -- pinning the selected layer
# regardless of which branch produced it.
HIDDEN_STATES_OFFSET = 1


def expected_num_hidden_states(config) -> int:
    """Number of entries ``outputs.hidden_states`` will have for this architecture."""
    return int(config.n_layer) + HIDDEN_STATES_OFFSET


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


def embed_batch_caduceus(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    sequences: List[str],
    max_length: int,
    device: str,
    layer_index: int,
) -> List[np.ndarray]:
    """Tokenize a batch of DNA sequences with the Caduceus tokenizer and extract per-token embeddings.

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
        # The Caduceus tokenizer declares model_input_names = ["input_ids"], so
        # the mask is omitted unless requested explicitly.
        return_attention_mask=True,
    )

    input_ids = enc["input_ids"].to(device)
    # forward() takes no attention_mask, so the mask stays on the host and is
    # only used to drop padding rows before pooling.
    attention_mask = enc["attention_mask"]

    # The mixer is bidirectional with no masking, so a padded position leaks
    # into every content position regardless of padding side. Every corpus run
    # here is fixed-length, so a ragged batch is a bug, not a case to handle --
    # fail loudly instead of silently corrupting the pooled vector.
    assert bool(attention_mask.bool().all()), (
        "Ragged batch detected: Caduceus has no attention masking inside the "
        "model, so padded positions would silently contaminate every pooled "
        "embedding in this batch."
    )

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

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


def _load_caduceus(checkpoint: str, revision: Optional[str], device: str):
    """Load the Caduceus tokenizer/model and return ``(tokenizer, model, model_max)``."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading Caduceus tokenizer/model from: {checkpoint} (revision={revision})")
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token is not None, "Caduceus tokenizer has no pad token"

    # Neither padding side is safe for this unmasked bidirectional mixer. Right
    # is forced to match the rest of the generators; embed_batch_caduceus
    # additionally asserts no batch is actually ragged.
    tokenizer.padding_side = "right"

    model_max = int(tokenizer.model_max_length)
    model.to(device)
    model.eval()
    return tokenizer, model, model_max


def _make_embed_chunk_caduceus(
    checkpoint: str,
    device: str,
    seq_length: int,
    batch_size: int,
    layer_index: int,
    revision: Optional[str] = None,
):
    """Load the model and return ``embed_chunk``: sequences -> mean-pooled ``[D]`` arrays.

    Sub-batches by ``batch_size`` and reuses ``embed_batch_caduceus``; the model
    is loaded once. Note that ``run_generation`` feeds this closure slices of
    ``gen_chunk_size`` samples, so the effective batch is
    ``min(batch_size, gen_chunk_size)``.
    """
    logger = logging.getLogger(__name__)
    tokenizer, model, model_max = _load_caduceus(checkpoint, revision, device)

    assert seq_length <= model_max, (
        f"Requested seq_length={seq_length} > tokenizer.model_max_length={model_max}"
    )

    # One 4 bp forward to check the hidden-state layout against this checkpoint.
    with torch.no_grad():
        probe = model(
            input_ids=tokenizer(
                ["ACGT"], return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(device),
            output_hidden_states=True,
        )
    num_hidden_states = len(probe.hidden_states)
    expected = expected_num_hidden_states(model.config)
    assert num_hidden_states == expected, (
        f"hidden_states has {num_hidden_states} entries, expected {expected} "
        f"(n_layer={model.config.n_layer} + {HIDDEN_STATES_OFFSET}). The layer "
        "indexing assumption in this generator no longer holds for this checkpoint."
    )
    resolved_index = resolve_layer_index(num_hidden_states, layer_index)

    # Pin the final layer regardless of which fused_add_norm branch appended it.
    assert torch.equal(probe.hidden_states[-1], probe.last_hidden_state), (
        "hidden_states[-1] does not match last_hidden_state for this checkpoint; "
        "the layer selected by layer_index=-1 would silently differ from the "
        "documented last hidden state."
    )

    # RCPSEmbedding concatenates forward and reverse-complement channels when
    # rcps=True on the -ps variant doubles the width vs config.d_model; -ph does
    # not. Derived either way rather than hardcoded. Trust the
    # observed tensor, but assert it against the expected width so a checkpoint
    # change that drops RCPS is caught rather than silently halving the vector.
    d_model = int(model.config.d_model)
    rcps = bool(getattr(model.config, "rcps", False))
    expected_width = 2 * d_model if rcps else d_model
    embedding_dim = int(probe.hidden_states[resolved_index].shape[-1])
    assert embedding_dim == expected_width, (
        f"Hidden state width {embedding_dim} does not match expected {expected_width} "
        f"(d_model={d_model}, rcps={rcps})"
    )
    logger.info(
        "[layer] n_layer=%d num_hidden_states=%d layer_index=%s -> "
        "resolved_layer_index=%d d_model=%d rcps=%s embedding_dim=%d",
        int(model.config.n_layer),
        num_hidden_states,
        layer_index,
        resolved_index,
        d_model,
        rcps,
        embedding_dim,
    )

    def embed_chunk(seqs: List[str]) -> List[np.ndarray]:
        return mean_pool_in_batches(
            seqs,
            batch_size,
            lambda b: embed_batch_caduceus(
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


@hydra.main(config_path="../conf", config_name="generate/caduceus", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint: generate sequences and extract Caduceus embeddings."""
    logger = logging.getLogger(__name__)
    logger.info("Generating DNA sequences and Caduceus embeddings...")

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
    n_layer = int(model_config.n_layer)
    num_hidden_states = expected_num_hidden_states(model_config)
    resolved_layer_index = resolve_layer_index(num_hidden_states, layer_index)
    logger.info(
        f"Extracting embeddings from hidden_states[{layer_index}] -> "
        f"resolved index {resolved_layer_index} of {num_hidden_states} "
        f"(n_layer={n_layer}, d_model={model_config.d_model})"
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
        make_embed_chunk=lambda: _make_embed_chunk_caduceus(
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
