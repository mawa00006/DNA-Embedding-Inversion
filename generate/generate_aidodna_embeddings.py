"""Generate DNA embeddings from AIDO.DNA-300M using sequences from a file.

The output is HDF5 files for train, val, and test with datasets 'sequences' and
'embeddings'. Only mean=true is supported: embeddings contains mean-pooled
embeddings of shape [embedding_dim] for each sequence.

UNVERIFIED / FLAG (measured from public HF metadata and the genbio-ai/
ModelGenerator source on GitHub, not by loading the model -- no GPU/model
download was in scope for this pass; re-probe on the cluster before spending
GPU time):

* Checkpoint identity. `genbio-ai/AIDO.DNA-300M` does not 404, but the HF Hub
  API's resolved `id` for it is `genbio-ai/GB.DNA-300M` -- the org's rename
  from AIDO.* to GB.* has already happened and AIDO.DNA-300M is now a redirect
  alias, not the canonical name. This generator's config therefore points at
  `genbio-ai/GB.DNA-300M` directly (same sha either way:
  a833c5602b0dad0728c0aeeecaf43086ebeb96ca).
* This is NOT loadable via plain `AutoTokenizer`/`AutoModel` +
  `trust_remote_code=True`, despite that being the pattern every other
  generator in this repo uses. The HF repo carries config.json + weights only
  -- no tokenizer files and no `auto_map` key and no modeling .py files for
  `trust_remote_code` to fetch. `AutoConfig.from_pretrained(..., model_type=
  "rnabert")` cannot resolve "rnabert" without something registering it, and
  nothing in this repo does. The classes actually implementing
  `RNABertForMaskedLM` live in the separate `modelgenerator` PyPI package
  (confirmed on PyPI as version 0.1.3.post0), imported directly here as
  `from modelgenerator.huggingface_models.rnabert import ...` rather than
  through `Auto*` dispatch. `modelgenerator` is not in this repo's
  requirements.txt and is not installed in the `dna-inversion` conda env.
* `modelgenerator==0.1.3.post0` pins `transformers==4.38.0`, which conflicts
  with this repo's pinned `transformers==4.37.2` (needed for DNABERT-2's
  remote code). Per the skill's guidance for checkpoints with special
  dependencies (e.g. Caduceus/mamba_ssm), this model should get its own conda
  env for extraction -- the HDF5 boundary makes that safe since training never
  imports the foundation model. Not yet confirmed on the cluster.
* Tokenizer vocab. `RNABertTokenizer` takes a local `vocab_file` path rather
  than pulling one from the checkpoint repo (there isn't one there); the file
  is bundled inside the `modelgenerator` package itself
  (`modelgenerator/huggingface_models/rnabert/vocab.txt`), resolved here via
  `importlib.resources`. Its 16 entries are single-character tokens (A, G, C,
  T, U, N + specials) -- vocab_size=16 in config.json matches.
* `RNABertTokenizer._tokenize` is a bare `text.split()`, which would treat an
  unspaced raw sequence as one token -- except `PreTrainedTokenizer.__init__`
  registers every vocab entry (including each single-nucleotide token) in the
  added-tokens trie as a "no-split" token, and that trie runs before
  `_tokenize` on the raw string. Since the whole alphabet is single characters,
  this means calling the tokenizer directly on a raw "ACGT..." string (no
  manual `" ".join`) should tokenize it one nucleotide at a time. Inferred from
  reading `tokenization_rnabert.py` and `transformers`' trie logic, not
  executed -- flagged rather than assumed.
* config.json's `tokenizer_type: "BertWordPieceLowerCase"` does not match the
  actual `RNABertTokenizer` class found (a fixed single-char vocab with no
  lowercasing, no WordPiece merges). Likely a stale/generic label inherited
  from ModelGenerator's broader tokenizer registry rather than a description
  of this checkpoint. Sequences reaching this generator are already
  upper-cased by `src.utils.load_sequences_from_file`, so this mismatch has no
  effect either way, but the config field itself should not be trusted.
* Hidden-state layout. `RNABertEncoder.forward` (in `modeling_rnabert.py`)
  appends the pre-layer hidden state once per layer, then appends the
  post-`ln`-normalized final state once more -- so hidden_states should have
  num_hidden_layers + 1 = 25 entries and hidden_states[-1] should equal
  last_hidden_state, matching every other generator's HIDDEN_STATES_OFFSET=1.
  Read from source, not measured; asserted at load time below regardless.
* The `genbio-ai/AIDO.DNA-300M` id string contains a literal ".": checked
  against this repo's Hydra configs -- `checkpoint` is never interpolated into
  a path or another config key anywhere in conf/ or src/, only ever passed
  verbatim as a string to `from_pretrained` and stamped as an HDF5 attribute,
  both of which handle dots without issue.

Architecture notes that are NOT in question (from config.json, taken as
given per the task):

* model_type="rnabert", architectures=["RNABertForMaskedLM"], hidden_size=1024,
  num_hidden_layers=24, vocab_size=16, pad_token_id=0.
* RNABertModel accepts `attention_mask` and applies it as a standard additive
  mask inside self-attention (`bert_extended_attention_mask`), so no host-only
  masking workaround is needed here, unlike Caduceus/HyenaDNA.
* The bare `RNABertModel` (base_model_prefix="bert") is loaded from an
  `RNABertForMaskedLM`-shaped checkpoint; missing_keys is asserted empty so a
  key-naming mismatch fails loudly instead of silently random-initialising
  part of the model.

Example usage:
    python generate_aidodna_embeddings.py input_path=data.csv \
        checkpoint=genbio-ai/GB.DNA-300M
"""

from __future__ import annotations

import importlib.resources
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

# Requires the `modelgenerator` PyPI package (not in this repo's
# requirements.txt -- see the module docstring). These classes are plain
# transformers.PreTrainedModel/PretrainedConfig/PreTrainedTokenizer subclasses,
# just not registered with transformers' Auto* dispatch, so they are imported
# and used directly rather than through AutoConfig/AutoTokenizer/AutoModel.
from modelgenerator.huggingface_models.rnabert import (
    RNABertConfig,
    RNABertModel,
    RNABertTokenizer,
)

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
# RNABertEncoder appends the pre-layer state once per layer and the
# post-loop, post-ln state once more, so the length is num_hidden_layers + 1.
# Read from source (see module docstring), asserted below rather than trusted.
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


def embed_batch_aidodna(
    tokenizer: RNABertTokenizer,
    model: RNABertModel,
    sequences: List[str],
    max_length: int,
    device: str,
    layer_index: int,
) -> List[np.ndarray]:
    """Tokenize a batch of DNA sequences with the RNABert tokenizer and extract per-token embeddings.

    Tokenizes without special tokens so embeddings align 1:1 with sequence
    (single-nucleotide) tokens. Returns a list of NumPy arrays, each of shape
    (num_tokens_i, embedding_dim).
    """
    assert sequences, "Empty batch"
    enc = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_attention_mask=True,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

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


def _load_aidodna(checkpoint: str, revision: Optional[str], device: str):
    """Load the RNABert tokenizer/model and return ``(tokenizer, model, model_max)``.

    Unlike every other generator in this repo, this does not go through
    AutoTokenizer/AutoModel -- see the module docstring for why.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading AIDO.DNA tokenizer/model from: {checkpoint} (revision={revision})")

    # The checkpoint repo ships no tokenizer assets; the vocab lives inside the
    # modelgenerator package itself.
    vocab_file = str(
        importlib.resources.files("modelgenerator.huggingface_models.rnabert") / "vocab.txt"
    )
    tokenizer = RNABertTokenizer(vocab_file, version="v2")

    config = RNABertConfig.from_pretrained(checkpoint, revision=revision)
    # RNABertModel loads the bare backbone against masked-LM weights, so
    # missing weights would otherwise be silently random-initialised and only
    # logged.
    model, loading_info = RNABertModel.from_pretrained(
        checkpoint, config=config, revision=revision, output_loading_info=True
    )
    assert not loading_info["missing_keys"], (
        f"Missing keys loading bare backbone from masked-LM checkpoint: "
        f"{loading_info['missing_keys']}"
    )

    assert tokenizer.pad_token is not None, "RNABert tokenizer has no pad token"
    # RNABertTokenizer does not set padding_side; forced explicitly to match
    # the rest of the generators rather than relying on the base-class default.
    tokenizer.padding_side = "right"

    model_max = int(config.max_position_embeddings)
    model.to(device)
    model.eval()
    return tokenizer, model, model_max


def _make_embed_chunk_aidodna(
    checkpoint: str,
    device: str,
    seq_length: int,
    batch_size: int,
    layer_index: int,
    revision: Optional[str] = None,
):
    """Load the model and return ``embed_chunk``: sequences -> mean-pooled ``[D]`` arrays.

    Sub-batches by ``batch_size`` and reuses ``embed_batch_aidodna``; the model
    is loaded once. Note that ``run_generation`` feeds this closure slices of
    ``gen_chunk_size`` samples, so the effective batch is
    ``min(batch_size, gen_chunk_size)``.
    """
    logger = logging.getLogger(__name__)
    tokenizer, model, model_max = _load_aidodna(checkpoint, revision, device)

    assert seq_length <= model_max, (
        f"Requested seq_length={seq_length} > config.max_position_embeddings={model_max}"
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
        f"Hidden state width {embedding_dim} does not match config.hidden_size "
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
            lambda b: embed_batch_aidodna(
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


@hydra.main(config_path="../conf", config_name="generate/aidodna", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint: generate sequences and extract AIDO.DNA embeddings."""
    logger = logging.getLogger(__name__)
    logger.info("Generating DNA sequences and AIDO.DNA embeddings...")

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
    model_config = RNABertConfig.from_pretrained(checkpoint, revision=revision)
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
        make_embed_chunk=lambda: _make_embed_chunk_aidodna(
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
