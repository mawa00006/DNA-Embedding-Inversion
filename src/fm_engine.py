"""Spec-driven loading, tokenization and embedding for the registry's models.

Everything here is driven by a ``ModelSpec``; the engine itself holds the parts
the benchmark fixes for every model (``add_special_tokens=False``, fp32,
mask-filtered mean pooling, right padding, a pinned revision) so a per-model
spec cannot silently regress comparability.
"""

from __future__ import annotations

import importlib.resources
import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from src.generation_io import mean_pool_in_batches
from src.model_registry import ModelSpec

MODEL_CLASSES = {
    "AutoModel": AutoModel,
    "AutoModelForMaskedLM": AutoModelForMaskedLM,
    "AutoModelForCausalLM": AutoModelForCausalLM,
}

# 8 bp: long enough that a k-mer spec (k=6) still yields tokens.
_PROBE_SEQUENCE = "ACGTACGT"


def _rnabert():
    """Import AIDO.DNA's classes from `modelgenerator`, which not every env has."""
    from modelgenerator.huggingface_models.rnabert import (
        RNABertConfig,
        RNABertModel,
        RNABertTokenizer,
    )

    return RNABertConfig, RNABertModel, RNABertTokenizer


def _load_tokenizer(spec: ModelSpec):
    if spec.loader.tokenizer_class == "RNABertTokenizer":
        # The checkpoint repo ships no tokenizer assets; the vocab is bundled
        # inside the modelgenerator package.
        _, _, rnabert_tokenizer = _rnabert()
        vocab = importlib.resources.files("modelgenerator.huggingface_models.rnabert") / "vocab.txt"
        return rnabert_tokenizer(str(vocab), version="v2")
    return AutoTokenizer.from_pretrained(
        spec.checkpoint,
        revision=spec.revision,
        trust_remote_code=spec.loader.tokenizer_trust_remote_code,
    )


def _load_backbone(spec: ModelSpec):
    if spec.loader.model_class == "RNABertModel":
        rnabert_config, rnabert_model, _ = _rnabert()
        config = rnabert_config.from_pretrained(spec.checkpoint, revision=spec.revision)
        return rnabert_model.from_pretrained(
            spec.checkpoint, config=config, revision=spec.revision, output_loading_info=True
        )
    return MODEL_CLASSES[spec.loader.model_class].from_pretrained(
        spec.checkpoint,
        revision=spec.revision,
        trust_remote_code=spec.loader.model_trust_remote_code,
        output_loading_info=True,
    )


def load_model(spec: ModelSpec, device: str) -> Tuple[Any, Any, Any]:
    """Load ``spec``'s checkpoint and return ``(tokenizer, model, config)``."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading {spec.display_name or spec.slug} from: {spec.checkpoint} "
                f"(revision={spec.revision})")
    tok = _load_tokenizer(spec)
    model, info = _load_backbone(spec)

    if spec.loader.missing_keys_policy == "strict":
        # A bare backbone loaded from masked-LM weights silently random-initialises
        # anything absent and only logs it. Assert, minus the known-benign.
        missing = [
            k
            for k in info["missing_keys"]
            if not any(sub in k for sub in spec.loader.missing_keys_ignore)
        ]
        assert not missing, f"{spec.slug}: unexpected missing keys {missing}"

    if spec.tokenization.force_pad_token_from_eos and tok.pad_token is None:
        tok.pad_token = tok.eos_token
    assert tok.pad_token is not None, f"{spec.slug}: tokenizer has no pad_token"
    # Several checkpoints default to left padding, which contaminates every
    # content position of an unmasked mixer.
    tok.padding_side = "right"

    # Captured before the descent: a submodule is not guaranteed to expose the
    # checkpoint's own config.
    config = model.config
    if spec.loader.submodule_attr:
        model = getattr(model, spec.loader.submodule_attr)

    return tok, model.to(device).eval(), config


def resolve_max_length(tok, config, spec: ModelSpec) -> int:
    """Positional capacity for this checkpoint, per the spec's resolution strategy."""
    src = spec.tokenization.max_length_source
    if src == "config":
        return int(config.max_position_embeddings)
    tok_max = int(getattr(tok, "model_max_length", 0) or 0)
    if src == "tokenizer":
        assert tok_max > 0, (
            f"{spec.slug}: tokenizer reports no model_max_length; "
            f"use max_length_source='config' or 'auto'"
        )
        return tok_max
    # auto: a legacy or absent `model_max_length` resolves to the ~1e30 sentinel,
    # in which case the architecture-level bound is the only real one.
    cfg_max = int(config.max_position_embeddings)
    return cfg_max if tok_max > 1_000_000 else min(tok_max, cfg_max)


def _prepare(seqs: List[str], spec: ModelSpec) -> List[str]:
    """Reshape raw ACGT strings into the form the spec's tokenizer expects."""
    if spec.tokenization.input_format != "whitespace_kmer":
        return seqs
    k = spec.tokenization.kmer_size
    assert k, f"{spec.slug}: input_format=whitespace_kmer requires kmer_size"
    # A slow BertTokenizer treats whitespace-split words as pre-tokenized vocab
    # units, so a raw string would become one out-of-vocabulary token.
    for s in seqs:
        assert len(s) >= k, f"{spec.slug}: sequence shorter than k={k}: {s!r}"
    return [" ".join(s[i : i + k] for i in range(len(s) - k + 1)) for s in seqs]


def embed_batch(
    tok,
    model,
    seqs: List[str],
    max_length: int,
    device: str,
    spec: ModelSpec,
    layer_index: int,
) -> List[np.ndarray]:
    """Embed one batch, returning a ``[num_tokens_i, D]`` array per sequence."""
    assert seqs, "Empty batch"
    kw = {"return_attention_mask": True} if spec.tokenization.needs_attention_mask_kw else {}
    enc = tok(
        _prepare(seqs, spec),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        **kw,
    )

    mask = enc["attention_mask"]
    if spec.tokenization.padding_side == "assert_uniform":
        # Bidirectional and unmasked: pad states leak from either end and cannot
        # be masked away afterwards. Fail loudly instead of corrupting silently.
        assert bool(mask.bool().all()), (
            f"{spec.slug}: ragged batch, padded positions would contaminate every "
            "pooled embedding in it"
        )

    fwd = {"input_ids": enc["input_ids"].to(device), "output_hidden_states": True}
    if spec.pass_attention_mask:
        fwd["attention_mask"] = mask.to(device)

    with torch.no_grad():
        out = model(**fwd)

    sel = out.hidden_states[layer_index].detach().float().cpu().numpy()
    m = mask.bool().cpu().numpy()
    return [sel[i][m[i]] for i in range(len(seqs))]


def make_embed_chunk(
    spec: ModelSpec, device: str, batch_size: int
) -> Tuple[Callable[[List[str]], List[np.ndarray]], Dict[str, Any]]:
    """Load the model once and return ``(embed_chunk, meta)``.

    ``embed_chunk`` maps sequences to mean-pooled ``[D]`` arrays, sub-batching by
    ``batch_size``; ``meta`` is the layer/width provenance measured while probing.
    """
    logger = logging.getLogger(__name__)
    tok, model, config = load_model(spec, device)
    depth = int(getattr(config, spec.layer.depth_attr))

    # One cheap forward to measure the layout instead of assuming it.
    enc = tok(_prepare([_PROBE_SEQUENCE], spec), return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        probe = model(input_ids=enc["input_ids"].to(device), output_hidden_states=True)

    n_states = len(probe.hidden_states)
    expected = depth + spec.layer.hidden_states_offset
    assert n_states == expected, (
        f"{spec.slug}: hidden_states has {n_states} entries, expected {expected} "
        f"({spec.layer.depth_attr}={depth} + offset {spec.layer.hidden_states_offset}). "
        "The spec's layer layout no longer holds for this checkpoint."
    )
    # Pins the selected layer regardless of which internal branch produced it.
    assert torch.equal(probe.hidden_states[-1], probe.last_hidden_state), (
        f"{spec.slug}: hidden_states[-1] is not last_hidden_state; layer_index=-1 "
        "would silently differ from the documented last hidden state."
    )

    idx = int(spec.layer.layer_index)
    assert -n_states <= idx < n_states, (
        f"{spec.slug}: layer_index={idx} out of range for {n_states} hidden states"
    )
    idx %= n_states

    # Width comes from the TENSOR; config is only the cross-check, so a
    # mismatch fails here rather than after all 14 lengths are on disk.
    observed = int(probe.hidden_states[idx].shape[-1])
    declared = int(getattr(config, spec.width.config_attr))
    if spec.width.multiplier_attr and getattr(config, spec.width.multiplier_attr, False):
        declared *= 2
    assert observed == declared, (
        f"{spec.slug}: observed width {observed} != declared {declared} "
        f"(config.{spec.width.config_attr}, multiplier={spec.width.multiplier_attr})"
    )

    max_len = resolve_max_length(tok, config, spec)
    logger.info(
        "[layer] %s: %s=%d num_hidden_states=%d layer_index=%s -> resolved=%d "
        "embedding_dim=%d max_length=%d",
        spec.slug,
        spec.layer.depth_attr,
        depth,
        n_states,
        spec.layer.layer_index,
        idx,
        observed,
        max_len,
    )

    def embed_chunk(chunk: List[str]) -> List[np.ndarray]:
        return mean_pool_in_batches(
            chunk,
            batch_size,
            lambda b: embed_batch(tok, model, b, max_len, device, spec, idx),
        )

    meta = {
        "embedding_dim": observed,
        "n_layer": depth,
        "num_hidden_states": n_states,
        "resolved_layer_index": idx,
        "layer_index": spec.layer.layer_index,
        "checkpoint": spec.checkpoint,
        "revision": spec.revision,
        "max_length": max_len,
    }
    return embed_chunk, meta
