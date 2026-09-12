#!/usr/bin/env python3
"""Probe a HuggingFace DNA foundation model for everything the generator needs.

Measures rather than assumes. Every field printed here corresponds to a decision
in the generator or to a way a copied generator silently produces wrong
embeddings. Run this before writing any code.

Usage:
    python probe_checkpoint.py <checkpoint> [--revision SHA] [--seq-len 25]
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

WIDTH_ATTRS = ("hidden_size", "d_model", "n_embd", "dim", "hidden_dim", "embed_dim")
DEPTH_ATTRS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers", "depth")


def _first_attr(cfg, names):
    for n in names:
        v = getattr(cfg, n, None)
        if isinstance(v, int) and v > 0:
            return n, v
    return None, None


def probe(checkpoint: str, revision: str | None, seq_len: int) -> dict:
    out: dict = {"checkpoint": checkpoint, "revision": revision}

    cfg = AutoConfig.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)
    out["config_class"] = type(cfg).__name__

    # The generator needs the width for `embedding_dim`, and data.py asserts it
    # against the real tensor. Not every config calls it hidden_size.
    w_attr, width = _first_attr(cfg, WIDTH_ATTRS)
    d_attr, depth = _first_attr(cfg, DEPTH_ATTRS)
    out["width_attr"], out["width"] = w_attr, width
    out["depth_attr"], out["depth"] = d_attr, depth

    tok = AutoTokenizer.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)
    out["tokenizer_class"] = type(tok).__name__
    out["pad_token"] = tok.pad_token
    out["padding_side"] = tok.padding_side
    out["model_input_names"] = list(getattr(tok, "model_input_names", []))
    mml = getattr(tok, "model_max_length", None)
    out["model_max_length"] = int(mml) if isinstance(mml, int) and mml < 10**12 else str(mml)
    out["model_max_length_is_sentinel"] = not (isinstance(mml, int) and mml < 10**12)

    try:
        model = AutoModel.from_pretrained(
            checkpoint, revision=revision, trust_remote_code=True
        ).eval()
    except ImportError as exc:
        # Remote code often needs packages that are not installable everywhere --
        # mamba_ssm, flash-attn and friends are CUDA-only. That is a finding, not
        # a crash: the model needs its own env, probably on a GPU node.
        missing = re.findall(r"[\w_]+", str(exc).split("not found in your environment:")[-1])
        out["model_load_error"] = str(exc).strip().splitlines()[0]
        out["missing_packages"] = sorted(
            {m for m in missing if m not in ("Run", "pip", "install")}
        )[:5]
        out["model_loaded"] = False
        return out
    except Exception as exc:
        out["model_load_error"] = repr(exc)
        out["model_loaded"] = False
        return out
    out["model_loaded"] = True
    out["model_class"] = type(model).__name__
    n_params = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    out["params"] = n_params
    out["buffers"] = n_buffers
    out["fp32_bytes"] = 4 * (n_params + n_buffers)
    out["fp32_gb"] = round(out["fp32_bytes"] / 1e9, 3)

    # Does forward() actually take a mask? Several DNA models do not, and calling
    # model(**enc) then raises TypeError at the first batch.
    sig = inspect.signature(model.forward)
    out["forward_params"] = list(sig.parameters)
    out["accepts_attention_mask"] = "attention_mask" in sig.parameters

    seqs = ["ACGT" * (seq_len // 4) or "ACGT", "ACGT"]
    enc = tok(seqs, return_tensors="pt", padding=True, add_special_tokens=False,
              return_attention_mask=True)
    out["tokenizer_returns_mask_by_default"] = "attention_mask" in tok(
        seqs, return_tensors="pt", padding=True, add_special_tokens=False
    )

    kwargs = {"input_ids": enc["input_ids"], "output_hidden_states": True}
    if out["accepts_attention_mask"]:
        kwargs["attention_mask"] = enc["attention_mask"]
    with torch.no_grad():
        res = model(**kwargs)

    hs = getattr(res, "hidden_states", None)
    out["returns_hidden_states"] = hs is not None
    if hs is not None:
        out["num_hidden_states"] = len(hs)
        out["hidden_states_offset"] = len(hs) - depth if depth else None
        out["last_hidden_state_index"] = len(hs) - 1
        lhs = getattr(res, "last_hidden_state", None)
        out["minus1_is_last_hidden_state"] = bool(
            lhs is not None and torch.equal(hs[-1], lhs)
        )
        out["observed_width"] = int(hs[-1].shape[-1])
        out["width_matches_config"] = (width == out["observed_width"])
        distinct = len({id(h) for h in hs})
        out["hidden_states_all_distinct_objects"] = distinct == len(hs)

    # Padding sensitivity. For an unmasked causal convolution, left padding leaks
    # pad embeddings into every content position while right padding does not.
    # This fails silently -- the embeddings are simply wrong -- so measure it.
    def pooled(side: str) -> np.ndarray:
        tok.padding_side = side
        e = tok(["ACGT", "ACGT" * 5], return_tensors="pt", padding=True,
                add_special_tokens=False, return_attention_mask=True)
        kw = {"input_ids": e["input_ids"], "output_hidden_states": True}
        if out["accepts_attention_mask"]:
            kw["attention_mask"] = e["attention_mask"]
        with torch.no_grad():
            r = model(**kw)
        h = r.hidden_states[-1][0].numpy()
        m = e["attention_mask"][0].bool().numpy()
        return h[m].mean(axis=0)

    original_side = tok.padding_side
    try:
        e1 = tok(["ACGT"], return_tensors="pt", add_special_tokens=False)
        kw = {"input_ids": e1["input_ids"], "output_hidden_states": True}
        if out["accepts_attention_mask"]:
            kw["attention_mask"] = torch.ones_like(e1["input_ids"])
        with torch.no_grad():
            ref = model(**kw).hidden_states[-1][0].numpy().mean(axis=0)
        scale = max(float(np.abs(ref).mean()), 1e-12)
        out["pad_err_right"] = round(float(np.abs(pooled("right") - ref).mean() / scale), 6)
        out["pad_err_left"] = round(float(np.abs(pooled("left") - ref).mean() / scale), 6)
        out["padding_side_matters"] = out["pad_err_left"] > 100 * max(out["pad_err_right"], 1e-9)
    except Exception as exc:  # a model that cannot batch at all still probes usefully
        out["pad_test_error"] = repr(exc)
    finally:
        tok.padding_side = original_side

    gb = out["fp32_gb"]
    out["recommended_gpu"] = (
        "1080ti" if gb <= 8 else "A4000" if gb <= 12 else "L40S"
    )
    return out


def report(p: dict) -> None:
    def line(k, v):
        print(f"  {k:34s} {v}")

    print(f"\n=== {p['checkpoint']} ({p['config_class']} / {p.get('model_class', 'not loaded')}) ===")

    if not p.get("model_loaded", True):
        print("\nMODEL COULD NOT BE LOADED HERE -- partial probe\n")
        print(f"  error            {p['model_load_error']}")
        if p.get("missing_packages"):
            print(f"  missing packages {p['missing_packages']}")
            print("\n  These are usually CUDA-only (mamba_ssm, flash-attn, causal-conv1d) and")
            print("  will not install on a laptop. Re-run this probe on a GPU node in an env")
            print("  that has them. The model also needs that env for extraction, which the")
            print("  HDF5 boundary makes safe -- training never imports the foundation model.")
        print("\nWHAT WE STILL LEARNED (config + tokenizer only)")
        print(f"  width (embedding_dim)    {p['width']}  via config.{p['width_attr']}")
        print(f"  depth                    {p['depth']}  via config.{p['depth_attr']}")
        print(f"  pad_token                {p['pad_token']}")
        print(f"  padding_side             {p['padding_side']}")
        print(f"  model_input_names        {p['model_input_names']}")
        print(f"  model_max_length         {p['model_max_length']}"
              + ("  (SENTINEL - drop the assert)" if p["model_max_length_is_sentinel"] else ""))
        print("\n  The layer layout, mask handling and padding sensitivity are UNKNOWN")
        print("  until the model loads. Do not guess them -- they are the traps.\n")
        return

    print("\nSIZE")
    line("width (embedding_dim)", f"{p['width']}  via config.{p['width_attr']}")
    line("depth", f"{p['depth']}  via config.{p['depth_attr']}")
    line("params / buffers", f"{p['params']:,} / {p['buffers']:,}")
    line("fp32 footprint", f"{p['fp32_gb']} GB  -> --gres=gpu:{p['recommended_gpu']}:1")

    print("\nTRAPS")
    line("forward takes attention_mask", p["accepts_attention_mask"])
    line("tokenizer returns mask default", p["tokenizer_returns_mask_by_default"])
    line("model_input_names", p["model_input_names"])
    line("padding_side default", p["padding_side"])
    line("pad_token", p["pad_token"])
    line("model_max_length", f"{p['model_max_length']}"
         + ("  (SENTINEL - drop the assert)" if p["model_max_length_is_sentinel"] else ""))
    if "pad_err_left" in p:
        line("pooled err, right padding", p["pad_err_right"])
        line("pooled err, left padding", p["pad_err_left"])
        line("padding side matters", p["padding_side_matters"])

    print("\nLAYERS")
    if p.get("returns_hidden_states"):
        off = p["hidden_states_offset"]
        line("len(hidden_states)", f"{p['num_hidden_states']}  (depth {p['depth']} + {off})")
        line("-1 == last_hidden_state", p["minus1_is_last_hidden_state"])
        line("observed width", f"{p['observed_width']}"
             + ("" if p["width_matches_config"] else "  <-- DISAGREES WITH CONFIG"))
        if off not in (1, None):
            print(f"\n  NOTE: offset is {off}, not the usual 1. Do not compute a layer")
            print("  index from depth; use -1 and assert len(hidden_states) against depth.")
    else:
        print("  output_hidden_states=True returned nothing -- inspect the model code.")

    print("\nWARNINGS")
    warn = []
    if not p["accepts_attention_mask"]:
        warn.append("forward() takes no attention_mask: pass input_ids only, "
                    "use the mask for pooling only.")
    if not p["tokenizer_returns_mask_by_default"]:
        warn.append("tokenizer omits attention_mask: pass return_attention_mask=True.")
    if p["padding_side"] != "right":
        warn.append(f"padding_side is '{p['padding_side']}': force 'right' before tokenizing.")
    if p.get("padding_side_matters"):
        warn.append("left padding measurably corrupts pooled output. Right padding is required.")
    if p["pad_token"] is None:
        warn.append("pad_token is None: set one or batching fails.")
    if p.get("width_matches_config") is False:
        warn.append("config width disagrees with the real tensor: trust the tensor.")
    if p.get("hidden_states_offset") not in (1, None):
        warn.append(f"hidden_states has depth+{p['hidden_states_offset']} entries.")
    print("  " + ("\n  ".join(f"- {w}" for w in warn) if warn else "- none"))
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--seq-len", type=int, default=25)
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    a = ap.parse_args()
    p = probe(a.checkpoint, a.revision, a.seq_len)
    if a.json:
        print(json.dumps(p, indent=2, default=str))
    else:
        report(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
