#!/usr/bin/env python3
"""Regenerate the multi-length training data configs from ``src.model_registry``.

Hydra selects ``data=hg38_multilen_seqdisjoint/<slug>_multi_..._mean`` by
filename, so these stay real files on disk -- but their ``checkpoint`` and
``embedding_dim`` duplicate registry content, and a disagreement is silent
until ``src/data.py`` asserts on the first batch. Generating them keeps the
registry the only place a model is described; ``--check`` makes the drift
visible in CI or a pre-commit hook.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import model_registry
from src.model_registry import ModelSpec

TAG = "hg38_multilen"
GROUP = "hg38_multilen_seqdisjoint"
CONF_DIR = REPO_ROOT / "conf" / "data" / GROUP


def _embedding_dim(spec: ModelSpec) -> int:
    assert spec.embedding_dim, f"{spec.slug}: registry carries no embedding_dim"
    return spec.embedding_dim


def _tokenizer_block(spec: ModelSpec) -> str:
    if spec.decoder_tokenizer == "huggingface":
        return "tokenizer:\n  type: huggingface\n  model_name: " + spec.checkpoint + "\n"
    return "tokenizer:\n  type: char\n"


def render(spec: ModelSpec) -> str:
    return (
        f"defaults:\n  - {GROUP}/_base@_here_\n  - _self_\n"
        f"\n"
        f"fm_slug: {spec.slug}\n"
        f"embedding_dim: {_embedding_dim(spec)}\n"
        f"{_tokenizer_block(spec)}"
        f"checkpoint: {spec.checkpoint}\n"
    )


def config_path(slug: str) -> Path:
    return CONF_DIR / f"{slug}_multi_{GROUP}_mean.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift and exit non-zero instead of rewriting",
    )
    args = parser.parse_args()

    slugs = model_registry.slugs()
    assert slugs, "empty registry in src/model_registry.py"

    stale = 0
    for slug in slugs:
        path = config_path(slug)
        want = render(model_registry.get(slug))
        have = path.read_text() if path.exists() else ""
        if have == want:
            continue
        stale += 1
        rel = path.relative_to(REPO_ROOT)
        if not args.check:
            path.write_text(want)
            print(f"wrote {rel}")
            continue
        print(f"drift: {rel}", file=sys.stderr)
        sys.stderr.writelines(
            difflib.unified_diff(
                have.splitlines(keepends=True),
                want.splitlines(keepends=True),
                fromfile=f"{rel} (on disk)",
                tofile=f"{rel} (registry)",
            )
        )

    orphans = sorted(
        p.name
        for p in CONF_DIR.glob("*_multi_*_mean.yaml")
        if p not in {config_path(s) for s in slugs}
    )
    if orphans:
        stale += len(orphans)
        print(f"not tagged {TAG!r} but present: {', '.join(orphans)}", file=sys.stderr)

    if args.check and stale:
        print(f"{stale} config(s) disagree with the registry", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
