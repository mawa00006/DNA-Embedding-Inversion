"""Tests for the engine's spec-driven tokenization and max-length resolution."""

import pytest

pytest.importorskip("torch")

from src.fm_engine import _prepare, resolve_max_length  # noqa: E402
from src.model_registry import ModelSpec, Tokenization, get  # noqa: E402

RAW_SPEC = get("grover")
KMER_SPEC = get("dnabert1")
SENTINEL = int(1e30)  # transformers' "unset model_max_length"


class _Tokenizer:
    def __init__(self, model_max_length: int):
        self.model_max_length = model_max_length


class _Config:
    def __init__(self, max_position_embeddings: int):
        self.max_position_embeddings = max_position_embeddings


def _spec(max_length_source: str) -> ModelSpec:
    return ModelSpec(
        "stub",
        "stub/checkpoint",
        "0" * 40,
        tokenization=Tokenization(max_length_source=max_length_source),
    )


def test_prepare_leaves_raw_sequences_untouched():
    sequences = ["ACGTACGTAC", "TTTTTTTTTT"]
    assert _prepare(sequences, RAW_SPEC) == sequences


def test_prepare_kmerises_a_25bp_window_reversibly():
    sequence = "ACGTACGTACGTAAACCCGGGTTTA"
    assert len(sequence) == 25
    (prepared,) = _prepare([sequence], KMER_SPEC)
    kmers = prepared.split(" ")
    assert len(kmers) == 20
    assert all(len(kmer) == 6 for kmer in kmers)
    assert kmers[0] + "".join(kmer[-1] for kmer in kmers[1:]) == sequence


def test_prepare_rejects_a_sequence_shorter_than_k():
    with pytest.raises(AssertionError, match="shorter than k=6"):
        _prepare(["ACGT"], KMER_SPEC)


def test_resolve_max_length_from_tokenizer_ignores_the_config():
    assert resolve_max_length(_Tokenizer(512), _Config(1024), _spec("tokenizer")) == 512


def test_resolve_max_length_from_config_ignores_the_tokenizer():
    assert resolve_max_length(_Tokenizer(512), _Config(1024), _spec("config")) == 1024


@pytest.mark.parametrize(
    "tokenizer_max, expected", [(SENTINEL, 1024), (512, 512), (4096, 1024)]
)
def test_resolve_max_length_auto_bounds_the_tokenizer_by_the_config(
    tokenizer_max, expected
):
    spec = _spec("auto")
    assert resolve_max_length(_Tokenizer(tokenizer_max), _Config(1024), spec) == expected
