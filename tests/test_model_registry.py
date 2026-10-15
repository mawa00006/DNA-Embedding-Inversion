"""Tests for the registry the engine, the plots and the shell drivers all read."""

import re

import pytest

from src.model_registry import REGISTRY, get, slugs, slugs_for

# Registry order fixes legend/marker order in every published figure.
EXPECTED_ORDER = [
    "hyenadna_large",
    "caduceus",
    "dnabert1",
    "grover",
    "genalm",
    "mistraldna",
    "seqlens",
    "aidodna",
    "ntv1",
    "dnabert2",
    "evo2",
    "ntv2",
]
# The three that keep their own generator; their revisions were never pinned.
LEGACY_SLUGS = {"dnabert2", "evo2", "ntv2"}
TAGS = ("hg38_multilen", "ood_1000g", "identification")
INPUT_FORMATS = {"raw", "whitespace_kmer"}
PADDING_SIDES = {"right", "assert_uniform"}
MAX_LENGTH_SOURCES = {"tokenizer", "auto", "config"}
MISSING_KEYS_POLICIES = {"strict", "skip"}


def test_registry_order_matches_the_published_figure_order():
    assert slugs() == EXPECTED_ORDER


def test_slugs_are_unique_and_carry_a_display_name():
    assert len(set(slugs())) == len(EXPECTED_ORDER)
    for slug, spec in REGISTRY.items():
        assert slug and spec.slug == slug
        assert spec.display_name
        assert spec.checkpoint


@pytest.mark.parametrize("slug", slugs())
def test_slug_survives_the_evaluate_filename_round_trip(slug):
    basename = f"train_{slug}_25_hg38_multilen_seqdisjoint.h5"
    match = re.search(r"train_(.+?)_\d+_hg38", basename)
    assert match is not None
    assert match.group(1) == slug


def test_get_rejects_an_unknown_slug():
    with pytest.raises(AssertionError, match="unknown fm_slug 'dnabert6'"):
        get("dnabert6")


@pytest.mark.parametrize("tag", TAGS)
def test_slugs_for_returns_a_non_empty_subset_in_registry_order(tag):
    tagged = slugs_for(tag)
    assert tagged
    assert tagged == [slug for slug in slugs() if slug in set(tagged)]
    assert all(tag in REGISTRY[slug].included_in for slug in tagged)


def test_no_spec_carries_a_tag_no_driver_queries():
    assert {tag for spec in REGISTRY.values() for tag in spec.included_in} == set(TAGS)


def test_engine_support_is_off_for_exactly_the_legacy_generators():
    assert {s for s, spec in REGISTRY.items() if not spec.engine_supported} == LEGACY_SLUGS


@pytest.mark.parametrize("slug", slugs())
def test_revision_is_pinned_outside_the_legacy_three(slug):
    spec = get(slug)
    if slug in LEGACY_SLUGS:
        assert spec.revision == ""
    else:
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision)


@pytest.mark.parametrize("slug", slugs())
def test_spec_fields_are_internally_consistent(slug):
    spec = get(slug)
    tokenization, loader = spec.tokenization, spec.loader
    assert tokenization.input_format in INPUT_FORMATS
    assert (tokenization.kmer_size is not None) == (
        tokenization.input_format == "whitespace_kmer"
    )
    assert tokenization.kmer_size is None or tokenization.kmer_size > 0
    assert tokenization.padding_side in PADDING_SIDES
    assert tokenization.max_length_source in MAX_LENGTH_SOURCES
    assert loader.missing_keys_policy in MISSING_KEYS_POLICIES
    if loader.missing_keys_policy != "strict":
        assert loader.missing_keys_ignore == ()
