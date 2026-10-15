"""Per-model specs for the generic embedding engine.

``REGISTRY`` is the single ordered source every consumer reads -- the engine,
``plot_cross_dataset.py``'s ``FM_CONFIG``, and the shell drivers' ``FMS``
arrays -- instead of maintaining its own parallel list. Insertion order is the
historical onboarding order and fixes legend/marker order in existing plots, so
new models append to the end.

Deliberately free of torch/transformers imports: the shell drivers query it
with a bare ``python -c``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Loader:
    """How the checkpoint's tokenizer and model objects are instantiated."""

    tokenizer_class: str = "AutoTokenizer"
    model_class: str = "AutoModel"
    tokenizer_trust_remote_code: bool = True
    model_trust_remote_code: bool = True      # independent of the tokenizer flag
    submodule_attr: str | None = None         # forward through getattr(model, attr)
    missing_keys_policy: str = "strict"       # strict | skip
    missing_keys_ignore: tuple[str, ...] = ()


@dataclass(frozen=True)
class Tokenization:
    """How raw ACGT strings are turned into the tensors the model expects."""

    input_format: str = "raw"                 # raw | whitespace_kmer
    kmer_size: int | None = None
    needs_attention_mask_kw: bool = False
    padding_side: str = "right"               # right | assert_uniform
    force_pad_token_from_eos: bool = False
    max_length_source: str = "tokenizer"      # tokenizer | auto | config


@dataclass(frozen=True)
class Layer:
    """Where in ``outputs.hidden_states`` the reported embedding lives."""

    depth_attr: str = "num_hidden_layers"
    hidden_states_offset: int = 1
    layer_index: int = -1


@dataclass(frozen=True)
class Width:
    """Config attributes the observed hidden-state width is checked against."""

    config_attr: str = "hidden_size"
    multiplier_attr: str | None = None        # bool config attr; doubles width when true


@dataclass(frozen=True)
class ModelSpec:
    """Everything that differs between one foundation model and the next."""

    slug: str
    checkpoint: str
    revision: str
    display_name: str = ""
    embedding_dim: int = 0                     # width training reads before any model loads
    decoder_tokenizer: str = "char"            # decoder target vocab: char | huggingface
    pass_attention_mask: bool = True
    loader: Loader = field(default_factory=Loader)
    tokenization: Tokenization = field(default_factory=Tokenization)
    layer: Layer = field(default_factory=Layer)
    width: Width = field(default_factory=Width)
    requires: tuple[str, ...] = ()            # extra deps -> may need its own env
    included_in: tuple[str, ...] = ()         # hg38_multilen | ood_1000g | identification
    engine_supported: bool = True


_SPECS = (
    # hidden_states has n_layer + 2 entries: the final state appears both as the
    # last block output and again after ln_f, and the two differ only by that
    # LayerNorm. Loaded without a missing-keys check, as the per-model generator
    # this replaces did.
    ModelSpec(
        "hyenadna_large",
        "LongSafari/hyenadna-large-1m-seqlen-hf",
        "0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446",
        "HyenaDNA-1M",
        embedding_dim=256,
        pass_attention_mask=False,
        loader=Loader(missing_keys_policy="skip"),
        width=Width(config_attr="d_model"),
        layer=Layer(depth_attr="n_layer", hidden_states_offset=2),
        tokenization=Tokenization(needs_attention_mask_kw=True, force_pad_token_from_eos=True),
        included_in=("hg38_multilen", "ood_1000g", "identification"),
    ),
    # Bidirectional and unmasked, so pad states leak from either end: right
    # padding is forced and every batch is additionally asserted unpadded.
    # ``rcps`` is False on the -ph variant and True on -ps, where RCPSEmbedding
    # concatenates forward and reverse-complement channels.
    ModelSpec(
        "caduceus",
        "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        "b0477522ac5d044ad03578aa724ec8e4bdbd405b",
        "Caduceus-PH",
        embedding_dim=256,
        pass_attention_mask=False,
        width=Width(config_attr="d_model", multiplier_attr="rcps"),
        layer=Layer(depth_attr="n_layer"),
        tokenization=Tokenization(
            needs_attention_mask_kw=True,
            padding_side="assert_uniform",
            force_pad_token_from_eos=True,
        ),
        requires=("mamba_ssm",),
        included_in=("hg38_multilen", "ood_1000g", "identification"),
    ),
    # Remote code for the tokenizer (its 6-mer vocabulary), NOT for the model:
    # stock BertModel rejects the checkpoint's own BertConfig and will not load
    # at all with both flags set. tokenizer_config.json uses the legacy
    # ``max_len`` key, so model_max_length falls back to the ~1e30 sentinel.
    ModelSpec(
        "dnabert1",
        "zhihan1996/DNA_bert_6",
        "c56e67ea5827e0ddc67ef059addcf71569b1216e",
        "DNABERT-1",
        embedding_dim=768,
        loader=Loader(model_trust_remote_code=False),
        tokenization=Tokenization(
            input_format="whitespace_kmer", kmer_size=6, max_length_source="auto"
        ),
        included_in=("hg38_multilen", "ood_1000g", "identification"),
    ),
    ModelSpec(
        "grover",
        "PoetschLab/GROVER",
        "6b223110f0d6963e849f55bc2a2f3cff0e38c7a4",
        "GROVER",
        embedding_dim=768,
        tokenization=Tokenization(max_length_source="auto"),
        included_in=("hg38_multilen", "ood_1000g", "identification"),
    ),
    # AutoModel resolves via auto_map to the checkpoint's own BertForMaskedLM,
    # whose forward returns MaskedLMOutput (hidden_states, no last_hidden_state);
    # .bert gives the usual pair and skips the unused MLM head.
    ModelSpec(
        "genalm",
        "AIRI-Institute/gena-lm-bert-base-t2t",
        "4f1352bd4e820f1dba341047f54ad7795e083bf2",
        "GENA-LM",
        embedding_dim=768,
        loader=Loader(submodule_attr="bert"),
        tokenization=Tokenization(max_length_source="auto"),
        included_in=("hg38_multilen", "ood_1000g", "identification"),
    ),
    # Stock Mixtral: no remote code on either side, and model_max_length is
    # unset in tokenizer_config.json so the real bound is the config's.
    ModelSpec(
        "mistraldna",
        "RaphaelMourad/Mistral-DNA-v1-17M-hg38",
        "e8e18f84ff71dcd4a3a84ff130fe763a7e636ebb",
        "Mistral-DNA",
        embedding_dim=256,
        loader=Loader(tokenizer_trust_remote_code=False, model_trust_remote_code=False),
        tokenization=Tokenization(force_pad_token_from_eos=True, max_length_source="config"),
        included_in=("hg38_multilen", "ood_1000g", "identification"),
    ),
    ModelSpec(
        "seqlens",
        "omicseye/seqLens_32768_512_30M",
        "ca7762d3178845439c5db5984db5904a906af4f0",
        "seqLens",
        embedding_dim=512,
        included_in=("hg38_multilen", "ood_1000g", "identification"),
    ),
    # No tokenizer assets and no auto_map in the checkpoint repo: the classes
    # and the vocab file both live in the `modelgenerator` package, which pins
    # transformers==4.38.0 against this repo's 4.37.2 and so needs its own env.
    ModelSpec(
        "aidodna",
        "genbio-ai/GB.DNA-300M",
        "a833c5602b0dad0728c0aeeecaf43086ebeb96ca",
        "AIDO.DNA",
        embedding_dim=1024,
        loader=Loader(tokenizer_class="RNABertTokenizer", model_class="RNABertModel"),
        tokenization=Tokenization(needs_attention_mask_kw=True, max_length_source="config"),
        requires=("modelgenerator",),
        included_in=("hg38_multilen", "ood_1000g", "identification"),
    ),
    # EsmModel always adds a pooler the masked-LM checkpoint does not carry. It
    # is never read, so it is excluded by name rather than by disabling the check.
    ModelSpec(
        "ntv1",
        "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        "b746b125aacd1b0970c05a32fd71ba726754542e",
        "NT-v1",
        embedding_dim=2560,
        loader=Loader(missing_keys_ignore=("pooler",)),
        included_in=("hg38_multilen", "ood_1000g", "identification"),
    ),
    # The three below keep their own generators: DNABERT-2 needs an import-time
    # Triton patch, NTv2 still carries the legacy per-token path, and Evo 2 is
    # not a transformers model at all. Their revisions were never pinned.
    ModelSpec(
        "dnabert2",
        "zhihan1996/DNABERT-2-117M",
        "",
        "DNABERT-2",
        decoder_tokenizer="huggingface",
        embedding_dim=768,
        requires=("einops", "triton"),
        included_in=("hg38_multilen", "ood_1000g", "identification"),
        engine_supported=False,
    ),
    ModelSpec(
        "evo2",
        "evo2_7b",
        "",
        "Evo 2",
        embedding_dim=4096,
        requires=("evo2",),
        included_in=("hg38_multilen", "ood_1000g", "identification"),
        engine_supported=False,
    ),
    ModelSpec(
        "ntv2",
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "",
        "NTv2",
        decoder_tokenizer="huggingface",
        embedding_dim=1024,
        loader=Loader(model_class="AutoModelForMaskedLM"),
        included_in=("hg38_multilen", "ood_1000g", "identification"),
        engine_supported=False,
    ),
)

REGISTRY: dict[str, ModelSpec] = {s.slug: s for s in _SPECS}
assert len(REGISTRY) == len(_SPECS), "duplicate slug in the registry"


def get(slug: str) -> ModelSpec:
    """Return the spec for ``slug``."""
    assert slug in REGISTRY, f"unknown fm_slug {slug!r}; known: {', '.join(REGISTRY)}"
    return REGISTRY[slug]


def slugs() -> List[str]:
    """Every registered slug, in registry order."""
    return list(REGISTRY)


def slugs_for(tag: str) -> List[str]:
    """Slugs tagged ``tag`` in ``included_in``, in registry order."""
    return [s for s, spec in REGISTRY.items() if tag in spec.included_in]


def generator_argv(slug: str) -> list[str]:
    """Script and overrides that produce embeddings for ``slug``."""
    spec = get(slug)
    if spec.engine_supported:
        return ["generate/generate_embeddings.py", f"fm_slug={slug}"]
    return [f"generate/generate_{slug}_embeddings.py"]
