"""Identification-style evaluation for the DNA embedding inversion attack.

Given a per-individual identification HDF5 (produced by the identification-mode
embedding generators), this module:
    1. Loads embeddings and (individual_id, locus_id, haplotype) labels.
    2. Inverts every embedding with the trained reconstruction model.
    3. For each target individual ``t`` and budget ``k``, scores every candidate
       ``c`` in the cohort by aggregated Levenshtein similarity over ``k`` sampled
       loci. Records top-1 / top-5 accuracy.
    4. Reports the smallest ``k`` at which top-1 accuracy first reaches each
       configured threshold (default 0.95).
    5. Emits a plot of top-1 accuracy vs. ``k`` per locus mode.

The identification block is invoked from ``evaluate.py`` / ``evaluate_1000g.py``
after the standard reconstruction eval finishes. It reuses the already-loaded
model, tokenizer, and reconstruction utilities.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import h5py
import hydra.utils as hy_utils
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.evaluate import reconstruct_sequences, levenshtein_similarity
from src.evaluation.locus_blind import (
    compute_locus_blind_distances,
    locus_blind_topk_sweep,
)
from src.plotting_utils import (
    PLOT_STYLE,
    _save_fig,
    configure_plot_style,
    get_series_color,
)
from src.utils import save_json, is_topk_hit


def _resolve_h5_path(pattern: str, foundation_model: str, seq_length: int, mode: str) -> str:
    """Resolve a Hydra-style identification H5 path pattern using {fm}/{L}/{mode}."""
    return pattern.format(fm=foundation_model, L=seq_length, mode=mode)


class _H5EmbeddingView:
    """Sliceable lazy view that keeps large identification embeddings on disk."""

    def __init__(self, path: str):
        self._handle = h5py.File(path, "r", swmr=True)
        self._dataset = self._handle["embeddings"]

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, item):
        return self._dataset[item]

    def close(self) -> None:
        self._handle.close()


def _load_identification_h5(
    path: str,
) -> Tuple[List[str], _H5EmbeddingView, List[str], List[int], List[int]]:
    """Load sequences, embeddings, and per-row labels from an identification H5.

    Embeddings are returned as a list of 1D float32 arrays exactly as stored
    (flattened per-token, or mean-pooled vectors).
    """
    with h5py.File(path, "r", swmr=True) as f:
        assert "identification_mode" in f.attrs, (
            f"Identification H5 missing 'identification_mode' attribute: {path}"
        )
        assert "individual_ids" in f, f"Missing 'individual_ids' dataset: {path}"
        assert "locus_ids" in f, f"Missing 'locus_ids' dataset: {path}"
        assert "haplotypes" in f, f"Missing 'haplotypes' dataset: {path}"

        n = len(f["sequences"])
        sequences = [
            (s.decode("utf-8") if isinstance(s, bytes) else str(s)) for s in f["sequences"][:]
        ]
        individual_ids = [
            (s.decode("utf-8") if isinstance(s, bytes) else str(s))
            for s in f["individual_ids"][:]
        ]
        locus_ids = list(map(int, f["locus_ids"][:]))
        haplotypes = list(map(int, f["haplotypes"][:]))
    # Do not materialise the embedding matrix: an Evo 2 N=1000/K=4096 cohort
    # exceeds 130 GB. reconstruct_sequences consumes this view in GPU batches.
    return sequences, _H5EmbeddingView(path), individual_ids, locus_ids, haplotypes


def _build_lookup_tables(
    individual_ids: List[str],
    locus_ids: List[int],
    haplotypes: List[int],
    sequences: List[str],
    predicted: List[str],
) -> Dict[str, Any]:
    """Pivot the flat arrays into (individual, locus) -> per-haplotype dicts."""
    individuals = sorted(set(individual_ids))
    loci = sorted(set(locus_ids))
    ind_to_idx = {ind: i for i, ind in enumerate(individuals)}
    loc_to_idx = {loc: i for i, loc in enumerate(loci)}

    n_ind = len(individuals)
    n_loc = len(loci)

    # For each (ind, locus), we store up to two haplotypes.
    # Some haplotypes may be absent if the prep wrote fewer; assert population.
    true_h0 = np.full((n_ind, n_loc), "", dtype=object)
    true_h1 = np.full((n_ind, n_loc), "", dtype=object)
    pred_h0 = np.full((n_ind, n_loc), "", dtype=object)
    pred_h1 = np.full((n_ind, n_loc), "", dtype=object)

    for row_idx in range(len(sequences)):
        ind_i = ind_to_idx[individual_ids[row_idx]]
        loc_i = loc_to_idx[locus_ids[row_idx]]
        hap = haplotypes[row_idx]
        if hap == 0:
            true_h0[ind_i, loc_i] = sequences[row_idx]
            pred_h0[ind_i, loc_i] = predicted[row_idx]
        else:
            assert hap == 1, f"Unexpected haplotype value {hap}"
            true_h1[ind_i, loc_i] = sequences[row_idx]
            pred_h1[ind_i, loc_i] = predicted[row_idx]

    # All (ind, locus) cells must be populated for both haplotypes.
    for arr, name in [
        (true_h0, "true_h0"),
        (true_h1, "true_h1"),
        (pred_h0, "pred_h0"),
        (pred_h1, "pred_h1"),
    ]:
        missing = (arr == "").sum()
        assert missing == 0, f"Identification table {name} has {missing} unfilled cells"

    return {
        "individuals": individuals,
        "loci": loci,
        "true_h0": true_h0,
        "true_h1": true_h1,
        "pred_h0": pred_h0,
        "pred_h1": pred_h1,
    }


def _load_sequence_set(path: str, chunk_size: int = 100_000) -> set[str]:
    """Load exact training strings for strict identification-locus auditing."""
    sequences: set[str] = set()
    with h5py.File(path, "r", swmr=True) as handle:
        dataset = handle["sequences"]
        for start in range(0, len(dataset), chunk_size):
            sequences.update(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in dataset[start : min(start + chunk_size, len(dataset))]
            )
    return sequences


def _subset_tables_to_loci(
    tables: Dict[str, Any], allowed_loci: set[int]
) -> Dict[str, Any]:
    """Return lookup tables restricted to whole loci in ``allowed_loci``."""
    indices = [
        index for index, locus in enumerate(tables["loci"]) if locus in allowed_loci
    ]
    assert indices, "Strict identification stratum contains no loci"
    return {
        "individuals": tables["individuals"],
        "loci": [tables["loci"][index] for index in indices],
        "true_h0": tables["true_h0"][:, indices],
        "true_h1": tables["true_h1"][:, indices],
        "pred_h0": tables["pred_h0"][:, indices],
        "pred_h1": tables["pred_h1"][:, indices],
    }


def _score_matrix(
    pred_h0: np.ndarray,
    pred_h1: np.ndarray,
    true_h0: np.ndarray,
    true_h1: np.ndarray,
    locus_idx: int,
) -> np.ndarray:
    """For a fixed locus, return an (n_ind x n_ind) similarity matrix between
    predicted (target) and true (candidate) sequences.

    Phase is arbitrary, so the score is the better of the two consistent
    haplotype pairings - (pred0<->true0, pred1<->true1) vs the swap - averaged
    over the two haplotypes. Unlike matching each predicted haplotype
    independently, this never lets both predicted haplotypes collapse onto the
    same candidate haplotype.
    """
    n = pred_h0.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    for t in range(n):
        ph0, ph1 = pred_h0[t, locus_idx], pred_h1[t, locus_idx]
        for c in range(n):
            th0, th1 = true_h0[c, locus_idx], true_h1[c, locus_idx]
            s00 = levenshtein_similarity(ph0, th0)
            s01 = levenshtein_similarity(ph0, th1)
            s10 = levenshtein_similarity(ph1, th0)
            s11 = levenshtein_similarity(ph1, th1)
            out[t, c] = 0.5 * max(s00 + s11, s01 + s10)
    return out


def _topk_accuracy(scores: np.ndarray, k: int, rng: np.random.Generator) -> float:
    """Return the fraction of rows where the diagonal entry is within the top-k of its row.

    Equal scores are broken uniformly at random so tied candidates do not bias
    the accuracy toward any particular index.
    """
    n = scores.shape[0]
    hits = sum(is_topk_hit(scores[t], t, k, rng, descending=True) for t in range(n))
    return hits / n


def _score_locus_compact(
    pred_h0: np.ndarray,
    pred_h1: np.ndarray,
    true_h0: np.ndarray,
    true_h1: np.ndarray,
    locus_idx: int,
) -> np.ndarray:
    """Return one locus score matrix while scoring repeated diplotypes once.

    A locus often has only a handful of distinct candidate diplotypes (and a
    monomorphic locus has exactly one), even in a 1,000-person cohort. The old
    nested loop recomputed the same four Levenshtein similarities for every
    carrier. This implementation scores each unique unordered diplotype once per
    target, then expands through an integer inverse map.
    """
    n_targets = pred_h0.shape[0]
    n_candidates = true_h0.shape[0]
    pair_to_idx: Dict[Tuple[str, str], int] = {}
    unique_pairs: List[Tuple[str, str]] = []
    inverse = np.empty(n_candidates, dtype=np.int32)
    for candidate in range(n_candidates):
        pair = tuple(sorted((true_h0[candidate, locus_idx], true_h1[candidate, locus_idx])))
        if pair not in pair_to_idx:
            pair_to_idx[pair] = len(unique_pairs)
            unique_pairs.append(pair)
        inverse[candidate] = pair_to_idx[pair]

    compact = np.empty((n_targets, len(unique_pairs)), dtype=np.float32)
    for target in range(n_targets):
        ph0 = pred_h0[target, locus_idx]
        ph1 = pred_h1[target, locus_idx]
        for pair_idx, (th0, th1) in enumerate(unique_pairs):
            s00 = levenshtein_similarity(ph0, th0)
            s01 = levenshtein_similarity(ph0, th1)
            s10 = levenshtein_similarity(ph1, th0)
            s11 = levenshtein_similarity(ph1, th1)
            compact[target, pair_idx] = 0.5 * max(s00 + s11, s01 + s10)
    return compact[:, inverse]


def _stream_oracle_sweep(
    tables: Dict[str, Any],
    k_grid: List[int],
    num_repetitions: int,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> Tuple[Dict[int, float], Dict[int, float], float]:
    """Compute nested-locus identification curves without a 3-D score cube.

    Memory falls from ``O(K*N^2)`` to ``O(N^2)``. Within each repetition a
    random locus permutation is accumulated once; every configured ``k`` is a
    prefix of that permutation. This is both cheaper and statistically cleaner
    than independently re-sampling all loci for every point on the curve.
    """
    assert num_repetitions > 0
    n_ind = len(tables["individuals"])
    n_loc = len(tables["loci"])
    valid_k = sorted({int(k) for k in k_grid if 0 < int(k) <= n_loc})
    assert valid_k, "No identification k values fit the available locus count"
    max_k = max(valid_k)
    top1_by_k: Dict[int, List[float]] = {k: [] for k in valid_k}
    top5_by_k: Dict[int, List[float]] = {k: [] for k in valid_k}
    diag_sum = 0.0
    diag_count = 0

    for repetition in range(num_repetitions):
        chosen = rng.permutation(n_loc)[:max_k]
        aggregate = np.zeros((n_ind, n_ind), dtype=np.float32)
        for step, locus_idx in enumerate(chosen, start=1):
            locus_scores = _score_locus_compact(
                tables["pred_h0"],
                tables["pred_h1"],
                tables["true_h0"],
                tables["true_h1"],
                int(locus_idx),
            )
            aggregate += locus_scores
            if repetition == 0:
                diag_sum += float(np.trace(locus_scores))
                diag_count += n_ind
            if step in top1_by_k:
                top1_by_k[step].append(_topk_accuracy(aggregate, 1, rng))
                top5_by_k[step].append(_topk_accuracy(aggregate, min(5, n_ind), rng))
        logger.info(
            "[identification] oracle repetition %d/%d complete (%d loci)",
            repetition + 1,
            num_repetitions,
            max_k,
        )

    return (
        {k: float(np.mean(values)) for k, values in top1_by_k.items()},
        {k: float(np.mean(values)) for k, values in top5_by_k.items()},
        diag_sum / diag_count,
    )


def _smallest_k(k_to_top1: Dict[int, float], threshold: float) -> int | None:
    """Return the smallest k in ``k_to_top1`` whose top-1 accuracy is >= threshold."""
    for k in sorted(k_to_top1.keys()):
        if k_to_top1[k] >= threshold:
            return k
    return None


def evaluate_identification_block(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    foundation_model: str,
    seq_length: int,
    mode: str,
    embedding_dim: int,
    normalization_stats: Dict[str, float],
    normalization_method: str,
    data_is_mean: bool,
    identification_cfg: Dict[str, Any],
    output_dir: str,
    logger: logging.Logger,
    model_color: str,
) -> Dict[str, Any]:
    """Run identification evaluation for all configured locus modes.

    ``normalization_stats`` are the training-set embedding statistics and are
    mandatory: the cohort HDF5 stores raw embeddings, so omitting them feeds the
    decoder inputs scaled by 1/std relative to training. The damage scales with
    1/std, which silently destroys the attack for foundation models whose
    activations are small (Evo 2, std 0.05) while merely degrading the others.

    Returns a dict (also written to ``identification_results.json``) summarising
    top-1 / top-5 accuracy vs. k for each locus mode, plus the smallest k for
    each threshold.
    """
    assert normalization_stats, (
        "normalization_stats is required; identification must reproduce the "
        "z-scoring applied during training."
    )
    modes = list(identification_cfg["modes"])
    k_grid = list(identification_cfg["k_grid"])
    thresholds = list(identification_cfg["thresholds"])
    h5_pattern = identification_cfg["h5_pattern"]
    seed = int(identification_cfg["seed"])
    num_target_reps = int(identification_cfg["num_target_repetitions"])
    lb_enabled = bool(identification_cfg["locus_blind"]["enabled"])
    lb_reference = identification_cfg["locus_blind"]["reference"]
    lb_vcf_dir = identification_cfg["locus_blind"]["vcf_dir"]
    lb_top_h_hits = int(identification_cfg["locus_blind"]["top_h_hits"])
    lb_bwa_threads = int(identification_cfg["locus_blind"]["bwa_threads"])
    lb_workdir_root = identification_cfg["locus_blind"]["workdir_root"]
    lb_reuse_alignment = bool(
        identification_cfg["locus_blind"].get("reuse_alignment_if_present", False)
    )
    strict_enabled = bool(
        identification_cfg.get("strict_train_unseen_loci", False)
    )
    training_sequences: set[str] | None = None
    if strict_enabled:
        training_h5 = identification_cfg.get("training_sequence_h5")
        assert training_h5 is not None, (
            "strict_train_unseen_loci requires training_sequence_h5 from the "
            "evaluated reconstruction run"
        )
        training_h5 = hy_utils.to_absolute_path(training_h5)
        logger.info(
            "[identification] loading exact training strings for strict audit: %s",
            training_h5,
        )
        training_sequences = _load_sequence_set(training_h5)

    rng = np.random.default_rng(seed)
    all_results: Dict[str, Any] = {
        "protocol": {
            "seq_length": seq_length,
            "k_grid_requested": k_grid,
            "num_target_repetitions": num_target_reps,
            "seed": seed,
            "locus_blind_enabled": lb_enabled,
            "locus_blind_top_h_hits": lb_top_h_hits,
            "strict_train_unseen_loci": strict_enabled,
        },
        "per_mode": {},
    }

    for locus_mode in modes:
        h5_rel = _resolve_h5_path(h5_pattern, foundation_model, seq_length, locus_mode)
        h5_abs = hy_utils.to_absolute_path(h5_rel)
        assert os.path.exists(h5_abs), (
            f"Identification H5 missing for mode={locus_mode}: {h5_abs}. "
            f"Remove this mode from cfg.identification.modes or run "
            f"prepare_and_generate_identification.sh --full first."
        )

        logger.info(f"[identification] mode={locus_mode} loading {h5_abs}")
        true_seqs, embeddings, ind_ids, loc_ids, haps = _load_identification_h5(h5_abs)
        logger.info(
            f"[identification] mode={locus_mode}: {len(true_seqs)} rows, "
            f"{len(set(ind_ids))} individuals x {len(set(loc_ids))} loci"
        )

        # Reconstruct predicted sequences.
        recon_data = {"embeddings": embeddings}
        try:
            predicted = reconstruct_sequences(
                model,
                recon_data,
                device,
                tokenizer,
                mode,
                seq_length,
                embedding_dim,
                normalization_stats=normalization_stats,
                normalization_method=normalization_method,
                data_is_mean=data_is_mean,
                inference_batch_size=int(identification_cfg.get("inference_batch_size", 256)),
            )
        finally:
            embeddings.close()
        # Length is decided by the model via EOS; no external truncation.
        assert len(predicted) == len(true_seqs)

        # A decoder fed mis-scaled inputs stops responding to them and emits
        # essentially one sequence for every locus. The identification sweep
        # still produces a smooth-looking (chance-level) curve in that case, so
        # check input-responsiveness explicitly rather than trusting the curve.
        sample = predicted[:: max(1, len(predicted) // 20000)]
        distinct_frac = len(set(sample)) / len(sample)
        logger.info(
            f"[identification] mode={locus_mode}: reconstruction diversity "
            f"{distinct_frac:.4f} distinct over {len(sample)} sampled rows"
        )
        assert distinct_frac > 0.01, (
            f"Decoder emitted only {len(set(sample))} distinct sequences over "
            f"{len(sample)} sampled rows: it is ignoring its input. This is the "
            f"signature of an embedding-scaling mismatch -- verify that "
            f"normalization_stats match the run's training statistics."
        )

        # Pivot.
        tables = _build_lookup_tables(ind_ids, loc_ids, haps, true_seqs, predicted)
        n_ind = len(tables["individuals"])
        n_loc = len(tables["loci"])
        logger.info(
            f"[identification] mode={locus_mode}: streaming compact oracle scores "
            f"({n_loc} loci, {n_ind} candidates, repetitions={num_target_reps})..."
        )
        k_to_top1, k_to_top5, levenshtein_diag_mean = _stream_oracle_sweep(
            tables=tables,
            k_grid=k_grid,
            num_repetitions=num_target_reps,
            rng=rng,
            logger=logger,
        )
        epsilon_emp = 1.0 - levenshtein_diag_mean

        k_star: Dict[str, int | None] = {
            f"top1_geq_{t}": _smallest_k(k_to_top1, t) for t in thresholds
        }

        mode_results: Dict[str, Any] = {
            "h5": h5_abs,
            "num_individuals": n_ind,
            "num_loci": n_loc,
            "epsilon_emp": epsilon_emp,
            "levenshtein_diag_mean": levenshtein_diag_mean,
            "k_grid": sorted(k_to_top1.keys()),
            "oracle": {
                "top1": k_to_top1,
                "top5": k_to_top5,
                "smallest_k": k_star,
            },
        }

        strict_loci: set[int] | None = None
        if strict_enabled:
            assert training_sequences is not None
            contaminated_loci = {
                locus_id
                for sequence, locus_id in zip(true_seqs, loc_ids, strict=True)
                if sequence in training_sequences
            }
            strict_loci = set(tables["loci"]) - contaminated_loci
            assert strict_loci, (
                f"All {n_loc} identification loci contain a sequence seen in training"
            )
            strict_tables = _subset_tables_to_loci(tables, strict_loci)
            strict_top1, strict_top5, strict_diag_mean = _stream_oracle_sweep(
                tables=strict_tables,
                k_grid=k_grid,
                num_repetitions=num_target_reps,
                rng=np.random.default_rng(seed + 10_000),
                logger=logger,
            )
            mode_results["strict_train_unseen_loci"] = {
                "definition": (
                    "Exclude a whole locus if any cohort haplotype sequence at "
                    "that locus occurs exactly in reconstruction training data."
                ),
                "num_loci": len(strict_loci),
                "num_excluded_loci": len(contaminated_loci),
                "fraction_loci_retained": len(strict_loci) / n_loc,
                "epsilon_emp": 1.0 - strict_diag_mean,
                "levenshtein_diag_mean": strict_diag_mean,
                "k_grid": sorted(strict_top1),
                "oracle": {
                    "top1": strict_top1,
                    "top5": strict_top5,
                    "smallest_k": {
                        f"top1_geq_{threshold}": _smallest_k(
                            strict_top1, threshold
                        )
                        for threshold in thresholds
                    },
                },
            }
            logger.info(
                "[identification] mode=%s strict audit retained %d/%d loci",
                locus_mode,
                len(strict_loci),
                n_loc,
            )

        lb_k_to_top1: Dict[int, float] = {}
        lb_k_to_top5: Dict[int, float] = {}
        if lb_enabled:
            logger.info(
                f"[identification] mode={locus_mode}: starting locus-blind pipeline"
            )
            workdir = os.path.join(
                hy_utils.to_absolute_path(lb_workdir_root),
                f"{foundation_model}_L{seq_length}_{locus_mode}",
            )
            distances = compute_locus_blind_distances(
                predicted=predicted,
                individual_ids_per_row=ind_ids,
                cohort_samples=tables["individuals"],
                seq_length=seq_length,
                reference_path=hy_utils.to_absolute_path(lb_reference),
                vcf_dir=hy_utils.to_absolute_path(lb_vcf_dir),
                top_h_hits=lb_top_h_hits,
                bwa_threads=lb_bwa_threads,
                workdir=workdir,
                logger=logger,
                reuse_alignment_if_present=lb_reuse_alignment,
            )
            lb_k_to_top1, lb_k_to_top5 = locus_blind_topk_sweep(
                distances=distances,
                individual_ids_per_row=ind_ids,
                cohort_samples=tables["individuals"],
                locus_ids=loc_ids,
                haplotypes=haps,
                k_grid=k_grid,
                num_target_repetitions=num_target_reps,
                seed=seed,
                logger=logger,
            )
            lb_k_star: Dict[str, int | None] = {
                f"top1_geq_{t}": _smallest_k(lb_k_to_top1, t) for t in thresholds
            }
            mode_results["locus_blind"] = {
                "top1": lb_k_to_top1,
                "top5": lb_k_to_top5,
                "smallest_k": lb_k_star,
                "top_h_hits": lb_top_h_hits,
            }
            if strict_loci is not None:
                strict_lb_top1, strict_lb_top5 = locus_blind_topk_sweep(
                    distances=distances,
                    individual_ids_per_row=ind_ids,
                    cohort_samples=tables["individuals"],
                    locus_ids=loc_ids,
                    haplotypes=haps,
                    k_grid=k_grid,
                    num_target_repetitions=num_target_reps,
                    seed=seed + 10_000,
                    logger=logger,
                    allowed_locus_ids=strict_loci,
                )
                strict_result = mode_results["strict_train_unseen_loci"]
                strict_result["locus_blind"] = {
                    "top1": strict_lb_top1,
                    "top5": strict_lb_top5,
                    "smallest_k": {
                        f"top1_geq_{threshold}": _smallest_k(
                            strict_lb_top1, threshold
                        )
                        for threshold in thresholds
                    },
                    "top_h_hits": lb_top_h_hits,
                }

        all_results["per_mode"][locus_mode] = mode_results

        configure_plot_style()
        plot_stem = os.path.join(output_dir, f"identification_top1_vs_k_{locus_mode}")
        ks_oracle = sorted(k_to_top1.keys())
        # Locus-blind is the headline (realistic) attack, so it carries the
        # foundation-model colour; the oracle is the idealised upper bound and is
        # drawn in grey to read as a reference rather than a result.
        oracle_color = get_series_color("random baseline")
        fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
        ax.plot(
            ks_oracle,
            [k_to_top1[k] for k in ks_oracle],
            marker="o",
            color=oracle_color,
            linewidth=PLOT_STYLE["line_width"],
            markersize=PLOT_STYLE["marker_size"],
            label="Oracle top-1",
        )
        ax.plot(
            ks_oracle,
            [k_to_top5[k] for k in ks_oracle],
            marker="s",
            color=oracle_color,
            linestyle="--",
            linewidth=PLOT_STYLE["line_width"],
            markersize=PLOT_STYLE["marker_size"],
            label="Oracle top-5",
        )
        if lb_enabled and len(lb_k_to_top1) > 0:
            ks_lb = sorted(lb_k_to_top1.keys())
            ax.plot(
                ks_lb,
                [lb_k_to_top1[k] for k in ks_lb],
                marker="o",
                color=model_color,
                linewidth=PLOT_STYLE["line_width"],
                markersize=PLOT_STYLE["marker_size"],
                label="Locus-blind top-1",
            )
            ax.plot(
                ks_lb,
                [lb_k_to_top5[k] for k in ks_lb],
                marker="s",
                color=model_color,
                linestyle="--",
                linewidth=PLOT_STYLE["line_width"],
                markersize=PLOT_STYLE["marker_size"],
                label="Locus-blind top-5",
            )
        # Faint guide lines, kept clearly lighter than the dark-grey oracle
        # series (#555555) so the two greys don't read as the same thing.
        for t in thresholds:
            ax.axhline(t, color="lightgrey", linewidth=0.8, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k (loci per target)", fontsize=PLOT_STYLE["label_fontsize"])
        ax.set_ylabel("Identification accuracy", fontsize=PLOT_STYLE["label_fontsize"])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(
            f"Identification (mode={locus_mode}, N={n_ind})",
            fontsize=PLOT_STYLE["title_fontsize"],
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=PLOT_STYLE["legend_fontsize"])
        fig.tight_layout()
        _save_fig(plot_stem)
        logger.info(f"[identification] Saved plot: {plot_stem}.pdf (and .png)")

    save_json(all_results, os.path.join(output_dir, "identification_results.json"))
    logger.info(
        f"[identification] Wrote {os.path.join(output_dir, 'identification_results.json')}"
    )
    return all_results
