"""Locus-blind identification scoring.

The attacker has only the predicted (inverted) sequences and a list of N
candidate individuals' phased genotypes. They do **not** know which genomic
locus each predicted sequence came from. Algorithm:

    for each predicted sequence q:
        align q to hg38 with BWA-MEM (substitution-only parameters)
        for each top hit (chrom, pos):
            for each candidate c, haplotype h:
                d_hit[c, h] = Hamming distance between q and c's haplotype at this position
        d_q[c] = min over (hits, hap) of d_hit[c, h]

    aggregate over k predicted sequences for target t:
        D[c] = sum_{i in 1..k} d_q_i[c]
    predicted identity = argmin_c D[c]

This module produces the per-(query, candidate) distance array. Downstream
identification sweeps live in ``src.evaluation.identification``.
"""

from __future__ import annotations

from bisect import bisect_right
import logging
import os
import shutil
import subprocess
from typing import Dict, List, Tuple

import Levenshtein
import numpy as np

from src.utils import is_topk_hit

CHROMS_AUTOSOMAL = [f"chr{i}" for i in range(1, 23)]

# BWA-MEM minimum seed length. Queries shorter than this cannot seed, so
# compute_locus_blind_distances refuses inputs below it rather than aligning
# nothing and returning a uniform (uninformative) distance matrix.
_BWA_SEED_LENGTH = 12


def _which(name: str) -> str:
    path = shutil.which(name)
    assert path is not None, f"{name} not on PATH; run init.sh to install it"
    return path


def _ensure_bwa_index(reference: str) -> None:
    """Fail fast if the BWA index is missing."""
    for ext in (".bwt", ".pac", ".sa", ".ann", ".amb"):
        idx_path = reference + ext
        assert os.path.exists(idx_path), (
            f"BWA index file missing: {idx_path}. Run scripts/build_bwa_index.sh first."
        )


def _ensure_fai(reference: str) -> Dict[str, int]:
    """Read chromosome lengths from <reference>.fai. The .fai must already exist."""
    fai = reference + ".fai"
    assert os.path.exists(fai), f"Missing FASTA index: {fai}. Run samtools faidx."
    lengths: Dict[str, int] = {}
    with open(fai, "r") as f:
        for line in f:
            parts = line.split("\t")
            lengths[parts[0]] = int(parts[1])
    return lengths


def _write_fasta(queries: List[str], path: str) -> None:
    with open(path, "w") as f:
        for i, q in enumerate(queries):
            f.write(f">q{i}\n{q}\n")


def _fasta_matches(queries: List[str], path: str) -> bool:
    """Return whether ``path`` is exactly the indexed FASTA for ``queries``."""
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        for i, query in enumerate(queries):
            if f.readline().rstrip("\n") != f">q{i}":
                return False
            if f.readline().rstrip("\n") != query:
                return False
        return f.read(1) == ""


def _bwa_align(
    bwa: str,
    reference: str,
    queries_fa: str,
    sam_out: str,
    threads: int,
) -> None:
    """Run BWA-MEM with gap penalties high enough to force ungapped alignment.

    ``-a`` emits all secondary alignments. ``-O/-E`` push gap penalties so the
    optimal alignment is substitution-only (length-preserving), which is what
    the Hamming scorer expects.
    """
    cmd = [
        bwa,
        "mem",
        "-t",
        str(threads),
        "-a",
        "-k",
        str(_BWA_SEED_LENGTH),
        "-T",
        "0",
        "-O",
        "100",
        "-E",
        "100",
        "-L",
        "100",
        reference,
        queries_fa,
    ]
    with open(sam_out, "w") as out:
        result = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True, check=False)
    assert result.returncode == 0, f"bwa mem failed: {result.stderr}"


def _parse_sam_hits(
    sam_path: str,
    n_queries: int,
    top_h_hits: int,
    query_lengths: List[int],
    require_all_queries: bool = False,
) -> List[List[Tuple[str, int, int]]]:
    """Return ``hits[query_idx]`` = list of up to ``top_h_hits`` (chrom, start_1based, nm) tuples.

    Reverse-strand hits, unmapped reads, hits with CIGAR containing indels, and
    hits whose target chromosome is not autosomal are dropped. ``nm`` is the
    edit distance reported by BWA.
    """
    assert len(query_lengths) == n_queries
    hits: List[List[Tuple[str, int, int]]] = [[] for _ in range(n_queries)]
    seen_queries = np.zeros(n_queries, dtype=np.bool_) if require_all_queries else None
    autosomal = set(CHROMS_AUTOSOMAL)
    with open(sam_path, "r") as f:
        for line in f:
            if line.startswith("@"):
                continue
            parts = line.rstrip("\n").split("\t")
            assert len(parts) >= 11, f"Malformed SAM line: {line!r}"
            qname = parts[0]
            assert qname.startswith("q"), f"Unexpected qname format: {qname}"
            q_idx = int(qname[1:])
            assert 0 <= q_idx < n_queries, (
                f"SAM query index {q_idx} outside expected range [0, {n_queries})"
            )
            if seen_queries is not None:
                seen_queries[q_idx] = True
            flag = int(parts[1])
            chrom = parts[2]
            pos = int(parts[3])
            cigar = parts[5]
            if flag & 0x4:
                continue
            if flag & 0x10:
                continue
            if chrom not in autosomal:
                continue
            # Require the complete EOS-decoded query to align, without soft
            # clipping or truncation. Over- and under-length predictions are
            # therefore exposed to the coordinate-recovery stage exactly as
            # emitted by the decoder.
            if cigar != f"{query_lengths[q_idx]}M":
                continue
            nm = -1
            for tag in parts[11:]:
                if tag.startswith("NM:i:"):
                    nm = int(tag[5:])
                    break
            assert nm >= 0, f"Missing NM tag in SAM record for {qname}"
            hits[q_idx].append((chrom, pos, nm))

    if seen_queries is not None:
        missing = int((~seen_queries).sum())
        assert missing == 0, (
            f"Refusing to reuse incomplete SAM {sam_path}: "
            f"{missing}/{n_queries} query IDs are absent"
        )

    for q_idx in range(n_queries):
        hits[q_idx].sort(key=lambda x: x[2])
        hits[q_idx] = hits[q_idx][:top_h_hits]
    return hits


def _load_reference(reference: str) -> Dict[str, str]:
    """Load autosomal chromosomes from FASTA (uppercase). Uses miniFasta."""
    import miniFasta as mf

    fa_objects = mf.read(reference, upper=True)
    out: Dict[str, str] = {}
    for obj in fa_objects:
        name = obj.head.lstrip(">").split()[0]
        if name in CHROMS_AUTOSOMAL:
            out[name] = obj.body
    return out


def _fetch_cohort_snps_at_windows(
    bcftools: str,
    vcf_dir: str,
    windows: List[Tuple[str, int, int]],
    samples: List[str],
    workdir: str,
) -> Dict[Tuple[str, int, int], List[Dict]]:
    """For each (chrom, start_1, end_1) window, return list of biallelic SNP records.

    Each record contains: pos (1-based int), ref, alt, and per-sample phased
    genotype strings aligned to ``samples`` order.
    """
    samples_file = os.path.join(workdir, "cohort_samples.txt")
    with open(samples_file, "w") as f:
        f.write("\n".join(samples) + "\n")

    by_chrom: Dict[str, List[Tuple[int, int]]] = {}
    for chrom, s, e in windows:
        by_chrom.setdefault(chrom, []).append((s, e))

    out: Dict[Tuple[str, int, int], List[Dict]] = {w: [] for w in windows}

    for chrom, intervals in by_chrom.items():
        vcf_path = os.path.join(vcf_dir, f"{chrom}.vcf.gz")
        assert os.path.exists(vcf_path), f"Missing VCF for {chrom}: {vcf_path}"
        # Merge overlapping intervals to keep the region list small for bcftools.
        merged: List[Tuple[int, int]] = []
        for s, e in sorted(intervals):
            if merged and s <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Do not pass the regions as a comma-separated ``-r`` argument. A full
        # identification run can produce hundreds of thousands of distinct hit
        # windows, which exceeds Linux ARG_MAX before bcftools is even started.
        # ``-R`` reads the same regions from disk and keeps the process argument
        # vector bounded independently of cohort/run size.
        regions_file = os.path.join(workdir, f"cohort_hit_regions_{chrom}.tsv")
        with open(regions_file, "w") as f:
            for s, e in merged:
                f.write(f"{chrom}\t{s}\t{e}\n")
        cmd = [
            bcftools,
            "query",
            "-R",
            regions_file,
            "-S",
            samples_file,
            "-f",
            "%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n",
            "-i",
            'TYPE="snp" && N_ALT=1',
            vcf_path,
        ]
        # A line contains one genotype column per cohort member, so even a
        # moderate region set can yield hundreds of MB. Spool stdout to disk
        # rather than holding the complete chromosome response in memory.
        query_output = os.path.join(workdir, f"cohort_snps_{chrom}.tsv")
        with open(query_output, "w") as query_out:
            result = subprocess.run(
                cmd,
                stdout=query_out,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        assert result.returncode == 0, f"bcftools failed for {chrom}: {result.stderr}"

        sorted_iv = sorted(intervals)
        starts = [s for s, _e in sorted_iv]
        # prefix_max_end lets each returned SNP find only the windows that can
        # contain it. The previous scan restarted at the first interval for
        # every SNP, making assignment O(num_snps * num_windows) after the
        # bcftools query had completed.
        prefix_max_end: List[int] = []
        max_end = -1
        for _s, e in sorted_iv:
            max_end = max(max_end, e)
            prefix_max_end.append(max_end)
        with open(query_output, "r") as query_in:
            for raw_line in query_in:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                assert len(parts) == 4 + len(samples), (
                    f"Unexpected bcftools query line column count: got {len(parts)}, "
                    f"expected {4 + len(samples)}"
                )
                pos = int(parts[1])
                ref = parts[2].upper()
                alt = parts[3].upper()
                assert len(ref) == 1 and len(alt) == 1
                gts = parts[4:]
                record = {"pos": pos, "ref": ref, "alt": alt, "genotypes": gts}
                idx = bisect_right(starts, pos) - 1
                while idx >= 0 and prefix_max_end[idx] >= pos:
                    s, e = sorted_iv[idx]
                    if e >= pos:
                        out[(chrom, s, e)].append(record)
                    idx -= 1
        os.remove(query_output)
    return out


def _genotype_matrix(snps: List[Dict], n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-haplotype ALT-carrier indicator matrices for a window.

    ``G_h0[c, s]`` is 1 if sample ``c`` carries the ALT on haplotype 0 at SNP
    ``s``, else 0. Same for ``G_h1`` and haplotype 1.
    """
    n_snps = len(snps)
    G_h0 = np.zeros((n_samples, n_snps), dtype=np.uint8)
    G_h1 = np.zeros((n_samples, n_snps), dtype=np.uint8)
    for s_idx, snp in enumerate(snps):
        gts = snp["genotypes"]
        for c_idx, gt in enumerate(gts):
            assert "|" in gt, f"Unphased GT encountered in locus-blind eval: {gt}"
            a0, a1 = gt.split("|")
            assert a0 in ("0", "1") and a1 in ("0", "1"), f"Non-biallelic GT: {gt}"
            if a0 == "1":
                G_h0[c_idx, s_idx] = 1
            if a1 == "1":
                G_h1[c_idx, s_idx] = 1
    return G_h0, G_h1


def _candidate_haplotypes(
    ref_window: str,
    window_start: int,
    snps: List[Dict],
    n_samples: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Collapse cohort haplotypes at one window to unique sequence strings.

    Returns the unique candidate sequences plus one inverse index per phased
    haplotype. This lets every query compute exact Levenshtein distance only to
    distinct haplotypes rather than repeating it for all cohort members.
    """
    G_h0, G_h1 = _genotype_matrix(snps, n_samples)
    unique: List[str] = []
    signature_to_idx: Dict[bytes, int] = {}

    def intern(signature: np.ndarray) -> int:
        key = signature.tobytes()
        existing = signature_to_idx.get(key)
        if existing is not None:
            return existing
        chars = list(ref_window)
        for carries_alt, snp in zip(signature, snps):
            offset = snp["pos"] - window_start
            assert 0 <= offset < len(chars)
            assert chars[offset] == snp["ref"], (
                f"Reference/VCF base mismatch at offset {offset}: "
                f"FASTA={chars[offset]!r}, VCF REF={snp['ref']!r}"
            )
            if carries_alt:
                chars[offset] = snp["alt"]
        idx = len(unique)
        signature_to_idx[key] = idx
        unique.append("".join(chars))
        return idx

    h0_inverse = np.fromiter(
        (intern(G_h0[sample]) for sample in range(n_samples)),
        dtype=np.int32,
        count=n_samples,
    )
    h1_inverse = np.fromiter(
        (intern(G_h1[sample]) for sample in range(n_samples)),
        dtype=np.int32,
        count=n_samples,
    )
    return unique, h0_inverse, h1_inverse


def compute_locus_blind_distances(
    predicted: List[str],
    individual_ids_per_row: List[str],
    cohort_samples: List[str],
    seq_length: int,
    reference_path: str,
    vcf_dir: str,
    top_h_hits: int,
    bwa_threads: int,
    workdir: str,
    logger: logging.Logger,
    reuse_alignment_if_present: bool = False,
) -> np.ndarray:
    """Run the locus-blind alignment + scoring pipeline.

    Parameters
    ----------
    predicted
        ``2*N*K`` predicted sequences (one per identification-H5 row).
    individual_ids_per_row
        Parallel array of true individual labels for each predicted sequence.
    cohort_samples
        Ordered list of cohort sample names. ``individual_ids_per_row`` values
        must all be present here.
    seq_length, reference_path, vcf_dir
        Reference + VCF inputs for haplotype reconstruction at hit positions.
    top_h_hits
        Keep up to this many top hits per query (ranked by NM ascending).
    bwa_threads
        Threads passed to ``bwa mem``.
    workdir
        Directory for intermediate FASTA/SAM/genotype files.
    logger
        Caller's logger.

    Returns
    -------
    np.ndarray
        Distance matrix of shape ``(n_queries, N)`` where ``N = len(cohort_samples)``.
        Entry ``[q, c]`` is the minimum (over hits, over candidate haplotype)
        normalized Levenshtein distance between the complete EOS-decoded query
        ``q`` and candidate ``c``'s true length-``seq_length`` haplotype.
        Queries with no usable hit get 1.0 for every candidate (max distance).
    """
    assert seq_length >= _BWA_SEED_LENGTH, (
        f"locus-blind alignment requires seq_length >= {_BWA_SEED_LENGTH} (the BWA-MEM "
        f"seed length); got {seq_length}. Shorter queries cannot seed, which would "
        f"leave every candidate equidistant."
    )

    bwa = _which("bwa")
    bcftools = _which("bcftools")
    _ensure_bwa_index(reference_path)
    chrom_lengths = _ensure_fai(reference_path)

    n_queries = len(predicted)
    n_cohort = len(cohort_samples)
    assert n_queries == len(individual_ids_per_row)
    assert all(ind in set(cohort_samples) for ind in individual_ids_per_row)

    os.makedirs(workdir, exist_ok=True)

    # Preserve every EOS-decoded sequence exactly. In particular, do not use
    # the known true length to trim over-long predictions: that would give the
    # attacker privileged information and remove insertion errors before both
    # alignment and candidate scoring.
    queries = list(predicted)
    n_non_nominal = sum(1 for q in queries if len(q) != seq_length)
    logger.info(
        f"[locus-blind] aligning full EOS output; {n_non_nominal}/{n_queries} "
        f"reconstructions differ from nominal length {seq_length}"
    )

    queries_fa = os.path.join(workdir, "queries.fa")
    sam_out = os.path.join(workdir, "queries.sam")

    logger.info(f"[locus-blind] writing {n_queries} queries to {queries_fa}")
    reuse_alignment = (
        reuse_alignment_if_present
        and os.path.isfile(sam_out)
        and os.path.getsize(sam_out) > 0
        and _fasta_matches(queries, queries_fa)
    )
    if reuse_alignment:
        logger.info(
            f"[locus-blind] reusing matching existing alignment: {sam_out}"
        )
    else:
        _write_fasta(queries, queries_fa)
        logger.info(f"[locus-blind] running bwa mem (threads={bwa_threads}) ...")
        _bwa_align(bwa, reference_path, queries_fa, sam_out, bwa_threads)

    logger.info("[locus-blind] parsing SAM hits ...")
    hits_per_query = _parse_sam_hits(
        sam_out,
        n_queries,
        top_h_hits,
        [len(query) for query in queries],
        require_all_queries=reuse_alignment,
    )
    n_with_hits = sum(1 for h in hits_per_query if len(h) > 0)
    logger.info(
        f"[locus-blind] {n_with_hits}/{n_queries} queries have at least one usable hit"
    )

    # Collect unique hit windows.
    unique_windows: Dict[Tuple[str, int, int], None] = {}
    for hits in hits_per_query:
        for chrom, pos, _nm in hits:
            assert pos >= 1 and pos + seq_length - 1 <= chrom_lengths[chrom]
            unique_windows[(chrom, pos, pos + seq_length - 1)] = None
    windows_list = list(unique_windows.keys())
    logger.info(f"[locus-blind] {len(windows_list)} unique hit windows")

    logger.info("[locus-blind] loading reference into memory ...")
    chrom_to_seq = _load_reference(reference_path)

    logger.info("[locus-blind] fetching cohort SNPs at hit windows via bcftools ...")
    snps_by_window = _fetch_cohort_snps_at_windows(
        bcftools, vcf_dir, windows_list, cohort_samples, workdir
    )

    logger.info("[locus-blind] scoring queries vs cohort haplotypes ...")
    # Initialize all distances to max (= 1.0) so unmapped queries are
    # uninformative (equal to every candidate).
    # This matrix is ~16 GB for N=1000, K=4096 and two haplotypes. Keep it as a
    # disk-backed memmap so locus-blind evaluation does not require that much
    # additional RAM; NumPy indexing in the downstream sweep remains unchanged.
    distance_path = os.path.join(workdir, "locus_blind_distances.uint16.mmap")
    distances = np.memmap(
        distance_path,
        mode="w+",
        dtype=np.float16,
        shape=(n_queries, n_cohort),
    )
    distances[:] = 1.0

    # Precompute unique true haplotype strings and candidate inverse maps per
    # window. Candidate sequences remain the nominal true length; predictions
    # are not trimmed to match them.
    window_cache: Dict[Tuple[str, int, int], Tuple[List[str], np.ndarray, np.ndarray]] = {}
    for w in windows_list:
        snps = snps_by_window[w]
        chrom, start, _end = w
        ref_window = chrom_to_seq[chrom][start - 1 : start - 1 + seq_length]
        window_cache[w] = _candidate_haplotypes(
            ref_window, start, snps, n_cohort
        )

    for q_idx in range(n_queries):
        hits = hits_per_query[q_idx]
        if not hits:
            continue
        q = queries[q_idx]
        best = np.ones(n_cohort, dtype=np.float32)
        for chrom, pos, _nm in hits:
            window = (chrom, pos, pos + seq_length - 1)
            unique_haps, h0_inverse, h1_inverse = window_cache[window]
            denominator = max(len(q), seq_length)
            if denominator == 0:
                unique_distances = np.zeros(len(unique_haps), dtype=np.float32)
            else:
                unique_distances = np.fromiter(
                    (
                        Levenshtein.distance(q, haplotype) / denominator
                        for haplotype in unique_haps
                    ),
                    dtype=np.float32,
                    count=len(unique_haps),
                )
            cur = np.minimum(
                unique_distances[h0_inverse], unique_distances[h1_inverse]
            )
            np.minimum(best, cur, out=best)
        distances[q_idx] = best.astype(np.float16)

        if (q_idx + 1) % 100000 == 0:
            logger.info(f"[locus-blind] scored {q_idx + 1}/{n_queries} queries")

    distances.flush()
    logger.info(f"[locus-blind] distance matrix complete: {distance_path}")
    return distances


def locus_blind_topk_sweep(
    distances: np.ndarray,
    individual_ids_per_row: List[str],
    cohort_samples: List[str],
    locus_ids: List[int],
    haplotypes: List[int],
    k_grid: List[int],
    num_target_repetitions: int,
    seed: int,
    logger: logging.Logger,
    allowed_locus_ids: set[int] | None = None,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Identification sweep over the locus-blind distance matrix.

    For each k, repeatedly sample k loci per target, sum the per-embedding
    distances over those loci (using BOTH haplotypes — i.e., 2k rows), then
    take the argmin over candidates. Reports mean top-1 / top-5 accuracy.
    """
    rng = np.random.default_rng(seed)
    cohort_idx = {ind: i for i, ind in enumerate(cohort_samples)}

    n_rows = len(individual_ids_per_row)
    observed_loci = set(locus_ids)
    if allowed_locus_ids is not None:
        assert allowed_locus_ids <= observed_loci
        observed_loci = allowed_locus_ids
    all_loci_sorted = sorted(observed_loci)
    n_loc = len(all_loci_sorted)
    n_ind = len(cohort_samples)
    locus_to_idx = {locus: idx for idx, locus in enumerate(all_loci_sorted)}

    # Dense row lookup is only 2*N*K int32 values (~33 MB at full scale) and
    # turns the Python target/locus loops into one vectorized gather per locus.
    row_lookup = np.full((n_ind, n_loc, 2), -1, dtype=np.int32)
    for row_idx in range(n_rows):
        if locus_ids[row_idx] not in locus_to_idx:
            continue
        target = cohort_idx[individual_ids_per_row[row_idx]]
        locus = locus_to_idx[locus_ids[row_idx]]
        hap = int(haplotypes[row_idx])
        assert hap in (0, 1)
        assert row_lookup[target, locus, hap] == -1, (
            f"Duplicate row for target={target}, locus={locus_ids[row_idx]}, hap={hap}"
        )
        row_lookup[target, locus, hap] = row_idx
    assert np.all(row_lookup >= 0), "Identification rows do not cover every target/locus/haplotype"

    valid_k = sorted({int(k) for k in k_grid if 0 < int(k) <= n_loc})
    assert valid_k and num_target_repetitions > 0
    max_k = max(valid_k)
    top1_by_k: Dict[int, List[float]] = {k: [] for k in valid_k}
    top5_by_k: Dict[int, List[float]] = {k: [] for k in valid_k}

    for repetition in range(num_target_repetitions):
        chosen = rng.permutation(n_loc)[:max_k]
        aggregate = np.zeros((n_ind, n_ind), dtype=np.float32)
        for step, locus_idx in enumerate(chosen, start=1):
            rows = row_lookup[:, int(locus_idx), :].reshape(-1)
            locus_distances = np.asarray(distances[rows], dtype=np.float32).reshape(
                n_ind, 2, n_ind
            )
            aggregate += locus_distances.sum(axis=1)
            if step in top1_by_k:
                hits1 = sum(
                    is_topk_hit(aggregate[t], t, 1, rng, descending=False)
                    for t in range(n_ind)
                )
                hits5 = sum(
                    is_topk_hit(
                        aggregate[t], t, min(5, n_ind), rng, descending=False
                    )
                    for t in range(n_ind)
                )
                top1_by_k[step].append(hits1 / n_ind)
                top5_by_k[step].append(hits5 / n_ind)
        logger.info(
            f"[locus-blind] repetition {repetition + 1}/{num_target_repetitions} complete"
        )

    k_to_top1 = {k: float(np.mean(values)) for k, values in top1_by_k.items()}
    k_to_top5 = {k: float(np.mean(values)) for k, values in top5_by_k.items()}
    for k in valid_k:
        logger.info(
            f"[locus-blind] k={k}: top-1={k_to_top1[k]:.3f}, "
            f"top-5={k_to_top5[k]:.3f}"
        )
    return k_to_top1, k_to_top5
