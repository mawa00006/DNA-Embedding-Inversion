"""Prepare per-individual 1000G sequences for identification evaluation.

Builds a CSV with columns ``individual_id,locus_id,haplotype,sequence`` where each
row is one haplotype of one individual at one locus. The same K loci are used for
all N individuals so that, at evaluation time, an inverted embedding from a target
individual can be scored against every other candidate's ground-truth sequence at
the same locus.

Pipeline
--------
1. Read sample names from the VCF header; sample ``N`` individuals (seeded).
2. Pick ``K`` loci of length ``L`` according to ``locus_mode``:
     - ``random``: uniform autosomal windows.
     - ``snp_enriched``: windows containing >=1 PASS biallelic SNP with MAF >= maf_min.
   Loci are spaced >= ``min_locus_spacing`` bp apart to avoid LD redundancy.
3. Use ``bcftools`` to extract a per-cohort, per-locus genotype table restricted
   to biallelic SNPs (so applying variants preserves length).
4. For each (individual, locus), apply phased GTs to the reference window to
   produce two haplotype sequences.
5. Write the CSV + a manifest JSON containing locus coordinates / MAFs / seed.

Example
-------
    python scripts/prepare_identification.py \\
        --reference data/hg38.fa \\
        --vcf_dir data/1000g_vcfs \\
        --output_csv data/1000g_id_random_seq50_N1000_K2000.csv \\
        --seq_length 50 --cohort_size 1000 --num_loci 2000 \\
        --locus_mode random --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import miniFasta as mf
import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

CHROMS_AUTOSOMAL = [f"chr{i}" for i in range(1, 23)]
MULTILEN_SOURCE_SCHEMA = "dna_inversion_hg38_multilen_v2"


def _which_bcftools() -> str:
    """Return the bcftools binary path, asserting it exists on PATH."""
    path = shutil.which("bcftools")
    assert path is not None, "bcftools not found on PATH; activate the project conda env"
    return path


def _run(cmd: List[str], stdin_text: str | None = None) -> str:
    """Run a subprocess and return stdout as text. Fail fast on non-zero exit."""
    result = subprocess.run(
        cmd,
        input=stdin_text,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"Command failed ({result.returncode}): {' '.join(cmd)}\nSTDERR: {result.stderr}"
    return result.stdout


def read_vcf_samples(bcftools: str, vcf_path: str) -> List[str]:
    """Return ordered list of sample names from a VCF header."""
    out = _run([bcftools, "query", "-l", vcf_path])
    samples = [s for s in out.strip().split("\n") if s]
    assert len(samples) > 0, f"No samples found in VCF {vcf_path}"
    return samples


def read_chrom_lengths(reference: str) -> Dict[str, int]:
    """Read chromosome lengths from a FASTA index (.fai). Build it if missing."""
    fai = reference + ".fai"
    if not os.path.exists(fai):
        samtools = shutil.which("samtools")
        assert samtools is not None, f"FAI index missing and samtools not on PATH: {fai}"
        _run([samtools, "faidx", reference])
        assert os.path.exists(fai), f"samtools faidx did not produce {fai}"

    lengths: Dict[str, int] = {}
    with open(fai, "r") as f:
        for line in f:
            parts = line.split("\t")
            assert len(parts) >= 2, f"Malformed FAI line: {line.rstrip()}"
            lengths[parts[0]] = int(parts[1])
    return lengths


def load_reference_sequences(reference: str) -> Dict[str, str]:
    """Load uppercase autosomal sequences from the reference FASTA (read once).

    Uses miniFasta to read the whole FASTA once (~3GB; this is the same approach as
    ``scripts/prepare_hg38.py``). The returned per-chromosome sequences are reused
    both to reject N-containing candidate windows during locus selection and to
    slice the final per-locus windows, so the FASTA is read only a single time.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading reference FASTA: {reference}")
    fa_objects = mf.read(reference, upper=True)

    chrom_to_seq: Dict[str, str] = {}
    for obj in fa_objects:
        name = obj.head.lstrip(">").split()[0]
        if name in CHROMS_AUTOSOMAL:
            chrom_to_seq[name] = obj.body
    assert len(chrom_to_seq) == len(CHROMS_AUTOSOMAL), (
        f"Expected {len(CHROMS_AUTOSOMAL)} autosomes in {reference}, "
        f"loaded {len(chrom_to_seq)}"
    )
    return chrom_to_seq


def extract_reference_windows(
    chrom_to_seq: Dict[str, str], loci: List[Tuple[str, int, int]]
) -> Dict[Tuple[str, int, int], str]:
    """Slice the reference sequence for each (chrom, start_1based, end_inclusive) locus.

    Loci selected by ``pick_loci`` are already N-free; the N assertion here is a
    safety net.
    """
    logger = logging.getLogger(__name__)
    windows: Dict[Tuple[str, int, int], str] = {}
    for chrom, start_1, end_1 in loci:
        assert chrom in chrom_to_seq, f"Reference missing autosomal chrom {chrom}"
        seq = chrom_to_seq[chrom][start_1 - 1 : end_1]
        assert len(seq) == end_1 - start_1 + 1, (
            f"Reference window length mismatch at {chrom}:{start_1}-{end_1}: "
            f"got {len(seq)}, expected {end_1 - start_1 + 1}"
        )
        assert "N" not in seq, f"Reference window contains N: {chrom}:{start_1}-{end_1}"
        windows[(chrom, start_1, end_1)] = seq
    logger.info(f"Loaded {len(windows)} reference windows")
    return windows


def load_multilen_test_candidates(
    source_h5: str, seq_length: int
) -> List[Tuple[str, int, int]]:
    """Return candidate loci contained inside held-out multilen intervals.

    Every source interval is globally coordinate-disjoint. Taking one centred
    ``seq_length`` window from each test interval therefore gives an
    identification candidate pool that cannot overlap the reconstruction
    training/validation data or another sequence length's test locus.
    """
    candidates: List[Tuple[str, int, int]] = []
    with h5py.File(source_h5, "r", swmr=True) as handle:
        assert handle.attrs.get("schema", "") == MULTILEN_SOURCE_SCHEMA, (
            f"Identification release data requires a sequence-disjoint hg38 v2 "
            f"source: {source_h5}"
        )
        assert bool(handle.attrs["sequence_content_split_disjoint"])
        assert bool(handle.attrs["generation_complete"])
        for key in sorted(handle["lengths"].keys(), key=int):
            source_length = int(key)
            if source_length < seq_length:
                continue
            group = handle[f"lengths/{key}"]
            test_indices = np.flatnonzero(np.asarray(group["split"][:], dtype=np.uint8) == 2)
            chrom_values = group["chrom"][test_indices]
            start_values = group["start"][test_indices]
            for chrom_value, source_start_value in zip(chrom_values, start_values):
                chrom = (
                    chrom_value.decode("utf-8")
                    if isinstance(chrom_value, bytes)
                    else str(chrom_value)
                )
                if chrom not in CHROMS_AUTOSOMAL:
                    continue
                source_start0 = int(source_start_value)
                offset = (source_length - seq_length) // 2
                start_1 = source_start0 + offset + 1
                candidates.append((chrom, start_1, start_1 + seq_length - 1))
    assert candidates, (
        f"No autosomal test intervals of length >= {seq_length} in {source_h5}"
    )
    logging.getLogger(__name__).info(
        f"Loaded {len(candidates)} coordinate-disjoint identification candidates "
        f"from held-out multilen intervals"
    )
    return candidates


def pick_loci(
    chrom_lengths: Dict[str, int],
    chrom_to_seq: Dict[str, str],
    seq_length: int,
    num_loci: int,
    locus_mode: str,
    min_spacing: int,
    bcftools: str,
    vcf_paths: Dict[str, str],
    maf_min: float,
    rng: random.Random,
    candidate_oversample: int = 8,
    candidate_loci: List[Tuple[str, int, int]] | None = None,
) -> List[Tuple[str, int, int]]:
    """Pick K loci according to the requested mode.

    Returns a list of (chrom, start_1based, end_inclusive). Loci are spaced at
    least ``min_spacing`` bp apart on the same chromosome to limit LD redundancy.
    """
    logger = logging.getLogger(__name__)
    assert locus_mode in ("random", "snp_enriched"), f"Unknown locus_mode: {locus_mode}"

    chrom_weights = [chrom_lengths[c] for c in CHROMS_AUTOSOMAL]

    candidate_pool = list(candidate_loci) if candidate_loci is not None else None
    candidate_cursor = 0
    if candidate_pool is not None:
        rng.shuffle(candidate_pool)

    def generate_batch(n: int) -> List[Tuple[str, int, int]]:
        """Draw ``n`` N-free random autosomal windows of length ``seq_length``."""
        nonlocal candidate_cursor
        if candidate_pool is not None:
            end = min(candidate_cursor + n, len(candidate_pool))
            batch = candidate_pool[candidate_cursor:end]
            candidate_cursor = end
            return batch
        batch: List[Tuple[str, int, int]] = []
        while len(batch) < n:
            chrom = rng.choices(CHROMS_AUTOSOMAL, weights=chrom_weights, k=1)[0]
            max_start = chrom_lengths[chrom] - seq_length
            assert max_start > 0, f"Chromosome {chrom} shorter than seq_length"
            start_1 = rng.randint(1, max_start)
            end_1 = start_1 + seq_length - 1
            # Reject windows overlapping assembly gaps / centromeres (runs of N);
            # these have no usable reference sequence and would fail downstream.
            if "N" in chrom_to_seq[chrom][start_1 - 1 : end_1]:
                continue
            batch.append((chrom, start_1, end_1))
        return batch

    accepted: List[Tuple[str, int, int]] = []
    accepted_by_chrom: Dict[str, List[Tuple[int, int]]] = {c: [] for c in CHROMS_AUTOSOMAL}

    def try_accept(filtered: List[Tuple[str, int, int]]) -> None:
        """Greedily add spacing-compatible loci until ``num_loci`` is reached."""
        rng.shuffle(filtered)
        for chrom, start_1, end_1 in filtered:
            too_close = False
            for s, e in accepted_by_chrom[chrom]:
                if not (end_1 + min_spacing < s or start_1 > e + min_spacing):
                    too_close = True
                    break
            if too_close:
                continue
            accepted.append((chrom, start_1, end_1))
            accepted_by_chrom[chrom].append((start_1, end_1))
            if len(accepted) == num_loci:
                return

    # Adaptively generate + filter in batches until enough loci are accepted.
    # snp_enriched at short seq_length keeps only a small fraction of candidates
    # (a 25 bp window rarely contains a common SNP), so a single fixed-size pool
    # may fall short; we keep drawing until we hit num_loci or the cap. The cap
    # makes the failure fast and explicit rather than looping forever.
    batch_size = num_loci * candidate_oversample
    max_candidates = batch_size * 50
    total_generated = 0
    while len(accepted) < num_loci:
        batch = generate_batch(batch_size)
        assert batch, (
            f"Exhausted the {len(candidate_pool) if candidate_pool is not None else 0} "
            f"held-out multilen candidates after accepting {len(accepted)}/{num_loci} loci. "
            "Reduce --num_loci/--min_locus_spacing or expand the source test split."
        )
        total_generated += len(batch)
        if locus_mode == "random":
            filtered = batch
        else:
            assert locus_mode == "snp_enriched"
            filtered = _filter_snp_enriched(batch, bcftools, vcf_paths, maf_min)
        try_accept(filtered)
        logger.info(
            f"After {locus_mode} filter: {len(accepted)}/{num_loci} loci accepted "
            f"({total_generated} candidates generated)"
        )
        assert (
            candidate_pool is not None
            or total_generated < max_candidates
            or len(accepted) == num_loci
        ), (
            f"Could not pick {num_loci} non-overlapping {locus_mode} loci of length "
            f"{seq_length}; only got {len(accepted)} after {total_generated} candidates. "
            f"Lower --min_locus_spacing or relax filters."
        )

    accepted.sort()
    return accepted


def _filter_snp_enriched(
    candidates: List[Tuple[str, int, int]],
    bcftools: str,
    vcf_paths: Dict[str, str],
    maf_min: float,
) -> List[Tuple[str, int, int]]:
    """Keep candidates that contain >=1 PASS biallelic SNP with AF in [maf_min, 1-maf_min]."""
    # Group candidates by chromosome to issue one bcftools call per chromosome.
    by_chrom: Dict[str, List[Tuple[int, int]]] = {}
    for chrom, s, e in candidates:
        by_chrom.setdefault(chrom, []).append((s, e))

    keep: List[Tuple[str, int, int]] = []
    for chrom, intervals in by_chrom.items():
        assert chrom in vcf_paths, f"Missing VCF for {chrom}"
        regions = ",".join(f"{chrom}:{s}-{e}" for s, e in intervals)
        raw = _run(
            [
                bcftools,
                "query",
                "-r",
                regions,
                "-f",
                "%CHROM\t%POS\t%AF\n",
                "-i",
                f'TYPE="snp" && N_ALT=1 && AF>={maf_min} && AF<={1.0 - maf_min}',
                vcf_paths[chrom],
            ]
        )
        snp_positions: Dict[Tuple[int, int], bool] = {(s, e): False for s, e in intervals}
        # Sort intervals to enable a linear scan.
        sorted_iv = sorted(intervals)
        for line in raw.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            pos = int(parts[1])
            # Mark any interval containing this position.
            for s, e in sorted_iv:
                if s <= pos <= e:
                    snp_positions[(s, e)] = True
                elif pos < s:
                    break
        for s, e in intervals:
            if snp_positions[(s, e)]:
                keep.append((chrom, s, e))
    return keep


def query_genotypes(
    bcftools: str,
    vcf_paths: Dict[str, str],
    loci: List[Tuple[str, int, int]],
    samples: List[str],
) -> Dict[Tuple[str, int, int], List[Dict]]:
    """For each locus, return the list of biallelic SNP records inside it.

    Each record dict contains:
        - pos: int (1-based)
        - ref: str (single base)
        - alt: str (single base)
        - af: float
        - genotypes: list[str] of length len(samples) in the same order as ``samples``,
                     each like '0|0', '0|1', '1|0', or '1|1'.
    """
    logger = logging.getLogger(__name__)

    # samples file
    samples_file = "/tmp/identification_cohort_samples.txt"
    with open(samples_file, "w") as f:
        f.write("\n".join(samples) + "\n")

    # Group loci by chromosome.
    by_chrom: Dict[str, List[Tuple[int, int]]] = {}
    for chrom, s, e in loci:
        by_chrom.setdefault(chrom, []).append((s, e))

    out: Dict[Tuple[str, int, int], List[Dict]] = {locus: [] for locus in loci}

    for chrom, intervals in by_chrom.items():
        assert chrom in vcf_paths, f"Missing VCF for {chrom}"
        regions = ",".join(f"{chrom}:{s}-{e}" for s, e in intervals)
        # Single bcftools call extracting POS, REF, ALT, AF, and per-sample GT.
        fmt = "%CHROM\t%POS\t%REF\t%ALT\t%INFO/AF[\t%GT]\n"
        raw = _run(
            [
                bcftools,
                "query",
                "-r",
                regions,
                "-S",
                samples_file,
                "-f",
                fmt,
                "-i",
                'TYPE="snp" && N_ALT=1',
                vcf_paths[chrom],
            ]
        )
        sorted_iv = sorted(intervals)
        for line in raw.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            assert len(parts) == 5 + len(samples), (
                f"Unexpected bcftools query line column count at {chrom}: "
                f"got {len(parts)}, expected {5 + len(samples)}"
            )
            pos = int(parts[1])
            ref, alt = parts[2], parts[3]
            assert len(ref) == 1 and len(alt) == 1, (
                f"Expected biallelic SNP (single base) but got REF={ref} ALT={alt}"
            )
            af_str = parts[4]
            assert af_str not in ("", "."), f"Missing AF at {chrom}:{pos} (1000G should always have AF)"
            af = float(af_str)
            gts = parts[5:]
            record = {
                "pos": pos,
                "ref": ref.upper(),
                "alt": alt.upper(),
                "af": af,
                "genotypes": gts,
            }
            for s, e in sorted_iv:
                if s <= pos <= e:
                    out[(chrom, s, e)].append(record)
                elif pos < s:
                    break

    logger.info(
        f"Genotype extraction complete: "
        f"{sum(len(v) for v in out.values())} SNP records across {len(loci)} loci"
    )
    return out


def apply_haplotype(ref_window: str, locus_start_1: int, snps: List[Dict], sample_idx: int, hap: int) -> str:
    """Apply a sample's phased haplotype variants to a reference window.

    Parameters
    ----------
    ref_window : str
        Reference sequence for the locus.
    locus_start_1 : int
        1-based start coordinate of the window.
    snps : list[dict]
        SNP records produced by ``query_genotypes`` for this locus.
    sample_idx : int
        Index into each SNP record's ``genotypes`` list.
    hap : int
        Haplotype index, 0 or 1.

    Returns
    -------
    str
        The reconstructed sequence (same length as ``ref_window``).
    """
    assert hap in (0, 1), f"hap must be 0 or 1, got {hap}"
    chars = list(ref_window)
    L = len(chars)
    # A multiallelic site (e.g. G>A and G>T) is split into one biallelic record
    # per ALT, so several records can share one position. Apply at most one ALT
    # per offset on a given haplotype, and always check REF against the original
    # reference (ref_window) rather than the in-progress buffer (chars), which a
    # prior record at the same offset may already have mutated.
    applied_offset_to_alt: Dict[int, str] = {}
    for record in snps:
        gt = record["genotypes"][sample_idx]
        assert "|" in gt, f"Expected phased GT (1000G is phased), got {gt}"
        alleles = gt.split("|")
        assert len(alleles) == 2, f"Expected diploid GT, got {gt}"
        a = alleles[hap]
        assert a in ("0", "1"), f"Unexpected allele {a} in GT {gt} (biallelic SNP expected)"
        if a == "0":
            continue
        pos = record["pos"]
        offset = pos - locus_start_1
        assert 0 <= offset < L, f"SNP pos {pos} outside locus [{locus_start_1}, {locus_start_1+L-1}]"
        ref_in_window = ref_window[offset]
        assert (
            ref_in_window == record["ref"]
        ), f"REF mismatch at {pos}: window has {ref_in_window}, VCF says {record['ref']}"
        # A single haplotype cannot carry two different ALT alleles at one site.
        # Loci where this happens are dropped upstream by ``find_conflicted_loci``;
        # this assertion is a safety net that should never fire after filtering.
        assert offset not in applied_offset_to_alt, (
            f"Conflicting ALT alleles on haplotype {hap} at {pos}: "
            f"{applied_offset_to_alt[offset]} then {record['alt']} "
            f"(sample_idx={sample_idx})"
        )
        applied_offset_to_alt[offset] = record["alt"]
        chars[offset] = record["alt"]
    return "".join(chars)


def find_conflicted_loci(
    loci: List[Tuple[str, int, int]],
    snps_by_locus: Dict[Tuple[str, int, int], List[Dict]],
    n_samples: int,
) -> set:
    """Return the set of loci that cannot yield an unambiguous haplotype sequence.

    A multiallelic site (REF=G, ALT=A and ALT=T) is stored as several biallelic
    records at the same position. In a clean phased split, each haplotype carries
    at most one ALT there, but the 1000G callset also contains overlapping records
    where a single haplotype is marked ALT in more than one of them -- a
    contradiction (it cannot be both A and T). Such loci are dropped rather than
    silently picking one allele.
    """
    conflicted: set = set()
    for locus in loci:
        records = snps_by_locus[locus]
        # Only positions shared by >1 record can conflict.
        pos_counts: Dict[int, int] = {}
        for r in records:
            pos_counts[r["pos"]] = pos_counts.get(r["pos"], 0) + 1
        dup_positions = {p for p, c in pos_counts.items() if c > 1}
        if not dup_positions:
            continue
        dup_records = [r for r in records if r["pos"] in dup_positions]
        found = False
        for sample_idx in range(n_samples):
            if found:
                break
            for hap in (0, 1):
                alts_here: Dict[int, int] = {}
                for r in dup_records:
                    alleles = r["genotypes"][sample_idx].replace("|", "/").split("/")
                    if len(alleles) > hap and alleles[hap] == "1":
                        alts_here[r["pos"]] = alts_here.get(r["pos"], 0) + 1
                        if alts_here[r["pos"]] > 1:
                            conflicted.add(locus)
                            found = True
                            break
                if found:
                    break
    return conflicted


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 1000G per-individual sequences for identification eval")
    parser.add_argument("--reference", type=str, required=True, help="Path to hg38.fa (with .fai index)")
    parser.add_argument(
        "--vcf_dir",
        type=str,
        required=True,
        help="Directory containing per-chromosome 1000G VCFs (chr1.vcf.gz, ...)",
    )
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None, help="Path for manifest JSON (defaults to <output_csv>.manifest.json)")
    parser.add_argument("--seq_length", type=int, required=True)
    parser.add_argument("--cohort_size", type=int, required=True, help="N: number of individuals")
    parser.add_argument("--num_loci", type=int, required=True, help="K: number of loci")
    parser.add_argument(
        "--locus_mode",
        type=str,
        choices=["random", "snp_enriched"],
        required=True,
    )
    parser.add_argument("--min_locus_spacing", type=int, default=50000)
    parser.add_argument("--maf_min", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate_oversample", type=int, default=8)
    parser.add_argument(
        "--source_multilen_h5",
        type=str,
        default=None,
        help="Optional hg38_multilen.h5; restrict loci to its globally disjoint test intervals",
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Preparing 1000G identification dataset")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 80)

    rng = random.Random(args.seed)

    bcftools = _which_bcftools()

    # Discover VCF paths per autosomal chromosome.
    vcf_paths: Dict[str, str] = {}
    for chrom in CHROMS_AUTOSOMAL:
        p = os.path.join(args.vcf_dir, f"{chrom}.vcf.gz")
        assert os.path.exists(p), f"Missing VCF for {chrom}: {p}"
        vcf_paths[chrom] = p

    # Cohort sampling: use chr1's sample list as canonical (1000G VCFs share samples).
    all_samples = read_vcf_samples(bcftools, vcf_paths["chr1"])
    logger.info(f"VCF contains {len(all_samples)} samples")
    assert args.cohort_size <= len(all_samples), (
        f"Cohort size {args.cohort_size} exceeds available samples {len(all_samples)}"
    )
    samples = rng.sample(all_samples, args.cohort_size)
    samples.sort()
    logger.info(f"Sampled cohort of {len(samples)} individuals")

    chrom_lengths = read_chrom_lengths(args.reference)

    # Load the reference once: used both to reject N windows during locus
    # selection and to slice the final per-locus windows.
    chrom_to_seq = load_reference_sequences(args.reference)

    candidate_loci = None
    if args.source_multilen_h5:
        candidate_loci = load_multilen_test_candidates(
            args.source_multilen_h5, args.seq_length
        )

    loci = pick_loci(
        chrom_lengths=chrom_lengths,
        chrom_to_seq=chrom_to_seq,
        seq_length=args.seq_length,
        num_loci=args.num_loci,
        locus_mode=args.locus_mode,
        min_spacing=args.min_locus_spacing,
        bcftools=bcftools,
        vcf_paths=vcf_paths,
        maf_min=args.maf_min,
        rng=rng,
        candidate_oversample=args.candidate_oversample,
        candidate_loci=candidate_loci,
    )
    logger.info(f"Selected {len(loci)} loci for mode={args.locus_mode}")

    ref_windows = extract_reference_windows(chrom_to_seq, loci)
    snps_by_locus = query_genotypes(bcftools, vcf_paths, loci, samples)

    # Drop loci with overlapping/multiallelic records that would assign two ALT
    # alleles to a single haplotype at one position (ambiguous reference). The
    # eval tolerates fewer than num_loci loci, so we simply exclude these.
    conflicted = find_conflicted_loci(loci, snps_by_locus, len(samples))
    if conflicted:
        logger.info(
            f"Dropping {len(conflicted)} of {len(loci)} loci with conflicting "
            f"multiallelic records; keeping {len(loci) - len(conflicted)}."
        )
        loci = [locus for locus in loci if locus not in conflicted]

    # Build sequences.
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing sequences to {args.output_csv}")
    n_written = 0
    with open(args.output_csv, "w") as out:
        out.write("individual_id,locus_id,haplotype,sequence\n")
        for locus_id, locus in enumerate(loci):
            chrom, start_1, end_1 = locus
            ref_window = ref_windows[locus]
            snps = snps_by_locus[locus]
            for sample_idx, sample in enumerate(samples):
                for hap in (0, 1):
                    seq = apply_haplotype(ref_window, start_1, snps, sample_idx, hap)
                    assert len(seq) == args.seq_length, (
                        f"Sequence length mismatch: got {len(seq)}, expected {args.seq_length}"
                    )
                    assert "N" not in seq, f"N in reconstructed sequence at {chrom}:{start_1}-{end_1}, sample {sample}, hap {hap}"
                    out.write(f"{sample},{locus_id},{hap},{seq}\n")
                    n_written += 1

    logger.info(f"Wrote {n_written} sequences")

    # Manifest.
    manifest_path = args.manifest or (args.output_csv + ".manifest.json")
    manifest = {
        "seq_length": args.seq_length,
        "cohort_size": args.cohort_size,
        "num_loci": args.num_loci,
        "locus_mode": args.locus_mode,
        "min_locus_spacing": args.min_locus_spacing,
        "maf_min": args.maf_min,
        "seed": args.seed,
        "reference": args.reference,
        "vcf_dir": args.vcf_dir,
        "source_multilen_h5": args.source_multilen_h5,
        "samples": samples,
        "loci": [
            {
                "locus_id": i,
                "chrom": c,
                "start_1based": s,
                "end_inclusive": e,
                "num_snps": len(snps_by_locus[(c, s, e)]),
                "snp_positions": [r["pos"] for r in snps_by_locus[(c, s, e)]],
                "snp_afs": [r["af"] for r in snps_by_locus[(c, s, e)]],
            }
            for i, (c, s, e) in enumerate(loci)
        ],
    }
    with open(manifest_path, "w") as mf_out:
        json.dump(manifest, mf_out, indent=2)
    logger.info(f"Wrote manifest to {manifest_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
