"""Prepare 1000G sequences (deduplicated) for out-of-distribution evaluation.

Samples random autosomal windows of length ``seq_length``, assigns each window
to a random 1000G individual/haplotype, applies the individual's biallelic SNP
variants via ``bcftools query``, and writes unique sequences (no 'N') to a CSV
(one sequence per line, no header) matching the format expected by the
``generate/generate_*_embeddings.py`` scripts.

Example:
    python scripts/prepare_1000g.py --seq_length 100 --output data/1000g_seq100.csv --seed 42
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import miniFasta as mf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

CHROMS_AUTOSOMAL = [f"chr{i}" for i in range(1, 23)]


def _which_bcftools() -> str:
    """Return the bcftools binary path, asserting it exists on PATH."""
    path = shutil.which("bcftools")
    assert path is not None, "bcftools not found on PATH; activate the project conda env"
    return path


def _run(cmd: List[str]) -> str:
    """Run a subprocess and return stdout as text. Fail fast on non-zero exit."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"Command failed ({result.returncode}): {' '.join(cmd)}\nSTDERR: {result.stderr[:1000]}"
    )
    return result.stdout


def read_vcf_samples(bcftools: str, vcf_path: str) -> List[str]:
    """Return ordered list of sample names from a VCF header."""
    out = _run([bcftools, "query", "-l", vcf_path])
    samples = [s for s in out.strip().split("\n") if s]
    assert len(samples) > 0, f"No samples found in VCF {vcf_path}"
    return samples


def load_reference_seqs(reference: str) -> Dict[str, str]:
    """Load autosomal chromosome sequences from a FASTA. Uppercase, in-memory."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading reference FASTA: {reference} (this takes ~30s for hg38)")
    fa_objects = mf.read(reference, upper=True)
    chrom_seqs: Dict[str, str] = {}
    for obj in fa_objects:
        name = obj.head.lstrip(">").split()[0]
        if name in CHROMS_AUTOSOMAL:
            chrom_seqs[name] = str(obj.body)
    missing = [c for c in CHROMS_AUTOSOMAL if c not in chrom_seqs]
    assert not missing, f"Reference missing autosomal chroms: {missing}"
    return chrom_seqs


def _query_snps_in_regions(
    bcftools: str,
    vcf_path: str,
    regions: List[Tuple[int, int]],
    samples: List[str],
    tmp_samples_file: str,
) -> Dict[int, Dict]:
    """Query biallelic SNPs across multiple regions in a single bcftools call.

    Returns a dict keyed by 1-based POS, each value containing ``ref``, ``alt``,
    and ``gts`` (list of per-sample phased genotype strings in the same order
    as ``samples``).
    """
    with open(tmp_samples_file, "w") as f:
        f.write("\n".join(samples) + "\n")

    regions_str = ",".join(f"{s}-{e}" for s, e in regions)
    raw = _run(
        [
            bcftools,
            "query",
            "-r",
            regions_str,
            "-S",
            tmp_samples_file,
            "-f",
            "%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n",
            "-i",
            'TYPE="snp" && N_ALT=1',
            vcf_path,
        ]
    )

    out: Dict[int, Dict] = {}
    for line in raw.split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4 + len(samples):
            continue
        pos = int(parts[1])
        ref, alt = parts[2].upper(), parts[3].upper()
        if len(ref) != 1 or len(alt) != 1:
            continue
        out[pos] = {"ref": ref, "alt": alt, "gts": parts[4:]}
    return out


def generate_sequences(
    bcftools: str,
    vcf_dir: str,
    chrom_seqs: Dict[str, str],
    samples: List[str],
    seq_length: int,
    num_sequences: int,
    rng: random.Random,
    batch_size: int,
    max_oversample: float,
) -> List[str]:
    """Generate ``num_sequences`` unique 1000G sequences via batched bcftools queries."""
    logger = logging.getLogger(__name__)
    chrom_lengths = {c: len(chrom_seqs[c]) for c in CHROMS_AUTOSOMAL}
    weights = [chrom_lengths[c] for c in CHROMS_AUTOSOMAL]

    vcf_paths = {c: os.path.join(vcf_dir, f"{c}.vcf.gz") for c in CHROMS_AUTOSOMAL}
    for c, p in vcf_paths.items():
        assert os.path.exists(p), f"Missing VCF for {c}: {p}"

    unique_seqs: set[str] = set()
    attempts_cap = int(num_sequences * max_oversample)
    total_attempts = 0
    tmp_samples_file = f"/tmp/_prep_1000g_samples_{os.getpid()}.txt"

    try:
        while len(unique_seqs) < num_sequences and total_attempts < attempts_cap:
            # Build a batch of candidate (chrom, start, end, sample, hap, ref_window) tuples.
            batch: List[Tuple[str, int, int, str, int, str]] = []
            for _ in range(batch_size):
                chrom = rng.choices(CHROMS_AUTOSOMAL, weights=weights, k=1)[0]
                start_1 = rng.randint(1, chrom_lengths[chrom] - seq_length + 1)
                end_1 = start_1 + seq_length - 1
                window = chrom_seqs[chrom][start_1 - 1 : end_1]
                if "N" in window:
                    continue
                sample = rng.choice(samples)
                hap = rng.randint(0, 1)
                batch.append((chrom, start_1, end_1, sample, hap, window))
            total_attempts += batch_size

            # Group by chromosome for one bcftools query per chrom.
            by_chrom: Dict[str, List[int]] = defaultdict(list)
            for i, item in enumerate(batch):
                by_chrom[item[0]].append(i)

            for chrom, indices in by_chrom.items():
                items = [batch[i] for i in indices]
                # Unique samples in this batch (alphabetical for stable indexing).
                samples_in_batch = sorted({it[3] for it in items})
                sample_idx_map = {s: i for i, s in enumerate(samples_in_batch)}

                regions = [(it[1], it[2]) for it in items]
                snp_map = _query_snps_in_regions(
                    bcftools, vcf_paths[chrom], regions, samples_in_batch, tmp_samples_file
                )
                if not snp_map:
                    # No variants; the reference windows themselves go into the set.
                    for _, _, _, _, _, window in items:
                        unique_seqs.add(window)
                        if len(unique_seqs) >= num_sequences:
                            break
                    continue

                # Build sorted index for fast pos->region lookup.
                sorted_pos = sorted(snp_map.keys())
                # For each candidate, find SNPs within [s, e] via bisect.
                import bisect

                for _, start_1, end_1, sample, hap, window in items:
                    chars = list(window)
                    lo = bisect.bisect_left(sorted_pos, start_1)
                    hi = bisect.bisect_right(sorted_pos, end_1)
                    si = sample_idx_map[sample]
                    for pos in sorted_pos[lo:hi]:
                        rec = snp_map[pos]
                        gt = rec["gts"][si]
                        if "|" not in gt:
                            # Skip unphased / missing.
                            continue
                        alleles = gt.split("|")
                        if len(alleles) != 2:
                            continue
                        a = alleles[hap]
                        if a != "1":
                            continue
                        offset = pos - start_1
                        if not (0 <= offset < len(chars)):
                            continue
                        # Only apply if reference base matches (sanity).
                        if chars[offset] == rec["ref"]:
                            chars[offset] = rec["alt"]
                    seq = "".join(chars)
                    if "N" not in seq:
                        unique_seqs.add(seq)
                        if len(unique_seqs) >= num_sequences:
                            break

                if len(unique_seqs) >= num_sequences:
                    break

            logger.info(
                f"Progress: {len(unique_seqs)}/{num_sequences} unique sequences "
                f"(attempts={total_attempts}, cap={attempts_cap})"
            )
    finally:
        if os.path.exists(tmp_samples_file):
            os.remove(tmp_samples_file)

    if len(unique_seqs) < num_sequences:
        logging.getLogger(__name__).warning(
            f"Reached attempt cap before target: produced {len(unique_seqs)}/{num_sequences}. "
            f"Increase --max_oversample to grow further."
        )
    return list(unique_seqs)[:num_sequences]


def save_sequences_to_csv(sequences: List[str], output_path: str) -> None:
    """Save sequences to a CSV (one sequence per line, no header)."""
    logger = logging.getLogger(__name__)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for seq in sequences:
            f.write(f"{seq}\n")
    logger.info(f"Wrote {len(sequences)} sequences to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare 1000G sequences via bcftools for OOD evaluation"
    )
    parser.add_argument("--seq_length", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=20000,
        help="Target number of unique sequences (default: 20000; generate scripts use the first 15000)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="data/hg38.fa",
        help="Path to hg38 FASTA",
    )
    parser.add_argument(
        "--vcf_dir",
        type=str,
        default="data/1000g_vcfs",
        help="Directory containing per-chromosome 1000G VCFs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5000,
        help="Number of candidate windows per bcftools batch",
    )
    parser.add_argument(
        "--max_oversample",
        type=float,
        default=4.0,
        help="Hard cap on attempts as a multiple of num_sequences",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Preparing 1000G sequences (bcftools)")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 80)

    rng = random.Random(args.seed)
    bcftools = _which_bcftools()

    chr1_vcf = os.path.join(args.vcf_dir, "chr1.vcf.gz")
    assert os.path.exists(chr1_vcf), f"Missing chr1 VCF: {chr1_vcf}"
    samples = read_vcf_samples(bcftools, chr1_vcf)
    logger.info(f"VCF contains {len(samples)} samples")

    chrom_seqs = load_reference_seqs(args.reference)

    sequences = generate_sequences(
        bcftools=bcftools,
        vcf_dir=args.vcf_dir,
        chrom_seqs=chrom_seqs,
        samples=samples,
        seq_length=args.seq_length,
        num_sequences=args.num_sequences,
        rng=rng,
        batch_size=args.batch_size,
        max_oversample=args.max_oversample,
    )

    assert len(sequences) > 0, "No sequences generated"
    save_sequences_to_csv(sequences, args.output)
    logger.info("Done.")


if __name__ == "__main__":
    main()
