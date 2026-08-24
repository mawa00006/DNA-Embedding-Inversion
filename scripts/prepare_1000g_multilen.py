"""Materialise a real-genome 1000G test corpus on held-out multilen loci.

This script reads the test intervals from the sequence-disjoint hg38 corpus, assigns each
interval to a random phased 1000 Genomes individual/haplotype, and applies that
haplotype's biallelic SNPs. It therefore evaluates the trained decoder on real
individual variation at loci that remain disjoint across sequence lengths.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_1000g import (
    CHROMS_AUTOSOMAL,
    _run,
    _which_bcftools,
    read_vcf_samples,
)


LOGGER = logging.getLogger(__name__)
SCHEMA = "dna_inversion_1000g_multilen_v2"
SOURCE_SCHEMA = "dna_inversion_hg38_multilen_v2"
TEST_SPLIT = 2


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _ordered_sequence_sha256(sequences: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(sequence.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def query_snps_in_regions(
    bcftools: str,
    vcf_path: str,
    chrom: str,
    regions: Sequence[Tuple[int, int]],
    samples: Sequence[str],
    tmp_samples_file: str,
) -> Dict[int, Dict]:
    """Query phased biallelic SNPs for one chromosome and a region batch."""
    with open(tmp_samples_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(samples) + "\n")
    regions_arg = ",".join(f"{chrom}:{start}-{end}" for start, end in regions)
    raw = _run(
        [
            bcftools,
            "query",
            "-r",
            regions_arg,
            "-S",
            tmp_samples_file,
            "-f",
            "%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n",
            "-i",
            'TYPE="snp" && N_ALT=1',
            vcf_path,
        ]
    )
    result: Dict[int, Dict] = {}
    for line in raw.splitlines():
        fields = line.split("\t")
        if len(fields) != 4 + len(samples):
            continue
        pos = int(fields[1])
        ref, alt = fields[2].upper(), fields[3].upper()
        if len(ref) == len(alt) == 1:
            result[pos] = {"ref": ref, "alt": alt, "gts": fields[4:]}
    return result


def load_test_windows(
    source_path: str, length: int, limit: int
) -> List[Tuple[str, int, str]]:
    """Load up to ``limit`` autosomal test windows as ``(chrom,start,ref)``."""
    with h5py.File(source_path, "r", swmr=True) as handle:
        assert handle.attrs.get("schema", "") == SOURCE_SCHEMA, (
            f"1000G v2 must be derived from a sequence-disjoint hg38 v2 corpus: "
            f"{source_path}"
        )
        assert bool(handle.attrs["sequence_content_split_disjoint"])
        assert bool(handle.attrs["generation_complete"])
        group = handle[f"lengths/{length}"]
        splits = group["split"]
        chroms = group["chrom"]
        starts = group["start"]
        sequences = group["sequences"]
        windows: List[Tuple[str, int, str]] = []
        for idx in range(len(sequences)):
            if int(splits[idx]) != TEST_SPLIT:
                continue
            chrom = _decode(chroms[idx])
            if chrom not in CHROMS_AUTOSOMAL:
                continue
            windows.append((chrom, int(starts[idx]), _decode(sequences[idx])))
            if len(windows) == limit:
                break
    assert len(windows) == limit, (
        f"Only {len(windows)} autosomal test windows of length {length}; requested {limit}"
    )
    return windows


def apply_haplotypes(
    *,
    windows: Sequence[Tuple[str, int, str]],
    samples: Sequence[str],
    vcf_dir: str,
    bcftools: str,
    rng: random.Random,
    batch_size: int,
    tmp_samples_file: str,
) -> Tuple[List[str], List[str], List[int], int]:
    """Apply one random sample/haplotype to every fixed genomic window."""
    output: List[str] = []
    output_samples: List[str] = []
    output_haps: List[int] = []
    mutated = 0

    assignments = [(rng.choice(samples), rng.randint(0, 1)) for _ in windows]
    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start : batch_start + batch_size]
        batch_assignments = assignments[batch_start : batch_start + batch_size]
        by_chrom: Dict[str, List[int]] = defaultdict(list)
        for local_idx, (chrom, _start, _reference) in enumerate(batch_windows):
            by_chrom[chrom].append(local_idx)

        batch_out = [reference for _chrom, _start, reference in batch_windows]
        for chrom, local_indices in by_chrom.items():
            chrom_samples = sorted({batch_assignments[i][0] for i in local_indices})
            sample_to_idx = {sample: i for i, sample in enumerate(chrom_samples)}
            regions = [
                (
                    batch_windows[i][1] + 1,
                    batch_windows[i][1] + len(batch_windows[i][2]),
                )
                for i in local_indices
            ]
            snp_map = query_snps_in_regions(
                bcftools,
                os.path.join(vcf_dir, f"{chrom}.vcf.gz"),
                chrom,
                regions,
                chrom_samples,
                tmp_samples_file,
            )
            sorted_positions = sorted(snp_map)
            for local_idx in local_indices:
                _chrom, start0, reference = batch_windows[local_idx]
                sample, hap = batch_assignments[local_idx]
                start1 = start0 + 1
                end1 = start0 + len(reference)
                chars = list(reference)
                lo = bisect.bisect_left(sorted_positions, start1)
                hi = bisect.bisect_right(sorted_positions, end1)
                sample_idx = sample_to_idx[sample]
                for pos in sorted_positions[lo:hi]:
                    record = snp_map[pos]
                    gt = record["gts"][sample_idx]
                    if "|" not in gt:
                        continue
                    alleles = gt.split("|")
                    if len(alleles) != 2 or alleles[hap] != "1":
                        continue
                    offset = pos - start1
                    if chars[offset] == record["ref"]:
                        chars[offset] = record["alt"]
                sequence = "".join(chars)
                batch_out[local_idx] = sequence
                mutated += int(sequence != reference)

        output.extend(batch_out)
        output_samples.extend(sample for sample, _hap in batch_assignments)
        output_haps.extend(hap for _sample, hap in batch_assignments)
        LOGGER.info("Materialised %d/%d windows", min(batch_start + batch_size, len(windows)), len(windows))

    return output, output_samples, output_haps, mutated


def write_group(
    handle: h5py.File,
    length: int,
    windows: Sequence[Tuple[str, int, str]],
    sequences: Sequence[str],
    samples: Sequence[str],
    haplotypes: Sequence[int],
) -> None:
    group = handle.require_group("lengths").create_group(str(length))
    n = len(sequences)
    assert n == len(windows) == len(samples) == len(haplotypes)
    group.create_dataset(
        "sequences",
        data=np.asarray([s.encode("ascii") for s in sequences], dtype=f"S{length}"),
        compression="lzf",
    )
    references = [reference for _chrom, _start, reference in windows]
    differs = np.asarray(
        [sequence != reference for sequence, reference in zip(sequences, references, strict=True)],
        dtype=np.bool_,
    )
    group.create_dataset(
        "reference_sequences",
        data=np.asarray([s.encode("ascii") for s in references], dtype=f"S{length}"),
        compression="lzf",
    )
    group.create_dataset(
        "differs_from_reference",
        data=differs,
        compression="lzf",
        shuffle=True,
    )
    group.create_dataset(
        "chrom",
        data=np.asarray([chrom.encode("ascii") for chrom, _start, _seq in windows], dtype="S16"),
        compression="lzf",
    )
    group.create_dataset(
        "start",
        data=np.asarray([start for _chrom, start, _seq in windows], dtype=np.uint32),
        compression="lzf",
        shuffle=True,
    )
    dt = h5py.string_dtype("utf-8")
    group.create_dataset("individual_id", data=np.asarray(samples, dtype=object), dtype=dt)
    group.create_dataset("haplotype", data=np.asarray(haplotypes, dtype=np.uint8))
    group.attrs["seq_length"] = length
    group.attrs["num_sequences"] = n
    group.attrs["num_sequences_differing_from_reference"] = int(differs.sum())
    group.attrs["ordered_sequence_sha256"] = _ordered_sequence_sha256(sequences)
    group.attrs["source_test_sequence_sha256"] = _ordered_sequence_sha256(references)


def build_1000g_corpus(
    *,
    source_path: str,
    output_path: str,
    vcf_dir: str,
    lengths: Sequence[int],
    num_sequences: int,
    seed: int,
    batch_size: int,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists(), f"Refusing to overwrite existing corpus: {output}"
    partial = Path(str(output) + ".partial")
    assert not partial.exists(), f"Remove or inspect incomplete corpus first: {partial}"

    bcftools = _which_bcftools()
    chr1_vcf = os.path.join(vcf_dir, "chr1.vcf.gz")
    samples = read_vcf_samples(bcftools, chr1_vcf)
    rng = random.Random(seed)
    tmp_samples_file = str(output.parent / f".samples_{os.getpid()}.txt")

    try:
        with h5py.File(partial, "w") as handle:
            handle.attrs["schema"] = SCHEMA
            handle.attrs["source_schema"] = SOURCE_SCHEMA
            handle.attrs["source_corpus"] = os.path.abspath(source_path)
            handle.attrs["seed"] = seed
            handle.attrs["generation_complete"] = False
            total_mutated = 0
            total = 0
            for length in lengths:
                LOGGER.info("Loading held-out length-%d loci", length)
                windows = load_test_windows(source_path, length, num_sequences)
                sequences, chosen_samples, haps, mutated = apply_haplotypes(
                    windows=windows,
                    samples=samples,
                    vcf_dir=vcf_dir,
                    bcftools=bcftools,
                    rng=rng,
                    batch_size=batch_size,
                    tmp_samples_file=tmp_samples_file,
                )
                write_group(handle, length, windows, sequences, chosen_samples, haps)
                total_mutated += mutated
                total += len(sequences)
                LOGGER.info(
                    "Length %d: %d/%d windows differ from hg38",
                    length,
                    mutated,
                    len(sequences),
                )
            handle.attrs["num_sequences"] = total
            handle.attrs["num_sequences_differing_from_reference"] = total_mutated
            handle.attrs["generation_complete"] = True
            handle.flush()
        os.replace(partial, output)
    finally:
        if os.path.exists(tmp_samples_file):
            os.remove(tmp_samples_file)
    LOGGER.info("Wrote complete 1000G multi-length corpus: %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 1000G OOD data on multilen test loci")
    parser.add_argument(
        "--source",
        default=(
            "data/hg38_multilen_seqdisjoint/hg38_multilen_seqdisjoint.h5"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "data/1000g_multilen_seqdisjoint/1000g_multilen_seqdisjoint.h5"
        ),
    )
    parser.add_argument("--vcf-dir", default="data/1000g_vcfs")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100],
    )
    parser.add_argument("--num-sequences", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build_1000g_corpus(
        source_path=args.source,
        output_path=args.output,
        vcf_dir=args.vcf_dir,
        lengths=args.lengths,
        num_sequences=args.num_sequences,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
