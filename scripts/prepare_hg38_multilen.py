"""Build a coordinate- and sequence-disjoint hg38 corpus.

A shuffled multiset of requested lengths is tiled over shuffled,
non-overlapping reference blocks, and every candidate interval advances the
genomic cursor. A candidate is accepted only if its exact sequence has not
already been accepted at that length. Consequently no two rows overlap in
coordinates and no exact sequence can occur in more than one
train/validation/test split.

The output is a source corpus, not an embedding file.  It contains one group per
length under ``/lengths/<L>`` with ``sequences``, ``chrom``, ``start`` (zero-based,
half-open together with ``L``), and ``split`` datasets.  The embedding generators
can read a group directly through ``load_sequences_from_file`` by passing this
file and the requested ``seq_length``.

Examples
--------
Full benchmark defaults (33 million windows; ~1.94 Gbp):

    python scripts/prepare_hg38_multilen.py \
        --fasta data/hg38.fa \
        --output data/hg38_multilen_seqdisjoint/hg38_multilen_seqdisjoint.h5

Small smoke corpus:

    python scripts/prepare_hg38_multilen.py \
        --fasta data/hg38.fa --output /tmp/hg38_multilen_smoke.h5 \
        --counts 10:1000,25:2000,100:5000 --val-size 100 --test-size 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, Iterator, List, Mapping, Tuple

import h5py
import numpy as np


LOGGER = logging.getLogger(__name__)
SCHEMA = "dna_inversion_hg38_multilen_v2"
SPLIT_NAMES = ("train", "val", "test")

# Longer sequences deliberately receive more samples.  Validation and test are
# fixed-count per length; the increase therefore goes entirely into training.
DEFAULT_COUNTS: Dict[int, int] = {
    10: 1_000_000,
    15: 1_000_000,
    20: 1_000_000,
    25: 2_000_000,
    30: 2_000_000,
    35: 2_000_000,
    40: 2_000_000,
    45: 2_000_000,
    50: 3_000_000,
    60: 3_000_000,
    70: 3_000_000,
    80: 3_000_000,
    90: 4_000_000,
    100: 4_000_000,
}


@dataclass(frozen=True)
class FaiRecord:
    """One FASTA-index record."""

    name: str
    length: int
    offset: int
    line_bases: int
    line_width: int


class IndexedFasta:
    """Minimal random-access FASTA reader backed by a samtools ``.fai`` index."""

    def __init__(self, fasta_path: str):
        self.path = os.fspath(fasta_path)
        fai_path = self.path + ".fai"
        assert os.path.exists(self.path), f"Reference FASTA not found: {self.path}"
        assert os.path.exists(fai_path), (
            f"FASTA index not found: {fai_path}. Run `samtools faidx {self.path}` first."
        )
        self.records: Dict[str, FaiRecord] = {}
        with open(fai_path, "r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                assert len(fields) >= 5, f"Malformed .fai line: {line!r}"
                rec = FaiRecord(
                    name=fields[0],
                    length=int(fields[1]),
                    offset=int(fields[2]),
                    line_bases=int(fields[3]),
                    line_width=int(fields[4]),
                )
                self.records[rec.name] = rec
        assert self.records, f"No contigs found in {fai_path}"
        self._handle: BinaryIO | None = None

    def __enter__(self) -> "IndexedFasta":
        self._handle = open(self.path, "rb")
        return self

    def __exit__(self, *_exc) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def fetch(self, chrom: str, start: int, end: int) -> bytes:
        """Return uppercase bases in the zero-based half-open interval."""
        assert self._handle is not None, "IndexedFasta must be used as a context manager"
        rec = self.records[chrom]
        assert 0 <= start <= end <= rec.length, (chrom, start, end, rec.length)
        if start == end:
            return b""
        byte_start = rec.offset + (start // rec.line_bases) * rec.line_width + (
            start % rec.line_bases
        )
        n_bases = end - start
        first_line_offset = start % rec.line_bases
        n_lines = math.ceil((first_line_offset + n_bases) / rec.line_bases)
        newline_bytes = rec.line_width - rec.line_bases
        self._handle.seek(byte_start)
        raw = self._handle.read(n_bases + n_lines * newline_bytes + 2)
        seq = raw.replace(b"\n", b"").replace(b"\r", b"")[:n_bases].upper()
        assert len(seq) == n_bases, (
            f"Short FASTA read for {chrom}:{start}-{end}: got {len(seq)} bases"
        )
        return seq


def parse_counts(value: str | None) -> Dict[int, int]:
    """Parse ``L:N,L:N`` or a JSON mapping; return sorted positive counts."""
    if value is None:
        return dict(DEFAULT_COUNTS)
    candidate = Path(value)
    if candidate.exists():
        with open(candidate, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        counts = {int(k): int(v) for k, v in raw.items()}
    else:
        counts = {}
        for item in value.split(","):
            length, count = item.split(":", maxsplit=1)
            counts[int(length)] = int(count)
    assert counts and all(length > 0 and count > 0 for length, count in counts.items())
    return dict(sorted(counts.items()))


def deterministic_split_codes(
    n: int, val_size: int, test_size: int, seed: int
) -> np.ndarray:
    """Return split codes matching ``generation_io.deterministic_split_labels``.

    Codes are 0=train, 1=validation, 2=test.  Matching the embedding writer's
    permutation means coordinates in this source file can be audited against the
    eventual embedded split without copying coordinate columns into every FM H5.
    """
    assert 0 <= val_size <= n
    assert 0 <= test_size <= n - val_size
    train_n = n - val_size - test_size
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    split = np.empty(n, dtype=np.uint8)
    split[perm[:train_n]] = 0
    split[perm[train_n : train_n + val_size]] = 1
    split[perm[train_n + val_size :]] = 2
    return split


def select_chromosomes(
    available: Iterable[str], requested: List[str] | None
) -> List[str]:
    """Select primary hg38 chromosomes, retaining FASTA order."""
    available_list = list(available)
    if requested:
        missing = [chrom for chrom in requested if chrom not in available_list]
        assert not missing, f"Requested chromosomes missing from FASTA: {missing}"
        return requested
    canonical = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}
    selected = [chrom for chrom in available_list if chrom in canonical]
    if selected:
        return selected
    # Support references named 1..22,X,Y,MT while still excluding alt contigs.
    canonical_no_prefix = {str(i) for i in range(1, 23)} | {"X", "Y", "MT", "M"}
    selected = [chrom for chrom in available_list if chrom in canonical_no_prefix]
    assert selected, "Could not identify primary chromosomes; pass --chromosomes explicitly"
    return selected


def make_blocks(
    records: Mapping[str, FaiRecord], chromosomes: Iterable[str], block_size: int
) -> List[Tuple[str, int, int]]:
    """Partition chromosomes into mutually disjoint fixed-size blocks."""
    assert block_size > 0
    blocks: List[Tuple[str, int, int]] = []
    for chrom in chromosomes:
        chrom_len = records[chrom].length
        blocks.extend(
            (chrom, start, min(start + block_size, chrom_len))
            for start in range(0, chrom_len, block_size)
        )
    return blocks


def iter_acgt_runs(sequence: bytes) -> Iterator[Tuple[int, int]]:
    """Yield zero-based A/C/G/T-only runs inside ``sequence``."""
    for match in re.finditer(b"[ACGT]+", sequence):
        yield match.start(), match.end()


class MultiLengthWriter:
    """Buffered writer for the grouped multi-length source HDF5."""

    def __init__(
        self,
        path: str,
        counts: Mapping[int, int],
        split_codes: Mapping[int, np.ndarray],
        buffer_size: int,
    ) -> None:
        self.path = path
        self.counts = dict(counts)
        self.split_codes = split_codes
        self.buffer_size = buffer_size
        self.handle = h5py.File(path, "w")
        self.groups: Dict[int, h5py.Group] = {}
        self.buffers: Dict[int, List[Tuple[bytes, bytes, int]]] = {
            length: [] for length in counts
        }
        self.written = {length: 0 for length in counts}
        self.sequence_hashers = {length: hashlib.sha256() for length in counts}
        root = self.handle.create_group("lengths")
        for length, target in counts.items():
            group = root.create_group(str(length))
            chunk = min(buffer_size, target)
            group.create_dataset(
                "sequences",
                shape=(0,),
                maxshape=(None,),
                dtype=f"S{length}",
                chunks=(chunk,),
                compression="lzf",
            )
            group.create_dataset(
                "chrom",
                shape=(0,),
                maxshape=(None,),
                dtype="S16",
                chunks=(chunk,),
                compression="lzf",
            )
            group.create_dataset(
                "start",
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint32,
                chunks=(chunk,),
                compression="lzf",
                shuffle=True,
            )
            group.create_dataset(
                "split",
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint8,
                chunks=(chunk,),
                compression="lzf",
                shuffle=True,
            )
            group.attrs["seq_length"] = length
            group.attrs["target_count"] = target
            self.groups[length] = group

    def append(self, length: int, sequence: bytes, chrom: str, start: int) -> None:
        assert len(sequence) == length
        self.sequence_hashers[length].update(sequence)
        self.sequence_hashers[length].update(b"\n")
        self.buffers[length].append((sequence, chrom.encode("ascii"), start))
        if len(self.buffers[length]) >= self.buffer_size:
            self.flush_length(length)

    def flush_length(self, length: int) -> None:
        rows = self.buffers[length]
        if not rows:
            return
        group = self.groups[length]
        n0 = self.written[length]
        n1 = n0 + len(rows)
        assert n1 <= self.counts[length]
        for name in ("sequences", "chrom", "start", "split"):
            group[name].resize((n1,))
        group["sequences"][n0:n1] = [row[0] for row in rows]
        group["chrom"][n0:n1] = [row[1] for row in rows]
        group["start"][n0:n1] = np.asarray([row[2] for row in rows], dtype=np.uint32)
        group["split"][n0:n1] = self.split_codes[length][n0:n1]
        self.written[length] = n1
        rows.clear()

    def close(self) -> None:
        for length in self.counts:
            self.flush_length(length)
            group = self.groups[length]
            group.attrs["ordered_sequence_sha256"] = self.sequence_hashers[
                length
            ].hexdigest()
            group.attrs["num_unique_sequences"] = self.written[length]
        self.handle.flush()
        self.handle.close()


def build_multilen_corpus(
    *,
    fasta_path: str,
    output_path: str,
    counts: Mapping[int, int],
    val_size: int,
    test_size: int,
    seed: int,
    block_size: int,
    buffer_size: int,
    chromosomes: List[str] | None = None,
) -> None:
    """Generate a complete cross-length-disjoint HDF5 source corpus."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists(), (
        f"Refusing to overwrite existing corpus: {output}. Move it aside or choose a new path."
    )
    partial = Path(str(output) + ".partial")
    assert not partial.exists(), (
        f"Partial corpus exists: {partial}. Inspect/remove it before regenerating."
    )

    counts = dict(sorted((int(k), int(v)) for k, v in counts.items()))
    for length, count in counts.items():
        assert count <= 4**length, (
            f"Cannot draw {count:,} unique DNA strings of length {length}; "
            f"the theoretical maximum is {4**length:,}"
        )
    assert all(val_size + test_size < n for n in counts.values()), (
        "Every length needs at least one training row after fixed validation/test allocation"
    )
    total_rows = sum(counts.values())
    total_bases = sum(length * count for length, count in counts.items())
    LOGGER.info(
        "Requested %s windows spanning %s bases (%s lengths)",
        f"{total_rows:,}",
        f"{total_bases:,}",
        len(counts),
    )

    # A compact int16 schedule is a shuffled multiset with exact per-length
    # quotas.  It costs ~66 MB for the full 33M-row benchmark and avoids tens of
    # millions of Python-level weighted-random draws.
    schedule = np.repeat(
        np.asarray(list(counts.keys()), dtype=np.int16),
        np.asarray(list(counts.values()), dtype=np.int64),
    )
    schedule_rng = np.random.default_rng(seed)
    schedule_rng.shuffle(schedule)

    split_codes = {
        length: deterministic_split_codes(count, val_size, test_size, seed)
        for length, count in counts.items()
    }

    with IndexedFasta(fasta_path) as fasta:
        chosen_chroms = select_chromosomes(fasta.records.keys(), chromosomes)
        blocks = make_blocks(fasta.records, chosen_chroms, block_size)
        block_rng = np.random.default_rng(seed + 1)
        block_rng.shuffle(blocks)
        available_bases = sum(end - start for _, start, end in blocks)
        assert total_bases <= available_bases, (
            f"Requested {total_bases:,} bases but selected chromosomes contain only "
            f"{available_bases:,} before N filtering"
        )

        writer = MultiLengthWriter(
            str(partial), counts=counts, split_codes=split_codes, buffer_size=buffer_size
        )
        writer.handle.attrs["schema"] = SCHEMA
        writer.handle.attrs["reference"] = os.path.abspath(fasta_path)
        writer.handle.attrs["coordinate_system"] = "0-based half-open"
        writer.handle.attrs["seed"] = seed
        writer.handle.attrs["block_size"] = block_size
        writer.handle.attrs["val_size_per_length"] = val_size
        writer.handle.attrs["test_size_per_length"] = test_size
        writer.handle.attrs["split_names"] = json.dumps(SPLIT_NAMES)
        writer.handle.attrs["sequence_content_deduplicated"] = True
        writer.handle.attrs["sequence_content_unique_per_length"] = True
        writer.handle.attrs["sequence_content_split_disjoint"] = True
        writer.handle.attrs["cross_length_coordinate_overlap"] = False
        writer.handle.attrs["generation_complete"] = False

        schedule_idx = 0
        seen_sequences = {length: set() for length in counts}
        candidates_examined = {length: 0 for length in counts}
        duplicate_rejections = {length: 0 for length in counts}
        try:
            for block_idx, (chrom, block_start, block_end) in enumerate(blocks):
                if schedule_idx == total_rows:
                    break
                block = fasta.fetch(chrom, block_start, block_end)
                for run_start, run_end in iter_acgt_runs(block):
                    cursor = run_start
                    while schedule_idx < total_rows:
                        length = int(schedule[schedule_idx])
                        if cursor + length > run_end:
                            break
                        sequence = block[cursor : cursor + length]
                        sequence_start = block_start + cursor
                        cursor += length
                        candidates_examined[length] += 1
                        if sequence in seen_sequences[length]:
                            duplicate_rejections[length] += 1
                            continue
                        seen_sequences[length].add(sequence)
                        writer.append(length, sequence, chrom, sequence_start)
                        schedule_idx += 1
                if (block_idx + 1) % 100 == 0:
                    LOGGER.info(
                        "Processed %d/%d blocks; accepted %s/%s windows",
                        block_idx + 1,
                        len(blocks),
                        f"{schedule_idx:,}",
                        f"{total_rows:,}",
                    )

            assert schedule_idx == total_rows, (
                f"Reference exhausted after {schedule_idx:,}/{total_rows:,} windows. "
                "The requested exact-content-unique quotas cannot be met from the "
                "selected reference. Reduce counts/block fragmentation or include "
                "more chromosomes."
            )
            writer.close()
        except BaseException:
            writer.handle.close()
            raise

    with h5py.File(partial, "r+") as handle:
        for length, target in counts.items():
            group = handle[f"lengths/{length}"]
            assert len(group["sequences"]) == target
            assert len(seen_sequences[length]) == target
            assert int(group.attrs["num_unique_sequences"]) == target
            starts = group["start"]
            seq_lengths = group["sequences"]
            assert len(starts) == len(seq_lengths)
            split = np.asarray(group["split"][:], dtype=np.uint8)
            observed = np.bincount(split, minlength=3)
            expected = np.asarray([target - val_size - test_size, val_size, test_size])
            assert np.array_equal(observed, expected), (length, observed, expected)
            group.attrs["candidates_examined"] = candidates_examined[length]
            group.attrs["duplicate_candidates_rejected"] = duplicate_rejections[
                length
            ]
        handle.attrs["num_sequences"] = total_rows
        handle.attrs["num_bases"] = total_bases
        handle.attrs["candidate_bases_examined"] = sum(
            length * candidates_examined[length] for length in counts
        )
        handle.attrs["duplicate_candidates_rejected"] = sum(
            duplicate_rejections.values()
        )
        handle.attrs["generation_complete"] = True
        handle.flush()

    os.replace(partial, output)
    LOGGER.info("Wrote complete multi-length corpus: %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a coordinate- and exact-content-disjoint multi-length hg38 corpus"
        )
    )
    parser.add_argument("--fasta", default="data/hg38.fa")
    parser.add_argument(
        "--output",
        default=(
            "data/hg38_multilen_seqdisjoint/hg38_multilen_seqdisjoint.h5"
        ),
    )
    parser.add_argument(
        "--counts",
        default=None,
        help="L:N comma list or JSON mapping; defaults to the publication allocation",
    )
    parser.add_argument("--val-size", type=int, default=50_000)
    parser.add_argument("--test-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--block-size",
        type=int,
        default=1_000_000,
        help="Genome block size shuffled before tiling (default: 1 Mb)",
    )
    parser.add_argument("--buffer-size", type=int, default=8192)
    parser.add_argument("--chromosomes", nargs="*", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    build_multilen_corpus(
        fasta_path=args.fasta,
        output_path=args.output,
        counts=parse_counts(args.counts),
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        block_size=args.block_size,
        buffer_size=args.buffer_size,
        chromosomes=args.chromosomes,
    )


if __name__ == "__main__":
    main()
