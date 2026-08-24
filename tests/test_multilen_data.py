"""Tests for the coordinate-aware multi-length source corpus."""

from pathlib import Path

import h5py
import numpy as np

from scripts.prepare_hg38_multilen import (
    build_multilen_corpus,
    deterministic_split_codes,
)
from scripts.prepare_1000g_multilen import write_group
from scripts.prepare_identification import load_multilen_test_candidates
from src.generation_io import (
    deterministic_split_labels,
    generate_split_h5_resumable,
)
from src.utils import load_sequences_from_file
from evaluate import _redirect_test_csv_to_1000g


def _write_indexed_fasta(path: Path, sequence: str, line_bases: int = 50) -> None:
    header = b">chr1\n"
    encoded = sequence.encode("ascii")
    lines = [encoded[i : i + line_bases] for i in range(0, len(encoded), line_bases)]
    path.write_bytes(header + b"\n".join(lines) + b"\n")
    line_width = line_bases + 1
    path.with_suffix(path.suffix + ".fai").write_text(
        f"chr1\t{len(sequence)}\t{len(header)}\t{line_bases}\t{line_width}\n",
        encoding="utf-8",
    )


def test_multilen_corpus_has_no_cross_length_or_split_overlap(tmp_path: Path):
    fasta = tmp_path / "tiny.fa"
    rng = np.random.default_rng(123)
    reference = "".join(rng.choice(list("ACGT"), size=1500))
    _write_indexed_fasta(fasta, reference)
    output = tmp_path / "tiny_multilen.h5"
    counts = {5: 12, 11: 20, 23: 24}

    build_multilen_corpus(
        fasta_path=str(fasta),
        output_path=str(output),
        counts=counts,
        val_size=2,
        test_size=2,
        seed=7,
        block_size=250,
        buffer_size=4,
        chromosomes=["chr1"],
    )

    intervals = []
    with h5py.File(output, "r") as handle:
        assert handle.attrs["generation_complete"]
        for length, count in counts.items():
            group = handle[f"lengths/{length}"]
            assert len(group["sequences"]) == count
            decoded_sequences = {
                sequence.decode("ascii") for sequence in group["sequences"][:]
            }
            assert len(decoded_sequences) == count
            assert np.array_equal(
                np.bincount(group["split"][:], minlength=3),
                np.asarray([count - 4, 2, 2]),
            )
            for sequence, start in zip(group["sequences"][:], group["start"][:]):
                start = int(start)
                decoded = sequence.decode("ascii")
                assert decoded == reference[start : start + length]
                intervals.append((start, start + length))

    intervals.sort()
    assert all(left_end <= right_start for (_left_start, left_end), (right_start, _right_end) in zip(intervals, intervals[1:]))

    length_11 = load_sequences_from_file(
        str(output), max_length=11, max_sequences=7
    )
    assert len(length_11) == 7
    assert all(len(sequence) == 11 for sequence in length_11)

    identification_loci = load_multilen_test_candidates(str(output), seq_length=5)
    zero_based = sorted((start - 1, end) for _chrom, start, end in identification_loci)
    assert all(
        left_end <= right_start
        for (_left_start, left_end), (right_start, _right_end) in zip(
            zero_based, zero_based[1:]
        )
    )


def test_source_split_codes_match_embedding_writer():
    source_codes = deterministic_split_codes(101, val_size=13, test_size=17, seed=42)
    writer_labels = deterministic_split_labels(
        101, 0.9, 0.05, 42, val_size=13, test_size=17
    )
    writer_codes = np.asarray(
        [{"train": 0, "val": 1, "test": 2}[label] for label in writer_labels],
        dtype=np.uint8,
    )
    assert np.array_equal(source_codes, writer_codes)


def test_1000g_group_uses_the_same_sequence_loader(tmp_path: Path):
    output = tmp_path / "tiny_1000g.h5"
    windows = [("chr1", 10, "AAAAA"), ("chr1", 30, "CCCCC")]
    with h5py.File(output, "w") as handle:
        handle.attrs["schema"] = "dna_inversion_1000g_multilen_v2"
        write_group(
            handle,
            length=5,
            windows=windows,
            sequences=["AAAAT", "CCCCG"],
            samples=["HG00001", "HG00002"],
            haplotypes=[0, 1],
        )
    assert load_sequences_from_file(str(output), max_length=5) == ["AAAAT", "CCCCG"]
    with h5py.File(output, "r") as handle:
        group = handle["lengths/5"]
        assert list(group["differs_from_reference"][:]) == [True, True]
        assert [value.decode("ascii") for value in group["reference_sequences"][:]] == [
            "AAAAA",
            "CCCCC",
        ]


def test_multilen_ood_path_redirects_to_real_data_directory():
    assert _redirect_test_csv_to_1000g(
        "data/hg38_multilen_seqdisjoint/test_ntv2_25_hg38_multilen_seqdisjoint.h5"
    ) == (
        "data/1000g_multilen_seqdisjoint/"
        "test_ntv2_25_1000g_multilen_seqdisjoint.h5"
    )



