"""Correctness checks for compact/streamed identification scoring."""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.evaluation.identification import (
    _score_locus_compact,
    _score_matrix,
    _stream_oracle_sweep,
)
from src.evaluation.locus_blind import (
    _candidate_haplotypes,
    _fasta_matches,
    _fetch_cohort_snps_at_windows,
    _parse_sam_hits,
    _write_fasta,
)


def _tables():
    true_h0 = np.asarray(
        [["AAAA", "CCCC"], ["AAAA", "CCCT"], ["AAAT", "CCCT"]], dtype=object
    )
    true_h1 = np.asarray(
        [["AAAT", "CCCT"], ["AAAA", "CCCC"], ["AAAT", "CCCC"]], dtype=object
    )
    pred_h0 = np.asarray(
        [["AAAA", "CCCC"], ["AAAA", "CCCT"], ["AAAT", "CCCT"]], dtype=object
    )
    pred_h1 = np.asarray(
        [["AAAT", "CCCT"], ["AAAA", "CCCC"], ["AAAT", "CCCC"]], dtype=object
    )
    return {
        "individuals": ["i0", "i1", "i2"],
        "loci": [0, 1],
        "true_h0": true_h0,
        "true_h1": true_h1,
        "pred_h0": pred_h0,
        "pred_h1": pred_h1,
    }


def test_compact_locus_scores_equal_reference_nested_loop():
    tables = _tables()
    for locus in range(2):
        reference = _score_matrix(
            tables["pred_h0"],
            tables["pred_h1"],
            tables["true_h0"],
            tables["true_h1"],
            locus,
        )
        compact = _score_locus_compact(
            tables["pred_h0"],
            tables["pred_h1"],
            tables["true_h0"],
            tables["true_h1"],
            locus,
        )
        assert np.allclose(compact, reference)


def test_streamed_oracle_sweep_returns_all_requested_k_values():
    top1, top5, diagonal = _stream_oracle_sweep(
        tables=_tables(),
        k_grid=[1, 2, 4],
        num_repetitions=2,
        rng=np.random.default_rng(3),
        logger=logging.getLogger(__name__),
    )
    assert set(top1) == {1, 2}
    assert set(top5) == {1, 2}
    assert all(0.0 <= value <= 1.0 for value in top1.values())
    assert 0.0 <= diagonal <= 1.0


def test_locus_blind_bcftools_regions_use_file_not_argument_list(tmp_path, monkeypatch):
    """Large hit sets must not be expanded into the process argument vector."""
    vcf_dir = tmp_path / "vcfs"
    vcf_dir.mkdir()
    (vcf_dir / "chr1.vcf.gz").touch()

    # This is deliberately much larger than a normal unit-test fixture. The
    # command itself must remain constant-sized regardless of this list.
    windows = [("chr1", i * 10 + 1, i * 10 + 5) for i in range(20_000)]
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        kwargs["stdout"].write("chr1\t1\tA\tG\t0|1\n")
        return SimpleNamespace(
            returncode=0,
            stderr="",
        )

    monkeypatch.setattr("src.evaluation.locus_blind.subprocess.run", fake_run)
    result = _fetch_cohort_snps_at_windows(
        "bcftools",
        str(vcf_dir),
        windows,
        ["sample-1"],
        str(tmp_path),
    )

    cmd = captured["cmd"]
    assert "-r" not in cmd
    assert "-R" in cmd
    assert len(cmd) < 20
    regions_file = Path(cmd[cmd.index("-R") + 1])
    assert len(regions_file.read_text().splitlines()) == len(windows)
    assert result[windows[0]][0]["genotypes"] == ["0|1"]


def test_alignment_resume_requires_exact_query_fasta(tmp_path):
    queries = ["ACGT", "TGCA"]
    fasta = tmp_path / "queries.fa"
    _write_fasta(queries, str(fasta))

    assert _fasta_matches(queries, str(fasta))
    assert not _fasta_matches(["ACGT", "TGCC"], str(fasta))
    assert not _fasta_matches(["ACGT"], str(fasta))


def test_sam_parser_requires_complete_eos_decoded_query(tmp_path):
    sam = tmp_path / "queries.sam"
    sam.write_text(
        "@HD\tVN:1.6\n"
        "q0\t0\tchr1\t10\t60\t6M\t*\t0\t0\tACGTAA\t*\tNM:i:0\n"
        "q1\t0\tchr1\t20\t60\t4M1S\t*\t0\t0\tACGTA\t*\tNM:i:0\n"
    )
    hits = _parse_sam_hits(
        str(sam),
        n_queries=2,
        top_h_hits=5,
        query_lengths=[6, 5],
        require_all_queries=True,
    )
    assert hits[0] == [("chr1", 10, 0)]
    assert hits[1] == []


def test_candidate_haplotypes_preserve_true_length_and_phase():
    snps = [
        {"pos": 11, "ref": "A", "alt": "G", "genotypes": ["0|1", "1|0"]},
        {"pos": 13, "ref": "A", "alt": "T", "genotypes": ["0|0", "1|1"]},
    ]
    unique, h0_inverse, h1_inverse = _candidate_haplotypes(
        "AAAA", window_start=10, snps=snps, n_samples=2
    )
    assert all(len(haplotype) == 4 for haplotype in unique)
    assert unique[h0_inverse[0]] == "AAAA"
    assert unique[h1_inverse[0]] == "AGAA"
    assert unique[h0_inverse[1]] == "AGAT"
    assert unique[h1_inverse[1]] == "AAAT"
