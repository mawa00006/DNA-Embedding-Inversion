"""Generate DNA embeddings for any registered foundation model.

One Hydra entrypoint over ``src.model_registry``: ``fm_slug`` selects the spec,
``src.fm_engine`` turns it into the ``embed_chunk`` closure, and the shared
resumable engine in ``src/generation_io.py`` does the splitting, hashing and
HDF5 writing. The output is HDF5 files for train, val and test with datasets
'sequences' and 'embeddings'.

Only mean=true is supported: embeddings contains mean-pooled embeddings of shape
[embedding_dim] for each sequence.

Example usage:
    python generate/generate_embeddings.py fm_slug=grover input_path=data.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
import hydra.utils as hy_utils
from omegaconf import DictConfig

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import model_registry
from src.fm_engine import make_embed_chunk
from src.generation_io import run_generation
from src.utils import (
    set_determinism,
    load_sequences_from_file,
    load_identification_csv,
)


@hydra.main(config_path="../conf", config_name="generate/model", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint: load sequences and extract embeddings for cfg.fm_slug."""
    logger = logging.getLogger(__name__)

    spec = model_registry.get(str(cfg.fm_slug))
    assert spec.engine_supported, (
        f"{spec.slug} is not ported to the generic engine; it keeps its own "
        f"generate/generate_{spec.slug}_embeddings.py"
    )
    # run_generation stamps cfg.checkpoint as input provenance, and the registry
    # is the only place a checkpoint id is declared.
    cfg.checkpoint = spec.checkpoint
    logger.info(f"Generating DNA sequences and {spec.display_name or spec.slug} embeddings...")

    set_determinism(int(cfg.seed))

    # Load sequences from file or generate random ones
    input_path = hy_utils.to_absolute_path(cfg.input_path)
    identification_mode = cfg.identification_mode

    if identification_mode:
        logger.info(f"Loading identification CSV (preserving order, no dedup): {input_path}")
        id_data = load_identification_csv(input_path)
        sequences = id_data["sequences"]
        individual_ids = id_data["individual_ids"]
        locus_ids = id_data["locus_ids"]
        haplotypes = id_data["haplotypes"]
        # Validate sequence length consistency.
        assert all(len(s) == cfg.seq_length for s in sequences), (
            "Identification CSV contains sequences not matching cfg.seq_length"
        )
        logger.info(
            f"Loaded {len(sequences)} identification rows "
            f"({len(set(individual_ids))} individuals, {len(set(locus_ids))} loci)"
        )
    else:
        logger.info(f"Loading sequences from file: {input_path}")
        logger.info(f"Max sequences (num_sequences): {cfg.num_sequences}")
        logger.info(f"Max sequence length (seq_length): {cfg.seq_length}")
        sequences = load_sequences_from_file(
            input_path, max_length=cfg.seq_length, max_sequences=cfg.num_sequences
        )
        individual_ids = None
        locus_ids = None
        haplotypes = None

        if bool(cfg.get("deduplicate_sequences", True)):
            original_count = len(sequences)
            sequences = list(dict.fromkeys(sequences))
            duplicates_removed = original_count - len(sequences)
            if duplicates_removed > 0:
                logger.warning(f"Removed {duplicates_removed} duplicate sequences")

        logger.info(
            f"Loaded {len(sequences)} sequence windows from file "
            f"(limited to {cfg.num_sequences}, target length {cfg.seq_length})"
        )

    device = cfg.device
    logger.info(f"Using checkpoint: {spec.checkpoint} (revision={spec.revision})")
    logger.info(f"Using device: {device}")

    meta: Dict[str, Any] = {}

    def build_embed_chunk():
        embed_chunk, probed = make_embed_chunk(spec, device, int(cfg.batch_size))
        assert int(cfg.seq_length) <= probed["max_length"], (
            f"Requested seq_length={cfg.seq_length} > model max positions="
            f"{probed['max_length']}"
        )
        meta.update(probed)
        return embed_chunk

    # run_generation calls build_embed_chunk before finalize writes attrs, so
    # meta is populated by then. On the fast-exit path it stays empty and the
    # existing files keep the provenance of the run that wrote them.
    if run_generation(
        cfg,
        sequences=sequences,
        individual_ids=individual_ids,
        locus_ids=locus_ids,
        haplotypes=haplotypes,
        make_embed_chunk=build_embed_chunk,
        logger=logger,
        extra_attrs=meta,
    ):
        return

    # Only reachable for mean=false, which this pipeline never uses.
    raise NotImplementedError(
        "mean=false (per-token embeddings) is not supported. "
        "The multilen benchmark always runs mean=true."
    )


if __name__ == "__main__":  # pragma: no cover
    main()  # type: ignore
