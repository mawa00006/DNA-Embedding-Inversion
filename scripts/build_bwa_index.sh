#!/bin/bash
# Build a BWA index for the hg38 reference if it does not already exist.
#
# The locus-blind identification path aligns predicted sequences to hg38 using
# BWA-MEM with substitution-only parameters; this builds the index it needs.

set -euo pipefail

REFERENCE="data/hg38.fa"

if ! command -v bwa >/dev/null 2>&1; then
    echo "ERROR: bwa not on PATH. Activate the project conda env (init.sh installs it)." >&2
    exit 1
fi

if [ ! -f "$REFERENCE" ]; then
    echo "ERROR: reference not found at $REFERENCE" >&2
    exit 1
fi

if [ -f "${REFERENCE}.bwt" ] && [ -f "${REFERENCE}.pac" ] && [ -f "${REFERENCE}.sa" ]; then
    echo "[Skip] BWA index already exists for $REFERENCE"
    exit 0
fi

echo "Building BWA index for $REFERENCE (this takes ~30 min)..."
bwa index "$REFERENCE"
echo "Done."
