#!/bin/bash
set -euo pipefail

# Identification data restricted to held-out intervals from the sequence-
# disjoint source corpus. Distinct filenames keep every earlier cohort intact.
SOURCE_MULTILEN_H5=${SOURCE_MULTILEN_H5:-data/hg38_multilen_seqdisjoint/hg38_multilen_seqdisjoint.h5}
export SOURCE_MULTILEN_H5
export ID_VARIANT_PREFIX=multilen_seqdisjoint_

bash prepare_and_generate_identification.sh "$@"
