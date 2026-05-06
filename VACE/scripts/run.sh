#!/usr/bin/env bash
# Reproduce VACE main results on TSB-AD-M.
# Usage: bash scripts/run.sh /path/to/TSB-AD-M [output_dir]

DATA_DIR=${1:?"Usage: $0 <data_dir> [output_dir]"}
OUTPUT_DIR=${2:-"results/full_model"}

python main.py \
  --config configs/full_model.yaml \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR"
