#!/usr/bin/env bash
# Reproduce the ablation table (Table 2) on TSB-AD-M.
# Runs all five configurations (full model + 4 ablations) sequentially.
# Usage: bash scripts/run_ablation.sh /path/to/TSB-AD-M [seed]

DATA_DIR=${1:?"Usage: $0 <data_dir> [seed]"}
SEED=${2:-2027}

CONFIGS=(
  "configs/full_model.yaml"
  "configs/ablation_no_channel.yaml"
  "configs/ablation_no_mahalanobis.yaml"
  "configs/ablation_no_vel_pretext.yaml"
  "configs/ablation_no_vel_scoring.yaml"
)

NAMES=(
  "full_model"
  "no_channel"
  "no_mahalanobis"
  "no_vel_pretext"
  "no_vel_scoring"
)

for i in "${!CONFIGS[@]}"; do
  echo "===== Running: ${NAMES[$i]} ====="
  python main.py \
    --config "${CONFIGS[$i]}" \
    --data_dir "$DATA_DIR" \
    --output_dir "results/ablation/${NAMES[$i]}" \
    --seed "$SEED"
done

echo "Done. Results in results/ablation/"
