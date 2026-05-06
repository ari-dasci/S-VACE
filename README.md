# VACE: Learning Geometrically Structured Representations for Time Series Anomaly Detection

VACE frames multivariate time-series anomaly detection as a geometric problem in the embedding space. A patch is anomalous if it sits at an unusual **position** or moves in an unusual **direction** through the learned manifold.

## Method Overview

| Component | Description |
|-----------|-------------|
| **Channel-Aware Encoder** | Depthwise-separable 1D-CNN: per-channel convolutions first, then pointwise mixing. Prevents cross-channel dilution in multivariate data. |
| **Velocity Pretext Task** | Trains the encoder so that temporal velocity vectors are directionally coherent. Aligns training objective with test-time scoring. |
| **Mahalanobis Scoring** | Fits a full-covariance Gaussian to training embeddings. Scores test patches by Mahalanobis distance (position). |
| **Velocity Bank Scoring** | Builds a KMeans bank of normalised velocity vectors from training. Scores test patches by cosine distance to nearest velocity prototype (direction). |
| **Combined Score** | `score = mahal_norm × (1 + w × vel_norm)`. High score when both state AND dynamics are unusual. |

## Requirements

- Python ≥ 3.9
- CUDA GPU recommended (runs on CPU but is slow)

```bash
pip install -r requirements.txt
```

Wandb is optional. Set `no_wandb: true` in the config (already the default) to run without it.

## Data

Download **TSB-AD-M** from the [TSB-AD benchmark](https://github.com/TheDatumOrg/TSB-AD). Place the 200 CSV files in a directory (e.g. `data/TSB-AD-M/`).

Each CSV file follows the naming convention `NNN_CATEGORY_id_N_..._tr_TTTT_....csv`, where `TTTT` is the train/test split index. The last column is the anomaly label.

## Reproducing Main Results

```bash
bash scripts/run.sh /path/to/TSB-AD-M results/full_model
```

Expected results on TSB-AD-M (200 files, seed=2027):

| VUS-PR | VUS-ROC | AUC-ROC | AUC-PR | BestF1 | RangeF1 |
|--------|---------|---------|--------|--------|---------|
| 0.487  | 0.788   | 0.760   | 0.434  | 0.470  | 0.462   |

Results are written to `results/full_model/summary_metrics.csv` and `categorical_metrics.csv`.

## Reproducing the Ablation Table

```bash
bash scripts/run_ablation.sh /path/to/TSB-AD-M
```

This runs all five configurations sequentially:

| Configuration | VUS-PR |
|---------------|--------|
| Full model    | 0.487  |
| No channel encoder | 0.443 |
| No Mahalanobis (KNN) | 0.477 |
| No velocity pretext | 0.450 |
| No velocity scoring | 0.483 |

To reproduce the mean-over-10-seeds results from the paper, run each ablation with seeds `2000 389 2 2027 54321 789 1234 5678 9999 6789` and average VUS-PR across seeds.

## Configuration

All hyperparameters are set in YAML files under `configs/`. Key parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `patch_size` | 96 | Temporal patch length |
| `num_iters` | 20 | Total training iterations |
| `pretext_fraction` | 1.0 | Fraction of iters with velocity loss (linear decay) |
| `velocity_delta` | 48 | Temporal stride for velocity computation (≈ patch_size / 2) |
| `channel_expansion` | 8 | Depthwise expansion factor in Stage 1 |
| `use_mahalanobis` | true | Full-covariance Mahalanobis scoring |
| `velocity_weight` | 1.0 | Weight of velocity score in combined scoring |
| `use_revin` | true | RevIN instance normalisation |

CLI overrides take priority over the YAML config:

```bash
python main.py --config configs/full_model.yaml --data_dir /data/TSB-AD-M --seed 42
```

## Embedding Geometry Diagnostics

`utils/geometry.py` exposes `compute_embedding_geometry(model, train_loader, device)`, which computes four spectral diagnostics of the training embedding distribution:

| Metric | Description |
|--------|-------------|
| `eff_rank_pr` | Participation-ratio effective rank / d |
| `eff_rank_entropy` | Entropy effective rank / d |
| `n_active_dims` | Number of eigenvalues above 1e-10 |
| `var_top1` | Fraction of variance in the leading component |

These are the metrics used in the geometry analysis of the paper (Section 4.3 and Appendix).

## Project Structure

```
VACE/
├── main.py                  # Entry point and evaluation loop
├── model.py                 # PatchChannelEncoder (full model) + PatchEncoder (ablation)
├── train.py                 # Velocity pretext + BN calibration training
├── utils/
│   ├── data_preprocess.py   # Data loading, patch creation, sliding window
│   ├── evaluation.py        # Mahalanobis, velocity bank, and KNN scorers
│   ├── geometry.py          # Embedding geometry diagnostics (eff. rank, var_top1, active dims)
│   ├── metrics.py           # VUS-PR, VUS-ROC, AUC-ROC, AUC-PR, F1 metrics
│   └── utils.py             # Memory bank (KMeans coreset), RevIN
├── affiliation/             # Affiliation-F metric (from TSB-AD)
├── configs/
│   ├── full_model.yaml      # Paper main results config
│   ├── ablation_no_channel.yaml
│   ├── ablation_no_mahalanobis.yaml
│   ├── ablation_no_vel_pretext.yaml
│   └── ablation_no_vel_scoring.yaml
└── scripts/
    ├── run.sh               # Reproduce main results
    └── run_ablation.sh      # Reproduce ablation table
```
