import os
import sys
import time
import random
import warnings
import yaml
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import UndefinedMetricWarning

from model import PatchEncoder, PatchChannelEncoder
from train import train_model
from utils.data_preprocess import *
from utils.utils import *
from utils.evaluation import *
from utils.metrics import get_metrics

warnings.simplefilter("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ENCODER_REGISTRY = {
    "PatchEncoder": PatchEncoder,
    "PatchChannelEncoder": PatchChannelEncoder,
}


class VACE:
    def __init__(self, data_dir, output_dir=None, patch_size=64,
                 num_iters=20, lr=1e-4, batch_size=512, random_seed=2027,
                 device=None, use_revin=True,
                 encoder="PatchChannelEncoder",
                 datasets=None, file_list=None,
                 pretext_fraction=1.0, use_mahalanobis=True,
                 velocity_weight=1.0, velocity_delta=48,
                 channel_expansion=8, use_bn=True):

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.num_iters = num_iters
        self.lr = lr
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_revin = use_revin

        if encoder not in ENCODER_REGISTRY:
            raise ValueError(f"Unknown encoder '{encoder}'. Choose from: {list(ENCODER_REGISTRY)}")
        self.encoder = encoder

        self.datasets = [d.upper() for d in datasets] if datasets else None
        self.file_list = None
        if file_list is not None:
            with open(file_list) as f:
                lines = [l.strip() for l in f if l.strip()]
            self.file_list = set(lines[1:] if lines[0].lower() == 'file_name' else lines)

        self.pretext_fraction = pretext_fraction
        self.use_mahalanobis = use_mahalanobis
        self.velocity_weight = velocity_weight
        self.velocity_delta = velocity_delta
        self.channel_expansion = channel_expansion
        self.use_bn = use_bn

    def run(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)

        dis_aurocs, dis_auprcs, dis_vuspr = [], [], []
        dis_vusroc, dis_f1, dis_Rfl = [], [], []
        results_by_category = defaultdict(list)
        summary_rows = []

        all_csv_files = sorted(f for f in os.listdir(self.data_dir) if f.endswith('.csv'))
        if self.file_list is not None:
            all_csv_files = [f for f in all_csv_files if f in self.file_list]
            print(f"VACE running... (file_list: {len(all_csv_files)} files)")
        elif self.datasets is not None:
            all_csv_files = [f for f in all_csv_files
                             if f.split('_')[1].upper() in self.datasets]
            print(f"VACE running... (filter: {self.datasets}, {len(all_csv_files)} files)")
        else:
            print(f"VACE running... ({len(all_csv_files)} files)")

        total = len(all_csv_files)

        for idx, file_name in enumerate(all_csv_files, 1):
            file_path = os.path.join(self.data_dir, file_name)
            print(f"\033[1m══ ({idx}/{total})\033[0m {file_name}")

            train_data, train_labels, test_data, test_labels = load_and_split_data(file_path)
            train_data = np.array(train_data, dtype=np.float32)
            test_data  = np.array(test_data,  dtype=np.float32)
            test_labels = np.array(test_labels, dtype=np.float32)

            if not self.use_revin:
                train_mean = np.mean(train_data, axis=0, keepdims=True).astype(np.float32)
                train_std  = np.std(train_data,  axis=0, keepdims=True).astype(np.float32)
                train_std  = np.where(train_std == 0.0, 1e-8, train_std)
                train_data = (train_data - train_mean) / train_std
                test_data  = (test_data  - train_mean) / train_std

            if test_data.ndim == 1:
                sliding_input = test_data.reshape(-1, 1)
            else:
                sliding_input = test_data[:, 0].reshape(-1, 1)
            slidingWindow = find_length_rank(sliding_input, rank=1)

            t0 = time.time()
            patch_creator = PatchCreator(L=self.patch_size, s=1, random_seed=self.random_seed)

            train_loader, _, _ = patch_creator.create_dataloaders(
                train_data, train_data, train_labels, batch_size=self.batch_size)
            _, test_loader, true_test_labels = patch_creator.create_dataloaders(
                train_data, test_data, test_labels, batch_size=self.batch_size)

            xb, _ = next(iter(train_loader))
            C = xb.shape[1]
            print(f"    in_channels={C}, encoder={self.encoder}")

            model = ENCODER_REGISTRY[self.encoder](
                in_channels=C, use_revin=self.use_revin,
                **({'channel_expansion': self.channel_expansion, 'use_bn': self.use_bn}
                   if self.encoder == 'PatchChannelEncoder' else {})
            ).to(self.device)

            train_model(
                model,
                train_loader,
                preprocess_to_patches(train_data, patch_size=self.patch_size, stride=1),
                self.device,
                num_iter=self.num_iters,
                pretext_step=self.patch_size,
                lr=self.lr,
                file_name=file_name,
                pretext_fraction=self.pretext_fraction,
            )

            # --- Anomaly scoring ---
            mahal_params = None
            if self.use_mahalanobis:
                mahal_params = fit_mahalanobis(model, train_loader, self.device, reg=1e-6, mode="full")
                all_scores = calculate_anomaly_scores_mahalanobis(model, test_loader, mahal_params, self.device)
            else:
                memory_bank, _ = create_memory_bank(model, train_loader, self.device, num_cores=0.1)
                all_scores = calculate_anomaly_scores(model, test_loader, memory_bank, top_k=3, device=self.device)

            if self.velocity_weight != 0.0:
                train_embs = collect_ordered_embeddings(model, train_loader, self.device)
                vel_bank   = build_velocity_bank(train_embs, delta=self.velocity_delta, num_cores=0.1)
                vel_scores = calculate_anomaly_scores_velocity(
                    model, test_loader, vel_bank, self.device,
                    delta=self.velocity_delta, top_k=3)

                def _minmax(arr):
                    a = np.array(arr, dtype=np.float32)
                    lo, hi = a.min(), a.max()
                    return a if hi == lo else (a - lo) / (hi - lo)

                mb_norm  = _minmax(all_scores)
                vel_norm = _minmax(vel_scores)
                all_scores = (mb_norm * (1.0 + self.velocity_weight * vel_norm)).tolist()

            dist_scores = distribute_patch_scores_to_points(
                all_scores, patch_size=self.patch_size, num_points=len(test_labels))
            t1 = time.time()

            if test_labels.sum() == 0:
                print(f"    [SKIP] No anomalies in test set.")
                continue

            results = get_metrics(dist_scores, test_labels,
                                  slidingWindow=slidingWindow, pred=None,
                                  version='opt', thre=250)

            dis_aurocs.append(results['AUC-ROC'])
            dis_auprcs.append(results['AUC-PR'])
            dis_vuspr.append(results['VUS-PR'])
            dis_vusroc.append(results['VUS-ROC'])
            dis_f1.append(results['Standard-F1'])
            dis_Rfl.append(results['R-based-F1'])

            category = file_name.split('_')[1]
            ds_id    = f"id_{file_name.split('_')[3]}" if len(file_name.split('_')) > 3 else file_name.split('_')[0]
            results_by_category[category].append(results)

            print(f"    AUC-ROC={results['AUC-ROC']:.4f}  AUC-PR={results['AUC-PR']:.4f}  "
                  f"VUS-PR={results['VUS-PR']:.4f}  VUS-ROC={results['VUS-ROC']:.4f}  "
                  f"BestF1={results['Standard-F1']:.4f}  RangeF1={results['R-based-F1']:.4f}  "
                  f"({t1-t0:.1f}s)")

            if self.output_dir:
                summary_rows.append({
                    'file': file_name, 'Category': category,
                    'AUC-ROC': results['AUC-ROC'], 'AUC-PR': results['AUC-PR'],
                    'VUS-PR':  results['VUS-PR'],  'VUS-ROC': results['VUS-ROC'],
                    'BestF1':  results['Standard-F1'], 'RangeF1': results['R-based-F1'],
                })
                scores_dir = os.path.join(self.output_dir, "Filewise_scores")
                os.makedirs(scores_dir, exist_ok=True)
                pd.DataFrame({'True Labels': test_labels, 'Anomaly scores': dist_scores}).to_csv(
                    os.path.join(scores_dir, f"{file_name}_output.csv"), index=False)

        # --- Aggregate results ---
        dist_auroc, dist_auprc, dist_vuspr, dist_vusroc, dist_F1, dist_RF1 = map(
            np.mean, [dis_aurocs, dis_auprcs, dis_vuspr, dis_vusroc, dis_f1, dis_Rfl])

        print(f"\nVACE Final Results ({len(dis_aurocs)} files): "
              f"AUROC={dist_auroc:.4f}  AUPRC={dist_auprc:.4f}  "
              f"VUSPR={dist_vuspr:.4f}  VUSROC={dist_vusroc:.4f}  "
              f"F1={dist_F1:.4f}  RangeF1={dist_RF1:.4f}")

        print("\nPer-category breakdown:")
        for cat_name, cat_results in sorted(results_by_category.items()):
            print(f"  {cat_name:20s}  N={len(cat_results):3d}  "
                  f"VUS-PR={np.mean([r['VUS-PR'] for r in cat_results]):.4f}  "
                  f"AUC-ROC={np.mean([r['AUC-ROC'] for r in cat_results]):.4f}")

        if self.output_dir and summary_rows:
            summary_df = pd.concat([
                pd.DataFrame(summary_rows),
                pd.DataFrame([{'file': 'AVERAGE',
                               'AUC-ROC': float(dist_auroc), 'AUC-PR': float(dist_auprc),
                               'VUS-PR':  float(dist_vuspr), 'VUS-ROC': float(dist_vusroc),
                               'BestF1':  float(dist_F1),    'RangeF1': float(dist_RF1)}])
            ], ignore_index=True)
            summary_path = os.path.join(self.output_dir, 'summary_metrics.csv')
            summary_df.to_csv(summary_path, index=False)

            cat_df = (
                summary_df[summary_df['file'] != 'AVERAGE']
                .groupby('Category')
                .agg(N=('file', 'count'),
                     AUC_ROC=('AUC-ROC', 'mean'), AUC_PR=('AUC-PR', 'mean'),
                     VUS_PR=('VUS-PR', 'mean'),   VUS_ROC=('VUS-ROC', 'mean'),
                     BestF1=('BestF1', 'mean'),   RangeF1=('RangeF1', 'mean'))
                .reset_index()
            )
            cat_df.to_csv(os.path.join(self.output_dir, 'categorical_metrics.csv'), index=False)
            print(f"\nResults saved to {self.output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VACE: Anomaly Detection")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a YAML config file. CLI flags override YAML values.')
    args, extra = parser.parse_known_args()

    defaults = dict(
        data_dir=None, output_dir=None, patch_size=96, num_iters=20,
        batch_size=512, lr=1e-4, seed=2027, use_revin=True,
        encoder='PatchChannelEncoder',
        datasets=None, file_list=None,
        pretext_fraction=1.0, use_mahalanobis=True,
        velocity_weight=1.0, velocity_delta=48,
        channel_expansion=8, use_bn=True,
    )

    cfg = {}
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # Parse extra CLI overrides: --key value or --bool_flag
    cli_overrides = {}
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok.startswith('--'):
            key = tok.lstrip('-').replace('-', '_')
            if i + 1 < len(extra) and not extra[i + 1].startswith('--'):
                raw = extra[i + 1]
                i += 2
            else:
                raw = True
                i += 1
            ref = defaults.get(key)
            if isinstance(raw, str) and ref is not None:
                if isinstance(ref, bool):
                    raw = raw.lower() in ('true', '1', 'yes')
                elif isinstance(ref, int):
                    raw = int(raw)
                elif isinstance(ref, float):
                    raw = float(raw)
            cli_overrides[key] = raw
        else:
            i += 1

    final = {**defaults, **cfg, **cli_overrides}

    if final['data_dir'] is None:
        parser.error("data_dir is required (--data_dir or via config YAML)")

    experiment = VACE(
        data_dir=final['data_dir'],
        output_dir=final['output_dir'],
        patch_size=final['patch_size'],
        num_iters=final['num_iters'],
        batch_size=final['batch_size'],
        lr=final['lr'],
        random_seed=final['seed'],
        use_revin=final['use_revin'],
        encoder=final['encoder'],
        datasets=final['datasets'],
        file_list=final['file_list'],
        pretext_fraction=final['pretext_fraction'],
        use_mahalanobis=final['use_mahalanobis'],
        velocity_weight=final['velocity_weight'],
        velocity_delta=final['velocity_delta'],
        channel_expansion=final['channel_expansion'],
        use_bn=final['use_bn'],
    )
    experiment.run()
