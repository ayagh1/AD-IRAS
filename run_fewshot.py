"""
run_fewshot.py — Few-Shot Learning Curve Experiment
Run on local PC with RTX 2070 GPU.
"""

import sys
import os
import yaml
import time
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone

if __name__ == '__main__':
    import torch
    import gc

    sys.path.insert(0, str(Path(__file__).parent))

    from src.utils import load_config, set_seed
    from src.benchmarker import BenchmarkRunner
    from src.data_loader import DatasetFactory

    cfg_path     = Path('config/settings.yaml')
    results_dir  = Path('results')
    dataset_base = Path('datasets/FewShotCurve')
    results_dir.mkdir(parents=True, exist_ok=True)

    k_values = [5, 10, 20, 30, 46]
    models   = ['Uflow', 'Fastflow', 'Padim']

    all_results = []

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"k = {k} training images")
        print(f"{'='*60}")

        data_source   = f'FewShotCurve_k{k}'
        mother_folder = dataset_base / f'k{k}'

        if not mother_folder.exists():
            print(f"  ❌ Dataset not found: {mother_folder} — skipping")
            continue

        # ── Register dataset with correct folder names ─────────────────
        DatasetFactory.SUPPORTED[data_source] = {
            "split":          "folder",
            "masks":          True,
            "normal_dir":     "train/good",
            "test_good":      "test/good",
            "test_bad":       "test/bad",
            "mask_dir":       "ground_truth/bad",
            "rename_anomaly": False,
        }

        # ── Update config ──────────────────────────────────────────────
        with open(cfg_path, 'r') as f:
            cfg_raw = yaml.safe_load(f)

        cfg_raw['run']['data_source']     = data_source
        cfg_raw['run']['categories']      = ['Computer']
        cfg_raw['run']['seeds']           = [420]
        cfg_raw['models']['to_run']       = models
        cfg_raw['training']['resize_img'] = 256
        cfg_raw['training']['num_workers'] = 0  # Windows fix

        with open(cfg_path, 'w') as f:
            yaml.dump(cfg_raw, f, default_flow_style=False, allow_unicode=True)

        cfg = load_config(cfg_path)
        set_seed(42)

        ts         = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        session_id = f'{ts}_256px_fewshot_k{k}'

        workspace  = Path(cfg['paths']['local_workspace'])
        output_dir = Path(cfg['paths']['output_dir'])
        local_path = workspace / data_source / 'Computer'

        workspace.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Copy dataset to local workspace ────────────────────────────
        if local_path.exists():
            shutil.rmtree(local_path)
        shutil.copytree(mother_folder / 'Computer', local_path)
        print(f"  Dataset copied to local workspace.")

        # ── Rename anomaly → bad ───────────────────────────────────────
        for kind in ('test', 'ground_truth'):
            src = local_path / kind / 'anomaly'
            dst = local_path / kind / 'bad'
            if src.exists() and not dst.exists():
                src.rename(dst)
                print(f"  Renamed: {kind}/anomaly → {kind}/bad")

        # ── Verify folders ─────────────────────────────────────────────
        required = [
            local_path / 'train' / 'good',
            local_path / 'test'  / 'good',
            local_path / 'test'  / 'bad',
            local_path / 'ground_truth' / 'bad',
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            print(f"  ❌ Missing: {missing} — skipping k={k}")
            shutil.rmtree(local_path, ignore_errors=True)
            continue

        n_train = len(list((local_path / 'train' / 'good').glob('*')))
        n_test  = len(list((local_path / 'test'  / 'bad').glob('*')))
        n_masks = len(list((local_path / 'ground_truth' / 'bad').glob('*')))
        print(f"  Train: {n_train} | Test bad: {n_test} | Masks: {n_masks}")

        # ── Run benchmark ──────────────────────────────────────────────
        runner = BenchmarkRunner(
            data_source = data_source,
            category    = 'Computer',
            base_path   = workspace,
            output_path = output_dir,
            session_id  = session_id,
            cfg         = cfg,
        )

        t0 = time.time()
        raw_df, agg_df, lb_df, pw_df, pred_df = runner.run_benchmark(
            model_list = models,
            seeds      = [420],
        )
        elapsed = time.time() - t0

        # ── Collect results ────────────────────────────────────────────
        if not agg_df.empty:
            for _, row in agg_df.iterrows():
                auroc = row.get('Image AUROC mean', row.get('Image AUROC', 0))
                aupro = row.get('Pixel AUPRO mean', row.get('Pixel AUPRO', 0))
                all_results.append({
                    'k':           k,
                    'Model':       row['Model'],
                    'Image AUROC': auroc,
                    'Pixel AUPRO': aupro,
                })
                print(f"  ✅ {row['Model']:<15} AUROC={auroc:.4f}")
        else:
            print(f"  ❌ No results for k={k}")

        # ── Save partial results after each k ──────────────────────────
        if all_results:
            pd.DataFrame(all_results).to_csv(
                results_dir / 'fewshot_curve.csv', index=False)
            print(f"  💾 Saved ({len(all_results)} rows so far)")

        # ── Cleanup ────────────────────────────────────────────────────
        torch.cuda.empty_cache()
        gc.collect()
        shutil.rmtree(local_path, ignore_errors=True)
        print(f"  ⏱ k={k} done in {elapsed/60:.1f} mins")

    # ── Reset settings.yaml ────────────────────────────────────────────
    with open(cfg_path, 'r') as f:
        cfg_raw = yaml.safe_load(f)
    cfg_raw['run']['data_source'] = 'MaskedDataset'
    cfg_raw['models']['to_run']   = ['AnomalyDINO', 'Padim', 'Patchcore',
                                      'WinClip', 'Fastflow', 'Uflow']
    cfg_raw['training']['num_workers'] = 0
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg_raw, f, default_flow_style=False, allow_unicode=True)
    print("\n✅ settings.yaml reset to MaskedDataset")

    # ── Final plot ─────────────────────────────────────────────────────
    if not all_results:
        print("❌ No results collected!")
        sys.exit(1)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / 'fewshot_curve.csv', index=False)
    print(f"\n✅ Final results saved! ({len(results_df)} rows)")

    pivot = results_df.pivot(index='k', columns='Model', values='Image AUROC')
    print("\nImage AUROC by k:")
    print(pivot.round(4).to_string())

    BG     = '#ffffff'
    TEXT   = '#0b0b0b'
    MUTED  = '#898781'
    GRID   = '#e8e7e0'
    COLORS = {'Uflow': '#2a78d6', 'Fastflow': '#1baf7a', 'Padim': '#eda100'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle('Few-Shot Learning Curve\nPerformance vs. Number of Training Images',
                 fontsize=14, fontweight='bold', color=TEXT)

    for model in models:
        df_m = results_df[results_df['Model'] == model].sort_values('k')
        if df_m.empty:
            continue
        axes[0].plot(df_m['k'], df_m['Image AUROC']*100,
                     marker='o', label=model,
                     color=COLORS.get(model, '#888888'),
                     linewidth=2, markersize=7)
        axes[1].plot(df_m['k'], df_m['Pixel AUPRO']*100,
                     marker='o', label=model,
                     color=COLORS.get(model, '#888888'),
                     linewidth=2, markersize=7)

    for ax, metric in zip(axes, ['Image AUROC (%)', 'Pixel AUPRO (%)']):
        ax.set_xlabel('Number of training images (k)', color=MUTED, fontsize=10)
        ax.set_ylabel(metric, color=MUTED, fontsize=10)
        ax.set_xticks(k_values)
        ax.axhline(50, color='#e34948', linestyle='--',
                   linewidth=1.2, label='Random baseline (50%)')
        ax.legend(fontsize=9, frameon=False)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=MUTED)

    plt.tight_layout()
    plt.savefig(results_dir / 'fewshot_curve.png',
                dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()
    print("✅ Plot saved to results/fewshot_curve.png")