import sys
import os
import shutil
import time
import gc
from pathlib import Path
from datetime import datetime, timezone

if __name__ == '__main__':
    import torch
    from src.utils import load_config, set_seed
    from src.benchmarker import BenchmarkRunner
    from src.augmentation import build_augmentation_pipeline, inject_augmented_images
    from src.utils import _collect_images, _count_images

    sys.path.insert(0, str(Path(__file__).parent))

    cfg = load_config(Path('config/settings.yaml'))
    set_seed(cfg['training'].get('seed', 42))

    paths_cfg   = cfg['paths']
    run_cfg     = cfg['run']
    aug_cfg     = cfg['augmentation']
    data_source = run_cfg['data_source']
    seeds       = run_cfg['seeds']
    models      = cfg['models']['to_run']

    output_dir  = Path(paths_cfg['output_dir'])
    workspace   = Path(paths_cfg['local_workspace'])
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    ts           = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    image_size   = cfg['training']['resize_img']
    n_models     = len(models)
    model_suffix = models[0] if n_models == 1 else f'{n_models}models'
    session_id   = f'{ts}_{image_size}px_noise_injection_{model_suffix}'
    run_folder   = output_dir / f'{data_source}_{session_id}'
    run_folder.mkdir(parents=True, exist_ok=True)

    print(f"Session: {session_id}")
    print(f"Models:  {models}")
    print(f"Dataset: {data_source}")
    print(f"Output:  {run_folder}")

    mother_folder = Path(paths_cfg['base_dataset_root']) / data_source
    category      = 'Computer'
    local_path    = workspace / data_source / category

    if local_path.exists():
        shutil.rmtree(local_path)
    shutil.copytree(mother_folder / category, local_path)
    print(f"\nDataset copied to local workspace.")

    for kind in ('test', 'ground_truth'):
        src = local_path / kind / 'anomaly'
        dst = local_path / kind / 'bad'
        if src.exists() and not dst.exists():
            src.rename(dst)

    aug_pipeline = build_augmentation_pipeline(aug_cfg)
    train_dir    = local_path / 'train' / 'good'
    if aug_cfg.get('enabled', False):
        n_orig    = len(_collect_images(train_dir))
        print(f"Augmentation | {n_orig} originals x {aug_cfg['images_per_original']}")
        aug_files = inject_augmented_images(
            train_good_dir      = train_dir,
            images_per_original = aug_cfg['images_per_original'],
            pipeline            = aug_pipeline,
            aug_seed            = aug_cfg['seed'],
        )
        print(f"{len(aug_files)} images added — Train-Set: {n_orig + len(aug_files)}")

    # Also set num_workers to 0 for Windows
    cfg['training']['num_workers'] = 0

    print(f"\n{'='*60}")
    print(f"Starting benchmark — {len(models)} model(s), {len(seeds)} seed(s)")
    print(f"{'='*60}")

    runner = BenchmarkRunner(
        data_source = data_source,
        category    = category,
        base_path   = workspace,
        output_path = output_dir,
        session_id  = session_id,
        cfg         = cfg,
    )

    t0 = time.time()
    raw_df, agg_df, lb_df, pw_df, pred_df = runner.run_benchmark(
        model_list     = models,
        seeds          = seeds,
        checkpoint_dir = run_folder / 'checkpoints',
    )
    elapsed = time.time() - t0
    print(f"\n⏱ Total time: {elapsed/60:.1f} minutes")

    if not raw_df.empty:
        raw_df.to_csv(run_folder / 'ALL_raw_seeds.csv',        index=False)
        agg_df.to_csv(run_folder / 'ALL_aggregated_stats.csv', index=False)
        lb_df.to_csv( run_folder / 'ALL_leaderboard.csv',      index=False)
        print(f"\n✅ Results saved to: {run_folder}")
        print(f"\nLEADERBOARD:")
        print(lb_df[['Model', 'Image AUROC']].to_string(index=False))

    torch.cuda.empty_cache()
    gc.collect()
    shutil.rmtree(local_path, ignore_errors=True)
    print("\n🧹 Cleanup done!")