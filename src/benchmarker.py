"""Orchestrates the training and evaluation of all models and seeds."""

import math
import shutil
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import pandas as pd
import torch
from anomalib.data import Folder, MVTecAD
from anomalib.engine import Engine
from anomalib.metrics import AUPR, AUPRO, AUROC, F1Score, Evaluator
from anomalib.post_processing import PostProcessor
from anomalib.visualization import ImageVisualizer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from .augmentation import patch_normalize_in_transform
from .data_loader import (
    copy_images_to_pseudo_val,
    generate_black_masks,
    generate_pseudo_anomaly_val_set,
    DatasetFactory,
)
from .engine import PredictionCollector, TrainLossAlias
from .models import IMAGE_ONLY_MODELS, MODEL_MAPPING
from .reporter import DualImageSaver
from .statengine import SafeAUPIMO, StatEngine
from .utils import _collect_images, _count_images, clear_memory


class BenchmarkRunner:
    """
    Orchestrates the dataset benchmark across all specified models and seeds.
    All execution parameters are controlled via the configuration dictionary.
    """

    GRADIENT_MODELS = {"Fastflow", "Cflow", "Csflow", "Uflow", "UniNet"}

    def __init__(
        self,
        data_source: str,
        category: str,
        base_path: Path,
        output_path: Path,
        session_id: str,
        cfg: dict,
        norm_stats: Optional[Tuple[List[float], List[float]]] = None,
    ):
        self.data_source  = data_source
        self.category     = category
        self.base_path    = Path(base_path)
        self.output_path  = Path(output_path)
        self.session_id   = session_id
        self.cfg          = cfg
        self.norm_stats   = norm_stats
        self.device       = "gpu" if torch.cuda.is_available() else "cpu"
        self.root         = self._resolve_root()
        print(f"Dataset Root: {self.root}")

    # ------------------------------------------------------------------
    # Resolve absolute project root directory    
    # ------------------------------------------------------------------

    def _resolve_root(self) -> Path:
        roots = {
            "MVTecAD":           self.base_path / "MVTecAD"    / self.category,
            "MVTec2":            self.base_path / "mvtec_ad_2" / self.category,
            "VisA":              self.base_path / "VisA"       / self.category,
            self.data_source:    self.base_path / self.data_source / self.category,
        }
        root = roots.get(self.data_source,
                         self.base_path / self.data_source / self.category)
        # Validate existence for custom datasets
        if self.data_source not in ("MVTecAD", "MVTec2", "VisA"):
            mother = root.parent
            if not mother.exists():
                raise FileNotFoundError(f"Mother folder not found at {mother}")
            if not root.exists():
                available = [d.name for d in mother.iterdir() if d.is_dir()]
                raise FileNotFoundError(
                    f"Dataset '{self.category}' not found. Available: {available}"
                )
        return root

    # ------------------------------------------------------------------
    # DataModule-Builder
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_eval_batch_size(total: int, requested: int) -> int:
        if requested == 1:
            return 1
        bs = max(requested, 2)
        if total <= 1:
            return bs
        # Try to find a batch size close to requested that avoids edge cases
        for delta in range(0, bs):
            candidate = bs - delta
            if candidate < 2:
                candidate = bs + (delta + 1)
            # Check if this batch size creates a valid split (no remainder of 1)
            if total % candidate != 1:
                if candidate != requested:
                    print(f"  Auto-adjusted eval_batch_size {requested} -> {candidate} "
                          f"(total test images: {total})")
                return candidate
        # Fallback: use total as batch size to ensure all images fit
        print(f"  Auto-adjusted eval_batch_size {requested} -> {total}")
        return total

    def _common_dm_kwargs(self, train_bs: int, eval_bs: int) -> dict:
        cfg = self.cfg
        kw  = dict(
            train_batch_size=train_bs,
            eval_batch_size=eval_bs,
            num_workers=cfg["training"]["num_workers"],
        )
        # Optionally add Perlin noise augmentation if enabled in config
        if cfg.get("augmentation", {}).get("perlin", {}).get("enabled", False):
            from anomalib.data.utils.generators import PerlinAnomalyGenerator
            kw["augmentations"] = PerlinAnomalyGenerator(
                **{k: v for k, v in cfg["augmentation"]["perlin"].items() if k != "enabled"}
            )
        return kw

    def get_datamodule(self, train_bs: int, eval_bs: int):
      val_pseudo_dir = self.root / "pseudo_val"
      if val_pseudo_dir.exists():
          shutil.rmtree(val_pseudo_dir)

      ds_cfg   = DatasetFactory.get(self.data_source)
      test_dir = self.root / ("test_public" if self.data_source == "MVTec2" else "test")
      total_test = _count_images(test_dir)
      safe_bs    = self._safe_eval_batch_size(total_test, eval_bs)
      common     = self._common_dm_kwargs(train_bs, safe_bs)

      if self.data_source == "MVTecAD":
          return MVTecAD(root=str(self.root.parent), category=self.category, **common)

      train_dir = self.root / ds_cfg["normal_dir"]
      n_orig    = len(_collect_images(train_dir))
      n_train   = _count_images(train_dir)
      print(f"\n--- DATASET LOAD SUMMARY: {self.category.upper()} ---")
      print(f"  Train (Good):   {n_train}  ({n_orig} original + {n_train - n_orig} augmented)")
      print(f"  Test (Good):    {_count_images(self.root / ds_cfg['test_good'])}")
      print(f"  Test (Bad):     {_count_images(self.root / ds_cfg['test_bad'])}")
      print(f"  Masks:          {_count_images(self.root / ds_cfg['mask_dir'])}")

      _abn_dir = None if _count_images(self.root / ds_cfg["test_bad"]) < 1 else ds_cfg["test_bad"]
      val_from_good = self.cfg["threshold"]["method"] == "HARD_THRESHOLD"

      if val_from_good:
          return Folder(
              name=self.category, root=str(self.root),
              normal_dir=ds_cfg["normal_dir"],
              abnormal_dir=_abn_dir,
              normal_test_dir=ds_cfg["test_good"],
              mask_dir=ds_cfg["mask_dir"],
              val_split_mode="from_train", val_split_ratio=0.2,
              **common,
          )
      return Folder(
          name=self.category, root=str(self.root),
          normal_dir=ds_cfg["normal_dir"],
          abnormal_dir=_abn_dir,
          normal_test_dir=ds_cfg["test_good"],
          mask_dir=ds_cfg["mask_dir"],
          val_split_mode="from_test", val_split_ratio=0.0,
          **common,
      )

    def get_train_val_datamodule(self, train_bs: int, eval_bs: int, seed: int):
        """built a DataModule with a Pseudo-Anomalie-Val-Set (for Threshold calibration)."""
        val_pseudo_dir = self.root / "pseudo_val"
        if val_pseudo_dir.exists():
            shutil.rmtree(val_pseudo_dir)

        sample_save_dir = (
            self.output_path / f"{self.data_source}_{self.session_id}" / "pseudo_val_samples" / self.category
        )
        train_good_dir  = self.root / "train" / "good"
        threshold_method = self.cfg["threshold"]["method"]
        print(f"  Threshold method: {threshold_method}")

        if threshold_method == "MANUAL_ANOMALIES":
            mc = self.cfg["threshold"]["manual_anomalies"]
            n_good, n_bad = copy_images_to_pseudo_val(
                source_dir      = Path(self.cfg["paths"]["golden_unit_defects"]),
                train_good_dir  = train_good_dir,
                val_pseudo_dir  = val_pseudo_dir,
                n_images        = mc.get("n_images"),
                n_samples_save  = mc.get("n_samples_save", 10),
                sample_save_dir = sample_save_dir,
            )
        elif threshold_method == "GENERATIVE_AI":
            gc_cfg = self.cfg["threshold"]["generative_ai"]
            n_good, n_bad = copy_images_to_pseudo_val(
                source_dir      = Path(self.cfg["paths"]["generated_anomalies"]),
                train_good_dir  = train_good_dir,
                val_pseudo_dir  = val_pseudo_dir,
                n_images        = gc_cfg.get("n_images"),
                n_samples_save  = gc_cfg.get("n_samples_save", 10),
                sample_save_dir = sample_save_dir,
            )
        else:
            n_good, n_bad = generate_pseudo_anomaly_val_set(
                train_good_dir  = train_good_dir,
                val_pseudo_dir  = val_pseudo_dir,
                config          = self.cfg["threshold"]["noise_injection"],
                seed            = self.cfg["augmentation"]["seed"],
                sample_save_dir = sample_save_dir,
            )

        print(f"  Pseudo-anomaly val created: {n_good} normal | {n_bad} pseudo-anomalous")
        total_pseudo = _count_images(val_pseudo_dir)
        safe_bs      = self._safe_eval_batch_size(total_pseudo, eval_bs)
        common       = self._common_dm_kwargs(train_bs, safe_bs)

        return Folder(
            name            = self.category,
            root            = str(self.root),
            normal_dir      = "train/good",
            abnormal_dir    = "pseudo_val/bad",
            normal_test_dir = "pseudo_val/good",
            val_split_mode  = "from_test",
            val_split_ratio = 0.8,
            seed            = seed,
            **common,
        )

    def get_test_only_datamodule(self, train_bs: int, eval_bs: int):
      ds_cfg   = DatasetFactory.get(self.data_source)
      test_dir = self.root / ("test_public" if self.data_source == "MVTec2" else "test")
      total_test = _count_images(test_dir)
      safe_bs    = self._safe_eval_batch_size(total_test, eval_bs)

      if self.data_source == "MVTecAD":
          return MVTecAD(
              root=str(self.root.parent), category=self.category,
              train_batch_size=train_bs, eval_batch_size=safe_bs,
              num_workers=self.cfg["training"]["num_workers"]
          )

      _abn_dir = None if _count_images(self.root / ds_cfg["test_bad"]) < 1 else ds_cfg["test_bad"]
      return Folder(
          name=self.category, root=str(self.root),
          normal_dir=ds_cfg["normal_dir"],
          abnormal_dir=_abn_dir,
          normal_test_dir=ds_cfg["test_good"],
          mask_dir=ds_cfg["mask_dir"],
          val_split_mode="none",
          train_batch_size=train_bs, eval_batch_size=safe_bs,
          num_workers=self.cfg["training"]["num_workers"],
      )
    def get_evaluator(self, model_name: str, disable_aupimo: bool = False) -> Evaluator:
        if _count_images(self.root / "test" / "bad") < 1:
            return Evaluator(test_metrics=[], compute_on_cpu=False)
        
        # Image-level metrics: score-based classification
        metrics = [
            AUROC(fields=["pred_score", "gt_label"],  prefix="image_"),
            F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
            AUPR(fields=["pred_score", "gt_label"],   prefix="image_"),
        ]
        # Pixel-level metrics: for models that output anomaly maps/masks
        if model_name not in IMAGE_ONLY_MODELS:
            metrics += [
                AUROC(fields=["anomaly_map", "gt_mask"],   prefix="pixel_"),
                F1Score(fields=["pred_mask", "gt_mask"],   prefix="pixel_"),
                AUPR(fields=["anomaly_map", "gt_mask"],    prefix="pixel_"),
                AUPRO(fields=["anomaly_map", "gt_mask"],   prefix="pixel_"),
            ]
            # Add AUPIMO (per-region metric) unless disabled
            if not disable_aupimo:
                metrics.append(SafeAUPIMO(fields=["anomaly_map", "gt_mask"], prefix="pixel_"))
        return Evaluator(test_metrics=metrics, compute_on_cpu=False)

    # ------------------------------------------------------------------
    # Engine-Builder
    # ------------------------------------------------------------------

    def _build_engine(
        self,
        run_path: Path,
        target_epochs: int,
        logger: CSVLogger,
        extra_callbacks: Optional[List] = None,
        model_name: str = "",
        skip_val: bool = False, 
    ) -> Engine:
        callbacks = [TQDMProgressBar(refresh_rate=0), TrainLossAlias()]
        if model_name in self.GRADIENT_MODELS:
            callbacks.append(EarlyStopping(
                monitor="train_loss", patience=30, min_delta=1e-4, mode="min",
            ))
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        return Engine(
            default_root_dir=str(run_path),
            max_epochs=target_epochs,
            accelerator=self.device, devices=1,
            enable_progress_bar=False,
            callbacks=callbacks,
            logger=logger,
            limit_val_batches=0 if skip_val else 1.0,
        )

    # ------------------------------------------------------------------
    # Single-Seed Run
    # ------------------------------------------------------------------

    @staticmethod
    def _get_met(res: Dict, level: str, metric: str) -> float:
        target = f"{level}_{metric}".lower()
        for k, v in res.items():
            if k.lower().endswith(target):
                return float(v)
        return float("nan")

    def _run_single_seed(
        self,
        model_name: str,
        model_args: dict,
        train_bs: int,
        eval_bs: int,
        target_epochs: int,
        seed: int,
        run_path: Path,
        dst: Path,
    ) -> Optional[Tuple[Dict, List[Dict], Optional[Path]]]:
        threshold_method = self.cfg["threshold"]["method"]
        models_to_save   = set(self.cfg["run"].get("models_to_save", []))

        try:
            if threshold_method not in ("HARD_THRESHOLD", "SAME_AS_TEST") \
                    and self.data_source not in ("MVTecAD", "MVTec2", "VisA"):
                train_val_dm = self.get_train_val_datamodule(train_bs, eval_bs, seed)
                test_dm      = self.get_test_only_datamodule(train_bs, eval_bs)
            else:
                train_val_dm = self.get_datamodule(train_bs, eval_bs)
                test_dm      = train_val_dm

            model_class = MODEL_MAPPING[model_name]

            # Initialize pre-processor and apply normalization patches if required
            try:
                resize = self.cfg["training"]["resize_img"]
                custom_pre_processor = model_class.configure_pre_processor(
                    image_size=(resize, resize)
                )
            except (AttributeError, TypeError):
                custom_pre_processor = None

            if custom_pre_processor is not None and self.norm_stats is not None:
                mean, std = self.norm_stats
                try:
                    patched = patch_normalize_in_transform(custom_pre_processor, mean, std)
                    if patched:
                        print(f"  Norm patched -> mean={[round(m,4) for m in mean]} "
                              f"std={[round(s,4) for s in std]}")
                    else:
                        print("  No Normalize layer found — ImageNet defaults kept.")
                except Exception as e:
                    print(f"  Norm patch failed: {e} — ImageNet defaults kept.")

            # PostProcessor / Sensitivity
            threshold_cfg = self.cfg["threshold"]
            _method_cfg   = threshold_cfg.get(threshold_method.lower(), {})
            _sens         = (
                threshold_cfg["hard"]["image_sensitivity"]
                if threshold_method == "HARD_THRESHOLD"
                else _method_cfg.get("sensitivity_override")
            )

            if _sens is not None:
                pp = PostProcessor(
                    enable_normalization=True,
                    image_sensitivity=_sens,
                    pixel_sensitivity=_sens,
                )
                print(f"  PostProcessor: sensitivity={_sens} -> threshold={1.0 - _sens:.2f}")
                model_kwargs = {"evaluator": self.get_evaluator(model_name),
                                "post_processor": pp, **model_args}
            else:
                print("  PostProcessor: F1AdaptiveThreshold")
                model_kwargs = {"evaluator": self.get_evaluator(model_name), **model_args}

            if custom_pre_processor is not None:
                model_kwargs["pre_processor"] = custom_pre_processor

            viz_cfg = self.cfg.get("visualization", {})
            if viz_cfg.get("heatmap", {}).get("enabled", False):
                model_kwargs["visualizer"] = ImageVisualizer(
                    fields_config=viz_cfg["heatmap"].get("fields_config", {})
                )

            model     = model_class(**model_kwargs)
            logger    = CSVLogger(save_dir=str(run_path), name="logs")
            collector = PredictionCollector()
            extra_cbs = [collector]
            if viz_cfg.get("custom", {}).get("enabled", False):
                img_threshold = (1.0 - _sens) if _sens is not None else 0.5
                extra_cbs.append(DualImageSaver(
                    save_dir=run_path / "custom_images",
                    cfg=viz_cfg["custom"],
                    img_threshold=img_threshold,
                ))

            engine = self._build_engine(
                run_path, target_epochs, logger,
                extra_callbacks=extra_cbs, model_name=model_name,
                skip_val=(model_name == "WinClip"),
            )
            t0 = time.time()

            save_ckpt = model_name in models_to_save
            ctx       = {} if save_ckpt else {"patch": patch("lightning.pytorch.trainer.trainer.Trainer.save_checkpoint")}

            def _do_run():
              if model_name != "WinClip":
                  print(f"  Training {model_name} (seed {seed})...")
                  engine.fit(model=model, datamodule=train_val_dm)
              train_time = time.time() - t0
              print(f"  Training done in {train_time:.1f}s — running evaluation...")
              if model_name == "WinClip":
                  dm = self.get_test_only_datamodule(train_bs, eval_bs)
              else:
                  dm = test_dm
              results = engine.test(model=model, datamodule=dm)
              return train_time, results

            if save_ckpt:
                train_time, test_results = _do_run()
            else:
                with patch("lightning.pytorch.trainer.trainer.Trainer.save_checkpoint"):
                    train_time, test_results = _do_run()

            if not test_results:
                return None

            r = test_results[0]
            _tp = _fp = _tn = _fn = 0
            for _row in collector.rows:
                _pl, _gl = _row.get("pred_label"), _row.get("gt_label")
                if _pl is None or _gl is None:
                    continue
                if   _gl == 1 and _pl == 1: _tp += 1
                elif _gl == 0 and _pl == 1: _fp += 1
                elif _gl == 0 and _pl == 0: _tn += 1
                elif _gl == 1 and _pl == 0: _fn += 1

            metrics_out = {
                "Train Time":   train_time,
                "Image AUROC":  self._get_met(r, "image", "auroc"),
                "Pixel AUROC":  self._get_met(r, "pixel", "auroc"),
                "Image F1":     self._get_met(r, "image", "f1score"),
                "Pixel F1":     self._get_met(r, "pixel", "f1score"),
                "Image AUPRC":  self._get_met(r, "image", "aupr"),
                "Pixel AUPRC":  self._get_met(r, "pixel", "aupr"),
                "Pixel AUPRO":  self._get_met(r, "pixel", "aupro"),
                "Pixel AUPIMO": self._get_met(r, "pixel", "safeaupimo"),
            }

            pred_rows = [
                {
                    "model":      model_name,
                    "seed":       seed,
                    "image_path": row["image_path"],
                    "pred_score": row["pred_score"],
                    "gt_label":   row["gt_label"],
                }
                for row in collector.rows
            ]

            staged_ckpt = None
            if save_ckpt:
                ckpt_files = sorted(run_path.rglob("*.ckpt"))
                if ckpt_files:
                    holding_dir = Path(self.cfg["paths"].get("temp_checkpoints", "/content/temp_checkpoints"))
                    holding_dir.mkdir(parents=True, exist_ok=True)
                    staged_ckpt = holding_dir / f"{model_name}_{self.category}_seed{seed}.ckpt"
                    shutil.copy2(ckpt_files[0], staged_ckpt)

            return metrics_out, pred_rows, staged_ckpt

        except Exception:
            print(f"  FAILED: {model_name} seed {seed}")
            traceback.print_exc()
            try:
                dst.rename(dst.parent / f"{dst.name}_CRASHED")
            except Exception:
                pass
            return None

        finally:
            self._salvage_to_drive(run_path, dst)
            for obj_name in ("engine", "model", "train_val_dm", "test_dm", "logger"):
                try:
                    del locals()[obj_name]
                except Exception:
                    pass
            clear_memory()

    # ------------------------------------------------------------------
    # Main Benchmark Loop
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        model_list: List[str],
        seeds: List[int],
        checkpoint_dir: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        from .models import build_model_config

        primary_metric   = self.cfg["evaluation"]["primary_metric"]
        alpha            = self.cfg["statistics"]["alpha"]
        bonferroni       = self.cfg["statistics"]["bonferroni"]
        models_to_save   = set(self.cfg["run"].get("models_to_save", []))

        print(f"Session: {self.session_id} | Dataset: {self.category} | Seeds: {seeds}")

        raw_rows:    List[Dict]                    = []
        model_seed_metrics: Dict[str, List[Dict]]  = {}
        all_pred_rows: List[Dict]                  = []

        for model_name in model_list:
            if model_name not in MODEL_MAPPING:
                print(f"  '{model_name}' not in MODEL_MAPPING — skipping.")
                continue

            print(f"\nModel: {model_name}")
            mc        = build_model_config(model_name, self.cfg)
            train_bs  = mc["train_batch_size"]
            eval_bs   = mc["eval_batch_size"]
            epochs    = mc["epochs"]
            model_args = mc["model_args"].copy()

            if model_name == "WinClip":
                model_args["class_name"] = self._resolve_winclip_prompt()

            seed_metrics: List[Dict]           = []
            best_ckpt_info: Optional[Tuple]    = None

            for seed in seeds:
                print(f"  Seed: {seed}")
                seed_everything(seed, workers=True)
                clear_memory()

                temp_results = self.cfg["paths"].get("temp_results", "/content/temp_results")
                run_path = Path(temp_results) / f"{model_name}_{self.category}_seed{seed}_{self.session_id}"
                dst = (
                    self.output_path / f"{self.data_source}_{self.session_id}"
                    / self.category / model_name / f"seed_{seed}"
                )
                if run_path.exists():
                    shutil.rmtree(run_path)

                result = self._run_single_seed(
                    model_name, model_args, train_bs, eval_bs, epochs, seed, run_path, dst
                )
                if result is not None:
                    metrics, pred_rows, staged_ckpt = result
                    metrics["seed"] = seed
                    all_pred_rows.extend(pred_rows)
                    seed_metrics.append(metrics)
                    raw_rows.append({"Model": model_name, **metrics})

                    if checkpoint_dir is not None:
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(raw_rows).to_csv(
                            checkpoint_dir / f"{self.category}_{self.session_id}_raw_seeds_partial.csv",
                            index=False,
                        )
                        if all_pred_rows:
                            pd.DataFrame(all_pred_rows).to_csv(
                                checkpoint_dir / f"{self.category}_{self.session_id}_raw_predictions_partial.csv",
                                index=False,
                            )

                    if staged_ckpt is not None and staged_ckpt.exists():
                        auroc = metrics.get("Image AUROC", float("nan"))
                        if not math.isnan(auroc):
                            if best_ckpt_info is None or auroc > best_ckpt_info[0]:
                                best_ckpt_info = (auroc, staged_ckpt, seed)

            if seed_metrics:
                model_seed_metrics[model_name] = seed_metrics
                model_run_dir = (
                    self.output_path / f"{self.data_source}_{self.session_id}"
                    / self.category / model_name
                )
                model_run_dir.mkdir(parents=True, exist_ok=True)
                model_raw_df = pd.DataFrame([{"Model": model_name, **m} for m in seed_metrics])
                model_raw_df.to_csv(model_run_dir / "raw_seeds.csv", index=False)
                print(f"  raw_seeds.csv saved for {model_name} ({len(seed_metrics)} seed(s))")

                if best_ckpt_info is not None:
                    best_auroc, best_path, best_seed = best_ckpt_info
                    if best_path.exists():
                        ckpt_dst = model_run_dir / f"best_{model_name}_seed{best_seed}_auroc{best_auroc:.4f}.ckpt"
                        shutil.copy2(best_path, ckpt_dst)
                        print(f"  Best checkpoint -> {ckpt_dst.name}  (AUROC={best_auroc:.4f})")

                # Resolve old staged checkpoints 
                holding_dir = Path(self.cfg["paths"].get("temp_checkpoints", "/content/temp_checkpoints"))
                for stale in holding_dir.glob(f"{model_name}_{self.category}_seed*.ckpt"):
                    stale.unlink(missing_ok=True)

        agg_rows = [StatEngine.summarise(n, sm) for n, sm in model_seed_metrics.items()]
        agg_df   = pd.DataFrame(agg_rows) if agg_rows else pd.DataFrame()
        pw_df    = StatEngine.pairwise_tests(
            model_seed_metrics, primary_metric, alpha=alpha, bonferroni=bonferroni
        )
        lb_df    = StatEngine.leaderboard(agg_df, pw_df, primary_metric, alpha=alpha)
        pred_df  = pd.DataFrame(all_pred_rows) if all_pred_rows else pd.DataFrame()

        return pd.DataFrame(raw_rows), agg_df, lb_df, pw_df, pred_df

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _resolve_winclip_prompt(self) -> str:
        if self.cfg["run"].get("use_folder_prompts", False):
            prompt_file = self.root / "prompt.txt"
            if prompt_file.exists():
                prompt = prompt_file.read_text().strip()
                print(f"  Using folder prompt: '{prompt}'")
                return prompt
            print(f"  'prompt.txt' not found in {self.root.name}, using manual fallback.")
        return self.cfg["run"].get("winclip_manual_prompt", "")

    def _salvage_to_drive(self, run_path: Path, dst: Path) -> None:
        if not run_path.exists():
            return
        print("  Saving outputs to Drive...")
        try:
            dst.mkdir(parents=True, exist_ok=True)
            for img_dir in run_path.rglob("images"):
                if img_dir.is_dir() and "checkpoints" not in img_dir.parts:
                    shutil.copytree(img_dir, dst / "images", dirs_exist_ok=True)
                    break
            custom_img_dir = run_path / "custom_images"
            if custom_img_dir.exists():
                shutil.copytree(custom_img_dir, dst / "custom_images", dirs_exist_ok=True)
            for metrics_file in run_path.rglob("metrics.csv"):
                shutil.copy2(metrics_file, dst / "metrics.csv")
                break
        except Exception as e:
            print(f"  Failed to salvage files to Drive: {e}")
        shutil.rmtree(run_path, ignore_errors=True)
