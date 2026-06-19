"""Trainings Callbacks: TrainLossAlias and PredictionCollector."""

from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch


class TrainLossAlias(pl.Callback):
    """Aliases 'loss' as 'train_loss' to enable compatibility with EarlyStopping."""
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        if "train_loss" not in metrics and "loss" in metrics:
            pl_module.log("train_loss", metrics["loss"], prog_bar=False)


class PredictionCollector(pl.Callback):
    """Collects image-level predictions (Score, Label, Path) during the test phase."""

    def __init__(self):
        """Initialize the prediction collector with empty storage."""
        super().__init__()
        self.rows: List[Dict]         = []
        self.norm_min: Optional[float] = None
        self.norm_max: Optional[float] = None

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Extract PostProcessor normalization bounds at end of test epoch."""
        try:
            pp = pl_module.post_processor
            self.norm_min = float(pp.image_min.item())
            self.norm_max = float(pp.image_max.item())
        except Exception:
            pass

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if outputs is None:
            return
        try:
            scores      = outputs.pred_score.cpu().numpy() if outputs.pred_score is not None else []
            labels      = outputs.gt_label.cpu().numpy()   if outputs.gt_label  is not None else []
            pred_labels = (
                outputs.pred_label.cpu().numpy()
                if outputs.pred_label is not None
                else [None] * len(scores)
            )
            paths = outputs.image_path if hasattr(outputs, "image_path") else [""] * len(scores)
            for path, score, label, pred_label in zip(paths, scores, labels, pred_labels):
                self.rows.append({
                    "image_path": str(path),
                    "pred_score": float(np.asarray(score).item()),
                    "gt_label":   int(np.asarray(label).item()),
                    "pred_label": int(np.asarray(pred_label).item()) if pred_label is not None else None,
                })
        except Exception:
            pass
