"""Result Reporting: CSV-Export, Leaderboard, DualImageSaver."""

import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from .utils import ensure_dir


# ---------------------------------------------------------------------------
# DualImageSaver Callback
# ---------------------------------------------------------------------------

class DualImageSaver(pl.Callback):
    """
    Saves 5 visualization variants per test image during engine.test().

    All anomaly maps are buffered until on_test_epoch_end so that global
    min/max normalization can be applied across the entire test set, matching
    anomalib's own visualizer scale (low-anomaly images stay cold, high stay hot).

    Output subdirectories created at init time:
        heatmap_only/{good,bad}/      global-normalized colormap, no mask overlay
        heatmap_mask/{good,bad}/      global-normalized colormap + GT-mask contour
        heatmap_local/{good,bad}/     per-image normalized (full dynamic range always)
        heatmap_compare/{good,bad}/   side-by-side global vs local + caption strip
        worker_inspection/{good,bad}/ original + red overlay + bbox + PASS/DEFECT banner

    Images are sorted into good/ and bad/ subfolders by ground-truth label.

    Args:
        save_dir:      Root directory in which all subdirectories are created.
        cfg:           ``visualization.custom`` section from settings.yaml.
        img_threshold: Normalized decision threshold in [0, 1]. Pre-computed by
                       BenchmarkRunner as ``1.0 - sensitivity``. Defaults to 0.5.
    """

    def __init__(self, save_dir: Path, cfg: dict, img_threshold: float = 0.5):
        super().__init__()
        self.save_dir      = Path(save_dir)
        self.cfg           = cfg
        self.img_threshold = img_threshold
        self._buffer: List[Dict] = []
        for sub in (
            "heatmap_only/good",       "heatmap_only/bad",
            "heatmap_mask/good",       "heatmap_mask/bad",
            "heatmap_local/good",      "heatmap_local/bad",
            "heatmap_compare/good",    "heatmap_compare/bad",
            "worker_inspection/good",  "worker_inspection/bad",
        ):
            (self.save_dir / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Static post-processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _postproc_amap(amap: np.ndarray, pp_cfg: dict) -> np.ndarray:
        """Gaussian blur on raw anomaly map to smooth blocky patch artifacts."""
        if not pp_cfg.get("enabled", True):
            return amap
        out   = amap.astype(np.float32)
        sigma = pp_cfg.get("gaussian_sigma", 2.0)
        if sigma > 0:
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma)
        return out

    @staticmethod
    def _postproc_mask(mask: np.ndarray, pp_cfg: dict) -> np.ndarray:
        """Morphological closing + small connected-component removal on binary mask."""
        if not pp_cfg.get("enabled", True):
            return mask
        out = mask.copy()
        k   = pp_cfg.get("closing_kernel", 5)
        if k > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            out    = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
        min_area = pp_cfg.get("min_defect_area", 10)
        if min_area > 0 and out.max() > 0:
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out)
            clean = np.zeros_like(out)
            for i in range(1, n_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    clean[labels == i] = 1
            out = clean
        return out

    # ------------------------------------------------------------------
    # Worker inspection view
    # ------------------------------------------------------------------

    def _make_worker_view(
        self, r: dict, norm_min: Optional[float], norm_max: Optional[float]
    ) -> np.ndarray:
        """
        Original photo + semi-transparent red anomaly overlay + bounding box
        around the largest defect region + PASS/DEFECT banner + score bar.
        Uses per-image p2-p98 clip so single outlier pixels do not dominate scale.
        """
        amap   = r["amap"]
        h, w   = amap.shape[:2]
        pp_cfg = self.cfg.get("postproc", {})

        orig = cv2.imread(r["fpath"])
        if orig is None:
            orig = np.full((h, w, 3), 80, dtype=np.uint8)
        else:
            orig = cv2.resize(orig, (w, h))

        # Per-image p2-p98 normalized map for overlay
        p_lo       = float(np.percentile(amap, 2))
        p_hi       = float(np.percentile(amap, 98))
        denom      = p_hi - p_lo if p_hi > p_lo else 1.0
        amap_norm  = np.clip((amap - p_lo) / denom, 0.0, 1.0)

        # Normalize score and pixel map using PostProcessor bounds if available
        if norm_min is not None and norm_max is not None and norm_max > norm_min:
            score_norm = float(np.clip((r["score"] - norm_min) / (norm_max - norm_min), 0.0, 1.0))
            pixel_norm = np.clip((amap - norm_min) / (norm_max - norm_min), 0.0, 1.0)
        else:
            score_norm = float(np.clip(r["score"], 0.0, 1.0))
            pixel_norm = amap_norm

        defect_mask = (pixel_norm >= self.img_threshold).astype(np.uint8)
        defect_mask = self._postproc_mask(defect_mask, pp_cfg)

        # Red overlay on defect pixels
        red_layer          = np.zeros_like(orig)
        red_layer[:, :, 2] = (amap_norm * 255).astype(np.uint8)
        overlay = cv2.addWeighted(orig, 0.55, red_layer, 0.45, 0)
        result  = orig.copy()
        result[defect_mask == 1] = overlay[defect_mask == 1]

        # Bounding box around largest defect contour
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 4:
                x, y, bw, bh = cv2.boundingRect(largest)
                cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

        # PASS / DEFECT banner
        banner_h  = max(20, h // 12)
        is_defect = r["outcome"] in ("TP", "FP") or score_norm >= self.img_threshold
        col       = (0, 0, 200) if is_defect else (0, 160, 0)
        banner    = np.full((banner_h, w, 3), col, dtype=np.uint8)
        label_str = (
            f"{'DEFECT' if is_defect else 'PASS'}  {r['outcome']}  "
            f"s:{score_norm:.2f} t:{self.img_threshold:.2f}"
        )
        cv2.putText(banner, label_str, (4, banner_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Score bar
        bar_h  = max(10, h // 20)
        bar_bg = np.full((bar_h, w, 3), 50, dtype=np.uint8)
        bar_bg[:, : int(score_norm * w)] = col
        cv2.line(bar_bg,
                 (int(self.img_threshold * w), 0),
                 (int(self.img_threshold * w), bar_h),
                 (255, 255, 255), 1)

        return np.concatenate([banner, result, bar_bg], axis=0)

    # ------------------------------------------------------------------
    # Lightning callbacks
    # ------------------------------------------------------------------

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if outputs is None:
            return
        try:
            amaps  = outputs.anomaly_map
            masks  = outputs.gt_mask  if hasattr(outputs, "gt_mask")  else None
            scores = outputs.pred_score
            paths  = outputs.image_path
            labels = outputs.gt_label if hasattr(outputs, "gt_label") else None

            batch_masks  = [None] * len(paths) if masks  is None else masks
            batch_labels = [None] * len(paths) if labels is None else labels

            for amap, mask, score, label, path in zip(
                amaps, batch_masks, scores, batch_labels, paths
            ):
                gt = int(label.cpu()) if label is not None else -1
                self._buffer.append({
                    "amap":      amap.squeeze().cpu().float().numpy(),
                    "mask":      mask.squeeze().cpu().numpy() if mask is not None else None,
                    "score":     float(score.cpu()),
                    "fname":     Path(str(path)).stem,
                    "fpath":     str(path),
                    "subfolder": "bad" if gt == 1 else "good",
                    "gt":        gt,
                    "outcome":   "?",
                })
        except Exception:
            pass

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if not self._buffer:
            return
        try:
            # Read PostProcessor normalization bounds
            norm_min = norm_max = None
            try:
                pp       = pl_module.post_processor
                norm_min = float(pp.image_min.item())
                norm_max = float(pp.image_max.item())
            except Exception:
                pass

            # Recompute TP/TN/FP/FN outcomes now that norm bounds are known.
            # on_test_batch_end runs before PostProcessor sets pred_label,
            # so outcome was left as "?" there.
            for r in self._buffer:
                raw = r["score"]
                if norm_min is not None and norm_max is not None and norm_max > norm_min:
                    ns   = max(0.0, min(1.0, (raw - norm_min) / (norm_max - norm_min)))
                    pred = 1 if ns >= self.img_threshold else 0
                else:
                    pred = -1
                gt = r["gt"]
                if gt != -1 and pred != -1:
                    if   gt == 1 and pred == 1: r["outcome"] = "TP"
                    elif gt == 0 and pred == 0: r["outcome"] = "TN"
                    elif gt == 0 and pred == 1: r["outcome"] = "FP"
                    else:                       r["outcome"] = "FN"

            # Gaussian blur before computing global stats
            pp_cfg = self.cfg.get("postproc", {})
            for r in self._buffer:
                r["amap"] = self._postproc_amap(r["amap"], pp_cfg)

            # Global min/max across entire test set for consistent colour scale
            all_vals       = np.concatenate([r["amap"].ravel() for r in self._buffer])
            g_min, g_max   = float(all_vals.min()), float(all_vals.max())
            g_denom        = g_max - g_min if g_max > g_min else 1.0

            colormap   = self.cfg.get("colormap", cv2.COLORMAP_VIRIDIS)
            font_scale = self.cfg.get("font_scale", 0.30)
            font_thick = self.cfg.get("font_thickness", 1)
            mask_color = tuple(self.cfg.get("mask_color", [0, 0, 255]))
            mask_thick = self.cfg.get("mask_thickness", 2)

            for r in self._buffer:
                amap_u8 = (((r["amap"] - g_min) / g_denom).clip(0, 1) * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(amap_u8, colormap)

                raw  = r["score"]
                if norm_min is not None and norm_max is not None and norm_max > norm_min:
                    norm_score = max(0.0, min(1.0, (raw - norm_min) / (norm_max - norm_min)))
                    has_norm   = True
                else:
                    norm_score = None
                    has_norm   = False

                standalone_label = (
                    f"[G] {r['outcome']} s:{norm_score:.2f} t:{self.img_threshold:.2f} RAW:{raw:.1f}"
                    if has_norm else f"[G] {r['outcome']} RAW:{raw:.1f}"
                )
                global_compare_label = f"[G] dataset RAW:{g_min:.1f}-{g_max:.1f}"
                cv2.putText(heatmap, standalone_label, (4, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), font_thick, cv2.LINE_AA)

                sub     = r["subfolder"]
                mask_np = r["mask"]

                def _with_contour(base, _m=mask_np):
                    out = base.copy()
                    if _m is not None:
                        mu8 = (_m * 255).astype(np.uint8)
                        if mu8.max() > 0:
                            cnts, _ = cv2.findContours(
                                mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            cv2.drawContours(out, cnts, -1, mask_color, mask_thick)
                    return out

                # Per-image (local) normalization
                l_min, l_max  = float(r["amap"].min()), float(r["amap"].max())
                l_denom       = l_max - l_min if l_max > l_min else 1.0
                amap_local_u8 = (((r["amap"] - l_min) / l_denom).clip(0, 1) * 255).astype(np.uint8)
                heatmap_local = cv2.applyColorMap(amap_local_u8, colormap)
                cv2.putText(
                    heatmap_local,
                    f"[L] {r['outcome']} RAW:{l_min:.1f}-{l_max:.1f}",
                    (4, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thick, cv2.LINE_AA,
                )

                # ── Version 1: global heatmap only ──────────────────────────
                cv2.imwrite(
                    str(self.save_dir / "heatmap_only" / sub / f"{r['fname']}.png"),
                    heatmap,
                )

                # ── Version 2: global heatmap + GT mask contour ─────────────
                cv2.imwrite(
                    str(self.save_dir / "heatmap_mask" / sub / f"{r['fname']}.png"),
                    _with_contour(heatmap),
                )

                # ── Version 3: per-image normalized + GT mask contour ────────
                cv2.imwrite(
                    str(self.save_dir / "heatmap_local" / sub / f"{r['fname']}.png"),
                    _with_contour(heatmap_local),
                )

                # ── Version 4: side-by-side comparison panel ─────────────────
                heatmap_g_cmp = cv2.applyColorMap(amap_u8, colormap)
                cv2.putText(heatmap_g_cmp, global_compare_label, (4, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), font_thick, cv2.LINE_AA)
                heatmap_l_cmp = cv2.applyColorMap(amap_local_u8, colormap)
                cv2.putText(
                    heatmap_l_cmp,
                    f"[L] this-image RAW:{l_min:.1f}-{l_max:.1f}",
                    (4, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thick, cv2.LINE_AA,
                )
                panel_g      = _with_contour(heatmap_g_cmp)
                panel_l      = _with_contour(heatmap_l_cmp)
                divider      = np.full((panel_g.shape[0], 2, 3), 255, dtype=np.uint8)
                side_by_side = np.concatenate([panel_g, divider, panel_l], axis=1)

                # Caption strip
                total_px = r["amap"].size
                if has_norm:
                    pix_norm_cap = np.clip(
                        (r["amap"] - norm_min) / (norm_max - norm_min), 0.0, 1.0
                    )
                else:
                    pix_norm_cap = np.clip(r["amap"], 0.0, 1.0)
                defect_px    = int((pix_norm_cap >= self.img_threshold).sum())
                coverage_pct = defect_px / total_px * 100
                n_reg, _, stats_cap, _ = cv2.connectedComponentsWithStats(
                    (pix_norm_cap >= self.img_threshold).astype(np.uint8)
                )
                min_area = pp_cfg.get("min_defect_area", 0)
                n_reg    = sum(
                    1 for i in range(1, n_reg)
                    if stats_cap[i, cv2.CC_STAT_AREA] >= max(min_area, 1)
                )
                sigma   = pp_cfg.get("gaussian_sigma", 0)
                shared  = (
                    f"{r['outcome']}  s:{norm_score:.2f}  t:{self.img_threshold:.2f}"
                    if has_norm else f"{r['outcome']}"
                )
                cap_txt = (
                    f"{shared} | def:{coverage_pct:.1f}% "
                    f"reg_count:{n_reg} blur:σ={sigma}"
                )
                cap_h   = max(16, panel_g.shape[0] // 14)
                cap_w   = side_by_side.shape[1]
                caption = np.full((cap_h, cap_w, 3), 30, dtype=np.uint8)
                cv2.putText(caption, cap_txt, (4, cap_h - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.imwrite(
                    str(self.save_dir / "heatmap_compare" / sub / f"{r['fname']}.png"),
                    np.concatenate([side_by_side, caption], axis=0),
                )

                # ── Version 5: worker inspection view ────────────────────────
                cv2.imwrite(
                    str(self.save_dir / "worker_inspection" / sub / f"{r['fname']}.png"),
                    self._make_worker_view(r, norm_min, norm_max),
                )

        except Exception as e:
            print(f"  DualImageSaver visualization error: {e}")
            traceback.print_exc()
        finally:
            self._buffer.clear()


# ---------------------------------------------------------------------------
# CSV / Leaderboard Export
# ---------------------------------------------------------------------------

def save_results_csv(results_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)


def save_leaderboard(
    results_df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
) -> pd.DataFrame:
    lb = (
        results_df.groupby("Model")[metric]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    lb.insert(0, "Rank", range(1, len(lb) + 1))
    save_results_csv(lb, output_path)
    return lb


def save_run_metadata(metadata: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
