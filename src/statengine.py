"""Statistical Evaluation: Summary, Significance Tests, Leaderboard, Bootstrap."""

import math
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from anomalib.metrics import AUPIMO


class SafeAUPIMO(AUPIMO):
    """AUPIMO wrapper that catches exceptions and returns NaN."""

    def compute(self):
        """Compute AUPIMO metric safely, returning NaN on error."""
        try:
            return super().compute()
        except Exception as e:
            print(f"\n  AUPIMO computation failed: {e}. Returning NaN.")
            return torch.tensor(float("nan"), device=self.device)


class StatEngine:
    """Statistical Evaluation over all Models and Seeds."""

    # ------------------------------------------------------------------
    # Summary per Model
    # ------------------------------------------------------------------

    @classmethod
    def summarise(cls, model_name: str, seed_metrics: List[Dict]) -> Dict:
        row: Dict = {"Model": model_name, "N_seeds": len(seed_metrics)}
        row["Train Time (s) mean"] = round(
            float(np.nanmean([m["Train Time"] for m in seed_metrics])), 2
        )
        skip_keys = {"Train Time", "seed"}
        for key in (k for k in seed_metrics[0] if k not in skip_keys):
            vals = np.array([m[key] for m in seed_metrics], dtype=float)
            row.update(cls._describe(key, vals))
        return row

    @classmethod
    def _describe(cls, key: str, vals: np.ndarray) -> Dict:
        valid   = vals[~np.isnan(vals)]
        n_valid = len(valid)
        
        # Return all-NaN result if no valid values
        if n_valid == 0:
            na = float("nan")
            return {
                f"{key} mean":    na, f"{key} std":     na,
                f"{key} median":  na, f"{key} IQR":     na,
                f"{key} CI95_lo": na, f"{key} CI95_hi": na,
                f"{key} min":     na, f"{key} max":     na,
                f"{key} N_valid": 0,
            }
        
        # Compute basic statistics
        mu  = float(np.mean(valid))
        std = float(np.std(valid, ddof=1)) if n_valid > 1 else 0.0
        med = float(np.median(valid))
        q25, q75 = (
            (float(np.percentile(valid, 25)), float(np.percentile(valid, 75)))
            if n_valid > 1 else (mu, mu)
        )
        
        return {
            f"{key} mean":    round(mu,  4),
            f"{key} std":     round(std, 4),
            f"{key} median":  round(med, 4),
            f"{key} IQR":     round(q75 - q25, 4),
            f"{key} CI95_lo": float("nan"),
            f"{key} CI95_hi": float("nan"),
            f"{key} min":     round(float(np.min(valid)), 4),
            f"{key} max":     round(float(np.max(valid)), 4),
            f"{key} N_valid": n_valid,
        }

    # ------------------------------------------------------------------
    # Pairwise Tests
    # ------------------------------------------------------------------

    @classmethod
    def pairwise_tests(
        cls,
        model_seed_metrics: Dict[str, List[Dict]],
        metric: str,
        alpha: float = 0.05,
        bonferroni: bool = True,
        seed_key: str = "seed",
    ) -> pd.DataFrame:
        models = list(model_seed_metrics.keys())
        if len(models) < 2:
            return pd.DataFrame()
        pairs         = list(combinations(models, 2))
        n_comparisons = len(pairs)
        rows = []
        for (a, b) in pairs:
            # Extract metric values for models a and b, matching by seed
            dict_a = {s[seed_key]: s[metric] for s in model_seed_metrics[a]
                      if metric in s and seed_key in s}
            dict_b = {s[seed_key]: s[metric] for s in model_seed_metrics[b]
                      if metric in s and seed_key in s}
            
            # Find seeds that both models have data for
            common_seeds = sorted(set(dict_a) & set(dict_b))
            if not common_seeds:
                continue
            xa = np.array([dict_a[seed] for seed in common_seeds], dtype=float)
            xb = np.array([dict_b[seed] for seed in common_seeds], dtype=float)
            
            # Perform statistical test
            result = cls._test_pair(xa, xb, alpha=alpha)
            
            # Apply Bonferroni correction if requested
            p_adj  = min(result["p_raw"] * n_comparisons, 1.0) if bonferroni else result["p_raw"]
            
            rows.append({
                "Model_A":     a,
                "Model_B":     b,
                "test":        result["test"],
                "statistic":   round(result["statistic"], 4),
                "p_raw":       round(result["p_raw"],     4),
                "p_adj":       round(p_adj,               4),
                "effect_size": round(result["effect"],    4),
                "significant": p_adj < alpha,
                "note":        result["note"],
                "n_seeds":     len(common_seeds),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _cohens_d(xa: np.ndarray, xb: np.ndarray) -> float:
        diff = xa - xb
        sd   = np.std(diff, ddof=1)
        return float(np.mean(diff) / sd) if sd != 0 else 0.0

    @staticmethod
    def _rank_biserial(xa: np.ndarray, xb: np.ndarray) -> float:
        diff  = xa - xb
        diff  = diff[diff != 0]
        if len(diff) == 0:
            return 0.0
        ranks   = scipy_stats.rankdata(np.abs(diff))
        w_plus  = np.sum(ranks[diff > 0])
        w_minus = np.sum(ranks[diff < 0])
        n       = len(diff)
        return float((w_plus - w_minus) / (n * (n + 1) / 2))

    @classmethod
    def _test_pair(cls, xa: np.ndarray, xb: np.ndarray, alpha: float = 0.05) -> Dict:
        n = len(xa)
        if n < 3:
            return {"test": "N/A", "statistic": float("nan"),
                    "p_raw": float("nan"), "effect": float("nan"),
                    "note": f"n={n} < 3; test not applicable"}
        diff = xa - xb
        _, p_sw     = scipy_stats.shapiro(diff)
        is_normal   = p_sw > alpha
        
        if is_normal:
            stat, p   = scipy_stats.ttest_rel(xa, xb)
            effect    = cls._cohens_d(xa, xb)
            test_name = "paired t-test"
            note      = f"Shapiro-Wilk p={p_sw:.3f} -> normal -> {test_name}"
        else:
            try:
                stat, p   = scipy_stats.wilcoxon(xa, xb, alternative="two-sided")
                effect    = cls._rank_biserial(xa, xb)
                test_name = "Wilcoxon signed-rank"
            except ValueError:
                stat, p, effect = 0.0, 1.0, 0.0
                test_name = "Wilcoxon signed-rank"
            note = f"Shapiro-Wilk p={p_sw:.3f} -> non-normal -> {test_name}"
        return {
            "test": test_name,
            "statistic": float(stat),
            "p_raw":     float(p),
            "effect":    float(effect),
            "note":      note,
        }

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    @classmethod
    def leaderboard(
        cls,
        agg_df: pd.DataFrame,
        pairwise_df: pd.DataFrame,
        primary_metric: str,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        col_mean = f"{primary_metric} mean"
        col_lo   = f"{primary_metric} CI95_lo"
        col_hi   = f"{primary_metric} CI95_hi"
        if col_mean not in agg_df.columns:
            return agg_df

        has_ci = col_lo in agg_df.columns and col_hi in agg_df.columns
        cols   = ["Model", col_mean] + ([col_lo, col_hi] if has_ci else [])
        lb     = agg_df[cols].copy()
        if not has_ci:
            lb[col_lo] = float("nan")
            lb[col_hi] = float("nan")

        lb = lb.sort_values(col_mean, ascending=False).reset_index(drop=True)
        lb.insert(0, "Rank", lb.index + 1)

        def _fmt(r):
            if pd.isna(r[col_lo]) or pd.isna(r[col_hi]):
                return f"{r[col_mean]:.4f}  [CI pending]"
            return f"{r[col_mean]:.4f}  [{r[col_lo]:.4f}, {r[col_hi]:.4f}]"

        lb[primary_metric] = lb.apply(_fmt, axis=1)

        best_model = lb.at[0, "Model"]
        sig_col    = []
        for _, row in lb.iterrows():
            if row["Model"] == best_model:
                sig_col.append("—")
            else:
                sig_col.append(cls._sig_marker(pairwise_df, best_model, row["Model"], alpha))
        lb["vs Best"] = sig_col
        return lb[["Rank", "Model", primary_metric, "vs Best"]]

    @staticmethod
    def _sig_marker(df: pd.DataFrame, best: str, other: str, alpha: float = 0.05) -> str:
        if df.empty:
            return "N/A"
        mask = (
            ((df["Model_A"] == best)  & (df["Model_B"] == other)) |
            ((df["Model_A"] == other) & (df["Model_B"] == best))
        )
        rows = df[mask]
        if rows.empty:
            return "N/A"
        p = rows.iloc[0]["p_adj"]
        if pd.isna(p):    return "N/A"
        if p < 0.001:     return "***"
        if p < 0.01:      return "**"
        if p < alpha:     return "*"
        return "ns"


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def boot_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
) -> Tuple[float, float]:
    """Bootstrap-Konfidenzintervall für den Mittelwert."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(values[0]), float(values[0])
    boot = np.array([np.mean(rng.choice(values, size=n, replace=True))
                     for _ in range(n_bootstrap)])
    boot = boot[~np.isnan(boot)]
    if len(boot) == 0:
        return float("nan"), float("nan")
    return (
        float(np.percentile(boot, 100 * alpha / 2)),
        float(np.percentile(boot, 100 * (1 - alpha / 2))),
    )


def bootstrap_from_raw_predictions(
    pred_df: pd.DataFrame,
    raw_df:  pd.DataFrame,
    agg_df:  pd.DataFrame,
    n_bootstrap: int = 10000,
    ci_level:    float = 0.95,
) -> pd.DataFrame:
    """Calculate Bootstrap-CIs out of raw Image-Predictions and seed-level Metrics."""
    rng   = np.random.default_rng(seed=0)
    alpha = 1.0 - ci_level

    def _auroc(s, y):
        return roc_auc_score(y, s) if len(np.unique(y)) >= 2 else float("nan")

    def _auprc(s, y):
        return average_precision_score(y, s) if len(np.unique(y)) >= 2 else float("nan")

    def _f1(s, y):
        if len(np.unique(y)) < 2:
            return float("nan")
        best = max(f1_score(y, (s >= t).astype(int), zero_division=0)
                   for t in np.unique(s))
        return best

    METRIC_FNS = {
        "Image AUROC": _auroc,
        "Image AUPRC": _auprc,
        "Image F1":    _f1,
    }

    updated = agg_df.copy()

    if not pred_df.empty:
        print(f"\n  Raw-image bootstrap ({n_bootstrap:,} resamples, CI={ci_level*100:.0f}%) ...")
        model_raw_boot: Dict[str, Dict[str, List[float]]] = {}

        for (model, seed), group in pred_df.groupby(["model", "seed"]):
            scores = group["pred_score"].values.astype(float)
            labels = group["gt_label"].values.astype(int)
            n      = len(scores)
            if n < 2:
                continue
            if model not in model_raw_boot:
                model_raw_boot[model] = {m: [] for m in METRIC_FNS}

            if len(np.unique(labels)) >= 2:
                best_t = max(
                    np.unique(scores),
                    key=lambda t: f1_score(labels, (scores >= t).astype(int), zero_division=0),
                )
            else:
                best_t = 0.5

            def _f1_fixed(s, y, _threshold=best_t):
                if len(np.unique(y)) < 2:
                    return float("nan")
                return f1_score(y, (s >= _threshold).astype(int), zero_division=0)

            metric_fns_for_group = {
                "Image AUROC": _auroc,
                "Image AUPRC": _auprc,
                "Image F1":    _f1_fixed,
            }

            boot_indices = rng.choice(n, size=(n_bootstrap, n), replace=True)
            for metric_name, fn in metric_fns_for_group.items():
                boot_vals = np.array([fn(scores[idx], labels[idx]) for idx in boot_indices])
                model_raw_boot[model][metric_name].extend(
                    boot_vals[~np.isnan(boot_vals)].tolist()
                )
            print(f"    {model} seed {seed}: {n} images.")

        for model, metric_pools in model_raw_boot.items():
            mask = updated["Model"] == model
            if not mask.any():
                continue
            for metric_name, pool in metric_pools.items():
                arr = np.array(pool)
                if arr.size == 0:
                    lo, hi = float("nan"), float("nan")
                else:
                    lo = round(float(np.percentile(arr, 100 * alpha / 2)),        4)
                    hi = round(float(np.percentile(arr, 100 * (1 - alpha / 2))), 4)
                updated.loc[mask, f"{metric_name} CI95_lo"] = lo
                updated.loc[mask, f"{metric_name} CI95_hi"] = hi

    if not raw_df.empty:
        print(f"\n  Seed-level bootstrap ({n_bootstrap:,} resamples) ...")
        for model, group in raw_df.groupby("Model"):
            mask = updated["Model"] == model
            if not mask.any():
                continue
            for metric_name in METRIC_FNS:
                if metric_name not in raw_df.columns:
                    continue
                vals = group[metric_name].dropna().values.astype(float)
                lo, hi = boot_ci(vals, rng, n_bootstrap, alpha)
                updated.loc[mask, f"{metric_name} CI95_seed_lo"] = round(lo, 4) if not math.isnan(lo) else lo
                updated.loc[mask, f"{metric_name} CI95_seed_hi"] = round(hi, 4) if not math.isnan(hi) else hi

    return updated
