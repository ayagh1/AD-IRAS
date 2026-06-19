"""General Helper Functions: Set Seed, Create Folders, Metadata, Image Utilities."""

import gc
import os
import platform
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#    Supported image file extensions (in lowercase)
VALID_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg"}
AUG_MARKER: str = "_aug_"


# ---------------------------------------------------------------------------
# Configuration Management
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_folder(output_dir: str | Path, run_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder = Path(output_dir) / f"{run_name}_{timestamp}"
    return ensure_dir(folder)


# ---------------------------------------------------------------------------
# Image File Utilities
# ---------------------------------------------------------------------------

def _collect_images(folder: Path) -> List[Path]:
    """Returns only the original source images, excluding any augmented copies."""
    if not folder.exists():
        return []
    return [
        p for p in folder.iterdir()
        if p.suffix.lower() in VALID_EXTENSIONS and AUG_MARKER not in p.stem
    ]


def _count_images(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    return sum(1 for p in dir_path.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS)


def files_in(folder: Path) -> set[str]:
    if not folder.exists():
        return set()
    return {f.name for f in folder.iterdir() if f.is_file()}


def imgs_in(folder: Path) -> List[Path]:
    return sorted(f for f in folder.iterdir() if f.is_file()) if folder.exists() else []


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def collect_metadata(session_id: str = "", seeds: List[int] = None, models: List[str] = None, config: dict = None) -> dict:
    """Collects system and library versions for reproducibility."""
    cfg = config or {}
    meta: dict = {
        "session_id":       session_id,
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "seeds":            seeds or [],
        "models":           models or [],
        "threshold_method": cfg.get("threshold", {}).get("method", ""),
        "primary_metric":   cfg.get("evaluation", {}).get("primary_metric", ""),
        "ci_level":         cfg.get("statistics", {}).get("ci_level", 0.95),
        "n_bootstrap":      cfg.get("statistics", {}).get("n_bootstrap", 10000),
        "alpha":            cfg.get("statistics", {}).get("alpha", 0.05),
        "bonferroni":       cfg.get("statistics", {}).get("bonferroni", True),
        "hardware": {},
        "software": {},
    }
    
    # Collect hardware information
    meta["hardware"]["python"]   = platform.python_version()
    meta["hardware"]["platform"] = platform.platform()
    if torch.cuda.is_available():
        meta["hardware"]["cuda_version"] = torch.version.cuda
        meta["hardware"]["gpu_name"]     = torch.cuda.get_device_name(0)
        meta["hardware"]["gpu_vram_gb"]  = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    else:
        meta["hardware"]["gpu_name"] = "CPU only"
    for pkg in ("torch", "anomalib", "albumentations", "cv2", "numpy", "pandas", "scipy"):
        try:
            mod = __import__(pkg)
            meta["software"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            meta["software"][pkg] = "not installed"
    
    return meta


# ---------------------------------------------------------------------------
# Safe Memory Management
# ---------------------------------------------------------------------------

def clear_memory() -> None:
    """
    Clear GPU and system memory caches.
    Useful to call between model training runs to free up resources.
    """
    gc.collect()  # Collect Python garbage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU memory cache
