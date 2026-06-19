"""Data Augmentation: Image Transformations and Normalization Patching."""

from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torchvision.transforms.v2 import Normalize

from .utils import AUG_MARKER, VALID_EXTENSIONS, _collect_images


def build_augmentation_pipeline(aug_config: dict) -> A.Compose:
    """Creates Albumentations pipeline from the YAML augmentation block."""
    import cv2 as _cv2

    _border_modes = {
        "BORDER_REPLICATE": _cv2.BORDER_REPLICATE,
        "BORDER_REFLECT":   _cv2.BORDER_REFLECT,
        "BORDER_WRAP":      _cv2.BORDER_WRAP,
    }
    
    transforms = []
    # Parse each transformation specification from the config
    for step in aug_config.get("pipeline", []):
        t      = step["type"]
        params = {k: v for k, v in step.items() if k != "type"}
        # Map border_mode string from YAML to the corresponding OpenCV constant
        if "border_mode" in params and isinstance(params["border_mode"], str):
            params["border_mode"] = _border_modes.get(params["border_mode"], _cv2.BORDER_REPLICATE)
        cls = getattr(A, t, None)
        if cls is None:
            raise ValueError(f"Unknown Albumentations Transformation: {t}")
        transforms.append(cls(**params))
    seed = aug_config.get("seed", 42)
    return A.Compose(transforms, seed=seed)


def inject_augmented_images(
    train_good_dir: Path,
    images_per_original: int,
    pipeline: A.Compose,
    aug_seed: int,
) -> List[Path]:
    """Generates augmented copies of all original images within the train/good directory."""
    originals = _collect_images(train_good_dir)
    if not originals:
        print(f"Augmentation failed: no original images in {train_good_dir}")
        return []
    
    created: List[Path] = []
    for img_path in originals:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Could not read {img_path.name}, skipping.")
            continue
        for i in range(images_per_original):
            aug_image = pipeline(image=image)["image"]
            out_path  = train_good_dir / f"{img_path.stem}{AUG_MARKER}{i}{img_path.suffix}"
            cv2.imwrite(str(out_path), aug_image)
            created.append(out_path)
    return created


def patch_normalize_in_transform(transform, mean: List[float], std: List[float]) -> bool:
    """Searches for the Normalize layer in a transform tree and overrides its mean/std."""
    if isinstance(transform, Normalize):
        # Update normalization parameters
        transform.mean = torch.tensor(mean, dtype=torch.float32)
        transform.std  = torch.tensor(std,  dtype=torch.float32)
        return True
    for sub in getattr(transform, "transforms", []):
        if patch_normalize_in_transform(sub, mean, std):
            return True
    inner = getattr(transform, "transform", None)
    if inner is not None:
        if patch_normalize_in_transform(inner, mean, std):
            return True
    
    return False
