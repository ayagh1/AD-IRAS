"""Handles the loading, preprocessing, and preparation of anomaly detection datasets."""

import csv
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np

from .utils import AUG_MARKER, VALID_EXTENSIONS, _collect_images, _count_images

class DatasetFactory:
    SUPPORTED = {
        "MVTecAD": {"split": "folder", "masks": True},
        "VisA":    {"split": "csv",    "masks": True},
        "custom":  {"split": "folder", "masks": False}
    }
    
    @staticmethod
    def get(data_source):
        if data_source in DatasetFactory.SUPPORTED:
            return DatasetFactory.SUPPORTED[data_source]
        return DatasetFactory.SUPPORTED["custom"]
# ---------------------------------------------------------------------------
# Grayscale conversion
# ---------------------------------------------------------------------------

def convert_dataset_to_greyscale(dataset_dir: Path) -> int:
    """Converts all images in-place to grayscale (BGR→GRAY→BGR). Returns count."""
    count = 0
    # Process all image files recursively
    for img_path in dataset_dir.rglob("*"):
        if img_path.suffix.lower() in VALID_EXTENSIONS:
            img = cv2.imread(str(img_path))
            if img is not None:
                grey_3ch = cv2.cvtColor(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    cv2.COLOR_GRAY2BGR,
                )
                cv2.imwrite(str(img_path), grey_3ch)
                count += 1
    return count


def compute_dataset_grey_norm(train_good_dir: Path) -> Tuple[List[float], List[float]]:
    """Computes normalization statistics for grayscale training images."""
    vals: List[float] = []
    for p in _collect_images(train_good_dir):
        img = cv2.imread(str(p))
        if img is not None:
            grey = img[:, :, 0].astype(np.float32) / 255.0
            vals.append(float(np.mean(grey)))
    if not vals:
        print("  compute_dataset_grey_norm: no images found — falling back to ImageNet defaults.")
        return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    mean_y = float(np.mean(vals))
    px_vals: List[np.ndarray] = []
    for p in _collect_images(train_good_dir):
        img = cv2.imread(str(p))
        if img is not None:
            px_vals.append(img[:, :, 0].astype(np.float32).ravel() / 255.0)
    std_y = float(np.std(np.concatenate(px_vals))) if px_vals else 0.2265
    # Ensure std is not too small to avoid numerical issues
    std_y = max(std_y, 1e-6)
    print(f"  Grey-adapted norm  ->  mean={mean_y:.4f}  std={std_y:.4f}  (applied to all 3 channels)")
    return ([mean_y] * 3, [std_y] * 3)


# ---------------------------------------------------------------------------
# Channel-Shuffle
# ---------------------------------------------------------------------------

def apply_channel_shuffle_dataset(dataset_dir: Path, permutation: Tuple[int, int, int]) -> int:
    """Applies channel permutation in-place to all images. Returns count."""
    count = 0
    perm = list(permutation)
    for img_path in dataset_dir.rglob("*"):
        if img_path.suffix.lower() in VALID_EXTENSIONS:
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.imwrite(str(img_path), img[:, :, perm])
                count += 1
    return count


# ---------------------------------------------------------------------------
# Pseudo Anomalie for Validiation Set
# ---------------------------------------------------------------------------

def generate_pseudo_anomaly_val_set(
    train_good_dir: Path,
    val_pseudo_dir: Path,
    config: dict,
    seed: int,
    sample_save_dir: Optional[Path] = None,
    aug_marker: str = AUG_MARKER,
) -> Tuple[int, int]:
    """Creates synthetic anomalies for the validation set via noise."""
    rng = np.random.default_rng(seed)
    originals = _collect_images(train_good_dir)

    if not originals:
        print(f"WARNING: No originals found in {train_good_dir}, pseudo-anomaly val skipped.")
        return 0, 0

    # Create directory structure
    val_good_dir = val_pseudo_dir / "good"
    val_bad_dir  = val_pseudo_dir / "bad"
    val_good_dir.mkdir(parents=True, exist_ok=True)
    val_bad_dir.mkdir(parents=True, exist_ok=True)

    if sample_save_dir:
        (sample_save_dir / "good").mkdir(parents=True, exist_ok=True)
        (sample_save_dir / "bad").mkdir(parents=True, exist_ok=True)

    if config.get("gaussian_std") is not None:
        noise_std = float(config["gaussian_std"])
        print(f"Fixed noise std: {noise_std} (override applied)")
    else:
        stds = []
        for img_path in originals:
            img = cv2.imread(str(img_path))
            if img is not None:
                stds.append(float(np.std(img.astype(np.float32))))
        avg_std    = float(np.mean(stds)) if stds else 40.0
        multiplier = config.get("std_multiplier", 1.0)
        noise_std  = avg_std * multiplier
        if not stds:
            print(f"  Warning: Could not compute adaptive std. Falling back to: {noise_std}")
        else:
            print(f"  Adaptive noise std: avg_std={avg_std:.1f} * {multiplier} = {noise_std:.1f}")

    s = noise_std / 255.0

    active      = config.get("active_methods", ["gauss", "coarse_dropout", "multiplicative", "shot"])
    patch_holes  = config.get("patch_holes",  (1, 3))
    patch_height = config.get("patch_height", (0.08, 0.30))
    patch_width  = config.get("patch_width",  (0.08, 0.30))

    def _noise_patch_fn(image, **kwargs):
        _rng = np.random.default_rng()
        out  = image.astype(np.float32)
        ih, iw = image.shape[:2]
        n = int(_rng.integers(patch_holes[0], patch_holes[1] + 1))
        for _ in range(n):
            ph = max(1, int(_rng.uniform(patch_height[0], patch_height[1]) * ih))
            pw = max(1, int(_rng.uniform(patch_width[0],  patch_width[1])  * iw))
            py = int(_rng.integers(0, max(1, ih - ph)))
            px = int(_rng.integers(0, max(1, iw - pw)))
            out[py:py+ph, px:px+pw] += _rng.normal(0, noise_std * 2.0, (ph, pw, image.shape[2]))
        return np.clip(out, 0, 255).astype(np.uint8)

    _method_map = {
        "gauss": A.GaussNoise(std_range=(s * 0.8, s * 1.2), p=1.0),
        "noise_patch": A.Lambda(image=_noise_patch_fn, p=1.0),
        "color_patch": A.CoarseDropout(
            num_holes_range=patch_holes,
            hole_height_range=patch_height,
            hole_width_range=patch_width,
            fill="random_uniform",
            p=1.0,
        ),
        "multiplicative": A.MultiplicativeNoise(
            multiplier_range=(max(0.0, 1.0 - s * 4), 1.0 + s * 4),
            per_channel=False,
            p=1.0,
        ),
        "shot": A.ShotNoise(scale_range=(s * 0.5, s * 1.5), p=1.0),
    }

    unknown = set(active) - _method_map.keys()
    if unknown:
        raise ValueError(f"Unknown active_methods: {unknown}. Valid: {list(_method_map.keys())}")

    pseudo_anomaly_pipeline = A.Compose(
        [A.OneOf([_method_map[m] for m in active], p=1.0)],
        seed=seed,
    )
    print(f"  Pseudo-anomaly methods active: {active}")

    n_create       = config.get("n_images") or len(originals)
    n_samples_save = config.get("n_samples_save", 10)
    n_good, n_bad  = 0, 0
    sampled_indices = rng.integers(0, len(originals), size=n_create)

    for i in range(n_create):
        img_path = originals[sampled_indices[i]]
        image    = cv2.imread(str(img_path))
        if image is None:
            continue

        good_out = val_good_dir / f"{img_path.stem}{aug_marker}vgood{i}{img_path.suffix}"
        cv2.imwrite(str(good_out), image)
        if sample_save_dir and n_good < n_samples_save:
            cv2.imwrite(str(sample_save_dir / "good" / f"sample_{n_good:02d}{img_path.suffix}"), image)
        n_good += 1

        noisy_img = pseudo_anomaly_pipeline(image=image)["image"]
        bad_out   = val_bad_dir / f"{img_path.stem}{aug_marker}vbad{i}{img_path.suffix}"
        cv2.imwrite(str(bad_out), noisy_img)
        if sample_save_dir and n_bad < n_samples_save:
            cv2.imwrite(str(sample_save_dir / "bad" / f"sample_{n_bad:02d}{img_path.suffix}"), noisy_img)
        n_bad += 1

    return n_good, n_bad


def copy_images_to_pseudo_val(
    source_dir: Path,
    train_good_dir: Path,
    val_pseudo_dir: Path,
    n_images: Optional[int],
    n_samples_save: int,
    sample_save_dir: Optional[Path],
    aug_marker: str = AUG_MARKER,
) -> Tuple[int, int]:
    """Copies real anomalies to the pseudo-validation set."""
    val_good_dir = val_pseudo_dir / "good"
    val_bad_dir  = val_pseudo_dir / "bad"
    val_good_dir.mkdir(parents=True, exist_ok=True)
    val_bad_dir.mkdir(parents=True, exist_ok=True)
    if sample_save_dir:
        (sample_save_dir / "good").mkdir(parents=True, exist_ok=True)
        (sample_save_dir / "bad").mkdir(parents=True, exist_ok=True)

    originals = _collect_images(train_good_dir)
    n_good = 0
    for img_path in originals:
        dst = val_good_dir / f"{img_path.stem}{aug_marker}vgood{n_good}{img_path.suffix}"
        shutil.copy2(img_path, dst)
        if sample_save_dir and n_good < n_samples_save:
            shutil.copy2(img_path, sample_save_dir / "good" / f"sample_{n_good:02d}{img_path.suffix}")
        n_good += 1

    if not source_dir.exists():
        print(f"  Source folder not found: {source_dir} — pseudo_val/bad will be empty.")
        return n_good, 0

    sources = _collect_images(source_dir)
    if not sources:
        print(f"  No images found in {source_dir} — pseudo_val/bad will be empty.")
        return n_good, 0

    n_use  = len(sources) if n_images is None else min(n_images, len(sources))
    n_bad  = 0
    for img_path in sources[:n_use]:
        dst = val_bad_dir / f"{img_path.stem}{aug_marker}vbad{n_bad}{img_path.suffix}"
        shutil.copy2(img_path, dst)
        if sample_save_dir and n_bad < n_samples_save:
            shutil.copy2(img_path, sample_save_dir / "bad" / f"sample_{n_bad:02d}{img_path.suffix}")
        n_bad += 1

    return n_good, n_bad


# ---------------------------------------------------------------------------
# Black Masks for normal-only Categories
# ---------------------------------------------------------------------------

def generate_black_masks(test_good_dir: Path, mask_dir: Path) -> int:
    """Creates black PNG masks for images without Ground-Truth masks."""
    mask_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for img_path in _collect_images(test_good_dir):
        mask_path = mask_dir / (img_path.stem + ".png")
        if mask_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        cv2.imwrite(str(mask_path), np.zeros((h, w), dtype=np.uint8))
        written += 1
    return written


# ---------------------------------------------------------------------------
# Dataset-Split Utilities
# ---------------------------------------------------------------------------

def kfold_indices(n: int, k: int, seed: int) -> List[List[int]]:
    """Shuffle indices and split into k folds."""
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    return [indices[i::k] for i in range(k)]


def get_train_good(split_dir: Path) -> set[str]:
    folder = split_dir / "train" / "good"
    if not folder.exists():
        return set()
    return {f.name for f in folder.iterdir() if f.is_file()}


def apply_visa_split(visa_root: Path, category: str) -> None:
    """Applies the VISA dataset split via split_csv/1cls.csv."""
    split_file = visa_root / "split_csv" / "1cls.csv"
    with split_file.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["object"] != category:
                continue
            label_dir = "good" if row["label"] == "normal" else "bad"
            img_src   = visa_root / row["image"]
            img_dst   = visa_root / category / row["split"] / label_dir / img_src.name
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            if not img_dst.exists():
                shutil.copyfile(img_src, img_dst)
            if row["label"] == "anomaly" and row.get("mask"):
                mask_src = visa_root / row["mask"]
                mask_dst = visa_root / category / "ground_truth" / "bad" / Path(row["mask"]).name
                mask_dst.parent.mkdir(parents=True, exist_ok=True)
                if not mask_dst.exists():
                    shutil.copyfile(mask_src, mask_dst)


def classify_role(img: str, train: set, test: set) -> str:
    if img in train:
        return "train"
    if img in test:
        return "test"
    return "-"


def split_note(roles: List[str]) -> str:
    unique = set(roles)
    if unique == {"-"}:            return "missing_all"
    if "-" in unique:              return "missing_some"
    if unique == {"train"}:        return "all_train"
    if unique == {"test"}:         return "all_test"
    if unique == {"train", "test"}: return "SWAPPED"
    return "mixed"


# ---------------------------------------------------------------------------
# File Operationen
# ---------------------------------------------------------------------------

def copy_file(src: Path, dst_folder: Path, dry_run: bool = False) -> None:
    if dry_run:
        print(f"    [DRY] COPY  {src.name}  ->  {dst_folder}/")
    else:
        dst_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst_folder / src.name))


def move_file(src: Path, dst_folder: Path, dry_run: bool = False) -> None:
    if dry_run:
        print(f"    [DRY] MOVE  {src.name}  ->  {dst_folder}/")
    else:
        dst_folder.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst_folder / src.name))
        print(f"    MOVED  {src.name}  ->  {dst_folder.name}/")
