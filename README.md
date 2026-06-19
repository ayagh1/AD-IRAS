# Anomaly Detection Benchmark

A comprehensive benchmarking framework for unsupervised and zero-shot anomaly detection, built on [Anomalib](https://github.com/openvinotoolkit/anomalib) and PyTorch Lightning. It supports 19+ models, multiple datasets, configurable preprocessing, rigorous statistical evaluation, and structured result export.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Running the Benchmark](#running-the-benchmark)
4. [Configuration Reference (`settings.yaml`)](#configuration-reference-settingsyaml)
5. [Supported Models](#supported-models)
6. [Output Files](#output-files)

---

## Project Structure

```
anomaly_detection/
├── config/
│   └── settings.yaml        ← All configuration lives here
├── src/
│   ├── augmentation.py      ← Albumentations pipeline builder & image augmentation
│   ├── benchmarker.py       ← Core training/evaluation orchestrator (BenchmarkRunner)
│   ├── data_loader.py       ← Dataset preprocessing utilities & pseudo-anomaly generation
│   ├── engine.py            ← PyTorch Lightning callbacks (prediction collection, loss aliasing)
│   ├── models.py            ← Model registry & hyperparameter builder
│   ├── reporter.py          ← Visualization callbacks & CSV/JSON export
│   ├── statengine.py        ← Statistical analysis: bootstrap CIs & pairwise hypothesis tests
│   └── utils.py             ← Helpers: seed setting, directory management, metadata collection
└── main.ipynb               ← Main entry point (designed for Google Colab)
```

---

## Setup

### Requirements

- Python 3.9+
- CUDA-capable GPU (strongly recommended)
- Google Colab or a local environment with Google Drive access (for the default path setup)

### Installation

The notebook (`main.ipynb`) installs all dependencies automatically in its first cell. For a manual install:

```bash
pip install anomalib lightning pyyaml albumentations opencv-python-headless
pip install torch torchvision timm numpy pandas scipy scikit-learn tqdm
```

Or use the included requirements file:

```bash
pip install -r requirements.txt
```

### File Directory Configuration

**Before running anything, you must update all paths in `config/settings.yaml` to match your environment.**

The default paths are set for Google Colab + Google Drive. If you run locally or use a different cloud storage, change every path under `project_dir` and `paths:` to reflect your actual directory layout.

Key paths to adjust:

| Key | Default | What it points to |
|---|---|---|
| `project_dir` | `/content/drive/MyDrive/anomaly_detection` | Root of this `anomaly_detection/` folder |
| `paths.base_dataset_root` | `/content/drive/MyDrive/datasets/` | Parent folder containing all dataset folders |
| `paths.output_dir` | `/content/drive/MyDrive/results_elma_benchmark/` | Where final CSVs and results are saved |
| `paths.local_workspace` | `/content/local_workspace` | Fast local SSD scratch space (Colab: `/content/`) |
| `paths.temp_results` | `/content/temp_results` | Temporary per-category result storage |
| `paths.temp_checkpoints` | `/content/temp_checkpoints` | Temporary model checkpoint storage |
| `paths.draem_textures` | `/content/drive/MyDrive/datasets/DRAEM_TEXTURES` | Only required when running DRAEM |
| `paths.imagenette` | `/content/drive/MyDrive/datasets/imagenette` | Only required when running EfficientAd |

---

## Running the Benchmark

Open `main.ipynb` and run all cells in order:

1. **Cell 0** – Installs dependencies
2. **Cell 1** – Mounts Google Drive and loads `settings.yaml`
3. **Cell 2** – Imports all modules
4. **Cell 3** – Sets up output directories and the augmentation pipeline
5. **Cell 4** – Main benchmark loop: for each category and colour mode, trains and evaluates all configured models across all seeds
6. **Cell 5** – Concatenates per-category results and saves consolidated output CSVs

---

## Configuration Reference (`settings.yaml`)

### `project_dir`

Absolute path to the `anomaly_detection/` folder. Must be updated to your environment.

---

### `paths`

All filesystem paths used by the framework. All must be valid directories. See [File Directory Configuration](#file-directory-configuration) above.

- `all_section_images`, `fewshot_dataset`, `golden_unit_defects`, `generated_anomalies` – paths to custom datasets; only required if the corresponding `data_source` is selected.
- `draem_textures` – texture dataset for DRAEM's anomaly synthesis; required only when running DRAEM.
- `imagenette` – ImageNet subset for EfficientAd's teacher network; required only when running EfficientAd.

---

### `run`

Controls which data and experiments to execute.

| Parameter | Type | Description |
|---|---|---|
| `data_source` | string | Dataset to use. Options: `"MVTecAD"`, `"MVTec2"`, `"VisA"`, or any custom folder name (e.g. `"FewShot1344GridDataset_8020"`). The framework looks for `<base_dataset_root>/<data_source>/<category>/`. |
| `categories` | list or `null` | List of category subfolder names to process (e.g. `["bottle", "cable"]`). Set to `null` to process all categories found in the dataset folder. |
| `seeds` | list of int | Random seeds for repeated runs. Multiple seeds enable statistical analysis. Example: `[42, 123, 420]`. |
| `colour_modes` | list of string | Preprocessing colour modes to benchmark. Options: `"rgb"` (no change), `"grey_imagenet"` (greyscale with ImageNet normalisation), `"grey_adapted"` (greyscale with per-dataset normalisation), `"shuffle"` (channel permutation). |
| `channel_shuffle_permutation` | list of 3 int | Channel index permutation used when `colour_modes` includes `"shuffle"`. Example: `[2, 0, 1]` = RGB → BRG. |
| `allow_normal_only_categories` | bool | Whether to process categories that have no anomalous test images. |
| `save_run_models` | bool | Whether to persist trained model checkpoints to `output_dir`. |
| `models_to_save` | list of string | If `save_run_models` is `true`, only these model names are saved. Empty list = save all. |
| `winclip_manual_prompt` | string | Text prompt used by WinCLIP for zero-shot detection. Describe the object class in plain language. |
| `use_folder_prompts` | bool | If `true`, reads per-category text prompt files instead of `winclip_manual_prompt`. |

---

### `training`

Global training and data-loading defaults. These are **fallback values** — every parameter listed here applies to all models unless a model explicitly overrides it in `models.config.<ModelName>` (see below).

| Parameter | Type | Description |
|---|---|---|
| `resize_img` | int | Images are resized to this resolution (square) before training. Default: `256`. |
| `batch_size` | int | **Global fallback** batch size. Override per model via `models.config.<Model>.batch_size`. Default: `16`. |
| `num_workers` | int | DataLoader worker threads. Set to `0` on Windows if you encounter multiprocessing errors. Default: `2`. |
| `seed` | int | Global seed for reproducibility (in addition to per-run seeds). Default: `42`. |
| `baseline_config` | bool | If `true`, uses the default Anomalib pre-processor (ImageNet normalisation, standard resize). |

---

### `augmentation`

Offline augmentation applied to training images before any run.

| Parameter | Type | Description |
|---|---|---|
| `enabled` | bool | Enable or disable augmentation entirely. |
| `images_per_original` | int | Number of augmented copies to generate per original training image. |
| `seed` | int | Seed for the augmentation RNG. |
| `pipeline` | list | List of Albumentations transforms. Each entry has a `type` (class name) and optional keyword arguments. Supported types: any `albumentations` transform (e.g. `HorizontalFlip`, `VerticalFlip`, `Rotate`, `RandomBrightnessContrast`). |
| `perlin.enabled` | bool | Apply Perlin-noise-based anomaly augmentation (experimental). |
| `perlin.anomaly_source_path` | string or `null` | Path to texture images for Perlin anomaly synthesis. |
| `perlin.probability` | float | Probability of applying Perlin augmentation per image. |
| `perlin.blend_factor` | [float, float] | Min/max blend factor range for the anomaly overlay. |
| `perlin.rotation_range` | [int, int] | Rotation range (degrees) for texture placement. |

---

### `threshold`

Controls how the anomaly score threshold for the validation set is determined.

| Parameter | Type | Description |
|---|---|---|
| `method` | string | Threshold strategy. Options: `"NOISE_INJECTION"`, `"MANUAL_ANOMALIES"`, `"GENERATIVE_AI"`, `"HARD_THRESHOLD"`, `"SAME_AS_TEST"`. |

**`NOISE_INJECTION`** – Synthetic anomalies are generated from normal training images:

| Sub-parameter | Type | Description |
|---|---|---|
| `gaussian_std` | float or `null` | Fixed Gaussian noise std. `null` = auto-scale from image variance. |
| `std_multiplier` | float | Scales the auto-computed std. Default: `1.0`. |
| `n_images` | int | Number of synthetic anomaly images to generate per category. |
| `active_methods` | list of string | Noise types to use: `"gauss"`, `"noise_patch"`, `"color_patch"`, `"multiplicative"`, `"shot"`. |
| `patch_holes` | [int, int] | Min/max number of noise patch regions per image. |
| `patch_height` | [float, float] | Min/max patch height as a fraction of image height. |
| `patch_width` | [float, float] | Min/max patch width as a fraction of image width. |
| `n_samples_save` | int | Number of synthetic anomaly samples to save as preview images. |
| `sensitivity_override` | float or `null` | Override the model's threshold sensitivity directly. `null` = use adaptive F1 threshold. |

**`MANUAL_ANOMALIES`** – Uses real anomalous images placed in a separate folder:

| Sub-parameter | Description |
|---|---|
| `n_images` | Max anomaly images to use (`null` = all available). |
| `n_samples_save` | Number of samples to save as previews. |
| `sensitivity_override` | Direct threshold override or `null`. |

**`GENERATIVE_AI`** – Uses AI-generated anomaly images from `paths.generated_anomalies`. Same sub-parameters as `manual_anomalies`.

**`HARD_THRESHOLD`** – Fixed score thresholds, no validation set needed:

| Sub-parameter | Description |
|---|---|
| `image_sensitivity` | Fixed image-level threshold (0–1). |
| `pixel_sensitivity` | Fixed pixel-level threshold (0–1). |

**`SAME_AS_TEST`** – Threshold is determined from the test set directly (upper-bound / oracle mode).

---

### `evaluation`

| Parameter | Type | Description |
|---|---|---|
| `primary_metric` | string | Metric used for leaderboard ranking. Options: any metric in `metrics`. Default: `"Image AUROC"`. |
| `metrics` | list of string | All metrics to compute. Options: `"Image AUROC"`, `"Image AUPRC"`, `"Image F1"`, `"Pixel AUROC"`, `"Pixel F1"`, `"Pixel AUPRC"`, `"Pixel AUPRO"`, `"Pixel AUPIMO"`. Note: pixel metrics are skipped for image-only models (`Dfkde`, `Dfm`). |

---

### `statistics`

| Parameter | Type | Description |
|---|---|---|
| `alpha` | float | Significance level for hypothesis tests (default `0.05`). |
| `ci_level` | float | Confidence interval level (default `0.95`). |
| `n_bootstrap` | int | Number of bootstrap resamples for CI estimation (default `10000`). |
| `bonferroni` | bool | Apply Bonferroni correction for multiple comparisons in pairwise tests. |
| `bootstrap_on_raw` | bool | If `true`, bootstraps CIs directly from per-image predictions rather than from seed-level metrics. Requires multiple seeds. |

---

### `visualization`

**`heatmap`** – Uses Anomalib's built-in visualiser:

| Sub-parameter | Description |
|---|---|
| `enabled` | Enable/disable heatmap saving. |
| `fields_config.anomaly_map.colormap` | Matplotlib colormap name (e.g. `"hot"`, `"jet"`). |
| `fields_config.anomaly_map.normalize` | Normalise anomaly map to [0, 1] before display. |
| `fields_config.gt_mask.mode` | Mask overlay mode: `"fill"` or `"contour"`. |
| `fields_config.gt_mask.alpha` | Opacity of the mask overlay (0–1). |
| `fields_config.gt_mask.color` | RGB colour for the mask overlay. |

**`custom`** – Side-by-side composite image saver (original | heatmap | contour mask):

| Sub-parameter | Description |
|---|---|
| `enabled` | Enable/disable custom image saving. |
| `colormap` | OpenCV colormap integer (e.g. `16` = `cv2.COLORMAP_VIRIDIS`). |
| `font_scale` | Font size for the prediction label overlay. |
| `font_thickness` | Thickness of the label text. |
| `mask_color` | BGR colour for contour lines (e.g. `[0, 0, 255]` = red). |
| `mask_thickness` | Pixel thickness of contour lines. |
| `pred_anomaly_threshold` | Score threshold for colouring the label as anomaly (red) vs normal (green). |

---

### `models`

| Parameter | Type | Description |
|---|---|---|
| `to_run` | list of string | Models to benchmark. Must match names in `models.config`. See [Supported Models](#supported-models). |
| `image_only_models` | list of string | Models that produce only image-level scores (no pixel maps). Pixel metrics are automatically skipped. |
| `gradient_models` | list of string | Models requiring gradient computation. Listed for reference; handled automatically. |
| `backbone_cnn` | string | **Global** CNN backbone for all compatible models. Override per model via `model_args.backbone`. Default: `"wide_resnet50_2"`. |
| `backbone_layers` | list of string | **Global** feature extraction layers for CNN backbones. Override per model via `model_args.layers`. Default: `["layer2", "layer3"]`. |
| `backbone_vit` | string | **Global** ViT backbone for transformer-based models. Override per model via `model_args.encoder_name`. Default: `"dinov2reg_vit_base_14"`. |

**`models.config.<ModelName>`** – Per-model overrides:

Any key not set here falls back to the global default in `training:` or `models:`.

| Sub-parameter | Type | Description |
|---|---|---|
| `epochs` | int | Training epochs for this model. |
| `batch_size` | int | Overrides `training.batch_size` for this model. |
| `train_batch_size` | int | Overrides `batch_size` for training only (takes priority over `batch_size`). |
| `eval_batch_size` | int | Overrides `batch_size` for evaluation only. Falls back to `batch_size` / `train_batch_size` if not set. |
| `model_args.backbone` | string | Overrides `models.backbone_cnn` for this model (CNN models only). |
| `model_args.layers` | list | Overrides `models.backbone_layers` for this model (CNN models only). |
| `model_args.encoder_name` | string | Overrides `models.backbone_vit` for this model (ViT models only). |
| `model_args.<key>` | any | Any other key is passed directly to the Anomalib model constructor. |

**Override priority (highest → lowest):**

```
Batch size:   train_batch_size  >  batch_size  >  training.batch_size
CNN backbone: model_args.backbone              >  models.backbone_cnn
CNN layers:   model_args.layers                >  models.backbone_layers
ViT backbone: model_args.encoder_name          >  models.backbone_vit
```

**Example** – lighter backbone and smaller batch for a single model, everything else stays global:

```yaml
models:
  backbone_cnn: "wide_resnet50_2"   # used by all CNN models ...
  batch_size: 16                    # ... unless overridden below

  config:
    Padim:
      batch_size: 8                 # overrides global batch size only for Padim
      model_args:
        backbone: "resnet18"        # overrides global backbone only for Padim
```

---

## Supported Models

| Model | Type | Notes |
|---|---|---|
| `WinClip` | Zero-shot | Requires `winclip_manual_prompt` |
| `AnomalyDINO` | Zero-shot | ViT backbone |
| `Dfkde` | Image-only | No pixel metrics |
| `Dfm` | Image-only | No pixel metrics |
| `Padim` | Feature extraction | |
| `Patchcore` | Feature extraction | Coreset-based memory bank |
| `Fre` | Feature reconstruction | |
| `Dinomaly` | Feature reconstruction | ViT backbone |
| `Fastflow` | Normalising flow | Gradient model |
| `Cflow` | Normalising flow | Gradient model |
| `Csflow` | Normalising flow | Gradient model |
| `Uflow` | Normalising flow | Gradient model |
| `ReverseDistillation` | Knowledge distillation | |
| `Stfpm` | Knowledge distillation | |
| `UniNet` | Knowledge distillation | Gradient model |
| `EfficientAd` | Knowledge distillation | Requires `paths.imagenette` |
| `Draem` | Generative | Requires `paths.draem_textures` |
| `Dsr` | Generative | |
| `Cfa` | Coupling-feature alignment | |
| `SuperSimpleNet` | Hybrid | |

---

## Output Files

After a completed run the following files are written to `paths.output_dir`:

| File | Description |
|---|---|
| `ALL_raw_seeds.csv` | Per-model, per-seed, per-category metrics |
| `ALL_aggregated_stats.csv` | Mean, std, median, IQR, 95% CI per model across seeds |
| `ALL_leaderboard.csv` | Models ranked by `primary_metric` with statistical significance markers |
| `ALL_pairwise_tests.csv` | Pairwise hypothesis test results (t-test or Wilcoxon, with effect sizes) |
| `ALL_raw_predictions.csv` | Per-image prediction scores, ground-truth labels, and predicted labels |
| `metadata.json` | System info, library versions, and run configuration snapshot |
