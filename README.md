readme = """# AD-IRAS — Anomaly Detection Benchmark for Industrial Computer Inspection

## Background
This repository is part of a HiWi project at IRAS-HKA, building on a bachelor thesis:
> *"Deep Learning-based Anomaly Detection in the Assembly of Industrial Computers"*
> Bachelor Thesis, HKA 2026

The thesis investigates Visual Anomaly Detection (VAD) applied to automated quality
inspection of custom-built industrial computers. These computers are produced in a
High-Mix Low-Volume (HMLV) manufacturing environment, where each batch is small and
product configurations vary frequently — making traditional supervised defect detection
economically infeasible.

The core idea: train AI models only on **normal** (defect-free) images, then flag
anything that deviates as a potential anomaly. No labelled defect examples needed.

## Project Context
- **Use case**: Top-down camera inspection of industrial computer assemblies
- **Challenge**: Small training datasets, high visual variance (cables, custom layouts)
- **Approach**: Unsupervised Visual Anomaly Detection (VAD)
- **Framework**: Built on [Anomalib](https://github.com/openvinotoolkit/anomalib)
- **Environment**: Google Colab + Google Drive + NVIDIA T4 GPU

---

## My Contributions (HiWi Extension)

### Bug Fixes
- **WinCLIP integration**: Fixed a pipeline crash caused by WinCLIP's zero-shot
  nature — it requires no training and therefore no validation split. The fix routes
  WinCLIP through a dedicated test-only evaluation path.
- **Memory management**: Fixed GPU RAM crashes during multi-model runs by adding
  proper cleanup between models.

### Pipeline Improvements
- **DatasetFactory pattern**: Clean dataset abstraction — adding a new dataset requires
  only one registration, no changes to pipeline logic
- **Auto dummy mask generation**: Automatically creates placeholder masks when no
  ground truth masks exist, allowing pixel-level metrics to run without crashing
- **GPU memory cleanup**: `torch.cuda.empty_cache()` + `gc.collect()` between model
  runs to prevent RAM crashes during multi-model benchmarks
- **Time tracking**: Per-model timing printed after each run
- **Dynamic results dashboard**: Auto-generates from latest CSV files — works for
  any number of models without hardcoding

### Original Experiments
- **ROI experiment**: Implemented and tested the ROI approach proposed but never
  evaluated in the thesis. Automatically derived the Region of Interest from ground
  truth masks (~33.4% of image area) and re-ran all models on cropped images.
  Finding: ROI cropping hurt most models — global context is important for this use case.
- **WinCLIP integration**: Added zero-shot language-based anomaly detection as a new
  model in the benchmark, with a defect-specific text prompt.
- **7-model benchmark**: Extended the original 5-model comparison with U-Flow and
  Dinomaly, using real annotated ground truth masks from supervisors.

---

## Results (MaskedDataset, Official Ground Truth Masks, Seed 420)

### Image-Level Classification
| Rank | Model | Image AUROC | Train Time | Type |
|------|-------|-------------|------------|------|
| 1 | **U-Flow** | **74.69%** | ~26 min | Norm. flow |
| 2 | FastFlow | 73.29% | ~39 min | Norm. flow |
| 3 | Dinomaly | 56.50% | ~173 min | ViT-based |
| 4 | PatchCore | 53.99% | ~1 min | Feature ext. |
| 5 | PaDiM | 53.43% | ~1 min | Feature ext. |
| 6 | WinCLIP | 49.23% | 0s (zero-shot) | Language |
| 7 | AnomalyDINO | 48.95% | ~1 min | ViT-based |

### Pixel-Level Localisation (with real masks)
| Model | Pixel AUPRO | Pixel AUPIMO |
|-------|-------------|--------------|
| U-Flow | 77.97% | 12.43% |
| Dinomaly | 75.60% | 2.68% |
| FastFlow | 69.88% | 17.84% |
| PaDiM | 68.99% | — |
| AnomalyDINO | 47.29% | 2.07% |
| PatchCore | 46.54% | 4.38% |
| WinCLIP | 16.32% | 0.00% |

### ROI Experiment Results
| Model | Full Image AUROC | ROI AUROC | Change |
|-------|-----------------|-----------|--------|
| U-Flow | 74.69% | 74.69% | = unchanged |
| FastFlow | 73.29% | 65.31% | ↓ -7.98% |
| PatchCore | 53.99% | 46.01% | ↓ -7.98% |
| PaDiM | 53.43% | 49.23% | ↓ -4.20% |
| WinCLIP | 49.23% | 46.85% | ↓ -2.38% |
| AnomalyDINO | 48.95% | 46.71% | ↓ -2.24% |

**Finding**: ROI cropping hurts most models — global context matters.
U-Flow's multi-scale architecture makes it naturally robust to spatial cropping.

---

Normalising flows (U-Flow, FastFlow) dominate both classification and localisation
WinCLIP achieves 49.23% AUROC with ZERO training — validates language-based VAD
ROI cropping removes useful global context — full images perform better
U-Flow is robust to ROI cropping — its multi-scale design focuses internally
All models struggle near random chance except the two normalising flows

→ confirms the difficulty of this HMLV use case


---

## Project Structure
AD-IRAS/

├── config/
│   └── settings.yaml        ← All configuration (models, paths, metrics)
├── src/
│   ├── benchmarker.py       ← Core orchestrator — WinCLIP fix, DatasetFactory
│   ├── data_loader.py       ← Dataset preprocessing + DatasetFactory
│   ├── models.py            ← Model registry
│   ├── augmentation.py      ← Albumentations pipeline
│   ├── reporter.py          ← CSV/JSON export
│   ├── statengine.py        ← Statistical analysis
│   └── utils.py             ← Helpers
└── main.ipynb               ← Main entry point (Google Colab)

## Setup
1. Clone this repo into your Google Drive
2. Update all paths in `config/settings.yaml`
3. Open `main.ipynb` in Google Colab
4. Enable T4 GPU (Runtime → Change runtime type → T4 GPU)
5. Run all cells in order

## Supported Datasets
Registered in `DatasetFactory` (`src/data_loader.py`):
- `MVTecAD` — standard anomaly detection benchmark
- `MVTec2` — extended MVTec dataset
- `VisA` — visual anomaly detection dataset
- `MaskedDataset` — official industrial computer dataset (this project)
- `ROIDataset` — cropped version derived from ground truth masks
- `custom` — any folder-based dataset (fallback)

## Next Steps (Planned)
- Few-shot experiments (k=10) — RQ3 from thesis
- Multi-seed evaluation (seeds: 420, 72, 42) — statistical reliability
- Grid-cell spatial approach — RQ2 from thesis

## Author
Aya Gherri | HiWi | IRAS-HKA | 2026
Supervised by: Philipp Augenstein & Till Weber
"""

with open('/content/drive/MyDrive/ADIRAS/anomaly_detection/README.md', 'w') as f:
    f.write(readme)
print("✅ README updated!")

import os
os.chdir('/content/drive/MyDrive/ADIRAS/anomaly_detection')
!git add README.md
!git commit -m "Update README with full results, ROI experiment and contributions"
!git push origin master
