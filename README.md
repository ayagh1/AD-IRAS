readme = """# AD-IRAS — Anomaly Detection Benchmark for Industrial Computer Inspection

## Background
This repository is part of a HiWi project at IRAS-HKA, building on the master thesis:
> *"Deep Learning-based Anomaly Detection in the Assembly of Industrial Computers"*  
> by Philipp Augenstein, HKA 2026

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

## My Contributions (HiWi Extension)
Starting from the original thesis pipeline, the following work was done:

### Bug Fixes
- **WinClip integration fix**: WinClip is a zero-shot model requiring no training.  
  The pipeline crashed because it expected a validation split that WinClip never creates.  
  Fixed by adding `limit_val_batches=0` to the Engine and giving WinClip its own  
  `test_only_datamodule`, bypassing the validation step entirely.
- **Model name fix**: Corrected `"WinCLIP"` to `"WinClip"` (case-sensitive MODEL_MAPPING)

### Pipeline Improvements
- **Auto dummy mask generation**: Automatically creates black placeholder masks when  
  no ground truth masks exist, allowing pixel-level metrics to run without crashing
- **GPU memory cleanup**: Added `torch.cuda.empty_cache()` and `gc.collect()` between  
  model runs to prevent RAM crashes during multi-model benchmarks
- **Time tracking**: Added per-model timing — prints finish time after each model
- **Cleaner code structure**: Moved all imports to top of benchmark cell

### Experiments
- Tested 5 models on the official masked dataset provided by supervisors
- First run with real ground truth masks (55 annotated anomaly masks)
- Added WinClip (zero-shot, language-based) as a new model comparison

## Results (Official Masked Dataset, Seed 420)

### Image-Level Classification
| Rank | Model | Image AUROC | Training Time |
|------|-------|-------------|---------------|
| 1 | FastFlow | 67.76% | ~62 min (GPU) |
| 2 | PaDiM | 52.87% | ~45s (GPU) |
| 3 | WinClip | 49.23% | ~30 min (CPU, zero-shot) |
| 4 | PatchCore | 47.55% | ~57s (GPU) |
| 5 | AnomalyDINO | 47.41% | ~51s (GPU) |

### Pixel-Level Localisation (with real masks)
| Model | Pixel AUPRO | Pixel AUPRC | Pixel AUPIMO |
|-------|-------------|-------------|--------------|
| FastFlow | 51.77% | 32.71% | 11.96% |
| PaDiM | 68.89% | 33.91% | 0.16% |
| PatchCore | 41.00% | 18.82% | 4.22% |
| AnomalyDINO | 47.13% | 21.66% | 1.63% |
| WinClip | 16.32% | 0.33% | 0.00% |

### Key Findings
- **FastFlow** is the best classifier (67.76% AUROC) — consistent with thesis findings
- **PaDiM** is the best localiser (68.89% AUPRO) — finds WHERE defects are most accurately
- **WinClip** achieves competitive classification (49.23%) with **zero training** using only  
  a text description — validates the value of language-based anomaly detection
- All models score near random chance on classification except FastFlow — confirming  
  the difficulty of this use case identified in the original thesis

## Project Structure
