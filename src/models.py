"""Model Registry and Configuration Builder for all anomalib Models."""

from pathlib import Path
from typing import Any, Dict

from anomalib.models import AnomalyDINO
from anomalib.models.image import (
    Cfa, Cflow, Csflow, Dfkde, Dfm, Dinomaly, Draem, Dsr,
    EfficientAd, Fastflow, Fre, Padim, Patchcore,
    ReverseDistillation, Stfpm, Supersimplenet, Uflow, WinClip,
)
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize

# Dictionary mapping model names to their corresponding anomalib classes
# Used for dynamic model instantiation during training/evaluation
MODEL_MAPPING: Dict[str, Any] = {
    "WinClip":             WinClip,
    "Patchcore":           Patchcore,
    "Padim":               Padim,
    "Dfkde":               Dfkde,
    "Dfm":                 Dfm,
    "Cfa":                 Cfa,
    "Cflow":               Cflow,
    "Csflow":              Csflow,
    "Fastflow":            Fastflow,
    "Uflow":               Uflow,
    "Draem":               Draem,
    "Dsr":                 Dsr,
    "Fre":                 Fre,
    "Stfpm":               Stfpm,
    "ReverseDistillation": ReverseDistillation,
    "EfficientAd":         EfficientAd,
    "AnomalyDINO":         AnomalyDINO,
    "SuperSimpleNet":      Supersimplenet,
    "Dinomaly":            Dinomaly,
}

try:
    from anomalib.models.image import UniNet
    MODEL_MAPPING["UniNet"] = UniNet
except ImportError:
    pass

# Models that only produce image-level predictions (no per-pixel anomaly maps)
# These models are excluded from pixel-level metrics calculations
IMAGE_ONLY_MODELS: set[str] = {"Dfkde", "Dfm"}


def build_model_config(model_name: str, cfg: dict) -> dict:
    """
    Constructs the Anomalib model arguments from the configuration dictionary.
    Integrates backbone and ViT references from the top-level configuration.
    """
    models_cfg   = cfg.get("models", {})
    model_cfgs   = models_cfg.get("config", {})
    model_entry  = model_cfgs.get(model_name, {})
    # Define default backbones for CNN and Vision Transformer models
    backbone_cnn = models_cfg.get("backbone_cnn",    "wide_resnet50_2")
    backbone_vit = models_cfg.get("backbone_vit",    "dinov2reg_vit_base_14")
    cnn_layers   = models_cfg.get("backbone_layers", ["layer2", "layer3"])
    paths_cfg    = cfg.get("paths", {})

    # Extract batch sizes and epochs, with fallback defaults
    train_bs = model_entry.get("train_batch_size",
               model_entry.get("batch_size",
               cfg["training"]["batch_size"]))
    eval_bs  = model_entry.get("eval_batch_size", train_bs)
    epochs   = model_entry.get("epochs", 10)

    # Start with model-specific arguments from config, if provided
    model_args = dict(model_entry.get("model_args", {}))

    # Auto-configure CNN backbone for supported models
    if model_name in ("Padim", "Patchcore", "Fre", "Stfpm",
                      "ReverseDistillation", "Fastflow", "Cflow", "Cfa", "SuperSimpleNet"):
        model_args.setdefault("backbone", backbone_cnn)

    if model_name in ("Padim", "Patchcore", "SuperSimpleNet"):
        model_args.setdefault("layers", cnn_layers)

    # ViT-Encoder-Defaults
    if model_name in ("Dinomaly", "AnomalyDINO"):
        model_args.setdefault("encoder_name", backbone_vit)

    # EfficientAd: imagenet_dir from paths
    if model_name == "EfficientAd":
        imagenet_dir = paths_cfg.get("imagenette", "")
        model_args.setdefault("imagenet_dir", str(imagenet_dir))
        size_str = model_args.pop("model_size", "M")
        model_args["model_size"] = EfficientAdModelSize[size_str] if isinstance(size_str, str) else size_str

    # Draem: dtd_dir from paths
    if model_name == "Draem":
        dtd_dir = paths_cfg.get("draem_textures", "")
        model_args.setdefault("dtd_dir", str(dtd_dir))
        beta = model_args.get("beta")
        if isinstance(beta, list):
            model_args["beta"] = tuple(beta)

    return {
        "train_batch_size": train_bs,
        "eval_batch_size":  eval_bs,
        "epochs":           epochs,
        "model_args":       model_args,
    }
