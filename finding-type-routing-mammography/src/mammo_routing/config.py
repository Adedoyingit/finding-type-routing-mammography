"""Training recipe and model registry.

The values below are the ones used for every model in the paper. They are
collected in one place so that training, evaluation and analysis agree.
"""
from dataclasses import dataclass, field

import numpy as np

CNN_BACKBONES = ["resnet50", "efficientnet_b0", "densenet121", "convnext_tiny"]
VIT_BACKBONE = "vit_base_patch16_224"
FINDING_TYPES = ["Mass", "Calcification"]  # category_idx 0 and 1
SEEDS = [42, 123, 456]


@dataclass
class TrainConfig:
    img_size: int = 224
    batch_size: int = 32
    learning_rate: float = 1e-4
    vit_learning_rate: float = 5e-5      # ViT-B/16 specialists only
    weight_decay: float = 1e-4
    max_epochs: int = 60
    patience: int = 15                   # early stopping on validation AUC
    min_delta: float = 1e-4
    focal_gamma: float = 2.0
    aux_lambda: float = 0.3              # weight of the auxiliary finding-type loss
    minority_boost: float = 1.5          # sampler boost for the rarest finding x label stratum
    gradient_clip: float = 1.0
    num_workers: int = 2
    seeds: list = field(default_factory=lambda: list(SEEDS))


# Ensemble conditions. 'beta' is the weight of the explicit affinity prior (Eq. 4);
# 'rand_affinity' initialises the affinity matrix at random instead of from specialist AUCs.
CONDITIONS = {
    "C1":         dict(arch="UnifiedEnsemble",        has_aux=False, beta=None, rand_affinity=False),
    "C1_aux":     dict(arch="UnifiedEnsembleWithAux", has_aux=True,  beta=None, rand_affinity=False),
    "C2_gate":    dict(arch="AFAGEGateOnly",          has_aux=False, beta=0.0,  rand_affinity=True),
    "C3_rand":    dict(arch="AFAGE",                  has_aux=True,  beta=1.0,  rand_affinity=True),
    "C3_beta0":   dict(arch="AFAGE",                  has_aux=True,  beta=0.0,  rand_affinity=False),
    "C3_beta025": dict(arch="AFAGE",                  has_aux=True,  beta=0.25, rand_affinity=False),
    "C3_beta05":  dict(arch="AFAGE",                  has_aux=True,  beta=0.5,  rand_affinity=False),
    "C3_beta075": dict(arch="AFAGE",                  has_aux=True,  beta=0.75, rand_affinity=False),
    "C3_beta1":   dict(arch="AFAGE",                  has_aux=True,  beta=1.0,  rand_affinity=False),
    "C3_oracle":  dict(arch="AFAGEOracle",            has_aux=True,  beta=1.0,  rand_affinity=False),
    "C3_static":  dict(arch="AFAGEStaticAffinity",    has_aux=False, beta=None, rand_affinity=False),
    "C3_vit5bb":  dict(arch="AFAGE",                  has_aux=True,  beta=0.0,  rand_affinity=False,
                       backbones=CNN_BACKBONES + [VIT_BACKBONE], affinity="five_backbone"),
}

CONDITION_LABELS = {
    "C1": "C1", "C1_aux": "C1+Aux", "C2_gate": "C2-Gate", "C3_rand": "C3-Rand",
    "C3_beta0": "C3 β=0", "C3_beta025": "C3 β=0.25", "C3_beta05": "C3 β=0.5",
    "C3_beta075": "C3 β=0.75", "C3_beta1": "C3 β=1", "C3_oracle": "C3-Oracle",
    "C3_static": "C3-Static", "C3_vit5bb": "C3+ViT-B/16",
}
BACKBONE_LABELS = {
    "resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0", "densenet121": "DenseNet-121",
    "convnext_tiny": "ConvNeXt-Tiny", VIT_BACKBONE: "ViT-B/16",
}


def condition_backbones(key):
    return CONDITIONS[key].get("backbones", CNN_BACKBONES)


def affinity_for(key, affinity_4, affinity_5=None):
    """Affinity matrix used to initialise condition `key` (rows = backbones, cols = Mass, Calc)."""
    if CONDITIONS[key].get("affinity") == "five_backbone":
        if affinity_5 is None:
            raise ValueError(f"{key} needs the five-backbone affinity matrix")
        return np.asarray(affinity_5)
    return np.asarray(affinity_4)
