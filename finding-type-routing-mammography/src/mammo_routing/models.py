"""Model architectures.

Single-backbone models
    SingleBackbone          one backbone + classification head (unified models and specialists)

Ensembles over four CNN backbones (features concatenated, 5,120 dimensions)
    UnifiedEnsemble         C1: jointly trained ensemble, no finding-type information
    UnifiedEnsembleWithAux  C1+Aux: C1 with an auxiliary finding-type loss
    AFAGEGateOnly           C2-Gate: adaptive gate on backbone features only
    AFAGE                   C3: auxiliary predictor + learnable affinity matrix + adaptive gate
    AFAGEOracle             C3-Oracle: AFAGE routed by the true finding type
    AFAGEStaticAffinity     C3-Static: weights fixed to softmax(A) p, no gate

Attribute names and construction order are part of the checkpoint format
(state_dict keys and random initialisation); do not rename or reorder them.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CONDITIONS, VIT_BACKBONE, affinity_for, condition_backbones


def _backbones(names, pretrained):
    return nn.ModuleList([timm.create_model(n, pretrained=pretrained, num_classes=0, global_pool="avg")
                          for n in names])


def _ensemble_head(in_dim, num_classes=2):
    return nn.Sequential(nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.4),
                         nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
                         nn.Linear(256, num_classes))


def _finding_predictor(in_dim, num_types=2):
    return nn.Sequential(nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(0.3),
                         nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, num_types))


def _projectors(dims):
    return nn.ModuleList([nn.Sequential(nn.Linear(d, 128), nn.LayerNorm(128), nn.ReLU(inplace=True))
                          for d in dims])


def _gate(in_dim, num_backbones):
    return nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
                         nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, num_backbones))


def _fuse(weights, features):
    """Weighted concatenation [w1 f1 || ... || wB fB] (Eq. 5)."""
    return torch.cat([weights[:, i:i + 1] * f for i, f in enumerate(features)], dim=1)


class SingleBackbone(nn.Module):
    def __init__(self, backbone, num_classes=2, pretrained=True, pool="avg"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool=pool)
        d = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes))

    def forward(self, x):
        f = self.backbone(x)
        return {"main_logits": self.classifier(f), "features": f}


class UnifiedEnsemble(nn.Module):
    def __init__(self, backbone_names, num_classes=2, pretrained=True):
        super().__init__()
        self.backbones = _backbones(backbone_names, pretrained)
        self.total_feat = sum(b.num_features for b in self.backbones)
        self.classifier = _ensemble_head(self.total_feat, num_classes)

    def forward(self, x):
        return {"main_logits": self.classifier(torch.cat([b(x) for b in self.backbones], dim=1))}


class UnifiedEnsembleWithAux(nn.Module):
    def __init__(self, backbone_names, num_classes=2, num_finding_types=2, pretrained=True):
        super().__init__()
        self.backbones = _backbones(backbone_names, pretrained)
        self.total_feat = sum(b.num_features for b in self.backbones)
        self.finding_predictor = _finding_predictor(self.total_feat, num_finding_types)
        self.classifier = _ensemble_head(self.total_feat, num_classes)

    def forward(self, x):
        f = torch.cat([b(x) for b in self.backbones], dim=1)
        finding_logits = self.finding_predictor(f)
        return {"main_logits": self.classifier(f), "finding_logits": finding_logits,
                "finding_probs": F.softmax(finding_logits, dim=1)}


class AFAGEGateOnly(nn.Module):
    def __init__(self, backbone_names, num_classes=2, pretrained=True):
        super().__init__()
        self.backbones = _backbones(backbone_names, pretrained)
        dims = [b.num_features for b in self.backbones]
        self.total_feat = sum(dims)
        self.projectors = _projectors(dims)
        self.gate = _gate(128 * len(backbone_names), len(backbone_names))
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.classifier = _ensemble_head(self.total_feat, num_classes)

    def forward(self, x):
        feats = [b(x) for b in self.backbones]
        g_raw = self.gate(torch.cat([p(f) for p, f in zip(self.projectors, feats)], dim=1))
        weights = F.softmax(g_raw / self.temperature.clamp(0.5, 3.0), dim=1)
        return {"main_logits": self.classifier(_fuse(weights, feats)), "fusion_weights": weights}


class AFAGE(nn.Module):
    """Affinity-guided adaptive ensemble (Eqs. 1-5).

    beta = 0: the affinity matrix A does not enter the weights; the predicted finding
    type reaches the gate through its input. beta > 0 adds the explicit prior softmax(A) p.
    """

    def __init__(self, backbone_names, num_classes=2, num_finding_types=2, pretrained=True,
                 affinity_matrix=None, beta=0.0, rand_affinity_init=False):
        super().__init__()
        self.beta = beta
        self.backbones = _backbones(backbone_names, pretrained)
        dims = [b.num_features for b in self.backbones]
        self.total_feat = sum(dims)
        self.projectors = _projectors(dims)
        self.finding_predictor = _finding_predictor(self.total_feat, num_finding_types)
        if rand_affinity_init or affinity_matrix is None:
            a_init = torch.rand(len(backbone_names), num_finding_types)
        else:
            a_init = torch.tensor(affinity_matrix, dtype=torch.float32)
        self.affinity_matrix = nn.Parameter(a_init)
        self.gate = _gate(num_finding_types + 128 * len(backbone_names), len(backbone_names))
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.classifier = _ensemble_head(self.total_feat, num_classes)

    def _weights(self, finding_probs, feats):
        g_aff = finding_probs @ F.softmax(self.affinity_matrix, dim=0).T
        gate_in = torch.cat([finding_probs] + [p(f) for p, f in zip(self.projectors, feats)], dim=1)
        g_raw = self.gate(gate_in)
        return F.softmax((g_raw + self.beta * g_aff) / self.temperature.clamp(0.5, 3.0), dim=1)

    def forward(self, x):
        feats = [b(x) for b in self.backbones]
        finding_logits = self.finding_predictor(torch.cat(feats, dim=1))
        finding_probs = F.softmax(finding_logits, dim=1)
        weights = self._weights(finding_probs, feats)
        return {"main_logits": self.classifier(_fuse(weights, feats)), "finding_logits": finding_logits,
                "finding_probs": finding_probs, "fusion_weights": weights}


class AFAGEOracle(nn.Module):
    """AFAGE whose gate and prior receive the true finding type instead of the prediction."""

    def __init__(self, backbone_names, num_classes=2, num_finding_types=2, pretrained=True,
                 affinity_matrix=None, beta=1.0):
        super().__init__()
        self.beta = beta
        self.num_finding_types = num_finding_types
        self.backbones = _backbones(backbone_names, pretrained)
        dims = [b.num_features for b in self.backbones]
        self.total_feat = sum(dims)
        self.projectors = _projectors(dims)
        self.finding_predictor = _finding_predictor(self.total_feat, num_finding_types)
        a_init = (torch.tensor(affinity_matrix, dtype=torch.float32) if affinity_matrix is not None
                  else torch.rand(len(backbone_names), num_finding_types))
        self.affinity_matrix = nn.Parameter(a_init)
        self.gate = _gate(num_finding_types + 128 * len(backbone_names), len(backbone_names))
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.classifier = _ensemble_head(self.total_feat, num_classes)

    def forward(self, x, true_category_idx=None):
        feats = [b(x) for b in self.backbones]
        finding_logits = self.finding_predictor(torch.cat(feats, dim=1))
        if true_category_idx is not None:
            finding_probs = F.one_hot(true_category_idx, self.num_finding_types).float()
        else:
            finding_probs = F.softmax(finding_logits, dim=1)
        g_aff = finding_probs @ F.softmax(self.affinity_matrix, dim=0).T
        gate_in = torch.cat([finding_probs] + [p(f) for p, f in zip(self.projectors, feats)], dim=1)
        weights = F.softmax((self.gate(gate_in) + self.beta * g_aff) / self.temperature.clamp(0.5, 3.0), dim=1)
        return {"main_logits": self.classifier(_fuse(weights, feats)), "finding_logits": finding_logits,
                "finding_probs": finding_probs, "fusion_weights": weights}


class AFAGEStaticAffinity(nn.Module):
    """Backbone weights fixed to softmax(A) p (A frozen), no gate."""

    def __init__(self, backbone_names, num_classes=2, num_finding_types=2, pretrained=True, affinity_matrix=None):
        super().__init__()
        self.backbones = _backbones(backbone_names, pretrained)
        self.total_feat = sum(b.num_features for b in self.backbones)
        self.finding_predictor = _finding_predictor(self.total_feat, num_finding_types)
        self.register_buffer("affinity_matrix", torch.tensor(affinity_matrix, dtype=torch.float32))
        self.classifier = _ensemble_head(self.total_feat, num_classes)

    def forward(self, x):
        feats = [b(x) for b in self.backbones]
        finding_logits = self.finding_predictor(torch.cat(feats, dim=1))
        finding_probs = F.softmax(finding_logits, dim=1)
        weights = finding_probs @ F.softmax(self.affinity_matrix, dim=0).T
        return {"main_logits": self.classifier(_fuse(weights, feats)), "finding_logits": finding_logits,
                "finding_probs": finding_probs, "fusion_weights": weights}


def build_condition(key, affinity_matrix, pretrained=True):
    """Instantiate ensemble condition `key` (see config.CONDITIONS)."""
    c = CONDITIONS[key]
    names = condition_backbones(key)
    arch = c["arch"]
    if arch == "UnifiedEnsemble":
        return UnifiedEnsemble(names, 2, pretrained)
    if arch == "UnifiedEnsembleWithAux":
        return UnifiedEnsembleWithAux(names, 2, 2, pretrained)
    if arch == "AFAGEGateOnly":
        return AFAGEGateOnly(names, 2, pretrained)
    if arch == "AFAGE":
        return AFAGE(names, 2, 2, pretrained, affinity_matrix=affinity_matrix, beta=c["beta"],
                     rand_affinity_init=c["rand_affinity"])
    if arch == "AFAGEOracle":
        return AFAGEOracle(names, 2, 2, pretrained, affinity_matrix=affinity_matrix, beta=c["beta"])
    if arch == "AFAGEStaticAffinity":
        return AFAGEStaticAffinity(names, 2, 2, pretrained, affinity_matrix=affinity_matrix)
    raise ValueError(f"unknown architecture {arch}")


def build_single(backbone, pretrained=True):
    """Single-backbone model; ViT-B/16 uses its class token, CNNs average pooling."""
    return SingleBackbone(backbone, 2, pretrained, pool="token" if backbone == VIT_BACKBONE else "avg")


def model_for_key(key, affinity_4=None, affinity_5=None, pretrained=True):
    """Model for a prediction key: 'U_<backbone>', 'S_<finding>_<backbone>' or a condition name."""
    if key.startswith("U_"):
        return build_single(key[2:], pretrained)
    if key.startswith("S_"):
        return build_single(key.split("_", 2)[2], pretrained)
    return build_condition(key, affinity_for(key, affinity_4, affinity_5), pretrained)


