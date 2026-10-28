"""Prediction and re-scoring of trained checkpoints."""
import numpy as np
import torch
import torch.nn.functional as F

from .reproducibility import DEVICE

META_COLUMNS = ["patient_id", "laterality", "view", "abnormality_id", "category", "category_idx",
                "label", "label_idx", "roi_path", "v1_crop_was_wrong", "is_bwc"]


@torch.no_grad()
def predict(model, loader, oracle=False, device=None):
    """Malignancy logits and probabilities, finding-type probabilities and gate weights."""
    device = device or DEVICE
    model.eval()
    logits, fprobs, weights = [], [], []
    for imgs, _labels, cats in loader:
        imgs, cats = imgs.to(device), cats.to(device)
        out = model(imgs, true_category_idx=cats) if oracle else model(imgs)
        logits.append(out["main_logits"].float().cpu())
        if "finding_logits" in out:
            fprobs.append(F.softmax(out["finding_logits"].float(), dim=1).cpu())
        if "fusion_weights" in out:
            weights.append(out["fusion_weights"].float().cpu())
    lg = torch.cat(logits)
    res = {"logit_benign": lg[:, 0].numpy(), "logit_malignant": lg[:, 1].numpy(),
           "prob_malignant": F.softmax(lg, dim=1)[:, 1].numpy()}
    if fprobs:
        fp = torch.cat(fprobs).numpy()
        res["pf_mass"], res["pf_calc"] = fp[:, 0], fp[:, 1]
    if weights:
        res["_weights"] = torch.cat(weights).numpy()
    return res


def average_predictions(results):
    """Average several predict() outputs (e.g. over CLAHE clip limits)."""
    return {k: np.mean([r[k] for r in results], axis=0) for k in results[0]}


def predictions_frame(df, res, model_key, seed, split, backbone_names=None):
    """One row per ROI with its metadata and the model outputs."""
    cols = [c for c in META_COLUMNS if c in df.columns]
    out = df[cols].copy()
    out.insert(0, "split", split)
    out.insert(0, "seed", seed)
    out.insert(0, "model_key", model_key)
    for k, v in res.items():
        if k == "_weights":
            for i in range(v.shape[1]):
                out[f"w_{backbone_names[i]}"] = v[:, i]
        else:
            out[k] = v
    return out
