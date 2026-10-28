"""External validation on INbreast (Table 2, Table S6).

Masses are the primary external test. Radiologist-drawn calcification clusters are
reported descriptively, because all but one are malignant (AUC undefined).
"""
import numpy as np
import pandas as pd

from ..config import CNN_BACKBONES, SEEDS
from ..stats import Bootstrap, auc, summarise, threshold_for_sensitivity

NEXT_SEED = {42: 123, 123: 456, 456: 42}
MODELS = ["C1", "C1_aux", "C3_beta0", "C3_beta1", "C3_oracle", "C3_rand", "C3_beta05", "C3_static", "C3_vit5bb",
          "C2_gate", "U4", "U8"] + [f"S_Mass_{b}" for b in CNN_BACKBONES]
COMPARISONS = [("C3_beta0", "C1"), ("C3_beta1", "C1"), ("C1_aux", "C1"), ("C3_beta0", "U8"), ("C3_beta1", "U8"),
               ("C1", "U8")]


def _filename(paths):
    return paths.astype(str).str.split("/").str[-1]


def _scores(pred, seed):
    w = pred[pred.seed == seed].pivot_table(index="roi", columns="model_key", values="prob_malignant")
    out = pd.DataFrame(index=w.index)
    out["U4"] = w[[f"U_{b}" for b in CNN_BACKBONES]].mean(axis=1)
    for k in MODELS:
        if k in w and k not in out:
            out[k] = w[k]
    return out


def mass_results(inbreast_predictions, inbreast_rois, n_replicates=1000):
    """AUC of every model on INbreast masses and paired comparisons (seed x patient bootstrap)."""
    pred = inbreast_predictions[inbreast_predictions.split == "inbreast"].copy()
    pred["roi"] = _filename(pred.roi_path)
    scores = {s: _scores(pred, s) for s in SEEDS}
    for s in SEEDS:
        scores[s]["U8"] = (scores[s]["U4"] + scores[NEXT_SEED[s]]["U4"]) / 2
    rois = inbreast_rois.assign(roi=_filename(inbreast_rois.roi_path))
    mass = rois[rois.category == "Mass"].reset_index(drop=True)
    y = mass.label_idx.values
    keys = [k for k in MODELS if k in scores[SEEDS[0]]]
    v = {s: scores[s].loc[mass.roi] for s in SEEDS}

    def stat(rows, seeds):
        o = {k: np.mean([auc(y[rows], v[s][k].values[rows]) for s in seeds]) for k in keys}
        o.update({f"{a} − {b}": o[a] - o[b] for a, b in COMPARISONS})
        return o

    res = summarise(*Bootstrap(mass[["roi", "patient_id", "label_idx"]].assign(category="Mass"), n_replicates).run(stat))
    is_diff = res.quantity.str.contains(" − ")
    return res[~is_diff].reset_index(drop=True), res[is_diff].reset_index(drop=True)


def clusters_descriptive(inbreast_predictions, cbis_predictions,
                         models=("C1", "C3_beta0", "C3_beta1", "U_efficientnet_b0",
                                 "S_Calcification_efficientnet_b0", "S_Calcification_densenet121")):
    """Mean malignancy score and detection of malignant clusters at each model's CBIS-DDSM 80% threshold."""
    ib = inbreast_predictions[(inbreast_predictions.split == "inbreast")]
    val = cbis_predictions[cbis_predictions.split == "val"]
    rows = []
    for k in models:
        for s in SEEDS:
            v = val[(val.model_key == k) & (val.seed == s)]
            thr = threshold_for_sensitivity(v.label_idx.values, v.prob_malignant.values, 0.80)
            d = ib[(ib.model_key == k) & (ib.seed == s) & (ib.category == "Calcification")]
            rows.append({"model": k, "seed": s, "n_clusters": len(d), "malignant": int(d.label_idx.sum()),
                         "mean_prob_malignant": d.prob_malignant.mean(),
                         "share_malignant_detected_at_cbis_80pct_threshold": (d[d.label_idx == 1].prob_malignant >= thr).mean()})
    return pd.DataFrame(rows).groupby("model").mean(numeric_only=True).drop(columns="seed")
