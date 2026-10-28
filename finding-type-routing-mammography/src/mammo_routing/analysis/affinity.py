"""Architecture-finding affinity as a backbone x finding-type interaction (Fig. 2, Table S1)."""
import numpy as np
import pandas as pd

from ..config import CNN_BACKBONES, FINDING_TYPES, SEEDS, VIT_BACKBONE
from ..results_io import wide
from ..stats import Bootstrap, auc, holm, summarise

PLANNED = ("densenet121", "resnet50")      # hypothesis: DenseNet-121 relatively better on calcifications


def specialist_auc(predictions):
    """AUC of each specialist on its own finding type, validation and test (mean and s.d. over seeds)."""
    rows = []
    for split in ("val", "test"):
        w, m = wide(predictions, split)
        y, cat = m.label_idx.values, m.category.values
        for b in CNN_BACKBONES + [VIT_BACKBONE]:
            for f in FINDING_TYPES:
                mask = cat == f
                vals = [auc(y[mask], w[(f"S_{f}_{b}", s)].values[mask]) for s in SEEDS]
                rows.append({"split": split, "backbone": b, "finding": f, "mean": np.mean(vals), "sd": np.std(vals, ddof=1)})
    return pd.DataFrame(rows)


def cross_finding(predictions):
    """Test AUC of every specialist on its own and on the other finding type."""
    w, m = wide(predictions, "test")
    y, cat = m.label_idx.values, m.category.values
    rows = [{"specialist": f"{f} ({b})", "evaluated_on": g,
             "auc": np.mean([auc(y[cat == g], w[(f"S_{f}_{b}", s)].values[cat == g]) for s in SEEDS])}
            for b in CNN_BACKBONES + [VIT_BACKBONE] for f in FINDING_TYPES for g in FINDING_TYPES]
    return pd.DataFrame(rows).pivot(index="specialist", columns="evaluated_on", values="auc")


def interaction(predictions, n_replicates=1000):
    """Interaction terms (Eq. 7) and pairwise contrasts (Eq. 8) with bootstrap intervals.

    Returns (interaction_terms, contrasts). Contrasts are (A - B)_calc - (A - B)_mass; the
    planned DenseNet-121 vs ResNet-50 contrast is pre-specified, the others are Holm-corrected.
    """
    w, m = wide(predictions, "test")
    y, cat = m.label_idx.values, m.category.values
    pairs = [(a, b) for i, a in enumerate(CNN_BACKBONES) for b in CNN_BACKBONES[i + 1:]]

    def stat(rows, seeds):
        auc_bf = {}
        for b in CNN_BACKBONES:
            for f in FINDING_TYPES:
                r = rows[(cat == f)[rows]]
                auc_bf[(b, f)] = np.mean([auc(y[r], w[(f"S_{f}_{b}", s)].values[r]) for s in seeds])
        mat = np.array([[auc_bf[(b, f)] for f in FINDING_TYPES] for b in CNN_BACKBONES])
        terms = mat - mat.mean(0, keepdims=True) - mat.mean(1, keepdims=True) + mat.mean()   # double centring
        out = {f"inter|{b}|{f}": terms[i, j] for i, b in enumerate(CNN_BACKBONES) for j, f in enumerate(FINDING_TYPES)}
        for a, b in pairs:
            out[f"contrast|{a} vs {b}"] = ((auc_bf[(a, "Calcification")] - auc_bf[(b, "Calcification")])
                                           - (auc_bf[(a, "Mass")] - auc_bf[(b, "Mass")]))
        return out

    res = summarise(*Bootstrap(m, n_replicates).run(stat))
    terms = res[res.quantity.str.startswith("inter|")].copy()
    terms[["_", "backbone", "finding"]] = terms.quantity.str.split("|", expand=True)
    contrasts = res[res.quantity.str.startswith("contrast|")].copy()
    contrasts["pair"] = contrasts.quantity.str.replace("contrast|", "", regex=False)
    contrasts["planned"] = contrasts.pair.isin([f"{PLANNED[0]} vs {PLANNED[1]}", f"{PLANNED[1]} vs {PLANNED[0]}"])
    contrasts["p_holm_exploratory"] = np.nan
    contrasts.loc[~contrasts.planned, "p_holm_exploratory"] = holm(contrasts.loc[~contrasts.planned, "p_boot"].values)
    return terms.drop(columns="_"), contrasts
