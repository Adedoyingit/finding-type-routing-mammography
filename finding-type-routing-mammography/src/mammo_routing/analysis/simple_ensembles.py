"""AFAGE and C1 against simple averages of independently trained backbones (Fig. 4b, Table 2, Table S5).

U4: mean malignancy probability of the four single-backbone models of one seed.
U8: mean of the U4 ensembles of that seed and the next seed (eight models).
"""
import numpy as np

from ..config import CNN_BACKBONES, SEEDS
from ..results_io import wide
from ..stats import Bootstrap, auc, equivalent, minimum_detectable_effect, summarise

NEXT_SEED = {42: 123, 123: 456, 456: 42}
COMPARISONS = [("C3_beta0", "U4"), ("C3_beta1", "U4"), ("C3_beta0", "U8"), ("C3_beta1", "U8"),
               ("C1", "U4"), ("C1", "U8")]


def simple_ensemble_scores(w, seed):
    u4 = w[[(f"U_{b}", seed) for b in CNN_BACKBONES]].mean(axis=1).values
    u4_next = w[[(f"U_{b}", NEXT_SEED[seed]) for b in CNN_BACKBONES]].mean(axis=1).values
    return {"U4": u4, "U8": (u4 + u4_next) / 2}


def compare(predictions, n_replicates=1000):
    """AUCs (overall, mass, calcification) and paired differences, for one evaluation setting."""
    wt, mt = wide(predictions, "test")
    y, cat = mt.label_idx.values, mt.category.values
    scores = {s: {**simple_ensemble_scores(wt, s), **{k: wt[(k, s)].values for k in ("C1", "C3_beta0", "C3_beta1")}}
              for s in SEEDS}

    def stat(rows, seeds):
        out = {}
        for subset, mask in (("all", None), ("mass", cat == "Mass"), ("calc", cat == "Calcification")):
            r = rows if mask is None else rows[mask[rows]]
            for k in ("U4", "U8", "C1", "C3_beta0", "C3_beta1"):
                out[f"AUC|{k}|{subset}"] = np.mean([auc(y[r], scores[s][k][r]) for s in seeds])
        for a, b in COMPARISONS:
            out[f"D|{a} − {b}|all"] = out[f"AUC|{a}|all"] - out[f"AUC|{b}|all"]
        return out

    res = summarise(*Bootstrap(mt, n_replicates).run(stat))
    res[["kind", "model", "subset"]] = res.quantity.str.split("|", expand=True)
    res["MDE"] = minimum_detectable_effect(res.se)
    for m in (0.02, 0.03, 0.05):
        res[f"equiv_{m}"] = [equivalent(a, b, m) for a, b in zip(res.ci90_lo, res.ci90_hi)]
    return res
