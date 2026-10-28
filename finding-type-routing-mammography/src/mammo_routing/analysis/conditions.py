"""Ensemble conditions against the jointly trained baseline C1 (Figs 3-5, Table 2, Tables S2-S3)."""
import numpy as np
import pandas as pd

from ..config import CONDITIONS, SEEDS
from ..results_io import wide
from ..stats import (TARGET_SENSITIVITIES, Bootstrap, auc, equivalent, holm, minimum_detectable_effect,
                     operating_point, partial_auc_high_sensitivity, summarise, threshold_for_sensitivity)

PRIMARY = ("C3_beta0", "C3_beta1")


def available_conditions(predictions):
    present = set(predictions.model_key)
    return [c for c in CONDITIONS if c in present]


def condition_auc(predictions):
    """Test AUC (overall, mass, calcification) and partial AUC per condition and seed."""
    wt, mt = wide(predictions, "test")
    y, cat = mt.label_idx.values, mt.category.values
    rows = []
    for c in available_conditions(predictions):
        for s in SEEDS:
            sc = wt[(c, s)].values
            rows.append({"condition": c, "seed": s, "auc": auc(y, sc),
                         "auc_mass": auc(y[cat == "Mass"], sc[cat == "Mass"]),
                         "auc_calc": auc(y[cat == "Calcification"], sc[cat == "Calcification"]),
                         "pauc80": partial_auc_high_sensitivity(y, sc)})
    return pd.DataFrame(rows)


def thresholds(predictions, conditions):
    """Validation-frozen thresholds for each condition, seed and target sensitivity."""
    wv, mv = wide(predictions, "val")
    yv = mv.label_idx.values
    return {(c, s, t): threshold_for_sensitivity(yv, wv[(c, s)].values, t)
            for c in conditions for s in SEEDS for t in TARGET_SENSITIVITIES}


def comparisons_vs_c1(predictions, n_replicates=1000):
    """Paired differences from C1 in AUC, partial AUC and false positives at fixed sensitivity."""
    conds = available_conditions(predictions)
    others = [c for c in conds if c != "C1"]
    wt, mt = wide(predictions, "test")
    yt = mt.label_idx.values
    thr = thresholds(predictions, conds)

    def stat(rows, seeds):
        y = yt[rows]

        def over_seeds(c, metric):
            return np.mean([metric(wt[(c, s)].values[rows], s, c) for s in seeds])

        def auc_of(x, s, c):
            return auc(y, x)

        def pauc_of(x, s, c):
            return partial_auc_high_sensitivity(y, x)

        def fp_at(t):
            return lambda x, s, c: operating_point(y, x, thr[(c, s, t)])["fp"]

        out = {}
        for c in others:
            out[f"{c}|dAUC"] = over_seeds(c, auc_of) - over_seeds("C1", auc_of)
            out[f"{c}|dpAUC"] = over_seeds(c, pauc_of) - over_seeds("C1", pauc_of)
            for t in TARGET_SENSITIVITIES:
                out[f"{c}|dFP{int(t * 100)}"] = over_seeds(c, fp_at(t)) - over_seeds("C1", fp_at(t))
        return out

    point, reps = Bootstrap(mt, n_replicates).run(stat)
    res = summarise(point, reps)
    res[["condition", "endpoint"]] = res.quantity.str.split("|", expand=True)
    res["primary"] = res.condition.isin(PRIMARY)
    res["MDE"] = minimum_detectable_effect(res.se)
    for m in (0.02, 0.03, 0.05):
        res[f"equiv_{m}"] = [equivalent(a, b, m) for a, b in zip(res.ci90_lo, res.ci90_hi)]
    is_auc = res.endpoint == "dAUC"
    res["p_adj"] = np.nan
    prim = is_auc & res.primary
    res.loc[prim, "p_adj"] = np.minimum(1, res.loc[prim, "p_boot"] * len(PRIMARY))        # Bonferroni
    expl = is_auc & ~res.primary
    if expl.any():
        res.loc[expl, "p_adj"] = holm(res.loc[expl, "p_boot"].values)
    return res


def operating_points(predictions):
    """False positives, achieved sensitivity and specificity on test at validation-frozen thresholds."""
    conds = available_conditions(predictions)
    wt, mt = wide(predictions, "test")
    yt = mt.label_idx.values
    thr = thresholds(predictions, conds)
    return pd.DataFrame([{"condition": c, "seed": s, "target": t, **operating_point(yt, wt[(c, s)].values, thr[(c, s, t)])}
                         for c in conds for s in SEEDS for t in TARGET_SENSITIVITIES])


def unchanged_roi_sensitivity(predictions, n_replicates=1000):
    """Primary comparisons restricted to test ROIs whose crop was not affected by the extraction fault."""
    wt, mt = wide(predictions, "test")
    keep = ~mt.v1_crop_was_wrong.values.astype(bool)
    mk, wk, yk = mt[keep], wt[keep], mt.label_idx.values[keep]

    def stat(rows, seeds):
        base = np.mean([auc(yk[rows], wk[("C1", s)].values[rows]) for s in seeds])
        return {c: np.mean([auc(yk[rows], wk[(c, s)].values[rows]) for s in seeds]) - base for c in PRIMARY}

    out = summarise(*Bootstrap(mk, n_replicates).run(stat))
    out["n_test"] = int(keep.sum())
    return out
