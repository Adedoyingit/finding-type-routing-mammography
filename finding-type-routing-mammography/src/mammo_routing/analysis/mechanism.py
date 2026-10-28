"""Why routing has little to exploit: headroom, affinity prior, gate weights, finding-type predictor (Fig. 6, Table S4)."""
import numpy as np
import pandas as pd

from ..config import CNN_BACKBONES, FINDING_TYPES, SEEDS, VIT_BACKBONE
from ..results_io import wide
from ..stats import auc

GATED = ["C2_gate", "C3_rand", "C3_beta0", "C3_beta025", "C3_beta05", "C3_beta075", "C3_beta1", "C3_oracle", "C3_static"]


def _scorers(w, seed):
    d = {"C1": w[("C1", seed)], "U(4)": w[[(f"U_{b}", seed) for b in CNN_BACKBONES]].mean(axis=1)}
    for f in FINDING_TYPES:
        d[f"S_{f}(4)"] = w[[(f"S_{f}_{b}", seed) for b in CNN_BACKBONES]].mean(axis=1)
    for b in CNN_BACKBONES + [VIT_BACKBONE]:
        for f in FINDING_TYPES:
            d[f"S_{f}_{b}"] = w[(f"S_{f}_{b}", seed)]
    for b in CNN_BACKBONES:
        d[f"U_{b}"] = w[(f"U_{b}", seed)]
    return pd.DataFrame(d)


def routing_headroom(predictions):
    """AUC of a composite that uses the best available scorer separately per finding type.

    The scorer for each finding type is chosen on validation (achievable) or on test (an
    optimistic upper bound for any finding-type router).
    """
    wt, mt = wide(predictions, "test")
    wv, mv = wide(predictions, "val")
    yt, ct, yv, cv = mt.label_idx.values, mt.category.values, mv.label_idx.values, mv.category.values
    rows = []
    for s in SEEDS:
        tt, tv = _scorers(wt, s), _scorers(wv, s)
        base = auc(yt, tt["C1"])
        for mode, src, ysrc, csrc in (("oracle (chosen on test)", tt, yt, ct), ("honest (chosen on validation)", tv, yv, cv)):
            pick = {f: max(src.columns, key=lambda k: auc(ysrc[csrc == f], src[k].values[csrc == f])) for f in FINDING_TYPES}
            comp = np.where(ct == "Mass", tt[pick["Mass"]], tt[pick["Calcification"]])
            rows.append({"seed": s, "mode": mode, "pick_mass": pick["Mass"], "pick_calc": pick["Calcification"],
                         "composite_auc": auc(yt, comp), "C1_auc": base, "headroom": auc(yt, comp) - base,
                         "mass_auc": auc(yt[ct == "Mass"], comp[ct == "Mass"]),
                         "calc_auc": auc(yt[ct == "Calcification"], comp[ct == "Calcification"])})
    return pd.DataFrame(rows)


def affinity_prior(registries, affinity_init):
    """Learned affinity matrices and temperatures against the initial matrix."""
    rows = []
    a0 = np.asarray(affinity_init)
    for _, r in registries[registries.get("A_learned").notna()].iterrows():
        a, sm = np.array(r.A_learned), np.array(r.softmax_A)
        rows.append({"condition": r.key, "seed": r.seed, "temperature": r.temperature,
                     "max_abs_change_from_init": float(np.abs(a - a0).max()) if a.shape == a0.shape else np.nan,
                     "max_between_finding_diff_softmaxA": float(np.abs(sm[:, 0] - sm[:, 1]).max())})
    return pd.DataFrame(rows)


def gate_weights(predictions):
    """Mean gate weight per backbone for mass and calcification ROIs, per-ROI s.d., share of variance by finding."""
    t = predictions[(predictions.split == "test") & predictions.model_key.isin(GATED)]
    rows = []
    for (k, s), d in t.groupby(["model_key", "seed"]):
        for b in CNN_BACKBONES:
            col = f"w_{b}"
            m, c = d[d.category == "Mass"][col], d[d.category == "Calcification"][col]
            total = d[col].var()
            between = (len(m) * (m.mean() - d[col].mean()) ** 2 + len(c) * (c.mean() - d[col].mean()) ** 2) / (len(d) - 1)
            rows.append({"condition": k, "seed": s, "backbone": b, "w_mass": m.mean(), "w_calc": c.mean(),
                         "abs_diff": abs(m.mean() - c.mean()), "per_roi_sd": d[col].std(),
                         "share_var_by_finding": between / total if total > 0 else np.nan})
    return pd.DataFrame(rows)


def finding_predictor(predictions):
    """Accuracy and recall of the auxiliary finding-type predictor on test."""
    t = predictions[(predictions.split == "test") & predictions.pf_calc.notna()]
    rows = []
    for (k, s), d in t.groupby(["model_key", "seed"]):
        pred_calc, true_calc = d.pf_calc > 0.5, d.category == "Calcification"
        rows.append({"condition": k, "seed": s, "accuracy": (pred_calc == true_calc).mean(),
                     "mass_recall": (~pred_calc[~true_calc]).mean(), "calc_recall": pred_calc[true_calc].mean()})
    return pd.DataFrame(rows)


def routing_subsets(predictions, conditions=("C3_beta0", "C3_beta1")):
    """AUC of AFAGE and of C1 on the same test ROIs, split by whether routing was correct."""
    wt, mt = wide(predictions, "test")
    yt = mt.label_idx.values
    rows = []
    for k in conditions:
        for s in SEEDS:
            d = predictions[(predictions.split == "test") & (predictions.model_key == k) & (predictions.seed == s)]
            d = d.set_index("roi").loc[mt.index]
            ok = ((d.pf_calc > 0.5) == (d.category == "Calcification")).values
            for name, m in (("correctly routed", ok), ("misrouted", ~ok)):
                rows.append({"condition": k, "seed": s, "subset": name, "n": int(m.sum()),
                             "auc_afage": auc(yt[m], wt[(k, s)].values[m]), "auc_C1": auc(yt[m], wt[("C1", s)].values[m])})
    out = pd.DataFrame(rows)
    out["afage_minus_C1"] = out.auc_afage - out.auc_C1
    return out
