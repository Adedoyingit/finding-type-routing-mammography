"""Metrics and the paired seed x patient bootstrap used for every interval in the paper."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import SEEDS

TARGET_SENSITIVITIES = (0.80, 0.85, 0.90)


def auc(y, s):
    """ROC AUC via the Mann-Whitney statistic (average ranks for ties; equals sklearn)."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def partial_auc_high_sensitivity(y, s, min_sensitivity=0.80):
    """Standardised partial AUC over sensitivity >= min_sensitivity (McClish, Eq. 9).

    Classes are swapped so that the high-sensitivity region becomes the low-FPR region;
    the result equals sklearn.metrics.roc_auc_score(1 - y, -s, max_fpr=1 - min_sensitivity).
    """
    y = 1 - np.asarray(y)
    s = -np.asarray(s, float)
    max_fpr = 1 - min_sensitivity
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    thr_idx = np.r_[np.where(np.diff(s))[0], y.size - 1]
    tps = np.cumsum(y)[thr_idx]
    fps = 1 + thr_idx - tps
    tpr = np.r_[0, tps / tps[-1]]
    fpr = np.r_[0, fps / fps[-1]]
    stop = np.searchsorted(fpr, max_fpr, "right")
    tpr = np.append(tpr[:stop], np.interp(max_fpr, [fpr[stop - 1], fpr[stop]], [tpr[stop - 1], tpr[stop]]))
    fpr = np.append(fpr[:stop], max_fpr)
    area = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
    min_area = 0.5 * max_fpr ** 2
    return 0.5 * (1 + (area - min_area) / (max_fpr - min_area))


def threshold_for_sensitivity(y, s, target):
    """Highest threshold reaching at least `target` sensitivity on (y, s); chosen on validation."""
    pos = np.sort(np.asarray(s)[np.asarray(y) == 1])[::-1]
    return pos[int(np.ceil(target * len(pos))) - 1]


def operating_point(y, s, threshold):
    y = np.asarray(y)
    pred = np.asarray(s) >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return {"fp": fp, "sens": tp / max(1, (y == 1).sum()), "spec": 1 - fp / max(1, (y == 0).sum())}


class Bootstrap:
    """Paired hierarchical bootstrap over training seeds and patients.

    Each replicate resamples the training seeds with replacement and, independently,
    the patients with replacement (keeping all ROIs of a sampled patient). Statistics
    are computed on identical replicates for all models, so differences are paired.
    """

    def __init__(self, meta, n_replicates=1000, random_state=2026, seeds=SEEDS):
        self.meta = meta
        self.seeds = list(seeds)
        patients = meta["patient_id"].values
        self.patients = np.unique(patients)
        self.rows_of = {p: np.where(patients == p)[0] for p in self.patients}
        rng = np.random.default_rng(random_state)
        labels = meta["label_idx"].values
        self.draws = []
        while len(self.draws) < n_replicates:
            seeds_b = list(rng.choice(self.seeds, len(self.seeds), replace=True))
            rows = np.concatenate([self.rows_of[p] for p in rng.choice(self.patients, len(self.patients))])
            if 0 < labels[rows].sum() < len(rows):          # both classes present
                self.draws.append((seeds_b, rows))
        self.all_rows = np.arange(len(meta))

    def run(self, statistic):
        """statistic(rows, seeds) -> dict of scalars. Returns (point estimates, replicate table)."""
        point = statistic(self.all_rows, self.seeds)
        replicates = pd.DataFrame([statistic(rows, seeds) for seeds, rows in self.draws])
        return point, replicates


def summarise(point, replicates, names=None):
    """Point estimate, 95% percentile interval, 90% interval, s.e. and two-sided bootstrap p."""
    rows = []
    for k in names or point.keys():
        r = replicates[k].dropna().values
        lo, hi = np.percentile(r, [2.5, 97.5])
        lo90, hi90 = np.percentile(r, [5, 95])
        p = min(1.0, 2 * min((r <= 0).mean(), (r >= 0).mean()))
        rows.append({"quantity": k, "estimate": point[k], "ci_lo": lo, "ci_hi": hi,
                     "ci90_lo": lo90, "ci90_hi": hi90, "se": r.std(ddof=1), "p_boot": p})
    return pd.DataFrame(rows)


def holm(p_values):
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (len(p) - i) * p[j])
        adjusted[j] = min(1.0, running)
    return adjusted


def equivalent(ci90_lo, ci90_hi, margin):
    """Two one-sided tests at alpha = 0.05: the 90% interval lies inside (-margin, margin)."""
    return bool(ci90_lo > -margin and ci90_hi < margin)


def minimum_detectable_effect(se):
    """Effect detectable with 80% power at two-sided alpha = 0.05 (2.8 x s.e.)."""
    return 2.8 * se
