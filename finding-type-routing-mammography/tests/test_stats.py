"""Checks of the statistical functions against reference implementations."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mammo_routing.stats import Bootstrap, auc, holm, partial_auc_high_sensitivity, threshold_for_sensitivity


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = np.round(rng.random(n) + 0.4 * y, 2)          # rounding creates ties
    return y, s


def test_auc_matches_sklearn():
    for seed in range(5):
        y, s = _data(seed=seed)
        assert np.isclose(auc(y, s), roc_auc_score(y, s))


def test_partial_auc_matches_sklearn():
    for seed in range(5):
        y, s = _data(seed=seed)
        assert np.isclose(partial_auc_high_sensitivity(y, s), roc_auc_score(1 - y, -s, max_fpr=0.2))


def test_threshold_reaches_target_sensitivity():
    y, s = _data()
    for t in (0.8, 0.85, 0.9):
        thr = threshold_for_sensitivity(y, s, t)
        assert (s[y == 1] >= thr).mean() >= t


def test_bootstrap_is_reproducible_and_keeps_patients_together():
    y, _ = _data(200)
    meta = pd.DataFrame({"patient_id": np.repeat(np.arange(100), 2), "label_idx": y})
    a, b = Bootstrap(meta, 50), Bootstrap(meta, 50)
    assert all((ra == rb).all() for (_, ra), (_, rb) in zip(a.draws, b.draws))
    rows = a.draws[0][1]
    pid = meta.patient_id.values
    assert all(pid[rows[i]] == pid[rows[i + 1]] for i in range(0, len(rows), 2))


def test_holm():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
