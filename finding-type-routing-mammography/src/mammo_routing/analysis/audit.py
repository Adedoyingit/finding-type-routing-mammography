"""Training audit: completeness, software environment, starting weights and convergence (Methods, Fig. S2)."""
import json
from pathlib import Path

import pandas as pd

EXPECTED_RUNS = 78


def audit(run_root, registries):
    """Summary checks over all training runs below `run_root`."""
    envs = [json.loads(p.read_text()) for p in Path(run_root).rglob("environment.json")]
    env_keys = ["python", "torch", "timm", "albumentations", "opencv", "gpu"]
    same_env = len({tuple(e.get(k) for k in env_keys) for e in envs}) == 1 if envs else None
    reg = registries.copy()
    reg["arch"] = reg.key.str.replace(r"^(U_|S_Mass_|S_Calcification_)", "", regex=True)
    single = reg[reg.key.str.startswith(("U_", "S_"))].groupby("arch").init_fingerprint.nunique()
    ensemble = reg[~reg.key.str.startswith(("U_", "S_"))].groupby("key").init_fingerprint.nunique()
    convergence = (reg.assign(early_peak=reg.best_epoch <= 3)
                      .groupby("arch").agg(runs=("tag", "size"), best_val_auc=("best_val_auc", "mean"),
                                           median_best_epoch=("best_epoch", "median"),
                                           median_epochs=("epochs_run", "median"),
                                           early_peak_runs=("early_peak", "sum"), gpu_minutes=("minutes", "sum")))
    summary = {"runs": int(len(reg)), "expected_runs": EXPECTED_RUNS, "complete": int(len(reg)) == EXPECTED_RUNS,
               "single_environment": same_env,
               "identical_initial_weights": bool((single == 1).all() and (ensemble == 1).all()),
               "gpu_hours": round(reg.minutes.sum() / 60, 1), "early_peak_runs": int((reg.best_epoch <= 3).sum())}
    return summary, convergence.reset_index()


def training_curves(registries):
    """Long table of validation AUC per epoch for every run."""
    rows = [{"tag": r.tag, "key": r.key, "seed": r.seed, **h} for _, r in registries.iterrows() for h in r.history]
    return pd.DataFrame(rows)
