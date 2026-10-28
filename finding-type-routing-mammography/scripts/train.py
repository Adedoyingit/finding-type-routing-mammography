"""Train the models of the paper (three seeds each).

Families
--------
single       four single-backbone models trained on all lesions (simple-ensemble members)
specialists  five backbones x two finding types; also writes the affinity matrices
ensembles    the twelve ensemble conditions (needs the affinity matrices from 'specialists')

Examples
--------
    python scripts/train.py single      --data data/cbis --out runs/single_backbone
    python scripts/train.py specialists --data data/cbis --out runs/specialists
    python scripts/train.py ensembles   --data data/cbis --out runs/ensembles --affinity runs/specialists
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing the package
import argparse
import json

import numpy as np
import pandas as pd

from mammo_routing.config import CNN_BACKBONES, CONDITIONS, FINDING_TYPES, VIT_BACKBONE, TrainConfig, affinity_for
from mammo_routing.data import load_splits
from mammo_routing.models import build_condition, build_single
from mammo_routing.reproducibility import environment
from mammo_routing.stats import auc
from mammo_routing.training import run_jobs


def affinity_matrices(run_dir, val_df):
    """Mean validation AUC of each specialist on its own finding type (rows = backbones; cols = Mass, Calc)."""
    preds = pd.concat([pd.read_csv(f) for f in sorted((run_dir / "predictions").glob("*.csv.gz"))])
    preds = preds[preds.split == "val"]
    table = {}
    for b in CNN_BACKBONES + [VIT_BACKBONE]:
        for f in FINDING_TYPES:
            d = preds[(preds.model_key == f"S_{f}_{b}") & (preds.category == f)]
            table[(b, f)] = np.mean([auc(g.label_idx, g.prob_malignant) for _, g in d.groupby("seed")])
    a4 = np.array([[table[(b, f)] for f in FINDING_TYPES] for b in CNN_BACKBONES])
    a5 = np.array([[table[(b, f)] for f in FINDING_TYPES] for b in CNN_BACKBONES + [VIT_BACKBONE]])
    return a4, a5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family", choices=["single", "specialists", "ensembles"])
    ap.add_argument("--data", type=Path, required=True, help="folder with train/val/test_df.csv and cbis_rois/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--affinity", type=Path, help="specialists run folder (for 'ensembles')")
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS), help="subset of ensemble conditions")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--resume", nargs="*", type=Path, default=[], help="earlier run folders to skip finished jobs")
    args = ap.parse_args()

    cfg = TrainConfig()
    seeds = args.seeds or cfg.seeds
    train_df, val_df, test_df = load_splits(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "environment.json").write_text(json.dumps(environment(), indent=2))

    if args.family == "single":
        jobs = [dict(key=f"U_{b}", backbone=b, seed=s, train_df=train_df, val_df=val_df, has_aux=False, oracle=False)
                for b in CNN_BACKBONES for s in seeds]
        build = lambda job: build_single(job["backbone"])
    elif args.family == "specialists":
        jobs = []
        for f in FINDING_TYPES:
            tr, va = (d[d.category == f].reset_index(drop=True) for d in (train_df, val_df))
            for b in CNN_BACKBONES + [VIT_BACKBONE]:
                lr = cfg.vit_learning_rate if b == VIT_BACKBONE else cfg.learning_rate
                jobs += [dict(key=f"S_{f}_{b}", backbone=b, seed=s, train_df=tr, val_df=va, has_aux=False,
                              oracle=False, lr=lr) for s in seeds]
        build = lambda job: build_single(job["backbone"])
    else:
        if args.affinity is None:
            ap.error("'ensembles' needs --affinity <specialists run folder>")
        a4, a5 = np.load(args.affinity / "affinity_matrix.npy"), np.load(args.affinity / "affinity_matrix_5bb.npy")
        np.save(args.out / "affinity_matrix_used.npy", a4)
        jobs = [dict(key=k, seed=s, train_df=train_df, val_df=val_df, has_aux=CONDITIONS[k]["has_aux"],
                     oracle=CONDITIONS[k]["arch"] == "AFAGEOracle",
                     backbone_names=CONDITIONS[k].get("backbones", CNN_BACKBONES))
                for k in args.conditions for s in seeds]
        build = lambda job: build_condition(job["key"], affinity_for(job["key"], a4, a5))

    run_jobs(jobs, build, val_df, test_df, args.out, cfg, resume_dirs=args.resume)
    if args.family == "specialists":
        a4, a5 = affinity_matrices(args.out, val_df)
        np.save(args.out / "affinity_matrix.npy", a4)
        np.save(args.out / "affinity_matrix_5bb.npy", a5)
        print("Affinity matrix (rows = backbones; cols = Mass, Calcification):\n", np.round(a4, 4))


if __name__ == "__main__":
    main()
