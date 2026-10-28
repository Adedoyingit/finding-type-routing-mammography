"""Run every statistical analysis in the paper from the released per-ROI predictions (CPU only).

Example
-------
    python scripts/run_analyses.py --results results --out results/analysis
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing the package
import argparse
import json

import numpy as np
import pandas as pd

from mammo_routing.analysis import affinity, audit, conditions, external, mechanism, simple_ensembles
from mammo_routing.results_io import load_predictions, load_registries

SETTINGS = {"clip4.0": "clip limit 4.0 (primary)", "clip2.5": "clip limit 2.5", "clip_average": "average over clip limits 1-4"}


def save(df, path, index=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"), help="folder with predictions/ and data/")
    ap.add_argument("--out", type=Path, default=Path("results/analysis"))
    ap.add_argument("--replicates", type=int, default=1000, help="bootstrap replicates")
    args = ap.parse_args()
    pred_root, out, B = args.results / "predictions", args.out, args.replicates

    cbis = load_predictions(pred_root / "cbis" / "clip4.0")
    registries = load_registries(pred_root / "cbis" / "clip4.0")

    print("Training audit")
    summary, convergence = audit.audit(pred_root / "cbis" / "clip4.0", registries)
    (out / "audit").mkdir(parents=True, exist_ok=True)
    (out / "audit" / "summary.json").write_text(json.dumps(summary, indent=2))
    save(convergence, out / "audit" / "convergence.csv")
    save(audit.training_curves(registries), out / "audit" / "training_curves.csv")

    print("Architecture-finding affinity")
    save(affinity.specialist_auc(cbis), out / "affinity" / "specialist_auc.csv")
    save(affinity.cross_finding(cbis), out / "affinity" / "cross_finding.csv", index=True)
    terms, contrasts = affinity.interaction(cbis, B)
    save(terms, out / "affinity" / "interaction_terms.csv")
    save(contrasts, out / "affinity" / "contrasts.csv")

    for setting in SETTINGS:
        print(f"Ensemble conditions vs C1 and vs simple averages ({SETTINGS[setting]})")
        p = cbis if setting == "clip4.0" else load_predictions(pred_root / "cbis" / setting)
        d = out / "conditions" / setting
        save(conditions.condition_auc(p), d / "condition_auc.csv")
        save(conditions.comparisons_vs_c1(p, B), d / "comparisons_vs_C1.csv")
        save(conditions.operating_points(p), d / "operating_points.csv")
        save(simple_ensembles.compare(p, B), out / "simple_ensembles" / setting / "afage_vs_simple_ensembles.csv")
    save(conditions.unchanged_roi_sensitivity(cbis, B), out / "conditions" / "clip4.0" / "unchanged_roi_sensitivity.csv")

    print("Mechanism")
    affinity_init = np.load(pred_root / "cbis" / "clip4.0" / "ensembles_part_a" / "affinity_matrix_used.npy")
    save(mechanism.routing_headroom(cbis), out / "mechanism" / "headroom.csv")
    save(mechanism.affinity_prior(registries, affinity_init), out / "mechanism" / "affinity_prior.csv")
    save(mechanism.gate_weights(cbis), out / "mechanism" / "gate_weights.csv")
    save(mechanism.finding_predictor(cbis), out / "mechanism" / "finding_predictor.csv")
    save(mechanism.routing_subsets(cbis), out / "mechanism" / "routing_subsets.csv")

    print("External validation (INbreast)")
    rois = pd.read_csv(args.results / "data" / "inbreast_rois.csv")
    for setting in ("clip4.0", "clip_average"):
        ib = load_predictions(pred_root / "inbreast" / setting)
        auc_table, comparisons = external.mass_results(ib, rois, B)
        save(auc_table, out / "external" / setting / "inbreast_mass_auc.csv")
        save(comparisons, out / "external" / setting / "inbreast_mass_comparisons.csv")
    ib4 = load_predictions(pred_root / "inbreast" / "clip4.0")
    save(external.clusters_descriptive(ib4, cbis), out / "external" / "clip4.0" / "inbreast_clusters.csv", index=True)


if __name__ == "__main__":
    main()
