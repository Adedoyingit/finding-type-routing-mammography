"""Score trained checkpoints on CBIS-DDSM (validation and test) or INbreast, deterministically.

The CLAHE clip limit is fixed at evaluation (default 4.0). '--clip average' averages
predictions over clip limits 1, 2, 3 and 4. Output files use the same layout as training.

Examples
--------
    python scripts/rescore.py --runs runs/single_backbone runs/specialists runs/ensembles --data data/cbis --out results/predictions/cbis/clip4.0
    python scripts/rescore.py --runs ... --data data/cbis --clip 2.5 --out results/predictions/cbis/clip2.5
    python scripts/rescore.py --runs ... --inbreast data/inbreast/inbreast_rois.csv --out results/predictions/inbreast/clip4.0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing the package
import argparse
import json
import shutil

import numpy as np
import pandas as pd

from mammo_routing.config import TrainConfig, condition_backbones
from mammo_routing.data import eval_loader, load_splits
from mammo_routing.inference import average_predictions, predict, predictions_frame
from mammo_routing.models import AFAGEOracle, model_for_key
from mammo_routing.reproducibility import environment, free_memory, set_seed
from mammo_routing.training import load_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True, help="training run folders (with models/ and run_registry.json)")
    ap.add_argument("--data", type=Path, help="CBIS-DDSM data folder")
    ap.add_argument("--inbreast", type=Path, help="INbreast ROI table (instead of --data)")
    ap.add_argument("--clip", default="4.0", help="fixed CLAHE clip limit, or 'average' for clip limits 1-4")
    ap.add_argument("--keys", nargs="*", help="score only these model keys")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = TrainConfig()
    clips = [1.0, 2.0, 3.0, 4.0] if args.clip == "average" else [float(args.clip)]
    if args.inbreast is not None:
        splits = [("inbreast", pd.read_csv(args.inbreast))]
    else:
        _, val_df, test_df = load_splits(args.data)
        splits = [("val", val_df), ("test", test_df)]
    set_seed(42)
    for run in args.runs:
        registry = json.loads((run / "run_registry.json").read_text())
        dst = args.out / run.name
        (dst / "predictions").mkdir(parents=True, exist_ok=True)
        for f in ("run_registry.json", "affinity_matrix.npy", "affinity_matrix_5bb.npy", "affinity_matrix_used.npy"):
            if (run / f).exists():
                shutil.copy(run / f, dst / f)
        (dst / "environment.json").write_text(json.dumps({**environment(), "clip": args.clip}, indent=2))
        a4 = np.load(run / "affinity_matrix_used.npy") if (run / "affinity_matrix_used.npy").exists() else None
        a5 = next((np.load(p) for p in (r / "affinity_matrix_5bb.npy" for r in args.runs) if p.exists()), None)
        for tag, info in registry.items():
            key, seed = info["key"], info["seed"]
            if args.keys and key not in args.keys:
                continue
            model = load_checkpoint(model_for_key(key, a4, a5, pretrained=False), run / "models" / f"{tag}_best.pth")
            oracle = isinstance(model, AFAGEOracle)
            names = None if key.startswith(("U_", "S_")) else condition_backbones(key)
            frames = [predictions_frame(df, average_predictions([predict(model, eval_loader(df, cfg, clip=c), oracle)
                                                                 for c in clips]), key, seed, split, names)
                      for split, df in splits]
            pd.concat(frames).to_csv(dst / "predictions" / f"{tag}.csv.gz", index=False, compression="gzip")
            print(f"{run.name}/{tag}")
            del model
            free_memory()


if __name__ == "__main__":
    main()
