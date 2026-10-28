"""Check that inference is repeatable: score checkpoints twice on GPU and once on CPU.

Example
-------
    python scripts/check_determinism.py --run runs/ensembles --tags C1_seed42 C3_beta0_seed42 --data data/cbis
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing the package
import argparse
import json

import numpy as np
import torch

from mammo_routing.config import TrainConfig
from mammo_routing.data import eval_loader, load_splits
from mammo_routing.inference import predict
from mammo_routing.models import AFAGEOracle, model_for_key
from mammo_routing.stats import auc


def score(run, tag, key, df, device, cfg):
    a4 = np.load(run / "affinity_matrix_used.npy") if (run / "affinity_matrix_used.npy").exists() else None
    model = model_for_key(key, a4, pretrained=False)
    state = torch.load(run / "models" / f"{tag}_best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"], strict=True)
    model = model.to(device)
    return predict(model, eval_loader(df, cfg), isinstance(model, AFAGEOracle), device=torch.device(device))["prob_malignant"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    cfg = TrainConfig()
    _, _, test_df = load_splits(args.data)
    registry = json.loads((args.run / "run_registry.json").read_text())
    y = test_df.label_idx.values
    for tag in args.tags:
        key = registry[tag]["key"]
        g1, g2 = (score(args.run, tag, key, test_df, "cuda", cfg) for _ in range(2))
        c = score(args.run, tag, key, test_df, "cpu", cfg)
        for name, a, b in (("GPU vs GPU", g1, g2), ("GPU vs CPU", g1, c)):
            print(f"{tag:<22} {name}: max |dp| {np.abs(a - b).max():.2e}  |dAUC| {abs(auc(y, a) - auc(y, b)):.2e}")


if __name__ == "__main__":
    main()
