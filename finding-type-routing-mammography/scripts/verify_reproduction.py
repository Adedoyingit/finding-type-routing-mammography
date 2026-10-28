"""Recompute every analysis from the released predictions and compare with the released results.

Example
-------
    python scripts/verify_reproduction.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing the package

import subprocess  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    released = ROOT / "results" / "analysis"
    with tempfile.TemporaryDirectory() as tmp:
        print("Recomputing all analyses from results/predictions (a few minutes) ...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "run_analyses.py"), "--results", str(ROOT / "results"),
                        "--out", tmp], check=True, stdout=subprocess.DEVNULL)
        files = sorted(released.rglob("*.csv"))
        failures = 0
        for f in files:
            a, b = pd.read_csv(f), pd.read_csv(Path(tmp) / f.relative_to(released))
            num = a.select_dtypes("number").columns
            same_shape = a.shape == b.shape
            diff = float(np.nanmax(np.abs(a[num].values - b[num].values))) if same_shape and len(num) else 0.0
            ok = same_shape and diff < 1e-12
            failures += not ok
            print(f"{'OK  ' if ok else 'DIFF'}  {f.relative_to(released)}  (max |difference| {diff:.1e})")
    print(f"\n{len(files) - failures} of {len(files)} result tables reproduced exactly.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
