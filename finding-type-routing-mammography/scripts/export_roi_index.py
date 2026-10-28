"""Write the ROI index (lesion keys, labels, split and crop hash) from an extracted dataset.

Run once on the dataset used for the paper to produce results/data/cbis_roi_index.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing the package
import argparse

import pandas as pd

COLUMNS = ["patient_id", "laterality", "view", "abnormality_id", "category", "category_idx", "label", "label_idx",
           "cbis_split", "pathology_raw", "is_bwc", "crop_md5", "v1_crop_was_wrong"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="folder with train_df.csv, val_df.csv, test_df.csv")
    ap.add_argument("--out", type=Path, default=Path("results/data/cbis_roi_index.csv"))
    args = ap.parse_args()
    frames = [pd.read_csv(args.data / f"{s}_df.csv").assign(split=s) for s in ("train", "val", "test")]
    index = pd.concat(frames, ignore_index=True)[COLUMNS + ["split"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.out, index=False)
    print(f"{len(index)} ROIs -> {args.out}")


if __name__ == "__main__":
    main()
