"""Extract INbreast mass and calcification-cluster crops for external validation.

Example
-------
    python scripts/extract_inbreast.py --inbreast /data/INbreast --out data/inbreast
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing the package
import argparse

from mammo_routing.extraction import extract_inbreast


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inbreast", type=Path, required=True, help="folder containing AllXML/, AllDICOMs/ and INbreast.xls")
    ap.add_argument("--out", type=Path, default=Path("data/inbreast"))
    args = ap.parse_args()
    rois = extract_inbreast(args.inbreast / "AllXML", args.inbreast / "AllDICOMs", args.inbreast / "INbreast.xls", args.out)
    rois.to_csv(args.out / "inbreast_rois.csv", index=False)
    print(rois.groupby(["category", "label"]).agg(rois=("roi_path", "size"), patients=("patient_id", "nunique")))


if __name__ == "__main__":
    main()
