"""Extract the CBIS-DDSM lesion crops used in the paper from the TCIA download.

The released ROI index (results/data/cbis_roi_index.csv) fixes which lesions are used
and which split each belongs to; the script finds each lesion's own DICOM folder,
extracts the crop and mask, and checks every crop against the recorded MD5 hash.

Example
-------
    python scripts/extract_cbis.py --dicom-root /data/CBIS-DDSM --index results/data/cbis_roi_index.csv --out data/cbis
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing the package
import argparse

import pandas as pd

from mammo_routing.extraction import extract_cbis_roi, index_cbis_dicoms, pick_crop_and_mask


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dicom-root", type=Path, required=True, help="unzipped TCIA CBIS-DDSM ROI download")
    ap.add_argument("--index", type=Path, default=Path("results/data/cbis_roi_index.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/cbis"))
    args = ap.parse_args()

    index = pd.read_csv(args.index)
    print(f"Indexing DICOM files under {args.dicom_root} ...")
    dicoms = index_cbis_dicoms(args.dicom_root)
    records, mismatches = [], 0
    for r in index.itertuples():
        key = (r.category, r.cbis_split, r.patient_id, r.laterality, r.view, int(r.abnormality_id))
        crop, mask = pick_crop_and_mask(dicoms.get(key, []))
        if crop is None:
            raise FileNotFoundError(f"no cropped image for {key}; is the download complete?")
        rel = f"{r.cbis_split}/{r.category}/{r.label}/{r.patient_id}_{r.laterality}_{r.view}_abn{int(r.abnormality_id)}.png"
        rec = extract_cbis_roi(crop, mask, args.out, rel)
        mismatches += rec["crop_md5"] != r.crop_md5
        records.append({**r._asdict(), **rec, "roi_path": f"cbis_rois/{rel}", "native_relpath": rel})
    table = pd.DataFrame(records).drop(columns=["Index"])
    for split in ("train", "val", "test"):
        table[table.split == split].to_csv(args.out / f"{split}_df.csv", index=False)
    print(f"Extracted {len(table)} ROIs; crops differing from the recorded hash: {mismatches}")


if __name__ == "__main__":
    main()
