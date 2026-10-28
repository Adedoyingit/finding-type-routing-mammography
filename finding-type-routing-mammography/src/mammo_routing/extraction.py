"""Extraction of lesion crops from CBIS-DDSM and INbreast.

CBIS-DDSM
    Each abnormality folder of the TCIA download is identified by the six fields in its
    name (finding type, original split, patient, laterality, view, abnormality number).
    Within a folder, the cropped lesion image and the binary lesion mask are told apart
    by pixel content (a mask has at most two grey levels) rather than by the series
    description, which is unreliable in the collection.

INbreast
    One crop per radiologist-drawn mass outline and per radiologist-drawn calcification
    cluster, cut from the raw DICOM pixels around the outline's bounding box with 6.35%
    padding per side (reproducing the median CBIS-DDSM crop-to-lesion-box area ratio of
    1.27), then processed exactly like the CBIS-DDSM crops. Labels follow image BI-RADS:
    1-2 benign, 4a-6 malignant, 3 excluded.
"""
import hashlib
import plistlib
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FOLDER_KEY = re.compile(r"(Mass|Calc)-(Training|Test)_(P_\d+)_(LEFT|RIGHT)_(CC|MLO)_(\d+)", re.I)
INBREAST_PAD = 0.0635
BENIGN_BIRADS, MALIGNANT_BIRADS = {"1", "2"}, {"4a", "4b", "4c", "5", "6"}


def to_uint8(arr, photometric="MONOCHROME2"):
    arr = arr.astype(np.float32)
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return (arr * 255).astype(np.uint8)


def parse_folder_key(text):
    """(finding type, original split, patient, laterality, view, abnormality number) or None."""
    m = FOLDER_KEY.search(str(text))
    if not m:
        return None
    cat, split, pid, lat, view, abn = m.groups()
    return ("Mass" if cat.lower() == "mass" else "Calcification", "train" if split.lower() == "training" else "test",
            pid.upper(), lat.upper(), view.upper(), int(abn))


def index_cbis_dicoms(dicom_root):
    """Map every abnormality key to the crop and mask DICOM files found in its folder."""
    import pydicom
    dicom_root = Path(dicom_root)
    index = defaultdict(list)
    for p in sorted(dicom_root.rglob("*.dcm")):
        key = next((parse_folder_key(a.name) for a in p.parents if FOLDER_KEY.fullmatch(a.name)), None)
        if key is None:
            continue                                   # full mammograms and other series
        dcm = pydicom.dcmread(str(p))
        arr = dcm.pixel_array
        index[key].append({"path": str(p), "shape": arr.shape,
                           "kind": "mask" if len(np.unique(arr)) <= 2 else "crop",
                           "photometric": str(getattr(dcm, "PhotometricInterpretation", "MONOCHROME2"))})
    return index


def pick_crop_and_mask(entries):
    crops = sorted((e for e in entries if e["kind"] == "crop"), key=lambda e: e["shape"][0] * e["shape"][1])
    masks = [e for e in entries if e["kind"] == "mask"]
    return (crops[0] if crops else None), (masks[0] if masks else None)


def extract_cbis_roi(crop_entry, mask_entry, out_root, relpath, img_size=224):
    """Write the 224-pixel crop, the native-resolution crop and the bounding-box-cropped mask."""
    import pydicom
    out_root = Path(out_root)
    img = to_uint8(pydicom.dcmread(crop_entry["path"]).pixel_array, crop_entry["photometric"])
    small = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    for sub, im in (("cbis_rois", small), ("cbis_rois_native", img)):
        (out_root / sub / relpath).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_root / sub / relpath), im)
    rec = {"crop_h": img.shape[0], "crop_w": img.shape[1], "crop_md5": hashlib.md5(small.tobytes()).hexdigest()}
    if mask_entry is not None:
        m = pydicom.dcmread(mask_entry["path"]).pixel_array > 0
        ys, xs = np.where(m)
        if len(ys):
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            (out_root / "cbis_masks" / relpath).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_root / "cbis_masks" / relpath), (m[y0:y1, x0:x1] * 255).astype(np.uint8))
            rec.update({"mask_relpath": relpath, "mask_bbox_y0": int(y0), "mask_bbox_y1": int(y1),
                        "mask_bbox_x0": int(x0), "mask_bbox_x1": int(x1), "mask_area_px": int(m.sum())})
    return rec


def inbreast_birads(xls_path):
    meta = pd.read_excel(xls_path)
    meta.columns = [str(c).strip() for c in meta.columns]

    def norm_id(v):
        try:
            return str(int(float(v)))
        except (ValueError, TypeError):
            return str(v).strip()

    meta["case_id"] = meta["File Name"].map(norm_id)
    meta["bi_rads"] = meta["Bi-Rads"].astype(str).str.strip().str.lower()
    return meta.dropna(subset=["case_id"]).drop_duplicates("case_id").set_index("case_id").bi_rads


def extract_inbreast(xml_dir, dicom_dir, xls_path, out_dir, img_size=224):
    """Crops for radiologist-drawn masses and calcification clusters; returns the ROI table."""
    import pydicom
    point = re.compile(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)")
    out_dir = Path(out_dir)
    (out_dir / "inbreast_rois").mkdir(parents=True, exist_ok=True)
    birads = inbreast_birads(xls_path)
    rows = []
    for xml in sorted(Path(xml_dir).glob("*.xml")):
        case = xml.stem
        dcms = sorted(Path(dicom_dir).glob(f"{case}_*.dcm"))
        br = birads.get(case)
        if not dcms or br not in BENIGN_BIRADS | MALIGNANT_BIRADS:
            continue
        rois = plistlib.load(open(xml, "rb"))["Images"][0].get("ROIs", [])
        wanted = [(r.get("Name", "").strip().lower(), r) for r in rois]
        wanted = [(n, r) for n, r in wanted if n in ("mass", "cluster")]
        if not wanted:
            continue
        ds = pydicom.dcmread(dcms[0])
        img_raw = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            img_raw = img_raw.max() - img_raw
        height, width = img_raw.shape
        parts = dcms[0].stem.split("_")
        for i, (name, r) in enumerate(wanted):
            pts = np.array([[float(a), float(b)] for a, b in (point.search(p).groups() for p in r.get("Point_px", []))])
            if len(pts) < 3:
                continue
            (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
            w, h = x1 - x0, y1 - y0
            x0, x1 = int(max(0, x0 - INBREAST_PAD * w)), int(min(width, x1 + INBREAST_PAD * w))
            y0, y1 = int(max(0, y0 - INBREAST_PAD * h)), int(min(height, y1 + INBREAST_PAD * h))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crop = cv2.resize(to_uint8(img_raw[y0:y1, x0:x1]), (img_size, img_size), interpolation=cv2.INTER_AREA)
            category = "Mass" if name == "mass" else "Calcification"
            rel = f"inbreast_rois/{case}_{category}_{i}.png"
            cv2.imwrite(str(out_dir / rel), crop)
            rows.append({"case_id": case, "patient_id": parts[1], "laterality": parts[3] if len(parts) > 3 else "",
                         "view": parts[4] if len(parts) > 4 else "", "abnormality_id": i, "category": category,
                         "category_idx": int(category == "Calcification"), "bi_rads": br,
                         "label": "Malignant" if br in MALIGNANT_BIRADS else "Benign",
                         "label_idx": int(br in MALIGNANT_BIRADS), "roi_path": str(out_dir / rel),
                         "box_w_px": x1 - x0, "box_h_px": y1 - y0, "v1_crop_was_wrong": False, "is_bwc": False})
    return pd.DataFrame(rows)
