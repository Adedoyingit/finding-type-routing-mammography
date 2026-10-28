"""Reading prediction files and run registries."""
import json
from pathlib import Path

import pandas as pd


def prediction_files(folder):
    """Prediction files below `folder`, as .csv.gz or .csv.

    Some file-sharing services unpack each archive into a folder of the same name
    (e.g. predictions/X.csv.gz/X.csv); those files are found as well.
    """
    files = [p for p in Path(folder).rglob("*.csv*") if p.is_file() and "predictions" in p.parts]
    return sorted(files)


def load_predictions(folder):
    """All per-ROI prediction files below `folder`, with an `roi` key relative to cbis_rois/."""
    files = prediction_files(folder)
    if not files:
        raise FileNotFoundError(f"no prediction files below {folder}")
    p = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    p["roi"] = p["roi_path"].astype(str).str.split("cbis_rois/").str[-1].str.split("/").str[-4:].str.join("/")
    return p


def load_registries(folder):
    rows = []
    for reg in sorted(Path(folder).rglob("run_registry.json")):
        for tag, v in json.loads(reg.read_text()).items():
            rows.append({"run_folder": reg.parent.name, "tag": tag, **v})
    return pd.DataFrame(rows)


def wide(predictions, split):
    """Scores as an ROI x (model, seed) table, with ROI metadata in the same row order."""
    t = predictions[predictions.split == split]
    w = t.pivot_table(index="roi", columns=["model_key", "seed"], values="prob_malignant")
    cols = [c for c in ["patient_id", "label_idx", "category", "v1_crop_was_wrong", "is_bwc"] if c in t.columns]
    meta = t.drop_duplicates("roi").set_index("roi").loc[w.index, cols]
    return w, meta
