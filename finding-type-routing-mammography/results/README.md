# Results

| Folder | Contents | Produced by |
|---|---|---|
| `data/inbreast_rois.csv` | INbreast lesions used for external validation: 103 masses and 27 calcification clusters, with BI-RADS-derived labels and pseudonymous case and patient codes (no images) | `scripts/extract_inbreast.py` |
| `data/cbis_roi_provenance_flags.csv` | Verification of an earlier CBIS-DDSM extraction: lesions whose crop came from another abnormality | data audit |
| `data/cbis_roi_index.csv` | Lesion keys, labels, split and crop hash of the 2,667 CBIS-DDSM lesions used | `scripts/export_roi_index.py` |
| `predictions/cbis/clip4.0/` | Validation and test predictions of all 78 models (primary evaluation setting), run registries with training histories, affinity matrices | `scripts/train.py`, `scripts/rescore.py` |
| `predictions/cbis/clip2.5/`, `clip_average/` | Predictions of the models used in the primary comparisons under the two sensitivity settings | `scripts/rescore.py` |
| `predictions/inbreast/` | INbreast predictions of all models | `scripts/rescore.py` |
| `analysis/` | Every statistical result in the paper | `scripts/run_analyses.py` |
| `figures/` | Figures 1-7 and Table 2 | `scripts/make_figures.py` |

Prediction files contain one row per lesion with the model key (`U_<backbone>`, `S_<finding>_<backbone>` or an ensemble condition), the seed, the split, lesion metadata, the malignancy probability and, for ensembles, the predicted finding type and backbone weights.

**INbreast identifiers.** Case and patient identifiers in `data/inbreast_rois.csv` and `predictions/inbreast/` are replaced by codes (`C001`..., `P01`...) assigned in ascending order of the original INbreast case and patient identifiers. Users with INbreast access can recover the correspondence by running `scripts/extract_inbreast.py` and sorting its identifiers. All analyses are unaffected by the coding.
