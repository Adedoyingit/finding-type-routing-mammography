# Finding-type-aware ensemble routing for mammographic lesion classification

Code, per-lesion predictions and analysis outputs for:

> A. E. Oyekanmi, Y. Sun, J. O. Olamijuwon and Z. Wang. *Finding-type-aware ensemble routing does not improve mammographic lesion classification: a controlled multi-seed evaluation.* Manuscript submitted for publication (2026).

The study asks whether CNN backbones are differentially suited to mammographic masses and calcifications, and whether an affinity-guided adaptive ensemble (AFAGE) that routes between backbones by predicted finding type improves on a jointly trained ensemble and on simple averaging of independently trained backbones. All models are trained on provenance-verified CBIS-DDSM lesion crops with one recipe and three seeds; comparisons use a paired bootstrap over training seeds and patients; INbreast provides external validation.

## Quick start: verify the results

```bash
git clone https://github.com/Adedoyingit/finding-type-routing-mammography.git
cd finding-type-routing-mammography
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install numpy pandas scipy scikit-learn matplotlib pytest
python scripts/verify_reproduction.py      # recomputes all 29 result tables and compares them with the released ones
python scripts/make_figures.py --out my_figures   # regenerates Figures 1-7 and Table 2
python -m pytest -q tests                  # unit tests (model tests run only if PyTorch is installed)
```

No GPU, images or datasets are needed: the scripts work from the per-lesion predictions in `results/predictions/`.

## Repository layout

```
src/mammo_routing/       Python package
    config.py              training recipe, backbones, the twelve ensemble conditions
    data.py                split loading, CLAHE and augmentation, datasets, sampler
    models.py              single-backbone models and ensemble architectures (AFAGE and ablations)
    training.py            training loop, checkpointing, multi-run execution
    inference.py           prediction export
    extraction.py          lesion-crop extraction from CBIS-DDSM and INbreast DICOM files
    stats.py               AUC, partial AUC, operating points, paired seed x patient bootstrap
    results_io.py          reading prediction files
    analysis/              the analyses reported in the paper
scripts/
    extract_cbis.py        lesion crops from the TCIA CBIS-DDSM download
    extract_inbreast.py    lesion crops from INbreast
    train.py               single-backbone models, specialists, ensemble conditions
    rescore.py             deterministic scoring of checkpoints (any CLAHE setting, CBIS-DDSM or INbreast)
    check_determinism.py   repeatability of inference (GPU vs GPU, GPU vs CPU)
    run_analyses.py        every statistical analysis in the paper
    make_figures.py        Figures 1-7 and Table 2
    verify_reproduction.py recompute all analyses and compare with the released results
results/
    data/                  lesion tables (no images)
    predictions/           per-lesion predictions of all 78 trained models
    analysis/              outputs of run_analyses.py
    figures/               outputs of make_figures.py
tests/                     unit tests
```

## Installation

```bash
pip install -e .              # analysis and figures (CPU)
pip install -e ".[train]"     # training, scoring and data extraction
```

Python 3.10 or later. The models in the paper were trained with Python 3.13, PyTorch 2.11, timm 1.0.29 and albumentations 2.0.8 on NVIDIA Tesla T4 GPUs (`requirements.txt`).

## Reproducing the results

**Statistics, figures and tables (CPU, about 5 minutes).** The released predictions are sufficient:

```bash
python scripts/run_analyses.py --results results --out results/analysis
python scripts/make_figures.py --results results --out results/figures
```

These reproduce every number, figure and table in the paper exactly (the bootstrap uses a fixed random state).

**Full pipeline (GPU).**

```bash
# 1. Data
python scripts/extract_cbis.py --dicom-root /path/to/CBIS-DDSM --index results/data/cbis_roi_index.csv --out data/cbis
python scripts/extract_inbreast.py --inbreast /path/to/INbreast --out data/inbreast

# 2. Training (three seeds per model; about 16 GPU-hours in total on a T4)
python scripts/train.py single      --data data/cbis --out runs/single_backbone
python scripts/train.py specialists --data data/cbis --out runs/specialists
python scripts/train.py ensembles   --data data/cbis --out runs/ensembles --affinity runs/specialists

# 3. Deterministic scoring: primary setting, two sensitivity settings, external data
python scripts/check_determinism.py --run runs/ensembles --tags C1_seed42 C3_beta0_seed42 --data data/cbis
python scripts/rescore.py --runs runs/* --data data/cbis --clip 4.0     --out results/predictions/cbis/clip4.0
python scripts/rescore.py --runs runs/* --data data/cbis --clip 2.5     --out results/predictions/cbis/clip2.5
python scripts/rescore.py --runs runs/* --data data/cbis --clip average --out results/predictions/cbis/clip_average
python scripts/rescore.py --runs runs/* --inbreast data/inbreast/inbreast_rois.csv --clip 4.0 --out results/predictions/inbreast/clip4.0

# 4. Analyses and figures, as above
```

## Notes on the method

- **Contrast enhancement.** In albumentations, `A.CLAHE(clip_limit=4.0)` draws the clip limit uniformly from [1, 4] on every call. Models were trained with this behaviour (an implicit augmentation); evaluation fixes the clip limit (4.0 primary; 2.5 and the average over 1-4 as sensitivity analyses) so that inference is deterministic. See `src/mammo_routing/data.py`.
- **Checkpoint format.** Attribute names and construction order in `models.py` define the state-dict keys of the trained checkpoints and must not be changed.
- **Lesion keys.** CBIS-DDSM lesions are identified by finding type, original split, patient, laterality, view and abnormality number; `extraction.py` matches each lesion to its own DICOM folder using all six fields. `results/data/cbis_roi_provenance_flags.csv` documents the verification of an earlier extraction that motivated this design.

## Data

- **CBIS-DDSM** — The Cancer Imaging Archive (https://www.cancerimagingarchive.net/collection/cbis-ddsm/).
- **INbreast** — available from its authors under their data-use terms. No images are included; the INbreast lesion table and predictions use pseudonymous case and patient codes (see `results/README.md`).

## Licence

MIT (see `LICENSE`).
