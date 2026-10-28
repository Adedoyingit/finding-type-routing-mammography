"""CBIS-DDSM regions of interest: splits, transforms, datasets and loaders.

Contrast enhancement
--------------------
All images are contrast-enhanced with CLAHE (8 x 8 tiles). In albumentations,
``A.CLAHE(clip_limit=4.0)`` draws the clip limit uniformly from [1, 4] on every
call. The models were trained with this behaviour, which acts as an additional
augmentation. For evaluation the clip limit is fixed (``clip_limit=(c, c)``) so
that inference is deterministic; the paper uses c = 4.0 and reports c = 2.5 and
an average over c = 1, 2, 3, 4 as sensitivity analyses.
"""
import os
from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .reproducibility import generator, seed_worker

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SPLIT_SIZES = (1728, 442, 497)


def load_splits(data_dir, image_dir=None, expected=SPLIT_SIZES):
    """Read train/val/test split tables and point ``roi_path`` at the local image folder.

    The split CSVs store image paths relative to a ``cbis_rois/`` folder; ``image_dir``
    is the local location of that folder (default: ``<data_dir>/cbis_rois``).
    """
    data_dir = Path(data_dir)
    image_dir = Path(image_dir) if image_dir else data_dir / "cbis_rois"
    splits = []
    for name in ("train", "val", "test"):
        df = pd.read_csv(data_dir / f"{name}_df.csv")
        df["roi_path"] = df["roi_path"].str.split("cbis_rois/").str[-1].map(lambda p: str(image_dir / p))
        missing = int((~df["roi_path"].map(os.path.exists)).sum())
        if missing:
            raise FileNotFoundError(f"{name}: {missing} images not found under {image_dir}")
        splits.append(df.reset_index(drop=True))
    train, val, test = splits
    if expected and (len(train), len(val), len(test)) != tuple(expected):
        raise ValueError(f"split sizes {(len(train), len(val), len(test))} differ from {expected}")
    patients = [set(d.patient_id) for d in splits]
    if patients[0] & patients[1] or patients[0] & patients[2] or patients[1] & patients[2]:
        raise ValueError("a patient appears in more than one split")
    return train, val, test


def clahe(clip=None):
    """CLAHE with a random clip limit in [1, 4] (clip=None, training) or a fixed clip limit."""
    limit = 4.0 if clip is None else (clip, clip)
    return A.CLAHE(clip_limit=limit, tile_grid_size=(8, 8), p=1.0)


def eval_transform(img_size=224, clip=4.0):
    return A.Compose([clahe(clip), A.Resize(img_size, img_size),
                      A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


def train_transform(finding_type, img_size=224):
    """Finding-type-specific training augmentation."""
    norm = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if finding_type == "Calcification":
        return A.Compose([
            clahe(), A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.6),
            A.RandomGamma(gamma_limit=(75, 125), p=0.4),
            A.GaussNoise(p=0.3), A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, p=0.4),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
            norm, ToTensorV2()])
    return A.Compose([
        clahe(), A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.RandomGamma(gamma_limit=(85, 115), p=0.3),
        A.ElasticTransform(alpha=1, sigma=10, p=0.2),
        A.OneOf([A.GaussNoise(p=1.0), A.GaussianBlur(p=1.0)], p=0.2),
        norm, ToTensorV2()])


class ROIDataset(Dataset):
    """Grey-scale ROI crops returned as (image tensor, pathology label, finding-type index)."""

    def __init__(self, df, train=False, img_size=224, clip=4.0):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.eval_tfm = eval_transform(img_size, clip)
        self.train_tfm = {f: train_transform(f, img_size) for f in ("Mass", "Calcification")} if train else {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = cv2.imread(str(row["roi_path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(row["roi_path"])
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        tfm = self.train_tfm.get(row["category"], self.eval_tfm) if self.train else self.eval_tfm
        return tfm(image=img)["image"], int(row["label_idx"]), int(row["category_idx"])


def stratum_sampler(df, minority_boost=1.5):
    """Sample finding-type x label strata equally, with a boost for the rarest stratum."""
    strata = df["category"] + "_" + df["label"]
    counts = strata.value_counts()
    weights = len(df) / (len(counts) * counts)
    weights[counts.idxmin()] *= minority_boost
    w = strata.map(weights).values
    return WeightedRandomSampler(torch.FloatTensor(w), num_samples=len(w), replacement=True)


def train_loader(df, seed, cfg):
    return DataLoader(ROIDataset(df, train=True, img_size=cfg.img_size), batch_size=cfg.batch_size,
                      sampler=stratum_sampler(df, cfg.minority_boost), num_workers=cfg.num_workers,
                      pin_memory=True, worker_init_fn=seed_worker, generator=generator(seed))


def eval_loader(df, cfg, clip=4.0):
    return DataLoader(ROIDataset(df, train=False, img_size=cfg.img_size, clip=clip), batch_size=cfg.batch_size,
                      shuffle=False, num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=seed_worker)
