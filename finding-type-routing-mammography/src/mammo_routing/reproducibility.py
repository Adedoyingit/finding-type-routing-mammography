"""Seeding, determinism and environment records."""
import gc
import hashlib
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """Seed Python, NumPy and PyTorch and select deterministic cuDNN kernels."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    ws = torch.initial_seed() % 2 ** 32
    np.random.seed(ws)
    random.seed(ws)


def generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def environment():
    """Software and hardware versions, stored with every run."""
    import albumentations
    import cv2
    import pandas
    import timm
    from . import __version__
    return {"package": __version__, "python": sys.version.split()[0], "torch": torch.__version__,
            "timm": timm.__version__, "albumentations": albumentations.__version__,
            "opencv": cv2.__version__, "numpy": np.__version__, "pandas": pandas.__version__,
            "device": str(DEVICE),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "time": datetime.now().isoformat(timespec="seconds")}


def backbone_fingerprint(model):
    """Hash of the pretrained backbone weights before training.

    Runs of the same architecture should share one fingerprint, which confirms
    that they started from identical ImageNet weights.
    """
    h = hashlib.md5()
    skip = ("classifier", "finding_predictor", "gate", "projector", "affinity", "temperature")
    for k, v in sorted(model.state_dict().items()):
        if any(t in k for t in skip):
            continue
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]
