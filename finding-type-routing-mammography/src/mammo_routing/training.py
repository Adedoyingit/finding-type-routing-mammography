"""Training loop and multi-run execution with resume."""
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .data import eval_loader, train_loader
from .inference import predict, predictions_frame
from .reproducibility import DEVICE, backbone_fingerprint, free_memory, set_seed


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


def focal_loss_for(df, gamma=2.0):
    """Focal loss with class weights inversely proportional to class frequency."""
    counts = df["label_idx"].value_counts().sort_index().values
    return FocalLoss(torch.FloatTensor(len(df) / (2.0 * counts)).to(DEVICE), gamma)


def train_one(model, train_df, val_df, seed, cfg, has_aux, oracle, ckpt_path, lr=None, log=print):
    """Train one model; keep the checkpoint with the highest validation AUC."""
    criterion = focal_loss_for(train_df, cfg.focal_gamma)
    train_ld, val_ld = train_loader(train_df, seed, cfg), eval_loader(val_df, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr or cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    best, best_epoch, waited, history = 0.0, 0, 0, []
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for imgs, labels, cats in train_ld:
            imgs, labels, cats = imgs.to(DEVICE), labels.to(DEVICE), cats.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                out = model(imgs, true_category_idx=cats) if oracle else model(imgs)
                loss = criterion(out["main_logits"], labels)
                if has_aux and "finding_logits" in out:
                    loss = loss + cfg.aux_lambda * F.cross_entropy(out["finding_logits"], cats)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            batches += 1
        sched.step()
        val_auc = roc_auc_score(val_df["label_idx"].values, predict(model, val_ld, oracle)["prob_malignant"])
        history.append({"epoch": epoch, "loss": total / batches, "val_auc": float(val_auc)})
        if val_auc > best + cfg.min_delta:
            best, best_epoch, waited = val_auc, epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_auc": float(val_auc),
                        "seed": seed, "saved_at": datetime.now().isoformat()}, ckpt_path)
        else:
            waited += 1
        log(f"epoch {epoch:2d}  loss {total / batches:.4f}  val AUC {val_auc:.4f}  best {best:.4f} (epoch {best_epoch})")
        if waited >= cfg.patience:
            break
    return {"best_val_auc": float(best), "best_epoch": best_epoch, "epochs_run": epoch,
            "minutes": round((time.time() - t0) / 60, 1), "history": history}


def load_checkpoint(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"], strict=True)
    return model.to(DEVICE)


def run_jobs(jobs, build_fn, val_df, test_df, out_dir, cfg, resume_dirs=()):
    """Train every job and export validation and test predictions.

    Each job is a dict with keys: key, seed, train_df, val_df, has_aux, oracle and,
    optionally, lr and backbone_names. A job is skipped if its prediction file already
    exists in `out_dir` or in one of `resume_dirs`. Results are recorded in
    `out_dir/run_registry.json`.
    """
    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)
    reg_path = out_dir / "run_registry.json"
    registry = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    for i, job in enumerate(jobs, 1):
        tag = f"{job['key']}_seed{job['seed']}"
        pred_file = out_dir / "predictions" / f"{tag}.csv.gz"
        if pred_file.exists() or any((Path(d) / "predictions" / f"{tag}.csv.gz").exists() for d in resume_dirs):
            print(f"[{i}/{len(jobs)}] {tag}: done, skipping")
            continue
        print(f"[{i}/{len(jobs)}] {tag}: training on {len(job['train_df'])} ROIs")
        free_memory()
        set_seed(job["seed"])
        model = build_fn(job).to(DEVICE)
        fingerprint = backbone_fingerprint(model)
        ckpt = out_dir / "models" / f"{tag}_best.pth"
        info = train_one(model, job["train_df"], job["val_df"], job["seed"], cfg, job["has_aux"],
                         job["oracle"], ckpt, lr=job.get("lr"))
        model = load_checkpoint(model, ckpt)
        frames = [predictions_frame(d, predict(model, eval_loader(d, cfg), job["oracle"]), job["key"],
                                    job["seed"], split, job.get("backbone_names"))
                  for split, d in (("val", val_df), ("test", test_df))]
        pd.concat(frames).to_csv(pred_file, index=False, compression="gzip")
        extra = {}
        if hasattr(model, "affinity_matrix"):
            a = model.affinity_matrix.detach().float().cpu()
            extra = {"A_learned": a.tolist(), "softmax_A": F.softmax(a, 0).tolist(),
                     "temperature": float(model.temperature.detach().clamp(0.5, 3.0))
                     if hasattr(model, "temperature") else None}
        registry[tag] = {**info, "key": job["key"], "seed": job["seed"], "init_fingerprint": fingerprint,
                         "train_n": len(job["train_df"]), "val_n": len(job["val_df"]), **extra}
        reg_path.write_text(json.dumps(registry, indent=1))
        print(f"  best validation AUC {info['best_val_auc']:.4f} at epoch {info['best_epoch']}")
        del model
        free_memory()
    return registry
