import os
import time
import warnings
from collections import defaultdict

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Assuming acoustic_features.py is in the root directory alongside model.py and train.py
from acoustic_features import AcousticFeatureExtractor
from model import (
    get_sequence_infos,
    EnergySegDataset,
    collate_fn,
    worker_init_fn,
    EnergyInstanceModel
)


# ── 1. CONFIG ─────────────────────────────────────────────────────────────────
FRAMES_BASE           = "STARSS23/frames_dev"
LABELS_BASE           = "STARSS23/labels_dev"
MIC_BASE              = "STARSS23/mic_dev"     
UPLAM_CHECKPOINT      = "UpLAM.pth"                

IMG_W, IMG_H          = 360, 180    # all frames normalised to this
N_CHANNELS            = 12          # RGB × 4 tiled
NUM_CLASSES           = 14          # updated from JSON before train()
NUM_EPOCHS            = 200         # bounded by early stopping
PATIENCE              = 10          # early stopping patience
LR                    = 1e-3        # head LR  (backbone gets LR/10)
BATCH_SIZE            = 128
NUM_WORKERS           = 2           # Added: dynamic thread pooling handling in the dataloader

TRAIN_FRAMES_PER_EPOCH = None   # None = use all frames every epoch
VAL_FRAMES_PER_EPOCH   = None   # None = use all frames every epoch

SCORE_THR             = 0.05
ENERGY_ANNOT_W        = 5.0         # extra MSE weight on annotated pixels (RoI head)
FULLMAP_ANNOT_W       = 10.0        # weight on annotated pixels in full-image loss
DIST_NORM             = 500.0       # raw distance / DIST_NORM  → [0,~1]
ENERGY_EXPORT_THR     = 0.10        # minimum predicted energy to include a pixel
INFERENCE_HZ          = 10
INFERENCE_SEC         = 5
CACHE_MAX_SIZE        = 200         # Maximum internal array size handling dataset objects caching


# ── Device Setup ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    _gpu_note = f"  [{torch.cuda.get_device_name(0)}]"
else:
    DEVICE = torch.device("cpu")
    _mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    _gpu_note = ("  [Apple Silicon detected — using CPU: MPS skipped because "
                 "torchvision RoIAlign/NMS ops are not MPS-native and trigger "
                 "slow CPU fallbacks with device-transfer overhead]"
                 if _mps_available else "")

print(f"[INFO] Device: {DEVICE}{_gpu_note}")


# ── 2. PLOTTING UTILITY ───────────────────────────────────────────────────────
def plot_losses(history: dict, save_path: str = "training_loss.png"):
    keys    = ["total", "loss_energy_mask", "loss_classifier",
               "loss_box_reg", "loss_fullmap", "loss_distance",
               "loss_rpn_box_score"]
    titles  = ["Total loss", "★ Energy mask (MSE)", "Classifier",
               "Box regression", "Full-image energy", "Distance",
               "RPN objectness"]
    epochs = range(1, len(history.get("total", [])) + 1)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    fig.suptitle("Training & Validation Loss Curves", fontsize=13)

    def smooth(v, k=7):
        k = max(3, min(k, (len(v) // 3) * 2 + 1) | 1)
        return np.convolve(v, np.ones(k)/k, mode="valid"), k // 2

    for i, (key, title) in enumerate(zip(keys, titles)):
        ax  = axes[i]
        vals = history.get(key, [])
        ax.plot(list(epochs), vals, lw=1, color="steelblue", alpha=0.4, label="train (raw)")
        
        if len(vals) >= 5:
            s, pad = smooth(vals)
            ax.plot(list(range(pad+1, len(vals)-pad+1)), s,
                    lw=2, color="orangered", label="train (smoothed)")
                    
        if key == "total" and "val_total" in history:
            ax.plot(list(epochs), history["val_total"], lw=2, color="green", label="validation")

        ax.legend(fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)

    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Loss curves → {save_path}")


# ── 3. TRAINING LOOP ──────────────────────────────────────────────────────────
def train(train_infos: list, val_infos: list):
    print(f"[TRAIN] NUM_CLASSES = {NUM_CLASSES}")

    acoustic_extractor = AcousticFeatureExtractor(
        uplam_checkpoint = UPLAM_CHECKPOINT,
        device           = torch.device("cpu"),
        num_bands        = 9,
    )
    acoustic_extractor.clear_cache()
    
    train_dataset = EnergySegDataset(
        sequence_infos     = train_infos, 
        frames_base        = FRAMES_BASE, 
        mic_base           = MIC_BASE, 
        acoustic_extractor = acoustic_extractor,
        frames_per_epoch   = TRAIN_FRAMES_PER_EPOCH,
        img_w=IMG_W, img_h=IMG_H, dist_norm=DIST_NORM, cache_max_size=CACHE_MAX_SIZE
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
        pin_memory=True, 
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        worker_init_fn=worker_init_fn if NUM_WORKERS > 0 else None
    )
    
    val_dataset = EnergySegDataset(
        sequence_infos     = val_infos, 
        frames_base        = FRAMES_BASE, 
        mic_base           = MIC_BASE, 
        acoustic_extractor = acoustic_extractor,
        frames_per_epoch   = VAL_FRAMES_PER_EPOCH,
        img_w=IMG_W, img_h=IMG_H, dist_norm=DIST_NORM, cache_max_size=CACHE_MAX_SIZE
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
        pin_memory=True, 
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        worker_init_fn=worker_init_fn if NUM_WORKERS > 0 else None
    )                              
    
    model = EnergyInstanceModel(
        num_classes     = NUM_CLASSES, 
        n_channels      = N_CHANNELS,
        img_w=IMG_W, img_h=IMG_H,
        energy_annot_w  = ENERGY_ANNOT_W,
        fullmap_annot_w = FULLMAP_ANNOT_W
    ).to(DEVICE)

    backbone_params = list(model.det.backbone.parameters())
    head_params     = (list(model.det.rpn.parameters())
                     + list(model.det.roi_heads.parameters())
                     + list(model.energy_head.parameters())
                     + list(model.energy_decoder.parameters())
                     + list(model.distance_head.parameters())
                     + list(model.energy_roi_pool.parameters()))

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LR * 0.1, "weight_decay": 1e-4},
        {"params": head_params,     "lr": LR,        "weight_decay": 0.0},
    ])

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr  = [LR * 0.1, LR],
        steps_per_epoch = max(len(train_loader), 1),
        epochs  = NUM_EPOCHS,
        pct_start = 0.05,
    )

    history = defaultdict(list)
    t0 = time.time()
    loss_keys = [
        "loss_rpn_box_score", "loss_rpn_box_reg",
        "loss_classifier", "loss_box_reg",
        "loss_energy_mask", "loss_fullmap", "loss_distance",
    ]
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"[TRAIN] Max {NUM_EPOCHS} epochs × {len(train_dataset)} train frames | device={DEVICE}")

    for epoch in range(1, NUM_EPOCHS + 1):
        # ─── PREPARE EPOCH SUBSET ──────────────────────────────────────────────
        train_dataset.reset_epoch()
        val_dataset.reset_epoch()
        
        # ─── TRAIN LOOP ────────────────────────────────────────────────────────
        model.train()
        ep = defaultdict(float)
        n  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{NUM_EPOCHS} [Train]", leave=False)
        for images, targets in pbar:
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total     = sum(loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            ep["total"] += total.item()
            for k in loss_keys:
                val = loss_dict.get(k)
                ep[k] += (val.item() if isinstance(val, torch.Tensor) else float(val or 0))
            n += 1

            pbar.set_postfix({
                "total":  f"{total.item():.3f}",
                "energy": f"{loss_dict.get('loss_energy_mask', 0):.3f}",
                "vmap":   f"{loss_dict.get('loss_fullmap', 0):.4f}",
            })

        for k in ["total"] + loss_keys:
            history[k].append(ep[k] / max(n, 1))
            
        # ─── VALIDATION LOOP ───────────────────────────────────────────────────
        val_loss_sum = 0.0
        val_n = 0
        
        if len(val_loader) > 0:
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch:3d}/{NUM_EPOCHS} [Val]", leave=False)
            with torch.no_grad():
                for images, targets in pbar_val:
                    images  = [img.to(DEVICE) for img in images]
                    targets = [{k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                                for k, v in t.items()} for t in targets]

                    loss_dict = model(images, targets)
                    val_total = sum(loss_dict.values())
                    val_loss_sum += val_total.item()
                    val_n += 1

        val_avg_loss = val_loss_sum / max(val_n, 1) if val_n > 0 else 0.0
        history["val_total"].append(val_avg_loss)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Train={history['total'][-1]:.4f}  "
              f"Val={val_avg_loss:.4f} | "
              f"Energy={history['loss_energy_mask'][-1]:.4f}  "
              f"VMap={history['loss_fullmap'][-1]:.4f} | "
              f"{time.time()-t0:.0f}s")
              
        # ─── EARLY STOPPING ────────────────────────────────────────────────────
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "energy_seg_best.pth")
            print("  [*] Validation loss improved. Best model saved.")
        else:
            epochs_no_improve += 1
            print(f"  [!] No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print(f"[TRAIN] Finished in {time.time()-t0:.1f}s")
    
    if os.path.exists("energy_seg_best.pth"):
        model.load_state_dict(torch.load("energy_seg_best.pth", map_location=DEVICE))
        
    return model, dict(history)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    train_infos = get_sequence_infos("train", LABELS_BASE, FRAMES_BASE)
    test_infos  = get_sequence_infos("test", LABELS_BASE, FRAMES_BASE)

    if not train_infos:
        raise FileNotFoundError(f"No train directories found in {LABELS_BASE}")
    if not test_infos:
        print(f"[WARN] No test directories found for cross-validation in {LABELS_BASE}")

    print(f"\n[MAIN] Found {len(train_infos)} training sequences.")
    print(f"[MAIN] Found {len(test_infos)} validation/test sequences.")
    print(f"\n[MAIN] Training config:")
    print(f"  TRAIN_FRAMES_PER_EPOCH = {TRAIN_FRAMES_PER_EPOCH}")
    print(f"  VAL_FRAMES_PER_EPOCH   = {VAL_FRAMES_PER_EPOCH}")
    print(f"  NUM_EPOCHS             = {NUM_EPOCHS} (Early stopping max)")
    print(f"  PATIENCE               = {PATIENCE}")
    print(f"  LR                     = {LR}  (backbone {LR*0.1})")
    print(f"  BATCH_SIZE             = {BATCH_SIZE}")
    print(f"  NUM_WORKERS            = {NUM_WORKERS}")
    print(f"  ENERGY_ANNOT_W         = {ENERGY_ANNOT_W}")    

    # 1. Train
    model, history = train(train_infos, test_infos)

    # 2. Loss curves
    plot_losses(history, "training_loss.png")

    print("\n[DONE]  Output files:")
    print("  training_loss.png        — 7-panel per-epoch loss curves")
    print("  energy_seg_best.pth      — trained model checkpoint (best val loss)")
