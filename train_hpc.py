import os
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from acoustic_features import AcousticFeatureExtractor
from model import (
    get_sequence_infos,
    EnergySegDataset,
    collate_fn,
    worker_init_fn,
    EnergyInstanceModel
)

# ── 1. AUTO-DETECT HPC RESOURCES ─────────────────────────────────────────────
def detect_resources():
    import psutil

    total_ram_gb = psutil.virtual_memory().total    / 1e9
    cpu_count    = os.cpu_count() or 1

    gpu_vram_gb = 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_vram_gb = props.total_memory / 1e9

    # 1. Cap workers to a safe middle-ground (8 is plenty for an A100 if prefetching)
    num_workers = min(max(cpu_count - 1, 0), 8)

    # 2. Hard-cap the cache. Do NOT trust psutil because Slurm cgroups hide the true limit.
    # 500 per dataset * 9 copies (8 workers + 1 main) * 2 datasets = ~9,000 frames. 
    # This is highly performant but uses a safe ~25-30 GB of RAM.
    cache_per_ds = 500

    # 3. Keep the massive batch size, the VRAM can handle it!
    if   gpu_vram_gb >= 75: batch_size = 256  
    elif gpu_vram_gb >= 40: batch_size = 128
    elif gpu_vram_gb >= 24: batch_size = 32
    elif gpu_vram_gb >= 16: batch_size = 16
    elif gpu_vram_gb >= 8:  batch_size = 8
    else:                   batch_size = 4

    pin = torch.cuda.is_available()

    print(f"\n[TUNING] BATCH_SIZE  = {batch_size}")
    print(f"[TUNING] NUM_WORKERS = {num_workers}")
    print(f"[TUNING] CACHE_MAX   = {cache_per_ds} frames per dataset")
    print(f"[TUNING] pin_memory  = {pin}\n")

    return batch_size, num_workers, cache_per_ds, pin


# ── 2. CONFIG ─────────────────────────────────────────────────────────────────
FRAMES_BASE           = "/data3/scratch/eez086/STARSS23/frames_dev"
LABELS_BASE           = "/data3/scratch/eez086/STARSS23/labels_dev"
MIC_BASE              = "/data3/scratch/eez086/STARSS23/mic_dev"
UPLAM_CHECKPOINT      = "UpLAM.pth"

IMG_W, IMG_H          = 360, 180
N_CHANNELS            = 12
NUM_CLASSES           = 14
NUM_EPOCHS            = 200
PATIENCE              = 10
LR                    = 1e-3

TRAIN_FRAMES_PER_EPOCH = None
VAL_FRAMES_PER_EPOCH   = None

SCORE_THR             = 0.05
ENERGY_ANNOT_W        = 5.0
FULLMAP_ANNOT_W       = 10.0
DIST_NORM             = 500.0
ENERGY_EXPORT_THR     = 0.10
INFERENCE_HZ          = 10
INFERENCE_SEC         = 5


# ── Device Setup ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE    = torch.device("cuda")
    _gpu_note = f"  [{torch.cuda.get_device_name(0)}]"
    # Enable cudnn benchmarking for faster convolutions on fixed input sizes
    torch.backends.cudnn.benchmark = True 
else:
    DEVICE        = torch.device("cpu")
    _mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    _gpu_note     = "  [Apple Silicon — CPU only]" if _mps_available else ""

print(f"[INFO] Device: {DEVICE}{_gpu_note}")


# ── 3. BATCHNORM FREEZE CONTEXT MANAGER ──────────────────────────────────────
@contextmanager
def freeze_bn(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    bns = [m for m in model.modules() if isinstance(m, bn_types)]
    for bn in bns:
        bn.eval()
    try:
        yield
    finally:
        for bn in bns:
            bn.train()


# ── 4. PLOTTING UTILITY ───────────────────────────────────────────────────────
def plot_losses(history: dict, save_path: str = "training_loss.png"):
    keys   = ["total", "loss_energy_mask", "loss_classifier",
              "loss_box_reg", "loss_fullmap", "loss_distance",
              "loss_rpn_box_score"]
    titles = ["Total loss", "★ Energy mask (MSE)", "Classifier",
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
        ax   = axes[i]
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


# ── 5. TRAINING LOOP ──────────────────────────────────────────────────────────
def train(train_infos: list, val_infos: list):

    batch_size, num_workers, cache_max, pin_memory = detect_resources()

    print(f"[TRAIN] NUM_CLASSES = {NUM_CLASSES}")

    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[TRAIN] Mixed precision (AMP) ENABLED")

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
        img_w=IMG_W, img_h=IMG_H, dist_norm=DIST_NORM,
        cache_max_size     = cache_max,
    )
    val_dataset = EnergySegDataset(
        sequence_infos     = val_infos,
        frames_base        = FRAMES_BASE,
        mic_base           = MIC_BASE,
        acoustic_extractor = acoustic_extractor,
        frames_per_epoch   = VAL_FRAMES_PER_EPOCH,
        img_w=IMG_W, img_h=IMG_H, dist_norm=DIST_NORM,
        cache_max_size     = cache_max // 2,
    )

    # ── HPC OPTIMIZATION: Restore DataLoader performance ─────────────────────
    # persistent_workers=True avoids restarting python processes every epoch.
    # prefetch_factor=2 keeps the GPU fed with the next batch proactively.
    # loader_kwargs = dict(
    #     collate_fn         = collate_fn,
    #     num_workers        = num_workers,
    #     pin_memory         = pin_memory,
    #     prefetch_factor    = 2 if num_workers > 0 else None,
    #     worker_init_fn     = worker_init_fn if num_workers > 0 else None,
    #     persistent_workers = True if num_workers > 0 else False,
    # )

    # Base kwargs for the Train Loader
    train_loader_kwargs = dict(
        collate_fn         = collate_fn,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        prefetch_factor    = 3 if num_workers > 0 else None, 
        worker_init_fn     = worker_init_fn if num_workers > 0 else None,
        persistent_workers = True if num_workers > 0 else False,
    )
    
    # Val Loader gets NO persistent workers. This ensures PyTorch destroys 
    # them and frees the RAM immediately after validation finishes.
    val_loader_kwargs = train_loader_kwargs.copy()
    val_loader_kwargs["persistent_workers"] = False
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  **train_loader_kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, **val_loader_kwargs)

    model = EnergyInstanceModel(
        num_classes     = NUM_CLASSES,
        n_channels      = N_CHANNELS,
        img_w=IMG_W, img_h=IMG_H,
        energy_annot_w  = ENERGY_ANNOT_W,
        fullmap_annot_w = FULLMAP_ANNOT_W,
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
        max_lr          = [LR * 0.1, LR],
        steps_per_epoch = max(len(train_loader), 1),
        epochs          = NUM_EPOCHS,
        pct_start       = 0.05,
    )

    history           = defaultdict(list)
    t0                = time.time()
    best_val_loss     = float('inf')
    epochs_no_improve = 0
    loss_keys = [
        "loss_rpn_box_score", "loss_rpn_box_reg",
        "loss_classifier", "loss_box_reg",
        "loss_energy_mask", "loss_fullmap", "loss_distance",
    ]

    def to_float(v):
        return v.item() if isinstance(v, torch.Tensor) else float(v or 0)

    print(f"[TRAIN] Max {NUM_EPOCHS} epochs × {len(train_dataset)} frames | "
          f"batch={batch_size} | workers={num_workers} | device={DEVICE}")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_dataset.reset_epoch()
        val_dataset.reset_epoch()

        # Keep cache clearing occasional so we aren't rebuilding caches unnecessarily
        if epoch % 10 == 1:
            acoustic_extractor.clear_cache()

        # ─── TRAIN ────────────────────────────────────────────────────────────
        model.train()
        ep = defaultdict(float)
        n  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{NUM_EPOCHS} [Train]", leave=False)
        for images, targets in pbar:
            images  = [img.to(DEVICE, non_blocking=pin_memory) for img in images]
            targets = [{k: v.to(DEVICE, non_blocking=pin_memory)
                        if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(images, targets)
                total     = sum(loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()

            ep["total"] += total.item()
            for k in loss_keys:
                ep[k] += to_float(loss_dict.get(k))
            n += 1

            pbar.set_postfix({
                "total":  f"{total.item():.3f}",
                "energy": f"{to_float(loss_dict.get('loss_energy_mask')):.3f}",
                "vmap":   f"{to_float(loss_dict.get('loss_fullmap')):.4f}",
            })

        for k in ["total"] + loss_keys:
            history[k].append(ep[k] / max(n, 1))

        # ─── VALIDATION ───────────────────────────────────────────────────────
        val_loss_sum = 0.0
        val_n        = 0

        if len(val_loader) > 0:
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch:3d}/{NUM_EPOCHS} [Val]", leave=False)
            with torch.no_grad(), freeze_bn(model):
                for images, targets in pbar_val:
                    images  = [img.to(DEVICE, non_blocking=pin_memory) for img in images]
                    targets = [{k: v.to(DEVICE, non_blocking=pin_memory)
                                if isinstance(v, torch.Tensor) else v
                                for k, v in t.items()} for t in targets]

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        loss_dict = model(images, targets)
                        val_total = sum(loss_dict.values())
                    val_loss_sum += val_total.item()
                    val_n        += 1

        val_avg_loss = val_loss_sum / max(val_n, 1) if val_n > 0 else 0.0
        history["val_total"].append(val_avg_loss)

        mem_str = ""
        if use_amp:
            alloc = torch.cuda.memory_allocated(0) / 1e9
            res   = torch.cuda.memory_reserved(0)  / 1e9
            mem_str = f"  VRAM={alloc:.1f}/{res:.1f}GB"

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Train={history['total'][-1]:.4f}  Val={val_avg_loss:.4f} | "
              f"Energy={history['loss_energy_mask'][-1]:.4f}  "
              f"VMap={history['loss_fullmap'][-1]:.4f} | "
              f"{time.time()-t0:.0f}s{mem_str}")

        # ─── EARLY STOPPING ───────────────────────────────────────────────────
        if val_avg_loss < best_val_loss:
            best_val_loss     = val_avg_loss
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
    test_infos  = get_sequence_infos("test",  LABELS_BASE, FRAMES_BASE)

    if not train_infos:
        raise FileNotFoundError(f"No train directories found in {LABELS_BASE}")
    if not test_infos:
        print(f"[WARN] No test directories found in {LABELS_BASE}")

    print(f"\n[MAIN] {len(train_infos)} training sequences | {len(test_infos)} val sequences")

    model, history = train(train_infos, test_infos)
    plot_losses(history, "training_loss.png")

    print("\n[DONE]")
    print("  training_loss.png   — loss curves")
    print("  energy_seg_best.pth — best checkpoint")