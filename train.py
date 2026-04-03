import os
import sys
import time
import warnings
import math
import random
import argparse
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
from augmentation import SeldAugmentor
from model import (
    get_sequence_infos,
    EnergySegDataset,
    collate_fn,
    worker_init_fn,
    EnergyInstanceModel,
)

# ── 0. MLOps UTILITIES ────────────────────────────────────────────────────────
class Logger(object):
    """Routes stdout/stderr to both the console and a specified log file."""
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def set_seed(seed=42):
    """Ensures deterministic runs for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── 1. AUTO-DETECT HPC RESOURCES ─────────────────────────────────────────────
def detect_resources():
    import psutil

    total_ram_gb = psutil.virtual_memory().total / 1e9
    cpu_count    = os.cpu_count() or 1

    gpu_vram_gb = 0.0
    if torch.cuda.is_available():
        props       = torch.cuda.get_device_properties(0)
        gpu_vram_gb = props.total_memory / 1e9

    num_workers  = min(max(cpu_count - 1, 0), 8)
    cache_per_ds = min(500, max(100, int(total_ram_gb * 5)))

    if   gpu_vram_gb >= 40: batch_size = 64
    elif gpu_vram_gb >= 24: batch_size = 32
    elif gpu_vram_gb >= 16: batch_size = 16
    elif gpu_vram_gb >=  8: batch_size = 8
    else:                   batch_size = 4

    pin = torch.cuda.is_available()

    print(f"\n[TUNING] BATCH_SIZE  = {batch_size}")
    print(f"[TUNING] NUM_WORKERS = {num_workers}")
    print(f"[TUNING] CACHE_MAX   = {cache_per_ds} frames per dataset")
    print(f"[TUNING] RAM         = {total_ram_gb:.1f} GB")
    print(f"[TUNING] pin_memory  = {pin}\n")

    return batch_size, num_workers, cache_per_ds, pin


# ── 2. CONFIG ─────────────────────────────────────────────────────────────────
FRAMES_BASE      = "/gpfs/scratch/eez086/STARSS23/frames_dev"
LABELS_BASE      = "/gpfs/scratch/eez086/STARSS23/labels_dev"
MIC_BASE         = "/gpfs/scratch/eez086/STARSS23/mic_dev"
UPLAM_CHECKPOINT = "UpLAM.pth"

IMG_W, IMG_H     = 360, 180
N_CHANNELS       = 12
N_ACOUSTIC       = 9
NUM_CLASSES      = 14
NUM_EPOCHS       = 200
PATIENCE         = 10          

TRAIN_FRAMES_PER_EPOCH = 15000
VAL_FRAMES_PER_EPOCH   = None

LR               = 1e-4
SCORE_THR        = 0.05
ENERGY_ANNOT_W   = 5.0
FULLMAP_ANNOT_W  = 10.0 
DIST_NORM        = 500.0
ENERGY_EXPORT_THR= 0.10
INFERENCE_HZ     = 10
INFERENCE_SEC    = 5

# ── Backbone progressive-unfreezing schedule ──────────────────────────────────
UNFREEZE_LAYER4_EPOCH = 2
UNFREEZE_LAYER3_EPOCH = 4   
PHASE_GRACE = 4


# ── Device Setup ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE    = torch.device("cuda")
    _gpu_note = f"  [{torch.cuda.get_device_name(0)}]"
    torch.backends.cudnn.benchmark = True
else:
    DEVICE        = torch.device("cpu")
    _mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    _gpu_note     = "  [Apple Silicon — CPU only]" if _mps_available else ""

print(f"[INFO] Device: {DEVICE}{_gpu_note}")


# ── 3. EVAL BEHAVIOR CONTEXT MANAGER (BN + DROPOUT) ──────────────────────────
@contextmanager
def eval_behavior_for_loss(model):
    target_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.Dropout, nn.Dropout2d, nn.Dropout3d,
    )
    modules     = [m for m in model.modules() if isinstance(m, target_types)]
    orig_states = {m: m.training for m in modules}
    for m in modules:
        m.eval()
    try:
        yield
    finally:
        for m, state in orig_states.items():
            m.train(mode=state)


# ── 4. BACKBONE PROGRESSIVE UNFREEZING ───────────────────────────────────────
def apply_backbone_freeze(model: nn.Module, epoch: int):
    body = model.det.backbone.body

    for p in list(body.conv1.parameters()) + list(body.bn1.parameters()):
        p.requires_grad = True

    for module in [body.layer1, body.layer2]:
        for p in module.parameters():
            p.requires_grad = False

    layer4_active = epoch >= UNFREEZE_LAYER4_EPOCH
    for p in body.layer4.parameters():
        p.requires_grad = layer4_active

    layer3_active = epoch >= UNFREEZE_LAYER3_EPOCH
    for p in body.layer3.parameters():
        p.requires_grad = layer3_active

    for p in model.det.backbone.fpn.parameters():
        p.requires_grad = True

    phase = "heads+FPN only"
    if layer3_active:
        phase = "layer3+4 + heads+FPN"
    elif layer4_active:
        phase = "layer4 + heads+FPN"

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    return phase, n_trainable, n_total


# ── 5. OPTIMIZER HELPERS ──────────────────────────────────────────────────────
def _get_pg(optimizer, name):
    return next(g for g in optimizer.param_groups if g.get("name") == name)

def freeze_backbone_bn(model: nn.Module):
    body = model.det.backbone.body
    for module in [body.layer1, body.layer2]:
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()


# ── 6. PLOTTING UTILITY ───────────────────────────────────────────────────────
def plot_losses(history: dict, save_path: str = "training_loss.png"):
    base_keys = [k for k in history.keys() if not k.startswith("val_") and k != "total"]
    all_keys  = ["total"] + sorted(base_keys)
    epochs    = range(1, len(history.get("total", [])) + 1)

    n_plots = len(all_keys)
    cols    = 4
    rows    = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axes      = np.array(axes).flatten()
    fig.suptitle("Training & Validation Loss Curves", fontsize=15, y=1.02)

    def smooth(v, k=7):
        if len(v) < k:
            return v, 0
        k = max(3, min(k, (len(v) // 3) * 2 + 1) | 1)
        return np.convolve(v, np.ones(k) / k, mode="valid"), k // 2

    for i, key in enumerate(all_keys):
        ax    = axes[i]
        vals  = history.get(key, [])
        title = "Total Loss" if key == "total" else key.replace("loss_", "").replace("_", " ").capitalize()

        ax.plot(list(epochs), vals, lw=1, color="steelblue", alpha=0.4, label="train (raw)")
        if len(vals) >= 5:
            s, pad = smooth(vals)
            ax.plot(list(range(pad + 1, len(vals) - pad + 1)), s,
                    lw=2, color="orangered", label="train (smoothed)")

        val_key = "val_total" if key == "total" else f"val_{key}"
        if val_key in history and len(history[val_key]) == len(epochs):
            ax.plot(list(epochs), history[val_key], lw=2, color="green", label="validation")

        ax.legend(fontsize=7)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Loss curves → {save_path}")


# ── 7. TRAINING LOOP ──────────────────────────────────────────────────────────
def train(train_infos: list, val_infos: list, exp_dir: str):

    best_model_path = os.path.join(exp_dir, "energy_seg_best.pth")

    batch_size, num_workers, cache_max, pin_memory = detect_resources()
    print(f"[TRAIN] NUM_CLASSES = {NUM_CLASSES}")

    use_amp = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("[TRAIN] Mixed precision (AMP) ENABLED")

    # ── Acoustic extractor (shared, CPU-side) ─────────────────────────────────
    acoustic_extractor = AcousticFeatureExtractor(
        uplam_checkpoint = UPLAM_CHECKPOINT,
        device           = torch.device("cpu"),
        num_bands        = N_ACOUSTIC,
    )
    acoustic_extractor.clear_cache()

    # ── Augmentor (train only) ────────────────────────────────────────────────
    train_augmentor = SeldAugmentor(
        img_w                 = IMG_W,
        img_h                 = IMG_H,
        n_acoustic            = N_ACOUSTIC,
        azimuth_rotate        = True,
        hflip_prob            = 0.5,
        max_bands_masked      = 4,
        intensity_scale_range = (0.6, 1.4),
        acoustic_noise_std    = 0.03,
        rgb_jitter_prob       = 0.8,
    )
    print(f"[TRAIN] Augmentation: azimuth_rotate=True | hflip_prob=0.5 | rgb_jitter=0.8 | "
          f"max_bands_masked=4 | intensity_scale=(0.6,1.4) | acoustic_noise_std=0.03")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = EnergySegDataset(
        sequence_infos     = train_infos,
        frames_base        = FRAMES_BASE,
        mic_base           = MIC_BASE,
        acoustic_extractor = acoustic_extractor,
        frames_per_epoch   = TRAIN_FRAMES_PER_EPOCH,
        img_w=IMG_W, img_h=IMG_H,
        dist_norm          = DIST_NORM,
        cache_max_size     = cache_max,
        augmentor          = train_augmentor,
    )
    
    val_dataset = EnergySegDataset(
        sequence_infos     = val_infos,
        frames_base        = FRAMES_BASE,
        mic_base           = MIC_BASE,
        acoustic_extractor = acoustic_extractor,
        frames_per_epoch   = VAL_FRAMES_PER_EPOCH,
        img_w=IMG_W, img_h=IMG_H,
        dist_norm          = DIST_NORM,
        cache_max_size     = max(1, cache_max // 2),
        augmentor          = None,
    )
    
    val_dataset.current_indices = np.arange(len(val_dataset.all_samples))
    if VAL_FRAMES_PER_EPOCH is not None:
        val_dataset.current_indices = val_dataset.current_indices[:VAL_FRAMES_PER_EPOCH]

    loader_kwargs = dict(
        collate_fn         = collate_fn,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        prefetch_factor    = 3 if num_workers > 0 else None,
        worker_init_fn     = worker_init_fn if num_workers > 0 else None,
        persistent_workers = False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    # shuffle the validation dataset so that the progress bar shows steady behavior
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=True, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EnergyInstanceModel(
        num_classes     = NUM_CLASSES,
        n_channels      = N_CHANNELS,
        img_w=IMG_W, img_h=IMG_H,
        energy_annot_w  = ENERGY_ANNOT_W,
        fullmap_annot_w = FULLMAP_ANNOT_W,        
    ).to(DEVICE)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    body = model.det.backbone.body
    
    backbone_stem_params = list(body.conv1.parameters()) + list(body.bn1.parameters())
    backbone_l3_params = list(body.layer3.parameters())
    backbone_l4_params = list(body.layer4.parameters())
    fpn_params         = list(model.det.backbone.fpn.parameters())

    rpn_params       = list(model.det.rpn.parameters())
    box_head_params  = list(model.det.roi_heads.box_head.parameters())
    box_pred_params  = list(model.det.roi_heads.box_predictor.parameters())
    
    custom_params    = (
        list(model.acoustic_norm.parameters()) + 
        list(model.energy_head.parameters()) +
        list(model.energy_decoder.parameters()) +
        list(model.distance_head.parameters())
    )

    optimizer = optim.AdamW([
        {"params": backbone_stem_params,   "lr": LR*0.2,   "weight_decay": 1e-4, "name": "bb_stem"},
        {"params": backbone_l3_params,     "lr": LR*0.05,  "weight_decay": 1e-4, "name": "bb_l3"},
        {"params": backbone_l4_params,     "lr": LR*0.1,   "weight_decay": 1e-4, "name": "bb_l4"},
        {"params": fpn_params,             "lr": LR*0.2,   "weight_decay": 1e-4, "name": "fpn"},
        {"params": rpn_params,             "lr": LR*0.1,   "weight_decay": 1e-3, "name": "rpn"},
        {"params": box_head_params,        "lr": LR*0.02,  "weight_decay": 5e-2, "name": "box_head"},
        {"params": box_pred_params,        "lr": LR*0.02,  "weight_decay": 5e-2, "name": "box_pred"},
        {"params": custom_params,          "lr": LR*0.5,   "weight_decay": 1e-2, "name": "custom"},
    ])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = 0.5,
        patience = 3,         
        min_lr   = 1e-8,
        verbose  = True,
    )

    history           = defaultdict(list)
    t0                = time.time()
    best_loss         = float("inf")
    epochs_no_improve = 0
    prev_phase        = None

    def to_float(v):
        return v.item() if isinstance(v, torch.Tensor) else float(v or 0)

    def abbreviate_loss_name(name):
        # More aggressive abbreviation to save space
        return (name.replace("loss_", "")
                    .replace("classifier", "cls")
                    .replace("objectness", "obj")
                    .replace("rpn_box_reg", "rpn_box")
                    .replace("box_reg", "box")
                    .replace("distance", "dist")
                    .replace("energy_mask", "mask")
                    .replace("fullmap", "fmap")[:7])

    print(f"[TRAIN] Max {NUM_EPOCHS} epochs × {len(train_dataset)} frames | "
          f"batch={batch_size} | workers={num_workers} | device={DEVICE}")

    for epoch in range(1, NUM_EPOCHS + 1):
                
        train_dataset.reset_epoch(balanced=False)

        if epoch % 10 == 1:
            acoustic_extractor.clear_cache()

        # ── Apply (or update) backbone freeze schedule ─────────────────────────
        phase, n_trainable, n_total = apply_backbone_freeze(model, epoch)
        if phase != prev_phase:
            print(f"\n[FREEZE] Epoch {epoch}: training phase → '{phase}' "
                  f"({n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M params active)")

            epochs_no_improve = max(0, epochs_no_improve - PHASE_GRACE)
            scheduler.num_bad_epochs = max(0, scheduler.num_bad_epochs - PHASE_GRACE)
            
            print(f"[FREEZE] Phase grace applied: patience counter → {epochs_no_improve}/{PATIENCE}")

            if epoch == UNFREEZE_LAYER4_EPOCH:
                print(f"[FREEZE] layer4 active. (head LR = {_get_pg(optimizer, 'custom')['lr']:.2e})")

            if epoch == UNFREEZE_LAYER3_EPOCH:
                print(f"[FREEZE] layer3 active. (head LR = {_get_pg(optimizer, 'custom')['lr']:.2e})")

            prev_phase = phase

        # ─── TRAIN ────────────────────────────────────────────────────────────
        model.train()
        freeze_backbone_bn(model)
        
        ep_losses = defaultdict(float)
        n = 0

        # Send tqdm explicitly to the original un-redirected stdout and use dynamic columns
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{NUM_EPOCHS} [Train]", 
                    leave=False, file=sys.__stdout__, dynamic_ncols=True)
        for images, targets in pbar:
            images  = [img.to(DEVICE, non_blocking=pin_memory) for img in images]
            targets = [
                {k: v.to(DEVICE, non_blocking=pin_memory) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()}
                for t in targets
            ]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_dict = model(images, targets)
                total     = sum(loss_dict.values())

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()

            ep_losses["total"] += total.item()
            for k, v in loss_dict.items():
                ep_losses[k] += to_float(v)
            n += 1

            # Calculate running average instead of showing instantaneous batch value
            postfix = {"Tot": f"{ep_losses['total']/n:.3f}"}
            for k in loss_dict.keys():
                postfix[abbreviate_loss_name(k)] = f"{ep_losses[k]/n:.3f}"
            pbar.set_postfix(postfix)

        for k, v in ep_losses.items():
            history[k].append(v / max(n, 1))

        # ─── VALIDATION ───────────────────────────────────────────────────────
        val_ep_losses = defaultdict(float)
        val_n         = 0

        if len(val_loader) > 0:
            # Same tqdm settings for validation
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch:3d}/{NUM_EPOCHS} [Val]", 
                            leave=False, file=sys.__stdout__, dynamic_ncols=True)
            with torch.no_grad(), eval_behavior_for_loss(model):
                for images, targets in pbar_val:
                    images  = [img.to(DEVICE, non_blocking=pin_memory) for img in images]
                    targets = [
                        {k: v.to(DEVICE, non_blocking=pin_memory) if isinstance(v, torch.Tensor) else v
                         for k, v in t.items()}
                        for t in targets
                    ]

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        loss_dict = model(images, targets)
                        val_total = sum(loss_dict.values())

                    val_ep_losses["total"] += val_total.item()
                    for k, v in loss_dict.items():
                        val_ep_losses[k] += to_float(v)
                    val_n += 1

                    # Validation running average
                    postfix_val = {"Tot": f"{val_ep_losses['total']/val_n:.3f}"}
                    for k in loss_dict.keys():
                        postfix_val[abbreviate_loss_name(k)] = f"{val_ep_losses[k]/val_n:.3f}"
                    pbar_val.set_postfix(postfix_val)

            for k, v in val_ep_losses.items():
                history[f"val_{k}"].append(v / max(val_n, 1))        

        # ─── EPOCH SUMMARY ────────────────────────────────────────────────────
        print(f"\n[{time.strftime('%H:%M:%S')}] Epoch {epoch}/{NUM_EPOCHS} Completed "
              f"({time.time()-t0:.1f}s) | phase={phase}")

        train_tot = history["total"][-1]
        train_sum = sum(history[k][-1] for k in ep_losses if k != "total")
        print(f"  [Train] Verification: Sum({train_sum:.4f}) == Total({train_tot:.4f})")
        for k in sorted(ep_losses):
            if k != "total":
                print(f"          - {k:<28}: {history[k][-1]:.4f}")

        val_avg_total = float("inf")
        if val_n > 0:
            val_avg_total = history["val_total"][-1]
            val_sum       = sum(history[f"val_{k}"][-1] for k in val_ep_losses if k != "total")
            print(f"  [Val]   Verification: Sum({val_sum:.4f}) == Total({val_avg_total:.4f})")
            for k in sorted(val_ep_losses):
                if k != "total":
                    print(f"          - {k:<28}: {history[f'val_{k}'][-1]:.4f}")

        # Explicit LR Logging
        head_lr = _get_pg(optimizer, "custom")["lr"]
        fpn_lr  = _get_pg(optimizer, "fpn")["lr"]
        print(f"  [LR] heads={head_lr:.2e} | FPN={fpn_lr:.2e}")

        # ─── EARLY STOPPING & SCHEDULING ──────────────────────────────────────
        track_raw = val_avg_total if val_n > 0 else train_tot
        
        scheduler.step(track_raw)

        if track_raw < best_loss:
            best_loss         = track_raw
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  [*] Epoch loss improved to {best_loss:.4f}. Model saved → {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"  [!] No improvement for {epochs_no_improve}/{PATIENCE} epochs. (Best: {best_loss:.4f}, Curr: {track_raw:.4f})")
            if epochs_no_improve >= PATIENCE:
                print(f"\n[!] Early stopping triggered after {epoch} epochs.")
                break

    print(f"\n[TRAIN] Finished in {time.time()-t0:.1f}s")
    if os.path.exists(best_model_path):
        model.load_state_dict(
            torch.load(best_model_path, map_location=DEVICE, weights_only=True)
        )
    return model, dict(history)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Train Energy Instance Model")
    parser.add_argument("--exp_name", type=str, default="baseline", help="Name of the experiment run")
    parser.add_argument("--seed",     type=int, default=42,         help="Random seed for reproducibility")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir   = os.path.join(os.getcwd(), "experiments", f"{args.exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(exp_dir, "train.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(exp_dir, "error.log"), sys.stderr)

    print(f"[MAIN] Starting experiment: {args.exp_name}")
    print(f"[MAIN] Output directory:    {exp_dir}")

    set_seed(args.seed)

    train_infos = get_sequence_infos("train", LABELS_BASE, FRAMES_BASE)
    test_infos  = get_sequence_infos("test",  LABELS_BASE, FRAMES_BASE)

    if not train_infos:
        raise FileNotFoundError(f"No train directories found in {LABELS_BASE}")
    if not test_infos:
        print(f"[WARN] No val directories found in {LABELS_BASE}")

    print(f"\n[MAIN] {len(train_infos)} training sequences | {len(test_infos)} val sequences")

    model, history = train(train_infos, test_infos, exp_dir)
    plot_losses(history, os.path.join(exp_dir, "training_loss.png"))

    print("\n[DONE]")
    print(f"  All artifacts (logs, checkpoints, loss curves) saved to: {exp_dir}")
