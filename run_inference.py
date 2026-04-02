# ============================================================
#  ENERGY-FIELD INSTANCE SEGMENTATION  —  INFERENCE SCRIPT
#  run_inference.py
#
#  Usage:
#      python run_inference.py --exp_dir experiments/baseline_2026...
#      python run_inference.py --exp_dir path --score_thr 0.35
#      python run_inference.py --exp_dir path --split dev-test --num_seqs 5
#
#  Filtering pipeline (in order):
#    1. Score threshold       — drop low-confidence detections
#    2. Per-class NMS         — remove spatially overlapping boxes
#    3. Per-frame class cap   — at most N detections per class per frame
#    4. Track confirmation    — only export tracks seen >= min_hits frames
#    5. Energy sparsification — export top-K energy points per detection
# ============================================================

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import warnings
import random
from collections import defaultdict, OrderedDict

warnings.filterwarnings("ignore")

# Disable torch.compile / dynamo entirely — torchvision detection models
# (RPN anchor generator, RoI pooler) are not compile-compatible and produce
# floods of graph-break warnings with zero speedup benefit.
import torch._dynamo
torch._dynamo.disable()

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

# ── Resolve script directory so imports always work ──────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ── Import model classes only from model.py (no module-level constants there)
_model_spec = importlib.util.spec_from_file_location(
    "model", os.path.join(_SCRIPT_DIR, "model.py"))
_model_mod = importlib.util.module_from_spec(_model_spec)
_model_mod.__name__ = "model"
_model_spec.loader.exec_module(_model_mod)

EnergyInstanceModel   = _model_mod.EnergyInstanceModel
InstanceTracker       = _model_mod.InstanceTracker
get_sequence_infos    = _model_mod.get_sequence_infos
scan_available_frames = _model_mod.scan_available_frames
frame_path            = _model_mod.frame_path

from acoustic_features import AcousticFeatureExtractor, wav_path_from_seq_dir

# ══════════════════════════════════════════════════════════════════════════════
#  MLOps UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS  (mirrors train.py — keep in sync if you change paths there)
# ══════════════════════════════════════════════════════════════════════════════

FRAMES_BASE      = "/data3/scratch/eez086/STARSS23/frames_dev"
LABELS_BASE      = "/data3/scratch/eez086/STARSS23/labels_dev"
MIC_BASE         = "/data3/scratch/eez086/STARSS23/mic_dev"
UPLAM_CHECKPOINT = "UpLAM.pth"

IMG_W            = 360
IMG_H            = 180
N_CHANNELS       = 12
N_ACOUSTIC       = 9
NUM_CLASSES      = 14
DIST_NORM        = 500.0
INFERENCE_HZ     = 10
SCORE_THR        = 0.05    
ENERGY_EXPORT_THR= 0.10

COAST_DECAY      = 0.9

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE    = torch.device("cuda")
    _gpu_name = torch.cuda.get_device_name(0)
    torch.backends.cudnn.benchmark = True
else:
    DEVICE    = torch.device("cpu")
    _gpu_name = "CPU"

# ── Worker count: leave 1 core free, cap at 16 ───────────────────────────────
_CPU_COUNT  = os.cpu_count() or 1
NUM_WORKERS = min(max(_CPU_COUNT - 1, 0), 16)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE DATASET
# ══════════════════════════════════════════════════════════════════════════════

class InferenceDataset(Dataset):
    def __init__(
        self,
        seq_dir          : str,
        seq_name         : str,
        frame_indices    : list,
        uplam_checkpoint : str = UPLAM_CHECKPOINT,
        n_acoustic       : int = N_ACOUSTIC,
        img_w            : int = IMG_W,
        img_h            : int = IMG_H,
    ):
        self.seq_dir          = seq_dir
        self.seq_name         = seq_name
        self.frame_indices    = frame_indices
        self.uplam_checkpoint = uplam_checkpoint
        self.n_acoustic       = n_acoustic
        self.img_w            = img_w
        self.img_h            = img_h
        self.wav_path         = wav_path_from_seq_dir(seq_dir, FRAMES_BASE, MIC_BASE)
        self._extractor       = None   

    def _ensure_extractor(self):
        if self._extractor is None:
            self._extractor = AcousticFeatureExtractor(
                uplam_checkpoint = self.uplam_checkpoint,
                device           = torch.device("cpu"),
                num_bands        = self.n_acoustic,
            )

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        self._ensure_extractor()
        fi   = self.frame_indices[idx]
        path = frame_path(self.seq_dir, self.seq_name, fi)

        if os.path.isfile(path):
            img = Image.open(path).convert("RGB")
            if img.size != (self.img_w, self.img_h):
                img = img.resize((self.img_w, self.img_h), Image.BILINEAR)
            rgb = torch.from_numpy(
                np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        else:
            rng = np.random.default_rng(
                abs(hash(self.seq_name)) % 2**31 + int(fi))
            rgb = torch.from_numpy(
                rng.random((3, self.img_h, self.img_w), dtype=np.float32))

        acoustic = self._extractor.get_frame_bands(self.wav_path, fi)
        tensor   = torch.cat([rgb, acoustic], dim=0)   
        return fi, tensor


def _worker_init(worker_id):
    ds = torch.utils.data.get_worker_info().dataset
    ds._extractor = AcousticFeatureExtractor(
        uplam_checkpoint = ds.uplam_checkpoint,
        device           = torch.device("cpu"),
        num_bands        = ds.n_acoustic,
    )


def _collate(batch):
    fis     = [x[0] for x in batch]
    tensors = [x[1] for x in batch]
    return fis, tensors


def build_loader(seq_dir, seq_name, frame_indices, batch_size, num_workers):
    ds = InferenceDataset(seq_dir=seq_dir, seq_name=seq_name,
                          frame_indices=frame_indices)
    return DataLoader(
        ds,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = torch.cuda.is_available(),
        prefetch_factor    = 4 if num_workers > 0 else None,
        worker_init_fn     = _worker_init if num_workers > 0 else None,
        persistent_workers = num_workers > 0,
        drop_last          = False,
        collate_fn         = _collate,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run trained EnergyInstanceModel on all test sequences.")

    p.add_argument("--exp_dir",    type=str, required=True,
                   help="Path to the experiment directory (e.g., experiments/run_name_timestamp)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Specific checkpoint to load. Defaults to energy_seg_best.pth in exp_dir")
    p.add_argument("--split",      default="test")
    p.add_argument("--num_seqs",   type=int, default=None,
                   help="Number of sequences to randomly sample for inference.")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed for sampling sequences.")

    p.add_argument("--score_thr",          type=float, default=0.35)
    p.add_argument("--nms_iou_thr",        type=float, default=0.30)
    p.add_argument("--max_dets_per_class", type=int,   default=3)
    p.add_argument("--min_hits",           type=int,   default=2)
    p.add_argument("--energy_top_k",       type=int,   default=20)
    p.add_argument("--energy_export_thr",  type=float, default=ENERGY_EXPORT_THR)

    p.add_argument("--iou_thr",     type=float, default=0.3)
    p.add_argument("--max_age",     type=int,   default=5)
    p.add_argument("--coast_decay", type=float, default=COAST_DECAY,
                   help="Confidence decay per coasting frame (default 0.9).")

    p.add_argument("--batch_size",  type=int, default=32,
                   help="Frames per forward pass (default 32 for A100-40GB).")
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                   help=f"DataLoader workers (default {NUM_WORKERS}).")
    p.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, num_classes: int) -> EnergyInstanceModel:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint : {checkpoint_path}")
    model = EnergyInstanceModel(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:    print(f"[WARN] Missing keys    : {missing}")
    if unexpected: print(f"[WARN] Unexpected keys: {unexpected}")

    model.to(DEVICE)
    model.eval()
    print("[INFO] Model ready in eval() mode.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-STAGE FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def filter_detections(det, score_thr, nms_iou_thr, max_dets_per_class):
    boxes       = det.get("boxes",       torch.zeros(0, 4))
    labels      = det.get("labels",      torch.zeros(0, dtype=torch.long))
    scores      = det.get("scores",      torch.zeros(0))
    energy_maps = det.get("energy_maps", torch.zeros(0, 28, 28))
    dist_pred   = det.get("dist_pred",   torch.zeros(0))
    N           = scores.shape[0]

    _empty = {**det,
              "boxes": torch.zeros(0, 4),
              "labels": torch.zeros(0, dtype=torch.long),
              "scores": torch.zeros(0),
              "energy_maps": torch.zeros(0, 28, 28),
              "dist_pred": torch.zeros(0)}

    if N == 0:
        return _empty

    keep = scores >= score_thr
    if not keep.any():
        return _empty

    def sel(t, mask):
        return t[mask] if isinstance(t, torch.Tensor) and t.shape[0] == N else t

    boxes, labels, scores = sel(boxes, keep), sel(labels, keep), sel(scores, keep)
    energy_maps = sel(energy_maps, keep)
    dist_pred   = sel(dist_pred, keep)
    N = scores.shape[0]

    keep_nms = []
    for cls in labels.unique():
        m   = labels == cls
        idx = m.nonzero(as_tuple=True)[0]
        keep_nms.append(idx[nms(boxes[idx], scores[idx], nms_iou_thr)])
    if not keep_nms:
        return _empty
    ki = torch.cat(keep_nms)
    ki = ki[scores[ki].argsort(descending=True)]

    def idx_sel(t):
        return t[ki] if isinstance(t, torch.Tensor) and t.shape[0] >= ki.max() + 1 else t

    boxes, labels, scores = idx_sel(boxes), idx_sel(labels), idx_sel(scores)
    energy_maps, dist_pred = idx_sel(energy_maps), idx_sel(dist_pred)

    keep_cap = []
    for cls in labels.unique():
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        keep_cap.append(idx[:max_dets_per_class])
    if not keep_cap:
        return _empty
    ki2 = torch.cat(keep_cap)

    def idx_sel2(t):
        return t[ki2] if isinstance(t, torch.Tensor) and t.shape[0] >= ki2.max() + 1 else t

    return {**det,
            "boxes": idx_sel2(boxes), "labels": idx_sel2(labels),
            "scores": idx_sel2(scores), "energy_maps": idx_sel2(energy_maps),
            "dist_pred": idx_sel2(dist_pred)}


def sparsify_energy_map(emap, box, top_k, thr):
    x0, y0, x1, y1 = box
    flat  = emap.flatten()
    n_pts = flat.shape[0]

    top_idx = (np.argpartition(flat, -top_k)[-top_k:]
               if 0 < top_k < n_pts else np.arange(n_pts))
    top_idx = top_idx[flat[top_idx] >= thr]

    triplets = []
    for i in top_idx:
        rj, ci = i // 28, i % 28
        # Sub-pixel accurate coordinate extraction for 28x28 RoI features.
        # This completely avoids the exclusive-box out-of-bounds overshoot error
        # while perfectly centering the bins mathematically.
        triplets.append([
            round(x0 + (ci + 0.5) / 28.0 * (x1 - x0), 4),
            round(y0 + (rj + 0.5) / 28.0 * (y1 - y0), 4),
            round(float(flat[i]), 6),
        ])

    if not triplets:
        pk = int(flat.argmax())
        rj, ci = pk // 28, pk % 28
        triplets.append([
            round(x0 + (ci + 0.5) / 28.0 * (x1 - x0), 4),
            round(y0 + (rj + 0.5) / 28.0 * (y1 - y0), 4),
            round(float(flat[pk]), 6),
        ])

    triplets.sort(key=lambda t: -t[2])
    return triplets


# ══════════════════════════════════════════════════════════════════════════════
#  PER-SEQUENCE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference_on_sequence(
    model, seq_dir, seq_name, frame_indices,
    score_thr, nms_iou_thr, max_dets_per_class,
    min_hits, iou_thr, max_age, batch_size, num_workers,
    coast_decay=COAST_DECAY,
):
    assert not model.training
    loader  = build_loader(seq_dir, seq_name, frame_indices, batch_size, num_workers)
    tracker = InstanceTracker(iou_thr=iou_thr, max_age=max_age)
    results = []
    track_score_memory: dict = {}  
    use_amp = torch.cuda.is_available()
    n_raw = n_kept = 0
    seq_t0 = time.time()

    with torch.no_grad():
        pbar = tqdm(loader, total=math.ceil(len(frame_indices)/batch_size),
                    desc=f"  {seq_name[:35]}", unit="batch",
                    dynamic_ncols=True, leave=True)

        for batch_fi, batch_tensors in pbar:
            images_gpu = [t.to(DEVICE, non_blocking=True) for t in batch_tensors]

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds_batch = model(images_gpu, None)

            for fi, det_raw in zip(batch_fi, preds_batch):
                det = {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                       for k, v in det_raw.items()}

                n_raw  += int(det.get("scores", torch.zeros(0)).shape[0])
                det     = filter_detections(det, score_thr, nms_iou_thr,
                                            max_dets_per_class)
                n_kept += int(det.get("scores", torch.zeros(0)).shape[0])

                tracked = tracker.update(det)

                # ── Score memory: Uses 'coasting' flag to bypass decay properly ──
                for obj in tracked:
                    tid = obj["track_id"]
                    
                    if not obj.get("coasting", False):
                        track_score_memory[tid] = float(obj.get("score", 0.0))
                    else:
                        prev = track_score_memory.get(tid, 0.0)
                        decayed = prev * coast_decay
                        track_score_memory[tid] = decayed
                        obj["score"] = decayed

                    obj["hits"] = (tracker.tracks[tid]["hits"]
                                   if tid in tracker.tracks else obj.get("hits", 1))

                full_e = (det["full_energy"].numpy()
                          if "full_energy" in det
                          else np.zeros((IMG_H, IMG_W), dtype=np.float32))

                results.append(dict(
                    frame       = int(fi),
                    time_s      = int(fi) / INFERENCE_HZ,
                    objects     = tracked,
                    full_energy = full_e,
                ))

            elapsed = max(time.time() - seq_t0, 1e-6)
            pbar.set_postfix({
                "fps":    f"{len(results)/elapsed:.1f}",
                "raw":    n_raw,
                "kept":   n_kept,
                "tracks": len(tracker.tracks),
            }, refresh=True)

    elapsed = time.time() - seq_t0
    n = len(frame_indices)
    print(f"  [INFO] {n} frames in {elapsed:.1f}s  ({n/max(elapsed,1e-6):.1f} fps) | "
          f"raw={n_raw}  after_filter={n_kept}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  JSON SERIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def dump_inference_json(results, save_path, min_hits, energy_top_k, energy_export_thr):
    max_hits: dict = {}
    for frame_result in results:
        for obj in frame_result["objects"]:
            tid = obj["track_id"]
            max_hits[tid] = max(max_hits.get(tid, 0), obj.get("hits", 1))

    annotations = []
    for frame_result in results:
        fi = frame_result["frame"]
        for obj in frame_result["objects"]:
            tid = obj["track_id"]
            if max_hits.get(tid, 0) < min_hits:  
                continue
            emap = obj["energy_map"]
            if isinstance(emap, torch.Tensor):
                emap = emap.numpy()
            triplets = sparsify_energy_map(emap, obj["box"],
                                           energy_top_k, energy_export_thr)
            annotations.append({
                "metadata_frame_index": int(fi),
                "instance_id"         : int(obj["track_id"]),
                "category_id"         : int(obj["label"]) - 1,
                "score"               : round(float(obj.get("score", 0.0)), 6),
                "distance"            : round(float(obj["dist_pred"]) * DIST_NORM, 4),
                "segmentation"        : [triplets],
            })

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({"annotations": annotations}, f, indent=2)

    n_frames_out = len({a["metadata_frame_index"] for a in annotations})
    print(f"  [JSON] {len(annotations):5d} annotations across "
          f"{n_frames_out:4d} frames  →  {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results, seq_name, min_hits):
    # Replicate the exact two-pass logic used by dump_inference_json
    max_hits_summary: dict = {}
    for r in results:
        for o in r["objects"]:
            tid = o["track_id"]
            max_hits_summary[tid] = max(max_hits_summary.get(tid, 0), o.get("hits", 1))

    confirmed = [o for r in results for o in r["objects"] 
                 if max_hits_summary.get(o["track_id"], 0) >= min_hits]
                 
    n_frames   = len(results)
    n_active   = sum(1 for r in results
                     if any(max_hits_summary.get(o["track_id"], 0) >= min_hits for o in r["objects"]))
                     
    cat_counts: dict = defaultdict(int)
    for o in confirmed:
        cat_counts[int(o["label"]) - 1] += 1
        
    duration_s = max((r["time_s"] for r in results), default=0.0)
    
    print(f"\n  +-- {seq_name}")
    print(f"  |  Frames          : {n_frames:>6d}  ({duration_s:.1f}s)")
    print(f"  |  Active frames   : {n_active:>6d}  ({100*n_active/max(n_frames,1):.1f}%)")
    print(f"  |  Confirmed objs  : {len(confirmed):>6d}")
    print(f"  |  Unique tracks   : {len({o['track_id'] for o in confirmed}):>6d}")
    if cat_counts:
        print(f"  |  Category counts : "
              + "  ".join(f"C{c}:{n}" for c, n in sorted(cat_counts.items())))
    print(f"  +{'--'*25}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.exp_dir, "energy_seg_best.pth")

    args.out_dir = os.path.join(args.exp_dir, "inference_outputs")
    os.makedirs(args.out_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(args.exp_dir, "inference.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(args.exp_dir, "inference_error.log"), sys.stderr)

    print(f"\n[INFO] Device: {DEVICE}  [{_gpu_name}]")
    print(f"[INFO] Experiment Dir      : {args.exp_dir}")
    print(f"[INFO] Output directory    : {args.out_dir}")
    print(f"[INFO] batch_size          : {args.batch_size}")
    print(f"[INFO] num_workers         : {args.num_workers}")
    print(f"\n[INFO] ── Filtering pipeline ──────────────────────────────────")
    print(f"[INFO]   Stage 1  score_thr          = {args.score_thr}")
    print(f"[INFO]   Stage 2  nms_iou_thr        = {args.nms_iou_thr}")
    print(f"[INFO]   Stage 3  max_dets_per_class = {args.max_dets_per_class}")
    print(f"[INFO]   Stage 4  min_hits           = {args.min_hits}")
    print(f"[INFO]   Stage 5  energy_top_k       = {args.energy_top_k}")
    print(f"[INFO]            energy_export_thr  = {args.energy_export_thr}")
    print(f"[INFO] ── Tracker ─────────────────────────────────────────────")
    print(f"[INFO]   iou_thr     = {args.iou_thr}")
    print(f"[INFO]   max_age     = {args.max_age}")
    print(f"[INFO]   coast_decay = {args.coast_decay}  "
          f"(coasting-frame score multiplier per frame)")
    print(f"[INFO] ────────────────────────────────────────────────────────\n")

    model      = load_model(args.checkpoint, args.num_classes)
    test_infos = get_sequence_infos(args.split, LABELS_BASE, FRAMES_BASE)

    if not test_infos:
        print(f"[ERROR] No sequences found — split='{args.split}'")
        sys.exit(1)

    if args.num_seqs is not None:
        if args.num_seqs >= len(test_infos):
            print(f"[INFO] Requested num_seqs ({args.num_seqs}) >= available "
                  f"({len(test_infos)}). Using all.")
        else:
            print(f"[INFO] Randomly sampling {args.num_seqs} sequences "
                  f"out of {len(test_infos)}")
            test_infos = random.sample(test_infos, args.num_seqs)

    print(f"[INFO] {len(test_infos)} sequences to process.\n")

    wall_t0      = time.time()
    output_files = []
    skipped      = []

    for seq_idx, (json_path, seq_dir, seq_name) in enumerate(test_infos, 1):
        print(f"\n{'='*60}")
        print(f"[{seq_idx:3d}/{len(test_infos)}]  {seq_name}")

        try:
            with open(json_path) as jf:
                ann_data = json.load(jf)
            annotated_ids = sorted(
                {int(a["metadata_frame_index"])
                 for a in ann_data.get("annotations", [])})
        except Exception as exc:
            print(f"  [WARN] JSON unreadable ({exc}) — using disk frames only.")
            annotated_ids = []

        disk_ids = set(scan_available_frames(seq_dir, seq_name))
        all_ids  = sorted(set(annotated_ids) | disk_ids)

        if not all_ids:
            print(f"  [WARN] No frames found — skipping.")
            skipped.append(seq_name)
            continue

        print(f"  Frames: {len(all_ids)}  "
              f"(annotated={len(annotated_ids)}, on-disk={len(disk_ids)})")

        results = run_inference_on_sequence(
            model              = model,
            seq_dir            = seq_dir,
            seq_name           = seq_name,
            frame_indices      = all_ids,
            score_thr          = args.score_thr,
            nms_iou_thr        = args.nms_iou_thr,
            max_dets_per_class = args.max_dets_per_class,
            min_hits           = args.min_hits,
            iou_thr            = args.iou_thr,
            max_age            = args.max_age,
            batch_size         = args.batch_size,
            num_workers        = args.num_workers,
            coast_decay        = args.coast_decay,
        )

        print_summary(results, seq_name, args.min_hits)

        out_json = os.path.join(args.out_dir, f"{seq_name}_inference.json")
        dump_inference_json(
            results           = results,
            save_path         = out_json,
            min_hits          = args.min_hits,
            energy_top_k      = args.energy_top_k,
            energy_export_thr = args.energy_export_thr,
        )
        output_files.append(out_json)

    wall_elapsed = time.time() - wall_t0
    print(f"\n{'='*60}")
    print(f"  DONE — {len(output_files)}/{len(test_infos)} sequences")
    if skipped:
        print(f"  Skipped  : {', '.join(skipped)}")
    print(f"  Wall time: {wall_elapsed:.1f}s  ({wall_elapsed/60:.1f} min)")
    print(f"  Output   : {os.path.abspath(args.out_dir)}")
    for fpath in output_files:
        print(f"    {os.path.basename(fpath):<50s}  "
              f"{os.path.getsize(fpath)/1024:.1f} KB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
