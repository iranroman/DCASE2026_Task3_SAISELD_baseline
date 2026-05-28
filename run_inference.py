import argparse
import importlib.util
import os
import sys
import contextlib
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

# ── resolve model.py from same directory ─────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_model_spec = importlib.util.spec_from_file_location(
    "model", os.path.join(_SCRIPT_DIR, "model.py"))
_model_mod = importlib.util.module_from_spec(_model_spec)
_model_mod.__name__ = "model"
_model_spec.loader.exec_module(_model_mod)

EnergyInstanceModel   = _model_mod.EnergyInstanceModel
scan_available_frames = _model_mod.scan_available_frames
frame_path            = _model_mod.frame_path

from acoustic_features import AcousticFeatureExtractor, wav_path_from_seq_dir

# ── constants ─────────────────────────────────────────────────────────────────
FRAMES_BASE      = "/gpfs/scratch/eez086/STARSS23/frames_dev"
MIC_BASE         = "/gpfs/scratch/eez086/STARSS23/mic_dev"
UPLAM_CHECKPOINT = "UpLAM.pth"

IMG_W, IMG_H           = 360, 180    # model input resolution
N_CHANNELS, N_ACOUSTIC = 12, 9
NUM_CLASSES            = 14          # 0=Background(skip), 1–13=foreground

# Submission coordinate space (kept small to reduce JSON size)
EVAL_W, EVAL_H = 100, 50
OUT_W, OUT_H = 360, 180

# Model label → submission category_id:  model label k → category k-1
MODEL_TO_CAT = {k: k - 1 for k in range(1, NUM_CLASSES)} 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ══════════════════════════════════════════════════════════════════════════════
# Dataset / DataLoader  
# ══════════════════════════════════════════════════════════════════════════════

class InferenceDataset(Dataset):
    def __init__(self, seq_dir, seq_name, frame_indices):
        self.seq_dir          = seq_dir
        self.seq_name         = seq_name
        self.frame_indices    = frame_indices
        self.uplam_checkpoint = UPLAM_CHECKPOINT
        self.n_acoustic       = N_ACOUSTIC
        self.img_w, self.img_h = IMG_W, IMG_H
        self.wav_path         = wav_path_from_seq_dir(seq_dir, FRAMES_BASE, MIC_BASE)
        self._extractor       = None

    def _ensure_extractor(self):
        if self._extractor is None:
            self._extractor = AcousticFeatureExtractor(
                uplam_checkpoint=self.uplam_checkpoint,
                device=torch.device("cpu"),
                num_bands=self.n_acoustic,
            )

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        self._ensure_extractor()
        fi = self.frame_indices[idx]
        p  = frame_path(self.seq_dir, self.seq_name, fi)
        if os.path.isfile(p):
            img = Image.open(p).convert("RGB")
            if img.size != (self.img_w, self.img_h):
                img = img.resize((self.img_w, self.img_h), Image.BILINEAR)
            rgb = torch.from_numpy(
                np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        else:
            rgb = torch.zeros((3, self.img_h, self.img_w), dtype=torch.float32)
        ac = self._extractor.get_frame_bands(self.wav_path, fi)
        t  = torch.cat([rgb, ac], dim=0)
        return fi, t


def _worker_init(wid):
    ds = torch.utils.data.get_worker_info().dataset
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        ds._extractor = AcousticFeatureExtractor(
            uplam_checkpoint=ds.uplam_checkpoint,
            device=torch.device("cpu"),
            num_bands=ds.n_acoustic,
        )


def _collate(b):
    return [x[0] for x in b], [x[1] for x in b]


def build_loader(sd, sn, fi, bs, nw):
    ds = InferenceDataset(seq_dir=sd, seq_name=sn, frame_indices=fi)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if nw > 0 else None,
        worker_init_fn=_worker_init if nw > 0 else None,
        collate_fn=_collate,
    )


def load_model(cp: str):
    if not os.path.isfile(cp):
        raise FileNotFoundError(f"Checkpoint not found: {cp}")
    print(f"[INFO] Loading checkpoint: {cp}")
    cw = torch.ones(NUM_CLASSES, dtype=torch.float32, device=DEVICE)
    m  = EnergyInstanceModel(
        num_classes=NUM_CLASSES, n_channels=N_CHANNELS,
        img_w=IMG_W, img_h=IMG_H, class_weights=cw,
    )
    state = torch.load(cp, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict):
        if   "model_state"      in state: state = state["model_state"]
        elif "model_state_dict" in state: state = state["model_state_dict"]
    m.load_state_dict(state, strict=True)
    m.to(DEVICE)
    m.eval()
    return m


# ══════════════════════════════════════════════════════════════════════════════
# NMS & Tracker
# ══════════════════════════════════════════════════════════════════════════════

def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    a1    = (box[2] - box[0]) * (box[3] - box[1])
    a2    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = a1 + a2 - inter + 1e-6
    return inter / union


def nms_per_class(
    boxes:   np.ndarray,   
    labels:  np.ndarray,   
    scores:  np.ndarray,   
    iou_thr: float = 0.45,
) -> np.ndarray:            
    keep = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        s   = scores[idx]
        b   = boxes[idx]
        order = np.argsort(-s)
        alive = np.ones(len(order), dtype=bool)
        for i, oi in enumerate(order):
            if not alive[i]:
                continue
            keep.append(idx[oi])
            ious = box_iou(b[oi], b[order[i + 1:]])
            for j, iou in enumerate(ious):
                if iou > iou_thr:
                    alive[i + 1 + j] = False
    return np.array(keep, dtype=int)


@dataclass
class Track:
    track_id:    int
    label:       int
    score:       float
    box:         np.ndarray   
    emap:        np.ndarray   
    dist_pred:   Optional[float]
    age:         int = 1      
    missed:      int = 0      
    confirmed:   bool = False 


class TemporalTracker:
    def __init__(
        self,
        iou_thr:    float = 0.30,
        min_age:    int   = 2,
        max_missed: int   = 2,
    ):
        self.iou_thr    = iou_thr
        self.min_age    = min_age
        self.max_missed = max_missed
        self._tracks: List[Track] = []
        self._next_id = 0

    def update(
        self,
        boxes:     np.ndarray,    
        labels:    np.ndarray,    
        scores:    np.ndarray,    
        emaps:     np.ndarray,    
        dist_preds: Optional[np.ndarray],  
    ) -> List[Track]:
        n_det = len(labels)
        unmatched_dets  = list(range(n_det))
        matched_track_i = set()

        if self._tracks and n_det > 0:
            track_boxes = np.stack([t.box for t in self._tracks])  
            cost = np.zeros((n_det, len(self._tracks)))
            for di in range(n_det):
                ious = box_iou(boxes[di], track_boxes)
                for ti, t in enumerate(self._tracks):
                    cost[di, ti] = ious[ti] if t.label == labels[di] else 0.0

            flat = np.argsort(-cost.ravel())
            for f in flat:
                di, ti = divmod(int(f), len(self._tracks))
                if cost[di, ti] < self.iou_thr:
                    break
                if di not in unmatched_dets or ti in matched_track_i:
                    continue
                
                t = self._tracks[ti]
                t.box      = boxes[di]
                t.score    = scores[di]
                t.emap     = emaps[di]
                t.dist_pred = float(dist_preds[di]) if dist_preds is not None else None
                t.age      += 1
                t.missed   = 0
                if t.age >= self.min_age:
                    t.confirmed = True
                unmatched_dets.remove(di)
                matched_track_i.add(ti)

        for ti, t in enumerate(self._tracks):
            if ti not in matched_track_i:
                t.missed += 1

        for di in unmatched_dets:
            self._tracks.append(Track(
                track_id  = self._next_id,
                label     = int(labels[di]),
                score     = float(scores[di]),
                box       = boxes[di].copy(),
                emap      = emaps[di].copy(),
                dist_pred = float(dist_preds[di]) if dist_preds is not None else None,
            ))
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]
        return [t for t in self._tracks if t.confirmed]


# ══════════════════════════════════════════════════════════════════════════════
# Peak extraction 
# ══════════════════════════════════════════════════════════════════════════════

def extract_peaks(
    emap_raw:  np.ndarray,   
    box_xyxy:  np.ndarray,   
    n_peaks:   int = 20,
) -> List[List[float]]:
    sx = EVAL_W / IMG_W
    sy = EVAL_H / IMG_H
    ex1 = int(np.clip(round(float(box_xyxy[0]) * sx), 0, EVAL_W - 1))
    ey1 = int(np.clip(round(float(box_xyxy[1]) * sy), 0, EVAL_H - 1))
    ex2 = int(np.clip(round(float(box_xyxy[2]) * sx), 0, EVAL_W))
    ey2 = int(np.clip(round(float(box_xyxy[3]) * sy), 0, EVAL_H))
    bw  = max(ex2 - ex1, 1)
    bh  = max(ey2 - ey1, 1)

    t = torch.from_numpy(emap_raw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(bh, bw), mode="bilinear", align_corners=False)    
    box_energy = t[0, 0].numpy()       

    emin, emax = box_energy.min(), box_energy.max()
    if emax - emin < 1e-8:
        norm = np.ones_like(box_energy)
    else:
        norm = (box_energy - emin) / (emax - emin)

    flat     = norm.ravel()
    n_peaks  = min(n_peaks, flat.size)
    top_flat = np.argpartition(flat, -n_peaks)[-n_peaks:]
    top_flat = top_flat[np.argsort(-flat[top_flat])]

    ox_scale = OUT_W / EVAL_W
    oy_scale = OUT_H / EVAL_H

    triplets = []
    for fi in top_flat:
        py, px = divmod(int(fi), bw)          
        ex = ex1 + px + 0.5                   
        ey = ey1 + py + 0.5
        ox = float(ex * ox_scale)             
        oy = float(ey * oy_scale)
        ox = min(ox, OUT_W - 1.0)
        oy = min(oy, OUT_H - 1.0)
        intensity = float(norm[py, px])
        triplets.append([round(ox, 2), round(oy, 2), round(intensity, 4)])

    return triplets


# ══════════════════════════════════════════════════════════════════════════════
# Per-sequence inference → list of annotation dicts
# ══════════════════════════════════════════════════════════════════════════════

def infer_sequence(
    model,
    sd:          str,
    sn:          str,
    bs:          int,
    nw:          int,
    score_thr:   float,
    nms_iou:     float,
    track_iou:   float,
    min_age:     int,
    max_missed:  int,
    n_peaks:     int,
    dist_scale:  float,   
) -> List[dict]:
    
    fis    = sorted(scan_available_frames(sd, sn))
    loader = build_loader(sd, sn, fis, bs, nw)
    tracker = TemporalTracker(
        iou_thr=track_iou, min_age=min_age, max_missed=max_missed)

    use_amp = torch.cuda.is_available()
    annotations: List[dict] = []

    # ── Pass 1 - Accumulate all predictions for the sequence ──────────────
    frame_data_accum = []
    class_scores = {c: [] for c in range(1, NUM_CLASSES)}

    with torch.no_grad():
        for bf, bt in tqdm(loader, desc=f"  {sn[:40]} [Infer]", unit="batch"):
            imgs = [t.to(DEVICE, non_blocking=True) for t in bt]
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds_batch = model(imgs, None)

            for fi, dr in zip(bf, preds_batch):
                fi = int(fi)

                def _get(keys):
                    for k in keys:
                        if k in dr:
                            v = dr[k]
                            return v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                    return None

                labels_np  = _get(["labels",      "pred_classes", "pred_labels", "classes"])
                scores_np  = _get(["scores",      "pred_scores",  "confidences"])
                boxes_np   = _get(["boxes",       "pred_boxes",   "bboxes"])
                emaps_np   = _get(["energy_maps","energy",        "heatmaps"])
                dist_np    = _get(["dist_pred"])

                if labels_np is not None and scores_np is not None and boxes_np is not None:
                    labels_np = labels_np.astype(int)
                    scores_np = scores_np.astype(np.float32)
                    boxes_np  = boxes_np.astype(np.float32)
                    if emaps_np is not None:
                        emaps_np = emaps_np.astype(np.float32)
                    
                    for l, s in zip(labels_np, scores_np):
                        if l > 0:
                            class_scores[l].append(s)
                else:
                    labels_np, scores_np, boxes_np = None, None, None

                frame_data_accum.append((fi, labels_np, scores_np, boxes_np, emaps_np, dist_np))

    # ── Calculate global min/max for per-class stretching ───────────────
    class_min_max = {}
    for c, s_list in class_scores.items():
        if s_list:
            class_min_max[c] = (float(np.min(s_list)), float(np.max(s_list)))
        else:
            class_min_max[c] = (0.0, 1.0) 
            
    stats_raw = {c: 0 for c in range(1, NUM_CLASSES)}
    stats_thresh = {c: 0 for c in range(1, NUM_CLASSES)}
    stats_nms = {c: 0 for c in range(1, NUM_CLASSES)}
    stats_emitted = {c: 0 for c in range(1, NUM_CLASSES)}

    # ── Pass 2 - Adaptive Gating & Fragment Merging ───────────────────────────
    for fi, labels_np, scores_np, boxes_np, emaps_np, dist_np in frame_data_accum:
        if labels_np is None:
            tracker.update(np.zeros((0,4)), np.zeros(0,int), np.zeros(0), np.zeros((0,28,28)), None)
            continue

        for l in labels_np:
            if l > 0: stats_raw[l] += 1

        # FIX 1: ADAPTIVE GATING 
        # We check the threshold using the scaled score, but we do NOT overwrite scores_np
        # This guarantees the Evaluator receives the RAW score to perfectly preserve mAP ranking
        keep_mask = np.zeros(len(scores_np), dtype=bool)

        for i in range(len(scores_np)):
            c = labels_np[i]
            raw_s = scores_np[i]
            if c > 0 and c in class_min_max:
                cmin, cmax = class_min_max[c]
                # Scale internally to "encourage" underconfident classes
                if cmax >= 0.08 and cmax > cmin:
                    scaled_s = 0.05 + 0.90 * ((raw_s - cmin) / (cmax - cmin))
                else:
                    scaled_s = raw_s
                
                # Check threshold against the SCALED score
                if scaled_s >= score_thr:
                    keep_mask[i] = True

        keep_fg = labels_np > 0
        keep    = keep_mask & keep_fg
        
        for l in labels_np[keep]: stats_thresh[l] += 1

        if keep.sum() == 0:
            tracker.update(np.zeros((0,4)), np.zeros(0,int), np.zeros(0), np.zeros((0,28,28)), None)
            continue

        # Extract features using the mask. scores_f contains the UNMODIFIED RAW scores.
        labels_f = labels_np[keep]
        scores_f = scores_np[keep]  
        boxes_f  = boxes_np[keep]
        emaps_f  = emaps_np[keep]  if emaps_np  is not None else np.zeros((keep.sum(),28,28))
        dist_f   = dist_np[keep]   if dist_np   is not None else None

        nms_idx  = nms_per_class(boxes_f, labels_f, scores_f, iou_thr=nms_iou)
        labels_f, scores_f, boxes_f = labels_f[nms_idx], scores_f[nms_idx], boxes_f[nms_idx]
        emaps_f, dist_f = emaps_f[nms_idx], dist_f[nms_idx] if dist_f is not None else None
        
        for l in labels_f: stats_nms[l] += 1

        confirmed = tracker.update(boxes_f, labels_f, scores_f, emaps_f, dist_f)

        # FIX 2: SPATIAL FRAGMENT MERGING
        # Group confirmed tracks by category to fix the 1-to-1 Hungarian matching penalty
        emitted_by_cat = defaultdict(list)
        for track in confirmed:
            cat_id = MODEL_TO_CAT.get(track.label)
            if cat_id is None: continue
            
            triplets = extract_peaks(track.emap, track.box, n_peaks=n_peaks)
            if not triplets: continue

            emitted_by_cat[cat_id].append({
                "box": track.box,
                "score": float(track.score),
                "triplets": triplets,
                "dist": float(track.dist_pred) * dist_scale if track.dist_pred is not None else None
            })

        for cat_id, fragments in emitted_by_cat.items():
            # Cluster adjacent fragments (centers within 50 pixels)
            clusters = []
            for frag in fragments:
                cx = (frag['box'][0] + frag['box'][2]) / 2.0
                cy = (frag['box'][1] + frag['box'][3]) / 2.0
                placed = False
                for clus in clusters:
                    for c_frag in clus:
                        ccx = (c_frag['box'][0] + c_frag['box'][2]) / 2.0
                        ccy = (c_frag['box'][1] + c_frag['box'][3]) / 2.0
                        if np.hypot(cx - ccx, cy - ccy) < 50.0:
                            clus.append(frag)
                            placed = True
                            break
                    if placed: break
                if not placed:
                    clusters.append([frag])
            
            # Emit one cohesive annotation per cluster
            for clus in clusters:
                all_triplets = []
                max_score = -1.0
                dists = []
                
                for frag in clus:
                    all_triplets.extend(frag['triplets'])
                    if frag['score'] > max_score: max_score = frag['score']
                    if frag['dist'] is not None: dists.append(frag['dist'])
                
                # Sort descending by energy to keep the best peaks of the merged blob
                all_triplets.sort(key=lambda x: x[2], reverse=True)
                
                # Find original class ID for logging
                original_class_id = next(k for k, v in MODEL_TO_CAT.items() if v == cat_id)
                stats_emitted[original_class_id] += 1
                
                entry: dict = {
                    "metadata_frame_index": fi,
                    "category_id":          cat_id,
                    "score":                round(max_score, 5), # Pure RAW score
                    "segmentation":         [all_triplets[:n_peaks * 2]], # Allow double peaks for merged blobs
                }
                if dists:
                    entry["distance"] = round(sum(dists) / len(dists), 1)

                annotations.append(entry)

    print("\n  → Per-Class Detection Funnel:")
    print("    Class |   Raw | >Thr |  NMS | Emitted | Raw Score Range")
    print("    " + "-" * 61)
    for c in range(1, NUM_CLASSES):
        if stats_raw[c] > 0 or stats_emitted[c] > 0:
            rmin, rmax = class_min_max[c]
            range_str = f"({rmin:.4f}, {rmax:.4f})"
            print(f"    {c:5d} | {stats_raw[c]:5d} | {stats_thresh[c]:4d} | {stats_nms[c]:4d} | {stats_emitted[c]:7d} | {range_str}")
    print()

    return annotations


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Submission-format inference — EnergyInstanceModel")
    
    parser.add_argument("--checkpoint",  default="experiments/audiotuned2_20260406_113306/energy_seg_best.pth")
    parser.add_argument("--split",       default="test")
    parser.add_argument("--n_seqs",      type=int,   default=-1)
    parser.add_argument("--output_dir",  default="submission_output")
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--num_workers", type=int,   default=4)
    
    parser.add_argument("--score_thr",   type=float, default=0.65)
    parser.add_argument("--nms_iou",     type=float, default=0.45)
    parser.add_argument("--track_iou",   type=float, default=0.30)
    parser.add_argument("--min_age",     type=int,   default=2)
    parser.add_argument("--max_missed",  type=int,   default=2)
    parser.add_argument("--n_peaks",     type=int,   default=20)
    parser.add_argument("--dist_scale",  type=float, default=1000.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_folders   = [d for d in os.listdir(FRAMES_BASE)
                      if os.path.isdir(os.path.join(FRAMES_BASE, d))]
    matched_splits = [d for d in base_folders if args.split in d]
    if not matched_splits:
        print(f"[ERROR] No directories matched split '{args.split}' in {FRAMES_BASE}")
        sys.exit(1)

    all_seqs: List[Tuple[str, str]] = []
    for split_dir in sorted(matched_splits):
        fsd = os.path.join(FRAMES_BASE, split_dir)
        for sn in sorted(os.listdir(fsd)):
            sd_full = os.path.join(fsd, sn)
            if os.path.isdir(sd_full):
                all_seqs.append((sd_full, sn))

    all_seqs.sort(key=lambda x: x[1])  
    
    if 0 < args.n_seqs < len(all_seqs):
        indices = np.linspace(0, len(all_seqs) - 1, args.n_seqs).astype(int)
        selected = [all_seqs[i] for i in indices]
    else:
        selected = all_seqs

    model = load_model(args.checkpoint)

    total_annots = 0
    for seq_idx, (sd, sn) in enumerate(selected, 1):
        print(f"[{seq_idx}/{len(selected)}] {sn}")
        annotations = infer_sequence(
            model       = model,
            sd          = sd,
            sn          = sn,
            bs          = args.batch_size,
            nw          = args.num_workers,
            score_thr   = args.score_thr,
            nms_iou     = args.nms_iou,
            track_iou   = args.track_iou,
            min_age     = args.min_age,
            max_missed  = args.max_missed,
            n_peaks     = args.n_peaks,
            dist_scale  = args.dist_scale,
        )

        json_path = out_dir / f"{sn}.json"
        with open(json_path, "w") as f:
            json.dump({"annotations": annotations}, f,
                      separators=(",", ":"))   

        n = len(annotations)
        total_annots += n
        size_kb = json_path.stat().st_size / 1024
        print(f"  → File footprint    : {size_kb:>7.1f} KB  →  {json_path.name}")

    print(f"\n[DONE] {len(selected)} JSON files written to {out_dir}/")
    print(f"       Total annotations : {total_annots:,}")

if __name__ == "__main__":
    main()
