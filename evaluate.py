import argparse
import json
import os
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

CANVAS_W, CANVAS_H = 100, 50
OUT_W, OUT_H       = 360, 180
MAX_JSON_BYTES     = 20 * 1024 * 1024   # 20 MB

DEFAULT_IOU_THRESHOLDS = np.round(np.arange(0.50, 1.00, 0.05), 2)

CLASS_NAMES = [
    "Female speech",       # 0
    "Male speech",         # 1
    "Clapping",            # 2
    "Telephone",           # 3
    "Laughter",            # 4
    "Domestic sounds",     # 5
    "Walk / footsteps",    # 6
    "Door open/close",     # 7
    "Music",               # 8
    "Musical instrument",  # 9
    "Water tap",           # 10
    "Bell",                # 11
    "Knock",               # 12
]
N_CLASSES = len(CLASS_NAMES)

DARK_BG  = "#0e0e0e"
PANEL_BG = "#1a1a1a"
WHITE    = "#e8e8e8"
CMAP_CLS = matplotlib.colormaps.get_cmap("tab20")


def cls_color(c: int):
    return CMAP_CLS(c / max(N_CLASSES - 1, 1))


# ---------------------------------------------------------------------------
# IoU threshold parsing
# ---------------------------------------------------------------------------

def parse_iou_thresholds(spec: str) -> np.ndarray:
    spec = spec.strip()
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"IoU range must be start:step:end  (got '{spec}')"
            )
        start, step, end = float(parts[0]), float(parts[1]), float(parts[2])
        thrs = np.round(np.arange(start, end + step * 0.5, step), 6)
        thrs = thrs[thrs <= end + 1e-9]
    elif "," in spec:
        thrs = np.array(sorted(set(round(float(v), 6) for v in spec.split(","))))
    else:
        thrs = np.array([round(float(spec), 6)])
    thrs = np.clip(thrs, 0.0, 1.0)
    if len(thrs) == 0:
        raise argparse.ArgumentTypeError("IoU threshold list is empty.")
    return thrs


# ---------------------------------------------------------------------------
# JSON loading + validation
# ---------------------------------------------------------------------------

def _validate_annotation_coords(annots: List[dict], source_label: str) -> None:
    """
    Raise ValueError if any coordinate in any polygon falls outside the
    canonical spatial range [0, OUT_W] x [0, OUT_H].
    We check x against OUT_W (360) and y against OUT_H (180).
    The third value per triplet is an intensity — not a coordinate — so it
    is deliberately not range-checked here.
    """
    for i, a in enumerate(annots):
        for poly in a.get("segmentation", []):
            if not poly:
                continue
            pts = np.array(poly, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError(
                    f"[{source_label}] annotation index {i}: "
                    f"segmentation polygon must be an array of [x, y, intensity] triplets, "
                    f"got shape {pts.shape}."
                )
            xs = pts[:, 0]
            ys = pts[:, 1]
            bad_x = np.where((xs < 0) | (xs > OUT_W))[0]
            bad_y = np.where((ys < 0) | (ys > OUT_H))[0]
            if len(bad_x):
                raise ValueError(
                    f"[{source_label}] annotation index {i}: "
                    f"x coordinate(s) at triplet indices {bad_x.tolist()} fall outside "
                    f"the valid spatial range [0, {OUT_W}].  "
                    f"Offending values: {xs[bad_x].tolist()}.  "
                    f"All coordinates MUST be expressed in the {OUT_W}×{OUT_H} spatial space."
                )
            if len(bad_y):
                raise ValueError(
                    f"[{source_label}] annotation index {i}: "
                    f"y coordinate(s) at triplet indices {bad_y.tolist()} fall outside "
                    f"the valid spatial range [0, {OUT_H}].  "
                    f"Offending values: {ys[bad_y].tolist()}.  "
                    f"All coordinates MUST be expressed in the {OUT_W}×{OUT_H} spatial space."
                )


def load_pred_json(path: str) -> List[dict]:
    size = os.path.getsize(path)
    if size > MAX_JSON_BYTES:
        raise ValueError(
            f"Prediction JSON '{path}' is {size / 1024 / 1024:.2f} MB, "
            f"which exceeds the 20 MB limit.  "
            f"JSON files must be under 20 MB."
        )
    with open(path) as f:
        data = json.load(f)
    annots = data.get("annotations", [])
    _validate_annotation_coords(annots, f"PRED:{os.path.basename(path)}")
    return annots


def load_gt_json(path: str) -> List[dict]:
    with open(path) as f:
        data = json.load(f)
    annots = data.get("annotations", [])
    normalised = []
    for a in annots:
        a = dict(a)
        a["category_id"] = int(a["category_id"])
        normalised.append(a)
    _validate_annotation_coords(normalised, f"GT:{os.path.basename(path)}")
    return normalised


def find_gt_json(seq_name: str, gt_dir: str) -> Optional[str]:
    direct = os.path.join(gt_dir, f"{seq_name}_std.json")
    if os.path.isfile(direct):
        return direct
    try:
        for sub in os.listdir(gt_dir):
            sub_full = os.path.join(gt_dir, sub)
            if not os.path.isdir(sub_full):
                continue
            p = os.path.join(sub_full, f"{seq_name}_std.json")
            if os.path.isfile(p):
                return p
    except PermissionError:
        pass
    return None


def find_frame_image(seq_name: str, frame_idx: int, frames_base: str) -> Optional[str]:
    for split_dir in sorted(os.listdir(frames_base)):
        split_full = os.path.join(frames_base, split_dir)
        if not os.path.isdir(split_full):
            continue
        seq_full = os.path.join(split_full, seq_name)
        if not os.path.isdir(seq_full):
            continue
        candidates = [
            f"{seq_name}_{frame_idx:04d}.png",
            f"{seq_name}_{frame_idx:04d}.jpg",
            f"{seq_name}_{frame_idx:06d}.png",
            f"{seq_name}_{frame_idx:06d}.jpg",
            f"frame_{frame_idx:06d}.png",
            f"frame_{frame_idx:06d}.jpg",
            f"{frame_idx:06d}.png",
            f"{frame_idx:06d}.jpg",
            f"{frame_idx:04d}.png",
            f"{frame_idx:04d}.jpg",
        ]
        for fname in candidates:
            fp = os.path.join(seq_full, fname)
            if os.path.isfile(fp):
                return fp
        tag4 = f"_{frame_idx:04d}."
        tag6 = f"{frame_idx:06d}"
        try:
            for fn in sorted(os.listdir(seq_full)):
                if fn.lower().endswith((".jpg", ".png")):
                    if tag4 in fn or tag6 in fn:
                        return os.path.join(seq_full, fn)
        except PermissionError:
            pass
    return None


# ---------------------------------------------------------------------------
# Mask rendering
# ---------------------------------------------------------------------------

def render_poly(
    triplets:  List[List[float]],
    src_w:     float = OUT_W,
    src_h:     float = OUT_H,
    canvas_w:  int   = CANVAS_W,
    canvas_h:  int   = CANVAS_H,
) -> np.ndarray:
    """
    Renders a single polygon onto a (canvas_h, canvas_w) canvas using bilinear
    interpolation of the per-point intensity values (third element of each triplet).

    Coordinates are expected in the [0, src_w] x [0, src_h] spatial range and
    are scaled to the canvas resolution.  360-degree azimuth wrap-around is
    handled by doubling the canvas width, shifting wrapped points, and then
    folding back with a max-merge.
    """
    canvas_ext_w = canvas_w * 2
    canvas_ext   = np.zeros((canvas_h, canvas_ext_w), dtype=np.float32)

    pts = np.array(triplets, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((canvas_h, canvas_w), dtype=np.float32)

    sx = canvas_w / src_w
    sy = canvas_h / src_h
    xs = pts[:, 0] * sx
    ys = pts[:, 1] * sy
    intensities = pts[:, 2]

    # Detect 360-wrap: find the largest gap in sorted x; if it exceeds half
    # the canvas width, points on the left side of the gap are shifted right
    # by canvas_w so that the polygon is rendered contiguously on the extended
    # canvas and then folded back.
    sorted_xs = np.sort(xs)
    gaps      = np.diff(sorted_xs)
    if len(gaps) > 0:
        max_gap_idx = np.argmax(gaps)
        if gaps[max_gap_idx] > canvas_w / 2:
            split_threshold = sorted_xs[max_gap_idx]
            xs = np.where(xs <= split_threshold, xs + canvas_w, xs)

    degenerate = len(pts) < 3
    if not degenerate:
        try:
            grid_x, grid_y = np.meshgrid(
                np.arange(canvas_ext_w), np.arange(canvas_h)
            )
            interpolated = griddata(
                (xs, ys),
                intensities,
                (grid_x, grid_y),
                method="linear",
                fill_value=0.0,
            )
            canvas_ext = np.nan_to_num(interpolated, nan=0.0)
            canvas_ext = np.clip(canvas_ext, 0.0, None)
        except Exception:
            degenerate = True

    if degenerate:
        canvas_ext.fill(0.0)
        for x, y, intensity in zip(xs, ys, intensities):
            xi = int(np.clip(round(x), 0, canvas_ext_w - 1))
            yi = int(np.clip(round(y), 0, canvas_h - 1))
            canvas_ext[yi, xi] = max(canvas_ext[yi, xi], intensity)

    canvas = np.maximum(
        canvas_ext[:, 0:canvas_w],
        canvas_ext[:, canvas_w: canvas_w * 2],
    )
    return canvas


def render_annotation(
    annot:    dict,
    src_w:    float = OUT_W,
    src_h:    float = OUT_H,
    canvas_w: int   = CANVAS_W,
    canvas_h: int   = CANVAS_H,
) -> np.ndarray:
    """
    Renders all polygons of a single annotation onto a canvas.
    Each polygon is rendered individually and combined with a pixel-wise max.
    The result is normalised to [0, 1].
    """
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    for poly in annot.get("segmentation", []):
        if not poly:
            continue
        poly_mask = render_poly(poly, src_w, src_h, canvas_w, canvas_h)
        canvas = np.maximum(canvas, poly_mask)
    m = canvas.max()
    if m > 1e-8:
        canvas /= m
    return canvas


# ---------------------------------------------------------------------------
# Metrics: soft-IoU  +  Pearson correlation (GT-masked)
# ---------------------------------------------------------------------------

def mask_soft_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.minimum(a, b).sum()
    union = np.maximum(a, b).sum()
    if union < 1e-8:
        return 0.0
    return float(inter / union)


def mask_pearson(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Pearson correlation between GT and Pred pixel intensities, computed only
    over the pixels where GT > 0 (i.e. the GT mask area).

    Both GT and Pred carry meaningful soft intensity values, so this measures
    how well the predicted energy field matches the ground-truth energy field
    *within the region the GT says is active*.

    Returns NaN when:
      - fewer than 2 GT-active pixels exist (correlation undefined), or
      - the GT or Pred values over those pixels have zero variance
        (perfectly flat — correlation undefined).
    """
    gt_mask = gt > 0
    n_active = gt_mask.sum()
    if n_active < 2:
        return float("nan")

    gt_vals   = gt[gt_mask].astype(np.float64)
    pred_vals = pred[gt_mask].astype(np.float64)

    gt_std   = gt_vals.std()
    pred_std = pred_vals.std()

    if gt_std < 1e-12 or pred_std < 1e-12:
        return float("nan")

    gt_z   = (gt_vals   - gt_vals.mean())   / gt_std
    pred_z = (pred_vals - pred_vals.mean()) / pred_std

    return float(np.mean(gt_z * pred_z))


# ---------------------------------------------------------------------------
# Frame-level matching
# ---------------------------------------------------------------------------

def match_frame(
    gt_annots:      List[dict],
    pr_annots:      List[dict],
    iou_thresholds: np.ndarray,
) -> List[dict]:
    """
    Hungarian-matched GT↔Pred pairs for one (frame, class) bucket.

    Each returned record is one of:
      - matched pair  : {score, tp[T], fn=False, iou, pearson}
      - unmatched pred: {score, tp=zeros[T], fn=False, iou=0, pearson=nan}
      - unmatched GT  : {score=0, tp=zeros[T], fn=True}
                        (no pearson — there is no paired prediction)

    pearson is computed only for matched pairs, using the GT mask as the
    active-pixel mask (GT > 0).
    """
    n_gt = len(gt_annots)
    n_pr = len(pr_annots)

    gt_masks = [render_annotation(a) for a in gt_annots]

    pr_masks  = []
    pr_scores = []
    for a in pr_annots:
        pr_masks.append(render_annotation(a))
        pr_scores.append(float(a.get("score", 0.0)))

    if n_pr == 0:
        return [
            {"score": 0.0,
             "tp":    np.zeros(len(iou_thresholds), bool),
             "fn":    True}
            for _ in range(n_gt)
        ]

    if n_gt == 0:
        return [
            {"score":   pr_scores[i],
             "tp":      np.zeros(len(iou_thresholds), bool),
             "fn":      False,
             "iou":     0.0,
             "pearson": float("nan")}
            for i in range(n_pr)
        ]

    iou_matrix = np.zeros((n_pr, n_gt), dtype=np.float32)
    for pi in range(n_pr):
        for gi in range(n_gt):
            iou_matrix[pi, gi] = mask_soft_iou(pr_masks[pi], gt_masks[gi])

    pr_idx, gt_idx = linear_sum_assignment(-iou_matrix)
    matched_pr = set(pr_idx.tolist())
    matched_gt = set(gt_idx.tolist())

    records = []

    # Matched pairs
    for pi, gi in zip(pr_idx.tolist(), gt_idx.tolist()):
        iou = float(iou_matrix[pi, gi])
        records.append({
            "score":   pr_scores[pi],
            "tp":      iou >= iou_thresholds,
            "fn":      False,
            "iou":     iou,
            "pearson": mask_pearson(gt_masks[gi], pr_masks[pi]),
        })

    # Unmatched predictions (false positives)
    for pi in range(n_pr):
        if pi not in matched_pr:
            records.append({
                "score":   pr_scores[pi],
                "tp":      np.zeros(len(iou_thresholds), bool),
                "fn":      False,
                "iou":     0.0,
                "pearson": float("nan"),
            })

    # Unmatched GT (false negatives — no prediction to correlate against)
    for gi in range(n_gt):
        if gi not in matched_gt:
            records.append({
                "score": 0.0,
                "tp":    np.zeros(len(iou_thresholds), bool),
                "fn":    True,
            })

    return records


# ---------------------------------------------------------------------------
# AP computation
# ---------------------------------------------------------------------------

def compute_ap(
    scores:   np.ndarray,
    tp_flags: np.ndarray,
    n_gt:     int,
) -> float:
    if n_gt == 0:
        return float("nan")
    order     = np.argsort(-scores)
    tp_ord    = tp_flags[order].astype(np.float32)
    fp_ord    = 1.0 - tp_ord
    tp_cum    = np.cumsum(tp_ord)
    fp_cum    = np.cumsum(fp_ord)
    recalls    = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
    rec_thrs   = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for r in rec_thrs:
        mask = recalls >= r
        ap  += precisions[mask].max() if mask.any() else 0.0
    return ap / 101.0


def compute_ap_at_thresholds(
    records:        List[dict],
    n_gt:           int,
    iou_thresholds: np.ndarray,
) -> np.ndarray:
    det_records = [r for r in records if not r.get("fn", False)]
    if not det_records:
        return np.full(len(iou_thresholds), float("nan"))
    scores = np.array([r["score"] for r in det_records], dtype=np.float64)
    tp_mat = np.stack([r["tp"] for r in det_records])
    aps    = np.zeros(len(iou_thresholds))
    for ti in range(len(iou_thresholds)):
        aps[ti] = compute_ap(scores, tp_mat[:, ti], n_gt)
    return aps


# ---------------------------------------------------------------------------
# Pearson aggregation helpers
# ---------------------------------------------------------------------------

def _pearson_stats(records: List[dict]) -> Tuple[float, int]:
    """
    Returns (mean_pearson, n_valid) from a list of match records.
    Only matched pairs (fn=False, pearson is not nan) contribute.
    """
    vals = [
        r["pearson"]
        for r in records
        if not r.get("fn", False) and "pearson" in r and not np.isnan(r["pearson"])
    ]
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

class EvalAccumulator:
    def __init__(self):
        self.cls_records:   Dict[int, List[dict]] = defaultdict(list)
        self.cls_n_gt:      Dict[int, int]        = defaultdict(int)
        self.micro_records: List[dict]            = []
        self.micro_n_gt:    int                   = 0
        self.frame_iou: Dict[str, Dict[int, Dict[int, float]]] = \
            defaultdict(lambda: defaultdict(dict))


# ---------------------------------------------------------------------------
# Sequence evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence(
    seq_name:       str,
    pred_annots:    List[dict],
    gt_annots:      List[dict],
    acc:            EvalAccumulator,
    iou_thresholds: np.ndarray,
):
    def group_by_frame_class(annots: List[dict]) -> Dict[Tuple[int, int], List[dict]]:
        d: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
        for a in annots:
            key = (int(a["metadata_frame_index"]), int(a["category_id"]))
            d[key].append(a)
        return d

    pred_by_fc = group_by_frame_class(pred_annots)
    gt_by_fc   = group_by_frame_class(gt_annots)

    for (fi, cls), glist in gt_by_fc.items():
        if not (0 <= cls < N_CLASSES):
            continue
        acc.cls_n_gt[cls] += len(glist)
        acc.micro_n_gt    += len(glist)

    all_keys = set(pred_by_fc.keys()) | set(gt_by_fc.keys())
    for (fi, cls) in all_keys:
        if not (0 <= cls < N_CLASSES):
            continue
        glist   = gt_by_fc.get((fi, cls), [])
        plist   = pred_by_fc.get((fi, cls), [])
        records = match_frame(
            gt_annots      = glist,
            pr_annots      = plist,
            iou_thresholds = iou_thresholds,
        )
        acc.cls_records[cls].extend(records)
        acc.micro_records.extend(records)

        iou_vals = [r["iou"] for r in records if "iou" in r]
        if iou_vals:
            acc.frame_iou[seq_name][fi][cls] = float(np.mean(iou_vals))


# ---------------------------------------------------------------------------
# Results printing
# ---------------------------------------------------------------------------

def print_results(
    acc:            EvalAccumulator,
    iou_thresholds: np.ndarray,
    breakdown_thrs: Optional[np.ndarray] = None,
) -> Dict:
    sep  = "=" * 88
    sep2 = "-" * 88

    def thr_idx(thr: float) -> Optional[int]:
        diffs = np.abs(iou_thresholds - thr)
        idx   = int(np.argmin(diffs))
        return idx if diffs[idx] < 1e-4 else None

    idx50 = thr_idx(0.50)
    idx75 = thr_idx(0.75)

    print(f"\n{sep}")
    print("  MASK mAP EVALUATION  (COCO-style, soft-IoU, 101-point interpolation)")
    print(sep)

    hdr_50 = f"{'AP@50':>8}" if idx50 is not None else f"{'':>8}"
    hdr_75 = f"{'AP@75':>8}" if idx75 is not None else f"{'':>8}"
    print(
        f"\n  {'Class':<22} {'GT':>6} {'Det':>6} "
        f"{hdr_50} {hdr_75} {'mAP':>8} {'PearsonR':>10} {'nPairs':>7}"
    )
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*7}")

    per_class_ap_mean: Dict[int, float]      = {}
    per_class_aps:     Dict[int, np.ndarray] = {}
    per_class_pearson: Dict[int, float]      = {}
    active_classes:    List[int]             = []

    for cls in range(N_CLASSES):
        n_gt  = acc.cls_n_gt[cls]
        recs  = acc.cls_records[cls]
        n_det = sum(1 for r in recs if not r.get("fn", False))

        pear_mean, n_pairs = _pearson_stats(recs)
        per_class_pearson[cls] = pear_mean

        if n_gt == 0:
            dash = f"{'—':>8}"
            pdash = f"{'—':>10}"
            print(
                f"  {CLASS_NAMES[cls]:<22} {'0':>6} {n_det:>6} "
                f"{dash} {dash} {dash} {pdash} {'0':>7}"
            )
            per_class_ap_mean[cls] = float("nan")
            per_class_aps[cls]     = np.full(len(iou_thresholds), float("nan"))
            continue

        aps   = compute_ap_at_thresholds(recs, n_gt, iou_thresholds)
        ap50  = f"{aps[idx50]:>8.4f}" if idx50 is not None else f"{'—':>8}"
        ap75  = f"{aps[idx75]:>8.4f}" if idx75 is not None else f"{'—':>8}"
        map_c = float(np.nanmean(aps))

        per_class_aps[cls]     = aps
        per_class_ap_mean[cls] = map_c
        active_classes.append(cls)

        pear_str = f"{pear_mean:>10.4f}" if not np.isnan(pear_mean) else f"{'—':>10}"
        print(
            f"  {CLASS_NAMES[cls]:<22} {n_gt:>6} {n_det:>6} "
            f"{ap50} {ap75} {map_c:>8.4f} {pear_str} {n_pairs:>7}"
        )

    # Macro averages
    valid_maps = [per_class_ap_mean[c] for c in active_classes
                  if not np.isnan(per_class_ap_mean[c])]
    macro_map  = float(np.mean(valid_maps)) if valid_maps else float("nan")

    macro_aps_per_thr = []
    for ti in range(len(iou_thresholds)):
        vals = [per_class_aps[c][ti] for c in active_classes
                if not np.isnan(per_class_aps[c][ti])]
        macro_aps_per_thr.append(float(np.mean(vals)) if vals else float("nan"))

    macro_ap50 = (f"{macro_aps_per_thr[idx50]:>8.4f}"
                  if idx50 is not None else f"{'—':>8}")
    macro_ap75 = (f"{macro_aps_per_thr[idx75]:>8.4f}"
                  if idx75 is not None else f"{'—':>8}")

    valid_pearson = [per_class_pearson[c] for c in active_classes
                     if not np.isnan(per_class_pearson[c])]
    macro_pearson = float(np.mean(valid_pearson)) if valid_pearson else float("nan")
    macro_pear_str = f"{macro_pearson:>10.4f}" if not np.isnan(macro_pearson) else f"{'—':>10}"

    # Micro averages
    micro_aps  = compute_ap_at_thresholds(acc.micro_records, acc.micro_n_gt, iou_thresholds)
    micro_map  = float(np.nanmean(micro_aps))
    micro_ap50 = (f"{float(micro_aps[idx50]):>8.4f}"
                  if idx50 is not None else f"{'—':>8}")
    micro_ap75 = (f"{float(micro_aps[idx75]):>8.4f}"
                  if idx75 is not None else f"{'—':>8}")

    micro_pear_mean, micro_pear_n = _pearson_stats(acc.micro_records)
    micro_pear_str = (f"{micro_pear_mean:>10.4f}"
                      if not np.isnan(micro_pear_mean) else f"{'—':>10}")

    n_micro_det = sum(1 for r in acc.micro_records if not r.get("fn", False))

    print(f"\n{sep2}")
    print(
        f"  {'MACRO mAP':<22} {'':>6} {'':>6} "
        f"{macro_ap50} {macro_ap75} {macro_map:>8.4f} {macro_pear_str} {'':>7}"
    )
    print(
        f"  {'MICRO mAP':<22} {acc.micro_n_gt:>6} {n_micro_det:>6} "
        f"{micro_ap50} {micro_ap75} {micro_map:>8.4f} {micro_pear_str} {micro_pear_n:>7}"
    )
    print(f"{sep}\n")

    # Per-threshold breakdown table
    bthr = breakdown_thrs if breakdown_thrs is not None else iou_thresholds
    bthr_indices = []
    for t in bthr:
        diffs = np.abs(iou_thresholds - t)
        idx   = int(np.argmin(diffs))
        if diffs[idx] < 1e-4:
            bthr_indices.append((t, idx))

    if bthr_indices:
        col_w      = 9
        thr_header = "".join(f"  AP@{t:.2f}".rjust(col_w) for t, _ in bthr_indices)
        print(f"  Per-threshold AP breakdown:")
        print(f"  {'Class':<22}{thr_header}  {'mAP':>{col_w}}  {'PearsonR':>{col_w}}")
        print(f"  {'-'*22}" + "-" * (col_w * (len(bthr_indices) + 2) + 2))

        for cls in range(N_CLASSES):
            name = CLASS_NAMES[cls]
            if acc.cls_n_gt[cls] == 0:
                row  = "".join(f"{'—':>{col_w}}" for _ in bthr_indices)
                row += f"{'—':>{col_w}}  {'—':>{col_w}}"
            else:
                aps  = per_class_aps[cls]
                row  = "".join(f"{aps[i]:>{col_w}.4f}" for _, i in bthr_indices)
                row += f"{per_class_ap_mean[cls]:>{col_w}.4f}"
                pv   = per_class_pearson[cls]
                row += f"  {pv:>{col_w}.4f}" if not np.isnan(pv) else f"  {'—':>{col_w}}"
            print(f"  {name:<22}{row}")

        macro_row  = "".join(f"{macro_aps_per_thr[i]:>{col_w}.4f}" for _, i in bthr_indices)
        macro_row += f"{macro_map:>{col_w}.4f}"
        macro_row += (f"  {macro_pearson:>{col_w}.4f}"
                      if not np.isnan(macro_pearson) else f"  {'—':>{col_w}}")
        micro_row  = "".join(f"{float(micro_aps[i]):>{col_w}.4f}" for _, i in bthr_indices)
        micro_row += f"{micro_map:>{col_w}.4f}"
        micro_row += (f"  {micro_pear_mean:>{col_w}.4f}"
                      if not np.isnan(micro_pear_mean) else f"  {'—':>{col_w}}")

        print(f"  {'-'*22}" + "-" * (col_w * (len(bthr_indices) + 2) + 2))
        print(f"  {'MACRO':<22}{macro_row}")
        print(f"  {'MICRO':<22}{micro_row}")
        print(f"{sep}\n")

    print("  AP across IoU thresholds (macro):")
    print("  " + "  ".join(f"{t:.2f}" for t in iou_thresholds))
    print("  " + "  ".join(
        f"{v:.3f}" if not np.isnan(v) else "  nan"
        for v in macro_aps_per_thr))
    print(f"\n{sep}\n")

    return {
        "macro_mAP":        macro_map,
        "macro_AP50":       float(macro_aps_per_thr[idx50]) if idx50 is not None else None,
        "macro_AP75":       float(macro_aps_per_thr[idx75]) if idx75 is not None else None,
        "macro_pearson_r":  float(macro_pearson) if not np.isnan(macro_pearson) else None,
        "micro_mAP":        micro_map,
        "micro_AP50":       float(micro_aps[idx50]) if idx50 is not None else None,
        "micro_AP75":       float(micro_aps[idx75]) if idx75 is not None else None,
        "micro_pearson_r":  float(micro_pear_mean) if not np.isnan(micro_pear_mean) else None,
        "micro_pearson_n":  micro_pear_n,
        "macro_AP_per_thr": [float(v) for v in macro_aps_per_thr],
        "micro_AP_per_thr": [float(v) for v in micro_aps],
        "iou_thresholds":   [float(t) for t in iou_thresholds],
        "breakdown_thresholds": [float(t) for t, _ in bthr_indices],
        "per_class": {
            CLASS_NAMES[c]: {
                "n_gt":      int(acc.cls_n_gt[c]),
                "n_det":     int(sum(1 for r in acc.cls_records[c]
                                     if not r.get("fn", False))),
                "AP50":      float(per_class_aps[c][idx50])
                             if idx50 is not None and not np.isnan(per_class_aps[c][idx50])
                             else None,
                "AP75":      float(per_class_aps[c][idx75])
                             if idx75 is not None and not np.isnan(per_class_aps[c][idx75])
                             else None,
                "mAP":       float(per_class_ap_mean[c])
                             if not np.isnan(per_class_ap_mean[c]) else None,
                "AP_per_thr": [
                    float(v) if not np.isnan(v) else None
                    for v in per_class_aps[c]
                ],
                "pearson_r": float(per_class_pearson[c])
                             if not np.isnan(per_class_pearson[c]) else None,
                "pearson_n": _pearson_stats(acc.cls_records[c])[1],
            }
            for c in range(N_CLASSES)
        },
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def render_annots_as_heatmap(annots: List[dict]) -> np.ndarray:
    canvas = np.zeros((CANVAS_H, CANVAS_W), np.float32)
    for a in annots:
        mask = render_annotation(
            a, src_w=OUT_W, src_h=OUT_H, canvas_w=CANVAS_W, canvas_h=CANVAS_H
        )
        canvas += mask
    m = canvas.max()
    if m > 1e-8:
        canvas /= m
    return canvas


def composite_heatmap_on_rgb(
    rgb_np: np.ndarray,
    heat:   np.ndarray,
    alpha:  float = 0.55,
    cmap          = plt.cm.inferno,
) -> np.ndarray:
    H, W = rgb_np.shape[:2]
    heat_up = np.array(
        Image.fromarray((heat * 255).astype(np.uint8)).resize(
            (W, H), Image.NEAREST
        )
    ) / 255.0
    colored = cmap(heat_up)[:, :, :3]
    out = (1 - alpha) * rgb_np.astype(np.float32) / 255.0 + alpha * colored
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def _class_summary(annots: List[dict]) -> Tuple[List[int], str]:
    counts: Dict[int, int] = defaultdict(int)
    for a in annots:
        cls = int(a.get("category_id", -1))
        if 0 <= cls < N_CLASSES:
            counts[cls] += 1
    if not counts:
        return [], "no annotations"
    parts = [f"{CLASS_NAMES[c]} ×{n}" for c, n in sorted(counts.items())]
    return sorted(counts.keys()), ",  ".join(parts)


def plot_comparison_frames(
    seq_name:    str,
    pred_annots: List[dict],
    gt_annots:   List[dict],
    acc:         EvalAccumulator,
    frames_base: str,
    out_path:    str,
    n_frames:    int = 8,
    strategy:    str = "mixed",
):
    frame_iou_dict = acc.frame_iou.get(seq_name, {})
    frame_scores   = [
        (fi, float(np.mean(list(cls_ious.values()))))
        for fi, cls_ious in frame_iou_dict.items()
        if cls_ious
    ]
    if not frame_scores:
        print(f"[WARN] No IoU data for {seq_name} — skipping comparison frames.")
        return

    frame_scores.sort(key=lambda x: x[1])

    if strategy == "best":
        selected       = [fs[0] for fs in frame_scores[-n_frames:]]
        strategy_label = f"top-{n_frames} frames by IoU (model performs best here)"
    elif strategy == "worst":
        selected       = [fs[0] for fs in frame_scores[:n_frames]]
        strategy_label = f"bottom-{n_frames} frames by IoU (model struggles most here)"
    else:
        half   = n_frames // 2
        worst  = [fs[0] for fs in frame_scores[:half]]
        best   = [fs[0] for fs in frame_scores[-(n_frames - half):]]
        selected = worst + best
        strategy_label = (
            f"{half} hardest + {n_frames - half} easiest frames by IoU "
            f"(range: {frame_scores[0][1]:.2f} – {frame_scores[-1][1]:.2f})"
        )

    selected = sorted(set(selected))[:n_frames]

    def group_by_frame(annots: List[dict]) -> Dict[int, List[dict]]:
        d: Dict[int, List[dict]] = defaultdict(list)
        for a in annots:
            d[int(a["metadata_frame_index"])].append(a)
        return d

    pred_by_f = group_by_frame(pred_annots)
    gt_by_f   = group_by_frame(gt_annots)

    n_rows       = len(selected)
    n_cols       = 4
    width_ratios = [3, 3, 3, 3, 0.25]

    fig, axes = plt.subplots(
        n_rows, n_cols + 1,
        figsize=(16.5, 2.8 * n_rows),
        facecolor=DARK_BG,
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.04, "hspace": 0.35},
    )
    fig.suptitle(
        f"GT vs Prediction — {seq_name}\n"
        f"Showing {strategy_label}",
        color=WHITE, fontsize=10, y=1.01,
    )

    col_titles = [
        "RGB frame",
        "Ground truth\n(green heat = GT mask energy)",
        "Model prediction\n(orange/yellow heat = predicted mask energy)",
        "Difference  (Pred − GT)\nred = pred only  |  blue = GT only  |  white = overlap",
    ]
    for col, ct in enumerate(col_titles):
        axes[0][col].set_title(ct, color=WHITE, fontsize=7, pad=4)
    for row in range(n_rows):
        axes[row][n_cols].set_visible(False)

    diff_norm = plt.Normalize(vmin=-1, vmax=1)

    for row, fi in enumerate(selected):
        fi_iou   = frame_iou_dict.get(fi, {})
        mean_iou = float(np.mean(list(fi_iou.values()))) if fi_iou else 0.0

        gt_frame   = gt_by_f.get(fi, [])
        pred_frame = pred_by_f.get(fi, [])

        img_path = find_frame_image(seq_name, fi, frames_base)
        if img_path:
            rgb_np = np.array(
                Image.open(img_path).convert("RGB").resize(
                    (OUT_W, OUT_H), Image.BILINEAR
                )
            )
        else:
            rgb_np = np.full((OUT_H, OUT_W, 3), 30, np.uint8)

        gt_heat   = render_annots_as_heatmap(gt_frame)
        pred_heat = render_annots_as_heatmap(pred_frame)
        diff      = pred_heat - gt_heat

        gt_comp   = composite_heatmap_on_rgb(rgb_np, gt_heat,   alpha=0.55, cmap=plt.cm.Greens)
        pred_comp = composite_heatmap_on_rgb(rgb_np, pred_heat, alpha=0.55, cmap=plt.cm.inferno)

        gt_cls_ids,   gt_cls_str   = _class_summary(gt_frame)
        pred_cls_ids, pred_cls_str = _class_summary(pred_frame)
        n_pred_classes             = len(pred_cls_ids)

        # Per-frame Pearson: average over all matched pairs in this frame
        frame_pearson_vals = []
        for cls in set(c for a in gt_frame + pred_frame
                       for c in [int(a.get("category_id", -1))]
                       if 0 <= c < N_CLASSES):
            gt_fc   = [a for a in gt_frame   if int(a.get("category_id", -1)) == cls]
            pred_fc = [a for a in pred_frame if int(a.get("category_id", -1)) == cls]
            if not gt_fc or not pred_fc:
                continue
            for a_gt, a_pr in zip(gt_fc, pred_fc):
                gm = render_annotation(a_gt)
                pm = render_annotation(a_pr)
                pv = mask_pearson(gm, pm)
                if not np.isnan(pv):
                    frame_pearson_vals.append(pv)
        frame_pearson_str = (
            f"  r={np.mean(frame_pearson_vals):.3f}"
            if frame_pearson_vals else ""
        )

        panels = [
            (rgb_np,    "rgb"),
            (gt_comp,   "rgb"),
            (pred_comp, "rgb"),
            (diff,      "diff"),
        ]
        for col, (img_data, kind) in enumerate(panels):
            ax = axes[row][col]
            ax.set_facecolor(PANEL_BG)
            for sp in ax.spines.values():
                sp.set_edgecolor("#444")
            if kind == "diff":
                ax.imshow(diff, cmap="RdBu_r", norm=diff_norm,
                          origin="upper", aspect="auto", interpolation="nearest")
            else:
                ax.imshow(img_data, aspect="auto", origin="upper", interpolation="nearest")
            ax.axis("off")

        axes[row][0].set_ylabel(
            f"frame {fi}\nIoU={mean_iou:.3f}{frame_pearson_str}",
            color=WHITE, fontsize=6.5, rotation=0, labelpad=52, va="center",
        )

        if gt_frame:
            axes[row][1].set_xlabel(gt_cls_str, color="#aaffaa", fontsize=5.5, labelpad=3)
        else:
            axes[row][1].set_xlabel(
                "No GT annotations in this frame", color="#888888", fontsize=5.5, labelpad=3
            )

        if pred_frame:
            axes[row][2].set_xlabel(
                pred_cls_str + f"\n({n_pred_classes} class"
                f"{'es' if n_pred_classes != 1 else ''}, heatmap shows combined energy)",
                color="#ffddaa", fontsize=5.5, labelpad=3,
            )
        else:
            axes[row][2].set_xlabel(
                "No predictions in this frame", color="#888888", fontsize=5.5, labelpad=3
            )

        axes[row][3].set_xlabel(
            f"mean IoU@0.5={mean_iou:.3f}",
            color=WHITE, fontsize=5.5, labelpad=3,
        )

        for col, cls_ids in [(1, gt_cls_ids), (2, pred_cls_ids)]:
            if cls_ids:
                patches = [
                    mpatches.Patch(
                        facecolor=cls_color(c), edgecolor="white",
                        linewidth=0.4, label=CLASS_NAMES[c],
                    )
                    for c in cls_ids
                ]
                axes[row][col].legend(
                    handles=patches, loc="upper left", fontsize=4.5,
                    facecolor="#000000cc", edgecolor="none", labelcolor=WHITE,
                    framealpha=0.85, handlelength=1.0, handletextpad=0.4,
                    borderpad=0.5, labelspacing=0.3, ncol=1,
                )

    fig.subplots_adjust(right=0.91)
    cbar_ax = fig.add_axes([0.925, 0.04, 0.012, 0.88])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=diff_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.ax.tick_params(colors=WHITE, labelsize=6)
    cb.set_label("Pred − GT energy", color=WHITE, fontsize=7, labelpad=6)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.set_ticklabels(
        ["-1\n(GT only)", "-0.5", "0\n(overlap)", "+0.5", "+1\n(Pred only)"],
        color=WHITE, fontsize=5,
    )

    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[VIZ] → {out_path}")


def plot_ap_curves(results: dict, out_path: str):
    iou_thrs = results["iou_thresholds"]
    active   = [
        (c, CLASS_NAMES[c]) for c in range(N_CLASSES)
        if results["per_class"][CLASS_NAMES[c]]["mAP"] is not None
    ]
    plot_items = active + [(-1, "MACRO"), (-2, "MICRO")]

    cols = min(4, len(plot_items))
    rows = (len(plot_items) + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 3.8, rows * 2.8),
        facecolor=DARK_BG,
        squeeze=False,
    )
    fig.suptitle(
        "AP vs IoU Threshold — Per-class + Macro/Micro",
        color=WHITE, fontsize=11, y=1.01,
    )

    for idx, (c, name) in enumerate(plot_items):
        r, col = divmod(idx, cols)
        ax = axes[r][col]
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.tick_params(colors=WHITE, labelsize=6)

        if c == -1:
            aps   = results["macro_AP_per_thr"]
            color = "#4fd1c5"
            mval  = results["macro_mAP"] or 0.0
            pval  = results.get("macro_pearson_r")
        elif c == -2:
            aps   = results["micro_AP_per_thr"]
            color = "#f6ad55"
            mval  = results["micro_mAP"] or 0.0
            pval  = results.get("micro_pearson_r")
        else:
            aps   = results["per_class"][name]["AP_per_thr"]
            color = cls_color(c)
            mval  = results["per_class"][name]["mAP"] or 0.0
            pval  = results["per_class"][name].get("pearson_r")

        aps_clean = [v if v is not None else 0.0 for v in aps]
        ax.plot(iou_thrs, aps_clean, color=color, linewidth=1.8, marker="o", markersize=4)
        ax.fill_between(iou_thrs, 0, aps_clean, color=color, alpha=0.18)

        lo, hi = min(iou_thrs) - 0.02, max(iou_thrs) + 0.02
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.05)
        for ref in [0.50, 0.75]:
            if lo < ref < hi:
                ax.axvline(ref, color="#ffffff33", lw=0.7, linestyle="--")

        pear_str = f"  r={pval:.3f}" if pval is not None else ""
        ax.set_title(
            f"{name[:16]}\nmAP={mval:.4f}{pear_str}",
            color=color, fontsize=7.5,
        )
        ax.set_xlabel("IoU threshold", color=WHITE, fontsize=6)
        ax.set_ylabel("AP", color=WHITE, fontsize=6)

    for idx in range(len(plot_items), rows * cols):
        r, col = divmod(idx, cols)
        axes[r][col].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[VIZ] → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="COCO-style mask mAP evaluation for STARSS23 submissions"
    )
    parser.add_argument("--pred_dir",    required=True,
                        help="Directory of prediction JSON files  (e.g. submission_output/)")
    parser.add_argument("--gt_dir",      required=True,
                        help="Root of GT JSON files  (e.g. labels_dev/)")
    parser.add_argument("--frames_base", required=True,
                        help="Root of dataset frame images  (e.g. frames_dev/)")
    parser.add_argument("--output_dir",  default="eval_output")
    parser.add_argument("--n_comp_seqs",    type=int, default=3,
                        help="Number of sequences for GT vs Pred comparison panels")
    parser.add_argument("--n_comp_frames",  type=int, default=8,
                        help="Frames per sequence in comparison panels")
    parser.add_argument("--frame_strategy", default="mixed",
                        choices=["best", "worst", "mixed"],
                        help=(
                            "Which frames to show in comparison panels:\n"
                            "  best  — highest IoU frames (where model succeeds)\n"
                            "  worst — lowest IoU frames (where model fails)\n"
                            "  mixed — half worst + half best (shows full range)"
                        ))
    parser.add_argument(
        "--n_sample", type=int, default=None, metavar="N",
        help=(
            "Randomly sample N prediction JSONs for a quick sanity check. "
            "Uses --seed for reproducibility. Omit to evaluate all sequences."
        ),
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --n_sample (default: 42)")
    parser.add_argument(
        "--iou_thresholds", type=str, default=None, metavar="SPEC",
        help=(
            "IoU threshold specification.\n"
            "  start:step:end  e.g. '0.50:0.05:0.95'  (COCO default)\n"
            "  comma list      e.g. '0.50,0.75'\n"
            "  single value    e.g. '0.50'\n"
            "Defaults to COCO standard 0.50:0.05:0.95."
        ),
    )
    parser.add_argument(
        "--breakdown_thresholds", type=str, default=None, metavar="SPEC",
        help=(
            "Subset of --iou_thresholds to show in the per-class breakdown table. "
            "Same format as --iou_thresholds. Defaults to all thresholds."
        ),
    )
    args = parser.parse_args()

    if args.iou_thresholds is not None:
        try:
            iou_thresholds = parse_iou_thresholds(args.iou_thresholds)
        except (ValueError, argparse.ArgumentTypeError) as e:
            print(f"[ERROR] --iou_thresholds: {e}")
            sys.exit(1)
    else:
        iou_thresholds = DEFAULT_IOU_THRESHOLDS

    if args.breakdown_thresholds is not None:
        try:
            breakdown_thrs = parse_iou_thresholds(args.breakdown_thresholds)
        except (ValueError, argparse.ArgumentTypeError) as e:
            print(f"[ERROR] --breakdown_thresholds: {e}")
            sys.exit(1)
    else:
        breakdown_thrs = None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(
        p for p in Path(args.pred_dir).glob("*.json")
        if not p.name.startswith(".")
    )
    if not pred_files:
        print(f"[ERROR] No JSON files found in {args.pred_dir}")
        sys.exit(1)

    if args.n_sample is not None and args.n_sample < len(pred_files):
        rng        = random.Random(args.seed)
        pred_files = sorted(rng.sample(pred_files, args.n_sample))
        print(
            f"[INFO] --n_sample {args.n_sample}: randomly selected "
            f"{args.n_sample} sequences  (seed={args.seed})"
        )
    else:
        print(f"[INFO] Evaluating all {len(pred_files)} sequences")

    print(f"[INFO] Prediction JSONs  : {len(pred_files)} files evaluated")
    print(f"[INFO] GT directory      : {args.gt_dir}")
    print(f"[INFO] Frames base       : {args.frames_base}")
    print(f"[INFO] Mask rendering    : Linear interpolation w/ 360 azimuth wrap support")
    print(f"[INFO] Canvas resolution : {CANVAS_W}×{CANVAS_H}")
    print(f"[INFO] Spatial range     : {OUT_W}×{OUT_H}  (coordinates MUST stay within this)")
    print(f"[INFO] Max JSON size     : {MAX_JSON_BYTES // 1024 // 1024} MB")
    print(f"[INFO] IoU thresholds    : {list(iou_thresholds)}")
    bthr_display = breakdown_thrs if breakdown_thrs is not None else iou_thresholds
    print(f"[INFO] Breakdown thrs    : {list(bthr_display)}")
    print(f"[INFO] GT category_id    : 0-indexed (matches prediction format)")
    print(f"[INFO] Pred category_id  : 0-indexed (as output by inference script)")
    print()

    acc      = EvalAccumulator()
    skipped  = []
    matched  = []
    all_pred = {}
    all_gt   = {}

    for pf in pred_files:
        seq_name = pf.stem
        gt_path  = find_gt_json(seq_name, args.gt_dir)
        if gt_path is None:
            skipped.append(seq_name)
            continue

        try:
            pred_annots = load_pred_json(str(pf))
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

        try:
            gt_annots = load_gt_json(gt_path)
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

        all_pred[seq_name] = pred_annots
        all_gt[seq_name]   = gt_annots

        print(f"  {seq_name:<40}  pred={len(pred_annots):>5}  gt={len(gt_annots):>5}")
        evaluate_sequence(seq_name, pred_annots, gt_annots, acc, iou_thresholds)
        matched.append(seq_name)

    print()
    print(f"[INFO] Matched {len(matched)} sequences  ({len(skipped)} skipped: {skipped})")

    results = print_results(acc, iou_thresholds, breakdown_thrs)

    jpath = out_dir / "metrics.json"
    with open(jpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Metrics saved → {jpath}")

    plot_ap_curves(results, str(out_dir / "00_ap_curves.png"))

    comp_candidates = [
        sn for sn in matched if all_pred.get(sn) and all_gt.get(sn)
    ]
    comp_candidates.sort(key=lambda sn: -len(all_pred.get(sn, [])))
    comp_seqs = comp_candidates[:args.n_comp_seqs]

    for sn in comp_seqs:
        out_path = str(out_dir / f"01_comparison_{sn}.png")
        plot_comparison_frames(
            seq_name    = sn,
            pred_annots = all_pred[sn],
            gt_annots   = all_gt[sn],
            acc         = acc,
            frames_base = args.frames_base,
            out_path    = out_path,
            n_frames    = args.n_comp_frames,
            strategy    = args.frame_strategy,
        )

    print(f"\n[DONE] All outputs written to: {out_dir}/")
    print(f"       metrics.json")
    print(f"       00_ap_curves.png")
    print(f"       01_comparison_<seq>.png  (×{len(comp_seqs)})")


if __name__ == "__main__":
    main()
