import argparse
import glob
import io
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_dilation
from tqdm import tqdm

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import pycocotools.mask as mask_utils
    HAS_COCO = True
except ImportError:
    HAS_COCO = False

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
LABELS_BASE   = "/data3/scratch/eez086/STARSS23/labels_dev"
IMG_W, IMG_H  = 360, 180
MATCH_THR_DEG = 20.0

DEFAULT_SIGMA_DEG  = 6.0
DEFAULT_ETH        = 0.10

# Blob pixels beyond this many sigma contribute < exp(-32) ~ 0% of peak.
_BLOB_CUTOFF_SIGMA = 4.0

CLASS_NAMES = {
    0:  "Female speech", 1:  "Male speech", 2:  "Clapping", 3:  "Telephone",
    4:  "Laughter", 5:  "Domestic sounds", 6:  "Walk/footsteps", 7:  "Door open/close",
    8:  "Music", 9:  "Musical instr.", 10: "Water tap/shower", 11: "Bell", 12: "Knock",
}
N_CLS = len(CLASS_NAMES)

COCO_CATS = [
    {"id": k + 1, "name": v, "supercategory": "sound"}
    for k, v in sorted(CLASS_NAMES.items())
]

# ── 1D Precomputed spherical grids ────────────────────────────────────────────
_AZ_1D     = np.radians((np.arange(IMG_W, dtype=np.float32) / IMG_W) * 360.0 - 180.0)
_EL_1D     = np.radians(90.0 - (np.arange(IMG_H, dtype=np.float32) / (IMG_H - 1)) * 180.0)
_COS_EL_1D = np.cos(_EL_1D).astype(np.float32)

# ── Argument parsing ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mask mAP + Pearson r + Distance MAPE evaluator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--exp_dir",    required=True)
    p.add_argument("--gt_root",    default=LABELS_BASE)
    p.add_argument("--split",      default="test")
    p.add_argument("--sigma",      type=float, default=DEFAULT_SIGMA_DEG)
    p.add_argument("--energy_thr", type=float, default=DEFAULT_ETH)
    return p.parse_args()

# ── Geometry helpers ───────────────────────────────────────────────────────────
def px_to_azel(x: float, y: float) -> Tuple[float, float]:
    az = (x / IMG_W) * 360.0 - 180.0
    el = 90.0 - (y / (IMG_H - 1)) * 180.0
    return az, el

def extract_peak(ann: dict) -> Tuple[float, float, float]:
    bx, by, be = 0.0, 0.0, 0.0
    for sub in ann.get("segmentation", []):
        for t in sub:
            x, y, e = float(t[0]), float(t[1]), float(t[2])
            if e > be:
                be, bx, by = e, x, y
    return bx, by, be

# ── High-Performance Gaussian Rendering ───────────────────────────────────────
def _add_spherical_blob(
    canvas: np.ndarray,
    az0_rad: float, el0_rad: float, energy: float,
    inv_2sigma2: float, cutoff_rad: float,
) -> None:
    cos_el0 = math.cos(el0_rad)

    el_lo = max(-math.pi / 2.0, el0_rad - cutoff_rad)
    el_hi = min( math.pi / 2.0, el0_rad + cutoff_rad)
    r0 = max(0, int(math.floor((math.pi / 2.0 - el_hi) / math.pi * (IMG_H - 1))))
    r1 = min(IMG_H, int(math.ceil( (math.pi / 2.0 - el_lo) / math.pi * (IMG_H - 1))) + 1)
    if r0 >= r1: return

    az_half_rad = min(math.pi, cutoff_rad / max(cos_el0, 0.087))
    az0_deg     = math.degrees(az0_rad)
    az_half_deg = math.degrees(az_half_rad)

    c_lo = int(math.floor((az0_deg - az_half_deg + 180.0) / 360.0 * IMG_W))
    c_hi = int(math.ceil( (az0_deg + az_half_deg + 180.0) / 360.0 * IMG_W)) + 1

    el_slice = _EL_1D[r0:r1]
    row_factor = cos_el0 * _COS_EL_1D[r0:r1]
    hav_el = np.sin((el_slice - el0_rad) / 2.0) ** 2

    def _compute_and_add(col_slice: slice) -> None:
        az_slice = _AZ_1D[col_slice]
        hav_az = np.sin((az_slice - az0_rad) / 2.0) ** 2
        
        hav = hav_el[:, None] + row_factor[:, None] * hav_az[None, :]
        hav *= (-4.0 * inv_2sigma2)
        np.exp(hav, out=hav)
        hav *= energy
        canvas[r0:r1, col_slice] += hav

    if c_lo >= 0 and c_hi <= IMG_W:
        _compute_and_add(slice(c_lo, c_hi))
    else:
        c_lo_w, c_hi_w = c_lo % IMG_W, c_hi % IMG_W or IMG_W
        if c_lo < 0:
            _compute_and_add(slice(0, c_hi_w))
            _compute_and_add(slice(c_lo_w, IMG_W))
        else:
            _compute_and_add(slice(c_lo_w, IMG_W))
            _compute_and_add(slice(0, c_hi_w))

def render_energy_map(ann: dict, sigma_rad: float, inv_2sigma2: float, cutoff_rad: float) -> np.ndarray:
    canvas = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    for sub in ann.get("segmentation", []):
        for t in sub:
            x, y, e = float(t[0]), float(t[1]), float(t[2])
            az_deg, el_deg = px_to_azel(x, y)
            _add_spherical_blob(
                canvas, math.radians(az_deg), math.radians(el_deg),
                e, inv_2sigma2, cutoff_rad
            )
    return canvas

# ── Unified Annotation Caching ────────────────────────────────────────────────
def preprocess_annotation(ann: dict, sigma_rad: float, inv_2sigma2: float, cutoff_rad: float, eth: float) -> dict:
    cid = int(ann.get("category_id", -1))
    
    if cid < 0 or cid >= N_CLS:
        return {"valid": False, "ann": ann}
        
    emap = render_energy_map(ann, sigma_rad, inv_2sigma2, cutoff_rad)
    peak = emap.max()
    
    if peak < 1e-9:
        return {"valid": False, "ann": ann}
        
    bmask = (emap >= eth * peak).astype(np.uint8)
    if not bmask.any():
        return {"valid": False, "ann": ann}
        
    ys, xs = np.where(bmask)
    
    rle = mask_utils.encode(np.asfortranarray(bmask))
    rle["counts"] = rle["counts"].decode("utf-8")
    
    return {
        "valid": True,
        "ann": ann,
        "cid": cid,
        "emap": emap,
        "bmask": bmask,
        "rle": rle,
        "area": int(bmask.sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
        "score": float(ann.get("score", 0.0))
    }

# ── Vectorized Hungarian Matching ──────────────────────────────────────────────
def hungarian_match(gts: List[dict], prs: List[dict]) -> List[Tuple[int, int, float]]:
    M, N = len(gts), len(prs)
    if M == 0 or N == 0: return []

    gt_peaks = np.array([extract_peak(a["ann"])[:2] for a in gts], dtype=np.float32)
    pr_peaks = np.array([extract_peak(a["ann"])[:2] for a in prs], dtype=np.float32)
    
    gt_cids = np.array([a["cid"] for a in gts])
    pr_cids = np.array([a["cid"] for a in prs])

    gt_az = np.radians((gt_peaks[:, 0] / IMG_W) * 360.0 - 180.0)
    gt_el = np.radians(90.0 - (gt_peaks[:, 1] / (IMG_H - 1)) * 180.0)
    pr_az = np.radians((pr_peaks[:, 0] / IMG_W) * 360.0 - 180.0)
    pr_el = np.radians(90.0 - (pr_peaks[:, 1] / (IMG_H - 1)) * 180.0)

    d_el = gt_el[:, None] - pr_el[None, :]
    d_az = gt_az[:, None] - pr_az[None, :]
    hav = np.sin(d_el / 2.0)**2 + np.cos(gt_el)[:, None] * np.cos(pr_el)[None, :] * np.sin(d_az / 2.0)**2
    
    cost = np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0))))
    
    cost[gt_cids[:, None] != pr_cids[None, :]] = 1000.0
    cost[cost > MATCH_THR_DEG] = 1000.0  

    row_ind, col_ind = linear_sum_assignment(cost)
    return [(r, c, cost[r, c]) for r, c in zip(row_ind, col_ind) if cost[r, c] <= MATCH_THR_DEG]


# ── File I/O ───────────────────────────────────────────────────────────────────
def load_annotations(path: str) -> Dict[int, List[dict]]:
    with open(path) as f:
        data = json.load(f)
    out: Dict[int, List[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        cid = int(ann.get("category_id", -1))
        if 0 <= cid < N_CLS:
            out[int(ann["metadata_frame_index"])].append(ann)
    return dict(out)

def find_gt_json(seq_name: str, gt_root: str, split: str):
    for pat in (f"{seq_name}_std.json", f"{seq_name}.json"):
        hits = glob.glob(os.path.join(gt_root, "**", pat), recursive=True)
        hits = [h for h in hits if split in h] or hits
        if hits: return hits[0]
    return None


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if not HAS_COCO:
        print("[ERROR] pycocotools is required for Mask mAP.")
        sys.exit(1)

    pred_dir = os.path.join(args.exp_dir, "inference_outputs")
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*_inference.json")))
    if not pred_files:
        print(f"[ERROR] No *_inference.json files found in {pred_dir}")
        sys.exit(1)

    sigma_rad   = math.radians(args.sigma)
    inv_2sigma2 = 0.5 / (sigma_rad * sigma_rad)
    cutoff_rad  = _BLOB_CUTOFF_SIGMA * sigma_rad

    coco_images, coco_gt_anns, coco_dt_anns = [], [], []
    img_id_map = {}
    img_id_counter, gt_ann_counter = 1, 1

    pearsons, dist_errors = [], []

    for pred_path in tqdm(pred_files, desc="Evaluating", unit="seq"):
        seq = os.path.basename(pred_path).replace("_inference.json", "")
        gt_path = find_gt_json(seq, args.gt_root, args.split)
        if not gt_path: continue

        gt_by_frame = load_annotations(gt_path)
        pr_by_frame = load_annotations(pred_path)
        all_fids    = sorted(set(gt_by_frame) | set(pr_by_frame))

        for fi in all_fids:
            key = (seq, fi)
            if key not in img_id_map:
                img_id_map[key] = img_id_counter
                coco_images.append({"id": img_id_counter, "width": IMG_W, "height": IMG_H})
                img_id_counter += 1
            iid = img_id_map[key]

            gts, prs = gt_by_frame.get(fi, []), pr_by_frame.get(fi, [])
            
            gt_data = [preprocess_annotation(a, sigma_rad, inv_2sigma2, cutoff_rad, args.energy_thr) for a in gts]
            pr_data = [preprocess_annotation(a, sigma_rad, inv_2sigma2, cutoff_rad, args.energy_thr) for a in prs]

            valid_gt = [obj for obj in gt_data if obj["valid"]]
            valid_pr = [obj for obj in pr_data if obj["valid"]]

            for obj in valid_gt:
                coco_gt_anns.append({
                    "id": gt_ann_counter, "image_id": iid, "category_id": obj["cid"] + 1,
                    "segmentation": obj["rle"], "area": obj["area"], "bbox": obj["bbox"], "iscrowd": 0,
                })
                gt_ann_counter += 1

            for obj in valid_pr:
                coco_dt_anns.append({
                    "image_id": iid, "category_id": obj["cid"] + 1,
                    "segmentation": obj["rle"], "score": obj["score"],
                })

            for gt_idx, pr_idx, _ in hungarian_match(valid_gt, valid_pr):
                gt_obj, pr_obj = valid_gt[gt_idx], valid_pr[pr_idx]
                
                # FIX: Switched to Mean Absolute Percentage Error (MAPE)
                gt_dist = float(gt_obj["ann"].get("distance", 0.0))
                pr_dist = float(pr_obj["ann"].get("distance", 0.0))
                if gt_dist > 1e-3:  # Guard against division by zero
                    dist_errors.append(abs(gt_dist - pr_dist) / gt_dist * 100.0)

                union = gt_obj["bmask"] | pr_obj["bmask"]
                if not union.any(): continue

                mask_dilated = binary_dilation(union, iterations=2)
                g_vec = gt_obj["emap"][mask_dilated]
                p_vec = pr_obj["emap"][mask_dilated]

                if np.ptp(g_vec) > 1e-7 and np.ptp(p_vec) > 1e-7:
                    r_val = float(np.corrcoef(g_vec, p_vec)[0, 1])
                    if not math.isnan(r_val): pearsons.append(r_val)

    # ── COCO Mask mAP ──────────────────────────────────────────────────────────
    print(f"\n[INFO] Running COCO Mask mAP ({len(coco_gt_anns)} GT | {len(coco_dt_anns)} pred | {len(coco_images)} frames) ...")
    
    coco_gt = COCO()
    coco_gt.dataset = {"images": coco_images, "annotations": coco_gt_anns, "categories": COCO_CATS}
    with redirect_stdout(io.StringIO()): coco_gt.createIndex()

    mask_ap, mask_ap50 = float("nan"), float("nan")
    per_cls_ap = {}

    if coco_dt_anns:
        with redirect_stdout(io.StringIO()): coco_dt = coco_gt.loadRes(coco_dt_anns)
        ev = COCOeval(coco_gt, coco_dt, iouType="segm")
        ev.evaluate()
        ev.accumulate()
        with redirect_stdout(io.StringIO()): ev.summarize()

        mask_ap, mask_ap50 = float(ev.stats[0]), float(ev.stats[1])
        prec = ev.eval["precision"] 
        for k_idx, cat in enumerate(COCO_CATS):
            cls_id = cat["id"] - 1
            p_kv = prec[:, :, k_idx, 0, 2]
            valid = p_kv[p_kv > -1]
            per_cls_ap[cls_id] = float(np.mean(valid)) if len(valid) > 0 else float("nan")
    else:
        print("[WARN] No predictions survived the energy threshold -- Mask mAP = N/A.")

    # ── Report ─────────────────────────────────────────────────────────────────
    mean_pearson  = float(np.mean(pearsons))    if pearsons    else float("nan")
    mean_dist_err = float(np.mean(dist_errors)) if dist_errors else float("nan")

    def fmt(v): return f"{v:.4f}" if not math.isnan(v) else "  N/A  "

    lines = [
        "=================================================================",
        "  ENERGY-FIELD SEGMENTATION -- EVALUATION REPORT",
        "=================================================================",
        f"  Gaussian sigma          : {args.sigma}deg of arc (spherical)",
        f"  Match threshold         : {MATCH_THR_DEG}deg (great-circle, peak-to-peak)",
        "-----------------------------------------------------------------",
        f"  Mask mAP  (IoU 0.50:0.95)  : {fmt(mask_ap)}",
        f"  Mask AP50 (IoU = 0.50)     : {fmt(mask_ap50)}",
        f"  Pearson r (alignment)      : {fmt(mean_pearson)}",
        f"  Distance MAPE (Relative)   : {mean_dist_err:.2f} %" if not math.isnan(mean_dist_err) else "  Distance MAPE (Relative)   :   N/A",
        "-----------------------------------------------------------------",
        "  PER-CLASS Mask mAP:",
    ]
    for cid in range(N_CLS): lines.append(f"  {CLASS_NAMES[cid]:<24s} : {fmt(per_cls_ap.get(cid, float('nan')))}")
    
    report = "\n".join(lines)
    print("\n" + report)

    out_path = os.path.join(args.exp_dir, "eval_summary.txt")
    with open(out_path, "w") as fh: fh.write(report + "\n")
    print(f"\n[INFO] Report saved -> {out_path}")

if __name__ == "__main__":
    main()
