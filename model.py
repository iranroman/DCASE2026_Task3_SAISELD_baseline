import os
import json
import glob
import re
import hashlib
import pickle
from collections import defaultdict, OrderedDict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou, MultiScaleRoIAlign
from scipy.optimize import linear_sum_assignment

# Assumes acoustic_features is accessible in your path
from acoustic_features import wav_path_from_seq_dir


# ── 1. SEQUENCE / FRAME UTILITIES ────────────────────────────────────────────

def json_to_seq_name(json_path: str) -> str:
    stem = os.path.splitext(os.path.basename(json_path))[0]
    return re.sub(r"_std$", "", stem, flags=re.IGNORECASE)

def frame_path(seq_dir: str, seq_name: str, idx: int) -> str:
    return os.path.join(seq_dir, f"{seq_name}_{idx:04d}.png")

def scan_available_frames(seq_dir: str, seq_name: str) -> list:
    pattern = os.path.join(seq_dir, f"{seq_name}_*.png")
    out = []
    for p in sorted(glob.glob(pattern)):
        m = re.search(r"_(\d{4})\.png$", os.path.basename(p))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)

def get_sequence_infos(split_keyword: str, labels_base: str, frames_base: str) -> list:
    """
    Scans labels_base and frames_base for subdirectories containing split_keyword.
    Returns a list of (json_path, seq_dir, seq_name).
    """
    infos = []
    if not os.path.isdir(labels_base) or not os.path.isdir(frames_base):
        return infos

    for split_dir in os.listdir(labels_base):
        if split_keyword not in split_dir:
            continue
        split_path = os.path.join(labels_base, split_dir)
        if not os.path.isdir(split_path):
            continue

        for json_file in sorted(glob.glob(os.path.join(split_path, "*.json"))):
            seq_name = json_to_seq_name(json_file)
            seq_dir = os.path.join(frames_base, split_dir, seq_name)
            infos.append((json_file, seq_dir, seq_name))
    return infos


# ── 2. DATASET & CACHING ──────────────────────────────────────────────────────

def _dataset_cache_fingerprint(sequence_infos: list) -> str:
    h = hashlib.md5()
    for json_path, seq_dir, seq_name in sorted(sequence_infos):
        if os.path.exists(json_path):
            mtime = str(os.path.getmtime(json_path))
        else:
            mtime = "missing"
        h.update(f"{json_path}:{mtime}:{seq_dir}:{seq_name}".encode())
    return h.hexdigest()[:16]

class EnergySegDataset(Dataset):
    CACHE_DIR = ".dataset_cache"

    def __init__(self, sequence_infos: list, frames_base: str, mic_base: str, 
                 acoustic_extractor, frames_per_epoch: int = None,
                 img_w: int = 360, img_h: int = 180, dist_norm: float = 500.0,
                 cache_max_size: int = 200):
        self.frames_base = frames_base
        self.mic_base = mic_base
        self.acoustic_extractor = acoustic_extractor
        self.frames_per_epoch = frames_per_epoch
        self.img_w = img_w
        self.img_h = img_h
        self.dist_norm = dist_norm
        self.cache_max_size = cache_max_size
        
        self.frame_cache = OrderedDict()
        self.current_indices = []
        self.all_samples = []

        os.makedirs(self.CACHE_DIR, exist_ok=True)
        fingerprint = _dataset_cache_fingerprint(sequence_infos)
        cache_path = os.path.join(self.CACHE_DIR, f"samples_{fingerprint}.pkl")

        if os.path.exists(cache_path):
            print(f"[Dataset] Cache hit  — loading from {cache_path}")
            with open(cache_path, "rb") as f:
                self.all_samples = pickle.load(f)
            n_pos = sum(1 for _, anns in self.all_samples if anns)
            n_neg = len(self.all_samples) - n_pos
            n_ann = sum(len(anns) for _, anns in self.all_samples)
            print(f"[Dataset] Loaded {len(self.all_samples)} frames from cache "
                  f"({n_pos} annotated / {n_neg} background-only) | {n_ann} annotations")
        else:
            print(f"[Dataset] Cache miss — scanning {len(sequence_infos)} sequences "
                  f"(this may take a minute; result will be cached at {cache_path})")
            self._scan(sequence_infos)
            with open(cache_path, "wb") as f:
                pickle.dump(self.all_samples, f)
            print(f"[Dataset] Scan complete. Cache saved to {cache_path}")

        if self.frames_per_epoch is None:
            print(f"[Dataset] frames_per_epoch=None → using all "
                  f"{len(self.all_samples)} frames per epoch (full dataset mode)")
        else:
            print(f"[Dataset] frames_per_epoch={self.frames_per_epoch} → random "
                  f"subsample drawn each epoch from {len(self.all_samples)} available")

        self.reset_epoch()

    def _scan(self, sequence_infos: list):
        n_pos, n_neg, n_ann = 0, 0, 0
        for json_path, seq_dir, seq_name in sequence_infos:
            with open(json_path) as f:
                data = json.load(f)

            by_frame: dict = defaultdict(list)
            for ann in data.get("annotations", []):
                by_frame[int(ann["metadata_frame_index"])].append(ann)

            annotated_ids = set(by_frame.keys())
            disk_ids      = set(scan_available_frames(seq_dir, seq_name))
            all_ids       = sorted(annotated_ids | disk_ids)

            for fi in all_ids:
                anns = by_frame.get(fi, [])
                self.all_samples.append(((seq_dir, seq_name, fi), anns))
                if anns:
                    n_pos += 1
                    n_ann += len(anns)
                else:
                    n_neg += 1

        print(f"[Dataset] Scanned {len(sequence_infos)} sequences | "
              f"{len(self.all_samples)} frames total ({n_pos} pos / {n_neg} neg) | "
              f"{n_ann} annotations")

    def clear_caches(self):
        self.frame_cache.clear()
        self.cache_max_size = 0  
        if self.acoustic_extractor is not None:
            self.acoustic_extractor.clear_cache()

    def load_frame_tensor(self, seq_dir: str, seq_name: str, frame_idx: int, warn_missing: bool = True) -> torch.Tensor:
        key = (seq_name, frame_idx)

        if key in self.frame_cache:
            tensor = self.frame_cache.pop(key)
            self.frame_cache[key] = tensor
            return tensor

        path = frame_path(seq_dir, seq_name, frame_idx)
        if os.path.isfile(path):
            img = Image.open(path).convert("RGB")
            if img.size != (self.img_w, self.img_h):
                img = img.resize((self.img_w, self.img_h), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            rgb = torch.from_numpy(arr).permute(2, 0, 1)          
        else:
            if warn_missing:
                print(f"[WARN] Frame not found — using noise: {path}")
            rng = np.random.default_rng(abs(hash(seq_name)) % 2**31 + int(frame_idx))
            rgb = torch.from_numpy(rng.random((3, self.img_h, self.img_w), dtype=np.float32))

        wav_path = wav_path_from_seq_dir(seq_dir, self.frames_base, self.mic_base)
        acoustic = self.acoustic_extractor.get_frame_bands(wav_path, frame_idx)   
        tensor = torch.cat([rgb, acoustic], dim=0)                 

        if self.cache_max_size > 0:
            self.frame_cache[key] = tensor
            if len(self.frame_cache) > self.cache_max_size:
                self.frame_cache.popitem(last=False)

        return tensor

    def precache_frames(self, seq_dir: str, seq_name: str, frame_indices: list, verbose: bool = True):
        indices = sorted(frame_indices)
        if len(indices) > self.cache_max_size:
            return
            
        if verbose: print(f"[INFO] Pre-caching {len(indices)} frames …")
        missing = sum(1 for fi in indices if not os.path.isfile(frame_path(seq_dir, seq_name, fi)))
        for fi in indices:
            self.load_frame_tensor(seq_dir, seq_name, fi, warn_missing=(missing < 10))
            
        if verbose:
            found  = len(indices) - missing
            status = "all found" if missing == 0 else f"{missing} replaced with noise"
            print(f"[INFO] Cache: {found}/{len(indices)} frames on disk ({status})")

    def build_annotation_target(self, frame_annots: list) -> dict:
        boxes, labels       = [], []
        bin_masks           = []
        energy_maps_list    = []
        energy_masks_list   = []
        distances, iids     = [], []

        combined_energy = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        combined_mask   = np.zeros((self.img_h, self.img_w), dtype=bool)

        for ann in frame_annots:
            cat  = int(ann["category_id"]) + 1
            dist = float(ann["distance"])
            iid  = int(ann["instance_id"])

            energy_map  = np.zeros((self.img_h, self.img_w), dtype=np.float32)
            energy_mask = np.zeros((self.img_h, self.img_w), dtype=bool)
            xs, ys      = [], []

            for sub in ann["segmentation"]:
                for triplet in sub:
                    x, y, v = float(triplet[0]), float(triplet[1]), float(triplet[2])
                    xi, yi  = int(round(x)), int(round(y))
                    if 0 <= xi < self.img_w and 0 <= yi < self.img_h:
                        energy_map[yi, xi]         = float(v)
                        energy_mask[yi, xi]        = True
                        combined_energy[yi, xi]    = max(combined_energy[yi, xi], float(v))
                        combined_mask[yi, xi]      = True
                        xs.append(xi); ys.append(yi)

            if not xs:
                continue

            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            if x0 == x1: x1 += 1
            if y0 == y1: y1 += 1

            boxes.append([float(x0), float(y0), float(x1), float(y1)])
            labels.append(cat)
            bin_masks.append(energy_mask.copy())
            energy_maps_list.append(energy_map.copy())
            energy_masks_list.append(energy_mask.copy())
            distances.append(dist / self.dist_norm)
            iids.append(iid)

        if not boxes:
            N = 0
            return dict(
                boxes        = torch.zeros(N, 4,              dtype=torch.float32),
                labels       = torch.zeros(N,                 dtype=torch.int64),
                masks        = torch.zeros(N, self.img_h, self.img_w, dtype=torch.bool),
                energy_maps  = torch.zeros(N, self.img_h, self.img_w, dtype=torch.float32),
                energy_masks = torch.zeros(N, self.img_h, self.img_w, dtype=torch.bool),
                vmap         = torch.zeros(self.img_h, self.img_w,    dtype=torch.float32),
                vmask        = torch.zeros(self.img_h, self.img_w,    dtype=torch.bool),
                distances    = torch.zeros(N,                 dtype=torch.float32),
                instance_ids = torch.zeros(N,                 dtype=torch.int64),
            )

        return dict(
            boxes        = torch.tensor(boxes,                           dtype=torch.float32),
            labels       = torch.tensor(labels,                          dtype=torch.int64),
            masks        = torch.tensor(np.stack(bin_masks),             dtype=torch.bool),
            energy_maps  = torch.tensor(np.stack(energy_maps_list),      dtype=torch.float32),
            energy_masks = torch.tensor(np.stack(energy_masks_list),     dtype=torch.bool),
            vmap         = torch.tensor(combined_energy,                 dtype=torch.float32),
            vmask        = torch.tensor(combined_mask,                   dtype=torch.bool),
            distances    = torch.tensor(distances,                       dtype=torch.float32),
            instance_ids = torch.tensor(iids,                            dtype=torch.int64),
        )

    def reset_epoch(self):
        total = len(self.all_samples)
        if self.frames_per_epoch is not None and self.frames_per_epoch < total:
            self.current_indices = np.random.choice(total, self.frames_per_epoch, replace=False)
        else:
            self.current_indices = np.arange(total)
            np.random.shuffle(self.current_indices)

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        real_idx = self.current_indices[idx]
        (seq_dir, seq_name, fi), anns = self.all_samples[real_idx]
        image  = self.load_frame_tensor(seq_dir, seq_name, fi)
        target = self.build_annotation_target(anns)
        return image, target

def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch]

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if hasattr(dataset, 'clear_caches'):
            dataset.clear_caches()


# ── 3. MODEL COMPONENTS ───────────────────────────────────────────────────────

class EnergyMaskHead(nn.Module):
    def __init__(self, in_ch: int = 256, mid: int = 256):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(4):
            layers += [
                nn.Conv2d(ch, mid, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
            ]
            ch = mid
        layers += [
            nn.ConvTranspose2d(mid, mid, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
            nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # [M, 1, 28, 28]

class FPNEnergyDecoder(nn.Module):
    def __init__(self, fpn_ch: int = 256, mid: int = 128, img_w: int = 360, img_h: int = 180):
        super().__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.lat    = nn.ModuleList([nn.Conv2d(fpn_ch, mid, 1) for _ in range(4)])
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid, mid, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(mid, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, fpn: dict) -> torch.Tensor:
        feats = [fpn[str(i)] for i in range(4)]
        x = self.smooth[3](self.lat[3](feats[3]))
        for i in (2, 1, 0):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="nearest")
            x = self.smooth[i](x + self.lat[i](feats[i]))
        x = F.interpolate(x, size=(self.img_h, self.img_w), mode="bilinear", align_corners=False)
        return self.head(x)

class DistanceHead(nn.Module):
    def __init__(self, feat_ch: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_ch, 128), nn.ReLU(inplace=True),
            nn.Linear(128,      64), nn.ReLU(inplace=True),
            nn.Linear(64,        1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.fc(x)).squeeze(-1)


class EnergyInstanceModel(nn.Module):
    def __init__(self, num_classes: int, n_channels: int = 12, img_w: int = 360, img_h: int = 180,
                 energy_annot_w: float = 5.0, fullmap_annot_w: float = 10.0):
        super().__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.energy_annot_w = energy_annot_w
        self.fullmap_annot_w = fullmap_annot_w

        base = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

        old = base.backbone.body.conv1
        new = nn.Conv2d(n_channels, old.out_channels,
                        old.kernel_size, old.stride,
                        old.padding, bias=(old.bias is not None))
        with torch.no_grad():
            new.weight.data[:, :3, :, :] = old.weight.data
            import torch.nn.init as init
            init.kaiming_uniform_(new.weight.data[:, 3:, :, :], a=0)
            new.weight.data[:, 3:, :, :] *= 0.01
            if old.bias is not None:
                new.bias.data.copy_(old.bias.data)
        base.backbone.body.conv1 = new

        in_box = base.roi_heads.box_predictor.cls_score.in_features
        base.roi_heads.box_predictor = FastRCNNPredictor(in_box, num_classes)

        base.roi_heads.mask_roi_pool  = None
        base.roi_heads.mask_head      = None
        base.roi_heads.mask_predictor = None

        base.transform.min_size = (min(img_h, img_w),)
        base.transform.max_size = max(img_h, img_w) * 4
        # Robust handling depending on arbitrary n_channels
        base.transform.image_mean = [0.485, 0.456, 0.406] + [0.5] * (n_channels - 3) 
        base.transform.image_std  = [0.229, 0.224, 0.225] + [0.25] * (n_channels - 3)

        self.det = base
        self.energy_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'], output_size=14, sampling_ratio=2)
        self.energy_head    = EnergyMaskHead(in_ch=256, mid=256)
        self.energy_decoder = FPNEnergyDecoder(fpn_ch=256, mid=128, img_w=img_w, img_h=img_h)
        self.distance_head  = DistanceHead(feat_ch=256)

    def _fpn4(self, features: dict) -> dict:
        return {k: features[k] for k in ('0','1','2','3') if k in features}

    def _roi_energy(self, features: dict, box_lists: list, image_sizes: list):
        M      = sum(len(b) for b in box_lists)
        device = next(iter(features.values())).device
        if M == 0:
            return (torch.zeros(0, 28, 28, device=device),
                    torch.zeros(0, device=device))

        roi_feats   = self.energy_roi_pool(features, box_lists, image_sizes)
        energy_flat = self.energy_head(roi_feats).squeeze(1)
        dist_feat   = roi_feats.mean(dim=[-2, -1])
        dist_flat   = self.distance_head(dist_feat)
        return energy_flat, dist_flat

    def _gt_roi_energy_targets(self, targets: list, device: torch.device):
        e_list, m_list = [], []
        for tgt in targets:
            boxes   = tgt["boxes"]
            emaps   = tgt["energy_maps"]
            emasks  = tgt["energy_masks"]
            for n in range(len(boxes)):
                x0, y0, x1, y1 = [max(0, int(v.item())) for v in boxes[n]]
                x1 = min(x1 + 1, self.img_w)
                y1 = min(y1 + 1, self.img_h)
                if x1 <= x0 or y1 <= y0:
                    e_list.append(torch.zeros(1, 1, 28, 28, device=device))
                    m_list.append(torch.zeros(1, 1, 28, 28, device=device))
                    continue
                crop_e = emaps[n, y0:y1, x0:x1].unsqueeze(0).unsqueeze(0).to(device)
                crop_m = emasks[n, y0:y1, x0:x1].float().unsqueeze(0).unsqueeze(0).to(device)
                r_e = F.interpolate(crop_e, (28, 28), mode='bilinear', align_corners=False)
                r_m = F.interpolate(crop_m, (28, 28), mode='bilinear', align_corners=False)
                e_list.append(r_e)
                m_list.append(r_m)

        if not e_list:
            return (torch.zeros(0, 1, 28, 28, device=device),
                    torch.zeros(0, 1, 28, 28, device=device))
        return torch.cat(e_list), torch.cat(m_list)

    def forward(self, images: list, targets: list = None):
        is_train   = self.training
        orig_sizes = [img.shape[-2:] for img in images]

        images_t, targets_t = self.det.transform(images, targets)
        img_sizes = images_t.image_sizes

        features = self.det.backbone(images_t.tensors)
        fpn4     = self._fpn4(features)
        full_energy = self.energy_decoder(fpn4)

        if is_train:
            device = images_t.tensors.device

            proposals,  rpn_losses = self.det.rpn(images_t, features, targets_t)
            _,          roi_losses = self.det.roi_heads(features, proposals, img_sizes, targets_t)

            gt_boxes = [tgt["boxes"].to(device) for tgt in targets]
            energy_flat, dist_flat = self._roi_energy(fpn4, gt_boxes, img_sizes)

            gt_e, gt_m = self._gt_roi_energy_targets(targets, device)
            M = energy_flat.shape[0]

            if M > 0:
                pred    = energy_flat.unsqueeze(1)
                loss_e_base = F.mse_loss(pred, gt_e)
                ann_px = (gt_m > 0.5)
                if ann_px.any():
                    loss_e_ann = F.mse_loss(pred[ann_px], gt_e[ann_px])
                else:
                    loss_e_ann = pred.new_zeros(1).squeeze()
                energy_loss = loss_e_base + self.energy_annot_w * loss_e_ann
            else:
                energy_loss = images_t.tensors.new_zeros(1).squeeze()

            gt_dists = torch.cat([tgt["distances"].to(device) for tgt in targets])
            if len(gt_dists) > 0 and len(dist_flat) == len(gt_dists):
                dist_loss = F.smooth_l1_loss(dist_flat, gt_dists)
            else:
                dist_loss = images_t.tensors.new_zeros(1).squeeze()

            vmap_loss = images_t.tensors.new_zeros(1).squeeze()
            for i, tgt in enumerate(targets):
                pred_i = full_energy[i, 0]
                gt_v   = tgt["vmap"].to(device)
                vm     = tgt["vmask"].to(device)

                W = torch.ones_like(pred_i)
                W[vm] = self.fullmap_annot_w
                vmap_loss = vmap_loss + (W * (pred_i - gt_v).pow(2)).mean()

            vmap_loss = vmap_loss / max(len(targets), 1)

            return {
                **rpn_losses,
                **roi_losses,
                "loss_energy_mask": energy_loss,
                "loss_fullmap":     vmap_loss,
                "loss_distance":    dist_loss,
            }

        else:
            proposals,        _ = self.det.rpn(images_t, features, None)
            detections_raw,   _ = self.det.roi_heads(features, proposals, img_sizes, None)

            all_boxes  = [raw["boxes"] for raw in detections_raw]
            counts     = [len(b) for b in all_boxes]

            if sum(counts) > 0:
                e_flat, d_flat = self._roi_energy(fpn4, all_boxes, img_sizes)
                e_per_img      = list(e_flat.split(counts))
                d_per_img      = list(d_flat.split(counts))
            else:
                e_per_img = [torch.zeros(0, 28, 28) for _ in detections_raw]
                d_per_img = [torch.zeros(0)         for _ in detections_raw]

            detections = self.det.transform.postprocess(detections_raw, img_sizes, orig_sizes)

            for i, det in enumerate(detections):
                det["energy_maps"] = e_per_img[i].detach().cpu()
                det["dist_pred"]   = d_per_img[i].detach().cpu()
                det["full_energy"] = full_energy[i, 0].detach().cpu()

            return detections


# ── 4. INSTANCE TRACKER ───────────────────────────────────────────────────────

class InstanceTracker:
    def __init__(self, iou_thr: float = 0.3, max_age: int = 5, img_w: int = 360, img_h: int = 180):
        self.iou_thr  = iou_thr
        self.max_age  = max_age
        self.img_w    = img_w
        self.img_h    = img_h
        self._nxt     = 0
        self.tracks: dict = {}

    def update(self, det: dict) -> list:
        boxes  = det.get("boxes",       torch.zeros(0, 4))
        labels = det.get("labels",      torch.zeros(0, dtype=torch.long))
        scores = det.get("scores",      torch.zeros(0))
        emaps  = det.get("energy_maps", torch.zeros(0, 28, 28))
        dists  = det.get("dist_pred",   torch.zeros(0))
        full_e = det.get("full_energy", torch.zeros(self.img_h, self.img_w))
        N      = len(boxes)

        if N == 0:
            for d in self.tracks.values(): d["age"] += 1
            dead = [t for t, d in self.tracks.items() if d["age"] > self.max_age]
            for t in dead: del self.tracks[t]
            return []

        track_ids   = list(self.tracks.keys())
        track_boxes = (torch.stack([self.tracks[t]["box"] for t in track_ids])
                       if track_ids else torch.zeros(0, 4))

        iou_mat = (box_iou(track_boxes.cpu(), boxes.cpu()).numpy()
                   if track_boxes.numel() > 0 else np.zeros((0, N), dtype=np.float32))

        matched_det, matched_trk = {}, set()
        if iou_mat.size > 0:
            row_ind, col_ind = linear_sum_assignment(1.0 - iou_mat)
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_thr:
                    tid = track_ids[r]
                    self.tracks[tid].update(
                        box=boxes[c].cpu(), age=0,
                        label=int(labels[c]),
                        energy_map=(emaps[c].cpu() if c < emaps.shape[0] else torch.zeros(28, 28)),
                        dist_pred=(float(dists[c]) if c < dists.shape[0] else 0.0),
                        full_energy=full_e,
                    )
                    self.tracks[tid]["hits"] += 1
                    matched_det[c] = tid
                    matched_trk.add(r)

        for c in range(N):
            if c not in matched_det:
                tid = self._nxt; self._nxt += 1
                self.tracks[tid] = dict(
                    box        = boxes[c].cpu(),
                    age        = 0, hits=1,
                    label      = int(labels[c]),
                    energy_map = (emaps[c].cpu() if c < emaps.shape[0] else torch.zeros(28, 28)),
                    dist_pred  = (float(dists[c]) if c < dists.shape[0] else 0.0),
                    full_energy= full_e,
                )
                matched_det[c] = tid

        for r, tid in enumerate(track_ids):
            if r not in matched_trk:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]

        results = []
        for c in range(N):
            tid = matched_det.get(c, -1)
            results.append(dict(
                track_id   = tid,
                label      = int(labels[c]),
                score      = float(scores[c]),
                box        = boxes[c].cpu().tolist(),
                energy_map = (emaps[c].cpu() if c < emaps.shape[0] else torch.zeros(28, 28)),
                dist_pred  = (float(dists[c]) if c < dists.shape[0] else 0.0),
                full_energy= full_e,
            ))
        return results