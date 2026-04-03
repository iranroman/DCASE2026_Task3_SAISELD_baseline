import os
import json
import glob
import re
import math
import hashlib
import pickle
import sqlite3
import types
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

_DB_SCHEMA_VERSION = "v3"   

def _dataset_cache_fingerprint(sequence_infos: list) -> str:
    h = hashlib.md5()
    h.update(_DB_SCHEMA_VERSION.encode())   
    for json_path, seq_dir, seq_name in sorted(sequence_infos):
        if os.path.exists(json_path):
            mtime = str(os.path.getmtime(json_path))
        else:
            mtime = "missing"
        h.update(f"{json_path}:{mtime}:{seq_dir}:{seq_name}".encode())
    return h.hexdigest()[:16]

class EnergySegDataset(Dataset):
    CACHE_DIR = ".dataset_cache"

    def __init__(
        self,
        sequence_infos: list,
        frames_base: str,
        mic_base: str,
        acoustic_extractor,
        frames_per_epoch: int = None,
        img_w: int = 360,
        img_h: int = 180,
        dist_norm: float = 500.0,
        cache_max_size: int = 200,
        augmentor=None,
    ):
        self.frames_base        = frames_base
        self.mic_base           = mic_base
        self.acoustic_extractor = acoustic_extractor
        self.frames_per_epoch   = frames_per_epoch
        self.img_w              = img_w
        self.img_h              = img_h
        self.dist_norm          = dist_norm
        self.cache_max_size     = cache_max_size
        self.augmentor          = augmentor

        self.frame_cache    = OrderedDict()

        self.seq_map        = {}
        self.all_samples    = []
        self.current_indices= []
        self.class_to_sample_indices: dict = defaultdict(list)

        self.db_conn        = None

        os.makedirs(self.CACHE_DIR, exist_ok=True)
        fingerprint   = _dataset_cache_fingerprint(sequence_infos)
        self.db_path  = os.path.join(self.CACHE_DIR, f"annotations_{fingerprint}.db")

        if os.path.exists(self.db_path):
            print(f"[Dataset] Cache hit  — loading index from {self.db_path}")
            self._load_index_from_db()
        else:
            print(f"[Dataset] Cache miss — building SQLite database from JSONs...")
            self._build_db(sequence_infos)

        if self.frames_per_epoch is None:
            print(f"[Dataset] frames_per_epoch=None → using all {len(self.all_samples)} frames")
        else:
            print(f"[Dataset] frames_per_epoch={self.frames_per_epoch} → subsampling")

        self.reset_epoch()

    def _get_db_conn(self):
        if self.db_conn is None:
            import pathlib
            db_uri       = pathlib.Path(self.db_path).absolute().as_uri()
            self.db_conn = sqlite3.connect(f"{db_uri}?mode=ro", uri=True)
        return self.db_conn

    def _build_db(self, sequence_infos: list):
        from tqdm import tqdm

        conn = sqlite3.connect(self.db_path)
        c    = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS annots (seq_id INTEGER, frame_idx INTEGER, data BLOB)")
        c.execute("CREATE TABLE IF NOT EXISTS seqs (seq_id INTEGER, seq_dir TEXT, seq_name TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS frame_classes (sample_idx INTEGER, category_id INTEGER)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_lookup ON annots (seq_id, frame_idx)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_fc ON frame_classes (category_id)")

        seq_id              = 0
        n_pos, n_neg, n_ann = 0, 0, 0

        for json_path, seq_dir, seq_name in tqdm(sequence_infos, desc="[Dataset] Building SQLite DB", leave=False):
            c.execute("INSERT INTO seqs VALUES (?, ?, ?)", (seq_id, seq_dir, seq_name))
            self.seq_map[seq_id] = (seq_dir, seq_name)

            with open(json_path) as f:
                data = json.load(f)

            by_frame = defaultdict(list)
            for ann in data.get("annotations", []):
                by_frame[int(ann["metadata_frame_index"])].append(ann)

            annotated_ids = set(by_frame.keys())
            disk_ids      = set(scan_available_frames(seq_dir, seq_name))
            all_ids       = sorted(annotated_ids | disk_ids)

            for fi in all_ids:
                anns       = by_frame.get(fi, [])
                sample_idx = len(self.all_samples)
                self.all_samples.append((seq_id, fi))
                c.execute("INSERT INTO annots VALUES (?, ?, ?)", (seq_id, fi, pickle.dumps(anns)))

                for ann in anns:
                    cat_id = int(ann["category_id"])
                    c.execute("INSERT INTO frame_classes VALUES (?, ?)", (sample_idx, cat_id))
                    self.class_to_sample_indices[cat_id].append(sample_idx)

                if anns:
                    n_pos += 1
                    n_ann += len(anns)
                else:
                    n_neg += 1

            seq_id += 1

        conn.commit()
        conn.close()
        print(f"[Dataset] SQLite DB built. {len(sequence_infos)} sequences | "
              f"{len(self.all_samples)} frames ({n_pos} pos / {n_neg} neg) | "
              f"{n_ann} annotations | {len(self.class_to_sample_indices)} classes indexed")

    def _load_index_from_db(self):
        from tqdm import tqdm

        conn = sqlite3.connect(self.db_path)
        c    = conn.cursor()

        for row in c.execute("SELECT seq_id, seq_dir, seq_name FROM seqs"):
            self.seq_map[row[0]] = (row[1], row[2])

        c.execute("SELECT COUNT(*) FROM annots")
        total_rows = c.fetchone()[0]

        c.execute("SELECT seq_id, frame_idx FROM annots")
        for row in tqdm(c, total=total_rows, desc="[Dataset] Loading Index from DB", leave=False):
            self.all_samples.append((row[0], row[1]))

        for row in c.execute("SELECT sample_idx, category_id FROM frame_classes"):
            self.class_to_sample_indices[row[1]].append(row[0])

        if self.class_to_sample_indices:
            n_pairs = sum(len(v) for v in self.class_to_sample_indices.values())
            print(f"[Dataset] Class index loaded: {len(self.class_to_sample_indices)} classes, "
                  f"{n_pairs} annotated frame-class pairs")

        conn.close()

    def clear_caches(self):
        self.frame_cache.clear()
        if self.acoustic_extractor is not None:
            self.acoustic_extractor.clear_cache()

    def __del__(self):
        if self.db_conn is not None:
            self.db_conn.close()

    def load_frame_tensor(self, seq_dir: str, seq_name: str, frame_idx: int, warn_missing: bool = True) -> torch.Tensor:
        key = (seq_name, frame_idx)

        if key in self.frame_cache:
            tensor = self.frame_cache.pop(key)
            self.frame_cache[key] = tensor    
            return tensor.clone()

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
        tensor   = torch.cat([rgb, acoustic], dim=0)

        if self.cache_max_size > 0:
            self.frame_cache[key] = tensor
            if len(self.frame_cache) > self.cache_max_size:
                self.frame_cache.popitem(last=False)

        return tensor.clone()

    def precache_frames(self, seq_dir: str, seq_name: str, frame_indices: list, verbose: bool = True):
        indices = sorted(frame_indices)
        if len(indices) > self.cache_max_size:
            return

        if verbose:
            print(f"[INFO] Pre-caching {len(indices)} frames …")
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
                        energy_map[yi, xi]      = float(v)
                        energy_mask[yi, xi]     = True
                        combined_energy[yi, xi] = max(combined_energy[yi, xi], float(v))
                        combined_mask[yi, xi]   = True
                        xs.append(xi); ys.append(yi)

            if not xs:
                continue

            x0 = min(xs)
            x1 = max(xs) + 1   # exclusive right
            y0 = min(ys)
            y1 = max(ys) + 1   # exclusive bottom

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

    # ADDED 'balanced' parameter to allow train.py to disable it dynamically
    def reset_epoch(self, balanced: bool = True):
        total = len(self.all_samples)
        n     = min(self.frames_per_epoch, total) if self.frames_per_epoch is not None else total

        use_balanced = (
            balanced
            and self.augmentor is not None
            and bool(self.class_to_sample_indices)
        )

        if not use_balanced:
            indices = np.arange(total)
            np.random.shuffle(indices)
            self.current_indices = indices[:n]
            return

        n_uniform  = min(n // 2, total)
        n_balanced = n - n_uniform

        uniform_part  = np.random.choice(total, n_uniform, replace=False)

        classes       = list(self.class_to_sample_indices.keys())
        balanced_part = np.empty(n_balanced, dtype=np.int64)
        for i in range(n_balanced):
            cls              = classes[int(np.random.randint(len(classes)))]
            pool             = self.class_to_sample_indices[cls]
            balanced_part[i] = pool[int(np.random.randint(len(pool)))]

        self.current_indices = np.random.permutation(
            np.concatenate([uniform_part, balanced_part])
        )

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        real_idx          = self.current_indices[idx]
        seq_id, fi        = self.all_samples[real_idx]
        seq_dir, seq_name = self.seq_map[seq_id]

        c = self._get_db_conn().cursor()
        c.execute("SELECT data FROM annots WHERE seq_id=? AND frame_idx=?", (seq_id, fi))
        row  = c.fetchone()
        anns = pickle.loads(row[0]) if row and row[0] else []

        image  = self.load_frame_tensor(seq_dir, seq_name, fi)
        target = self.build_annotation_target(anns)

        if self.augmentor is not None:
            image, target = self.augmentor(image, target)

        return image, target

def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch]

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if hasattr(dataset, "frame_cache"):
            dataset.frame_cache.clear()


# ── 3. MODEL COMPONENTS ───────────────────────────────────────────────────────

class EnergyMaskHead(nn.Module):
    def __init__(self, in_ch: int = 256, mid: int = 256, dropout: float = 0.2):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(4):
            layers += [
                nn.Conv2d(ch, mid, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
            ]
            if i in (1, 3):
                layers.append(nn.Dropout2d(dropout))
            ch = mid
        layers += [
            nn.ConvTranspose2d(mid, mid, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
            nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   


class FPNEnergyDecoder(nn.Module):
    def __init__(
        self,
        fpn_ch: int = 256,
        mid: int = 128,
        img_w: int = 360,
        img_h: int = 180,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img_w  = img_w
        self.img_h  = img_h
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
            nn.Dropout2d(dropout),
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
    def __init__(self, feat_ch: int = 256, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_ch, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.fc(x)).squeeze(-1)


class EnergyInstanceModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_channels: int = 12,
        img_w: int = 360,
        img_h: int = 180,
        energy_annot_w: float = 5.0,
        fullmap_annot_w: float = 10.0, # CHANGED default to 10.0 to match strategy
        dropout: float = 0.6,
    ):
        super().__init__()
        self.img_w           = img_w
        self.img_h           = img_h
        self.energy_annot_w  = energy_annot_w
        self.fullmap_annot_w = fullmap_annot_w

        base = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

        old = base.backbone.body.conv1
        new = nn.Conv2d(
            n_channels, old.out_channels,
            old.kernel_size, old.stride, old.padding,
            bias=(old.bias is not None),
        )
        with torch.no_grad():
            new.weight.data[:, :3, :, :] = old.weight.data
            import torch.nn.init as init
            init.kaiming_uniform_(new.weight.data[:, 3:, :, :], a=math.sqrt(5))
            if old.bias is not None:
                new.bias.data.copy_(old.bias.data)
        base.backbone.body.conv1 = new

        in_box = base.roi_heads.box_predictor.cls_score.in_features
        base.roi_heads.box_predictor = FastRCNNPredictor(in_box, num_classes)

        box_head = base.roi_heads.box_head
        if hasattr(box_head, 'fc6') and hasattr(box_head, 'fc7'):
            def _box_head_forward(self_box, x):
                x = x.flatten(start_dim=1)
                x = self_box.drop1(F.relu(self_box.fc6(x)))
                x = self_box.drop2(F.relu(self_box.fc7(x)))
                return x
            
            box_head.drop1 = nn.Dropout(p=dropout)
            box_head.drop2 = nn.Dropout(p=dropout)
            box_head.forward = types.MethodType(_box_head_forward, box_head)

        base.roi_heads.mask_roi_pool  = None
        base.roi_heads.mask_head      = None
        base.roi_heads.mask_predictor = None

        base.transform.min_size   = (min(img_h, img_w),)
        base.transform.max_size   = max(img_h, img_w) * 4
        
        base.transform.image_mean = [0.485, 0.456, 0.406] + [0.0] * (n_channels - 3)
        base.transform.image_std  = [0.229, 0.224, 0.225] + [1.0] * (n_channels - 3)

        self.det = base
        
        self.acoustic_norm = nn.InstanceNorm2d(n_channels - 3, affine=True)

        self.energy_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0","1","2","3"], output_size=14, sampling_ratio=2)
        self.energy_head     = EnergyMaskHead(in_ch=256, mid=256, dropout=0.2)
        self.energy_decoder  = FPNEnergyDecoder(fpn_ch=256, mid=128, img_w=img_w, img_h=img_h, dropout=0.1)
        self.distance_head   = DistanceHead(feat_ch=256, dropout=0.3)

    def _fpn4(self, features: dict) -> dict:
        return {k: features[k] for k in ("0","1","2","3") if k in features}

    @staticmethod
    def _jitter_boxes(
        boxes: torch.Tensor,
        img_h: int,
        img_w: int,
        jitter_frac: float = 0.1,
    ) -> torch.Tensor:
        if boxes.shape[0] == 0:
            return boxes
        w      = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
        h      = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
        scale  = torch.stack([w, h, w, h], dim=1) * jitter_frac
        delta  = (torch.rand_like(scale) * 2.0 - 1.0) * scale
        jittered        = boxes + delta
        jittered[:, 0] = jittered[:, 0].clamp(min=0.0, max=float(img_w) - 1.0)
        jittered[:, 1] = jittered[:, 1].clamp(min=0.0, max=float(img_h) - 1.0)
        jittered[:, 2] = torch.maximum(jittered[:, 2], jittered[:, 0] + 1.0).clamp_max(float(img_w))
        jittered[:, 3] = torch.maximum(jittered[:, 3], jittered[:, 1] + 1.0).clamp_max(float(img_h))
        return jittered

    def _roi_energy(self, features: dict, box_lists: list, image_sizes: list):
        M      = sum(len(b) for b in box_lists)
        device = next(iter(features.values())).device
        if M == 0:
            return (torch.zeros(0, 28, 28, device=device),
                    torch.zeros(0,         device=device))

        roi_feats   = self.energy_roi_pool(features, box_lists, image_sizes)
        energy_flat = self.energy_head(roi_feats).squeeze(1)
        dist_feat   = roi_feats.mean(dim=[-2, -1])
        dist_flat   = self.distance_head(dist_feat)
        return energy_flat, dist_flat

    def _gt_roi_energy_targets(
        self,
        targets: list,
        gt_boxes_t: list,
        img_sizes: list,
        orig_sizes: list,
        device: torch.device,
    ) -> tuple:
        e_list, m_list = [], []

        for tgt, boxes_t, (th, tw), (oh, ow) in zip(
            targets, gt_boxes_t, img_sizes, orig_sizes
        ):
            emaps  = tgt["energy_maps"]    
            emasks = tgt["energy_masks"]   

            for n in range(len(boxes_t)):
                x0 = max(0,  int(math.floor(boxes_t[n, 0].item() * ow / tw)))
                y0 = max(0,  int(math.floor(boxes_t[n, 1].item() * oh / th)))
                x1 = min(ow, int(math.ceil (boxes_t[n, 2].item() * ow / tw)))
                y1 = min(oh, int(math.ceil (boxes_t[n, 3].item() * oh / th)))

                if x1 <= x0 or y1 <= y0:
                    e_list.append(torch.zeros(1, 1, 28, 28, device=device))
                    m_list.append(torch.zeros(1, 1, 28, 28, device=device))
                    continue

                crop_e = emaps[n, y0:y1, x0:x1].unsqueeze(0).unsqueeze(0).to(device)
                crop_m = emasks[n, y0:y1, x0:x1].float().unsqueeze(0).unsqueeze(0).to(device)
                r_e = F.interpolate(crop_e, (28, 28), mode="bilinear", align_corners=False)
                r_m = F.interpolate(crop_m, (28, 28), mode="bilinear", align_corners=False)
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
        img_sizes   = images_t.image_sizes

        x = images_t.tensors
        rgb_feats = x[:, :3, :, :]
        acoustic_feats = x[:, 3:, :, :]
        
        acoustic_normed = self.acoustic_norm(acoustic_feats)
        
        x_normed = torch.cat([rgb_feats, acoustic_normed], dim=1)
        features = self.det.backbone(x_normed)

        fpn4        = self._fpn4(features)
        full_energy = self.energy_decoder(fpn4)

        if is_train:
            device = images_t.tensors.device

            proposals, rpn_losses = self.det.rpn(images_t, features, targets_t)
            _,         roi_losses = self.det.roi_heads(features, proposals, img_sizes, targets_t)

            gt_boxes_t = [
                self._jitter_boxes(tgt["boxes"].to(device), h, w)
                for tgt, (h, w) in zip(targets_t, img_sizes)
            ]
            energy_flat, dist_flat = self._roi_energy(fpn4, gt_boxes_t, img_sizes)

            gt_e, gt_m = self._gt_roi_energy_targets(
                targets, gt_boxes_t, img_sizes, orig_sizes, device
            )
            M = energy_flat.shape[0]

            if M > 0:
                pred        = energy_flat.unsqueeze(1)
                loss_e_base = F.mse_loss(pred, gt_e)
                ann_px      = gt_m > 0.5
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
                W      = torch.ones_like(pred_i)
                W[vm]  = self.fullmap_annot_w
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

            all_boxes = [raw["boxes"] for raw in detections_raw]
            counts    = [len(b) for b in all_boxes]

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

        def _track_to_result(tid: int, coasting: bool) -> dict:
            trk = self.tracks[tid]
            return dict(
                track_id    = tid,
                label       = trk["label"],
                score       = trk.get("last_score", 0.0) if coasting else None,
                box         = trk["box"].tolist(),
                energy_map  = trk["energy_map"],
                dist_pred   = trk["dist_pred"],
                full_energy = trk["full_energy"],
                coasting    = coasting,
            )

        if N == 0:
            results = []
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
                else:
                    results.append(_track_to_result(tid, coasting=True))
            return results

        track_ids    = list(self.tracks.keys())
        track_boxes  = (torch.stack([self.tracks[t]["box"] for t in track_ids])
                        if track_ids else torch.zeros(0, 4))
        
        track_labels = np.array([self.tracks[t]["label"] for t in track_ids])

        iou_mat = (box_iou(track_boxes.cpu(), boxes.cpu()).numpy()
                   if track_boxes.numel() > 0 else np.zeros((0, N), dtype=np.float32))

        matched_det, matched_trk = {}, set()
        if iou_mat.size > 0:
            cost = 1.0 - iou_mat
            
            for r in range(len(track_ids)):
                for c in range(N):
                    if track_labels[r] != int(labels[c]):
                        cost[r, c] = 1000.0
            
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_thr and cost[r, c] < 1000.0:
                    tid = track_ids[r]
                    self.tracks[tid].update(
                        box         = boxes[c].cpu(),
                        age         = 0,
                        label       = int(labels[c]),
                        energy_map  = (emaps[c].cpu() if c < emaps.shape[0] else torch.zeros(28, 28)),
                        dist_pred   = (float(dists[c]) if c < dists.shape[0] else 0.0),
                        full_energy = full_e,
                        last_score  = float(scores[c]), 
                    )
                    self.tracks[tid]["hits"] += 1
                    matched_det[c] = tid
                    matched_trk.add(r)

        for c in range(N):
            if c not in matched_det:
                tid = self._nxt; self._nxt += 1
                self.tracks[tid] = dict(
                    box         = boxes[c].cpu(),
                    age         = 0, hits=1,
                    label       = int(labels[c]),
                    energy_map  = (emaps[c].cpu() if c < emaps.shape[0] else torch.zeros(28, 28)),
                    dist_pred   = (float(dists[c]) if c < dists.shape[0] else 0.0),
                    full_energy = full_e,
                    last_score  = float(scores[c]), 
                )
                matched_det[c] = tid

        for r, tid in enumerate(track_ids):
            if r not in matched_trk:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]

        results = []
        for c in range(N):
            tid = matched_det[c]
            r   = _track_to_result(tid, coasting=False)
            r["score"] = float(scores[c])
            results.append(r)

        for r_idx, tid in enumerate(track_ids):
            if r_idx not in matched_trk and tid in self.tracks:
                results.append(_track_to_result(tid, coasting=True))

        return results
