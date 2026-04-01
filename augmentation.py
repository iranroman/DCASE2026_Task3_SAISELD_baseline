import torch
import numpy as np
import random
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _recompute_boxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    """
    Recompute tight axis-aligned bounding boxes from binary masks.

    masks : [N, H, W] bool
    returns [N, 4] float32  (x0, y0, x1, y1)

    x1 and y1 are *exclusive* (standard torchvision convention).
    """
    N     = masks.shape[0]
    boxes = torch.zeros(N, 4, dtype=torch.float32)
    for i in range(N):
        ys, xs = masks[i].nonzero(as_tuple=True)
        if xs.numel() == 0:
            continue

        x0 = xs.min().float()
        x1 = xs.max().float() + 1.0   
        y0 = ys.min().float()
        y1 = ys.max().float() + 1.0   

        boxes[i] = torch.tensor([x0, y0, x1, y1], dtype=torch.float32)
    return boxes


def _filter_empty_instances(target: dict) -> dict:
    masks = target["masks"]
    if masks.shape[0] == 0:
        return target

    valid = masks.flatten(1).any(dim=1)   # [N] bool
    if valid.all():
        return target

    out = dict(target)   
    per_instance_keys = (
        "boxes", "labels", "masks",
        "energy_maps", "energy_masks",
        "distances", "instance_ids",
    )
    for key in per_instance_keys:
        if key in out and isinstance(out[key], torch.Tensor) \
                and out[key].shape[0] == valid.shape[0]:
            out[key] = out[key][valid]
    return out


def _split_boundary_instances(target: dict, shift: int, img_w: int) -> dict:
    if shift == 0:
        return target

    masks = target["masks"]   # [N, H, W]
    N     = masks.shape[0]
    if N == 0:
        return target

    # Recompute preliminary boxes to check physical width
    temp_boxes = _recompute_boxes_from_masks(masks)
    widths     = temp_boxes[:, 2] - temp_boxes[:, 0]

    # An object is only split if it crosses the shift point AND its bounding 
    # box is physically massive (> W/2), proving it wrapped around the image edges.
    has_left  = masks[:, :, :shift].flatten(1).any(dim=1)   
    has_right = masks[:, :, shift:].flatten(1).any(dim=1)   
    crosses   = has_left & has_right & (widths > img_w / 2.0)

    if not crosses.any():
        out          = dict(target)
        out["boxes"] = temp_boxes
        return out

    scalar_keys  = ("labels", "distances", "instance_ids")
    spatial_keys = ("energy_maps", "energy_masks")

    idx_rows:    list[int]          = []          
    new_masks:   list[torch.Tensor] = []
    new_spatial: dict[str, list]    = {k: [] for k in spatial_keys if k in target}

    for i in range(N):
        if not crosses[i]:
            idx_rows.append(i)
            new_masks.append(masks[i].unsqueeze(0))
            for k in new_spatial:
                new_spatial[k].append(target[k][i].unsqueeze(0))
        else:
            for is_left in (True, False):
                frag = masks[i].clone()
                if is_left:
                    frag[:, shift:] = False    
                else:
                    frag[:, :shift] = False    

                idx_rows.append(i)
                new_masks.append(frag.unsqueeze(0))

                for k in new_spatial:
                    sp = target[k][i].clone()
                    is_bool_field = sp.dtype == torch.bool
                    if is_left:
                        sp[:, shift:] = False if is_bool_field else 0.0
                    else:
                        sp[:, :shift] = False if is_bool_field else 0.0
                    new_spatial[k].append(sp.unsqueeze(0))

    out        = dict(target)
    idx_tensor = torch.tensor(idx_rows, dtype=torch.long)

    for key in scalar_keys:
        if key in out and isinstance(out[key], torch.Tensor) and out[key].shape[0] == N:
            out[key] = out[key][idx_tensor]

    out["masks"] = torch.cat(new_masks, dim=0)
    for k, parts in new_spatial.items():
        out[k] = torch.cat(parts, dim=0)

    out["boxes"] = _recompute_boxes_from_masks(out["masks"])

    return out


def _roll_target_spatial(target: dict, shift: int, dim: int = -1) -> dict:
    out             = dict(target)
    per_image_2d    = ("vmap", "vmask")
    per_instance_3d = ("masks", "energy_maps", "energy_masks")

    for key in per_image_2d:
        if key in out:
            out[key] = torch.roll(out[key], shift, dims=dim)

    for key in per_instance_3d:
        if key in out and out[key].shape[0] > 0:
            out[key] = torch.roll(out[key], shift, dims=dim)

    return out


def _flip_target_spatial(target: dict, img_w: int) -> dict:
    out             = dict(target)
    per_image_2d    = ("vmap", "vmask")
    per_instance_3d = ("masks", "energy_maps", "energy_masks")

    for key in per_image_2d:
        if key in out:
            out[key] = torch.flip(out[key], dims=[-1])

    for key in per_instance_3d:
        if key in out and out[key].shape[0] > 0:
            out[key] = torch.flip(out[key], dims=[-1])

    if out["boxes"].shape[0] > 0:
        boxes = out["boxes"].clone()

        x0_new = img_w - boxes[:, 2]   
        x1_new = img_w - boxes[:, 0]   

        boxes[:, 0] = x0_new
        boxes[:, 2] = x1_new

        zero_w = boxes[:, 0] >= boxes[:, 2]
        if zero_w.any():
            boxes[zero_w, 2] = boxes[zero_w, 0] + 1.0

        out["boxes"] = boxes

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main augmentor
# ─────────────────────────────────────────────────────────────────────────────

class SeldAugmentor:
    def __init__(
        self,
        img_w: int                   = 360,
        img_h: int                   = 180,
        n_acoustic: int              = 9,
        azimuth_rotate: bool         = True,
        hflip_prob: float            = 0.5,
        max_bands_masked: int        = 2,
        intensity_scale_range: tuple = (0.6, 1.4),
        acoustic_noise_std: float    = 0.015,
        rgb_jitter_prob: float       = 0.8,
    ):
        self.img_w                 = img_w
        self.img_h                 = img_h
        self.n_acoustic            = n_acoustic
        self.azimuth_rotate        = azimuth_rotate
        self.hflip_prob            = hflip_prob
        self.max_bands_masked      = max_bands_masked
        self.intensity_scale_range = intensity_scale_range
        self.acoustic_noise_std    = acoustic_noise_std
        self.rgb_jitter_prob       = rgb_jitter_prob

    def __call__(self, image: torch.Tensor, target: dict) -> tuple[torch.Tensor, dict]:
        if self.azimuth_rotate:
            image, target = self._azimuth_rotate(image, target)

        if torch.rand(1).item() < self.hflip_prob:
            image, target = self._hflip(image, target)

        if torch.rand(1).item() < self.rgb_jitter_prob:
            image = self._rgb_jitter(image)

        image = self._band_mask(image)
        image = self._acoustic_intensity_scale(image)
        image = self._acoustic_noise(image)

        return image, target

    def _azimuth_rotate(self, image: torch.Tensor, target: dict) -> tuple[torch.Tensor, dict]:
        shift = int(torch.randint(0, self.img_w, (1,)).item())
        if shift == 0:
            return image, target

        image  = torch.roll(image,  shift, dims=-1)
        target = _roll_target_spatial(target, shift)
        target = _split_boundary_instances(target, shift, self.img_w)
        target = _filter_empty_instances(target)

        return image, target

    def _hflip(self, image: torch.Tensor, target: dict) -> tuple[torch.Tensor, dict]:
        image  = torch.flip(image, dims=[-1])
        target = _flip_target_spatial(target, self.img_w)
        return image, target

    # ------------------------------------------------------------------
    # Acoustic band masking  (SpecAugment-style)
    # ------------------------------------------------------------------

    def _band_mask(self, image: torch.Tensor) -> torch.Tensor:
        n_mask = int(torch.randint(0, self.max_bands_masked + 1, (1,)).item())
        if n_mask == 0 or self.n_acoustic == 0:
            return image

        RGB_OFFSET   = 3 # Explicitly enforce the [RGB, Acoustic] tensor contract
        band_indices = torch.randperm(self.n_acoustic)[:n_mask]
        image        = image.clone()
        for b in band_indices:
            image[RGB_OFFSET + int(b)] = 0.0
        return image

    # ------------------------------------------------------------------
    # Acoustic intensity scaling
    # ------------------------------------------------------------------

    def _acoustic_intensity_scale(self, image: torch.Tensor) -> torch.Tensor:
        if self.n_acoustic == 0:
            return image

        RGB_OFFSET = 3 # Explicit layout
        lo, hi     = self.intensity_scale_range
        scale      = lo + (hi - lo) * torch.rand(1).item()
        image      = image.clone()
        image[RGB_OFFSET:] = (image[RGB_OFFSET:] * scale).clamp(0.0, 1.0)
        return image

    # ------------------------------------------------------------------
    # Acoustic Gaussian noise
    # ------------------------------------------------------------------

    def _acoustic_noise(self, image: torch.Tensor) -> torch.Tensor:
        if self.n_acoustic == 0 or self.acoustic_noise_std <= 0.0:
            return image

        RGB_OFFSET = 3 # Explicit layout
        image      = image.clone()
        noise      = torch.randn_like(image[RGB_OFFSET:]) * self.acoustic_noise_std
        image[RGB_OFFSET:] = (image[RGB_OFFSET:] + noise).clamp(0.0, 1.0)
        return image

    def _rgb_jitter(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[0] < 3:
            return image
            
        image = image.clone()
        rgb   = image[:3]
        
        rgb = TF.adjust_brightness(rgb, random.uniform(0.7, 1.3))
        rgb = TF.adjust_contrast(rgb,   random.uniform(0.7, 1.3))
        rgb = TF.adjust_saturation(rgb, random.uniform(0.8, 1.2))
        
        # Explicit clamp to prevent blown-out float pixels from leaking
        image[:3] = rgb.clamp(0.0, 1.0)
        
        return image
