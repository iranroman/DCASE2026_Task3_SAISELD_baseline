import os
import numpy as np
import torch
import soundfile as sf
from scipy.spatial import cKDTree

# ── Import from companion LAM script (save Document-2 as lam_model.py) ────────
from lam_model import (
    UpLAM,
    get_field,
    cart2eq,
    wrapped_rad2deg,
    get_visibility_matrix,
    _UPLAM_MIC_INDICES_,
    load_checkpoint,
)

# ── Constants (must stay in sync with main training script) ───────────────────
IMG_W              = 360
IMG_H              = 180
NUM_ACOUSTIC_BANDS = 9       # UpLAM num_bands=9 → 9 output spatial maps
FS                 = 24000   # expected sample rate of all WAV files

# ── Derived timing constants ──────────────────────────────────────────────────
# These mirror the defaults inside get_visibility_matrix / form_visibility.
# If you ever change T_sti there, change it here too.
T_STI_S            = 10e-3                              # 10 ms per STI frame
N_STFT             = int(FS * T_STI_S)              # 240 samples per STI frame
N_BLK              = 10                             # STI frames per visibility frame
SAMPLES_PER_FRAME  = N_BLK * N_STFT                # 2400 samples per video/vis frame

# ── Pre-compute sphere → equirectangular nearest-neighbour mapping ─────────────
# Done once at import time; applying it later is a single fancy-index op.
_R_field               = get_field()                      # (3, N_px)
_, _R_el_rad, _R_az_rad = cart2eq(*_R_field)
_R_el_deg, _R_az_deg   = wrapped_rad2deg(_R_el_rad, _R_az_rad)

_sphere_pts = np.column_stack([_R_az_deg, _R_el_deg])    # (N_px, 2)

# Equirectangular grid:  az -180→+180 left→right,  el +90→-90 top→bottom
_az_lin = np.linspace(-180,  180, IMG_W)
_el_lin = np.linspace(  90,  -90, IMG_H)
_AZ, _EL = np.meshgrid(_az_lin, _el_lin)
_grid_pts  = np.column_stack([_AZ.ravel(), _EL.ravel()])  # (IMG_H*IMG_W, 2)

_kd_tree   = cKDTree(_sphere_pts)
_, _NN_IDX = _kd_tree.query(_grid_pts)   # (IMG_H*IMG_W,) — computed once
# ──────────────────────────────────────────────────────────────────────────────


def _latent_to_equirect_single(latent_np: np.ndarray) -> np.ndarray:
    """
    Project a single UpLAM frame's spatial activations to an equirectangular image.

    Parameters
    ----------
    latent_np : np.ndarray, shape (n_bands, N_px)

    Returns
    -------
    np.ndarray, shape (n_bands, IMG_H, IMG_W), float32, values in [0, 1]

    The horizontal flip corrects the mirror between the acoustic field
    and the camera's azimuth convention.
    """
    n_bands, N_px = latent_np.shape
    out = np.zeros((n_bands, IMG_H, IMG_W), dtype=np.float32)
    
    # 1. Scatter to 2D grid
    for b in range(n_bands):
        band = latent_np[b]                               # (N_px,)
        img  = band[_NN_IDX].reshape(IMG_H, IMG_W)       # scatter to 2-D
        out[b] = img[:, ::-1]                            # horizontal flip
            
    return out


# ── WAV path helper ────────────────────────────────────────────────────────────

def wav_path_from_seq_dir(seq_dir: str, frames_base: str, mic_base: str) -> str:
    """
    Derives the WAV path from the frame directory.

    Example
    -------
    seq_dir     = /STARSS23/frames_dev/dev-test-sony/fold4_room23_mix001
    frames_base = /STARSS23/frames_dev
    mic_base    = /STARSS23/mic_dev
    returns       /STARSS23/mic_dev/dev-test-sony/fold4_room23_mix001.wav
    """
    rel        = os.path.relpath(seq_dir, frames_base)  # dev-test-sony/fold4_room23_mix001
    split_name = os.path.dirname(rel)                    # dev-test-sony
    seq_name   = os.path.basename(rel)                   # fold4_room23_mix001
    return os.path.join(mic_base, split_name, seq_name + ".wav")


# ── Main extractor class ───────────────────────────────────────────────────────

class AcousticFeatureExtractor:
    """
    Lazily extracts 9-band equirectangular acoustic images one video frame at
    a time.  For each call to get_frame_bands():

      1. soundfile.read() seeks to the exact byte offset for that frame and
         reads SAMPLES_PER_FRAME (2400) samples — no full-file load.
      2. get_visibility_matrix() computes 1 visibility matrix from that slice.
      3. UpLAM runs on the single-frame batch.
      4. The result is projected to equirectangular and returned as a tensor.

    A small LRU result-cache keyed by (wav_path, frame_idx) avoids repeating
    the UpLAM call when a specific worker requests the same frame twice
    (e.g. repeated access via dataset repeats or intra-batch augmentations).
    
    NOTE: Because DataLoader workers operate in isolated memory spaces, 
    this cache is local to each worker, not shared across num_workers.

    Parameters
    ----------
    uplam_checkpoint : str
        Path to the UpLAM .pth file.
    device : torch.device
    num_bands : int
        Must match the UpLAM model's num_bands (default 9).
    result_cache_size : int
        How many (wav, frame) results to keep in RAM.  Each entry is
        ~9 * 180 * 360 * 4 bytes ≈ 2.3 MB, so 128 entries ≈ 300 MB per worker.
    """

    def __init__(
        self,
        uplam_checkpoint: str = "checkpoints/UpLAM.pth",
        device: torch.device  = torch.device("cpu"),
        num_bands: int        = NUM_ACOUSTIC_BANDS,
        result_cache_size: int = 128,
    ):
        self.device            = device
        self.num_bands         = num_bands
        self.result_cache_size = result_cache_size

        # LRU result cache:  (wav_path, frame_idx) -> torch.Tensor (9, H, W)
        from collections import OrderedDict
        self._result_cache: "OrderedDict" = OrderedDict()

        # Per-file metadata cache: wav_path -> total_frames (int)
        # Avoids re-opening the file just to check its length.
        self._file_meta: dict = {}

        # Build UpLAM
        self.model = UpLAM(num_bands=num_bands).to(device)
        if os.path.exists(uplam_checkpoint):
            state = load_checkpoint(uplam_checkpoint, device)
            self.model.load_state_dict(state)
            print(f"[Acoustic] UpLAM loaded from {uplam_checkpoint}")
        else:
            print(
                f"[Acoustic] WARNING: checkpoint not found at '{uplam_checkpoint}'. "
                "Acoustic channels will be noise."
            )
        self.model.eval()

    # ------------------------------------------------------------------
    def _total_frames(self, wav_path: str) -> int:
        """Return the number of video-aligned frames available in this WAV."""
        if wav_path not in self._file_meta:
            info = sf.info(wav_path)
            # soundfile reports sample rate; verify alignment
            if info.samplerate != FS:
                print(
                    f"[Acoustic] WARNING: {os.path.basename(wav_path)} is "
                    f"{info.samplerate} Hz, expected {FS} Hz. "
                    "Frame alignment may be off."
                )
            self._file_meta[wav_path] = info.frames // SAMPLES_PER_FRAME
        return self._file_meta[wav_path]

    # ------------------------------------------------------------------
    def _compute_one_frame(self, wav_path: str, frame_idx: int) -> torch.Tensor:
        """
        Core per-frame pipeline:
          read 2400-sample slice → visibility matrix → UpLAM → equirect tensor.
        """
        start = int(frame_idx) * SAMPLES_PER_FRAME

        # ── 1. Load the audio slice ───────────────────────────────────────────
        # sf.read with start= and frames= does a single seek + read — very fast.
        audio_slice, file_sr = sf.read(
            wav_path,
            start       = start,
            frames      = SAMPLES_PER_FRAME,
            always_2d   = True,
            dtype       = "float32",
            fill_value  = 0.0,   # handles zero-padding at EOF natively
        )
        # audio_slice: (SAMPLES_PER_FRAME, n_channels)

        # ── 2. Channel selection ──────────────────────────────────────────────
        if audio_slice.shape[1] > len(_UPLAM_MIC_INDICES_):
            audio_slice = audio_slice[:, _UPLAM_MIC_INDICES_]

        # ── 3. Visibility matrix ──────────────────────────────────────────────
        # get_visibility_matrix on SAMPLES_PER_FRAME samples produces exactly
        # 1 visibility frame (shape: n_bands, 1, 4, 4).
        S_np, _ = get_visibility_matrix(
            audio_slice, fs=FS, nbands=self.num_bands + 1
        )
        # S_np: (n_bands, n_vis_frames, N_ch, N_ch)

        if S_np.shape[1] == 0:
            # Degenerate slice (all zeros)
            return torch.zeros(self.num_bands, IMG_H, IMG_W, dtype=torch.float32)

        # ── 4. UpLAM ─────────────────────────────────────────────────────────
        # Permute to (n_vis_frames, n_bands, N_ch, N_ch) and take the first
        # (and only) frame as a batch of size 1.
        S_t = torch.from_numpy(S_np[:, :1, :, :]).to(self.device).permute(1, 0, 2, 3)
        # S_t: (1, n_bands, 4, 4)

        with torch.no_grad():
            _, latent = self.model(S_t)   # latent: (1, n_bands, N_px)

        latent_np = latent[0].cpu().float().numpy()   # (n_bands, N_px)

        # ── 5. Sphere → equirectangular ───────────────────────────────────────
        equirect = _latent_to_equirect_single(latent_np)   # (n_bands, H, W)
        return torch.from_numpy(equirect)

    # ------------------------------------------------------------------
    def get_frame_bands(self, wav_path: str, frame_idx: int) -> torch.Tensor:
        """
        Return a (9, IMG_H, IMG_W) float32 tensor for the given video frame.

        Handles missing files and out-of-range indices gracefully (zeros).
        Uses an LRU result cache so repeated requests are free.
        """
        cache_key = (wav_path, int(frame_idx))

        # ── LRU hit ───────────────────────────────────────────────────────────
        if cache_key in self._result_cache:
            tensor = self._result_cache.pop(cache_key)
            self._result_cache[cache_key] = tensor   # move to end (most recent)
            return tensor

        # ── Missing file ──────────────────────────────────────────────────────
        if not os.path.isfile(wav_path):
            # Warn once per path
            if wav_path not in self._file_meta:
                print(f"[Acoustic] WARNING: WAV not found -> zeros ({wav_path})")
                self._file_meta[wav_path] = 0
            return torch.zeros(self.num_bands, IMG_H, IMG_W, dtype=torch.float32)

        # ── Out-of-range frame ────────────────────────────────────────────────
        total = self._total_frames(wav_path)
        if total == 0 or int(frame_idx) >= total:
            return torch.zeros(self.num_bands, IMG_H, IMG_W, dtype=torch.float32)

        # ── Compute ───────────────────────────────────────────────────────────
        tensor = self._compute_one_frame(wav_path, frame_idx)

        # ── LRU insert ────────────────────────────────────────────────────────
        self._result_cache[cache_key] = tensor
        if len(self._result_cache) > self.result_cache_size:
            self._result_cache.popitem(last=False)   # evict oldest

        return tensor

    # ------------------------------------------------------------------
    def clear_cache(self):
        """Free result cache (call between train/val if RAM is tight)."""
        self._result_cache.clear()
