import argparse
import collections.abc as abc
import importlib
import json
import math
import os

import astropy.coordinates as coord
import astropy.units as u
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import mpl_toolkits.basemap as basemap
import numpy as np
import scipy.constants as constants
import scipy.linalg as linalg
import skimage.util as skutil
import torch
from scipy.signal import windows
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# CONSTANTS
# =============================================================================

_EIGENMIKE_ = {
    "1":  [69, 0,   0.042], "2":  [90, 32,  0.042], "3":  [111, 0,   0.042], "4":  [90, 328, 0.042],
    "5":  [32, 0,   0.042], "6":  [55, 45,  0.042], "7":  [90, 69,  0.042], "8":  [125, 45,  0.042],
    "9":  [148, 0,  0.042], "10": [125, 315, 0.042], "11": [90, 291, 0.042], "12": [55, 315, 0.042],
    "13": [21, 91,  0.042], "14": [58, 90,  0.042], "15": [121, 90,  0.042], "16": [159, 89,  0.042],
    "17": [69, 180, 0.042], "18": [90, 212, 0.042], "19": [111, 180, 0.042], "20": [90, 148, 0.042],
    "21": [32, 180, 0.042], "22": [55, 225, 0.042], "23": [90, 249, 0.042], "24": [125, 225, 0.042],
    "25": [148, 180, 0.042], "26": [125, 135, 0.042], "27": [90, 111, 0.042], "28": [55, 135, 0.042],
    "29": [21, 269, 0.042], "30": [58, 270, 0.042], "31": [122, 270, 0.042], "32": [159, 271, 0.042],
}

# Microphone subset used by UpLAM (4-channel sparse array)
_UPLAM_MIC_INDICES_ = [5, 9, 25, 21]  # 0-based indices into eigenmike channels

# =============================================================================
# UTILITIES & MATH
# =============================================================================

def initialize_config(module_cfg, pass_args=True):
    main_name = module_cfg["main"]
    # Prefer a locally-defined class (handles configs that point to repo modules
    # like "model.LAM" or "dataset.inference_dataloader" — those are equivalent
    # when running this self-contained script).
    func = globals().get(main_name)
    if func is None:
        # Fall back to the module path specified in the config
        try:
            func = getattr(importlib.import_module(module_cfg["module"]), main_name)
        except (ImportError, AttributeError) as e:
            raise RuntimeError(
                f"Could not resolve '{main_name}' from globals or module "
                f"'{module_cfg['module']}': {e}"
            )
    return func(**module_cfg.get("args", {})) if pass_args else func


def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(checkpoint_path)
    assert ext in (".pth", ".tar"), "Only .pth and .tar checkpoints are supported."
    ckpt = torch.load(checkpoint_path, map_location=device)
    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return ckpt
    print(f"Loading {checkpoint_path}, epoch = {ckpt['epoch']}.")
    return ckpt["model"]


def is_scalar(x):
    return not isinstance(x, abc.Container)


def _polar2cart(coords_dict, units=None):
    if units not in ("degrees", "radians"):
        raise ValueError("units must be 'degrees' or 'radians'")
    coords = {
        m: [math.radians(c[0]), math.radians(c[1]), c[2]] if units == "degrees" else list(c)
        for m, c in coords_dict.items()
    }
    return {
        m: [
            c[2] * math.sin(c[0]) * math.cos(c[1]),
            c[2] * math.sin(c[0]) * math.sin(c[1]),
            c[2] * math.cos(c[0]),
        ]
        for m, c in coords.items()
    }


def get_xyz():
    mic_coords = _polar2cart(_EIGENMIKE_, units="degrees")
    return [list(c) for c in mic_coords.values()]


# --- Coordinate conversions ---

def eq2cart(r, lat, lon):
    r = np.atleast_1d(np.array(r, copy=False) if not is_scalar(r) else np.array([r]))
    if np.any(r < 0):
        raise ValueError("r must be non-negative.")
    return (
        coord.SphericalRepresentation(lon * u.rad, lat * u.rad, r)
        .to_cartesian()
        .xyz.to_value(u.dimensionless_unscaled)
    )


def pol2cart(r, colat, lon):
    return eq2cart(r, (np.pi / 2) - colat, lon)


def cart2pol(x, y, z):
    sph = coord.SphericalRepresentation.from_cartesian(coord.CartesianRepresentation(x, y, z))
    r = sph.distance.to_value(u.dimensionless_unscaled)
    colat = u.Quantity(90 * u.deg - sph.lat).to_value(u.rad)
    lon = u.Quantity(sph.lon).to_value(u.rad)
    return r, colat, lon


def cart2eq(x, y, z):
    r, colat, lon = cart2pol(x, y, z)
    return r, (np.pi / 2) - colat, lon


def wrapped_rad2deg(lat_r, lon_r):
    lat_d = coord.Angle(lat_r * u.rad).to_value(u.deg)
    lon_d = coord.Angle(lon_r * u.rad).wrap_at(180 * u.deg).to_value(u.deg)
    return lat_d, lon_d


# --- Spatial sampling & steering ---

def fibonacci(N, direction=None, FoV=None, shift_lon=0, shift_colat=0):
    if direction is not None:
        direction = np.array(direction, dtype=float)
        direction /= linalg.norm(direction)
        if FoV is None or not (0 < np.rad2deg(FoV) < 360):
            raise ValueError("FoV must be in (0, 360) degrees when direction is given.")

    if N < 0:
        raise ValueError("N must be non-negative.")

    N_px = 4 * (N + 1) ** 2
    n = np.arange(N_px)
    colat = np.arccos(1 - (2 * n + 1) / N_px)
    lon = (4 * np.pi * n) / (1 + np.sqrt(5)) + shift_lon
    XYZ = np.stack(pol2cart(1, colat, lon), axis=0)

    if direction is not None:
        XYZ = XYZ[:, (direction @ XYZ) >= np.cos(FoV / 2)]

    return XYZ


def get_field(shift_lon=0, shift_colat=0):
    R = fibonacci(10, shift_lon=shift_lon, shift_colat=shift_colat)
    return R[:, np.abs(R[2, :]) < np.sin(np.deg2rad(90))]


def steering_operator(XYZ=None, R=None):
    if XYZ is None:
        XYZ = np.array(get_xyz()).T
    if R is None:
        R = get_field()
    wl = constants.speed_of_sound / (
        skutil.view_as_windows(np.linspace(1500, 4500, 10), (2,), 1).mean(axis=-1).max() + 500
    )
    return np.exp((-1j * 2 * np.pi / wl * XYZ.T) @ R)


# =============================================================================
# SIGNAL PROCESSING
# =============================================================================

def extract_visibilities(data, rate, T, fc, bw, alpha):
    N_stft = int(rate * T)
    if N_stft == 0:
        raise ValueError("Not enough samples per time frame.")
    N_ch = data.shape[1]
    N_sample = (data.shape[0] // N_stft) * N_stft
    stf_data = skutil.view_as_blocks(data[:N_sample], (N_stft, N_ch)).squeeze(axis=1)

    window = windows.tukey(M=N_stft, alpha=alpha, sym=True).reshape(1, -1, 1)
    stft = np.fft.fft(stf_data * window, axis=1)
    idx_start = int((fc - 0.5 * bw) * N_stft / rate)
    idx_end   = int((fc + 0.5 * bw) * N_stft / rate)
    spec = stft[:, idx_start:idx_end + 1, :].sum(axis=1)

    return spec.reshape(-1, N_ch, 1).conj() * spec.reshape(-1, 1, N_ch)


def form_visibility(data, rate, fc, bw, T_sti, T_stationarity):
    S_sti = extract_visibilities(data, rate, T_sti, fc, bw, alpha=1.0)
    N_ch = data.shape[1]
    N_blk = int(T_stationarity / T_sti)
    return (
        skutil.view_as_windows(S_sti, (N_blk, N_ch, N_ch), (N_blk, N_ch, N_ch))
        .squeeze(axis=(1, 2))
        .sum(axis=1)
    )


def get_visibility_matrix(audio_in, fs, T_sti=10e-3, scale="linear", nbands=9):
    if scale == "linear":
        freq = skutil.view_as_windows(np.linspace(1500, 4500, nbands), (2,), 1).mean(axis=-1)
    elif scale == "log":
        freq = librosa.mel_frequencies(n_mels=nbands, fmin=50, fmax=4500)
    else:
        raise ValueError("scale must be 'linear' or 'log'")
    bw = 50.0

    N_px = steering_operator(np.array(get_xyz()).T, get_field()).shape[1]
    visibilities = []

    for i in range(nbands - 1):
        S = form_visibility(audio_in, fs, freq[i], bw, T_sti, 10 * T_sti)
        frames = []
        for s in S:
            S_D, S_V = linalg.eigh(s)
            S_D = np.clip(S_D / S_D.max(), 0, None) if S_D.max() > 0 else np.zeros_like(S_D)
            frames.append((S_V * S_D) @ S_V.conj().T)
        visibilities.append(frames)

    n_frames = len(visibilities[0]) if visibilities else 0
    return np.array(visibilities), np.zeros((nbands - 1, n_frames, N_px))


# =============================================================================
# DATASET
# =============================================================================

class InferenceDataset(Dataset):
    def __init__(self, dataset, limit=None, offset=0, sample_length=16384):
        super().__init__()
        self.dataset_list = [
            os.path.join(dataset, f) for f in os.listdir(dataset) if f.endswith(".wav")
        ]
        self.length = len(self.dataset_list)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        path = self.dataset_list[item]
        name = os.path.splitext(os.path.basename(path))[0]
        audio, _ = librosa.load(os.path.abspath(os.path.expanduser(path)), sr=24000, mono=False)
        return audio, name


# =============================================================================
# MODEL — LAM
# =============================================================================

def _init_scaled_kaiming(layer, scale=1e-6):
    nn.init.kaiming_uniform_(layer.weight, a=0, mode="fan_in", nonlinearity="relu")
    layer.weight.data *= scale
    if layer.bias is not None:
        layer.bias.data.fill_(1e-6)


class LAM(nn.Module):
    def __init__(self, num_bands=16, Nch=32, tau=None, D=None):
        super().__init__()
        self.num_bands = num_bands
        self.A = torch.from_numpy(steering_operator())
        self.A.requires_grad = False
        Npx = self.A.shape[-1]

        if tau is None or D is None:
            self.tau = nn.Parameter(torch.empty((num_bands, Npx), dtype=torch.float64))
            self.D   = nn.Parameter(torch.empty((num_bands, Nch, Npx), dtype=torch.complex128))
            self.tau.data.normal_(0, 1e-7)
            self.D.data.normal_(0, 1e-5)
        else:
            self.tau = nn.Parameter(tau)
            self.D   = nn.Parameter(D)

        self.retanh = nn.ReLU()
        conv_kwargs = dict(dtype=torch.float64, padding="same")
        self.denoise1 = nn.Conv1d(num_bands, num_bands, kernel_size=3, **conv_kwargs)
        self.denoise2 = nn.Conv1d(num_bands, num_bands, kernel_size=5, **conv_kwargs)
        self.denoise3 = nn.Conv1d(num_bands, num_bands, kernel_size=7, **conv_kwargs)
        self.denoise4 = nn.Conv1d(num_bands, num_bands, kernel_size=9, **conv_kwargs)
        for layer in (self.denoise1, self.denoise2, self.denoise3, self.denoise4):
            _init_scaled_kaiming(layer)

    def forward(self, S):
        self.A = self.A.to(S.device)
        batch_size, freq_bands = S.shape[:2]

        latent_list = []
        for i in range(freq_bands):
            Ds, Vs = torch.linalg.eigh(S[:, i])
            Vs = Vs * torch.sqrt(torch.where(Ds > 0, Ds, torch.zeros_like(Ds))).unsqueeze(1)
            x = torch.linalg.norm(torch.matmul(self.D[i].conj().T, Vs), dim=2) ** 2 - self.tau[i]
            latent_list.append(x)

        latent_x = torch.stack(latent_list, dim=1)
        skip = latent_x.clone()

        for denoise in (self.denoise1, self.denoise2, self.denoise3, self.denoise4):
            latent_x = self.retanh(denoise(latent_x) + skip)

        A = self.A.unsqueeze(0)
        out = torch.stack([
            torch.einsum("nij,bjk,nkl->bil",
                         A,
                         torch.diag_embed(latent_x[:, i].cdouble()),
                         A.transpose(1, 2).conj())
            for i in range(latent_x.shape[1])
        ], dim=1)

        return out, latent_x


# =============================================================================
# MODEL — CDBPN (Complex Deep Back-Projection Network)
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1,
                 bias=True, activation='prelu', norm=None):
        super().__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding,
                              bias=bias, dtype=torch.double)
        self.norm = norm
        if norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if activation == 'relu':
            self.act = nn.ReLU(True)
        elif activation == 'prelu':
            self.act = nn.PReLU(dtype=torch.double)
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.bn(self.conv(x)) if self.norm is not None else self.conv(x)
        return self.act(out) if self.activation is not None else out


class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1,
                 bias=True, activation='prelu', norm=None):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding,
                                         bias=bias, dtype=torch.double)
        self.norm = norm
        if norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if activation == 'relu':
            self.act = nn.ReLU(True)
        elif activation == 'prelu':
            self.act = nn.PReLU(dtype=torch.double)
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.bn(self.deconv(x)) if self.norm is not None else self.deconv(x)
        return self.act(out) if self.activation is not None else out


class UpBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
                 bias=True, activation='prelu', norm=None):
        super().__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
                 bias=True, activation='prelu', norm=None):
        super().__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class CDBPN(nn.Module):
    """Complex Deep Back-Projection Network for visibility matrix upsampling."""

    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
        super().__init__()

        if scale_factor == 2:
            kernel, stride, padding = 6, 2, 2
        elif scale_factor == 4:
            kernel, stride, padding = 8, 4, 2
        elif scale_factor == 8:
            kernel, stride, padding = 12, 8, 2
        else:
            raise ValueError(f"Unsupported scale_factor: {scale_factor}. Choose from 2, 4, 8.")

        # Initial feature extraction (real & imaginary paths)
        self.feat0_rel  = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat0_imag = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1_rel  = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        self.feat1_imag = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)

        # Back-projection stages
        self.up1_rel    = UpBlock(base_filter, kernel, stride, padding)
        self.up1_imag   = UpBlock(base_filter, kernel, stride, padding)
        self.down1_rel  = DownBlock(base_filter, kernel, stride, padding)
        self.down1_imag = DownBlock(base_filter, kernel, stride, padding)
        self.up2_rel    = UpBlock(base_filter, kernel, stride, padding)
        self.up2_imag   = UpBlock(base_filter, kernel, stride, padding)

        # Reconstruction
        self.output_conv_rel  = ConvBlock(num_stages * base_filter // 5, num_channels,
                                          3, 1, 1, activation=None, norm=None)
        self.output_conv_imag = ConvBlock(num_stages * base_filter // 5, num_channels,
                                          3, 1, 1, activation=None, norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.kaiming_normal_(m.weight).double()
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.data = m.bias.data.double()
            elif classname.find('ConvTranspose2d') != -1:
                nn.init.kaiming_normal_(m.weight).double()
                if m.bias is not None:
                    m.bias.data = m.bias.data.double()

    def forward(self, x_rel, x_imag):
        x_rel  = self.feat1_rel(self.feat0_rel(x_rel))
        x_imag = self.feat1_imag(self.feat0_imag(x_imag))

        h1_rel  = self.up1_rel(x_rel)
        h2_rel  = self.up2_rel(self.down1_rel(h1_rel))
        h1_imag = self.up1_imag(x_imag)
        h2_imag = self.up2_imag(self.down1_imag(h1_imag))

        x_rel  = self.output_conv_rel(torch.cat((h2_rel,  h1_rel),  dim=1))
        x_imag = self.output_conv_imag(torch.cat((h2_imag, h1_imag), dim=1))
        return torch.complex(x_rel, x_imag)


# =============================================================================
# MODEL — UpLAM
# =============================================================================

class UpLAM(nn.Module):
    """
    Upsampling LAM: a sparse 4-channel visibility matrix is first upsampled
    to a full 32-channel one via CDBPN, then processed by LAM.
    """

    def __init__(self, num_bands=16, base_filter=32, feat=128,
                 num_stages=10, scale_factor=8):
        super().__init__()
        self.cdbpn = CDBPN(num_bands, base_filter, feat, num_stages, scale_factor)
        self.lam   = LAM(num_bands)

    def forward(self, S):
        S_pred      = self.cdbpn(S.real, S.imag)   # upsample: (B, bands, 4, 4) → (B, bands, 32, 32)
        out, latent = self.lam(S_pred)              # localize
        return out, latent


# =============================================================================
# VISUALIZATION
# =============================================================================

def cmap_from_list(name, colors, N=256, gamma=1.0):
    if not isinstance(colors, abc.Iterable):
        raise ValueError("colors must be iterable")

    if isinstance(colors[0], abc.Sized) and len(colors[0]) == 2 and not isinstance(colors[0], str):
        vals, colors = zip(*colors)
    else:
        vals = np.linspace(0, 1, len(colors))

    cdict = {k: [] for k in ("red", "green", "blue", "alpha")}
    for val, color in zip(vals, colors):
        r, g, b, a = mcolors.to_rgba(color)
        for ch, v in zip(("red", "green", "blue", "alpha"), (r, g, b, a)):
            cdict[ch].append((val, v, v))

    return mcolors.LinearSegmentedColormap(name, cdict, N, gamma)


def draw_map(I, R, lon_ticks, catalog=None, show_labels=False, show_axis=False,
             fig=None, ax=None, kmeans=False, gaussian_mixture=False):
    _, R_el, R_az = cart2eq(*R)
    R_el, R_az = wrapped_rad2deg(R_el, R_az)
    R_el_min, R_el_max = np.around([R_el.min(), R_el.max()])
    R_az_min, R_az_max = np.around([R_az.min(), R_az.max()])

    if ax is None:
        fig, ax = plt.subplots()

    bm = basemap.Basemap(
        projection="mill",
        llcrnrlat=R_el_min, urcrnrlat=R_el_max,
        llcrnrlon=R_az_min, urcrnrlon=R_az_max,
        resolution="c", ax=ax,
    )
    bm_labels = [1, 0, 0, 1] if show_axis else [0, 0, 0, 0]
    bm.drawparallels(np.linspace(R_el_min, R_el_max, 5),
                     color="w", dashes=[1, 0], labels=bm_labels,
                     labelstyle="+/-", textcolor="#565656", zorder=0, linewidth=2)
    bm.drawmeridians(lon_ticks,
                     color="w", dashes=[1, 0], labels=bm_labels,
                     labelstyle="+/-", textcolor="#565656", zorder=0, linewidth=2)

    if show_labels:
        ax.set_xlabel("Azimuth (degrees)", labelpad=20)
        ax.set_ylabel("Elevation (degrees)", labelpad=40)

    R_x, R_y = bm(R_az, R_el)
    triangulation = tri.Triangulation(R_x, R_y)
    N_px = I.shape[1]
    mycmap = cmap_from_list("mycmap", I.T, N=N_px)
    ax.tripcolor(triangulation, np.arange(N_px), cmap=mycmap,
                 shading="gouraud", alpha=0.9, edgecolors="w", linewidth=0.1)

    cluster_center = None
    if kmeans:
        Npts = 18
        max_idx = np.square(I).sum(axis=0).argsort()[-Npts:][::-1]
        x_y = np.column_stack((R_x[max_idx], R_y[max_idx]))
        clusters = KMeans(n_clusters=3).fit(x_y).cluster_centers_
        ax.scatter(R_x[max_idx], R_y[max_idx], c="b", s=5)
        ax.scatter(clusters[:, 0], clusters[:, 1], s=500, alpha=0.3)
        cluster_center = bm(clusters[0, 0], clusters[0, 1], inverse=True)

    return fig, ax, cluster_center


# =============================================================================
# INFERENCE
# =============================================================================

# Default configs for each model variant
_DEFAULT_CONFIG_LAM_ = {
    "model":      {"module": "__main__", "main": "LAM", "args": {"num_bands": 9}},
    "dataset":    {"module": "__main__", "main": "InferenceDataset",
                   "args": {"dataset": "/path/to/eigenmike/wavs"}},
    "model_path": "checkpoints/LAM.pth",
    "output_dir": "output_LAM",
    "FS":         24000,
    "n_max":      20,
    "T_sti_ms":   100,
    "alpha_ema":  0.65,
}

_DEFAULT_CONFIG_UPLAM_ = {
    "model":      {"module": "__main__", "main": "UpLAM",
                   "args": {"num_bands": 9, "base_filter": 32, "feat": 128,
                            "num_stages": 10, "scale_factor": 8}},
    "dataset":    {"module": "__main__", "main": "InferenceDataset",
                   "args": {"dataset": "/Users/iranroman/Downloads/tau"}},
    "model_path": "checkpoints/UpLAM.pth",
    "output_dir": "output_UpLAM",
    "FS":         24000,
    "n_max":      20,
    "T_sti_ms":   100,
    "alpha_ema":  0.65,
}


def main():
    parser = argparse.ArgumentParser(
        description="LAM / UpLAM: acoustic map visualisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-C", "--config",     type=str,   default=None,
                        help="Path to a JSON config file (overrides --model defaults).")
    parser.add_argument("-D", "--device",     type=str,   default="0",
                        help="CUDA device index, or 'cpu'.")
    parser.add_argument("-M", "--model",      type=str,   default="lam",
                        choices=["lam", "uplam"],
                        help="Which model variant to run: 'lam' (full 32-ch) or 'uplam' (sparse 4-ch).")
    parser.add_argument("-A", "--alpha-ema",  type=float, default=None,
                        help="EMA smoothing factor for normalization (0–1). Overrides config.")
    args = parser.parse_args()

    # ------------------------------------------------------------------ config
    if args.config and os.path.exists(args.config):
        config = json.load(open(args.config))
        print(f"Loaded config from {args.config}.")
    else:
        if args.config:
            print(f"Config file '{args.config}' not found — using built-in defaults.")
        else:
            print("No config provided — using built-in defaults.")
        config = _DEFAULT_CONFIG_UPLAM_ if args.model == "uplam" else _DEFAULT_CONFIG_LAM_

    # Determine which model class we're running (from config or CLI flag)
    model_main = config["model"].get("main", "LAM")
    use_uplam  = (model_main == "UpLAM") or (args.model == "uplam")
    if use_uplam and model_main != "UpLAM":
        # CLI flag wins: patch the config to UpLAM
        config = dict(_DEFAULT_CONFIG_UPLAM_, **{k: v for k, v in config.items()
                                                  if k not in ("model",)})
        print("  [--model uplam] overriding model config to UpLAM defaults.")

    print(f"Running model variant: {'UpLAM' if use_uplam else 'LAM'}")

    # ------------------------------------------------------------------ paths
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    ckpt_path  = config["model_path"]
    if not os.path.exists(ckpt_path):
        print(f"Warning: checkpoint not found at '{ckpt_path}'. Proceeding with random weights.")

    # ------------------------------------------------------------------ device
    device = (
        torch.device("cpu")
        if args.device.lower() == "cpu"
        else torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # --------------------------------------------------- EMA alpha resolution
    # Default 0.65 matches the reference fallback config; 0.1 was too conservative.
    alpha_ema = args.alpha_ema if args.alpha_ema is not None else config.get("alpha_ema", 0.65)
    print(f"EMA alpha: {alpha_ema}")

    # ----------------------------------------------------------------- dataset
    dataset    = initialize_config(config["dataset"])
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0)

    # ------------------------------------------------------------------ model
    model = initialize_config(config["model"])
    if os.path.exists(ckpt_path):
        model.load_state_dict(load_checkpoint(ckpt_path, device))
    model.to(device).eval()

    # Derive nbands so it always stays in sync with the model's num_bands.
    # get_visibility_matrix produces (nbands-1) bands, so nbands = num_bands + 1.
    model_num_bands = config["model"].get("args", {}).get("num_bands", 9)
    nbands = model_num_bands + 1
    print(f"Band alignment: model num_bands={model_num_bands}, "
          f"get_visibility_matrix nbands={nbands} → {nbands - 1} output bands")

    R_field   = get_field()
    lon_ticks = np.linspace(-180, 180, 5)
    T_sti_ms  = config.get("T_sti_ms", 10)

    # --------------------------------------------------------------- inference
    with torch.no_grad():
        for audio, name in dataloader:
            name  = name[0]
            audio = audio.cpu().numpy()[0].T   # (samples, channels)

            # uses channels as-is if already sparse
            if use_uplam:
                if audio.shape[1] > len(_UPLAM_MIC_INDICES_):
                    audio = audio[:, _UPLAM_MIC_INDICES_]
                    print(f"{name}: selected sparse subset {_UPLAM_MIC_INDICES_} → shape {audio.shape}")
                else:
                    print(f"{name}: audio already has {audio.shape[1]} channels, using as-is")

            S_in, _ = get_visibility_matrix(audio, fs=config["FS"], nbands=nbands)
            print(f"{name}: S_in shape = {S_in.shape}  (bands, frames, N_ch, N_ch)")

            S_in = torch.from_numpy(S_in).to(device).permute(1, 0, 2, 3)
            _, I_pred = model(S_in)

            I_pred_np = I_pred.cpu().numpy()
            n_frames, n_bands, N_px = I_pred_np.shape
            print(f"{name}: {n_frames} frames × {n_bands} bands × {N_px} pixels")

            clip_dir = os.path.join(output_dir, name)
            os.makedirs(clip_dir, exist_ok=True)

            running_max = np.zeros(n_bands)

            for i, frame_bands in enumerate(I_pred_np):
                t_ms = i * T_sti_ms

                for b, band in enumerate(frame_bands):
                    current_max = band.max()

                    if i == 0:
                        running_max[b] = current_max
                    else:
                        running_max[b] = alpha_ema * current_max + (1 - alpha_ema) * running_max[b]

                    norm_val       = running_max[b] if running_max[b] > 1e-10 else 1.0
                    band_normalized = np.clip(band / norm_val, 0, 1)
                    band_rgb        = np.tile(band_normalized[np.newaxis], (3, 1))

                    band_dir = os.path.join(clip_dir, "bands", f"band{b:02d}")
                    os.makedirs(band_dir, exist_ok=True)

                    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    draw_map(band_rgb, R_field, lon_ticks,
                             show_labels=True, show_axis=True, fig=fig, ax=ax)
                    ax.set_title(f"{name}  —  t = {t_ms} ms  |  band {b}")
                    fig.savefig(
                        os.path.join(band_dir, f"frame_{i:04d}_{t_ms:06d}ms_band{b:02d}.png"),
                        bbox_inches="tight", dpi=100,
                    )
                    plt.close(fig)

            print(f"  → saved {n_frames} frames × {n_bands} bands to '{clip_dir}'")


if __name__ == "__main__":
    main()