# DCASE 2026 Task 3 — Semantic Acoustic Imaging for SELD: Baseline System

Multimodal instance segmentation pipeline for the [DCASE 2026 Task 3](https://dcase.community/challenge2026/task-semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes) challenge. The system fuses RGB equirectangular frames with spatial audio to produce per-instance segmentation masks, acoustic energy fields, and source distance estimates on a 360×180 canvas at 10 FPS.

## How It Works

4-channel audio (tetrahedral mic array) is upsampled by [**UpLAM**](https://ccrma.stanford.edu/~iran/papers/Roman_et_al_WASPAA_2025.pdf) — a self-supervised complex-valued deep back-projection network — into a 9-band equirectangular acoustic feature map. These 9 channels are concatenated with the 3 RGB channels to form a 12-channel input tensor, which is processed by `EnergyInstanceModel` — a modified Mask R-CNN whose mask head is replaced by three sub-networks predicting:

1. **Energy masks** — 28×28 per-instance acoustic energy grids (sparsified to top-K peaks at export)
2. **Full-image energy maps** — global 360×180 energy fields
3. **Source distance** — scalar range regression from RoI-pooled features

A frame-level `InstanceTracker` links detections across time via IoU-constrained Hungarian assignment, applies score decay during coasting, and suppresses short-lived spurious tracks.

* **Audio-only mode** (Track A): RGB channels are zeroed; only acoustic features are used.
* **Audiovisual mode** (Track B): RGB and acoustic channels are used jointly.

---

## Repository Structure

```text
.
├── model.py                # EnergyInstanceModel, InstanceTracker, dataset utilities
├── acoustic_features.py    # AcousticFeatureExtractor (UpLAM wrapper)
├── train.py                # Training loop
├── run_inference.py        # Batch inference → submission JSON files
├── evaluate.py             # Evaluation: Mask mAP, Pearson r, RDE
├── UpLAM.pth               # Pre-trained UpLAM checkpoint (place here or set path)
└── experiments/            # Created automatically during training
    └── <run_name>/
        ├── energy_seg_best.pth
        ├── train.log
        ├── inference.log
        ├── eval_summary.txt
        └── inference_outputs/
            └── <seq_name>_inference.json
```

---

## Setup

### Requirements
- Python 3.10.19
- CUDA-capable GPU (baseline developed on A100-40GB; batch size defaults assume this)
- FFmpeg (for frame extraction)
- [STARSS23 dataset](https://zenodo.org/records/7880637) + [STAIRS26 labels](https://doi.org/10.5281/zenodo.18171005)

### Environment

```bash
conda create -n saiseld_env python=3.10.19 -y
conda activate saiseld_env

# Adjust cuda version to match your hardware
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install numpy scipy soundfile librosa scikit-image scikit-learn Pillow tqdm pycocotools
pip install astropy matplotlib basemap
```

### Hardcoded Paths

The following constants appear at the top of `train.py`, `run_inference.py`, and `evaluate.py` and **must be updated** to match your filesystem before running anything:

```python
FRAMES_BASE      = "/data3/scratch/eez086/STARSS23/frames_dev"
LABELS_BASE      = "/data3/scratch/eez086/STARSS23/labels_dev"
MIC_BASE         = "/data3/scratch/eez086/STARSS23/mic_dev"
UPLAM_CHECKPOINT = "UpLAM.pth"
```

---

## Data Preparation

### Expected Directory Layout

```text
STARSS23/
├── labels_dev/                 # STAIRS26 JSON annotation files
│   ├── train/
│   │   └── fold1_room1_mix001.json
│   └── test/
├── mic_dev/                    # 4-channel WAV files, 24 kHz
│   ├── train/
│   │   └── fold1_room1_mix001.wav
│   └── test/
└── frames_dev/                 # Extracted RGB frames (10 FPS, 360×180)
    ├── train/
    │   └── fold1_room1_mix001/
    │       ├── fold1_room1_mix001_0000.png
    │       ├── fold1_room1_mix001_0001.png
    │       └── ...
    └── test/
```

Frame filenames must follow the pattern `{sequence_name}_{frame_index:04d}.png` with **zero-based indexing** (frame 0 = `_0000.png`) to match `metadata_frame_index` values in the JSON annotations.

### Extracting Frames

Place raw STARSS23 videos under `STARSS23/video_dev/`, then from inside `STARSS23/` run:

```bash
find video_dev -type f -name "*.mp4" -exec bash -c '
  rel="${1#video_dev/}"
  base="$(basename "$rel" .mp4)"
  outdir="frames_dev/$(dirname "$rel")/$base"
  mkdir -p "$outdir"
  ffmpeg -hide_banner -loglevel error -i "$1" \
    -vf "fps=10,scale=360:180" \
    -start_number 0 \
    "$outdir/${base}_%04d.png"
' _ {} \;
```

*Note: The `-start_number 0` flag is critical. It ensures the first extracted frame is `_0000.png`. Without it, FFmpeg defaults to `_0001.png`, causing frame 0 to fall back to synthetic noise in the data loader.*

---

## Training

```bash
python train.py --exp_name my_run
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--exp_name` | **required** | Name for the experiment directory under `experiments/` |
| `--split` | `train` | Dataset split to train on |
| `--num_classes` | `14` | 13 sound classes + background |
| `--batch_size` | `4` | Sequences per batch |
| `--num_workers` | auto | DataLoader workers (capped at 16, 1 core reserved) |
| `--audio_only` | `False` | Zero out RGB channels (Track A mode) |

Checkpoints are saved to `experiments/<exp_name>/energy_seg_best.pth` whenever validation loss improves. Training logs are written to `experiments/<exp_name>/train.log`.

---

## Inference

Runs the trained model on all sequences in a dataset split and writes one submission-format JSON file per sequence.

```bash
python run_inference.py --exp_dir experiments/my_run
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--exp_dir` | **required** | Path to the experiment directory |
| `--checkpoint` | `<exp_dir>/energy_seg_best.pth` | Override checkpoint path |
| `--split` | `test` | Dataset split to run on |
| `--num_seqs` | all | Randomly sample N sequences (useful for debugging) |
| `--seed` | `42` | RNG seed for sequence sampling |
| `--batch_size` | `32` | Frames per forward pass |

### Filtering Pipeline

Detections pass through five sequential stages before export:

| Stage | Argument | Default | Effect |
|---|---|---|---|
| 1. Score threshold | `--score_thr` | `0.35` | Drop low-confidence detections |
| 2. Per-class NMS | `--nms_iou_thr` | `0.30` | Remove spatially redundant boxes |
| 3. Per-frame class cap | `--max_dets_per_class` | `3` | At most N detections per class per frame |
| 4. Track confirmation | `--min_hits` | `2` | Only export tracks seen ≥ N frames |
| 5. Energy sparsification | `--energy_top_k` | `20` | Export top-K energy points per detection |

### Tracker Arguments

| Argument | Default | Description |
|---|---|---|
| `--iou_thr` | `0.3` | Minimum IoU to associate a detection to an existing track |
| `--max_age` | `5` | Frames a track can coast (no detection) before deletion |
| `--coast_decay` | `0.9` | Score multiplier applied per coasting frame |

### Output

Results are written to `experiments/<exp_dir>/inference_outputs/<seq_name>_inference.json` in the DCASE 2026 Task 3 submission format. A summary of per-sequence track counts and throughput (fps) is printed to stdout and to `experiments/<exp_dir>/inference.log`.

---

## Evaluation

Computes Mask mAP, Pearson *r*, and Relative Distance Error against the STAIRS26 ground-truth annotations. Both ground-truth and prediction segmentations are rendered into continuous 360×180 energy maps through a spherical Gaussian kernel ($\sigma = 6^\circ$) before any metric is computed, making the evaluator equally valid for dense polygon and sparse peak representations.

```bash
python evaluate.py --exp_dir experiments/my_run
```

| Argument | Default | Description |
|---|---|---|
| `--exp_dir` | **required** | Experiment directory containing `inference_outputs/` |
| `--gt_root` | `LABELS_BASE` | Root directory of ground-truth JSON annotations |
| `--split` | `test` | Split to evaluate against |
| `--sigma` | `6.0` | Gaussian kernel standard deviation in degrees of arc |
| `--energy_thr` | `0.10` | Binary mask threshold as a fraction of peak energy |

### Metrics

- **Mask mAP (IoU 0.50:0.95)** — Primary ranking metric; simultaneously penalizes missed detections, false positives, mislocalization, and poor energy reconstruction.
- **Mask AP50** — mAP at IoU ≥ 0.50 only.
- **Pearson *r*** — Energy field shape fidelity over spatially matched pairs, macro-averaged across classes.
- **RDE (%)** — Mean absolute percentage error of predicted vs. reference source distance, macro-averaged across classes.

Matching uses the Hungarian algorithm with a 20° great-circle angular threshold. Cross-class matches are strictly rejected. Results are printed to stdout and saved to `experiments/<exp_dir>/eval_summary.txt`.

---

## Citation

If you use this system or components of its architecture, please cite the following:

**Baseline Architecture (UpLAM):**
```bibtex
@inproceedings{roman2025latent,
  title={Latent Acoustic Mapping for Direction of Arrival Estimation: A Self-Supervised Approach},
  author={Roman, Adrian S and Roman, Iran R and Bello, Juan P},
  booktitle={2025 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  pages={1--5},
  year={2025},
  organization={IEEE},
  url={[https://ccrma.stanford.edu/~iran/papers/Roman_et_al_WASPAA_2025.pdf](https://ccrma.stanford.edu/~iran/papers/Roman_et_al_WASPAA_2025.pdf)}
}
```

**Backbone Network (Mask R-CNN):**
```bibtex
@inproceedings{he2017mask,
  title={Mask {R-CNN}},
  author={He, Kaiming and Gkioxari, Georgia and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages={2961--2969},
  year={2017}
}
```
