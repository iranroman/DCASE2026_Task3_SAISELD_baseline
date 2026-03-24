This repository contains a multimodal deep learning pipeline designed to perform instance segmentation, energy mapping, and distance estimation using both visual and spatial audio data. 

The pipeline fuses RGB video with spatial audio for deep acoustic imaging. The audio is processed via [**UpLAM** (Latent Acoustic Mapping)](https://ccrma.stanford.edu/~iran/papers/Roman_et_al_WASPAA_2025.pdf), a self-supervised framework that leverages a complex-valued deep back-projection network (CDBPN) to upsample standard 4-channel audio into a 32-channel representation that is used to generate 9-band equirectangular acoustic images. These dense acoustic features are concatenated with RGB frames and fed into our modified **Mask R-CNN** (`EnergyInstanceModel`). Custom sub-networks replace the standard mask head to predict:
1. **Energy Masks** (Pixel-wise acoustic energy localization)
2. **Full-Image Energy Maps** (Global energy distribution)
3. **Source Distance** (Regression based on RoI features)

---

## 🛠️ Setup & Installation

This project requires a specific directory structure to align the 10 FPS video frames with the corresponding acoustic images.

### 1. Prerequisites
* **Conda:** Recommended for environment management.
* **FFmpeg:** Required for extracting frames from the raw dataset videos. (Install via `sudo apt install ffmpeg` on Ubuntu or `brew install ffmpeg` on macOS).
* [STARSS23 Dataset](https://zenodo.org/records/7880637)
* The pre-trained UpLAM.pth model checkpoint is provided here, but can also be found in [the model's original repository](https://github.com/adrianSRoman/LAM)

### 2. Environment Setup
The codebase was developed and tested on **Python 3.10.19**. To avoid dependency conflicts (especially with spatial mapping libraries), create a dedicated Conda environment:

```bash
# Create and activate the conda environment
conda create -n saiseld_env python=3.10.19 -y
conda activate saiseld_env
```

Install PyTorch and Torchvision. *(Note: Adjust the CUDA version below to match your local hardware).*

```bash
# Example for CUDA 11.8. 
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install the remaining Python dependencies:

```bash
# Core math, audio, and imaging libraries
pip install numpy scipy soundfile librosa scikit-image scikit-learn Pillow tqdm

# Astropy and Basemap are required for equirectangular projections
pip install astropy matplotlib basemap
```

### 3. Data Preparation (STARSS23)
The data loader (`EnergySegDataset`) expects a very specific directory structure where video frames, JSON labels, and WAV mic files mirror each other's split and sequence names.

Your project root must look exactly like this before running `train.py`:

```text
├── checkpoints/
│   └── UpLAM.pth                # Pre-trained UpLAM weights
├── STARSS23/
│   ├── labels_dev/              # Ground truth JSONs
│   │   ├── train/
│   │   │   └── fold1_room1_mix001.json
│   │   └── test/
│   ├── mic_dev/                 # 4-channel eigenmike WAVs (24kHz)
│   │   ├── train/
│   │   │   └── fold1_room1_mix001.wav
│   │   └── test/
│   └── frames_dev/              # Extracted RGB frames (10 FPS)
│       ├── train/
│       │   └── fold1_room1_mix001/
│       │       ├── fold1_room1_mix001_0000.png
│       │       ├── fold1_room1_mix001_0001.png
│       │       └── ...
│       └── test/
```

### 4. Extracting Frames from Video

The audio feature extractor (`acoustic_features.py`) calculates visibility matrices using exactly 2400 samples per video frame (assuming a 24,000 Hz sample rate). This corresponds strictly to **10 Frames Per Second (FPS)** during video preprocessing.

Each video must therefore be converted into 10 FPS frames and resized to **360×180**.

Assuming the raw STARSS23 videos are downloaded into the `STARSS23/video_dev/` directory, your input structure should look like this:

```text
video_dev/
├── dev-test-tau/
│   ├── fold4_room10_mix001.mp4
│   ├── fold4_room10_mix002.mp4
│   └── ...
├── dev-train-tau/
│   ├── fold1_room1_mix001.mp4
│   └── ...
```

**Output Structure**

Frames will be written to a parallel directory tree at `STARSS23/frames_dev/`. Each video receives its own directory, and frame filenames preserve the original sequence name:

```text
frames_dev/
├── dev-test-tau/
│   ├── fold4_room10_mix001/
│   │   ├── fold4_room10_mix001_0000.png
│   │   ├── fold4_room10_mix001_0001.png
│   │   └── ...
│   ├── fold4_room10_mix002/
│   │   └── ...
│   └── ...
```

**One-Line Frame Extraction Command**

Navigate into your `STARSS23` directory and run the following command:

```bash
find video_dev -type f -name "*.mp4" -exec bash -c 'rel="${1#video_dev/}"; base="$(basename "$rel" .mp4)"; outdir="frames_dev/$(dirname "$rel")/$base"; mkdir -p "$outdir"; ffmpeg -hide_banner -loglevel error -i "$1" -vf "fps=10,scale=360:180" "$outdir/${base}_%04d.png"' _ {} \;
```

**What This Command Does:**
For every `.mp4` file inside `video_dev`:
1.  **Finds all videos recursively** (supports nested directories).
2.  **Recreates the same directory structure** inside `frames_dev`.
3.  **Creates a specific folder** for each video sequence.
4.  **Extracts frames at exactly 10 FPS** (`fps=10`).
5.  **Resizes each frame** to 360×180 (`scale=360:180`).
6.  **Names frames sequentially**, starting at 0001 (`%04d.png`).

*(Example Output: `frames_dev/dev-test-tau/fold4_room10_mix001/fold4_room10_mix001_0001.png`)*
