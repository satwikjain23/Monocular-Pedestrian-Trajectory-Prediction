# Pedestrian Trajectory Prediction for Autonomous Vehicles using Monocular Camera

A lightweight monocular vision-based framework for pedestrian trajectory prediction and forward-collision warning using only RGB camera input.

The proposed system combines:

- Depth Anything V2 for monocular depth estimation
- Geometric 3D reconstruction
- Constant Velocity Model (CVM) for trajectory prediction
- SegFormer-B0 for road segmentation
- Forward collision detection

Unlike many existing approaches, this framework does not require LiDAR, stereo cameras, or trajectory-specific neural network training, making it suitable for lightweight autonomous driving and ADAS system.

---

## Features

- Monocular RGB-only perception
- Metric depth estimation using Depth Anything V2
- 3D pedestrian localization
- Constant Velocity trajectory prediction
- Real-time road segmentation
- Collision warning generation

---

## Repository Structure

```
Monocular-Pedestrian-Trajectory-Prediction/

├── scripts/
│   ├── main.py
│   └── generate_depth_images.py
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/satwikjain23/Monocular-Pedestrian-Trajectory-Prediction.git

cd Monocular-Pedestrian-Trajectory-Prediction
```

Create a virtual environment

```bash
python -m venv venv

source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Download nuScenes Dataset

Download the **nuScenes v1.0 Trainval** dataset from

https://www.nuscenes.org/download

After downloading, organize the dataset as

```
nuscenes_data/

├── samples/
│   └── CAM_FRONT/

├── sweeps/

├── maps/

└── v1.0-trainval/
```

---

# Install nuScenes DevKit

```bash
pip install nuscenes-devkit
```

---

# Install Depth Anything V2

Clone the official repository

https://github.com/DepthAnything/Depth-Anything-V2

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git

cd Depth-Anything-V2

pip install -r requirements.txt
```

The scripts import

```python
from depth_anything_v2.dpt import DepthAnythingV2
```

so ensure the package is available in your Python environment.

---

# Generate Depth Images

Edit the dataset paths in

```
scripts/generate_depth_images.py
```

Run

```bash
python scripts/generate_depth_images.py
```

The script

- loads Depth Anything V2
- generates dense monocular depth maps
- saves grayscale depth images

---

# Run Trajectory Prediction

Edit the paths inside

```
scripts/main.py
```

```
DATA_ROOT = "/path/to/nuscenes"

DEPTH_DIR = "/path/to/generated/depth/images"
```

Run

```bash
python scripts/main.py
```

The system performs

- pedestrian localization
- 3D reconstruction
- trajectory prediction
- road segmentation
- collision detection
- visualization

---

# Models Used

| Model | Purpose |
|--------|---------|
| Depth Anything V2 Small | Monocular depth estimation |
| SegFormer-B0 (Cityscapes) | Road segmentation |

---

# Dataset

- nuScenes v1.0 Trainval
