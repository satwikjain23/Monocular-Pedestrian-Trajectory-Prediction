import os
import cv2
import torch
import numpy as np

from tqdm import tqdm
from huggingface_hub import hf_hub_download
from depth_anything_v2.dpt import DepthAnythingV2


# =====================================
# PATHS
# =====================================

INPUT_DIR = "/home/satwik/nuscenes_data_1/samples/CAM_FRONT"
OUTPUT_DIR = "/home/satwik/nuscenes_data_1/depth/CAM_FRONT"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================
# LOAD MODEL
# =====================================

model = DepthAnythingV2(
    encoder="vits",
    features=64,
    out_channels=[48, 96, 192, 384]
)

checkpoint_path = hf_hub_download(
    repo_id="depth-anything/Depth-Anything-V2-Small",
    filename="depth_anything_v2_vits.pth"
)

state_dict = torch.load(checkpoint_path, map_location="cpu")

model.load_state_dict(state_dict)

model.eval()


# =====================================
# PROCESS IMAGES
# =====================================

image_files = sorted(os.listdir(INPUT_DIR))

for img_name in tqdm(image_files):

    img_path = os.path.join(INPUT_DIR, img_name)

    # Read image
    img = cv2.imread(img_path)

    if img is None:
        continue

    # Original resolution
    orig_h, orig_w = img.shape[:2]

    # =====================================
    # Resize ONLY for faster inference
    # =====================================

    img_resized = cv2.resize(img, (518, 518))

    # =====================================
    # DEPTH INFERENCE
    # =====================================

    with torch.no_grad():

        depth = model.infer_image(img_resized)

    # =====================================
    # Resize depth back to original size
    # =====================================

    depth = cv2.resize(
        depth,
        (orig_w, orig_h),
        interpolation=cv2.INTER_CUBIC
    )

    # =====================================
    # NORMALIZE DEPTH
    # =====================================

    depth = depth.astype(np.float32)

    depth = depth - depth.min()

    if depth.max() > 0:
        depth = depth / depth.max()

    # Convert to 8-bit
    depth_uint8 = (depth * 255).astype(np.uint8)

    # =====================================
    # SAVE DEPTH IMAGE
    # =====================================

    out_path = os.path.join(OUTPUT_DIR, img_name)

    cv2.imwrite(out_path, depth_uint8)

print("✅ Done processing all images!")