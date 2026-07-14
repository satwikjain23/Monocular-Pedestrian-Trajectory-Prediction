import os
import cv2
import torch
import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# =====================================
# SETTINGS
# =====================================

DATA_ROOT = "/home/satwik/nuscenes_data_1"
DEPTH_DIR = "/home/satwik/nuscenes_data_1/depth/CAM_FRONT"
SCENE_INDEX = 64

DEPTH_SCALE = 50.0
DEPTH_SHIFT = 0.0

tracks = {}

# =====================================
# DEPTH CONVERSION
# =====================================

def depth_to_meters(raw_depth):

    if raw_depth <= 0:
        return None

    # Estimated scaling from your calibration
    z = (1700.0 / raw_depth) - 1.0

    return z

# =====================================
# SEGMENTATION MODEL
# =====================================

class LaneDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
        )

        self.model.to(self.device)
        self.model.eval()

    def detect(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits

        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )

        return upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    def visualize(self, image, mask):
        color_seg = np.zeros_like(image)
        color_seg[mask == 0] = [0, 255, 0]  # road = 0
        return cv2.addWeighted(image, 0.6, color_seg, 0.4, 0)


# =====================================
# CROPPING
# =====================================

def crop_road_slanted(mask):
    h, w = mask.shape
    new_mask = np.ones_like(mask) * 255

    for y in range(h):
        t = y / h

        left = int((1 - t) * (0.9 * w) + t * (0.2 * w))
        right = int((1 - t) * (0.2 * w) + t * (0.8 * w))

        new_mask[y, left:right] = mask[y, left:right]

    return new_mask


# =====================================
# TRANSFORMS
# =====================================

def camera_to_global(point_cam, cs_record, pose_record):
    point = np.dot(Quaternion(cs_record['rotation']).rotation_matrix, point_cam)
    point += np.array(cs_record['translation'])

    point = np.dot(Quaternion(pose_record['rotation']).rotation_matrix, point)
    point += np.array(pose_record['translation'])

    return point


def global_to_camera(point_global, cs_record, pose_record):
    point = point_global - np.array(pose_record['translation'])
    point = np.dot(Quaternion(pose_record['rotation']).rotation_matrix.T, point)

    point = point - np.array(cs_record['translation'])
    point = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, point)

    return point


# =====================================
# FUTURE PREDICTION
# =====================================

def predict_future_3d(points):
    if len(points) < 5:
        return None

    pts = np.array(points[-5:])
    vx = np.mean(np.diff(pts[:, 0]))
    vy = np.mean(np.diff(pts[:, 1]))

    x, y = pts[-1, 0], pts[-1, 1]

    return np.array([x + vx, y + vy, pts[-1, 2]])


# =====================================
# COLLISION
# =====================================

def is_on_road(mask, u, v, radius=6):
    h, w = mask.shape

    u_min = max(u - radius, 0)
    u_max = min(u + radius, w)
    v_min = max(v - radius, 0)
    v_max = min(v + radius, h)

    region = mask[v_min:v_max, u_min:u_max]
    return np.any(region == 0)


# =====================================
# MAIN
# =====================================

def main():

    nusc = NuScenes("v1.0-trainval", DATA_ROOT, verbose=True)
    detector = LaneDetector()

    scene = nusc.scene[SCENE_INDEX]
    sample_token = scene['first_sample_token']

    while sample_token != "":

        sample = nusc.get('sample', sample_token)
        cam_token = sample['data']['CAM_FRONT']

        img_path, boxes, K = nusc.get_sample_data(cam_token)
        image = cv2.imread(img_path)

        cam_data = nusc.get('sample_data', cam_token)
        cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])

        # =====================================
        # LOAD DEPTH
        # =====================================
        depth_path = os.path.join(DEPTH_DIR, os.path.basename(img_path))
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        if depth_img is not None and depth_img.shape[:2] != image.shape[:2]:
            depth_img = cv2.resize(depth_img, (image.shape[1], image.shape[0]))

        # =====================================
        # SEGMENTATION
        # =====================================
        mask = detector.detect(image)
        mask = crop_road_slanted(mask)
        image = detector.visualize(image, mask)

        collision_flag = False

        for box in boxes:

            if "pedestrian" not in box.name:
                continue

            ann = nusc.get('sample_annotation', box.token)
            ped_id = ann['instance_token']

            corners = box.corners()
            pixels = view_points(corners, K, normalize=True)

            if np.any(pixels[2, :] <= 0):
                continue

            corners_2d = pixels[:2, :].astype(int)

            # DRAW BOX
            for i in range(4):
                cv2.line(image, tuple(corners_2d[:, i]),
                         tuple(corners_2d[:, (i+1)%4]), (255,0,0), 2)
                cv2.line(image, tuple(corners_2d[:, i+4]),
                         tuple(corners_2d[:, ((i+1)%4)+4]), (255,0,0), 2)
                cv2.line(image, tuple(corners_2d[:, i]),
                         tuple(corners_2d[:, i+4]), (255,0,0), 2)

            # FOOT
            bottom = corners[:, [6,7,2,3]]
            foot_cam = np.mean(bottom, axis=1)

            foot_pixel = view_points(foot_cam.reshape(3,1), K, normalize=True)
            u, v = int(foot_pixel[0,0]), int(foot_pixel[1,0])

            if not (0 <= u < image.shape[1] and 0 <= v < image.shape[0]):
                continue

            cv2.circle(image, (u, v), 5, (255, 0, 0), -1)

            # =====================================
            # DEPTH FROM IMAGE
            # =====================================

            if depth_img is None:
                continue

            fx = K[0,0]
            fy = K[1,1]
            cx = K[0,2]
            cy = K[1,2]

            patch = depth_img[
                max(0, v-5):v+6,
                max(0, u-5):u+6
            ]

            if patch.size == 0:
                continue

            raw_depth = np.median(patch)

            z = depth_to_meters(raw_depth)

            if z is None:
                continue

            X = (u - cx) * z / fx
            Y = (v - cy) * z / fy

            est_cam = np.array([X, Y, z])

            est_global = camera_to_global(
                est_cam,
                cs_record,
                pose_record
            )

            # =====================================
            # TRACK USING ESTIMATED DEPTH
            # =====================================

            tracks.setdefault(ped_id, []).append(est_global)

            future_global = predict_future_3d(tracks[ped_id])

            if future_global is None:
                continue
           

            future_cam = global_to_camera(future_global, cs_record, pose_record)
            if future_cam[2] <= 0:
                continue

            pix = view_points(future_cam.reshape(3,1), K, normalize=True)
            fu, fv = int(pix[0,0]), int(pix[1,0])

            if not (0 <= fu < image.shape[1] and 0 <= fv < image.shape[0]):
                continue

            collision = is_on_road(mask, fu, fv)
            if collision:
                collision_flag = True

            color = (0,0,255) if collision else (0,255,255)

            cv2.circle(image, (fu, fv), 6, color, -1)
            cv2.line(image, (u, v), (fu, fv), color, 2)

        if collision_flag:
            cv2.putText(image, "FUTURE COLLISION DETECTED",
                        (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 3)

        # MAIN WINDOW
        cv2.imshow("Scene", cv2.resize(image,(1280,720)))

        # DEPTH WINDOW
        if depth_img is not None:
            depth_vis = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)
            combined = np.hstack((image, depth_vis))
            cv2.imshow("RGB + Depth", cv2.resize(combined,(1600,500)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sample_token = sample['next']

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()