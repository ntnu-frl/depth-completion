import numpy as np
import torch
import yaml
from pathlib import Path

def load_intrinsics(path):
    with open(path, "r") as handle:
        config = yaml.safe_load(handle)
    fx, fy, cx, cy = config["intrinsics"]
    resolution = config["resolution"]
    matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float32)
    distortion = np.array(config["distortion_coeffs"], dtype=np.float32)
    return matrix, resolution, distortion


def load_prior_points(path):
    prior_path = Path(path)
    if not prior_path.is_file():
        return np.empty((3, 0), dtype=np.float32)
    return np.load(prior_path).astype(np.float32)

def resize_intrinsics(matrix, original_resolution, new_resolution):
    scale_x = new_resolution[0] / original_resolution[0]
    scale_y = new_resolution[1] / original_resolution[1]
    resized = matrix.copy()
    resized[0, 0] *= scale_x
    resized[1, 1] *= scale_y
    resized[0, 2] *= scale_x
    resized[1, 2] *= scale_y
    return resized
