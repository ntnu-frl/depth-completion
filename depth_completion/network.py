import numpy as np
import pathlib
import torch
import yaml

from pathlib import Path
from PIL import Image

from .model import DepthAnythingV2, model_configs
from .utils import invert_depth_repr, project_prior_points, resize_intrinsics

torch.serialization.add_safe_globals([pathlib.PosixPath])

def _resize_array(array, size, method):
    image = Image.fromarray(array)
    image = image.resize(size, resample=method)
    return np.array(image)


def _resize_sparse_mask(mask, target_size):
    src_h, src_w = mask.shape
    tgt_w, tgt_h = target_size
    scale_x = tgt_w / src_w
    scale_y = tgt_h / src_h
    nonzero_y, nonzero_x = np.nonzero(mask)
    new_x = np.clip((nonzero_x * scale_x).astype(int), 0, tgt_w - 1)
    new_y = np.clip((nonzero_y * scale_y).astype(int), 0, tgt_h - 1)
    new_mask = np.zeros((tgt_h, tgt_w), dtype=mask.dtype)
    new_mask[new_y, new_x] = mask[nonzero_y, nonzero_x]
    return new_mask


class DepthCompletionModel:
    def __init__(self, config_path=None, checkpoint_path=None, device=None):
        root_dir = Path(__file__).resolve().parent.parent
        self.config_path = Path(config_path) if config_path is not None else root_dir / "config.yaml"
        with self.config_path.open("r") as handle:
            self.config = yaml.safe_load(handle)

        if checkpoint_path is None:
            checkpoint_path = self.config_path.parent / self.config["checkpoint"]
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self):
        model = DepthAnythingV2(**model_configs[self.config["encoder"]])
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        return model.to(self.device).eval()

    def _preprocess_image(self, rgb_image):
        image = np.asarray(rgb_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("rgb_image must have shape (H, W, 3)")

        if np.issubdtype(image.dtype, np.floating) and image.size > 0 and float(image.max()) <= 1.0:
            image = image * 255.0

        if image.dtype != np.uint8:
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)

        image = _resize_array(image, tuple(self.config["input_resolution"]), Image.BILINEAR)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        return torch.tensor(image, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _project_prior(self, prior_points, intrinsics, image_size):
        points = np.asarray(prior_points, dtype=np.float32)
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] != 3:
            raise ValueError("prior_points must have shape (3, N)")
        if intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must have shape (3, 3)")

        depth_prior = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        projected_points, projected_depth = project_prior_points(
            points,
            intrinsics,
            image_size,
            1.0,
            self.config["depth_max"],
        )
        if projected_points is None:
            return depth_prior

        u = projected_points[0].astype(np.int64)
        v = projected_points[1].astype(np.int64)
        depth_prior[v, u] = projected_depth
        return depth_prior

    def _preprocess_prior(self, depth_prior, intrinsics):
        prior = _resize_sparse_mask(depth_prior, tuple(self.config["input_resolution"]))
        processed_prior = np.zeros_like(prior)
        nonzero_y, nonzero_x = np.nonzero(prior)

        for y, x in zip(nonzero_y, nonzero_x):
            start_y = (y // 14) * 14
            start_x = (x // 14) * 14
            processed_prior[start_y:start_y + 14, start_x:start_x + 14] = prior[y, x]

        resized_intrinsics = resize_intrinsics(
            intrinsics,
            depth_prior.shape[::-1],
            self.config["input_resolution"],
        )
        focal_length = float((resized_intrinsics[0, 0] + resized_intrinsics[1, 1]) / 2.0)
        prior_tensor = torch.tensor(processed_prior, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        return invert_depth_repr(
            prior_tensor,
            focal_length,
            self.config["focal_length_canonical"],
            self.config["depth_min"],
        )

    def _postprocess_prediction(self, prediction, intrinsics, output_size):
        focal_length = float((intrinsics[0, 0] + intrinsics[1, 1]) / 2.0)
        depth = invert_depth_repr(
            prediction.detach(),
            focal_length,
            self.config["focal_length_canonical"],
            self.config["depth_min"],
        ).squeeze().cpu().numpy()
        return _resize_array(depth, output_size, Image.BILINEAR)

    @torch.no_grad()
    def predict(self, rgb_image, prior_points, intrinsics):
        image = np.asarray(rgb_image)
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        output_size = (image.shape[1], image.shape[0])
        depth_prior = self._project_prior(prior_points, intrinsics, output_size)
        image_tensor = self._preprocess_image(image)
        prior_tensor = self._preprocess_prior(depth_prior, intrinsics)
        prediction = self.model(torch.concat([image_tensor, prior_tensor], dim=1))
        return self._postprocess_prediction(prediction, intrinsics, output_size)
