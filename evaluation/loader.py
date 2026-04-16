import numpy as np
from pathlib import Path
from PIL import Image

from .utils import load_intrinsics, load_prior_points, resize_intrinsics

class DatasetLoader:
    def __init__(self, dataset_dir, config):
        self.dataset_dir = Path(dataset_dir)
        self.config = config
        self.rgb_dir = self.dataset_dir / "rgb"
        self.depth_dir = self.dataset_dir / "depth"
        self.prior_dir = self.dataset_dir / "depth_prior"
        self.intrinsics, self.resolution, _ = load_intrinsics(self.dataset_dir / "intrinsics.txt")
        self.resized_intrinsics = resize_intrinsics(
            self.intrinsics,
            self.resolution,
            self.config["intermediate_resolution"],
        )
        self.indexes = [
            int(rgb_file.name.split("_")[0])
            for rgb_file in sorted(self.rgb_dir.glob("*_rgb.jpg"))
        ]

    def __iter__(self):
        target_width, target_height = self.config["target_resolution"]
        intermediate_width, intermediate_height = self.config["intermediate_resolution"]
        for index in self.indexes:
            image = Image.open(self.rgb_dir / f"{index:05d}_rgb.jpg").resize(
                (intermediate_width, intermediate_height),
                Image.BILINEAR,
            )
            image = np.array(image)

            depth = np.load(self.depth_dir / f"{index:05d}_d.npz")['depth'].astype(np.float32)
            depth = Image.fromarray(depth, mode="F").resize((intermediate_width, intermediate_height), Image.NEAREST)
            depth = np.array(depth)

            start_x = (intermediate_width - target_width) // 2
            start_y = (intermediate_height - target_height) // 2
            image = image[start_y:start_y + target_height, start_x:start_x + target_width, :]
            depth = depth[start_y:start_y + target_height, start_x:start_x + target_width]

            prior_points = load_prior_points(self.prior_dir / f"{index:05d}_rp.npy")

            yield {
                "image": image,
                "depth": depth,
                "prior_points": prior_points,
                "intrinsics": self.resized_intrinsics,
            }
