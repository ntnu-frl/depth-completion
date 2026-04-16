import numpy as np
import torch

def invert_depth_repr(depth, focal_length, focal_length_canonical, depth_min):
    mask = depth > 0
    focal_length = torch.as_tensor(focal_length, dtype=torch.float32, device=depth.device)
    depth_min = torch.as_tensor(depth_min, dtype=torch.float32, device=depth.device)
    output = torch.zeros_like(depth)
    output[mask] = 1.0 / depth[mask]
    if focal_length_canonical is not None:
        focal_length_canonical = torch.as_tensor(focal_length_canonical, dtype=torch.float32, device=depth.device)
        output *= depth_min.item() * focal_length.view(-1, 1, 1, 1) / focal_length_canonical.item()
    return output

def resize_intrinsics(matrix, original_resolution, new_resolution):
    scale_x = new_resolution[0] / original_resolution[0]
    scale_y = new_resolution[1] / original_resolution[1]
    resized = matrix.copy()
    resized[0, 0] *= scale_x
    resized[1, 1] *= scale_y
    resized[0, 2] *= scale_x
    resized[1, 2] *= scale_y
    return resized

def project_prior_points(depth_prior_points, matrix, resolution, scale_canonical, max_prior_depth):
    width, height = resolution
    if depth_prior_points is None:
        return None, None

    depth_prior_points = np.asarray(depth_prior_points, dtype=np.float32)
    if depth_prior_points.shape[0] != 3 or depth_prior_points.size == 0:
        return None, None

    depth_prior_image = matrix @ depth_prior_points
    depth_prior_depth = depth_prior_image[2, :].copy()
    depth_prior_image = depth_prior_image / depth_prior_depth
    u = depth_prior_image[0, :]
    v = depth_prior_image[1, :]

    inside_image_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid_mask = (depth_prior_depth > 0.0) & (depth_prior_depth < max_prior_depth) & inside_image_mask
    if valid_mask.sum() == 0:
        return None, None

    depth_prior_image = depth_prior_image[:, valid_mask]
    depth_prior_depth = depth_prior_depth[valid_mask] * scale_canonical
    return depth_prior_image, depth_prior_depth

