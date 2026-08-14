"""Geometry helpers for DA3 depth point clouds."""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger


def backproject_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project a depth image to camera-frame xyz points."""

    height, width = depth.shape
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    z = depth
    x = (x_grid - cx) * z / fx
    y = (y_grid - cy) * z / fy

    return np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)


def make_rgbd_point_cloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create camera-frame points and normalized RGB colors."""

    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Depth must be a 2D array, got {depth.shape}")

    depth_h, depth_w = depth.shape
    rgb_h, rgb_w = rgb.shape[:2]
    if (rgb_h, rgb_w) != (depth_h, depth_w):
        logger.warning(
            "Resize RGB from {}x{} to depth size {}x{}",
            rgb_w,
            rgb_h,
            depth_w,
            depth_h,
        )
        rgb = cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_AREA)

    points = backproject_depth(depth.astype(np.float32), intrinsics.astype(np.float32))
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    if max_points is not None and max_points > 0 and len(points) > max_points:
        stride = int(np.ceil(len(points) / float(max_points)))
        points = points[::stride]
        colors = colors[::stride]

    return points, colors
