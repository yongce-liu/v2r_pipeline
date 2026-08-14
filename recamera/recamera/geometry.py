"""Backproject masked depth into a 3D point cloud in the camera frame.

The human arm cloud is the geometric reference for the depth-match camera pose
optimization: the robot arm (retarget world frame) is posed so that, projected
through the solved camera extrinsics, it lands where the human arm was in the
original first-person video.
"""

from __future__ import annotations

import numpy as np


def backproject(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    *,
    stride: int = 1,
) -> np.ndarray:
    """Backproject masked depth pixels into camera-frame 3D points.

    ``mask`` is a (H, W) bool; ``depth`` a (H, W) metric depth in meters;
    ``K`` the 3x3 intrinsics. Returns an (N, 3) array of x,y,z points in the
    pinhole camera frame (z = depth along the optical axis).
    """
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    keep = (z > 1e-4) & np.isfinite(z)
    xs, ys, z = xs[keep], ys[keep], z[keep]
    if stride > 1:
        xs, ys, z = xs[::stride], ys[::stride], z[::stride]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    pts = np.stack(
        [(xs - cx) / fx * z, (ys - cy) / fy * z, z],
        axis=1,
    )
    return pts
