"""Image and mask helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load an image as an RGB numpy array."""

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a saved binary mask as a uint8 (0/255) numpy array."""

    image = Image.open(mask_path)
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.uint8)


def save_image(rgb: np.ndarray, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_path)


def apply_mask_overlay(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    mask_color_rgb: tuple[int, int, int],
) -> np.ndarray:
    """Blend a solid color over the mask region of a frame.

    Returns a copy; the frame is left untouched outside the mask.
    """

    if not 0 <= alpha <= 1:
        raise ValueError("--overlay-alpha must be within [0, 1].")

    frame_float = frame_rgb.astype(np.float32)
    mask_bool = mask > 0
    mask_color = np.array(mask_color_rgb, dtype=np.float32)

    vis = frame_float.copy()
    vis[mask_bool] = vis[mask_bool] * (1 - alpha) + mask_color * alpha
    return vis.astype(np.uint8)


@dataclass(frozen=True)
class ImageShape:
    """Resolution of one image."""

    height: int
    width: int

    def to_dict(self) -> dict:
        return {"height": self.height, "width": self.width}


def save_side_by_side(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    output_path: Path,
    overwrite: bool,
) -> None:
    """Write ``before | after`` concatenated horizontally as one image."""

    if output_path.exists() and not overwrite:
        return
    if before_rgb.shape[:2] != after_rgb.shape[:2]:
        raise ValueError("before/after frames must share the same resolution.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.concatenate([before_rgb, after_rgb], axis=1)
    Image.fromarray(canvas).save(output_path)
