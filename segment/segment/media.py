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


def mask_is_empty(mask: np.ndarray) -> bool:
    return not np.any(mask > 0)


@dataclass(frozen=True)
class MaskStats:
    """Geometry summary of one binary mask (row/col pixel coordinates)."""

    has_mask: bool
    area: int
    """Number of foreground pixels."""
    bbox: tuple[int, int, int, int] | None
    """Inclusive bounding box ``(min_row, min_col, max_row, max_col)`` or None."""

    def to_dict(self) -> dict:
        bbox = None
        if self.bbox is not None:
            min_row, min_col, max_row, max_col = self.bbox
            bbox = {
                "min_row": min_row,
                "min_col": min_col,
                "max_row": max_row,
                "max_col": max_col,
            }
        return {"has_mask": self.has_mask, "area": self.area, "bbox": bbox}


def mask_stats(mask: np.ndarray) -> MaskStats:
    """Compute the foreground area and inclusive bbox of a binary mask."""

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return MaskStats(has_mask=False, area=0, bbox=None)
    return MaskStats(
        has_mask=True,
        area=int(ys.size),
        bbox=(int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
    )


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a saved binary mask as a uint8 (0/255) numpy array."""

    image = Image.open(mask_path)
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.uint8)


def save_mask(mask: np.ndarray, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path)


def save_overlay(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    alpha: float,
    mask_color_rgb: tuple[int, int, int],
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        return
    if not 0 <= alpha <= 1:
        raise ValueError("--overlay-alpha must be within [0, 1].")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_float = frame_rgb.astype(np.float32)
    mask_bool = mask > 0
    mask_color = np.array(mask_color_rgb, dtype=np.float32)

    vis = frame_float.copy()
    vis[mask_bool] = vis[mask_bool] * (1 - alpha) + mask_color * alpha
    Image.fromarray(vis.astype(np.uint8)).save(output_path, quality=95)
