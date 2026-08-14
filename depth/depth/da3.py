"""Depth Anything 3 inference for a single image."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger

from depth import resolve_torch_device, set_cuda_device_if_indexed
from depth.geometry import backproject_depth


@dataclass(frozen=True)
class Da3ImageOutputs:
    """Paths produced by DA3 image inference."""

    image_path: Path
    depth_path: Path
    intrinsics_path: Path
    pointcloud_path: Path


def _processed_hw(prediction: Any) -> tuple[int, int]:
    processed = prediction.processed_images
    return int(processed.shape[1]), int(processed.shape[2])


def _as_numpy_array(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype)
    return array


class Da3Predictor:
    """Reusable Depth Anything 3 predictor for single images."""

    def __init__(self, model_path: Path | None, device: str = "auto") -> None:
        if model_path is None:
            raise ValueError("Pass --model-path to run DA3.")
        model_path = model_path.expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"DA3 model path not found: {model_path}")

        self.model_path = model_path
        self.device = resolve_torch_device(device)
        set_cuda_device_if_indexed(self.device)

        from depth_anything_3.api import DepthAnything3

        logger.info("[DA3] Loading model: device={}, path={}", self.device, model_path)
        self.model = DepthAnything3.from_pretrained(str(model_path)).to(self.device)
        self.model.eval()

    def predict_depth_arrays(
        self,
        image_path: Path,
        process_res: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run DA3 depth prediction on one image and return the raw arrays.

        Returns ``(depth, intrinsics)`` at the input image resolution: ``depth``
        is a ``(H, W)`` float32 map and ``intrinsics`` a ``(3, 3)`` float32
        camera matrix rescaled to the input resolution. Nothing is written to
        disk, which lets a video-mode caller reuse the same predictor per frame.
        """
        if process_res <= 0:
            raise ValueError("--process-res must be positive.")

        image_path = image_path.expanduser()
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]

        with torch.inference_mode():
            prediction = self.model.inference(
                image=[rgb],
                process_res=process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
                export_format="mini_npz",
            )

        if prediction.intrinsics is None:
            raise RuntimeError("DA3 did not return intrinsics for image.")

        depth = _as_numpy_array(prediction.depth[0], np.float32)
        intrinsics = _as_numpy_array(prediction.intrinsics[0], np.float32).copy()
        proc_h, proc_w = _processed_hw(prediction)

        scale_x = orig_w / float(proc_w)
        scale_y = orig_h / float(proc_h)
        intrinsics[0, 0] *= scale_x
        intrinsics[1, 1] *= scale_y
        intrinsics[0, 2] *= scale_x
        intrinsics[1, 2] *= scale_y

        if (proc_h, proc_w) != (orig_h, orig_w):
            depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return depth.astype(np.float32), intrinsics.astype(np.float32)

    def predict_image_depth(
        self,
        image_path: Path,
        output_dir: Path,
        process_res: int,
        overwrite: bool,
        output_image_name: str = "hand_inpaint.png",
    ) -> Da3ImageOutputs:
        """Run DA3 depth prediction on one image and write the outputs."""

        image_path = image_path.expanduser()
        output_dir = output_dir.expanduser()

        output_dir.mkdir(parents=True, exist_ok=True)
        staged_image_path = output_dir / output_image_name
        depth_path = output_dir / "depth_inpaint_image.npy"
        intrinsics_path = output_dir / "intrinsics_inpaint_image.npy"
        pointcloud_path = output_dir / "cloud_inpaint_image.npy"
        summary_path = output_dir / "summary.txt"

        required_outputs = [
            staged_image_path,
            depth_path,
            intrinsics_path,
            pointcloud_path,
            summary_path,
        ]
        if not overwrite and all(path.exists() for path in required_outputs):
            return Da3ImageOutputs(
                image_path=staged_image_path,
                depth_path=depth_path,
                intrinsics_path=intrinsics_path,
                pointcloud_path=pointcloud_path,
            )

        depth, intrinsics = self.predict_depth_arrays(image_path, process_res)
        orig_h, orig_w = depth.shape[:2]

        if (overwrite or not staged_image_path.exists()) and (
            image_path.resolve() != staged_image_path.resolve()
        ):
            shutil.copyfile(image_path, staged_image_path)

        points = backproject_depth(
            depth.astype(np.float32), intrinsics.astype(np.float32)
        )

        np.save(depth_path, depth.astype(np.float32))
        np.save(intrinsics_path, intrinsics.astype(np.float32))
        np.save(pointcloud_path, points)
        summary_path.write_text(
            "\n".join(
                [
                    f"image: {image_path}",
                    f"resolution: {orig_w}x{orig_h}",
                    f"model: {self.model_path}",
                    "depth_mode: image",
                    "",
                ]
            )
        )

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return Da3ImageOutputs(
            image_path=staged_image_path,
            depth_path=depth_path,
            intrinsics_path=intrinsics_path,
            pointcloud_path=pointcloud_path,
        )
