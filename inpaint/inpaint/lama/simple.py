"""LaMa (big-lama) inpainter via the ``simple-lama-inpainting`` TorchScript wrapper.

Drives ``SimpleLama`` — a pre-scripted big-lama FFC network shipped as a single
TorchScript file — behind the same ``inpaint(image, output_path, args, mask=...)``
entry point as :class:`inpaint.qwen.inpainter.QwenInpainter`, so the video
workflow can drive either backend.

Input/output conventions (per ``simple-lama-inpainting``):

- ``image``: a clean RGB frame — a 3-channel ``np.ndarray``, PIL image, or the
  CHW uint8 tensor the video workflow feeds in. The masked hole must NOT be
  painted: ``SimpleLama`` zeroes it internally (``img * (1 - mask)``), so the
  original background must still be there for the network to copy from.
- ``mask``: a 1-channel binary mask where nonzero pixels (255 when saved as a
  uint8 PNG) mark the region to inpaint. This matches the repo's
  :func:`inpaint.media.load_mask` convention (0/255).
- returns: a ``PIL.Image.Image`` (the inpainted frame, modulo the package's
  mod-8 symmetric padding).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from loguru import logger

from inpaint.device import resolve_torch_device, set_cuda_device_if_indexed
from inpaint.lama.args import LamaInpaintArgs

if TYPE_CHECKING:
    from PIL import Image


class SimpleLamaInpainter:
    """Reusable ``simple-lama-inpainting`` inpainter for masked frames."""

    def __init__(self, args: LamaInpaintArgs, device: str | None = None) -> None:
        self.args = args
        self.device = resolve_torch_device(device or args.device)
        set_cuda_device_if_indexed(self.device)
        self._simple_lama = None

    @property
    def simple_lama(self):
        """Lazily built ``SimpleLama`` (first use downloads/loads the weights)."""

        if self._simple_lama is None:
            self._simple_lama = self._build_simple_lama()
        return self._simple_lama

    def _build_simple_lama(self):
        """Point the package at a local TorchScript checkpoint when given.

        ``LAMA_MODEL`` is the only knob ``SimpleLama`` exposes; setting it here
        mirrors what the user could do in the shell, so an offline run does not
        hit the author's GitHub release.
        """
        if self.args.model_path is not None:
            model_path = self.args.model_path.expanduser()
            if not model_path.exists():
                raise FileNotFoundError(
                    f"simple-lama TorchScript model not found: {model_path}. "
                    "Drop --lama.model-path to let the package download "
                    "big-lama.pt from GitHub, or point it at a valid "
                    "simple-lama TorchScript checkpoint."
                )
            os.environ["LAMA_MODEL"] = str(model_path)

        from simple_lama_inpainting import SimpleLama

        return SimpleLama(device=self.device)

    def inpaint(
        self,
        image,
        output_path: Path,
        args: LamaInpaintArgs,
        mask=None,
    ) -> None:
        """Inpaint one frame and save the result.

        ``image`` is a clean RGB frame (ndarray / PIL / CHW uint8 tensor) and
        ``mask`` a binary 1-channel mask (ndarray / PIL) whose nonzero pixels
        mark the hole. The result is a PIL image saved to ``output_path``.
        """
        if mask is None:
            raise ValueError(
                "--lama-single.mask-path is required for the simple backend "
                "(simple-lama needs the real mask; it zeroes the hole itself)."
            )
        if output_path.exists() and not args.overwrite:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)

        started_at = time.perf_counter()
        pil_image = self._to_pil_rgb(image)
        pil_mask = self._to_pil_mask(mask)

        if args.dilate_ratio > 0:
            pil_mask = self._dilate_mask(pil_mask, args)

        # Repeat passes re-feed the previous output with the SAME mask: LaMa
        # zeroes the hole internally, so each pass reads progressively cleaner
        # boundary context and a large fill converges instead of staying a
        # single-shot result.
        result = pil_image
        for _ in range(args.repeat):
            result = self.simple_lama(result, pil_mask)
        result.save(output_path)

        logger.info(
            "[simple-lama] Inference complete: passes={} elapsed={:.2f}s, output={}",
            args.repeat,
            time.perf_counter() - started_at,
            output_path,
        )

    @staticmethod
    def _dilation_px_for_frame(min_side_px: int, dilate_ratio: float) -> int:
        """Outward dilation (px) for a frame whose shorter side is ``min_side_px``.

        Dilation is ``dilate_ratio`` of the frame's shorter side, so it scales
        with the source resolution instead of being a fixed pixel count (a fixed
        px is a tiny fraction of a 4K frame but a large chunk of a 720p one).
        Returns 0 for an empty frame or when dilation is disabled
        (``dilate_ratio <= 0``).
        """
        if dilate_ratio <= 0 or min_side_px <= 0:
            return 0
        return max(1, round(dilate_ratio * min_side_px))

    @staticmethod
    def _dilate_mask(mask_pil, args: LamaInpaintArgs):
        """Dilate a PIL ``L`` mask outward by a frame-relative pixel amount."""

        from PIL import Image

        arr = np.asarray(mask_pil)
        binary = (arr > 0).astype(np.uint8) * 255
        k = SimpleLamaInpainter._dilation_px_for_frame(
            min(mask_pil.size), args.dilate_ratio
        )
        if k > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)
            )
            binary = cv2.dilate(binary, kernel, iterations=1)
        logger.info(
            "[simple-lama] mask dilation: frame={} ratio={} -> {}px",
            mask_pil.size,
            args.dilate_ratio,
            k,
        )
        return Image.fromarray(binary, mode="L")

    @staticmethod
    def _to_pil_rgb(image) -> Image.Image:
        """Convert an accepted pipeline image to a PIL RGB frame."""

        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.ndim == 3:
                arr = tensor.permute(1, 2, 0).contiguous().cpu().numpy()
            elif tensor.ndim == 4:
                arr = tensor[0].permute(1, 2, 0).contiguous().cpu().numpy()
            else:
                raise ValueError(
                    f"Expected a 3-channel image, got {tuple(tensor.shape)}"
                )
        else:
            arr = np.asarray(image)

        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            arr = arr[:, :, :3]
        else:
            raise ValueError(f"Expected a 3-channel RGB image, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = arr.clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _to_pil_mask(mask) -> Image.Image:
        """Convert a binary mask to a PIL ``L`` image (nonzero = inpaint region)."""

        from PIL import Image

        if isinstance(mask, torch.Tensor):
            arr = mask.detach().squeeze().cpu().numpy()
        else:
            arr = np.asarray(mask)

        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            raise ValueError(f"Expected a 1-channel HW mask, got {arr.shape}")

        # Contract: nonzero pixels (255 when saved) mark the inpaint region.
        binary = ((arr > 0) * 255).astype(np.uint8)
        return Image.fromarray(binary, mode="L")
