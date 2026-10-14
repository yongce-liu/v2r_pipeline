"""MiniMax-Remover whole-video inpainting adapter.

The upstream repository ships a custom diffusers pipeline and transformer but
no installable package. This module adapts the vendored implementation in
``inpaint.minimax.upstream`` to the pipeline's frame/mask PNG contract.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from scipy import ndimage

from inpaint.device import resolve_torch_device, set_cuda_device_if_indexed
from inpaint.media import load_mask, load_rgb_image
from inpaint.minimax.args import UPSTREAM_REVISION, MiniMaxRemoverInpaintArgs
from inpaint.minimax.upstream import (
    Minimax_Remover_Pipeline,
    Transformer3DModel,
)

TEMPORAL_COMPRESSION = 4


def next_temporal_frame_count(frame_count: int) -> int:
    """Return the smallest ``4k + 1`` frame count that can hold ``frame_count``.

    ``AutoencoderKLWan`` temporal compression in this checkpoint is four, so
    MiniMax-Remover expects ``1, 5, 9, ...`` input frames. Shorter clips are
    padded by repeating the final frame and the generated padding is discarded.
    """

    if frame_count <= 0:
        raise ValueError("frame_count must be > 0.")
    remainder = (frame_count - 1) % TEMPORAL_COMPRESSION
    if remainder == 0:
        return frame_count
    return frame_count + TEMPORAL_COMPRESSION - remainder


class MiniMaxRemoverInpainter:
    """Reusable MiniMax-Remover inpainter for a whole video."""

    def __init__(self, args: MiniMaxRemoverInpaintArgs) -> None:
        self._validate_args(args)
        self.args = args

        if not torch.cuda.is_available():
            raise RuntimeError(
                "The MiniMax-Remover backend requires CUDA fp16 inference. "
                "Use --backend qwen or --backend lama on CPU-only machines."
            )
        self.device = resolve_torch_device(args.device)
        if self.device.type != "cuda":
            raise ValueError("--minimax.device must be a CUDA device.")
        set_cuda_device_if_indexed(self.device)
        self._pipeline = None

    @staticmethod
    def _validate_args(args: MiniMaxRemoverInpaintArgs) -> None:
        if args.model_path is None:
            raise ValueError("--minimax.model-path is required.")
        model_path = args.model_path.expanduser()
        if not model_path.exists():
            raise FileNotFoundError(
                f"MiniMax-Remover model snapshot not found: {model_path}. "
                "Download it explicitly with: hf download zibojia/minimax-remover "
                f"--local-dir {model_path}"
            )
        for component in ("vae", "transformer", "scheduler"):
            component_path = model_path / component
            if not component_path.is_dir():
                raise FileNotFoundError(
                    f"MiniMax-Remover snapshot is missing {component}/: "
                    f"{component_path}"
                )

        if args.height <= 0 or args.height % 16 != 0:
            raise ValueError("--minimax.height must be positive and divisible by 16.")
        if args.width <= 0 or args.width % 16 != 0:
            raise ValueError("--minimax.width must be positive and divisible by 16.")
        if args.steps <= 0:
            raise ValueError("--minimax.steps must be > 0.")
        if args.mask_dilation < 0:
            raise ValueError("--minimax.mask-dilation must be >= 0.")
        if args.post_mask_dilation < 0:
            raise ValueError("--minimax.post-mask-dilation must be >= 0.")

    @property
    def pipeline(self):
        """Lazily load the VAE, transformer and scheduler from local weights."""

        if self._pipeline is not None:
            return self._pipeline

        from diffusers.models import AutoencoderKLWan
        from diffusers.schedulers import UniPCMultistepScheduler

        model_path = self.args.model_path.expanduser()

        logger.info(
            "[minimax] loading local fp16 checkpoint: {} (vendored revision {})",
            model_path,
            UPSTREAM_REVISION,
        )
        vae = AutoencoderKLWan.from_pretrained(
            model_path / "vae",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        transformer = Transformer3DModel.from_pretrained(
            model_path / "transformer",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        scheduler = UniPCMultistepScheduler.from_pretrained(
            model_path / "scheduler",
            local_files_only=True,
        )
        pipeline = Minimax_Remover_Pipeline(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        ).to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        self._pipeline = pipeline
        return pipeline

    def _inference_size(self, frame_shape: tuple[int, int]) -> tuple[int, int]:
        """Return (height, width), matching the orientation policy used by the
        upstream demo when enabled."""

        frame_height, frame_width = frame_shape
        if self.args.auto_orientation and frame_height > frame_width:
            return self.args.width, self.args.height
        return self.args.height, self.args.width

    def _build_model_inputs(
        self,
        frame_paths: list[Path],
        mask_paths: list[Path],
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """Load and resize clean frames/masks to the upstream tensor contract."""

        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        expected_shape: tuple[int, int] | None = None

        for frame_path, mask_path in zip(frame_paths, mask_paths, strict=True):
            frame_rgb = load_rgb_image(frame_path)
            mask = load_mask(mask_path)
            if frame_rgb.shape[:2] != mask.shape[:2]:
                raise ValueError(
                    "frame/mask resolution mismatch: "
                    f"{frame_path} {frame_rgb.shape[:2]} vs "
                    f"{mask_path} {mask.shape[:2]}"
                )
            if expected_shape is None:
                expected_shape = frame_rgb.shape[:2]
            elif frame_rgb.shape[:2] != expected_shape:
                raise ValueError(
                    "MiniMax-Remover requires a constant frame resolution; "
                    f"expected {expected_shape}, got {frame_rgb.shape[:2]} for "
                    f"{frame_path}"
                )

            target_height, target_width = self._inference_size(frame_rgb.shape[:2])
            frame_resized = cv2.resize(
                frame_rgb,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            mask_resized = cv2.resize(
                mask,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
            images.append(frame_resized.astype(np.float32) / 127.5 - 1.0)
            masks.append((mask_resized > 0).astype(np.float32))

        model_frame_count = next_temporal_frame_count(len(images))
        while len(images) < model_frame_count:
            images.append(images[-1].copy())
            masks.append(masks[-1].copy())

        image_tensor = torch.from_numpy(np.stack(images)).half()
        mask_tensor = torch.from_numpy(np.stack(masks))[:, :, :, None].half()
        return (
            image_tensor,
            mask_tensor,
            model_frame_count,
            target_height,
            target_width,
        )

    @staticmethod
    def _as_uint8_rgb(frame) -> np.ndarray:
        if hasattr(frame, "convert"):
            frame = np.asarray(frame.convert("RGB"))
        else:
            frame = np.asarray(frame)
        if frame.dtype == np.uint8:
            return frame
        frame_float = frame.astype(np.float32)
        if frame_float.size and frame_float.max() <= 1.0 + 1e-6:
            frame_float *= 255.0
        return np.clip(frame_float, 0, 255).astype(np.uint8)

    def inpaint_video(
        self,
        frame_paths: list[Path],
        mask_paths: list[Path],
        output_paths: list[Path],
        args: MiniMaxRemoverInpaintArgs,
    ) -> None:
        """Remove masked objects from a sequence and save every output frame.

        Frames without a mask must receive a zero mask. The model runs on the
        full sequence, then each generated frame is resized back to the source
        resolution and composited only inside the dilated source mask.
        """

        if not (len(frame_paths) == len(mask_paths) == len(output_paths) > 0):
            raise ValueError(
                "frame/mask/output path lists must be non-empty and aligned."
            )
        if not args.overwrite and all(path.exists() for path in output_paths):
            return
        for path in output_paths:
            path.parent.mkdir(parents=True, exist_ok=True)

        images, masks, model_frame_count, target_height, target_width = (
            self._build_model_inputs(frame_paths, mask_paths)
        )
        started_at = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

        try:
            with torch.inference_mode():
                generated = self.pipeline(
                    images=images,
                    masks=masks,
                    num_frames=model_frame_count,
                    height=target_height,
                    width=target_width,
                    num_inference_steps=args.steps,
                    generator=torch.Generator(device=self.device).manual_seed(
                        args.seed
                    ),
                    iterations=args.mask_dilation,
                ).frames[0]
        except torch.cuda.OutOfMemoryError as err:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA out of memory during MiniMax-Remover inference. Lower "
                "--video.minimax.height/--video.minimax.width, reduce "
                "--video.max-frames, or use a GPU with more memory."
            ) from err

        if len(generated) < len(frame_paths):
            raise RuntimeError(
                "MiniMax-Remover returned fewer frames than requested: "
                f"{len(generated)} < {len(frame_paths)}."
            )

        for index, (frame_path, mask_path, output_path) in enumerate(
            zip(frame_paths, mask_paths, output_paths, strict=True)
        ):
            source_rgb = load_rgb_image(frame_path)
            source_height, source_width = source_rgb.shape[:2]
            generated_rgb = cv2.resize(
                self._as_uint8_rgb(generated[index]),
                (source_width, source_height),
                interpolation=cv2.INTER_LINEAR,
            )

            output_mask = load_mask(mask_path) > 0
            if output_mask.shape != (source_height, source_width):
                raise ValueError(
                    "mask resolution changed before MiniMax compositing: "
                    f"{mask_path} {output_mask.shape} != "
                    f"{(source_height, source_width)}"
                )
            if args.post_mask_dilation > 0:
                output_mask = ndimage.binary_dilation(
                    output_mask,
                    iterations=args.post_mask_dilation,
                )

            output = source_rgb.copy()
            output[output_mask] = generated_rgb[output_mask]
            from PIL import Image

            Image.fromarray(output).save(output_path)

        elapsed = time.perf_counter() - started_at
        peak_gib = (
            torch.cuda.max_memory_allocated(self.device) / 1024**3
            if self.device.type == "cuda"
            else 0.0
        )
        logger.info(
            "[minimax] Inference complete: frames={} padded={} elapsed={:.2f}s "
            "peak_allocated={:.2f}GiB out={}",
            len(frame_paths),
            model_frame_count - len(frame_paths),
            elapsed,
            peak_gib,
            output_paths[0].parent,
        )
