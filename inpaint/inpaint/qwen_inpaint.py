"""Qwen-Image-Edit inpainting for one masked frame (model wrapper)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

from inpaint.device import resolve_torch_device, set_cuda_device_if_indexed
from inpaint.qwen_model import QwenEditModelLoader

DEFAULT_INPAINT_PROMPT = "Discard and remove the hand and arms according to the colored masks of this image, keep the other part the same."
DEFAULT_INPAINT_NEGATIVE_PROMPT = (
    "hands, fingers, arms, human body parts, skin, person, red, blue"
)


@dataclass
class QwenInpaintArgs:
    """Arguments for Qwen image editing on a masked frame."""

    model_path: Path = Path(__file__).parents[2] / "ckpts/Qwen-Image-Edit-2511-NVFP4"
    """Qwen model: a diffusers directory (``model_index.json``) or a single-file
    ``.safetensors`` checkpoint (ComfyUI NVFP4). None disables inpainting."""

    base_model_path: Path = Path(__file__).parents[2] / "ckpts/Qwen-Image-Edit-2511"
    """Diffusers base model used with a single-file checkpoint. The single file
    only carries the quantized transformer; the VAE / text encoder / scheduler /
    tokenizer come from this base. Defaults to ``ckpts/Qwen-Image-Edit-2511``."""

    device: str = "auto"
    """Torch device: ``auto``, ``cuda[:N]``, or ``cpu``."""

    cpu_offload: bool = True
    """Offload whole pipeline components between stages to fit in 32GB VRAM."""

    downsample: float = 1.0
    """Inference size relative to the input image, in the range ``(0, 1]``."""

    prompt: str = DEFAULT_INPAINT_PROMPT
    negative_prompt: str = DEFAULT_INPAINT_NEGATIVE_PROMPT
    steps: int = 50
    true_cfg_scale: float = 6.0
    guidance_scale: float | None = None
    seed: int = 42
    overwrite: bool = True
    """Clear existing inpaint outputs and recompute. With it off, prior per-frame
    outputs are reused (idempotent re-runs)."""


class QwenInpainter:
    """Reusable Qwen-Image-Edit inpainter for masked frames."""

    def __init__(self, args: QwenInpaintArgs) -> None:
        if args.model_path is None:
            raise ValueError("Pass --qwen.model-path to run inpainting.")
        if not 0.0 < args.downsample <= 1.0:
            raise ValueError("--qwen.downsample must be in the range (0, 1].")

        self.device = resolve_torch_device(args.device)
        set_cuda_device_if_indexed(self.device)

        base_model_path = args.base_model_path

        self.pipeline = QwenEditModelLoader(
            model_path=args.model_path,
            device=self.device,
            base_model_path=base_model_path,
            cpu_offload=args.cpu_offload,
        ).load()

    @staticmethod
    def _downsample_image(image, ratio: float):
        """Resize an accepted pipeline image and align dimensions to 16 pixels."""
        from PIL import Image

        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        else:
            height, width = image.shape[-3:-1]

        if ratio == 1.0:
            target_width = max(16, width // 16 * 16)
            target_height = max(16, height // 16 * 16)
            resized = image
        elif isinstance(image, Image.Image):
            target_width = max(16, round(width * ratio / 16) * 16)
            target_height = max(16, round(height * ratio / 16) * 16)
            resized = image.resize(
                (target_width, target_height),
                resample=Image.Resampling.LANCZOS,
            )
        elif isinstance(image, torch.Tensor):
            target_width = max(16, round(width * ratio / 16) * 16)
            target_height = max(16, round(height * ratio / 16) * 16)
            source_dtype = image.dtype
            resized = torch.nn.functional.interpolate(
                image.unsqueeze(0).float(),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            if not source_dtype.is_floating_point:
                resized = resized.round().clamp(0, 255)
            resized = resized.to(source_dtype)
        else:
            target_width = max(16, round(width * ratio / 16) * 16)
            target_height = max(16, round(height * ratio / 16) * 16)
            resized = Image.fromarray(image).resize(
                (target_width, target_height),
                resample=Image.Resampling.LANCZOS,
            )
        logger.info(
            "[Qwen] Input downsample: {}x{} -> {}x{} ({:.3f})",
            width,
            height,
            target_width,
            target_height,
            ratio,
        )
        return resized, target_height, target_width

    def inpaint(
        self,
        image: torch.Tensor,
        output_path: Path,
        args: QwenInpaintArgs,
    ) -> None:
        if output_path.exists() and not args.overwrite:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        inference_image, height, width = self._downsample_image(
            image,
            args.downsample,
        )
        started_at = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        try:
            with torch.inference_mode():
                output = self.pipeline(
                    image=[inference_image],
                    prompt=args.prompt,
                    generator=generator,
                    true_cfg_scale=args.true_cfg_scale,
                    negative_prompt=(
                        args.negative_prompt if args.true_cfg_scale > 1.0 else None
                    ),
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                    height=height,
                    width=width,
                )
            output.images[0].save(output_path)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                peak_gib = torch.cuda.max_memory_allocated(self.device) / 2**30
            else:
                peak_gib = 0.0
            logger.info(
                "[Qwen] Inference complete: elapsed={:.2f}s, peak_cuda={:.2f}GiB, output={}",
                time.perf_counter() - started_at,
                peak_gib,
                output_path,
            )
        finally:
            del generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
