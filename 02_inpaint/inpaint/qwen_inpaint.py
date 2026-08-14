"""Qwen-Image-Edit inpainting for one masked frame (model wrapper)."""

from __future__ import annotations

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

    model_path: Path | None = None
    """Qwen model: a diffusers directory (``model_index.json``) or a single-file
    ``.safetensors`` checkpoint (ComfyUI NVFP4). None disables inpainting."""

    base_model_path: Path | None = None
    """Diffusers base model used with a single-file checkpoint. The single file
    only carries the quantized transformer; the VAE / text encoder / scheduler /
    tokenizer come from this base. Defaults to ``ckpts/Qwen-Image-Edit-2511``."""

    device: str = "auto"
    """Torch device: ``auto``, ``cuda[:N]``, or ``cpu``."""

    cpu_offload: bool = False
    """Offload pipeline modules to CPU sequentially to reduce VRAM."""

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

        self.device = resolve_torch_device(args.device)
        set_cuda_device_if_indexed(self.device)

        base_model_path = args.base_model_path
        if base_model_path is None:
            default_base = (
                Path(__file__).parents[2] / "ckpts" / "Qwen-Image-Edit-2511"
            )
            if default_base.exists():
                base_model_path = default_base

        self.pipeline = QwenEditModelLoader(
            model_path=args.model_path,
            device=self.device,
            base_model_path=base_model_path,
            cpu_offload=args.cpu_offload,
        ).load()

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

        try:
            with torch.inference_mode():
                output = self.pipeline(
                    image=[image],
                    prompt=args.prompt,
                    generator=generator,
                    true_cfg_scale=args.true_cfg_scale,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                )
            output.images[0].save(output_path)
        finally:
            del generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
