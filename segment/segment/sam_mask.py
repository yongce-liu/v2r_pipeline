"""SAM3 hand-mask segmentation for a single RGB image."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from loguru import logger
from PIL import Image

from segment import (
    load_rgb_image,
    resolve_torch_device,
    save_mask,
    save_overlay,
    set_cuda_device_if_indexed,
)

DEFAULT_TEXT_PROMPT = "human hand and arm"
DEFAULT_MASK_COLOR_RGB = (0, 0, 255)


@dataclass
class SamMaskArgs:
    """Arguments for SAM3 mask generation on a single image."""

    checkpoint: Path | None = Path(__file__).parents[2] / "ckpts/sam3/sam3.pt"
    allow_hf_download: bool = False
    device: str = "auto"
    text_prompt: str = DEFAULT_TEXT_PROMPT
    score_threshold: float = 0.1
    overlay_alpha: float = 0.5
    mask_color_rgb: tuple[int, int, int] = DEFAULT_MASK_COLOR_RGB
    overwrite: bool = True


@dataclass
class SamMaskCliArgs:
    """CLI wrapper for SAM3 mask generation on a single image."""

    image_path: Path
    output_dir: Path
    sam_mask: SamMaskArgs = field(default_factory=SamMaskArgs)


@dataclass(frozen=True)
class SamMaskOutputs:
    """Output paths produced by SAM3 mask generation."""

    mask_path: Path
    overlay_path: Path
    mask: np.ndarray


@dataclass(frozen=True)
class SamMaskInstance:
    """Single SAM3 mask instance returned for a text prompt."""

    mask: np.ndarray
    score: float


def _score_to_float(score: Any) -> float:
    if torch.is_tensor(score):
        return float(score.detach().cpu().item())
    return float(score)


def _mask_to_numpy(mask: Any) -> np.ndarray:
    if torch.is_tensor(mask):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)

    while mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np, axis=0)
    return mask_np


class Sam3MaskGenerator:
    """Reusable SAM3 image segmenter."""

    def __init__(self, args: SamMaskArgs) -> None:
        if not 0 <= args.score_threshold <= 1:
            raise ValueError("--score-threshold must be within [0, 1].")

        checkpoint = args.checkpoint.expanduser() if args.checkpoint else None
        if checkpoint is None and not args.allow_hf_download:
            raise ValueError(
                "Pass --sam-mask.checkpoint or --sam-mask.allow-hf-download "
                "to let SAM3 download weights from Hugging Face."
            )
        if checkpoint is not None and not checkpoint.exists():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")

        self.device = resolve_torch_device(args.device)
        self.score_threshold = args.score_threshold
        set_cuda_device_if_indexed(self.device)

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        logger.info(
            "[SAM3] Loading model: device={}, source={}",
            self.device,
            checkpoint or "Hugging Face",
        )
        build_device = str(self.device) if self.device.type == "cuda" else "cpu"
        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint) if checkpoint else None,
            load_from_HF=checkpoint is None,
            device=build_device,
        )
        model.to(self.device)

        self.processor = Sam3Processor(
            model,
            device=self.device,
            confidence_threshold=args.score_threshold,
        )

    def _inference_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def segment_instances(
        self,
        frame_rgb: np.ndarray,
        text_prompt: str,
        score_threshold: float | None = None,
    ) -> list[SamMaskInstance]:
        threshold = self.score_threshold if score_threshold is None else score_threshold
        previous_threshold = getattr(self.processor, "confidence_threshold", None)
        if hasattr(self.processor, "set_confidence_threshold"):
            self.processor.set_confidence_threshold(threshold)

        image = Image.fromarray(frame_rgb)
        try:
            with self._inference_context():
                inference_state = self.processor.set_image(image)
                output = self.processor.set_text_prompt(
                    state=inference_state,
                    prompt=text_prompt,
                )
        finally:
            if (
                previous_threshold is not None
                and previous_threshold != threshold
                and hasattr(self.processor, "set_confidence_threshold")
            ):
                self.processor.set_confidence_threshold(previous_threshold)

        instances: list[SamMaskInstance] = []
        expected_shape = frame_rgb.shape[:2]
        for raw_mask, raw_score in zip(output["masks"], output["scores"]):
            score = _score_to_float(raw_score)
            if score < threshold:
                continue
            mask_np = _mask_to_numpy(raw_mask)
            if mask_np.shape != expected_shape:
                raise ValueError(
                    f"SAM3 mask shape mismatch: expected {expected_shape}, got {mask_np.shape}"
                )
            instances.append(
                SamMaskInstance(
                    mask=(mask_np > 0).astype(np.uint8),
                    score=score,
                )
            )
        return instances

    def segment(
        self, frame_rgb: np.ndarray, text_prompt: str
    ) -> tuple[np.ndarray, int]:
        instances = self.segment_instances(frame_rgb, text_prompt)
        mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
        for instance in instances:
            mask[instance.mask > 0] = 255
        return mask, len(instances)


def process_sam_mask(
    image_path: Path,
    output_dir: Path,
    args: SamMaskArgs,
    generator: Sam3MaskGenerator | None = None,
) -> SamMaskOutputs:
    """Run SAM3 mask generation for one image."""

    image_path = image_path.expanduser()

    mask_path = output_dir / "hand_seg.png"
    overlay_path = output_dir / "hand_seg_vis.jpg"

    frame_rgb = load_rgb_image(image_path)
    active_generator = generator or Sam3MaskGenerator(args)
    mask, _kept_masks = active_generator.segment(frame_rgb, args.text_prompt)

    save_mask(mask, mask_path, args.overwrite)
    save_overlay(
        frame_rgb,
        mask,
        overlay_path,
        alpha=args.overlay_alpha,
        mask_color_rgb=args.mask_color_rgb,
        overwrite=args.overwrite,
    )

    return SamMaskOutputs(mask_path=mask_path, overlay_path=overlay_path, mask=mask)


if __name__ == "__main__":
    args = tyro.cli(SamMaskCliArgs)
    process_sam_mask(
        image_path=args.image_path,
        output_dir=args.output_dir,
        args=args.sam_mask,
    )
