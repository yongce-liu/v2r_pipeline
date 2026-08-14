"""Command-line entry point for the inpaint package.

Two axes, dispatched with ``--command`` and ``--backend``:

- ``command``: ``single`` (one masked frame) or ``video`` (frame-by-frame,
  reading the ``segment`` stage's ``masks.json``).
- ``backend``: ``qwen`` (Qwen-Image-Edit) or ``lama`` (simple-lama big-lama).

Usage:

.. code-block:: bash

    # Single masked frame with Qwen (already painted with the mask color)
    uv run python -m inpaint.cli --command single --backend qwen \
        --single.image-path frame.png --single.output-path out.png \
        --single.qwen.model-path ckpts/Qwen-Image-Edit-2511

    # Full video with Qwen (frame-by-frame), reading segment masks.json
    uv run python -m inpaint.cli --command video --backend qwen \
        --video.masks-json outputs/0/segment/masks.json \
        --video.vis --video.qwen.model-path ckpts/Qwen-Image-Edit-2511

    # Single masked frame with LaMa. The input must be the CLEAN frame + the
    # real binary mask (nonzero pixels are inpainted).
    uv run python -m inpaint.cli --command single --backend lama \
        --single.image-path frame.png --single.output-path out.png \
        --single.mask-path mask.png \
        --single.lama.model-path ckpts/big-lama/big-lama.pt

    # Full video with LaMa (simple-lama TorchScript wrapper)
    uv run python -m inpaint.cli --command video --backend lama \
        --video.masks-json outputs/0/segment/masks.json \
        --video.vis --video.lama.model-path ckpts/big-lama/big-lama.pt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from loguru import logger

from inpaint.lama.args import LamaInpaintArgs
from inpaint.lama.simple import SimpleLamaInpainter
from inpaint.lama.workflow import LamaVideoArgs, run_lama_video_inpaint
from inpaint.qwen.inpainter import QwenInpaintArgs, QwenInpainter
from inpaint.qwen.workflow import InpaintVideoArgs, run_video_inpaint


@dataclass
class SingleImageArgs:
    """Inputs for ``--command single`` (validated only when that mode runs)."""

    image_path: Path | None = None
    output_path: Path | None = None
    mask_path: Path | None = None
    """Binary mask where nonzero pixels (255) mark the region to inpaint.
    Required for ``--backend lama`` (the input must be the clean frame); ignored
    by Qwen (the frame is already painted with the mask color)."""

    qwen: QwenInpaintArgs = field(default_factory=QwenInpaintArgs)
    """Qwen-Image-Edit settings (used when ``--backend qwen``)."""

    lama: LamaInpaintArgs = field(default_factory=LamaInpaintArgs)
    """LaMa settings (used when ``--backend lama``)."""


@dataclass
class InpaintCliArgs:
    """Run hand/arm-removal inpainting (Qwen-Image-Edit or LaMa), single image
    or full video."""

    command: Literal["single", "video"] = "video"
    """One masked image (``single``) or every masked frame of a video
    (``video``, reading the segment stage's masks.json)."""

    backend: Literal["qwen", "lama"] = "lama"
    """Inpainting backend: ``qwen`` (Qwen-Image-Edit) or ``lama`` (big-lama via
    simple-lama-inpainting)."""

    single: SingleImageArgs = field(default_factory=SingleImageArgs)
    """Settings for ``--command single``."""

    video_qwen: InpaintVideoArgs = field(default_factory=InpaintVideoArgs)
    """Qwen settings for ``--command video --backend qwen``."""

    video_lama: LamaVideoArgs = field(default_factory=LamaVideoArgs)
    """LaMa settings for ``--command video --backend lama``."""


def inpaint_single_image(
    image_path: Path,
    output_path: Path,
    backend: str,
    qwen_args: QwenInpaintArgs,
    lama_args: LamaInpaintArgs,
    mask_path: Path | None = None,
) -> Path:
    """Run inpainting on one masked frame with the selected backend."""

    image_path = image_path.expanduser()
    output_path = output_path.expanduser()

    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    if backend == "lama":
        if mask_path is None:
            raise ValueError(
                "--single.mask-path is required for --backend lama: LaMa must "
                "be fed the clean frame + the real binary mask (it zeroes the "
                "hole itself)."
            )
        from inpaint.media import load_mask

        mask = load_mask(mask_path.expanduser())
        inpainter = SimpleLamaInpainter(lama_args)
        inpainter.inpaint(image, output_path, lama_args, mask=mask)
    else:
        inpainter = QwenInpainter(qwen_args)
        inpainter.inpaint(image, output_path, qwen_args)

    return output_path


def main() -> None:
    args = tyro.cli(InpaintCliArgs)

    if args.command == "single":
        if args.single.image_path is None or args.single.output_path is None:
            raise ValueError(
                "--command single requires --single.image-path and --single.output-path."
            )
        output = inpaint_single_image(
            image_path=args.single.image_path,
            output_path=args.single.output_path,
            backend=args.backend,
            qwen_args=args.single.qwen,
            lama_args=args.single.lama,
            mask_path=args.single.mask_path,
        )
        logger.info(
            "[inpaint] single-image {} edit complete: {}",
            args.backend,
            output,
        )
        return

    if args.backend == "lama":
        run_lama_video_inpaint(args.video_lama)
        logger.info(
            "[inpaint] lama video inpainting complete: {}",
            args.video_lama.masks_json,
        )
        return

    run_video_inpaint(args.video_qwen)
    logger.info(
        "[inpaint] qwen video inpainting complete: {}",
        args.video_qwen.masks_json,
    )


if __name__ == "__main__":
    main()
