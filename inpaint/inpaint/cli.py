"""Command-line entry point for the inpaint package.

Two modes, dispatched with ``--command``:

- ``single``: Qwen-Image-Edit editing of one already-masked frame.
- ``video``:  frame-by-frame arm-removal inpainting of a whole video, reading the
  ``segment`` stage's ``masks.json``.

Usage:

.. code-block:: bash

    # Single image (already painted with the mask color)
    uv run python -m inpaint.cli --command single \
        --single.image-path frame.png --single.output-path out.png \
        --single.qwen.model-path ckpts/Qwen-Image-Edit-2511

    # Full video (frame-by-frame), reading segment masks.json
    uv run python -m inpaint.cli --command video \
        --video.masks-json outputs/0/segment/masks.json \
        --video.vis --video.qwen.model-path ckpts/Qwen-Image-Edit-2511
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from loguru import logger

from inpaint.qwen_inpaint import QwenInpaintArgs, QwenInpainter
from inpaint.workflow import InpaintVideoArgs, run_video_inpaint


@dataclass
class SingleImageArgs:
    """Inputs for ``--command single`` (validated only when that mode runs)."""

    image_path: Path | None = None
    output_path: Path | None = None
    qwen: QwenInpaintArgs = field(default_factory=QwenInpaintArgs)


@dataclass
class InpaintCliArgs:
    """Run Qwen-Image-Edit arm-removal inpainting, single image or full video."""

    command: Literal["single", "video"] = "video"
    """Inpaint mode: one image (``single``) or a whole video (``video``)."""

    single: SingleImageArgs = field(default_factory=SingleImageArgs)
    """Settings for ``--command single``."""

    video: InpaintVideoArgs = field(default_factory=InpaintVideoArgs)
    """Settings for ``--command video``."""


def inpaint_single_image(
    image_path: Path,
    output_path: Path,
    args: QwenInpaintArgs,
    inpainter: QwenInpainter | None = None,
) -> Path:
    """Run Qwen image editing on one masked frame."""

    image_path = image_path.expanduser()
    output_path = output_path.expanduser()

    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    active_inpainter = inpainter or QwenInpainter(args)
    active_inpainter.inpaint(image, output_path, args)
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
            args=args.single.qwen,
        )
        logger.info("[inpaint] single-image edit complete: {}", output)
        return

    run_video_inpaint(args.video)
    logger.info("[inpaint] video inpainting complete: {}", args.video.masks_json)


if __name__ == "__main__":
    main()
