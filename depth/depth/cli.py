"""Command-line entry point for the depth package.

Two modes, dispatched with ``--command``:

- ``single``: DA3 depth estimation of one image (existing behavior).
- ``video``:  frame-by-frame depth estimation of a whole video, reading the
  ``process`` stage's ``frames.json`` and writing a single aggregate file plus
  per-frame depth maps.

Usage:

.. code-block:: bash

    # Single image
    uv run python -m depth.cli --command single \
        --single.image-path frame.png --single.output-dir out \
        --single.da3.model-path ckpts/depth_anything_v3

    # Full video (frame-by-frame), reading process frames.json
    uv run python -m depth.cli --command video \
        --video.frames-json outputs/0/process/frames.json \
        --video.vis --video.da3.model-path ckpts/depth_anything_v3

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from loguru import logger

from depth.da3 import Da3ImageOutputs, Da3Predictor
from depth.workflow import Da3Args, DepthVideoArgs, run_video_depth


@dataclass
class SingleImageDepthArgs:
    """Inputs for ``--command single`` (validated only when that mode runs)."""

    image_path: Path | None = None
    output_dir: Path | None = None
    da3: Da3Args = field(default_factory=Da3Args)


@dataclass
class DepthCliArgs:
    """Run DA3 depth estimation, single image or full video."""

    command: Literal["single", "video"] = "video"
    """Depth mode: one image (``single``) or a whole video (``video``)."""

    single: SingleImageDepthArgs = field(default_factory=SingleImageDepthArgs)
    """Settings for ``--command single``."""

    video: DepthVideoArgs = field(default_factory=DepthVideoArgs)
    """Settings for ``--command video``."""


def process_depth(
    image_path: Path,
    output_dir: Path,
    args: Da3Args,
    predictor: Da3Predictor | None = None,
) -> Da3ImageOutputs:
    """Run DA3 depth estimation for one image."""

    active_predictor = predictor or Da3Predictor(
        model_path=args.model_path,
        device=args.device,
    )
    return active_predictor.predict_image_depth(
        image_path=image_path,
        output_dir=output_dir,
        process_res=args.process_res,
        overwrite=args.overwrite,
    )


def main() -> None:
    args = tyro.cli(DepthCliArgs)

    if args.command == "single":
        if args.single.image_path is None or args.single.output_dir is None:
            raise ValueError(
                "--command single requires --single.image-path and --single.output-dir."
            )
        outputs = process_depth(
            image_path=args.single.image_path,
            output_dir=args.single.output_dir,
            args=args.single.da3,
        )
        logger.info(
            "[depth] single-image depth complete: depth={}, intrinsics={}",
            outputs.depth_path,
            outputs.intrinsics_path,
        )
        return

    run_video_depth(args.video)
    logger.info("[depth] video depth complete: {}", args.video.frames_json)


if __name__ == "__main__":
    main()
