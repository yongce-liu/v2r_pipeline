"""Command-line entry point for the composite step.

Reads the ``inpaint`` background, the DA3 depth of the inpainted frames (plus
optionally the DA3 depth of the original frames for calibration) and the
``retarget`` robot camera renders, then writes depth-aware composites.

Usage:

.. code-block:: bash

    uv run python -m composite.cli \
        --inpainted-json outputs/0/inpaint/inpainted.json \
        --depth-json outputs/0/depth/depth.json \
        --camera-json outputs/0/retarget/camera.json \
        --calibration-depth-json outputs/0/depth_orig/depth.json \
        --video

For the calibration input, first run the depth stage on the original frames:

.. code-block:: bash

    (cd depth && uv run python -m depth.cli --video.frames-json \
        ../outputs/0/process/frames.json --video.output-root ../outputs/**/depth_orig)
"""

from __future__ import annotations

import tyro

from composite.workflow import CompositeArgs, run_composite


def main() -> None:
    args = tyro.cli(CompositeArgs)
    outputs = run_composite(args)
    print(outputs.stage_dir)


if __name__ == "__main__":
    main()
