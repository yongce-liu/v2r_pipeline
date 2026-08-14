"""CLI for the first-person depth-match rendering step.

Reads the per-episode outputs of the earlier stages (segment masks, depth +
intrinsics, process frame manifest, retarget trajectory), solves a first-person
camera per frame so the robot arm covers the human arm, and writes
transparent-background RGBA frames to ``outputs/<stem>/first_person/frames/``.

Run from the repo root (asset paths are repo-root-relative):

    uv run recamera --input outputs/0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


from recamera.inputs import load_episode
from recamera.robot import build_model
from recamera.workflow import RenderConfig, process_episode


@dataclass
class RecameraArgs:
    """Inputs for one first-person depth-match render run."""

    input: Path
    """Episode dir under ``output_root`` (e.g. ``outputs/0``)."""

    output_root: Path | None = None
    """Root of the episode outputs. Defaults to the parent of ``input``."""

    output_dir: Path | None = None
    """Output dir for the first-person frames. Defaults to
    ``output_root/<stem>/first_person``."""

    robot_xml: Path | None = None
    """Robot MJCF used for rendering. Defaults to the retarget model at
    ``assets/unitree_g1_mjcf/g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml``."""

    cloud_stride: int = 1
    """Backprojection stride on the human arm mask (1 = full res)."""

    iou_render_width: int = 320
    """Render width used for the silhouette IoU refinement (reduced res, faster)."""

    device: str = "auto"
    """Unused placeholder for future GPU offload; kept for CLI parity."""


def main(args: RecameraArgs | None = None) -> None:
    import tyro

    if args is None:
        args = tyro.cli(RecameraArgs)

    cwd = Path.cwd()
    input_path = args.input
    if not input_path.is_absolute():
        input_path = cwd / input_path

    output_root = args.output_root or input_path.parent
    ep = load_episode(output_root, input_path.name)

    robot_xml = args.robot_xml or cwd / (
        "assets/unitree_g1_mjcf/g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml"
    )
    model = build_model(robot_xml, ep.width, ep.height)

    output_dir = args.output_dir or (output_root / input_path.name / "first_person")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = RenderConfig(
        cloud_stride=args.cloud_stride,
        iou_render_width=args.iou_render_width,
    )
    print(
        f"recamera: {ep.frame_count} frames, {ep.width}x{ep.height}, "
        f"{ep.fps:.0f} fps -> {output_dir}"
    )
    n = process_episode(model, ep, output_dir, cfg)
    print(f"Wrote {n} transparent frame(s) to {output_dir / 'frames'}")


if __name__ == "__main__":
    main()
