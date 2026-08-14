"""CLI for the retarget step: raw human data -> robot qpos/action trajectory.

The flow:

  1. read the IK config and pop its ``source_type`` (which loader reads input);
  2. dispatch the raw human calibration file to that source's dataloader, which
     decodes it into per-frame human/hand data;
  3. retarget every frame with :class:`RobotRetargeter` using the IK config;
  4. write ``trajectory.npz`` under ``--output-dir``, or when omitted under
     ``outputs/<input-stem>/retarget/``.

Optional visualization after writing:

  * ``--vis`` renders a third-person video of the retargeted motion to
    ``trajectory_vis.mp4`` next to the npz (offscreen, headless-safe);
  * ``--mujoco`` replays the motion in the interactive mujoco viewer, looping
    until the window is closed or the process is interrupted (Ctrl-C).

Frame timestamps come from the process step's frame manifest
``outputs/<input-stem>/process/frames.json`` (override with ``--frames-json``),
not from a fixed decode fps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from retarget.loaders import Frame, iter_source_frames


@dataclass
class RetargetArgs:
    """Inputs for one retarget run."""

    input: Path
    """Raw human calibration file (hdf5 for source_type=egodex)."""

    ik_config: Path
    """GMR IK config defining the human-to-robot matching tables, source_type,
    and robot_xml."""

    robot_xml: Path | None = None
    """MJCF robot model used by GMR. Defaults to the ikconfig's ``robot_xml``."""

    confidence_threshold: float = 0.5
    """Keypoint confidence below which previous-frame values are used."""

    output_dir: Path | None = None
    """Directory holding the fixed ``trajectory.npz`` output. Defaults to
    ``outputs/<input-stem>/retarget``."""

    frames_json: Path | None = None
    """Process-step frame manifest supplying output timestamps. Defaults to
    ``outputs/<input-stem>/process/frames.json``."""

    vis: bool = False
    """Render a third-person video of the retargeted motion to
    ``trajectory_vis.mp4`` next to the npz."""

    mujoco: bool = False
    """Replay the retargeted motion in the interactive mujoco viewer, looping
    until the window is closed or the process is interrupted (Ctrl-C)."""


TRAJECTORY_FILENAME = "trajectory.npz"
"""Trajectory filename inside the per-episode ``retarget/`` dir, fixed in code."""

VIS_FILENAME = "trajectory_vis.mp4"
"""Rendered video filename next to ``trajectory.npz``, written by ``--vis``."""


def load_source_type(ik_config: Path) -> str:
    """Read ``source_type`` from the IK config, defaulting to ``egodex``."""
    from retarget import load_ik_config

    return load_ik_config(ik_config).pop("source_type", "egodex")


def load_frame_timestamps(frames_json: Path) -> np.ndarray:
    """Frame timestamps from a process/ ``frames.json`` manifest, in frame order.

    Entries whose ``timestamp_sec`` is null fall back to their running index.
    """
    import json

    data = json.loads(Path(frames_json).read_text())
    entries = sorted(data["entries"], key=lambda entry: entry["index"])
    timestamps = [
        float(index)
        if entry.get("timestamp_sec") is None
        else float(entry["timestamp_sec"])
        for index, entry in enumerate(entries)
    ]
    return np.asarray(timestamps, dtype=np.float64)


def retarget_frames(
    retargeter,
    frames: Iterable[Frame],
    *,
    frame_timestamps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retarget an iterable of frames into qpos/action/timestamps arrays.

    When ``frame_timestamps`` is given (from ``frames.json``) its values are used
    in frame order; otherwise each frame's own ``timestamp_s`` (or its index) is
    the fallback.
    """
    qpos_all: list[np.ndarray] = []
    action_all: list[np.ndarray] = []
    timestamps: list[float] = []
    for index, frame in enumerate(frames):
        result = retargeter.retarget(frame)
        qpos_all.append(result.qpos.astype(np.float64, copy=False))
        action_all.append(result.action.astype(np.float32, copy=False))
        if frame_timestamps is not None and index < len(frame_timestamps):
            timestamps.append(float(frame_timestamps[index]))
        else:
            timestamps.append(float(getattr(frame, "timestamp_s", index)))
    if not qpos_all:
        raise RuntimeError("No frames retargeted from the input file")
    return (
        np.stack(qpos_all),
        np.stack(action_all),
        np.asarray(timestamps, dtype=np.float64),
    )


def main(args: RetargetArgs | None = None) -> None:
    import tyro

    if args is None:
        args = tyro.cli(RetargetArgs)

    from retarget import RobotRetargeter, load_ik_config

    cwd = Path.cwd()
    ik_config_path = args.ik_config
    ik_config_data = load_ik_config(ik_config_path)
    source_type = ik_config_data.pop("source_type", "egodex")

    robot_xml = args.robot_xml or ik_config_data.get("robot_xml")
    if robot_xml is None:
        raise ValueError(
            "robot_xml must be given as an arg or set in the ik config "
            "(ikconfig['robot_xml'])"
        )
    robot_xml = cwd / robot_xml

    retargeter = RobotRetargeter(
        robot_xml,
        ik_config_path,
        confidence_threshold=args.confidence_threshold,
    )
    print(f"source_type={source_type}, hand backend: {retargeter.hand_backend}")

    frames_json = args.frames_json or (
        cwd / f"outputs/{Path(args.input).stem}/process/frames.json"
    )
    frame_timestamps = load_frame_timestamps(frames_json)

    frames = iter_source_frames(
        source_type,
        args.input,
        ik_config_data,
        confidence_threshold=args.confidence_threshold,
    )
    qpos, action, timestamps = retarget_frames(
        retargeter, frames, frame_timestamps=frame_timestamps
    )

    output_dir = args.output_dir or (cwd / f"outputs/{Path(args.input).stem}/retarget")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / TRAJECTORY_FILENAME
    np.savez_compressed(
        output,
        qpos=qpos,
        action=action,
        timestamps=timestamps,
        action_names=np.asarray(retargeter.action_names),
    )
    print(f"Wrote {len(qpos)} frame(s), action shape {action.shape} -> {output}")

    if args.vis or args.mujoco:
        from retarget.visualize import (
            load_playback,
            launch_viewer,
            render_video,
        )

        model = retargeter.gmr.model
        playback = load_playback(model, output)
        if args.vis:
            render_video(playback, output_dir / VIS_FILENAME)
        if args.mujoco:
            launch_viewer(playback)


if __name__ == "__main__":
    main()
