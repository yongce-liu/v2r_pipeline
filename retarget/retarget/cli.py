"""CLI for the retarget step: raw human data -> robot qpos/action trajectory.

The flow:

  1. read the IK config and pop its ``source_type`` (which loader reads input);
  2. dispatch the raw human calibration file to that source's dataloader, which
     decodes it into per-frame human/hand data;
  3. retarget every frame with :class:`RobotRetargeter` using the IK config;
  4. write ``trajectory.npz`` under ``--output-dir``, or when omitted under
     ``outputs/<input-stem>/retarget/``.

Optional visualization after writing:

  * ``--third-person-vis`` renders a third-person video of the retargeted
    motion to ``trajectory_vis.mp4`` next to the npz (offscreen, headless-safe);
  * ``--mujoco`` replays the motion in the interactive mujoco viewer, looping
    until the window is closed or the process is interrupted (Ctrl-C);
  * a ``camera`` entry in the IK config renders per-frame views from a camera
    mounted on a robot body (see :func:`load_camera_config` for the schema).

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
    ``outputs/<input-stem>/process/frames.json``. When the default file is
    absent, each source frame's own timestamp is used instead."""

    third_person_vis: bool = False
    """Render a third-person video of the retargeted motion to
    ``trajectory_vis.mp4`` next to the npz."""

    mujoco: bool = False
    """Replay the retargeted motion in the interactive mujoco viewer, looping
    until the window is closed or the process is interrupted (Ctrl-C)."""


TRAJECTORY_FILENAME = "trajectory.npz"
"""Trajectory filename inside the per-episode ``retarget/`` dir, fixed in code."""

VIS_FILENAME = "trajectory_vis.mp4"
"""Third-person video filename next to ``trajectory.npz``."""


def load_camera_config(ik_config: dict) -> dict | None:
    """Normalize the optional ``camera`` entry of an IK config.

    The camera schema describes where to mount a render camera on the robot
    and which outputs to produce::

        "camera": {
          "link": "d435_link",        # robot body the camera follows
          "pos_offset": [0, 0, 0],    # position offset (x, y, z) in meters
          "orn_offset": [1, 0, 0, 0], # orientation offset (w, x, y, z)
          "width": 1280,              # optional render width (inferred from
          "height": 720,              #   source video when omitted)
          "depth": true               # optional: also write depth NPYs + manifest
        }

    Returns ``None`` when no camera is configured, otherwise a normalized dict
    with keys ``link``, ``pos_offset``, ``orn_offset``, ``width``, ``height``,
    and ``depth``.
    """
    camera = ik_config.get("camera")
    if not camera:
        return None
    if not isinstance(camera, dict):
        raise ValueError("ik config 'camera' must be a dict")

    link = camera.get("link")
    if not link:
        raise ValueError("ik config 'camera' requires a 'link' to mount on")

    pos_offset = tuple(camera.get("pos_offset", (0.0, 0.0, 0.0)))
    if len(pos_offset) != 3:
        raise ValueError("'camera.pos_offset' must have 3 elements (x, y, z)")
    orn_offset = tuple(camera.get("orn_offset", (1.0, 0.0, 0.0, 0.0)))
    if len(orn_offset) != 4:
        raise ValueError("'camera.orn_offset' must have 4 elements (w, x, y, z)")

    return {
        "link": str(link),
        "pos_offset": tuple(float(value) for value in pos_offset),
        "orn_offset": tuple(float(value) for value in orn_offset),
        "width": int(camera["width"]) if camera.get("width") is not None else None,
        "height": int(camera["height"]) if camera.get("height") is not None else None,
        "depth": bool(camera.get("depth", False)),
    }


CAMERA_JSON_FILENAME = "camera.json"
"""Per-frame camera manifest filename inside the retarget output dir."""

CAMERA_FOVY_DEG = 45.0
"""Camera vertical field of view used by the injected camera (MuJoCo default)."""

CAMERA_DEFAULT_WIDTH = 640
"""Fallback render width when the camera config omits it and the source video
resolution cannot be probed."""

CAMERA_DEFAULT_HEIGHT = 480
"""Fallback render height when the camera config omits it and the source video
resolution cannot be probed."""


def probe_video_resolution(video_path: Path) -> tuple[int, int]:
    """Probe a video file's pixel resolution with ffprobe."""
    import json
    import shutil
    import subprocess

    if not video_path.is_file():
        raise FileNotFoundError(f"source video not found: {video_path}")
    if shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffprobe executable not found on PATH")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise ValueError(f"no video stream found in {video_path}")
    width = int(streams[0]["width"])
    height = int(streams[0]["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid resolution {width}x{height} from {video_path}")
    return width, height


def resolve_camera_resolution(
    camera_cfg: dict,
    frames_json: Path | None,
) -> tuple[int, int]:
    """Pick the camera render resolution.

    Explicit ``width``/``height`` in the camera config win. Otherwise the
    resolution is inferred from the ``source_video`` entry of the process
    ``frames.json`` via ffprobe; when that is unavailable the built-in default
    is used.
    """
    width = camera_cfg.get("width")
    height = camera_cfg.get("height")
    if width is not None and height is not None:
        return int(width), int(height)

    if frames_json is not None and frames_json.is_file():
        try:
            import json

            data = json.loads(Path(frames_json).read_text(encoding="utf-8"))
            source_video = data.get("source_video")
            if source_video:
                video = Path(source_video)
                if not video.is_file():
                    video = Path.cwd() / video
                width, height = probe_video_resolution(video)
                print(f"Inferred camera resolution {width}x{height} from {video}")
                return width, height
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"Could not infer camera resolution from {frames_json}: {exc}")

    print(
        "Falling back to default camera resolution "
        f"{CAMERA_DEFAULT_WIDTH}x{CAMERA_DEFAULT_HEIGHT}"
    )
    return CAMERA_DEFAULT_WIDTH, CAMERA_DEFAULT_HEIGHT


def camera_intrinsics(width: int, height: int) -> list[list[float]]:
    """Square-pixel 3x3 intrinsics matrix for the injected camera."""
    focal = (height / 2.0) / np.tan(np.radians(CAMERA_FOVY_DEG) / 2.0)
    return [
        [float(focal), 0.0, width / 2.0],
        [0.0, float(focal), height / 2.0],
        [0.0, 0.0, 1.0],
    ]


def write_camera_manifest(
    output_dir: Path,
    camera_cfg: dict,
    timestamps: np.ndarray,
    depth_stats: list[dict] | None,
) -> Path:
    """Write ``camera.json`` describing the rendered camera frames.

    The manifest records the camera mount/intrinsics and one entry per frame
    (RGB filename, optional depth filename, timestamp). Depth stats come from
    :func:`retarget.visualize.render_camera_frames`.
    """
    import json

    frame_count = len(timestamps)
    depth_enabled = depth_stats is not None
    entries = []
    for index in range(frame_count):
        entries.append(
            {
                "index": index,
                "rgb_filename": f"{index:06d}.png",
                "depth_filename": (
                    depth_stats[index]["depth_filename"] if depth_enabled else None
                ),
                "timestamp_sec": float(timestamps[index]),
                "depth_min": depth_stats[index]["depth_min"] if depth_enabled else None,
                "depth_max": depth_stats[index]["depth_max"] if depth_enabled else None,
                "depth_mean": depth_stats[index]["depth_mean"]
                if depth_enabled
                else None,
            }
        )

    width = camera_cfg["width"]
    height = camera_cfg["height"]
    manifest = {
        "schema_version": "1.0",
        "stage": "retarget_camera",
        "camera": {
            "name": "camera",
            "link": camera_cfg["link"],
            "pos_offset": list(camera_cfg["pos_offset"]),
            "orn_offset": list(camera_cfg["orn_offset"]),
            "width": width,
            "height": height,
            "fovy_deg": CAMERA_FOVY_DEG,
            "intrinsics": camera_intrinsics(width, height),
        },
        "frame_count": frame_count,
        "rgb_dir": "camera_rgb",
        "depth_dir": "camera_depth" if depth_enabled else None,
        "depth_enabled": depth_enabled,
        "entries": entries,
    }
    path = output_dir / CAMERA_JSON_FILENAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


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
    if frames_json.is_file():
        frame_timestamps = load_frame_timestamps(frames_json)
    elif args.frames_json is not None:
        raise FileNotFoundError(f"frames-json not found: {frames_json}")
    else:
        # No process-step manifest yet: fall back to each source frame's own
        # timestamp (EgoDex frames carry index/fps), so retarget can run before
        # the video-ingest steps on a new episode.
        print(f"frames.json not found at {frames_json}; using source frame timestamps")
        frame_timestamps = None

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

    if args.third_person_vis or args.mujoco:
        from retarget.visualize import (
            load_playback,
            launch_viewer,
            render_video,
        )

        model = retargeter.gmr.model
        playback = load_playback(model, output)
        if args.third_person_vis:
            render_video(playback, output_dir / VIS_FILENAME)
        if args.mujoco:
            launch_viewer(playback)

    camera_cfg = load_camera_config(ik_config_data)
    if camera_cfg is not None:
        from retarget.visualize import (
            CAMERA_DEPTH_DIRNAME,
            CAMERA_RGB_DIRNAME,
            build_camera_model,
            load_playback,
            render_camera_frames,
        )

        camera_cfg["width"], camera_cfg["height"] = resolve_camera_resolution(
            camera_cfg, frames_json
        )
        camera_model, camera_id = build_camera_model(
            robot_xml,
            body_name=camera_cfg["link"],
            pos=camera_cfg["pos_offset"],
            quat=camera_cfg["orn_offset"],
            width=camera_cfg["width"],
            height=camera_cfg["height"],
        )
        camera_playback = load_playback(camera_model, output)
        depth_stats = render_camera_frames(
            camera_playback,
            camera_id,
            output_dir / CAMERA_RGB_DIRNAME,
            depth_dir=(
                output_dir / CAMERA_DEPTH_DIRNAME if camera_cfg["depth"] else None
            ),
            width=camera_cfg["width"],
            height=camera_cfg["height"],
        )
        manifest_path = write_camera_manifest(
            output_dir, camera_cfg, timestamps, depth_stats
        )
        print(f"Wrote camera manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
