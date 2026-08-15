"""Visualize a retargeted trajectory in the robot's own mujoco model.

Two complementary modes, both driven by the ``trajectory.npz`` written by the
retarget CLI:

* ``--third-person-vis`` renders the whole trajectory offscreen from a fixed
  third-person camera and writes ``trajectory_vis.mp4`` next to the npz. It
  runs headless (no display, no GLFW), so it can be invoked in any environment.
* ``--mujoco`` opens the interactive mujoco viewer and replays the trajectory
  on a loop until the user closes the window or hits Ctrl-C.
* A ``camera`` entry in the IK config renders per-frame first-person views from
  a camera mounted on any robot body (RGB PNGs always, float32 depth NPYs when
  requested) into separate ``camera_rgb/`` and ``camera_depth/``
  subdirectories. The camera is injected into a throwaway copy of the robot
  model, so the asset XML is never modified.

The replays work directly on the robot mujoco model, so ``--mujoco`` shows the
same robot model that was retargeted. The saved qpos already holds the full
60-dof pose (freejoint root + 59 joint DoFs), so playback only sets
``data.qpos`` and calls ``mj_forward``; physics is not stepped.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Timestamps are dt-frames apart, but the viewer replay rates the motion at a
# fixed fps so the motion is visible regardless of the source video cadence.
PLAYBACK_FPS = 30

CAMERA_NAME = "camera"
"""Name of the camera injected into the camera render model."""

CAMERA_LIGHT_NAME = "camera_light"
"""Name of the world light injected so fixed-camera renders are visible."""

CAMERA_RGB_DIRNAME = "camera_rgb"
"""Subdirectory (under the retarget output dir) holding per-frame RGB PNGs."""

CAMERA_DEPTH_DIRNAME = "camera_depth"
"""Subdirectory (under the retarget output dir) holding per-frame depth NPYs."""


@dataclass
class Playback:
    """Robot model and the per-frame data needed to replay it."""

    model: object
    data: object
    qpos: np.ndarray
    frame_delay: float


def load_playback(model: object, trajectory: Path) -> Playback:
    """Load ``trajectory.npz`` and build the replay state for ``model``.

    The qpos layout (freejoint root + joint DoFs) is derived from the model
    itself rather than assumed, so any retargeted model replays correctly.
    """
    import mujoco as mj

    arrays = np.load(trajectory)
    qpos = arrays["qpos"]
    data = mj.MjData(model)
    return Playback(
        model=model,
        data=data,
        qpos=qpos,
        frame_delay=1.0 / PLAYBACK_FPS,
    )


def set_frame(playback: Playback, frame_index: int) -> None:
    """Apply frame ``frame_index`` of the trajectory to the robot state."""
    import mujoco as mj

    playback.data.qpos[:] = playback.qpos[frame_index]
    mj.mj_forward(playback.model, playback.data)


def _write_video(
    playback: Playback,
    output: Path,
    *,
    fps: int = PLAYBACK_FPS,
    width: int = 1280,
    height: int = 720,
) -> None:
    """Render every trajectory frame offscreen to an mp4."""
    import imageio.v2 as imageio
    import mujoco as mj

    model = playback.model
    data = playback.data
    n_frames = len(playback.qpos)
    if n_frames == 0:
        raise ValueError(f"No frames to render in {output}")

    # The offscreen renderer is capped by the model's framebuffer (MuJoCo raises
    # if the requested size exceeds it), so clamp to the model's buffer.
    width = min(width, model.vis.global_.offwidth)
    height = min(height, model.vis.global_.offheight)

    renderer = mj.Renderer(model, height=height, width=width)
    camera = mj.MjvCamera()
    camera.azimuth = 180
    camera.elevation = -10
    camera.distance = 2.2
    camera.lookat = data.qpos[:3].copy()

    print(f"Rendering {n_frames} frame(s) to {output} ...")
    writer = imageio.get_writer(str(output), fps=fps)
    try:
        for frame_index in range(n_frames):
            set_frame(playback, frame_index)
            camera.lookat = data.qpos[:3].copy()
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()


def render_video(playback: Playback, output: Path) -> None:
    """Render the trajectory offscreen to ``output``."""
    _write_video(playback, output)


def _format_floats(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _inject_world_light(xml: str) -> str:
    """Add a fixed world light so fixed-camera renders are not pitch black."""
    light = (
        f'<light name="{CAMERA_LIGHT_NAME}" pos="0 0 3" dir="0 0 -1" '
        'diffuse="0.9 0.9 0.9" ambient="0.35 0.35 0.35"/>'
    )
    return re.sub(
        r"<worldbody\b[^>]*>",
        lambda match: match.group(0) + light,
        xml,
        count=1,
    )


def _inject_offscreen_buffer(xml: str, width: int, height: int) -> str:
    """Grow the model's offscreen framebuffer to hold ``width`` x ``height``."""
    global_clause = f'<global offwidth="{width}" offheight="{height}"/>'
    # Models that already declare a <visual> keep their own buffer; the
    # renderer clamps to it at render time instead of reordering clauses here.
    if "<visual" in xml:
        return xml
    return xml.replace("</mujoco>", f"<visual>{global_clause}</visual></mujoco>")


def build_camera_model(
    robot_xml: Path,
    *,
    body_name: str,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    width: int = 640,
    height: int = 480,
) -> tuple[object, int]:
    """Load a render copy of ``robot_xml`` with a camera injected.

    The camera is mounted on ``body_name`` with ``pos``/``quat`` (wxyz) offsets
    relative to that body and follows it during replay. A throwaway XML is
    written next to the source so relative mesh paths keep resolving; the
    source asset is never modified.

    Returns ``(model, camera_id)``. The model has the same qpos layout as the
    retarget model (cameras/lights do not change qpos), so any ``trajectory.npz``
    written for the original model replays unchanged.
    """
    import mujoco as mj

    robot_xml = Path(robot_xml)
    if not robot_xml.is_file():
        raise FileNotFoundError(f"robot_xml not found: {robot_xml}")

    xml = robot_xml.read_text(encoding="utf-8")
    body_pattern = re.compile(rf"<body\s+name=[\"']{re.escape(body_name)}[\"'][^>]*>")
    if not body_pattern.search(xml):
        raise ValueError(
            f"body {body_name!r} not found in {robot_xml}; cannot mount the camera"
        )

    camera_tag = (
        f'<camera name="{CAMERA_NAME}" pos="{_format_floats(pos)}" '
        f'quat="{_format_floats(quat)}"/>'
    )
    xml = body_pattern.sub(lambda match: match.group(0) + camera_tag, xml, count=1)
    xml = _inject_world_light(xml)
    xml = _inject_offscreen_buffer(xml, width, height)

    fd, tmp_name = tempfile.mkstemp(
        prefix=".camera_", suffix=".xml", dir=str(robot_xml.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(xml)
        model = mj.MjModel.from_xml_path(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    camera_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    return model, int(camera_id)


def render_camera_frames(
    playback: Playback,
    camera_id: int,
    rgb_dir: Path,
    *,
    depth_dir: Path | None = None,
    width: int = 640,
    height: int = 480,
) -> list[dict] | None:
    """Render first-person camera RGB PNGs (and optional depth) per frame.

    ``playback`` must be built from the camera-injected model (see
    :func:`build_camera_model`). RGB frames go to ``rgb_dir`` as
    ``000000.png``-style images. When ``depth_dir`` is given, depth frames are
    written there as float32 ``.npy`` arrays (linear meters, far-plane distance
    where nothing was hit).

    Returns a per-frame depth stats list (``index``, ``depth_filename``,
    ``depth_min``, ``depth_max``, ``depth_mean``) when ``depth_dir`` is given,
    otherwise ``None``.
    """
    import imageio.v2 as imageio
    import mujoco as mj

    rgb_dir.mkdir(parents=True, exist_ok=True)
    if depth_dir is not None:
        depth_dir.mkdir(parents=True, exist_ok=True)

    model = playback.model
    n_frames = len(playback.qpos)
    if n_frames == 0:
        raise ValueError("No frames to render for the camera")

    # Clamp to the model's framebuffer (MuJoCo raises if the requested size
    # exceeds it) when the source XML already declared its own <visual>.
    width = min(width, model.vis.global_.offwidth)
    height = min(height, model.vis.global_.offheight)

    renderer = mj.Renderer(model, height=height, width=width)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = camera_id

    suffix = f", {depth_dir}" if depth_dir is not None else ""
    print(f"Rendering {n_frames} camera frame(s) -> {rgb_dir}{suffix}")
    depth_stats: list[dict] = []
    try:
        for frame_index in range(n_frames):
            set_frame(playback, frame_index)
            renderer.update_scene(playback.data, camera=camera)
            imageio.imwrite(rgb_dir / f"{frame_index:06d}.png", renderer.render())
            if depth_dir is not None:
                renderer.enable_depth_rendering()
                depth = renderer.render().astype(np.float32)
                renderer.disable_depth_rendering()
                depth_filename = f"{frame_index:06d}.npy"
                np.save(depth_dir / depth_filename, depth)
                depth_stats.append(
                    {
                        "index": frame_index,
                        "depth_filename": depth_filename,
                        "depth_min": float(depth.min()),
                        "depth_max": float(depth.max()),
                        "depth_mean": float(depth.mean()),
                    }
                )
    finally:
        renderer.close()
    return depth_stats if depth_dir is not None else None


def launch_viewer(playback: Playback) -> None:
    """Replay the trajectory in an interactive mujoco viewer, looping forever.

    Exits when the user closes the viewer window or interrupts with Ctrl-C.
    """
    import mujoco.viewer as mjv

    model = playback.model
    data = playback.data

    print("Press Ctrl-C or close the viewer window to stop replay.")
    viewer = mjv.launch_passive(
        model=model,
        data=data,
        show_left_ui=True,
        show_right_ui=False,
    )
    # Third-person framing: the freejoint root (first 3 qpos entries) is the
    # robot's base position, which the camera follows at a fixed distance.
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -10
    viewer.cam.distance = 2.2
    viewer.cam.lookat = data.qpos[:3].copy()

    n_frames = len(playback.qpos)
    try:
        while viewer.is_running():
            for frame_index in range(n_frames):
                if not viewer.is_running():
                    break
                set_frame(playback, frame_index)
                viewer.cam.lookat = data.qpos[:3].copy()
                viewer.sync()
                time.sleep(playback.frame_delay)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
