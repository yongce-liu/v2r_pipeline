"""Visualize a retargeted trajectory in the robot's own mujoco model.

Two complementary modes, both driven by the ``trajectory.npz`` written by the
retarget CLI:

* ``--vis``  renders the whole trajectory offscreen from a fixed third-person
  camera and writes ``trajectory_vis.mp4`` next to the npz. It runs headless
  (no display, no GLFW), so it can be invoked in any environment.
* ``--mujoco`` opens the interactive mujoco viewer and replays the trajectory
  on a loop until the user closes the window or hits Ctrl-C.

The replays work directly on the robot mujoco model, so ``--mujoco`` shows the
same robot model that was retargeted. The saved qpos already holds the full
60-dof pose (freejoint root + 59 joint DoFs), so playback only sets
``data.qpos`` and calls ``mj_forward``; physics is not stepped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Timestamps are dt-frames apart, but the viewer replay rates the motion at a
# fixed fps so the motion is visible regardless of the source video cadence.
PLAYBACK_FPS = 30


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
