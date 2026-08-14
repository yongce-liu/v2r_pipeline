"""Tests for retarget trajectory visualization (offscreen render + npz replay).

The end-to-end CLI tests require real robot assets (relative to the repo root),
so they are skipped when the assets are absent. The pure-replay and offscreen
renderer tests build a tiny synthetic mujoco model instead, so they run anywhere
mujoco is importable.
"""

import numpy as np
import pytest
from pathlib import Path

# Mujoco is a hard dependency of the retarget step (used by GMR), so importing
# it here is fine even when the full robot assets are absent.
import mujoco as mj

from retarget.visualize import (
    load_playback,
    render_video,
    set_frame,
)


def _synthetic_model_xml() -> str:
    """A minimal 3-link arm with a freejoint root, mirroring the real layout.

    Freejoint root (pos + wxyz quat, 7 qpos entries) then two revolute joints
    with actuators — the same structure as the G1's ``pelvis`` freejoint.
    """
    return (
        "<mujoco>"
        "  <option timestep='0.01'/>"
        "  <worldbody>"
        "    <body name='base'>"
        "      <freejoint name='root'/>"
        "      <geom type='sphere' size='0.05'/>"
        "      <body name='link1' pos='0 0 0.2'>"
        "        <joint name='j1' type='hinge' axis='0 1 0'/>"
        "        <geom type='capsule' size='0.02' fromto='0 0 0 0 0 0.2'/>"
        "        <body name='link2' pos='0 0 0.2'>"
        "          <joint name='j2' type='hinge' axis='0 1 0'/>"
        "          <geom type='capsule' size='0.02' fromto='0 0 0 0 0 0.2'/>"
        "        </body>"
        "      </body>"
        "    </body>"
        "  </worldbody>"
        "</mujoco>"
    )


def _synthetic_trajectory(tmp_path, n_frames: int = 4) -> Path:
    """A trajectory.npz matching the CLI output layout, on a synthetic model."""
    model = mj.MjModel.from_xml_string(_synthetic_model_xml())
    assert model.nq == 9  # 7 root + 2 hinge
    qpos = np.zeros((n_frames, model.nq), dtype=np.float64)
    for i in range(n_frames):
        qpos[i, :3] = [0.0, 0.0, 0.5]
        qpos[i, 3:7] = [1.0, 0.0, 0.0, 0.0]
        qpos[i, 7] = float(i) * 0.1
        qpos[i, 8] = -float(i) * 0.1
    path = tmp_path / "trajectory.npz"
    np.savez(path, qpos=qpos)
    return path


def _synthetic_model() -> mj.MjModel:
    return mj.MjModel.from_xml_string(_synthetic_model_xml())


def test_load_playback_loads_qpos(tmp_path) -> None:
    model = _synthetic_model()
    path = _synthetic_trajectory(tmp_path, n_frames=5)
    playback = load_playback(model, path)
    assert playback.qpos.shape == (5, 9)
    assert playback.frame_delay == 1.0 / 30


def test_set_frame_applies_one_frame(tmp_path) -> None:
    model = _synthetic_model()
    path = _synthetic_trajectory(tmp_path, n_frames=3)
    playback = load_playback(model, path)
    set_frame(playback, 2)
    np.testing.assert_allclose(playback.data.qpos[:3], [0.0, 0.0, 0.5])
    np.testing.assert_allclose(playback.data.qpos[7], 0.2)
    np.testing.assert_allclose(playback.data.qpos[8], -0.2)


def test_render_video_writes_mp4(tmp_path) -> None:
    model = _synthetic_model()
    path = _synthetic_trajectory(tmp_path, n_frames=4)
    playback = load_playback(model, path)
    output = tmp_path / "trajectory_vis.mp4"
    render_video(playback, output)
    assert output.is_file()
    assert output.stat().st_size > 0
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(output))
    assert reader.count_frames() == 4
    reader.close()


def test_render_video_rejects_empty_trajectory(tmp_path) -> None:
    model = _synthetic_model()
    empty = tmp_path / "empty.npz"
    np.savez(empty, qpos=np.zeros((0, 9)))
    playback = load_playback(model, empty)
    with pytest.raises(ValueError, match="No frames"):
        render_video(playback, tmp_path / "x.mp4")
