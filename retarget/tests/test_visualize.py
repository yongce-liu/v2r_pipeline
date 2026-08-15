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
    CAMERA_DEPTH_DIRNAME,
    CAMERA_NAME,
    CAMERA_RGB_DIRNAME,
    build_camera_model,
    load_playback,
    render_camera_frames,
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


def _write_synthetic_xml(tmp_path: Path) -> Path:
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text(_synthetic_model_xml())
    return xml_path


def test_build_camera_model_injects_camera(tmp_path) -> None:
    xml_path = _write_synthetic_xml(tmp_path)
    model, camera_id = build_camera_model(
        xml_path, body_name="link2", width=320, height=240
    )
    assert model.ncam == 1
    assert mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, camera_id) == CAMERA_NAME
    body_id = model.cam_bodyid[camera_id]
    assert mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) == "link2"
    assert model.vis.global_.offwidth == 320
    assert model.vis.global_.offheight == 240
    # The camera must not change the qpos layout used by the retarget npz.
    assert model.nq == mj.MjModel.from_xml_string(_synthetic_model_xml()).nq


def test_build_camera_model_rejects_missing_body(tmp_path) -> None:
    xml_path = _write_synthetic_xml(tmp_path)
    with pytest.raises(ValueError, match="missing_body"):
        build_camera_model(xml_path, body_name="missing_body")


def test_render_camera_frames_writes_rgb_and_depth(tmp_path) -> None:
    path = _synthetic_trajectory(tmp_path, n_frames=4)

    xml_path = _write_synthetic_xml(tmp_path)
    camera_model, camera_id = build_camera_model(
        xml_path, body_name="link2", width=160, height=128
    )
    camera_playback = load_playback(camera_model, path)
    rgb_dir = tmp_path / CAMERA_RGB_DIRNAME
    depth_dir = tmp_path / CAMERA_DEPTH_DIRNAME
    depth_stats = render_camera_frames(
        camera_playback,
        camera_id,
        rgb_dir,
        depth_dir=depth_dir,
        width=160,
        height=120,
    )

    expected_names = [f"{index:06d}.png" for index in range(4)]
    assert sorted(path.name for path in rgb_dir.glob("*.png")) == expected_names
    assert sorted(path.name for path in depth_dir.glob("*.npy")) == [
        name.replace(".png", ".npy") for name in expected_names
    ]
    import imageio.v2 as imageio

    rgb = imageio.imread(rgb_dir / "000000.png")
    assert rgb.shape == (120, 160, 3)
    depth = np.load(depth_dir / "000000.npy")
    assert depth.shape == (120, 160)
    assert depth.dtype == np.float32
    assert len(depth_stats) == 4
    assert depth_stats[0]["depth_filename"] == "000000.npy"
    assert depth_stats[0]["depth_min"] <= depth_stats[0]["depth_mean"]
    assert depth_stats[0]["depth_mean"] <= depth_stats[0]["depth_max"]


def test_render_camera_frames_skips_depth_by_default(tmp_path) -> None:
    """Without depth_dir only RGB PNGs are written and no stats are returned."""
    path = _synthetic_trajectory(tmp_path, n_frames=2)

    xml_path = _write_synthetic_xml(tmp_path)
    camera_model, camera_id = build_camera_model(
        xml_path, body_name="link2", width=160, height=120
    )
    camera_playback = load_playback(camera_model, path)
    rgb_dir = tmp_path / CAMERA_RGB_DIRNAME
    depth_stats = render_camera_frames(
        camera_playback, camera_id, rgb_dir, width=160, height=120
    )

    assert len(list(rgb_dir.glob("*.png"))) == 2
    assert not (tmp_path / CAMERA_DEPTH_DIRNAME).exists()
    assert depth_stats is None


def test_render_camera_frames_rejects_empty_trajectory(tmp_path) -> None:
    empty = tmp_path / "empty.npz"
    np.savez(empty, qpos=np.zeros((0, 9)))
    xml_path = _write_synthetic_xml(tmp_path)
    camera_model, camera_id = build_camera_model(
        xml_path, body_name="link2", width=160, height=120
    )
    camera_playback = load_playback(camera_model, empty)
    with pytest.raises(ValueError, match="No frames"):
        render_camera_frames(
            camera_playback,
            camera_id,
            tmp_path / "rgb",
            depth_dir=tmp_path / "depth",
            width=160,
            height=120,
        )
