"""CLI tests for the retarget step.

The end-to-end test requires real robot assets (relative to the repo root), so
it is skipped when the assets are absent. The argument-parsing, help,
source-type dispatch, and timestamp-from-frames.json tests run anywhere.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PKG_ROOT.parent
_ASSET_XML = (
    _REPO_ROOT
    / "assets"
    / "unitree_g1_mjcf"
    / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml"
)
_IK_CONFIG = _REPO_ROOT / "configs" / "egodex_UnitreeG1InspireDfq.json"


def _write_egodex_hdf5(path: Path, n_frames: int = 3) -> None:
    """A tiny EgoDex episode covering every name the real IK config needs."""
    import h5py

    from retarget import load_ik_config
    from retarget.egodex import loader_targets_from_ik_config

    config = load_ik_config(_IK_CONFIG)
    body_names, _hand_points, _mano = loader_targets_from_ik_config(config)
    mano = config["dex_hand_config"]["mano_keypoint_names"]
    names = set(body_names)
    for side in ("left", "right"):
        for joint in mano:
            names.add(f"{side}{joint}")
    with h5py.File(path, "w") as f:
        for name in sorted(names):
            f.create_dataset(
                f"transforms/{name}", data=np.tile(np.eye(4), (n_frames, 1, 1))
            )
            f.create_dataset(f"confidences/{name}", data=np.ones(n_frames))
        f.attrs["task_description"] = "test task"


def _write_frames_json(path: Path, n_frames: int, start_s: float = 0.0) -> None:
    """A process-step manifest (FrameManifest.to_dict layout)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source_video": "x.mp4",
                "fps": 30.0,
                "width": None,
                "height": None,
                "format": "png",
                "frame_count": n_frames,
                "frames_dir": str(path.parent / "frames"),
                "entries": [
                    {
                        "index": i,
                        "frame_filename": f"frame_{i:06d}.png",
                        "timestamp_sec": start_s + i / 30.0,
                    }
                    for i in range(n_frames)
                ],
                "ffmpeg_version": "x",
            }
        )
    )


def test_tyro_args_parse() -> None:
    """RetargetArgs carries the flow inputs with sane defaults."""
    from retarget.cli import RetargetArgs

    args = RetargetArgs(input=Path("0.hdf5"), ik_config=Path("ik.json"))
    assert args.confidence_threshold == 0.5
    assert args.robot_xml is None  # falls back to ikconfig's robot_xml
    assert args.output_dir is None
    assert args.frames_json is None
    assert args.third_person_vis is False
    assert args.mujoco is False


def test_vis_and_mujoco_flags_parse() -> None:
    """--third-person-vis and --mujoco parse and appear in --help; the old
    head-camera args are gone."""
    result = subprocess.run(
        [sys.executable, "-m", "retarget.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--third-person-vis" in result.stdout
    assert "--mujoco" in result.stdout
    assert "--head-camera" not in result.stdout


def test_cli_help_runs() -> None:
    """The console entry point exposes --help without importing robot code."""
    result = subprocess.run(
        [sys.executable, "-m", "retarget.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--robot-xml" in result.stdout
    assert "--ik-config" in result.stdout
    assert "--confidence-threshold" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--frames-json" in result.stdout
    assert "--fps" not in result.stdout
    assert "--max-frames" not in result.stdout


def test_cli_runs_end_to_end(tmp_path: Path) -> None:
    """Running the CLI on an egodex hdf5 produces the fixed trajectory.npz."""
    if not _ASSET_XML.exists():
        pytest.skip("robot assets not present; skipping end-to-end CLI test")

    hdf5 = tmp_path / "in" / "task" / "0.hdf5"
    hdf5.parent.mkdir(parents=True)
    _write_egodex_hdf5(hdf5, n_frames=3)
    frames_json = tmp_path / "outputs" / "0" / "process" / "frames.json"
    _write_frames_json(frames_json, n_frames=3, start_s=0.5)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "retarget",
            "--input",
            str(hdf5),
            "--ik-config",
            str(_IK_CONFIG),
            "--output-dir",
            str(tmp_path / "out"),
            "--frames-json",
            str(frames_json),
        ],
        cwd=_REPO_ROOT,  # dex hand urdf_dir + ikconfig robot_xml are repo-root-relative
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "source_type=egodex" in result.stdout

    output = tmp_path / "out" / "trajectory.npz"
    assert output.is_file()
    data = np.load(output)
    assert set(data.files) == {"qpos", "action", "timestamps", "action_names"}
    assert data["qpos"].shape == (3, 60)
    assert data["action"].shape == (3, 53)  # G1 + 24 inspire finger joints
    # The dex hand joints are folded into qpos, so qpos is the complete robot
    # pose: every hand joint's qpos column equals its action column.
    import mujoco as mj

    model = mj.MjModel.from_xml_path(str(_ASSET_XML))
    names = [str(name) for name in data["action_names"]]
    for column, name in enumerate(names):
        if not name.startswith(("L_", "R_")):
            continue
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        qpos_column = model.jnt_qposadr[joint_id]
        np.testing.assert_allclose(
            data["qpos"][:, qpos_column], data["action"][:, column]
        )
    # timestamps come from frames.json, not frame indices.
    np.testing.assert_allclose(data["timestamps"], [0.5, 0.5 + 1 / 30, 0.5 + 2 / 30])


def test_output_dir_defaults_to_outputs_stem(tmp_path: Path) -> None:
    """Without --output-dir, output goes to outputs/<input-stem>/retarget."""
    if not _ASSET_XML.exists():
        pytest.skip("robot assets not present; skipping end-to-end CLI test")

    hdf5 = tmp_path / "in" / "task" / "7.hdf5"
    hdf5.parent.mkdir(parents=True)
    _write_egodex_hdf5(hdf5, n_frames=1)
    _write_frames_json(tmp_path / "outroot" / "7" / "process" / "frames.json", 1)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "retarget.cli",
            "--input",
            str(hdf5),
            "--robot-xml",
            str(_ASSET_XML),
            "--ik-config",
            str(_IK_CONFIG),
            "--output-root",
            str(tmp_path / "outroot"),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # --output-root was removed in the redesign; the CLI must reject it.
    assert result.returncode != 0
    assert "output-root" in (result.stdout + result.stderr).lower()


def test_robot_xml_from_ikconfig(tmp_path: Path) -> None:
    """With no --robot-xml, the ikconfig's robot_xml is used (cwd-relative)."""
    if not _ASSET_XML.exists():
        pytest.skip("robot assets not present; skipping end-to-end CLI test")

    hdf5 = tmp_path / "in" / "task" / "0.hdf5"
    hdf5.parent.mkdir(parents=True)
    _write_egodex_hdf5(hdf5, n_frames=1)
    _write_frames_json(tmp_path / "out" / "0" / "process" / "frames.json", 1)

    # A config with an explicit robot_xml (the real one, repo-root-relative).
    ik_cfg = json.loads(_IK_CONFIG.read_text())
    assert "robot_xml" in ik_cfg
    ik_path = tmp_path / "ik.json"
    ik_path.write_text(json.dumps(ik_cfg))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "retarget",
            "--input",
            str(hdf5),
            "--ik-config",
            str(ik_path),
            "--output-dir",
            str(tmp_path / "out2"),
            "--frames-json",
            str(tmp_path / "out" / "0" / "process" / "frames.json"),
        ],
        cwd=_REPO_ROOT,  # ikconfig robot_xml is repo-root-relative
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out2" / "trajectory.npz").is_file()


def test_output_filename_is_fixed() -> None:
    """The output name is fixed in code, not taken from an arg."""
    from retarget.cli import TRAJECTORY_FILENAME

    assert TRAJECTORY_FILENAME == "trajectory.npz"


def test_source_type_read_from_ik_config(tmp_path: Path) -> None:
    """The CLI's source_type comes from the ik config's source_type field."""
    from retarget.cli import load_source_type

    path = tmp_path / "ik.json"
    path.write_text(json.dumps({"source_type": "egodex", "human_root_name": "hip"}))
    assert load_source_type(path) == "egodex"

    # Missing field falls back to egodex (the only registered source today).
    path.write_text(json.dumps({"human_root_name": "hip"}))
    assert load_source_type(path) == "egodex"


def test_load_frame_timestamps_from_frames_json(tmp_path: Path) -> None:
    """Timestamps are read from the process manifest, in frame order."""
    from retarget.cli import load_frame_timestamps

    path = tmp_path / "frames.json"
    _write_frames_json(path, n_frames=3, start_s=1.0)
    timestamps = load_frame_timestamps(path)
    np.testing.assert_allclose(timestamps, [1.0, 1.0 + 1 / 30, 1.0 + 2 / 30])


def test_load_frame_timestamps_null_falls_back_to_index(tmp_path: Path) -> None:
    """Entries with null timestamp_sec fall back to their running index."""
    from retarget.cli import load_frame_timestamps

    path = tmp_path / "frames.json"
    _write_frames_json(path, n_frames=2)
    data = json.loads(path.read_text())
    data["entries"][0]["timestamp_sec"] = None
    data["entries"][1]["timestamp_sec"] = None
    path.write_text(json.dumps(data))

    timestamps = load_frame_timestamps(path)
    np.testing.assert_allclose(timestamps, [0.0, 1.0])


def test_load_camera_config_normalizes_schema() -> None:
    """The ik-config camera entry is normalized into render options."""
    from retarget.cli import load_camera_config

    cfg = load_camera_config(
        {
            "camera": {
                "link": "d435_link",
                "pos_offset": [0.1, 0.2, 0.3],
                "orn_offset": [0.1736482, 0, 0.9848078, 0],
                "width": 320,
                "height": 240,
                "depth": True,
            }
        }
    )
    assert cfg == {
        "link": "d435_link",
        "pos_offset": (0.1, 0.2, 0.3),
        "orn_offset": (0.1736482, 0.0, 0.9848078, 0.0),
        "width": 320,
        "height": 240,
        "depth": True,
    }


def test_load_camera_config_defaults_and_absence() -> None:
    """Missing camera -> None; optional camera keys default sensibly."""
    from retarget.cli import load_camera_config

    assert load_camera_config({"human_root_name": "hip"}) is None
    cfg = load_camera_config({"camera": {"link": "pelvis"}})
    assert cfg is not None
    assert cfg["pos_offset"] == (0.0, 0.0, 0.0)
    assert cfg["orn_offset"] == (1.0, 0.0, 0.0, 0.0)
    # Width/height are None so the caller infers them from the source video.
    assert cfg["width"] is None
    assert cfg["height"] is None
    assert cfg["depth"] is False


def test_load_camera_config_rejects_bad_schema() -> None:
    """A camera entry without a link, or with malformed offsets, is rejected."""
    from retarget.cli import load_camera_config

    with pytest.raises(ValueError, match="link"):
        load_camera_config({"camera": {"pos_offset": [0, 0, 0]}})
    with pytest.raises(ValueError, match="pos_offset"):
        load_camera_config({"camera": {"link": "pelvis", "pos_offset": [0, 0]}})
    with pytest.raises(ValueError, match="orn_offset"):
        load_camera_config({"camera": {"link": "pelvis", "orn_offset": [1, 0, 0]}})


def test_camera_intrinsics_match_resolution() -> None:
    """The manifest intrinsics are square-pixel and centered."""
    from retarget.cli import camera_intrinsics

    k = camera_intrinsics(640, 480)
    assert k[0][0] == k[1][1]  # square pixels
    assert k[0][2] == 320.0
    assert k[1][2] == 240.0
    assert k[2] == [0.0, 0.0, 1.0]
    # fx = (h/2) / tan(fovy/2) for the MuJoCo default 45 deg vertical fov.
    import math

    assert k[0][0] == pytest.approx(240.0 / math.tan(math.radians(45.0) / 2.0))


def test_write_camera_manifest(tmp_path: Path) -> None:
    """camera.json records mount, intrinsics and per-frame entries."""
    import json

    from retarget.cli import (
        CAMERA_JSON_FILENAME,
        load_camera_config,
        resolve_camera_resolution,
        write_camera_manifest,
    )

    camera_cfg = load_camera_config({"camera": {"link": "d435_link", "depth": True}})
    camera_cfg["width"], camera_cfg["height"] = resolve_camera_resolution(
        camera_cfg, None
    )
    assert (camera_cfg["width"], camera_cfg["height"]) == (640, 480)
    timestamps = np.asarray([0.0, 1 / 30.0, 2 / 30.0])
    depth_stats = [
        {
            "index": i,
            "depth_filename": f"{i:06d}.npy",
            "depth_min": 0.1,
            "depth_max": 5.0,
            "depth_mean": 1.0,
        }
        for i in range(3)
    ]
    path = write_camera_manifest(tmp_path, camera_cfg, timestamps, depth_stats)
    assert path == tmp_path / CAMERA_JSON_FILENAME

    data = json.loads(path.read_text())
    assert data["stage"] == "retarget_camera"
    assert data["camera"]["link"] == "d435_link"
    assert data["camera"]["intrinsics"][0][0] > 0
    assert data["depth_enabled"] is True
    assert data["frame_count"] == 3
    assert data["entries"][1] == {
        "index": 1,
        "rgb_filename": "000001.png",
        "depth_filename": "000001.npy",
        "timestamp_sec": 1 / 30.0,
        "depth_min": 0.1,
        "depth_max": 5.0,
        "depth_mean": 1.0,
    }


def test_write_camera_manifest_without_depth(tmp_path: Path) -> None:
    """Without depth, entries carry null depth fields."""
    import json

    from retarget.cli import (
        load_camera_config,
        resolve_camera_resolution,
        write_camera_manifest,
    )

    camera_cfg = load_camera_config({"camera": {"link": "pelvis"}})
    camera_cfg["width"], camera_cfg["height"] = resolve_camera_resolution(
        camera_cfg, None
    )
    path = write_camera_manifest(tmp_path, camera_cfg, np.asarray([0.0, 0.1]), None)
    data = json.loads(path.read_text())
    assert data["depth_enabled"] is False
    assert data["depth_dir"] is None
    assert data["entries"][0]["depth_filename"] is None


def test_resolve_camera_resolution_explicit_wins() -> None:
    """Explicit width/height in the camera config are used as-is."""
    from retarget.cli import load_camera_config, resolve_camera_resolution

    camera_cfg = load_camera_config(
        {"camera": {"link": "pelvis", "width": 320, "height": 240}}
    )
    assert resolve_camera_resolution(camera_cfg, None) == (320, 240)


def test_resolve_camera_resolution_probes_source_video(
    tmp_path: Path, monkeypatch
) -> None:
    """Missing width/height is inferred from frames.json's source_video."""
    from retarget.cli import load_camera_config, resolve_camera_resolution

    video = tmp_path / "source.mp4"
    frames_json = tmp_path / "frames.json"
    frames_json.write_text(json.dumps({"source_video": str(video), "entries": []}))
    monkeypatch.setattr("retarget.cli.probe_video_resolution", lambda path: (1280, 720))

    camera_cfg = load_camera_config({"camera": {"link": "pelvis"}})
    assert resolve_camera_resolution(camera_cfg, frames_json) == (1280, 720)

    # A partially-specified config is also completed from the source video.
    partial = load_camera_config({"camera": {"link": "pelvis", "width": 640}})
    assert resolve_camera_resolution(partial, frames_json) == (1280, 720)


def test_resolve_camera_resolution_falls_back_when_unavailable(
    tmp_path: Path,
) -> None:
    """No frames.json / no source_video -> built-in default resolution."""
    from retarget.cli import (
        CAMERA_DEFAULT_HEIGHT,
        CAMERA_DEFAULT_WIDTH,
        load_camera_config,
        resolve_camera_resolution,
    )

    camera_cfg = load_camera_config({"camera": {"link": "pelvis"}})
    assert resolve_camera_resolution(camera_cfg, None) == (
        CAMERA_DEFAULT_WIDTH,
        CAMERA_DEFAULT_HEIGHT,
    )
    missing = tmp_path / "missing.json"
    assert resolve_camera_resolution(camera_cfg, missing) == (
        CAMERA_DEFAULT_WIDTH,
        CAMERA_DEFAULT_HEIGHT,
    )
    no_video = tmp_path / "frames.json"
    no_video.write_text(json.dumps({"entries": []}))
    assert resolve_camera_resolution(camera_cfg, no_video) == (
        CAMERA_DEFAULT_WIDTH,
        CAMERA_DEFAULT_HEIGHT,
    )


def test_probe_video_resolution_parses_ffprobe(tmp_path: Path, monkeypatch) -> None:
    """ffprobe JSON is parsed into (width, height)."""
    import shutil
    import subprocess

    from retarget.cli import probe_video_resolution

    video = tmp_path / "source.mp4"
    video.touch()
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffprobe")

    class FakeProc:
        returncode = 0
        stdout = '{"streams": [{"width": 1920, "height": 1080}]}'
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: FakeProc())
    assert probe_video_resolution(video) == (1920, 1080)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="boom"
        ),
    )
    with pytest.raises(RuntimeError, match="boom"):
        probe_video_resolution(video)
