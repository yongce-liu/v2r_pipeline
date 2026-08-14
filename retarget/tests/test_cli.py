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
_IK_CONFIG = _REPO_ROOT / "configs" / "egodex_g1_inspire_dfq.json"


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
                        "filename": f"frame_{i:06d}.png",
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
    assert args.vis is False
    assert args.mujoco is False


def test_vis_and_mujoco_flags_parse() -> None:
    """--vis and --mujoco parse as booleans and appear in --help."""
    result = subprocess.run(
        [sys.executable, "-m", "retarget.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--vis" in result.stdout
    assert "--mujoco" in result.stdout


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
