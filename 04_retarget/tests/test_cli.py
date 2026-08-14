"""Minimal CLI tests for the retarget step.

The end-to-end test requires real robot assets (relative to the repo root), so
it is skipped when the assets are absent. The argument-parsing and help tests
run anywhere.
"""

import os
import subprocess
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
_ASSET_XML = (
    _PKG_ROOT.parent
    / "assets"
    / "unitree_g1_mjcf"
    / ("g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml")
)


def test_tyro_args_parse() -> None:
    """The RetargetArgs dataclass parses via tyro without a robot model."""
    from retarget.cli import RetargetArgs

    args = RetargetArgs()
    assert args.confidence_threshold == 0.5
    assert str(args.robot_xml).endswith(".xml")


def test_cli_help_runs() -> None:
    """The console entry point exposes --help without importing robot code."""
    result = subprocess.run(
        [sys.executable, "-m", "retarget.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--robot-xml" in result.stdout
    assert "--ik-config" in result.stdout
    assert "--confidence-threshold" in result.stdout


def test_cli_runs_end_to_end(tmp_path: Path) -> None:
    """Running the CLI produces an npz output with the real robot model."""
    if not _ASSET_XML.exists():
        import pytest

        pytest.skip("robot assets not present; skipping end-to-end CLI test")

    out = tmp_path / "trajectory.npz"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "retarget.cli",
            "--output",
            str(out),
        ],
        cwd=_PKG_ROOT.parent,  # relative asset paths resolve against repo root
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    import numpy as np

    data = np.load(out)
    assert "qpos" in data
    assert "action" in data
    assert data["action"].shape == (1, 53)  # G1 + 24 inspire finger joints
