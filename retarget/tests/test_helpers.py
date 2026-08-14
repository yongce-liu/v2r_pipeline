"""Unit tests for pure retargeting helpers (no robot assets required)."""

import numpy as np

from retarget import (
    load_ik_config,
    merge_gmr_hand_points,
    required_gmr_hand_points,
)


def _sample_ik_config() -> dict:
    """A minimal IK config exercising body + hand-point matching tables."""
    return {
        "human_root_name": "hip",
        "ik_match_table1": {
            "pelvis": ["hip", 100, 0, [0, 0, 0], [1, 0, 0, 0]],
            "left_elbow_link": [
                "leftForearm",
                20,
                0,
                [0, 0, 0],
                [1, 0, 0, 0],
            ],
        },
        "ik_match_table2": {
            "left_wrist_yaw_link": [
                "leftIndexFingerTip",
                80,
                80,
                [0, 0, 0],
                [1, 0, 0, 0],
            ],
            # A hand point with zero weight must be excluded.
            "right_wrist_yaw_link": [
                "rightIndexFingerTip",
                0,
                0,
                [0, 0, 0],
                [1, 0, 0, 0],
            ],
        },
    }


def test_required_gmr_hand_points_returns_weighted_hand_names() -> None:
    config = _sample_ik_config()
    points = required_gmr_hand_points(config)
    # leftIndexFingerTip is a hand point with nonzero weight -> included.
    # rightIndexFingerTip has zero weight -> excluded.
    # leftForearm is a body name -> excluded.
    assert "leftIndexFingerTip" in points
    assert "rightIndexFingerTip" not in points
    assert "leftForearm" not in points


def test_merge_gmr_hand_points_adds_identity_quat_targets() -> None:
    config = _sample_ik_config()
    required = required_gmr_hand_points(config)
    assert required

    base = {
        "hip": [np.zeros(3, dtype=np.float64), np.array([1.0, 0, 0, 0])],
        "leftForearm": [
            np.array([0.1, 0.2, 1.0], dtype=np.float64),
            np.array([1.0, 0, 0, 0]),
        ],
    }
    hand = {"leftIndexFingerTip": np.array([0.3, 0.4, 1.0], dtype=np.float64)}

    merged = merge_gmr_hand_points(base, hand, required)
    # base entries survive unchanged.
    assert set(base.keys()) <= set(merged.keys())
    # hand point becomes a position-only target with identity rotation.
    assert "leftIndexFingerTip" in merged
    pos, quat = merged["leftIndexFingerTip"]
    np.testing.assert_array_equal(pos, hand["leftIndexFingerTip"])
    np.testing.assert_array_equal(quat, [1.0, 0.0, 0.0, 0.0])
    # the merge must not alias the caller's arrays.
    merged["leftIndexFingerTip"][0][0] = 999.0
    assert hand["leftIndexFingerTip"][0] != 999.0


def test_load_ik_config_roundtrip(tmp_path) -> None:
    config = _sample_ik_config()
    path = tmp_path / "ik.json"
    import json

    path.write_text(json.dumps(config))
    loaded = load_ik_config(path)
    assert loaded["human_root_name"] == "hip"
