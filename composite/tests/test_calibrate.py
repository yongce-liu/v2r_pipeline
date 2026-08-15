"""Unit tests for the depth-matching math in composite.calibrate."""

from __future__ import annotations

import numpy as np
import pytest

from composite.calibrate import (
    arm_mask,
    calibrate_frame,
    composite_frame,
    fit_affine_robust,
)


def test_arm_mask_detects_near_surface() -> None:
    depth = np.full((10, 10), 73.9, dtype=np.float32)
    depth[2:6, 3:7] = 0.4
    mask = arm_mask(depth)
    assert mask.sum() == 16
    assert not mask[0, 0]


def test_fit_affine_robust_recovers_affine() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0.3, 0.6, size=2000)
    y = 2.5 * x + 0.3
    y[::50] += rng.uniform(-3, 3, size=40)  # outliers
    fit = fit_affine_robust(x, y)
    assert fit is not None
    slope, intercept, n = fit
    assert slope == pytest.approx(2.5, abs=0.02)
    assert intercept == pytest.approx(0.3, abs=0.02)
    assert n == 2000


def test_fit_affine_robust_rejects_degenerate() -> None:
    x = np.ones(100)
    y = np.linspace(0, 1, 100)
    assert fit_affine_robust(x, y) is None


def _synthetic_frame(shape=(80, 100)) -> dict:
    """Robot/DA3 arrays with a known arm-vs-background occlusion layout."""

    h, w = shape
    arm_slice = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
    # Background depth varies smoothly so the affine fits have x-variation.
    z_bg = np.linspace(0.6, 1.0, w, dtype=np.float32)[None, :]
    z_bg = np.repeat(z_bg, h, axis=0)
    # A closer foreground (0.2 m) under part of the arm that must hide it.
    z_bg[3 * h // 8 : 5 * h // 8, 5 * w // 8 : 7 * w // 8] = 0.2

    robot_depth = np.full(shape, 70.0, dtype=np.float32)
    # Arm depth varies 0.35..0.45 m so the arm calibration has x-variation.
    z_arm = np.linspace(0.35, 0.45, 3 * h // 4 - h // 4, dtype=np.float32)[:, None]
    robot_depth[arm_slice] = np.repeat(z_arm, 3 * w // 4 - w // 4, axis=1)

    scene_orig = (3.0 * z_bg + 0.1).astype(np.float32)
    scene_orig[arm_slice] = 3.0 * robot_depth[arm_slice] + 0.1
    scene_inp = (4.0 * z_bg + 0.2).astype(np.float32)

    robot_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    robot_rgb[arm_slice] = (255, 255, 255)
    inpainted_rgb = np.full((h, w, 3), (10, 200, 10), dtype=np.uint8)

    return {
        "robot_depth": robot_depth,
        "scene_orig": scene_orig,
        "scene_inp": scene_inp,
        "robot_rgb": robot_rgb,
        "inpainted_rgb": inpainted_rgb,
        "arm_slice": arm_slice,
        "occluder_slice": (
            slice(3 * h // 8, 5 * h // 8),
            slice(5 * w // 8, 7 * w // 8),
        ),
    }


def test_calibration_and_composite_depth_matching() -> None:
    f = _synthetic_frame()
    arm = arm_mask(f["robot_depth"])

    calib = calibrate_frame(
        f["robot_depth"], f["scene_orig"], f["scene_inp"], arm, max_samples=None
    )
    assert calib.valid
    # Robot arm at 0.4 m -> DA3_orig 1.3 -> DA3_inp 1.8.
    mapped = calib.arm_depth_in_scene_space(np.full_like(f["robot_depth"], 0.4))[
        f["arm_slice"]
    ]
    assert np.allclose(mapped, 1.8, atol=0.1)

    out, _, fraction, n_arm, _ = composite_frame(
        f["inpainted_rgb"],
        f["robot_rgb"],
        f["robot_depth"],
        f["scene_inp"],
        calib,
        margin_frac=0.0,
        feather_px=0,
    )
    arm_pixels = (3 * 80 // 4 - 80 // 4) * (3 * 100 // 4 - 100 // 4)
    assert n_arm == arm_pixels
    # Visible where the background is farther than the arm, hidden where the
    # 0.2 m foreground is closer to the camera.
    arm_white = np.zeros(out.shape[:2], dtype=bool)
    arm_white[f["arm_slice"]] = True
    arm_white[f["occluder_slice"]] = False
    assert np.all(out[arm_white] == (255, 255, 255))
    assert np.all(out[f["occluder_slice"]] == (10, 200, 10))
    assert 0.8 < fraction < 0.95


def test_composite_fallback_mask_without_calibration() -> None:
    f = _synthetic_frame()
    out, _, fraction, n_arm, n_visible = composite_frame(
        f["inpainted_rgb"],
        f["robot_rgb"],
        f["robot_depth"],
        f["scene_inp"],
        None,
        margin_frac=0.0,
        feather_px=0,
    )
    assert n_visible == n_arm
    assert fraction == 1.0
    assert np.all(out[f["arm_slice"]] == (255, 255, 255))
