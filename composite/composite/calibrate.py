"""Depth matching between the robot render and the inpainted scene.

The ``retarget`` camera render provides the robot arm as an RGB image plus a
*metric* depth buffer (meters; far-plane distance where nothing was hit). The
``depth`` stage provides *relative* scene depth in DA3 units, which is an
unknown per-image affine function of metric depth. To decide whether the arm is
in front of the scene at a given pixel we therefore need the affine mapping
between the two spaces.

Calibration strategy (per frame):

1. ``arm`` mask from the metric depth buffer: pixels closer than the far plane.
2. Arm correspondences — ``DA3_orig ≈ a_arm * z_robot + b_arm`` — sampled where
   the robot arm overlaps the depth estimate of the *original* frames (the
   human hand occupied the same 3D location as the robot arm, so its DA3 value
   is the value the arm surface should take). This anchors the arm in the
   scene-depth space.
3. Background correspondences — ``DA3_inp ≈ a_bg * DA3_orig + b_bg`` — sampled
   outside the arm region, where the inpainted and original frames are
   identical, mapping the original run's normalization into the inpainted
   run's normalization.
4. Occlusion test in the inpainted depth space: the arm pixel is shown iff
   ``a_bg * (a_arm * z_robot + b_arm) + b_bg < DA3_inp - margin``, i.e. the arm
   surface is closer to the camera than the scene surface behind it.

All fits are robust (median-split two-point) so a few misaligned boundary
pixels do not shift the mapping, and every fit is restricted to finite values.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FAR_EPS_M = 1e-3
"""Depth margin below the far plane (meters) used to detect the arm."""


def arm_mask(robot_depth: np.ndarray, far: float | None = None) -> np.ndarray:
    """Pixels belonging to the robot arm in a metric depth buffer.

    ``far`` defaults to the buffer's own maximum (the MuJoCo far plane); the
    arm is anything closer than ``far - FAR_EPS_M``.
    """

    depth = robot_depth.astype(np.float32, copy=False)
    if far is None:
        far = float(depth.max())
    return depth < far - FAR_EPS_M


def _sample_pixels(mask: np.ndarray, max_samples: int | None) -> np.ndarray:
    """Flat indices of ``mask`` pixels, optionally subsampled uniformly."""

    indices = np.flatnonzero(mask)
    if max_samples is not None and indices.size > max_samples:
        stride = max(1, indices.size // max_samples)
        indices = indices[::stride][:max_samples]
    return indices


def fit_affine_robust(
    x: np.ndarray,
    y: np.ndarray,
    min_samples: int = 64,
) -> tuple[float, float, int] | None:
    """Robust affine fit ``y ≈ slope * x + intercept``.

    Uses the median of the lower and upper halves of ``x`` as the two anchor
    points, which tolerates outliers far better than plain least squares.
    Returns ``None`` when there are not enough valid samples or the anchor
    spread is degenerate.
    """

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < min_samples:
        return None
    if x.size > 4_000_000:
        # Keep the two anchors cheap on large frames.
        order = np.argsort(x)
        step = max(1, x.size // 4_000_000)
        x = x[order[::step]]
        y = y[order[::step]]

    xm = np.median(x)
    lo = x <= xm
    hi = x > xm
    if lo.sum() < 2 or hi.sum() < 2:
        return None
    x_lo, y_lo = x[lo], y[lo]
    x_hi, y_hi = x[hi], y[hi]
    dx = float(np.median(x_hi) - np.median(x_lo))
    if abs(dx) < 1e-12:
        return None
    slope = float((np.median(y_hi) - np.median(y_lo)) / dx)
    intercept = float(np.median(y) - slope * np.median(x))
    return slope, intercept, int(x.size)


@dataclass(frozen=True)
class FrameCalibration:
    """Per-frame affine mapping from robot depth into inpainted depth space."""

    slope: float | None
    """Slope of the arm mapping ``DA3_orig ≈ slope * z_robot + intercept``."""
    intercept: float | None
    """Intercept of the arm mapping."""
    bg_slope: float | None
    """Slope mapping the original run into the inpainted run's space."""
    bg_intercept: float | None
    """Intercept mapping the original run into the inpainted run's space."""
    n_arm: int
    """Number of arm correspondences used."""
    n_bg: int
    """Number of background correspondences used."""

    @property
    def valid(self) -> bool:
        return (
            self.slope is not None
            and self.intercept is not None
            and self.bg_slope is not None
            and self.bg_intercept is not None
        )

    def arm_depth_in_scene_space(self, robot_depth: np.ndarray) -> np.ndarray:
        """Map metric robot depth into the inpainted scene-depth space."""

        if not self.valid:
            raise ValueError("calibration is not valid")
        assert self.slope is not None and self.intercept is not None
        assert self.bg_slope is not None and self.bg_intercept is not None
        return (
            self.bg_slope * (self.slope * robot_depth + self.intercept)
            + self.bg_intercept
        )


def calibrate_frame(
    robot_depth: np.ndarray,
    scene_orig: np.ndarray,
    scene_inp: np.ndarray,
    arm: np.ndarray,
    *,
    erode_px: int = 2,
    max_samples: int | None = 200_000,
    hand_mask: np.ndarray | None = None,
) -> FrameCalibration:
    """Fit the arm/background affines for one frame.

    ``scene_orig`` is the DA3 depth of the *original* frame, ``scene_inp`` the
    DA3 depth of the *inpainted* frame, and ``robot_depth`` the metric render
    depth. Pixels are matched on their flat indices, so all arrays must share
    the same shape.
    """

    if robot_depth.shape != scene_orig.shape or robot_depth.shape != scene_inp.shape:
        raise ValueError("robot_depth, scene_orig and scene_inp must share a shape")

    erode_px = max(0, int(erode_px))
    if hand_mask is not None and hand_mask.shape == arm.shape:
        # Prefer the human-hand mask (segment stage): the robot gripper only
        # overlaps the human hand, while the rest of the rendered arm has no
        # counterpart in the original frames.
        if erode_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
            )
            arm_core = cv2.erode(arm.astype(np.uint8), kernel).astype(bool)
        else:
            arm_core = arm
        arm_masked = arm_core & hand_mask
        if arm_masked.sum() >= 64:
            arm_core = arm_masked
    elif erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2)
        arm_core = cv2.erode(arm.astype(np.uint8), kernel).astype(bool)
    else:
        arm_core = arm
    if arm_core.sum() < 64:
        return FrameCalibration(None, None, None, None, 0, 0)

    # Arm correspondences: robot metric depth vs original-frame depth at the
    # location the human hand (and therefore the robot arm) occupied.
    arm_idx = _sample_pixels(arm_core & np.isfinite(scene_orig), max_samples)
    arm_fit = None
    n_arm = 0
    if arm_idx.size >= 64:
        arm_fit = fit_affine_robust(
            robot_depth.ravel()[arm_idx],
            scene_orig.ravel()[arm_idx],
        )
        if arm_fit is not None:
            n_arm = arm_fit[2]

    # Background correspondences: outside the (dilated) arm, original and
    # inpainted frames are identical, so this maps the two runs' normalizations.
    if erode_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
        )
        arm_dilated = cv2.dilate(arm.astype(np.uint8), kernel).astype(bool)
    else:
        arm_dilated = arm
    bg_mask = (~arm_dilated) & np.isfinite(scene_orig) & np.isfinite(scene_inp)
    bg_idx = _sample_pixels(bg_mask, max_samples)
    bg_fit = None
    n_bg = 0
    if bg_idx.size >= 64:
        bg_fit = fit_affine_robust(
            scene_orig.ravel()[bg_idx],
            scene_inp.ravel()[bg_idx],
        )
        if bg_fit is not None:
            n_bg = bg_fit[2]

    if arm_fit is None or bg_fit is None:
        return FrameCalibration(None, None, None, None, n_arm, n_bg)
    slope, intercept, _ = arm_fit
    bg_slope, bg_intercept, _ = bg_fit
    return FrameCalibration(slope, intercept, bg_slope, bg_intercept, n_arm, n_bg)


def composite_frame(
    inpainted_rgb: np.ndarray,
    robot_rgb: np.ndarray,
    robot_depth: np.ndarray,
    scene_inp: np.ndarray,
    calibration: FrameCalibration | None,
    *,
    margin_frac: float = 0.02,
    feather_px: int = 3,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """Blend the robot arm into the inpainted frame using depth.

    Returns ``(composited_rgb, alpha, visible_fraction, n_arm, n_visible)``.
    When ``calibration`` is ``None`` (no original-frame depth available) the
    arm is composited with a plain mask — depth matching is skipped and the
    caller should log that limitation.

    ``margin_frac`` biases the occlusion test toward showing the arm: the arm
    is hidden only when its mapped depth is *clearly* behind the scene surface
    (``arm_scene > scene + margin``). Near-ties, which are within depth-noise
    of each other, keep the arm visible because the arm is the subject of the
    frame.
    """

    inpainted = inpainted_rgb.astype(np.float32, copy=False)
    robot = robot_rgb.astype(np.float32, copy=False)
    arm = arm_mask(robot_depth)

    alpha = np.zeros(arm.shape, dtype=np.float32)
    if arm.any():
        if calibration is not None and calibration.valid:
            arm_scene = calibration.arm_depth_in_scene_space(robot_depth)
            scene = scene_inp.astype(np.float32, copy=False)
            finite = np.isfinite(arm_scene) & np.isfinite(scene)
            margin = (
                margin_frac * float(np.nanmax(scene[finite]) - np.nanmin(scene[finite]))
                if finite.any()
                else 0.0
            )
            visible = arm & finite & (arm_scene < scene + margin)
        else:
            visible = arm.copy()
        alpha[visible] = 1.0

    if feather_px > 0 and alpha.any():
        kernel = (2 * feather_px + 1, 2 * feather_px + 1)
        alpha = cv2.GaussianBlur(alpha, kernel, 0)

    n_arm = int(arm.sum())
    n_visible = int((alpha > 0.5).sum())
    visible_fraction = n_visible / n_arm if n_arm else 0.0

    blended = inpainted.copy()
    if alpha.any():
        alpha_b = alpha[..., None]
        blended = alpha_b * robot + (1.0 - alpha_b) * inpainted
    out = np.clip(blended, 0, 255).astype(np.uint8)
    return out, alpha, visible_fraction, n_arm, n_visible
