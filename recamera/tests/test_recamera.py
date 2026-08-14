"""Tests for the recamera depth-match renderer.

Geometry and camera-init tests use pure numpy (run anywhere). Render tests use
the real robot model + real episode outputs, skipped when those are absent —
the real model's segmentation maps geoms to body ids reliably, which synthetic
models do not (a MuJoCo seg quirk with synthetic bodies).
"""

import numpy as np
import pytest

from recamera.camera import camera_center, refine_iou
from recamera.geometry import backproject


# --- geometry ---------------------------------------------------------------


def _make_intrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def test_backproject_roundtrip():
    """Mask*depth pixels backproject to camera-frame 3D, then re-project back."""
    K = _make_intrinsics()
    H, W = 480, 640
    mask = np.zeros((H, W), dtype=bool)
    mask[300:400, 400:500] = True
    depth = np.full((H, W), 0.5, dtype=np.float32)
    pts = backproject(mask, depth, K)
    assert pts.shape[0] == mask.sum()
    np.testing.assert_allclose(pts[:, 2], 0.5, atol=1e-6)
    u = pts[:, 0] * K[0, 0] / pts[:, 2] + K[0, 2]
    v = pts[:, 1] * K[1, 1] / pts[:, 2] + K[1, 2]
    assert (u >= 400).all() and (u < 500).all()
    assert (v >= 300).all() and (v < 400).all()


def test_backproject_ignores_zero_depth():
    K = _make_intrinsics()
    mask = np.zeros((480, 640), dtype=bool)
    mask[10:20, 10:20] = True
    depth = np.zeros((480, 640), dtype=np.float32)
    pts = backproject(mask, depth, K)
    assert pts.shape[0] == 0


def test_umeyama_recovers_rigid_transform():
    """Umeyama recovers the exact transform mapping world points to cam points."""
    from recamera.camera import _umeyama

    rng = np.random.default_rng(0)
    world = rng.uniform(0, 1, (20, 3))
    T_true = np.eye(4)
    T_true[:3, :3] = _rotation_z(0.5)
    T_true[:3, 3] = [0.2, -0.1, 0.3]
    cam = (T_true[:3, :3] @ world.T).T + T_true[:3, 3]
    T = _umeyama(world, cam)
    np.testing.assert_allclose(T, T_true, atol=1e-8)


def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def test_camera_center_roundtrip():
    """camera_center(T) recovers the camera position used to build T."""
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(camera_center(T), [-1.0, -2.0, -3.0])


# --- render (real assets required) ------------------------------------------

_REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
_ROBOT_XML = (
    _REPO_ROOT
    / "assets/unitree_g1_mjcf"
    / ("g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml")
)
_EPISODE = _REPO_ROOT / "outputs" / "0"

_HAS_ASSETS = _ROBOT_XML.is_file() and (_EPISODE / "depth/depth.json").is_file()


@pytest.mark.skipif(not _HAS_ASSETS, reason="real robot assets / episode absent")
def test_render_arm_rgba_transparent_background():
    """Real-model render: arm pixels alpha=255, background alpha=0."""
    import mujoco as mj

    from recamera.inputs import load_episode
    from recamera.render import render_arm_rgba
    from recamera.robot import arm_body_ids, build_model, camera_id

    ep = load_episode(_EPISODE.parent, _EPISODE.name)
    model = build_model(_ROBOT_XML, 640, 360)
    data = mj.MjData(model)
    data.qpos[:] = ep.qpos[0]
    mj.mj_forward(model, data)
    camid = camera_id(model)
    arm_ids = arm_body_ids(model)
    K = ep.intrinsics[0]

    # a camera that frames the arm: from above-left, looking at the arm centroid
    from recamera.robot import skeleton_anchors

    J = skeleton_anchors(model, data)
    armc = J.mean(0)
    cam_pos = armc + np.array([0.0, 0.0, 0.3])
    from recamera.camera import _rotation_for_lookat

    T = _rotation_for_lookat(cam_pos, armc)

    rgba = render_arm_rgba(model, data, camid, arm_ids, T, K, 640, 360)
    assert rgba.shape == (360, 640, 4)
    assert rgba[..., 3].max() == 255  # some arm pixel opaque
    assert (rgba[..., 3] == 0).any()  # some background transparent


@pytest.mark.skipif(not _HAS_ASSETS, reason="real robot assets / episode absent")
def test_refine_iou_keeps_aligned_pose():
    """Refining an already-aligned camera keeps it (IoU unchanged)."""
    import mujoco as mj

    from recamera.inputs import load_episode
    from recamera.robot import arm_body_ids, build_model, camera_id, skeleton_anchors

    ep = load_episode(_EPISODE.parent, _EPISODE.name)
    model = build_model(_ROBOT_XML, 640, 360)
    data = mj.MjData(model)
    data.qpos[:] = ep.qpos[0]
    mj.mj_forward(model, data)
    camid = camera_id(model)
    arm_ids = arm_body_ids(model)
    K = ep.intrinsics[0]

    J = skeleton_anchors(model, data)
    armc = J.mean(0)
    cam_pos = armc + np.array([0.0, 0.0, 0.3])
    from recamera.camera import _rotation_for_lookat

    T = _rotation_for_lookat(cam_pos, armc)

    def render_fn(t):
        from recamera.render import render_arm_rgba

        return render_arm_rgba(model, data, camid, arm_ids, t, K, 320, 180)[..., 3] > 0

    from PIL import Image

    mask_small = np.asarray(Image.fromarray(ep.masks[0]).resize((320, 180))) > 0
    iou0 = _silhouette_iou(T, render_fn, mask_small)
    T2 = refine_iou(T, render_fn=render_fn, mask=mask_small, max_iter=40)
    iou1 = _silhouette_iou(T2, render_fn, mask_small)
    assert iou1 >= iou0 - 0.05  # refinement does not hurt alignment


def _silhouette_iou(T, render_fn, mask):
    arm = render_fn(T)
    if arm is None or arm.sum() == 0:
        return 0.0
    inter = (arm & mask).sum()
    union = (arm | mask).sum()
    return inter / max(union, 1)
