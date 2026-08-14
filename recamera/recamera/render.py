"""Render the robot arm under a solved camera pose with a transparent background.

Produces per-frame RGBA PNGs: the robot arm (and only the arm + hand bodies) is
opaque, every other pixel has alpha=0. We render the full scene normally for RGB,
and use a segmentation pass to decide which pixels belong to arm bodies — those
get alpha=255, everything else alpha=0.

Also exposes :func:`render_arm_cloud`, which renders the arm depth under a camera
pose and backprojects it — the signal used by the depth-match ICP refinement.
"""

from __future__ import annotations

import numpy as np

import mujoco as mj

from recamera.geometry import backproject


def camera_center(T: np.ndarray) -> np.ndarray:
    """World-frame camera center from a world->camera transform."""
    return -T[:3, :3].T @ T[:3, 3]


def camera_axes_world(T: np.ndarray) -> np.ndarray:
    """Camera axes (x, y, z) in world frame, as rows of a 3x3.

    ``T`` is world->camera in the pinhole convention. MuJoCo fixed cameras render
    with the y axis negated relative to a right-handed pinhole frame (verified
    empirically). This function returns ``T[:3,:3].T`` with the y row negated.
    """
    R = T[:3, :3].T  # rows = cam x, y, z axes in world (right-handed)
    R[1] *= -1.0
    return R


def _setup_camera(model, data, camid, T, K, height, width) -> None:
    data.cam_xpos[camid] = camera_center(T)
    data.cam_xmat[camid] = camera_axes_world(T).flatten()
    model.cam_fovy[camid] = float(np.degrees(2 * np.arctan2(height / 2, K[1, 1])))


def _fixed_camera_view(camid: int) -> mj.MjvCamera:
    cam = mj.MjvCamera()
    cam.fixedcamid = camid
    cam.type = mj.mjtCamera.mjCAMERA_FIXED
    return cam


def _decode_seg(seg: np.ndarray, width: int, height: int):
    """Split a segmentation render into body-id 2D array.

    Mujoco returns two layouts depending on buffer size: a packed (H, W) int
    (``id*1000 + class``) for small framebuffers, or a (H, W, 2) [id, class]
    layout at full resolution. Normalize both to a (H, W) body-id array.
    """
    seg = seg.reshape(height, width, -1)
    if seg.shape[-1] == 2:
        return seg[..., 0]
    return seg[..., 0] % 1000


def render_arm_rgba(
    model: mj.MjModel,
    data: mj.MjData,
    camid: int,
    arm_ids: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Render the robot arm under world->camera ``T`` as an (H, W, 4) RGBA.

    ``K`` is the (3, 3) intrinsics; the camera's vertical fov is set so the
    rendered depth matches the source camera's fy. Alpha is 255 exactly where a
    visible arm-body pixel is, else 0.
    """
    _setup_camera(model, data, camid, T, K, height, width)
    cam = _fixed_camera_view(camid)

    renderer = mj.Renderer(model, height, width)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        seg = renderer.render().copy().astype(np.int32)
        renderer.disable_segmentation_rendering()

        renderer.update_scene(data, camera=cam)
        rgb = renderer.render().copy()
    finally:
        renderer.close()

    segid = _decode_seg(seg, width, height)
    # Arm pixels are identified purely by body id; the class channel is a
    # mesh/geom-specific grouping that must not be filtered on.
    arm_px = np.isin(segid, arm_ids)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.where(arm_px, 255, 0).astype(np.uint8)
    return rgba


class ArmSilhouetteRenderer:
    """Reusable seg-only renderer: returns arm-silhouette masks for candidate T.

    The IoU optimization calls this many times per frame; reusing the renderer
    and skipping the RGB pass keeps it fast.
    """

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        camid: int,
        arm_ids: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        self.model = model
        self.data = data
        self.camid = camid
        self.arm_ids = arm_ids
        self.width = width
        self.height = height
        self._renderer = mj.Renderer(model, height, width)
        self._cam = _fixed_camera_view(camid)

    def __call__(self, T: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Return (H, W) bool arm-silhouette mask for pose ``T`` and intrinsics ``K``."""
        _setup_camera(self.model, self.data, self.camid, T, K, self.height, self.width)
        r = self._renderer
        r.enable_segmentation_rendering()
        r.update_scene(self.data, camera=self._cam)
        seg = r.render().copy().astype(np.int32)
        r.disable_segmentation_rendering()
        segid = _decode_seg(seg, self.width, self.height)
        return np.isin(segid, self.arm_ids)

    def close(self) -> None:
        self._renderer.close()


def render_arm_cloud(
    model: mj.MjModel,
    data: mj.MjData,
    camid: int,
    arm_ids: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Backproject the robot arm's depth under ``T`` into a camera-frame cloud.

    Returns an (N, 3) cloud of the visible arm surface, or None if the arm is not
    in view (fewer than 50 pixels). Used as the render/measure step for ICP.
    """
    _setup_camera(model, data, camid, T, K, height, width)
    cam = _fixed_camera_view(camid)

    renderer = mj.Renderer(model, height, width)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        seg = renderer.render().copy().astype(np.int32)
        renderer.disable_segmentation_rendering()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
    finally:
        renderer.close()

    segid = _decode_seg(seg, width, height)
    arm_px = np.isin(segid, arm_ids)
    depth = np.where(arm_px, depth, 0.0)
    cloud = backproject(arm_px, depth, K)
    if len(cloud) < 50:
        return None
    return cloud
