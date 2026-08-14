"""Camera pose estimation for the depth-match.

The task: find camera extrinsics ``T`` (world -> camera) so the robot arm,
projected through ``T``, sits where the human arm was in the original first-person
video. The robot is retargeted into a world frame where it occupies the human's
pose, so the human arm cloud (camera frame) and the robot arm (world frame)
differ only by ``T``: ``P_human_cam ≈ T · P_robot_world``.

Design (per user decisions — no robot camera-mount prior, Open3D):

1. **PnP seed** (:func:`solve_camera`): match the robot arm's skeleton chain
   (shoulder→elbow→wrist→fingers) to ordered points on the human arm's 3D medial
   curve (mask×depth backprojection), then solve the rigid transform via Umeyama.
   Tries both robot arms and keeps the one with higher silhouette overlap.
2. **Silhouette IoU refinement** (:func:`refine_iou`): optimize T so the rendered
   robot-arm silhouette overlaps the human arm mask (IoU) — the direct visual
   goal. Seeded from the PnP solve.

Later frames seed from the previous camera pose and refine the same way, so the
camera follows the arm across the episode.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

# --- helpers ----------------------------------------------------------------


def camera_center(T: np.ndarray) -> np.ndarray:
    """World-frame camera center from a world->camera transform."""
    return -T[:3, :3].T @ T[:3, 3]


def _T_from_params(p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler("xyz", np.deg2rad(p[3:6])).as_matrix()
    T[:3, 3] = p[:3]
    return T


def _params_from_T(T: np.ndarray) -> np.ndarray:
    """6-DoF params from T, robust to a slightly non-orthonormal R.

    Umeyama can produce a reflection (det=-1) when the correspondence is poor;
    projecting onto SO(3) via SVD keeps the optimizer in a valid pose space.
    """
    U, _S, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    T2 = np.eye(4)
    T2[:3, :3] = R
    T2[:3, 3] = T[:3, 3]
    return np.concatenate(
        [T2[:3, 3], np.degrees(Rot.from_matrix(T2[:3, :3]).as_euler("xyz"))]
    )


def _umeyama(world_pts: np.ndarray, cam_pts: np.ndarray) -> np.ndarray:
    """Rigid transform (4x4) mapping ``world_pts`` onto ``cam_pts``."""
    mw, mc = world_pts.mean(0), cam_pts.mean(0)
    H = (world_pts - mw).T @ (cam_pts - mc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    t = mc - R @ mw
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# --- human arm medial curve --------------------------------------------------


def medial_curve_3d(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, stride=4):
    """Ordered 3D polyline of the human arm's medial axis (camera frame).

    The arm mask is eroded to a ridge and backprojected; the points are ordered
    along the mask's principal axis. Returns an (N, 3) camera-frame curve.
    """
    m = mask.astype(np.uint8)
    skel = ndimage.binary_erosion(m, structure=np.ones((3, 3)))
    skel = ndimage.binary_erosion(m, structure=np.ones((5, 5)))
    dist = ndimage.distance_transform_edt(m)
    ridge = skel & (dist > dist.max() * 0.35)
    ys, xs = np.nonzero(ridge)
    z = depth[ys, xs]
    keep = (z > 1e-4) & np.isfinite(z)
    ys, xs, z = ys[keep], xs[keep], z[keep]
    if stride > 1:
        ys, xs, z = ys[::stride], xs[::stride], z[::stride]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    pts = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], axis=1)
    if len(pts) < 4:
        return pts
    Pc = pts - pts.mean(0)
    _, evecs = np.linalg.eigh(Pc.T @ Pc)
    axis = evecs[:, -1]
    order = np.argsort(pts @ axis)
    return pts[order]


# --- camera solve ------------------------------------------------------------


def solve_camera(
    skeleton: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    *,
    render_fn,
    iou_mask: np.ndarray | None = None,
    arm_offsets: tuple[int, ...] = (0, 8),
) -> np.ndarray:
    """Solve the camera for one frame.

    ``skeleton`` is the (16, 3) world-frame robot skeleton; ``mask``/``depth``/``K``
    are the FULL-resolution human arm mask, depth, and intrinsics — the medial
    curve used for the PnP seed is computed at full res because downscaling
    destroys its 3D geometry. ``render_fn(T)`` returns a boolean arm-silhouette
    at the IoU render resolution; ``iou_mask`` (same res) is the IoU target
    (defaults to ``mask``). Tries each arm, PnP-seeds from the medial curve, and
    returns the pose with the best silhouette overlap after refinement.
    """
    curve = medial_curve_3d(mask, depth, K)
    if len(curve) < 4:
        raise RuntimeError("Human arm medial curve is empty; cannot solve camera")
    if iou_mask is None:
        iou_mask = mask

    best_T, best_iou = None, -1.0
    for off in arm_offsets:
        robot_chain = skeleton[[off, off + 1, off + 2, off + 3]]
        fracs = [0.05, 0.4, 0.75, 0.98]
        n = len(curve)
        cam_pts = np.array([curve[int(f * (n - 1))] for f in fracs])
        T = _umeyama(robot_chain, cam_pts)
        T = refine_iou(T, render_fn=render_fn, mask=iou_mask)
        iou = _silhouette_iou(T, render_fn, iou_mask)
        if iou > best_iou:
            best_T, best_iou = T, iou
    if best_T is None:
        raise RuntimeError("No camera pose produced any arm pixels")
    return best_T


def refine_iou(
    T: np.ndarray,
    *,
    render_fn,
    mask: np.ndarray,
    max_iter: int = 60,
) -> np.ndarray:
    """Optimize ``T`` to maximize robot-arm/human-mask silhouette IoU.

    ``render_fn(T)`` returns a boolean arm-silhouette mask at the same resolution
    as ``mask``. Uses Nelder-Mead on the 6-DoF camera pose seeded from ``T``.
    """

    def cost(p: np.ndarray) -> float:
        t = _T_from_params(p)
        arm = render_fn(t)
        if arm is None or arm.sum() < 10:
            return 1e6
        inter = (arm & mask).sum()
        union = (arm | mask).sum()
        return float(-inter / max(union, 1))

    res = minimize(
        cost,
        _params_from_T(T),
        method="Nelder-Mead",
        options=dict(maxiter=max_iter, xatol=1e-3, fatol=1e-4),
    )
    return _T_from_params(res.x)


def _silhouette_iou(T: np.ndarray, render_fn, mask: np.ndarray) -> float:
    arm = render_fn(T)
    if arm is None or arm.sum() == 0:
        return -1.0
    inter = (arm & mask).sum()
    union = (arm | mask).sum()
    return float(inter / max(union, 1))


def _rotation_for_lookat(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """T world->cam looking from ``cam_pos`` toward ``target`` (pinhole)."""
    cam_pos = np.asarray(cam_pos, float)
    target = np.asarray(target, float)
    fwd = target - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    zc = fwd  # pinhole: third row = view direction
    xc = np.cross(np.array([0.0, 0.0, 1.0]), zc)
    if np.linalg.norm(xc) < 1e-9:
        xc = np.cross(np.array([0.0, 1.0, 0.0]), zc)  # fallback up
    xc = xc / np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    Rcw = np.stack([xc, yc, zc])  # cam axes in world (right-handed)
    R = Rcw.T.copy()
    R[2] *= -1.0  # third row = view direction (arm maps to +z)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ cam_pos
    return T
