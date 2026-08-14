"""Per-frame orchestration: depth-match camera pose + transparent-arm render.

Iterates the episode's frames in order. Frame 0 gets an absolute camera solve
(PnP from the human arm medial curve + silhouette IoU refinement); later frames
seed from the previous camera pose and refine the same way. Each frame is
rendered as a transparent-background RGBA and written as a PNG, plus a small
``config.json``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import mujoco as mj

from recamera.camera import refine_iou, solve_camera
from recamera.inputs import EpisodeInputs
from recamera.render import ArmSilhouetteRenderer, render_arm_rgba
from recamera.robot import arm_body_ids, camera_id, skeleton_anchors


@dataclass
class RenderConfig:
    """Tunables for the depth-match + render pass."""

    skeleton_anchor_names: tuple[str, ...] = (
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "L_thumb_distal",
        "L_index_intermediate",
        "L_middle_intermediate",
        "L_ring_intermediate",
        "L_pinky_intermediate",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
        "R_thumb_distal",
        "R_index_intermediate",
        "R_middle_intermediate",
        "R_ring_intermediate",
        "R_pinky_intermediate",
    )
    """Robot joint anchors used for the depth-match alignment signal."""

    cloud_stride: int = 1
    """Backprojection stride on the human arm mask (1 = full res)."""

    iou_render_width: int = 320
    """Render width used for the IoU refinement (reduced res, faster)."""


def write_config_json(output_dir: Path, cfg: RenderConfig, ep: EpisodeInputs) -> Path:
    """Write a small manifest mirroring the other stage layouts."""
    data = {
        "package": {"name": "recamera", "version": "0.1.0"},
        "source": {
            "depth_json": str(ep.depth_json_path),
            "trajectory": str(ep.trajectory_path),
            "frame_count": ep.frame_count,
            "fps": ep.fps,
            "width": ep.width,
            "height": ep.height,
        },
        "render": {
            "output": "frames",
            "format": "png",
            "transparent_background": True,
            "cloud_stride": cfg.cloud_stride,
        },
    }
    path = output_dir / "config.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def _render_silhouette_fn(model, data, camid, arm_ids, K, width, height, mask_shape):
    """Build ``render_fn(T) -> bool arm-silhouette`` at reduced res."""

    def render_fn(T):
        rgba = render_arm_rgba(model, data, camid, arm_ids, T, K, width, height)
        return rgba[..., 3] > 0

    return render_fn


def process_episode(
    model,
    ep: EpisodeInputs,
    output_dir: Path,
    cfg: RenderConfig | None = None,
) -> int:
    """Run the full first-person depth-match render for one episode.

    Returns the number of frames written. ``model`` is the robot model with the
    injected first-person camera (see :func:`recamera.robot.build_model`).
    """
    cfg = cfg or RenderConfig()
    data = mj.MjData(model)
    arm_ids = arm_body_ids(model)
    camid = camera_id(model)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    iou_h = max(1, int(cfg.iou_render_width * ep.height / ep.width))
    scale = cfg.iou_render_width / ep.width

    # Downscale mask and intrinsics to the IoU render resolution. (Depth is not
    # needed here: the medial-curve seed runs at full resolution.)
    from PIL import Image as _Image

    iou_masks = [
        np.asarray(
            _Image.fromarray(np.ascontiguousarray(ep.masks[i])).resize(
                (cfg.iou_render_width, iou_h)
            )
        )
        > 0
        for i in range(ep.frame_count)
    ]
    iou_K = ep.intrinsics.copy()
    iou_K[:, 0, 0] *= scale
    iou_K[:, 1, 1] *= scale
    iou_K[:, 0, 2] *= scale
    iou_K[:, 1, 2] *= scale

    t0 = time.time()
    T_prev: np.ndarray | None = None
    silhouette = ArmSilhouetteRenderer(
        model, data, camid, arm_ids, cfg.iou_render_width, iou_h
    )
    try:
        for i in range(ep.frame_count):
            data.qpos[:] = ep.qpos[i]
            mj.mj_forward(model, data)
            J = skeleton_anchors(model, data, cfg.skeleton_anchor_names)
            Ki = iou_K[i]

            def render_fn(T, _K=Ki):
                return silhouette(T, _K)

            if T_prev is None:
                # Frame 0: absolute PnP solve. The medial-curve seed uses the
                # FULL-resolution mask/depth/K (downscaling destroys its 3D
                # geometry); IoU refinement uses the render-res mask.
                T = solve_camera(
                    J,
                    ep.masks[i],
                    ep.depths[i],
                    ep.intrinsics[i],
                    render_fn=render_fn,
                    iou_mask=iou_masks[i],
                )
            else:
                # Later frames: the first-person camera is roughly stable; refine
                # the previous pose toward this frame's arm mask (IoU).
                T = refine_iou(T_prev, render_fn=render_fn, mask=iou_masks[i])

            rgba = render_arm_rgba(
                model, data, camid, arm_ids, T, ep.intrinsics[i], ep.width, ep.height
            )
            Image.fromarray(rgba).save(frames_dir / f"frame_{i:06d}.png")
            T_prev = T

            if (i + 1) % 10 == 0 or i == ep.frame_count - 1:
                print(f"  frame {i + 1}/{ep.frame_count}  ({time.time() - t0:.1f}s)")
    finally:
        silhouette.close()

    write_config_json(output_dir, cfg, ep)
    return ep.frame_count
