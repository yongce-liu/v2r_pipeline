"""Robot model handling for the first-person depth-match renderer.

The robot's MJCF is loaded once, its mesh paths absolutized so an extra fixed
camera can be injected in-memory (relative mesh paths break under
``from_xml_string``). The module exposes:

* :func:`build_model` — the robot model with a ``__fp__`` fixed camera appended,
  plus the model's offscreen framebuffer raised to the source video resolution.
* :func:`arm_body_ids` — body ids whose geoms are the arm + hand (masked out of
  the final RGBA by setting every other body's alpha to 0).
* :func:`skeleton_anchors` — named joint positions (arm links + finger links)
  that define the geometric skeleton used for the depth-match alignment.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import mujoco as mj

# Camera name injected into the MJCF under <worldbody>. ``mode="fixed"`` lets us
# drive it fully from data.cam_xpos / data.cam_xmat per frame.
CAMERA_NAME = "__fp__"

# Bodies whose geoms make up the robot arm + hand (everything we want to keep
# visible in the transparent first-person output). Finger links use the L_/R_
# prefix scheme of the G1 Inspire hand.
ARM_PREFIXES = (
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "L_",
    "R_",
)

# Skeleton anchor joints used for depth matching. The arm links give the elbow/
# wrist depth alignment (the user's chosen match target); the finger links keep
# the hand (which the human mask contains) from drifting.
SKELETON_ANCHORS = (
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


def build_model(xml_path: Path, width: int, height: int) -> mj.MjModel:
    """Load the robot model and append a fixed first-person camera.

    Mesh paths in ``xml_path`` are rewritten to absolute paths (relative to the
    MJCF's ``meshes/`` dir) so :func:`mujoco.MjModel.from_xml_string` can resolve
    them; the renderer needs ``from_xml_string`` because the camera is injected
    into the XML. The offscreen framebuffer is raised to the source resolution
    so full-res RGBA renders are allowed.
    """
    tree = ET.parse(xml_path)
    meshes_dir = xml_path.parent / "meshes"
    for m in tree.iter("mesh"):
        if m.get("file"):
            m.set("file", str((meshes_dir / m.get("file")).resolve()))

    worldbody = tree.getroot().find("./worldbody")
    cam = ET.SubElement(worldbody, "camera")
    cam.set("name", CAMERA_NAME)
    cam.set("mode", "fixed")

    model = mj.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
    model.vis.global_.offwidth = int(width)
    model.vis.global_.offheight = int(height)
    return model


def arm_body_ids(model: mj.MjModel) -> np.ndarray:
    """Body ids whose geoms should be visible in the output frame."""
    return np.asarray(
        [
            b
            for b in range(model.nbody)
            if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) or "").startswith(
                ARM_PREFIXES
            )
        ],
        dtype=np.int32,
    )


def camera_id(model: mj.MjModel) -> int:
    """Id of the injected first-person camera."""
    return next(
        i
        for i in range(model.ncam)
        if mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i) == CAMERA_NAME
    )


def skeleton_anchors(
    model: mj.MjModel, data: mj.MjData, anchors: tuple[str, ...] = SKELETON_ANCHORS
) -> np.ndarray:
    """World-frame positions of the named skeleton anchors, as (N, 3).

    ``data`` must already have the desired qpos applied and ``mj_forward`` run.
    """
    ids = [
        next(
            i
            for i in range(model.nbody)
            if mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) == name
        )
        for name in anchors
    ]
    return np.stack([data.xpos[i] for i in ids])
