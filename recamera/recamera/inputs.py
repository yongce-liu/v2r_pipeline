"""Load per-episode inputs for the first-person depth-match rendering.

All inputs live under ``outputs/<stem>/`` from the earlier pipeline stages:

* ``depth/depth.json``  — per-frame intrinsics (``entries[i].intrinsics``)
* ``depth/depths/depth_%06d.npy`` — per-frame depth maps
* ``segment/masks/mask_%06d.png`` — binary human-arm mask (the ``segment`` stage
  runs SAM3 with prompt "human hand and arm")
* ``process/frames.json`` — frame manifest (count, per-frame timestamps)
* ``retarget/trajectory.npz`` — retargeted robot pose (``qpos``)

This module exposes the decoded arrays and per-frame intrinsics in a flat,
stage-agnostic form so the matching/render code never touches stage paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class EpisodeInputs:
    """Per-episode decoded inputs for the depth-match renderer."""

    frame_count: int
    fps: float
    width: int
    height: int
    masks: np.ndarray
    """(T, H, W) bool, True where the human arm is present."""
    depths: np.ndarray
    """(T, H, W) float32 metric depth in meters."""
    intrinsics: np.ndarray
    """(T, 3, 3) per-frame camera intrinsics."""
    qpos: np.ndarray
    """(T, nq) full robot pose (freejoint + joints) from retarget."""
    timestamps: np.ndarray
    """(T,) frame timestamps in seconds."""
    trajectory_path: Path
    depth_json_path: Path


def _read_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def load_episode(output_root: Path, stem: str) -> EpisodeInputs:
    """Load all first-person render inputs for one episode.

    ``output_root`` is the root that contains ``<stem>/`` (repo root in the CLI
    default); ``stem`` is the episode dir name (e.g. ``0``).
    """
    base = output_root / stem

    depth_json_path = base / "depth" / "depth.json"
    depth_json = _read_manifest(depth_json_path)
    depth_entries = depth_json["entries"]
    depths_dir = Path(depth_json["depths_dir"])
    mask_dir = Path(_read_manifest(base / "segment" / "masks.json")["masks_dir"])
    frames_json = _read_manifest(base / "process" / "frames.json")
    traj = np.load(base / "retarget" / "trajectory.npz")

    n = depth_json["frame_count"]
    h = int(depth_json["height"])
    w = int(depth_json["width"])

    masks = np.empty((n, h, w), dtype=bool)
    depths = np.empty((n, h, w), dtype=np.float32)
    intrinsics = np.empty((n, 3, 3), dtype=np.float64)

    for i, entry in enumerate(depth_entries):
        masks[i] = (
            np.asarray(
                Image.open(mask_dir / entry["frame_filename"].replace("frame", "mask"))
            )
            > 0
        )
        depths[i] = np.load(depths_dir / entry["depth_filename"])
        intrinsics[i] = np.asarray(entry["intrinsics"], dtype=np.float64)

    timestamps = np.asarray(
        [
            float(e.get("timestamp_sec", i))
            if e.get("timestamp_sec") is not None
            else float(i)
            for i, e in enumerate(frames_json["entries"])
        ],
        dtype=np.float64,
    )

    return EpisodeInputs(
        frame_count=n,
        fps=float(depth_json["fps"]),
        width=w,
        height=h,
        masks=masks,
        depths=depths,
        intrinsics=intrinsics,
        qpos=traj["qpos"],
        timestamps=timestamps,
        trajectory_path=base / "retarget" / "trajectory.npz",
        depth_json_path=depth_json_path,
    )
