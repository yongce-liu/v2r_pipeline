"""Loading of the stage manifests consumed by the composite step.

The composite step joins three earlier stage outputs, all described by their
own JSON manifest (same shared frame-manifest conventions as the rest of the
pipeline):

- ``inpainted.json``  — the ``inpaint`` stage (background frames, hand removed);
- ``depth.json``      — the ``depth`` stage run on the *inpainted* frames
  (scene depth behind the robot arm);
- ``camera.json``     — the ``retarget`` stage's mounted-camera render manifest
  (robot arm RGB + metric depth);
- ``calibration-depth-json`` (optional) — the ``depth`` stage run on the
  *original* frames, used to map metric arm depth into the scene-depth space.

File paths recorded inside each manifest are resolved relative to the manifest
file itself when they are relative (``retarget/camera.json`` stores
``camera_rgb`` / ``camera_depth`` as bare directory names).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def _read_json(path: Path) -> dict:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"manifest {path} must be a JSON object")
    return data


def _resolve(path_value: str | None, base_dir: Path) -> Path | None:
    """Resolve a manifest-embedded path against ``base_dir`` (manifest dir).

    Absolute paths pass through. Relative paths are tried first relative to
    the manifest itself (the bare-directory convention used by
    ``retarget/camera.json``), then relative to the current working directory
    (some depth manifests record paths like ``outputs/<clip>/depth_orig/depths``
    relative to the pipeline root). The manifest-relative candidate wins when
    both exist so the documented convention keeps priority.
    """

    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    relative_to_manifest = base_dir / path
    if relative_to_manifest.exists():
        return relative_to_manifest
    relative_to_cwd = Path.cwd() / path
    if relative_to_cwd.exists():
        return relative_to_cwd
    return relative_to_manifest


def _entries_by_index(raw_entries: list) -> dict[int, dict]:
    entries: dict[int, dict] = {}
    for raw in raw_entries:
        try:
            index = int(raw["index"])
        except (KeyError, TypeError, ValueError):
            continue
        entries[index] = raw
    return entries


@dataclass(frozen=True)
class CameraEntry:
    """One rendered camera frame from ``retarget/camera.json``."""

    index: int
    rgb_filename: str
    depth_filename: str | None
    timestamp_sec: float | None
    depth_min: float | None
    depth_max: float | None
    depth_mean: float | None


@dataclass(frozen=True)
class CameraManifest:
    """Parsed ``retarget/camera.json`` (robot-mounted camera renders)."""

    width: int
    height: int
    frame_count: int
    rgb_dir: Path
    depth_dir: Path | None
    entries: list[CameraEntry]
    by_index: dict[int, CameraEntry]
    base_dir: Path

    def rgb_path(self, entry: CameraEntry) -> Path:
        return self.rgb_dir / entry.rgb_filename

    def depth_path(self, entry: CameraEntry) -> Path:
        if entry.depth_filename is None or self.depth_dir is None:
            raise FileNotFoundError(f"no depth render for frame {entry.index}")
        return self.depth_dir / entry.depth_filename


def load_camera_manifest(camera_json: Path) -> CameraManifest:
    """Read and validate the retarget camera render manifest."""

    data = _read_json(camera_json)
    base_dir = camera_json.expanduser().resolve().parent

    entries: list[CameraEntry] = []
    for raw in data.get("entries", []):
        entries.append(
            CameraEntry(
                index=int(raw["index"]),
                rgb_filename=str(raw["rgb_filename"]),
                depth_filename=raw.get("depth_filename"),
                timestamp_sec=raw.get("timestamp_sec"),
                depth_min=raw.get("depth_min"),
                depth_max=raw.get("depth_max"),
                depth_mean=raw.get("depth_mean"),
            )
        )
    if not entries:
        raise ValueError(f"no camera entries in {camera_json}")

    rgb_dir = _resolve(data.get("rgb_dir"), base_dir)
    if rgb_dir is None or not rgb_dir.exists():
        raise FileNotFoundError(f"camera rgb_dir not found: {rgb_dir}")
    depth_dir = _resolve(data.get("depth_dir"), base_dir)
    if depth_dir is not None and not depth_dir.exists():
        raise FileNotFoundError(f"camera depth_dir not found: {depth_dir}")

    by_index = {entry.index: entry for entry in entries}
    return CameraManifest(
        width=int(data.get("width", 0)),
        height=int(data.get("height", 0)),
        frame_count=int(data.get("frame_count", len(entries))),
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        entries=entries,
        by_index=by_index,
        base_dir=base_dir,
    )


@dataclass(frozen=True)
class DepthEntry:
    """One per-frame depth record from a ``depth.json`` manifest."""

    index: int
    frame_filename: str
    timestamp_sec: float | None
    depth_filename: str
    width: int | None
    height: int | None


@dataclass(frozen=True)
class DepthManifest:
    """Parsed ``depth.json`` (DA3 per-frame relative depth)."""

    fps: float | None
    width: int | None
    height: int | None
    frame_count: int
    depths_dir: Path
    entries: list[DepthEntry]
    by_index: dict[int, DepthEntry]
    base_dir: Path

    def depth_path(self, entry: DepthEntry) -> Path:
        return self.depths_dir / entry.depth_filename


def load_depth_manifest(depth_json: Path) -> DepthManifest:
    """Read a ``depth`` stage manifest (scene depth in DA3 units)."""

    data = _read_json(depth_json)
    base_dir = depth_json.expanduser().resolve().parent

    depths_dir = _resolve(data.get("depths_dir") or data.get("frames_dir"), base_dir)
    if depths_dir is None or not depths_dir.exists():
        raise FileNotFoundError(f"depth frames dir not found: {depths_dir}")

    entries: list[DepthEntry] = []
    for raw in data.get("entries", []):
        entries.append(
            DepthEntry(
                index=int(raw["index"]),
                frame_filename=str(raw.get("frame_filename", "")),
                timestamp_sec=raw.get("timestamp_sec"),
                depth_filename=str(raw.get("depth_filename", "")),
                width=raw.get("width"),
                height=raw.get("height"),
            )
        )
    if not entries:
        raise ValueError(f"no depth entries in {depth_json}")

    by_index = {entry.index: entry for entry in entries}
    return DepthManifest(
        fps=data.get("fps"),
        width=data.get("width"),
        height=data.get("height"),
        frame_count=int(data.get("frame_count", len(entries))),
        depths_dir=depths_dir,
        entries=entries,
        by_index=by_index,
        base_dir=base_dir,
    )


@dataclass(frozen=True)
class InpaintEntry:
    """One per-frame record from ``inpainted.json``."""

    index: int
    frame_filename: str
    timestamp_sec: float | None
    inpainted_filename: str


@dataclass(frozen=True)
class InpaintManifest:
    """Parsed ``inpainted.json`` (hand-removed background frames)."""

    fps: float | None
    width: int | None
    height: int | None
    frame_count: int
    frames_dir: Path
    entries: list[InpaintEntry]
    by_index: dict[int, InpaintEntry]
    base_dir: Path

    def frame_path(self, entry: InpaintEntry) -> Path:
        return self.frames_dir / entry.inpainted_filename


def load_inpaint_manifest(inpainted_json: Path) -> InpaintManifest:
    """Read an ``inpaint`` stage manifest (inpainted background frames)."""

    data = _read_json(inpainted_json)
    base_dir = inpainted_json.expanduser().resolve().parent

    frames_dir = _resolve(data.get("inpainted_dir") or data.get("frames_dir"), base_dir)
    if frames_dir is None or not frames_dir.exists():
        raise FileNotFoundError(f"inpainted frames dir not found: {frames_dir}")

    entries: list[InpaintEntry] = []
    for raw in data.get("entries", []):
        entries.append(
            InpaintEntry(
                index=int(raw["index"]),
                frame_filename=str(raw.get("frame_filename", "")),
                timestamp_sec=raw.get("timestamp_sec"),
                inpainted_filename=str(
                    raw.get("inpainted_filename", raw.get("frame_filename", ""))
                ),
            )
        )
    if not entries:
        raise ValueError(f"no inpainted entries in {inpainted_json}")

    by_index = {entry.index: entry for entry in entries}
    return InpaintManifest(
        fps=data.get("fps"),
        width=data.get("width"),
        height=data.get("height"),
        frame_count=int(data.get("frame_count", len(entries))),
        frames_dir=frames_dir,
        entries=entries,
        by_index=by_index,
        base_dir=base_dir,
    )


@dataclass(frozen=True)
class MaskEntry:
    """One per-frame record from ``masks.json`` (segment stage)."""

    index: int
    frame_filename: str
    timestamp_sec: float | None
    mask_filename: str


@dataclass(frozen=True)
class MaskManifest:
    """Parsed ``masks.json`` (human hand/arm masks on the original frames)."""

    fps: float | None
    width: int | None
    height: int | None
    frame_count: int
    masks_dir: Path
    entries: list[MaskEntry]
    by_index: dict[int, MaskEntry]
    base_dir: Path

    def mask_path(self, entry: MaskEntry) -> Path:
        return self.masks_dir / entry.mask_filename


def load_mask_manifest(masks_json: Path) -> MaskManifest:
    """Read a ``segment`` stage manifest (hand/arm segmentation masks)."""

    data = _read_json(masks_json)
    base_dir = masks_json.expanduser().resolve().parent

    masks_dir = _resolve(data.get("masks_dir") or data.get("frames_dir"), base_dir)
    if masks_dir is None or not masks_dir.exists():
        raise FileNotFoundError(f"mask frames dir not found: {masks_dir}")

    entries: list[MaskEntry] = []
    for raw in data.get("entries", []):
        entries.append(
            MaskEntry(
                index=int(raw["index"]),
                frame_filename=str(raw.get("frame_filename", "")),
                timestamp_sec=raw.get("timestamp_sec"),
                mask_filename=str(raw.get("mask_filename", "")),
            )
        )
    if not entries:
        raise ValueError(f"no mask entries in {masks_json}")

    by_index = {entry.index: entry for entry in entries}
    return MaskManifest(
        fps=data.get("fps"),
        width=data.get("width"),
        height=data.get("height"),
        frame_count=int(data.get("frame_count", len(entries))),
        masks_dir=masks_dir,
        entries=entries,
        by_index=by_index,
        base_dir=base_dir,
    )
