"""Loading the ``segment`` stage mask manifest (``masks.json``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaskEntry:
    """One per-frame mask record from the segment manifest."""

    index: int
    frame_filename: str
    timestamp_sec: float | None
    mask_filename: str | None
    vis_filename: str | None
    has_mask: bool
    instance_count: int
    area: int
    bbox: dict | None
    mask_path: Path | None
    """Absolute path to the mask image, or None when ``has_mask`` is False."""


@dataclass(frozen=True)
class MaskManifest:
    """Parsed ``masks.json`` written by the ``segment`` stage."""

    source_frames_json: str
    source_video: str
    fps: float | None
    width: int | None
    height: int | None
    frame_format: str
    frame_count: int
    masks_dir: Path
    mask_format: str
    entries: list[MaskEntry]


def load_mask_manifest(masks_json: Path) -> MaskManifest:
    """Read and validate a ``segment`` ``masks.json`` file.

    Mask images are resolved against the ``masks_dir`` recorded in the
    manifest. Every entry also carries the referenced ``frame_filename``, which
    the inpaint stage resolves against the process manifest.
    """
    masks_json = masks_json.expanduser()
    if not masks_json.exists():
        raise FileNotFoundError(f"masks.json not found: {masks_json}")

    data = json.loads(masks_json.read_text(encoding="utf-8"))
    masks_dir = Path(data.get("masks_dir", "")).expanduser()
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks_dir from masks.json not found: {masks_dir}")

    entries: list[MaskEntry] = []
    for raw in data.get("entries", []):
        index = int(raw["index"])
        has_mask = bool(raw.get("has_mask"))
        mask_filename = raw.get("mask_filename")
        mask_path = None
        if has_mask and mask_filename:
            mask_path = (masks_dir / mask_filename).resolve()
        entries.append(
            MaskEntry(
                index=index,
                frame_filename=raw.get("frame_filename", ""),
                timestamp_sec=raw.get("timestamp_sec"),
                mask_filename=mask_filename,
                vis_filename=raw.get("vis_filename"),
                has_mask=has_mask,
                instance_count=int(raw.get("instance_count", 0)),
                area=int(raw.get("area", 0)),
                bbox=raw.get("bbox"),
                mask_path=mask_path,
            )
        )
    if not entries:
        raise ValueError(f"No mask entries in masks.json: {masks_json}")

    return MaskManifest(
        source_frames_json=data.get("source_frames_json", ""),
        source_video=data.get("source_video", ""),
        fps=data.get("fps"),
        width=data.get("width"),
        height=data.get("height"),
        frame_format=data.get("frame_format", "png"),
        frame_count=int(data.get("frame_count", len(entries))),
        masks_dir=masks_dir.resolve(),
        mask_format=data.get("mask_format", "png"),
        entries=entries,
    )
