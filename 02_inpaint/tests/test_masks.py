"""Tests for loading the segment mask manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inpaint.masks import MaskManifest, load_mask_manifest


def build_segment_layout(tmp_path: Path, frame_count: int = 3) -> tuple[Path, Path]:
    """Create a synthetic ``segment`` stage output and return (masks.json, clip_root).

    Mirrors the real layout ``outputs/<clip>/segment/masks.json``; frames 0 and
    1 carry a mask, frame 2 has none.
    """

    clip_root = tmp_path / "0"
    masks_dir = clip_root / "segment" / "masks"
    masks_dir.mkdir(parents=True)
    for index in range(frame_count - 1):
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        Image.fromarray(mask).save(masks_dir / f"mask_{index:06d}.png")

    masks_json = clip_root / "segment" / "masks.json"
    entries = []
    for index in range(frame_count):
        has_mask = index < frame_count - 1
        entries.append(
            {
                "index": index,
                "frame_filename": f"frame_{index:06d}.png",
                "timestamp_sec": index / 10.0,
                "mask_filename": f"mask_{index:06d}.png" if has_mask else None,
                "vis_filename": f"vis_{index:06d}.jpg",
                "has_mask": has_mask,
                "instance_count": 1 if has_mask else 0,
                "area": 16 if has_mask else 0,
                "bbox": {"min_row": 2, "min_col": 2, "max_row": 5, "max_col": 5},
            }
        )
    masks_json.write_text(
        json.dumps(
            {
                "source_frames_json": str(clip_root / "process" / "frames.json"),
                "source_video": str(tmp_path / "clip.mp4"),
                "fps": 10.0,
                "width": 10,
                "height": 8,
                "frame_format": "png",
                "frame_count": frame_count,
                "masks_dir": str(masks_dir),
                "masks_vis_dir": str(clip_root / "segment" / "masks_vis"),
                "mask_format": "png",
                "vis_enabled": True,
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return masks_json, tmp_path


def test_load_mask_manifest(tmp_path: Path) -> None:
    masks_json, _ = build_segment_layout(tmp_path)
    manifest = load_mask_manifest(masks_json)

    assert isinstance(manifest, MaskManifest)
    assert manifest.frame_count == 3
    assert manifest.source_frames_json == str(
        tmp_path / "0" / "process" / "frames.json"
    )
    assert len(manifest.entries) == 3

    entry = manifest.entries[0]
    assert entry.index == 0
    assert entry.has_mask is True
    assert entry.mask_path is not None
    assert entry.mask_path.exists()
    assert entry.area == 16

    empty = manifest.entries[2]
    assert empty.has_mask is False
    assert empty.mask_path is None
    assert empty.mask_filename is None


def test_load_mask_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mask_manifest(tmp_path / "nope.json")


def test_load_mask_manifest_missing_masks_dir(tmp_path: Path) -> None:
    masks_json = tmp_path / "masks.json"
    masks_json.write_text(
        json.dumps({"masks_dir": str(tmp_path / "does-not-exist"), "entries": []}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        load_mask_manifest(masks_json)
