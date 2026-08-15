"""Tests for loading the shared frame manifest (common schema v1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from depth.frames import (
    FRAME_MANIFEST_SCHEMA_VERSION,
    FrameManifest,
    load_frame_manifest,
)


def _write_frame_images(frames_dir: Path, frame_count: int) -> None:
    frames_dir.mkdir(parents=True)
    for index in range(frame_count):
        rgb = np.full((8, 10, 3), index * 40, dtype=np.uint8)
        Image.fromarray(rgb).save(frames_dir / f"{index:06d}.png")


def build_process_layout(tmp_path: Path, frame_count: int = 3) -> tuple[Path, Path]:
    """Create a synthetic ``process`` stage output and return (frames.json, clip_root).

    Mirrors the real layout ``outputs/<clip>/process/frames.json`` so the clip
    stem resolves to ``tmp_path / "0"``.
    """

    clip_root = tmp_path / "0"
    frames_dir = clip_root / "process" / "frames"
    _write_frame_images(frames_dir, frame_count)

    frames_json = clip_root / "process" / "frames.json"
    frames_json.write_text(
        json.dumps(
            {
                "schema_version": FRAME_MANIFEST_SCHEMA_VERSION,
                "stage": "process",
                "source_video": str(tmp_path / "clip.mp4"),
                "fps": 10.0,
                "width": 10,
                "height": 8,
                "frame_count": frame_count,
                "frames_dir": str(frames_dir),
                "frame_format": "png",
                "entries": [
                    {
                        "index": index,
                        "frame_filename": f"{index:06d}.png",
                        "timestamp_sec": index / 10.0,
                    }
                    for index in range(frame_count)
                ],
            }
        ),
        encoding="utf-8",
    )
    return frames_json, tmp_path


def build_inpaint_layout(tmp_path: Path, frame_count: int = 2) -> tuple[Path, Path]:
    """Create a synthetic ``inpaint`` stage output and return (inpainted.json, clip_root).

    Mirrors the real layout ``outputs/<clip>/inpaint/inpainted.json``; the
    common ``frames_dir`` points at the stage's own output images.
    """

    clip_root = tmp_path / "0"
    inpainted_dir = clip_root / "inpaint" / "inpainted"
    _write_frame_images(inpainted_dir, frame_count)

    inpainted_json = clip_root / "inpaint" / "inpainted.json"
    inpainted_json.write_text(
        json.dumps(
            {
                "schema_version": FRAME_MANIFEST_SCHEMA_VERSION,
                "stage": "inpaint",
                "backend": "propainter",
                "source_frames_json": str(clip_root / "process" / "frames.json"),
                "source_video": str(tmp_path / "clip.mp4"),
                "fps": 10.0,
                "width": 10,
                "height": 8,
                "frame_count": frame_count,
                "frames_dir": str(inpainted_dir),
                "frame_format": "png",
                "inpainted_dir": str(inpainted_dir),
                "entries": [
                    {
                        "index": index,
                        "frame_filename": f"{index:06d}.png",
                        "timestamp_sec": index / 10.0,
                        "has_mask": True,
                        "mask_filename": f"{index:06d}.png",
                        "inpainted_filename": f"{index:06d}.png",
                        "vis_filename": f"{index:06d}.png",
                    }
                    for index in range(frame_count)
                ],
            }
        ),
        encoding="utf-8",
    )
    return inpainted_json, tmp_path


def build_segment_layout(tmp_path: Path, frame_count: int = 2) -> tuple[Path, Path]:
    """Create a synthetic ``segment`` stage output and return (masks.json, clip_root)."""

    clip_root = tmp_path / "0"
    masks_dir = clip_root / "segment" / "masks"
    _write_frame_images(masks_dir, frame_count)

    masks_json = clip_root / "segment" / "masks.json"
    masks_json.write_text(
        json.dumps(
            {
                "schema_version": FRAME_MANIFEST_SCHEMA_VERSION,
                "stage": "segment",
                "source_frames_json": str(clip_root / "process" / "frames.json"),
                "source_video": str(tmp_path / "clip.mp4"),
                "fps": 10.0,
                "width": 10,
                "height": 8,
                "frame_count": frame_count,
                "frames_dir": str(masks_dir),
                "frame_format": "png",
                "masks_dir": str(masks_dir),
                "entries": [
                    {
                        "index": index,
                        "frame_filename": f"{index:06d}.png",
                        "timestamp_sec": index / 10.0,
                        "mask_filename": f"{index:06d}.png",
                        "vis_filename": f"{index:06d}.jpg",
                        "has_mask": True,
                        "instance_count": 3,
                        "area": 42,
                        "bbox": {
                            "min_row": 1,
                            "min_col": 2,
                            "max_row": 3,
                            "max_col": 4,
                        },
                    }
                    for index in range(frame_count)
                ],
            }
        ),
        encoding="utf-8",
    )
    return masks_json, tmp_path


def test_load_process_manifest(tmp_path: Path) -> None:
    frames_json, _ = build_process_layout(tmp_path)
    manifest = load_frame_manifest(frames_json)

    assert isinstance(manifest, FrameManifest)
    assert manifest.stage == "process"
    assert manifest.frame_count == 3
    assert manifest.fps == 10.0
    assert manifest.width == 10
    assert manifest.height == 8
    assert manifest.format == "png"
    assert len(manifest.entries) == 3
    assert manifest.entries[0].index == 0
    assert manifest.entries[0].frame_filename == "000000.png"
    assert manifest.entries[0].timestamp_sec == 0.0
    assert manifest.entries[0].path.exists()


def test_load_inpaint_manifest(tmp_path: Path) -> None:
    inpainted_json, _ = build_inpaint_layout(tmp_path)
    manifest = load_frame_manifest(inpainted_json)

    assert manifest.stage == "inpaint"
    assert manifest.frames_dir.name == "inpainted"
    assert len(manifest.entries) == 2
    # The common schema resolves images against the stage's own output dir.
    assert manifest.entries[0].path == manifest.frames_dir / "000000.png"
    assert manifest.entries[0].path.exists()


def test_load_segment_manifest(tmp_path: Path) -> None:
    masks_json, _ = build_segment_layout(tmp_path)
    manifest = load_frame_manifest(masks_json)

    assert manifest.stage == "segment"
    assert manifest.frames_dir.name == "masks"
    assert len(manifest.entries) == 2
    assert manifest.entries[0].path == manifest.frames_dir / "000000.png"
    assert manifest.entries[0].path.exists()


def test_load_frame_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_frame_manifest(tmp_path / "nope.json")


def test_load_frame_manifest_missing_frames_dir(tmp_path: Path) -> None:
    frames_json = tmp_path / "frames.json"
    frames_json.write_text(
        json.dumps({"frames_dir": str(tmp_path / "does-not-exist"), "entries": []}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        load_frame_manifest(frames_json)


def test_load_frame_manifest_empty_entries(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames_json = tmp_path / "frames.json"
    frames_json.write_text(
        json.dumps({"frames_dir": str(frames_dir), "entries": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_frame_manifest(frames_json)
