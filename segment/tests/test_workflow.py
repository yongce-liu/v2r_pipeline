"""Tests for the video-mode (frame-by-frame) segmentation workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from segment.workflow import (
    SegmentVideoArgs,
    SegmentVideoOutputs,
    run_video_segment,
)
from tests.test_frames import build_process_layout


class FakeGenerator:
    """Duck-typed stand-in for Sam3MaskGenerator (no SAM3/torch needed)."""

    def __init__(
        self,
        mask: np.ndarray,
        instance_count: int = 1,
        empty_indices: set[int] | None = None,
    ) -> None:
        self._mask = mask
        self._instance_count = instance_count
        self._empty_indices = empty_indices or set()
        self.segment_calls: list[tuple[int, int]] = []

    def segment(
        self, frame_rgb: np.ndarray, text_prompt: str
    ) -> tuple[np.ndarray, int]:
        self.segment_calls.append(frame_rgb.shape[:2])
        return self._mask.copy(), self._instance_count


def _make_fake_generator(frame_shape=(8, 10), instance_count=1):
    mask = np.zeros(frame_shape, dtype=np.uint8)
    mask[2:6, 1:5] = 255
    return FakeGenerator(mask, instance_count)


def test_run_video_segment_with_vis(tmp_path: Path) -> None:
    frames_json, _clip_root = build_process_layout(tmp_path, frame_count=3)
    fake = _make_fake_generator()

    outputs = run_video_segment(
        SegmentVideoArgs(
            frames_json=frames_json,
            output_root=tmp_path,
            vis=True,
        ),
        generator=fake,
    )

    assert isinstance(outputs, SegmentVideoOutputs)
    assert outputs.stage_dir == tmp_path / "0" / "segment"
    assert outputs.masks_dir.exists()
    assert outputs.masks_vis_dir is not None and outputs.masks_vis_dir.exists()

    masks = sorted(p.name for p in outputs.masks_dir.glob("mask_*.png"))
    vis = sorted(p.name for p in outputs.masks_vis_dir.glob("vis_*.jpg"))
    assert masks == ["mask_000000.png", "mask_000001.png", "mask_000002.png"]
    assert vis == ["vis_000000.jpg", "vis_000001.jpg", "vis_000002.jpg"]

    manifest = json.loads(outputs.masks_json_path.read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 3
    assert manifest["processed_count"] == 3
    assert manifest["vis_enabled"] is True
    assert manifest["masks_vis_dir"] == str(outputs.masks_vis_dir)
    assert len(manifest["entries"]) == 3

    entry = manifest["entries"][0]
    assert entry["index"] == 0
    assert entry["frame_filename"] == "frame_000000.png"
    assert entry["mask_filename"] == "mask_000000.png"
    assert entry["vis_filename"] == "vis_000000.jpg"
    assert entry["has_mask"] is True
    assert entry["instance_count"] == 1
    assert entry["area"] == 4 * 4
    assert entry["bbox"] == {"min_row": 2, "min_col": 1, "max_row": 5, "max_col": 4}

    config = json.loads(outputs.config_json_path.read_text(encoding="utf-8"))
    assert config["package"]["name"] == "segment"
    assert config["source"]["frame_count"] == 3
    assert config["segment"]["vis"] is True
    assert config["segment"]["text_prompt"] == "人手"
    assert config["segment"]["mask_color_rgb"] == [0, 0, 255]

    # The generator was reused for every frame (model loaded once).
    assert len(fake.segment_calls) == 3


def test_run_video_segment_without_vis(tmp_path: Path) -> None:
    frames_json, _clip_root = build_process_layout(tmp_path, frame_count=2)
    outputs = run_video_segment(
        SegmentVideoArgs(frames_json=frames_json, output_root=tmp_path, vis=False),
        generator=_make_fake_generator(),
    )

    assert outputs.masks_vis_dir is None
    assert not (outputs.stage_dir / "masks_vis").exists()
    manifest = json.loads(outputs.masks_json_path.read_text(encoding="utf-8"))
    assert manifest["vis_enabled"] is False
    assert manifest["masks_vis_dir"] is None
    assert all(e["vis_filename"] is None for e in manifest["entries"])


def test_run_video_segment_max_frames(tmp_path: Path) -> None:
    frames_json, _clip_root = build_process_layout(tmp_path, frame_count=5)
    fake = _make_fake_generator()
    outputs = run_video_segment(
        SegmentVideoArgs(frames_json=frames_json, output_root=tmp_path, max_frames=2),
        generator=fake,
    )

    assert len(outputs.entries) == 2
    manifest = json.loads(outputs.masks_json_path.read_text(encoding="utf-8"))
    assert manifest["processed_count"] == 2
    assert [e["index"] for e in manifest["entries"]] == [0, 1]


def test_run_video_segment_empty_mask(tmp_path: Path) -> None:
    frames_json, _clip_root = build_process_layout(tmp_path, frame_count=1)
    fake = FakeGenerator(np.zeros((8, 10), dtype=np.uint8), instance_count=0)
    outputs = run_video_segment(
        SegmentVideoArgs(frames_json=frames_json, output_root=tmp_path),
        generator=fake,
    )

    entry = outputs.entries[0]
    assert not entry.has_mask
    assert entry.instance_count == 0
    assert entry.area == 0
    assert entry.bbox is None


def test_run_video_segment_missing_frames_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_video_segment(
            SegmentVideoArgs(frames_json=tmp_path / "nope.json", output_root=tmp_path),
            generator=_make_fake_generator(),
        )


def test_run_video_segment_negative_max_frames(tmp_path: Path) -> None:
    frames_json, _ = build_process_layout(tmp_path)
    with pytest.raises(ValueError):
        run_video_segment(
            SegmentVideoArgs(
                frames_json=frames_json, output_root=tmp_path, max_frames=-1
            ),
            generator=_make_fake_generator(),
        )
