"""Tests for the video-mode (frame-by-frame) inpaint workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inpaint.workflow import InpaintVideoArgs, run_video_inpaint
from tests.test_frames import build_process_layout
from tests.test_masks import build_segment_layout

FRAME_SHAPE = (8, 10)


class FakeInpainter:
    """Duck-typed stand-in for QwenInpainter (no diffusers/torch needed).

    The workflow passes the mask-colored overlay (blue-dominant pixels inside
    the hand/arm region). The fake turns those pixels red so the edit result is
    distinguishable from a passthrough, and records every call.
    """

    def __init__(self) -> None:
        self.calls: list[int] = []

    def inpaint(self, image, output_path: Path, args) -> None:
        self.calls.append(len(self.calls))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # The workflow passes a CHW uint8 torch tensor (the masked overlay).
        rgb = image.permute(1, 2, 0).numpy()
        out = rgb.copy()
        blue = rgb[..., 2] > rgb[..., 0]  # mask color (0,0,255) blended region
        out[blue] = (255, 0, 0)
        Image.fromarray(out).save(output_path)


def _build_segmented_clip(tmp_path: Path, frame_count: int = 3) -> tuple[Path, Path]:
    """Create a full process + segment layout; return (masks_json, tmp_path)."""

    build_process_layout(tmp_path, frame_count=frame_count)
    return build_segment_layout(tmp_path, frame_count=frame_count)


def test_run_video_inpaint_with_vis(tmp_path: Path) -> None:
    masks_json, _ = _build_segmented_clip(tmp_path, frame_count=3)
    fake = FakeInpainter()

    outputs = run_video_inpaint(
        InpaintVideoArgs(masks_json=masks_json, output_root=tmp_path, vis=True),
        inpainter=fake,
    )

    assert outputs.stage_dir == tmp_path / "0" / "inpaint"
    assert outputs.inpainted_dir.exists()
    assert outputs.inpainted_vis_dir is not None and outputs.inpainted_vis_dir.exists()

    inpainted = sorted(p.name for p in outputs.inpainted_dir.glob("inpainted_*.png"))
    vis = sorted(p.name for p in outputs.inpainted_vis_dir.glob("vis_*.png"))
    assert inpainted == [
        "inpainted_000000.png",
        "inpainted_000001.png",
        "inpainted_000002.png",
    ]
    assert vis == ["vis_000000.png", "vis_000001.png", "vis_000002.png"]

    manifest = json.loads(outputs.inpainted_json_path.read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 3
    assert manifest["processed_count"] == 3
    assert manifest["masked_count"] == 2
    assert manifest["vis_enabled"] is True
    assert manifest["inpainted_vis_dir"] == str(outputs.inpainted_vis_dir)
    assert len(manifest["entries"]) == 3

    # Frame 0 had a mask -> edited by the fake (overlay turned red).
    entry = manifest["entries"][0]
    assert entry["index"] == 0
    assert entry["has_mask"] is True
    assert entry["inpainted_filename"] == "inpainted_000000.png"
    assert entry["vis_filename"] == "vis_000000.png"
    edited = np.asarray(Image.open(outputs.inpainted_dir / entry["inpainted_filename"]))
    assert edited.shape == FRAME_SHAPE + (3,)
    assert tuple(edited[3, 3]) == (255, 0, 0)

    # Frame 2 had no mask -> passthrough, original unchanged.
    passthrough = np.asarray(Image.open(outputs.inpainted_dir / "inpainted_000002.png"))
    assert tuple(passthrough[3, 3]) == (80, 80, 80)  # frame index 2 gray

    config = json.loads(outputs.config_json_path.read_text(encoding="utf-8"))
    assert config["package"]["name"] == "inpaint"
    assert config["source"]["frame_count"] == 3
    assert config["inpaint"]["vis"] is True


def test_run_video_inpaint_without_vis(tmp_path: Path) -> None:
    masks_json, _ = _build_segmented_clip(tmp_path, frame_count=2)
    outputs = run_video_inpaint(
        InpaintVideoArgs(masks_json=masks_json, output_root=tmp_path, vis=False),
        inpainter=FakeInpainter(),
    )

    assert outputs.inpainted_vis_dir is None
    assert not (outputs.stage_dir / "inpainted_vis").exists()
    manifest = json.loads(outputs.inpainted_json_path.read_text(encoding="utf-8"))
    assert manifest["vis_enabled"] is False
    assert manifest["inpainted_vis_dir"] is None
    assert all(e["vis_filename"] is None for e in manifest["entries"])


def test_run_video_inpaint_max_frames(tmp_path: Path) -> None:
    masks_json, _ = _build_segmented_clip(tmp_path, frame_count=5)
    fake = FakeInpainter()
    outputs = run_video_inpaint(
        InpaintVideoArgs(masks_json=masks_json, output_root=tmp_path, max_frames=2),
        inpainter=fake,
    )

    assert len(outputs.entries) == 2
    manifest = json.loads(outputs.inpainted_json_path.read_text(encoding="utf-8"))
    assert manifest["processed_count"] == 2
    assert [e["index"] for e in manifest["entries"]] == [0, 1]
    assert len(fake.calls) == 2


def test_run_video_inpaint_idempotent_skip(tmp_path: Path) -> None:
    """A non-overwrite re-run reuses prior per-frame outputs (no re-inference)."""

    from inpaint.qwen_inpaint import QwenInpaintArgs

    masks_json, _ = _build_segmented_clip(tmp_path, frame_count=3)
    run_video_inpaint(
        InpaintVideoArgs(masks_json=masks_json, output_root=tmp_path),
        inpainter=FakeInpainter(),
    )

    second = run_video_inpaint(
        InpaintVideoArgs(
            masks_json=masks_json,
            output_root=tmp_path,
            qwen=QwenInpaintArgs(overwrite=False),
        ),
        inpainter=FakeInpainter(),
    )
    assert second.inpainted_json_path.exists()
    assert len(second.entries) == 3
    assert (second.inpainted_dir / "inpainted_000000.png").exists()


def test_run_video_inpaint_missing_masks_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_video_inpaint(
            InpaintVideoArgs(masks_json=tmp_path / "nope.json", output_root=tmp_path),
            inpainter=FakeInpainter(),
        )


def test_run_video_inpaint_negative_max_frames(tmp_path: Path) -> None:
    masks_json, _ = _build_segmented_clip(tmp_path)
    with pytest.raises(ValueError):
        run_video_inpaint(
            InpaintVideoArgs(
                masks_json=masks_json, output_root=tmp_path, max_frames=-1
            ),
            inpainter=FakeInpainter(),
        )


def test_run_video_inpaint_requires_masks_json(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_video_inpaint(
            InpaintVideoArgs(masks_json=None, output_root=tmp_path),
            inpainter=FakeInpainter(),
        )
