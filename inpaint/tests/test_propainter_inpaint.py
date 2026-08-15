"""Tests for the ProPainter (streaming) backend.

The real pytorchcv weights are never loaded (no downloads): the workflow is
driven with a fake inpainter, and the inpainter itself is only exercised on
argument validation / path sequencing that does not touch the models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from inpaint.propainter.args import ProPainterInpaintArgs
from inpaint.propainter.inpainter import ProPainterInpainter
from inpaint.propainter.workflow import (
    ProPainterVideoArgs,
    run_propainter_video_inpaint,
)
from tests.test_frames import build_process_layout
from tests.test_masks import build_segment_layout

FRAME_SHAPE = (8, 10)


class FakeProPainterInpainter:
    """Duck-typed stand-in for ProPainterInpainter (no pytorchcv models).

    Records every call (aligned frame/mask/output path lists) and writes a solid
    red frame per output so results are distinguishable from the gray originals.
    Also records whether each supplied mask is all-zero (frames without a mask
    must get a generated zero mask).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[Path], list[Path], list[Path]]] = []
        self.zero_mask_flags: list[list[bool]] = []

    def inpaint_video(
        self,
        frame_paths: list[Path],
        mask_paths: list[Path],
        output_paths: list[Path],
        args: ProPainterInpaintArgs,
    ) -> None:
        self.calls.append((list(frame_paths), list(mask_paths), list(output_paths)))
        self.zero_mask_flags.append(
            [not bool(np.asarray(Image.open(p).convert("L")).any()) for p in mask_paths]
        )
        for path in output_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (FRAME_SHAPE[1], FRAME_SHAPE[0]), (255, 0, 0)).save(path)


def _build_segmented_clip(tmp_path: Path, frame_count: int = 3) -> tuple[Path, Path]:
    """Create a full process + segment layout; return (masks_json, tmp_path)."""

    build_process_layout(tmp_path, frame_count=frame_count)
    return build_segment_layout(tmp_path, frame_count=frame_count)


def test_resolve_model_paths_defaults(tmp_path: Path) -> None:
    args = ProPainterInpaintArgs(model_dir=tmp_path / "ckpts" / "propainter")

    raft, rfc, pp = args.resolve_model_paths()

    assert raft == tmp_path / "ckpts" / "propainter" / "raft-things.pth"
    assert rfc == tmp_path / "ckpts" / "propainter" / "recurrent_flow_completion.pth"
    assert pp == tmp_path / "ckpts" / "propainter" / "ProPainter.pth"


def test_resolve_model_paths_overrides_and_none_dir(tmp_path: Path) -> None:
    overrides = ProPainterInpaintArgs(
        model_dir=tmp_path / "models",
        raft_model_path=tmp_path / "raft.pth",
        pp_model_path=tmp_path / "pp.pth",
    )
    raft, rfc, pp = overrides.resolve_model_paths()
    assert raft == tmp_path / "raft.pth"
    assert rfc == tmp_path / "models" / "recurrent_flow_completion.pth"
    assert pp == tmp_path / "pp.pth"

    no_dir = ProPainterInpaintArgs(model_dir=None)
    assert no_dir.resolve_model_paths() == (None, None, None)


def test_inpainter_validation() -> None:
    with pytest.raises(ValueError, match="mask-dilation"):
        ProPainterInpainter(ProPainterInpaintArgs(mask_dilation=0))
    with pytest.raises(ValueError, match="step"):
        ProPainterInpainter(ProPainterInpaintArgs(step=0))
    with pytest.raises(ValueError, match="resize-ratio"):
        ProPainterInpainter(ProPainterInpaintArgs(resize_ratio=0.0))
    with pytest.raises(ValueError, match="raft-iters"):
        ProPainterInpainter(ProPainterInpaintArgs(raft_iters=0))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the streaming propainter package is CUDA-only",
)
def test_inpaint_video_requires_aligned_paths(tmp_path: Path) -> None:
    """The alignment check runs before any model is built (no downloads)."""

    inpainter = ProPainterInpainter(ProPainterInpaintArgs(model_dir=None))
    with pytest.raises(ValueError, match="aligned"):
        inpainter.inpaint_video(
            [tmp_path / "a.png"],
            [],
            [],
            inpainter.args,
        )


def test_run_propainter_video_inpaint_with_vis(tmp_path: Path) -> None:
    masks_json, _ = _build_segmented_clip(tmp_path, frame_count=3)
    fake = FakeProPainterInpainter()

    outputs = run_propainter_video_inpaint(
        ProPainterVideoArgs(masks_json=masks_json, output_root=tmp_path, vis=True),
        inpainter=fake,
    )

    assert outputs.stage_dir == tmp_path / "0" / "inpaint"
    assert outputs.inpainted_dir.exists()
    assert outputs.inpainted_vis_dir is not None and outputs.inpainted_vis_dir.exists()

    inpainted = sorted(p.name for p in outputs.inpainted_dir.glob("*.png"))
    vis = sorted(p.name for p in outputs.inpainted_vis_dir.glob("*.png"))
    assert inpainted == ["000000.png", "000001.png", "000002.png"]
    assert vis == ["000000.png", "000001.png", "000002.png"]

    # One call over the whole clip, aligned lists, and the unmasked frame got a
    # generated zero mask.
    assert len(fake.calls) == 1
    frame_paths, mask_paths, out_paths = fake.calls[0]
    assert len(frame_paths) == len(mask_paths) == len(out_paths) == 3
    assert fake.zero_mask_flags == [[False, False, True]]

    # All frames were written by the fake (solid red).
    edited = np.asarray(Image.open(outputs.inpainted_dir / "000000.png"))
    assert edited.shape == FRAME_SHAPE + (3,)
    assert tuple(edited[3, 3]) == (255, 0, 0)
    passthrough_like = np.asarray(Image.open(outputs.inpainted_dir / "000002.png"))
    assert tuple(passthrough_like[3, 3]) == (255, 0, 0)

    manifest = json.loads(outputs.inpainted_json_path.read_text(encoding="utf-8"))
    assert manifest["backend"] == "propainter"
    assert manifest["frame_count"] == 3
    assert manifest["processed_count"] == 3
    assert manifest["masked_count"] == 2
    assert manifest["vis_enabled"] is True
    assert len(manifest["entries"]) == 3
    assert manifest["entries"][0]["index"] == 0
    assert manifest["entries"][0]["has_mask"] is True
    assert manifest["entries"][2]["has_mask"] is False

    config = json.loads(outputs.config_json_path.read_text(encoding="utf-8"))
    assert config["package"]["name"] == "inpaint.propainter"
    assert config["backend"] == "propainter"
    assert config["inpaint"]["mask_dilation"] == 4
    assert config["inpaint"]["vis"] is True


def test_run_propainter_video_inpaint_without_vis(tmp_path: Path) -> None:
    masks_json, _ = _build_segmented_clip(tmp_path, frame_count=2)
    outputs = run_propainter_video_inpaint(
        ProPainterVideoArgs(masks_json=masks_json, output_root=tmp_path, vis=False),
        inpainter=FakeProPainterInpainter(),
    )

    assert outputs.inpainted_vis_dir is None
    assert not (outputs.stage_dir / "inpainted_vis").exists()
    manifest = json.loads(outputs.inpainted_json_path.read_text(encoding="utf-8"))
    assert manifest["vis_enabled"] is False
    assert all(e["vis_filename"] is None for e in manifest["entries"])


def test_run_propainter_video_inpaint_max_frames(tmp_path: Path) -> None:
    masks_json, _ = _build_segmented_clip(tmp_path, frame_count=5)
    fake = FakeProPainterInpainter()
    outputs = run_propainter_video_inpaint(
        ProPainterVideoArgs(masks_json=masks_json, output_root=tmp_path, max_frames=2),
        inpainter=fake,
    )

    assert len(outputs.entries) == 2
    assert [e.index for e in outputs.entries] == [0, 1]
    assert len(fake.calls[0][0]) == 2
    manifest = json.loads(outputs.inpainted_json_path.read_text(encoding="utf-8"))
    assert manifest["processed_count"] == 2


def test_run_propainter_video_inpaint_idempotent_skip(tmp_path: Path) -> None:
    """A non-overwrite re-run with every output present skips inference."""

    masks_json, _ = _build_segmented_clip(tmp_path, frame_count=3)
    run_propainter_video_inpaint(
        ProPainterVideoArgs(masks_json=masks_json, output_root=tmp_path),
        inpainter=FakeProPainterInpainter(),
    )

    second_fake = FakeProPainterInpainter()
    second = run_propainter_video_inpaint(
        ProPainterVideoArgs(
            masks_json=masks_json,
            output_root=tmp_path,
            propainter=ProPainterInpaintArgs(overwrite=False),
        ),
        inpainter=second_fake,
    )

    assert len(second.entries) == 3
    assert (second.inpainted_dir / "000000.png").exists()
    assert second_fake.calls == []


def test_run_propainter_video_inpaint_requires_masks_json(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_propainter_video_inpaint(
            ProPainterVideoArgs(masks_json=None, output_root=tmp_path),
            inpainter=FakeProPainterInpainter(),
        )


def test_run_propainter_video_inpaint_negative_max_frames(tmp_path: Path) -> None:
    masks_json, _ = _build_segmented_clip(tmp_path)
    with pytest.raises(ValueError):
        run_propainter_video_inpaint(
            ProPainterVideoArgs(
                masks_json=masks_json, output_root=tmp_path, max_frames=-1
            ),
            inpainter=FakeProPainterInpainter(),
        )


def test_run_propainter_video_inpaint_missing_masks_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_propainter_video_inpaint(
            ProPainterVideoArgs(
                masks_json=tmp_path / "nope.json", output_root=tmp_path
            ),
            inpainter=FakeProPainterInpainter(),
        )
