"""Tests for the MiniMax-Remover video backend.

The 2.7 GB checkpoint and custom upstream model are never loaded here. Workflow
tests use a duck-typed inpainter; constructor tests only validate local paths
and arguments before model construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from inpaint.minimax.args import UPSTREAM_REVISION, MiniMaxRemoverInpaintArgs
from inpaint.minimax.inpainter import MiniMaxRemoverInpainter, next_temporal_frame_count
from inpaint.minimax.upstream import Minimax_Remover_Pipeline, Transformer3DModel
from inpaint.minimax.workflow import (
    MiniMaxVideoArgs,
    run_minimax_video_inpaint,
)
from tests.test_frames import build_process_layout
from tests.test_masks import build_segment_layout

FRAME_SHAPE = (8, 10)


class FakeMiniMaxInpainter:
    """Record aligned path lists and write red outputs without loading models."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[Path], list[Path], list[Path]]] = []
        self.zero_mask_flags: list[list[bool]] = []

    def inpaint_video(
        self,
        frame_paths: list[Path],
        mask_paths: list[Path],
        output_paths: list[Path],
        args: MiniMaxRemoverInpaintArgs,
    ) -> None:
        self.calls.append((list(frame_paths), list(mask_paths), list(output_paths)))
        self.zero_mask_flags.append(
            [
                not bool(np.asarray(Image.open(path).convert("L")).any())
                for path in mask_paths
            ]
        )
        for output_path in output_paths:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (FRAME_SHAPE[1], FRAME_SHAPE[0]),
                (255, 0, 0),
            ).save(output_path)


def _build_segmented_clip(tmp_path: Path, frame_count: int = 3) -> tuple[Path, Path]:
    build_process_layout(tmp_path, frame_count=frame_count)
    return build_segment_layout(tmp_path, frame_count=frame_count)


def _build_local_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "model"
    for component in ("vae", "transformer", "scheduler"):
        (model_path / component).mkdir(parents=True, exist_ok=True)
    return model_path


def test_vendored_upstream_imports() -> None:
    assert issubclass(Transformer3DModel, torch.nn.Module)
    assert Minimax_Remover_Pipeline.__name__ == "Minimax_Remover_Pipeline"
    assert UPSTREAM_REVISION == "28e12b450d8a72a7547b86940a4985e6ad90d75b"


def test_next_temporal_frame_count() -> None:
    assert next_temporal_frame_count(1) == 1
    assert next_temporal_frame_count(2) == 5
    assert next_temporal_frame_count(4) == 5
    assert next_temporal_frame_count(5) == 5
    assert next_temporal_frame_count(6) == 9
    with pytest.raises(ValueError):
        next_temporal_frame_count(0)


def test_inference_size_auto_orientation() -> None:
    inpainter = MiniMaxRemoverInpainter.__new__(MiniMaxRemoverInpainter)
    inpainter.args = MiniMaxRemoverInpaintArgs(height=480, width=832)

    assert inpainter._inference_size((1080, 1920)) == (480, 832)
    assert inpainter._inference_size((1920, 1080)) == (832, 480)
    inpainter.args.auto_orientation = False
    assert inpainter._inference_size((1920, 1080)) == (480, 832)


def test_inpainter_argument_validation(tmp_path: Path) -> None:
    model_path = _build_local_model(tmp_path)
    with pytest.raises(ValueError, match="divisible by 16"):
        MiniMaxRemoverInpainter(
            MiniMaxRemoverInpaintArgs(
                model_path=model_path,
                height=481,
            )
        )
    with pytest.raises(ValueError, match="steps"):
        MiniMaxRemoverInpainter(
            MiniMaxRemoverInpaintArgs(
                model_path=model_path,
                steps=0,
            )
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the upstream MiniMax-Remover pipeline is CUDA-only",
)
def test_inpaint_video_requires_aligned_paths(tmp_path: Path) -> None:
    model_path = _build_local_model(tmp_path)
    inpainter = MiniMaxRemoverInpainter(
        MiniMaxRemoverInpaintArgs(model_path=model_path)
    )
    with pytest.raises(ValueError, match="aligned"):
        inpainter.inpaint_video([tmp_path / "a.png"], [], [], inpainter.args)


def test_run_minimax_video_inpaint_with_vis(tmp_path: Path) -> None:
    masks_json, _ = _build_segmented_clip(tmp_path, frame_count=3)
    fake = FakeMiniMaxInpainter()

    outputs = run_minimax_video_inpaint(
        MiniMaxVideoArgs(masks_json=masks_json, output_root=tmp_path, vis=True),
        inpainter=fake,
    )

    assert outputs.stage_dir == tmp_path / "0" / "inpaint"
    assert outputs.inpainted_vis_dir is not None
    assert outputs.inpainted_vis_dir.exists()
    assert sorted(path.name for path in outputs.inpainted_dir.glob("*.png")) == [
        "000000.png",
        "000001.png",
        "000002.png",
    ]
    assert sorted(path.name for path in outputs.inpainted_vis_dir.glob("*.png")) == [
        "000000.png",
        "000001.png",
        "000002.png",
    ]

    assert len(fake.calls) == 1
    frame_paths, mask_paths, output_paths = fake.calls[0]
    assert len(frame_paths) == len(mask_paths) == len(output_paths) == 3
    assert fake.zero_mask_flags == [[False, False, True]]

    manifest = json.loads(outputs.inpainted_json_path.read_text(encoding="utf-8"))
    assert manifest["backend"] == "minimax"
    assert manifest["frame_count"] == 3
    assert manifest["processed_count"] == 3
    assert manifest["masked_count"] == 2
    assert manifest["vis_enabled"] is True
    assert manifest["entries"][0]["index"] == 0
    assert manifest["entries"][2]["has_mask"] is False

    config = json.loads(outputs.config_json_path.read_text(encoding="utf-8"))
    assert config["package"]["name"] == "inpaint.minimax"
    assert config["backend"] == "minimax"
    assert config["inpaint"]["height"] == 480
    assert config["inpaint"]["width"] == 832
    assert config["inpaint"]["auto_orientation"] is True
    assert config["inpaint"]["mask_dilation"] == 12
    assert config["inpaint"]["post_mask_dilation"] == 24
    assert config["inpaint"]["upstream_revision"] == UPSTREAM_REVISION
    assert config["inpaint"]["dtype"] == "float16"
    assert config["inpaint"]["vis"] is True


def test_run_minimax_video_inpaint_without_vis(tmp_path: Path) -> None:
    masks_json, _ = _build_segmented_clip(tmp_path, frame_count=2)
    outputs = run_minimax_video_inpaint(
        MiniMaxVideoArgs(masks_json=masks_json, output_root=tmp_path, vis=False),
        inpainter=FakeMiniMaxInpainter(),
    )

    assert outputs.inpainted_vis_dir is None
    assert not (outputs.stage_dir / "inpainted_vis").exists()
    manifest = json.loads(outputs.inpainted_json_path.read_text(encoding="utf-8"))
    assert manifest["vis_enabled"] is False
    assert all(entry["vis_filename"] is None for entry in manifest["entries"])


def test_run_minimax_video_inpaint_max_frames(tmp_path: Path) -> None:
    masks_json, _ = _build_segmented_clip(tmp_path, frame_count=5)
    fake = FakeMiniMaxInpainter()
    outputs = run_minimax_video_inpaint(
        MiniMaxVideoArgs(masks_json=masks_json, output_root=tmp_path, max_frames=2),
        inpainter=fake,
    )

    assert [entry.index for entry in outputs.entries] == [0, 1]
    assert len(fake.calls[0][0]) == 2
    manifest = json.loads(outputs.inpainted_json_path.read_text(encoding="utf-8"))
    assert manifest["processed_count"] == 2


def test_run_minimax_video_inpaint_idempotent_skip(tmp_path: Path) -> None:
    masks_json, _ = _build_segmented_clip(tmp_path, frame_count=3)
    run_minimax_video_inpaint(
        MiniMaxVideoArgs(masks_json=masks_json, output_root=tmp_path),
        inpainter=FakeMiniMaxInpainter(),
    )

    second_fake = FakeMiniMaxInpainter()
    second = run_minimax_video_inpaint(
        MiniMaxVideoArgs(
            masks_json=masks_json,
            output_root=tmp_path,
            minimax=MiniMaxRemoverInpaintArgs(overwrite=False),
        ),
        inpainter=second_fake,
    )

    assert len(second.entries) == 3
    assert second_fake.calls == []


def test_run_minimax_video_inpaint_requires_masks_json(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_minimax_video_inpaint(
            MiniMaxVideoArgs(masks_json=None, output_root=tmp_path),
            inpainter=FakeMiniMaxInpainter(),
        )


def test_run_minimax_video_inpaint_negative_max_frames(tmp_path: Path) -> None:
    masks_json, _ = _build_segmented_clip(tmp_path)
    with pytest.raises(ValueError):
        run_minimax_video_inpaint(
            MiniMaxVideoArgs(
                masks_json=masks_json,
                output_root=tmp_path,
                max_frames=-1,
            ),
            inpainter=FakeMiniMaxInpainter(),
        )


def test_run_minimax_video_inpaint_missing_masks_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_minimax_video_inpaint(
            MiniMaxVideoArgs(masks_json=tmp_path / "nope.json", output_root=tmp_path),
            inpainter=FakeMiniMaxInpainter(),
        )
