"""Tests for the video-mode (frame-by-frame) depth workflow."""

from __future__ import annotations

import base64
import gzip
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from depth.workflow import DepthVideoArgs, run_video_depth
from tests.test_frames import build_process_layout

FRAME_SHAPE = (8, 10)


class FakePredictor:
    """Duck-typed stand-in for Da3Predictor (no DA3/torch needed)."""

    def __init__(self, per_frame: bool = False) -> None:
        self.per_frame = per_frame
        self.calls: list[Path] = []

    def predict_depth_arrays(
        self, image_path: Path, process_res: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(image_path)
        h, w = FRAME_SHAPE
        # Vary the depth per frame so per-frame arrays and the aggregate
        # clearly differ across frames.
        frame_no = int(image_path.name.split("_")[1].split(".")[0])
        depth = np.full((h, w), frame_no, dtype=np.float32)
        intrinsics = np.array(
            [[w, 0.0, w / 2], [0.0, w, h / 2], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return depth, intrinsics


def _read_npz_values(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def test_run_video_depth_with_vis_npz(tmp_path: Path) -> None:
    frames_json, _ = build_process_layout(tmp_path, frame_count=3)
    fake = FakePredictor()

    outputs = run_video_depth(
        DepthVideoArgs(
            frames_json=frames_json,
            output_root=tmp_path,
            vis=True,
            aggregate_format="npz",
        ),
        predictor=fake,
    )

    assert outputs.stage_dir == tmp_path / "0" / "depth"
    assert outputs.depths_dir.exists()
    assert outputs.depths_vis_dir is not None and outputs.depths_vis_dir.exists()

    depths = sorted(p.name for p in outputs.depths_dir.glob("depth_*.npy"))
    vis = sorted(p.name for p in outputs.depths_vis_dir.glob("vis_*.png"))
    assert depths == ["depth_000000.npy", "depth_000001.npy", "depth_000002.npy"]
    assert vis == ["vis_000000.png", "vis_000001.png", "vis_000002.png"]

    manifest = json.loads(outputs.depth_json_path.read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 3
    assert manifest["processed_count"] == 3
    assert manifest["vis_enabled"] is True
    assert manifest["aggregate_format"] == "npz"
    assert manifest["depths_vis_dir"] == str(outputs.depths_vis_dir)
    assert len(manifest["entries"]) == 3

    entry = manifest["entries"][0]
    assert entry["index"] == 0
    assert entry["frame_filename"] == "frame_000000.png"
    assert entry["depth_filename"] == "depth_000000.npy"
    assert entry["vis_filename"] == "vis_000000.png"
    assert entry["height"] == 8
    assert entry["width"] == 10
    assert entry["depth_min"] == 0.0
    assert entry["depth_max"] == 0.0
    assert entry["depth_mean"] == 0.0
    assert entry["intrinsics"] == [
        [10.0, 0.0, 5.0],
        [0.0, 10.0, 4.0],
        [0.0, 0.0, 1.0],
    ]

    config = json.loads(outputs.config_json_path.read_text(encoding="utf-8"))
    assert config["package"]["name"] == "depth"
    assert config["source"]["frame_count"] == 3
    assert config["depth"]["vis"] is True
    assert config["depth"]["aggregate_format"] == "npz"

    # The aggregate file matches the per-frame arrays.
    agg = _read_npz_values(outputs.aggregate_path)
    assert agg["depth"].shape == (3, 8, 10)
    assert agg["depth"][1, 0, 0] == pytest.approx(1.0)
    assert agg["intrinsics"].shape == (3, 3, 3)
    assert np.allclose(agg["intrinsics"][0], entry["intrinsics"])
    assert agg["timestamps"].shape == (3,)

    # The predictor was reused for every frame (model loaded once).
    assert len(fake.calls) == 3


def test_run_video_depth_without_vis_pkl(tmp_path: Path) -> None:
    frames_json, _ = build_process_layout(tmp_path, frame_count=2)
    outputs = run_video_depth(
        DepthVideoArgs(
            frames_json=frames_json,
            output_root=tmp_path,
            vis=False,
            aggregate_format="pkl",
        ),
        predictor=FakePredictor(),
    )

    assert outputs.depths_vis_dir is None
    assert not (outputs.stage_dir / "depths_vis").exists()
    manifest = json.loads(outputs.depth_json_path.read_text(encoding="utf-8"))
    assert manifest["vis_enabled"] is False
    assert manifest["depths_vis_dir"] is None
    assert all(e["vis_filename"] is None for e in manifest["entries"])

    data = pickle.loads(outputs.aggregate_path.read_bytes())
    assert data["depth"].shape == (2, 8, 10)
    assert data["intrinsics"].shape == (2, 3, 3)
    assert data["timestamps"].shape == (2,)
    assert data["frames"] == ["frame_000000.png", "frame_000001.png"]


def test_run_video_depth_json_aggregate(tmp_path: Path) -> None:
    frames_json, _ = build_process_layout(tmp_path, frame_count=2)
    outputs = run_video_depth(
        DepthVideoArgs(
            frames_json=frames_json, output_root=tmp_path, aggregate_format="json"
        ),
        predictor=FakePredictor(),
    )

    payload = json.loads(outputs.aggregate_path.read_text(encoding="utf-8"))
    assert payload["encoding"] == "base64-gzip-npy"
    assert payload["frame_count"] == 2
    assert payload["frames"] == ["frame_000000.png", "frame_000001.png"]

    raw = base64.b64decode(payload["depth"])
    depth = np.load(io.BytesIO(gzip.decompress(raw)), allow_pickle=False)
    assert depth.shape == (2, 8, 10)
    assert depth[1, 0, 0] == pytest.approx(1.0)

    raw_i = base64.b64decode(payload["intrinsics"])
    intrinsics = np.load(io.BytesIO(gzip.decompress(raw_i)), allow_pickle=False)
    assert intrinsics.shape == (2, 3, 3)


def test_run_video_depth_max_frames(tmp_path: Path) -> None:
    frames_json, _ = build_process_layout(tmp_path, frame_count=5)
    fake = FakePredictor()
    outputs = run_video_depth(
        DepthVideoArgs(frames_json=frames_json, output_root=tmp_path, max_frames=2),
        predictor=fake,
    )

    assert len(outputs.entries) == 2
    manifest = json.loads(outputs.depth_json_path.read_text(encoding="utf-8"))
    assert manifest["processed_count"] == 2
    assert [e["index"] for e in manifest["entries"]] == [0, 1]
    assert len(fake.calls) == 2


def test_run_video_depth_idempotent_skip(tmp_path: Path) -> None:
    """A non-overwrite re-run reuses prior per-frame outputs (no re-inference)."""

    from depth.workflow import Da3Args

    frames_json, _ = build_process_layout(tmp_path, frame_count=3)
    fake = FakePredictor()
    run_video_depth(
        DepthVideoArgs(frames_json=frames_json, output_root=tmp_path), predictor=fake
    )
    assert len(fake.calls) == 3

    second = run_video_depth(
        DepthVideoArgs(
            frames_json=frames_json,
            output_root=tmp_path,
            da3=Da3Args(overwrite=False),
        ),
        predictor=FakePredictor(),
    )
    assert second.depth_json_path.exists()
    assert len(second.entries) == 3
    # depth_000000.npy came from the first run; no re-inference happened.
    first_depth = np.load(second.depths_dir / "depth_000000.npy")
    assert first_depth[0, 0] == pytest.approx(0.0)
    assert second.depths_dir.glob("depth_*.npy")


def test_run_video_depth_missing_frames_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_video_depth(
            DepthVideoArgs(frames_json=tmp_path / "nope.json", output_root=tmp_path),
            predictor=FakePredictor(),
        )


def test_run_video_depth_negative_max_frames(tmp_path: Path) -> None:
    frames_json, _ = build_process_layout(tmp_path)
    with pytest.raises(ValueError):
        run_video_depth(
            DepthVideoArgs(
                frames_json=frames_json, output_root=tmp_path, max_frames=-1
            ),
            predictor=FakePredictor(),
        )


def test_run_video_depth_bad_aggregate_format(tmp_path: Path) -> None:
    frames_json, _ = build_process_layout(tmp_path)
    with pytest.raises(ValueError):
        run_video_depth(
            DepthVideoArgs(
                frames_json=frames_json, output_root=tmp_path, aggregate_format="h5"
            ),
            predictor=FakePredictor(),
        )
