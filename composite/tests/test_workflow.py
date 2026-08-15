"""End-to-end tests for composite.workflow.run_composite on synthetic data."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from composite.workflow import CompositeArgs, _write_video, run_composite


def _write_manifest(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_stage(tmp_path, frames: list[dict], fps: float = 30.0) -> dict:
    """Synthetic stage outputs under tmp_path/<clip>/ with 2 frames."""

    clip = tmp_path / "0"
    n = len(frames)

    process_dir = clip / "process"
    frames_json = process_dir / "frames.json"
    _write_manifest(
        frames_json,
        {
            "schema_version": "1.0",
            "stage": "process",
            "source_video": "fake.mp4",
            "fps": fps,
            "width": 32,
            "height": 24,
            "frame_format": "png",
            "frame_count": n,
            "frames_dir": str(process_dir / "frames"),
            "entries": [
                {"index": i, "frame_filename": f"{i:06d}.png", "timestamp_sec": i / fps}
                for i in range(n)
            ],
        },
    )
    (process_dir / "frames").mkdir(parents=True, exist_ok=True)

    inpaint_dir = clip / "inpaint"
    inpainted_dir = inpaint_dir / "inpainted"
    inpainted_dir.mkdir(parents=True, exist_ok=True)
    inpainted_json = inpaint_dir / "inpainted.json"
    _write_manifest(
        inpainted_json,
        {
            "schema_version": "1.0",
            "stage": "inpaint",
            "source_frames_json": str(frames_json),
            "fps": fps,
            "width": 32,
            "height": 24,
            "frame_format": "png",
            "frame_count": n,
            "frames_dir": str(inpainted_dir),
            "inpainted_dir": str(inpainted_dir),
            "entries": [
                {
                    "index": i,
                    "frame_filename": f"{i:06d}.png",
                    "timestamp_sec": i / fps,
                    "inpainted_filename": f"{i:06d}.png",
                }
                for i in range(n)
            ],
        },
    )

    depth_dir = clip / "depth"
    depths_dir = depth_dir / "depths"
    depths_dir.mkdir(parents=True, exist_ok=True)
    depth_json = depth_dir / "depth.json"
    _write_manifest(
        depth_json,
        {
            "schema_version": "1.0",
            "stage": "depth",
            "source_frames_json": str(frames_json),
            "fps": fps,
            "width": 32,
            "height": 24,
            "frame_format": "png",
            "frame_count": n,
            "frames_dir": str(depths_dir),
            "depths_dir": str(depths_dir),
            "entries": [
                {
                    "index": i,
                    "frame_filename": f"{i:06d}.png",
                    "timestamp_sec": i / fps,
                    "depth_filename": f"{i:06d}.npy",
                }
                for i in range(n)
            ],
        },
    )

    retarget_dir = clip / "retarget"
    rgb_dir = retarget_dir / "camera_rgb"
    camera_depth_dir = retarget_dir / "camera_depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    camera_depth_dir.mkdir(parents=True, exist_ok=True)
    camera_json = retarget_dir / "camera.json"
    _write_manifest(
        camera_json,
        {
            "schema_version": "1.0",
            "stage": "retarget_camera",
            "camera": {
                "name": "camera",
                "width": 32,
                "height": 24,
                "fovy_deg": 45.0,
                "intrinsics": [[16.0, 0.0, 16.0], [0.0, 16.0, 12.0], [0.0, 0.0, 1.0]],
            },
            "frame_count": n,
            "rgb_dir": "camera_rgb",
            "depth_dir": "camera_depth",
            "depth_enabled": True,
            "entries": [
                {
                    "index": i,
                    "rgb_filename": f"{i:06d}.png",
                    "depth_filename": f"{i:06d}.npy",
                    "timestamp_sec": i / fps,
                }
                for i in range(n)
            ],
        },
    )

    for i, frame in enumerate(frames):
        inpainted_rgb = frame["inpainted_rgb"]
        robot_rgb = frame["robot_rgb"]
        robot_depth = frame["robot_depth"]
        scene_inp = frame["scene_inp"]
        scene_orig = frame["scene_orig"]
        cv2.imwrite(str(inpainted_dir / f"{i:06d}.png"), inpainted_rgb)
        cv2.imwrite(str(rgb_dir / f"{i:06d}.png"), robot_rgb)
        np.save(camera_depth_dir / f"{i:06d}.npy", robot_depth)
        np.save(depths_dir / f"{i:06d}.npy", scene_inp)

        orig_depth_dir = clip / "depth_orig" / "depths"
        orig_depth_dir.mkdir(parents=True, exist_ok=True)
        np.save(orig_depth_dir / f"{i:06d}.npy", scene_orig)

    orig_depth_json = clip / "depth_orig" / "depth.json"
    _write_manifest(
        orig_depth_json,
        {
            "schema_version": "1.0",
            "stage": "depth",
            "source_frames_json": str(frames_json),
            "fps": fps,
            "width": 32,
            "height": 24,
            "frame_format": "png",
            "frame_count": n,
            "frames_dir": str(orig_depth_dir),
            "depths_dir": str(orig_depth_dir),
            "entries": [
                {
                    "index": i,
                    "frame_filename": f"{i:06d}.png",
                    "timestamp_sec": i / fps,
                    "depth_filename": f"{i:06d}.npy",
                }
                for i in range(n)
            ],
        },
    )

    return {
        "clip": clip,
        "inpainted_json": inpainted_json,
        "depth_json": depth_json,
        "orig_depth_json": orig_depth_json,
        "camera_json": camera_json,
    }


def _synthetic_frame_data(h=24, w=32) -> dict:
    arm_slice = (slice(4, 20), slice(8, 24))
    z_bg = np.linspace(0.6, 1.0, w, dtype=np.float32)[None, :]
    z_bg = np.repeat(z_bg, h, axis=0)
    z_bg[8:16, 20:28] = 0.2  # closer foreground hides part of the arm

    robot_depth = np.full((h, w), 70.0, dtype=np.float32)
    z_arm = np.linspace(0.35, 0.45, 16, dtype=np.float32)[:, None]
    robot_depth[arm_slice] = np.repeat(z_arm, 16, axis=1)
    scene_orig = (3.0 * z_bg + 0.1).astype(np.float32)
    scene_orig[arm_slice] = 3.0 * robot_depth[arm_slice] + 0.1
    scene_inp = (4.0 * z_bg + 0.2).astype(np.float32)

    inpainted_rgb = np.full((h, w, 3), (10, 200, 10), dtype=np.uint8)
    robot_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    robot_rgb[arm_slice] = (255, 255, 255)
    return {
        "inpainted_rgb": inpainted_rgb,
        "robot_rgb": robot_rgb,
        "robot_depth": robot_depth,
        "scene_inp": scene_inp,
        "scene_orig": scene_orig,
        "arm_slice": arm_slice,
    }


@pytest.fixture()
def stage(tmp_path) -> dict:
    frame = _synthetic_frame_data()
    return _build_stage(tmp_path, [frame, frame])


def test_run_composite_with_calibration(stage) -> None:
    args = CompositeArgs(
        inpainted_json=stage["inpainted_json"],
        depth_json=stage["depth_json"],
        camera_json=stage["camera_json"],
        calibration_depth_json=stage["orig_depth_json"],
        output_root=stage["clip"].parent,
        vis=True,
        video=False,
        overwrite=True,
        feather_px=0,
        depth_margin_frac=0.0,
        smooth_window=1,
        max_corr_samples=None,
        calibration_erode_px=0,
        poisson_blend=False,
    )
    outputs = run_composite(args)

    assert outputs.stage_dir == stage["clip"] / "composite"
    assert outputs.composite_json_path.exists()
    assert outputs.config_json_path.exists()
    assert outputs.video_path is None

    manifest = json.loads(outputs.composite_json_path.read_text(encoding="utf-8"))
    assert manifest["stage"] == "composite"
    assert manifest["depth_matching_enabled"] is True
    assert len(manifest["entries"]) == 2
    entry = manifest["entries"][0]
    assert entry["calibration"] is not None
    assert entry["visible_pixels"] < entry["arm_pixels"]
    assert entry["visible_pixels"] > 0

    frame = _synthetic_frame_data()
    out = cv2.imread(str(outputs.frames_dir / "000000.png"))
    assert out is not None
    arm_white = np.zeros(out.shape[:2], dtype=bool)
    arm_white[frame["arm_slice"]] = True
    arm_white[8:16, 20:28] = False
    assert np.all(out[arm_white] == (255, 255, 255))
    assert np.all(out[8:16, 20:28] == (10, 200, 10))
    assert (outputs.frames_vis_dir / "000000.png").exists()


def test_run_composite_fallback_without_calibration(stage) -> None:
    args = CompositeArgs(
        inpainted_json=stage["inpainted_json"],
        depth_json=stage["depth_json"],
        camera_json=stage["camera_json"],
        calibration_depth_json=None,
        vis=False,
        video=False,
        overwrite=True,
        feather_px=0,
        smooth_window=1,
        poisson_blend=False,
    )
    outputs = run_composite(args)
    manifest = json.loads(outputs.composite_json_path.read_text(encoding="utf-8"))
    assert manifest["depth_matching_enabled"] is False
    entry = manifest["entries"][0]
    assert entry["visible_pixels"] == entry["arm_pixels"]
    assert entry["calibration"] is None


def test_write_video(stage, tmp_path) -> None:
    outputs_dir = tmp_path / "out"
    frames_dir = outputs_dir / "frames"
    frames_dir.mkdir(parents=True)
    img = np.full((24, 32, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(frames_dir / "000000.png"), img)
    cv2.imwrite(str(frames_dir / "000001.png"), img)
    video_path = outputs_dir / "composite.mp4"
    _write_video(frames_dir, video_path, 30.0)
    assert video_path.exists()
    assert video_path.stat().st_size > 0
