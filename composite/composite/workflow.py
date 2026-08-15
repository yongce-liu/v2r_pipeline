"""Video-mode compositing workflow for the composite step.

Reads the ``inpaint`` background frames, the DA3 depth of the inpainted frames
(plus optionally the DA3 depth of the original frames for calibration) and the
``retarget`` robot camera renders, and writes depth-aware composites:

.. code-block:: text

    outputs/<clip>/composite/
    ├── config.json       # effective run config
    ├── composite.json    # per-frame manifest (paths, stats, calibration)
    ├── composite.mp4     # only when --video (muxed from the frames)
    ├── frames/           # composited frames (000000.png, ...)
    └── frames_vis/       # only when --vis (inpainted | robot | composite)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from composite import __version__
from composite.calibrate import FrameCalibration, calibrate_frame, composite_frame
from composite.manifests import (
    CameraManifest,
    InpaintManifest,
    load_camera_manifest,
    load_depth_manifest,
    load_inpaint_manifest,
    load_mask_manifest,
)

OUTPUT_FILENAME_PATTERN = "{:06d}.png"
VIS_FILENAME_PATTERN = "{:06d}.png"
VIDEO_FILENAME = "composite.mp4"


@dataclass
class CompositeArgs:
    """Inputs for one composite run."""

    inpainted_json: Path | None = None
    """Path to the ``inpaint`` stage's ``inpainted.json`` (required)."""

    depth_json: Path | None = None
    """Path to the ``depth`` stage manifest computed on the *inpainted* frames
    (required; gives the scene depth behind the robot arm)."""

    camera_json: Path | None = None
    """Path to the ``retarget`` stage's ``camera.json`` (robot render RGB +
    metric depth; required)."""

    calibration_depth_json: Path | None = None
    """Optional ``depth`` stage manifest computed on the *original* frames
    (e.g. run the depth stage on ``outputs/0/process/frames.json`` with
    ``--video.output-root outputs_orig``). Enables true depth matching;
    without it the arm is composited with a plain mask."""

    masks_json: Path | None = None
    """Optional ``segment`` stage ``masks.json``. When given together with
    ``calibration_depth_json`` the arm calibration is fitted only where the
    rendered arm overlaps the human hand mask, which is much more reliable
    than fitting over the whole rendered arm."""

    output_root: Path = Path(__file__).parents[2] / "outputs"
    """Root under which ``<clip_stem>/composite/`` is created."""

    overwrite: bool = True
    """Clear existing composite outputs and recompute. With it off, prior
    per-frame outputs are reused (idempotent re-runs)."""

    vis: bool = True
    """Write side-by-side (inpainted | robot | composite) PNGs per frame."""

    video: bool = True
    """Mux the composited frames into ``composite.mp4``."""

    max_frames: int | None = None
    """Limit the number of frames processed (None = all shared frames)."""

    feather_px: int = 3
    """Gaussian feather radius (px) applied to the composited arm silhouette."""

    depth_margin_frac: float = 0.02
    """Fraction of the per-frame scene depth range used as the occlusion
    margin; prevents depth-noise pixels from flipping the arm visibility."""

    smooth_window: int = 5
    """Window for temporal gap-filling of missing per-frame calibrations."""

    max_corr_samples: int = 200_000
    """Cap on the number of correspondences used per fit per frame."""

    calibration_erode_px: int = 2
    """Erode the arm mask by this many pixels before sampling arm
    correspondences (avoids misaligned arm/hand boundary pixels)."""


@dataclass(frozen=True)
class CompositeOutputs:
    """Everything produced by one composite run."""

    clip_root: Path
    stage_dir: Path
    frames_dir: Path
    frames_vis_dir: Path | None
    composite_json_path: Path
    config_json_path: Path
    video_path: Path | None
    entries: list[dict]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _clip_stem(camera: CameraManifest) -> str:
    return camera.base_dir.parent.name


def _resolve_fps(inpainted: InpaintManifest, camera: CameraManifest) -> float | None:
    if inpainted.fps:
        return float(inpainted.fps)
    timestamps = [
        entry.timestamp_sec
        for entry in camera.entries
        if entry.timestamp_sec is not None
    ]
    if len(timestamps) > 1:
        diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
        diffs = diffs[diffs > 0]
        if diffs.size:
            return float(1.0 / np.median(diffs))
    return None


def _smooth_params(
    per_frame: list[FrameCalibration | None], window: int
) -> list[FrameCalibration | None]:
    """Fill calibration gaps with a centered temporal median.

    Per-frame fits are the most accurate estimate for their own frame (the DA3
    normalization changes frame to frame), so valid fits are kept untouched.
    Only frames whose fit failed inherit the median of their temporal
    neighbours, which keeps the video free of calibration holes.
    """

    if window <= 1 or len(per_frame) < 3:
        return per_frame

    window = min(window, len(per_frame))
    if window % 2 == 0:
        window -= 1
    half = window // 2

    def smooth_axis(values: list[float | None]) -> list[float | None]:
        valid_idx = [i for i, v in enumerate(values) if v is not None]
        if not valid_idx:
            return values
        arr = np.full(len(values), np.nan, dtype=np.float64)
        for i in valid_idx:
            arr[i] = values[i]
        out = arr.copy()
        padded = np.pad(arr, half, mode="edge")
        for i in range(len(values)):
            window_values = padded[i : i + window]
            valid_window = window_values[np.isfinite(window_values)]
            if valid_window.size:
                out[i] = float(np.median(valid_window))
        return [None if np.isnan(v) else float(v) for v in out]

    slopes = smooth_axis([c.slope if c and c.valid else None for c in per_frame])
    intercepts = smooth_axis(
        [c.intercept if c and c.valid else None for c in per_frame]
    )
    bg_slopes = smooth_axis([c.bg_slope if c and c.valid else None for c in per_frame])
    bg_intercepts = smooth_axis(
        [c.bg_intercept if c and c.valid else None for c in per_frame]
    )
    counts = [(c.n_arm, c.n_bg) if c and c.valid else (0, 0) for c in per_frame]

    smoothed: list[FrameCalibration | None] = []
    for i in range(len(per_frame)):
        if (
            slopes[i] is None
            or intercepts[i] is None
            or bg_slopes[i] is None
            or bg_intercepts[i] is None
        ):
            smoothed.append(None)
            continue
        n_arm, n_bg = counts[i]
        smoothed.append(
            FrameCalibration(
                slope=slopes[i],
                intercept=intercepts[i],
                bg_slope=bg_slopes[i],
                bg_intercept=bg_intercepts[i],
                n_arm=n_arm,
                n_bg=n_bg,
            )
        )
    return [
        per_frame[i] if per_frame[i] is not None and per_frame[i].valid else smoothed[i]
        for i in range(len(per_frame))
    ]


def _write_video(frames_dir: Path, video_path: Path, fps: float | None) -> None:
    pattern = sorted(frames_dir.glob("*.png"))
    if not pattern:
        raise ValueError(f"no composited frames to mux in {frames_dir}")

    first = cv2.imread(str(pattern[0]))
    if first is None:
        raise ValueError(f"cannot read first composite frame {pattern[0]}")
    height, width = first.shape[:2]
    rate = fps if fps and fps > 0 else 30.0

    video_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = video_path.with_name(f"{video_path.stem}.tmp.mp4")
    writer = cv2.VideoWriter(
        str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), rate, (width, height)
    )
    if not writer.isOpened():
        writer.release()
        _write_video_ffmpeg(pattern, tmp, rate)
        tmp.replace(video_path)
        logger.info(
            "[composite] wrote video (ffmpeg): {} ({}x{} @ {:.3g} fps)",
            video_path,
            width,
            height,
            rate,
        )
        return
    try:
        for frame_path in pattern:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"cannot read composite frame {frame_path}")
            writer.write(frame)
    finally:
        writer.release()
    tmp.replace(video_path)
    logger.info(
        "[composite] wrote video: {} ({}x{} @ {:.3g} fps)",
        video_path,
        width,
        height,
        rate,
    )


def _write_video_ffmpeg(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    """Mux PNG frames into an mp4 with the system ffmpeg."""

    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "cv2 video writer failed and ffmpeg is not on PATH; cannot write the video"
        )
    list_file = output_path.with_name("frames.txt")
    list_file.write_text(
        "\n".join(f"file '{frame_path.resolve()}'" for frame_path in frame_paths)
        + "\n",
        encoding="utf-8",
    )
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-r",
                str(fps),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg mux failed (rc={result.returncode}): {result.stderr[-2000:]}"
            )
    finally:
        list_file.unlink(missing_ok=True)


def _config_dict(args: CompositeArgs, frames: list[dict]) -> dict:
    return {
        "package": {"name": "composite", "version": __version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "inpainted_json": str(args.inpainted_json.expanduser().resolve()),
            "depth_json": str(args.depth_json.expanduser().resolve()),
            "calibration_depth_json": (
                str(args.calibration_depth_json.expanduser().resolve())
                if args.calibration_depth_json is not None
                else None
            ),
            "masks_json": (
                str(args.masks_json.expanduser().resolve())
                if args.masks_json is not None
                else None
            ),
            "camera_json": str(args.camera_json.expanduser().resolve()),
        },
        "composite": {
            "overwrite": args.overwrite,
            "vis": args.vis,
            "video": args.video,
            "max_frames": args.max_frames,
            "feather_px": args.feather_px,
            "depth_margin_frac": args.depth_margin_frac,
            "smooth_window": args.smooth_window,
            "max_corr_samples": args.max_corr_samples,
            "calibration_erode_px": args.calibration_erode_px,
        },
        "software": {"opencv": cv2.__version__},
        "frames": frames,
    }


def run_composite(args: CompositeArgs) -> CompositeOutputs:
    """Depth-aware compositing of robot renders into the inpainted scene."""

    if args.inpainted_json is None:
        raise ValueError("--inpainted-json is required.")
    if args.depth_json is None:
        raise ValueError("--depth-json is required.")
    if args.camera_json is None:
        raise ValueError("--camera-json is required.")
    if args.max_frames is not None and args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0.")

    inpainted = load_inpaint_manifest(args.inpainted_json)
    scene_depth = load_depth_manifest(args.depth_json)
    camera = load_camera_manifest(args.camera_json)
    calib_depth = (
        load_depth_manifest(args.calibration_depth_json)
        if args.calibration_depth_json is not None
        else None
    )
    masks = load_mask_manifest(args.masks_json) if args.masks_json is not None else None

    depth_matching = calib_depth is not None
    if not depth_matching:
        logger.warning(
            "[composite] no --calibration-depth-json given; depth matching is "
            "skipped and the arm is composited with a plain mask (run the depth "
            "stage on the original frames to enable it)."
        )

    clip_root = args.output_root.expanduser().resolve() / _clip_stem(camera)
    stage_dir = clip_root / "composite"
    frames_dir = stage_dir / "frames"
    frames_vis_dir = stage_dir / "frames_vis" if args.vis else None

    if args.overwrite:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        if frames_vis_dir is not None and frames_vis_dir.exists():
            shutil.rmtree(frames_vis_dir)
        stale_video = stage_dir / VIDEO_FILENAME
        if stale_video.exists():
            stale_video.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)
    if frames_vis_dir is not None:
        frames_vis_dir.mkdir(parents=True, exist_ok=True)

    indexes = sorted(
        set(inpainted.by_index) & set(scene_depth.by_index) & set(camera.by_index)
    )
    if calib_depth is not None:
        indexes = [i for i in indexes if i in calib_depth.by_index]
    if not indexes:
        raise ValueError("no frame index shared by all input manifests")
    if args.max_frames is not None:
        indexes = indexes[: args.max_frames]

    raw_calibrations: list[FrameCalibration | None] = []
    for index in indexes:
        if depth_matching:
            calib_entry = calib_depth.by_index[index]
            scene_orig_path = calib_depth.depth_path(calib_entry)
            scene_orig = np.load(scene_orig_path)
            robot_depth = np.load(camera.depth_path(camera.by_index[index]))
            arm = robot_depth < float(robot_depth.max()) - 1e-3
            scene_inp = np.load(scene_depth.depth_path(scene_depth.by_index[index]))
            hand_mask = None
            if masks is not None and index in masks.by_index:
                mask_entry = masks.by_index[index]
                hand_mask = (
                    cv2.imread(str(masks.mask_path(mask_entry)), cv2.IMREAD_GRAYSCALE)
                    > 0
                )
            raw_calibrations.append(
                calibrate_frame(
                    robot_depth,
                    scene_orig,
                    scene_inp,
                    arm,
                    erode_px=args.calibration_erode_px,
                    max_samples=args.max_corr_samples,
                    hand_mask=hand_mask,
                )
            )
        else:
            raw_calibrations.append(None)

    calibrations = _smooth_params(raw_calibrations, args.smooth_window)

    entries: list[dict] = []
    fps = _resolve_fps(inpainted, camera)
    for position, index in enumerate(indexes):
        inpaint_entry = inpainted.by_index[index]
        scene_entry = scene_depth.by_index[index]
        camera_entry = camera.by_index[index]

        inpainted_rgb = cv2.imread(str(inpainted.frame_path(inpaint_entry)))
        if inpainted_rgb is None:
            raise RuntimeError(f"cannot read inpainted frame {inpaint_entry.index}")
        robot_rgb = cv2.imread(str(camera.rgb_path(camera_entry)))
        if robot_rgb is None:
            raise RuntimeError(f"cannot read robot render frame {camera_entry.index}")
        robot_depth = np.load(camera.depth_path(camera_entry))
        scene_inp = np.load(scene_depth.depth_path(scene_entry))

        calibration = calibrations[position]
        out, _, visible_fraction, n_arm, n_visible = composite_frame(
            inpainted_rgb,
            robot_rgb,
            robot_depth,
            scene_inp,
            calibration,
            margin_frac=args.depth_margin_frac,
            feather_px=args.feather_px,
        )

        output_filename = OUTPUT_FILENAME_PATTERN.format(index)
        output_path = frames_dir / output_filename
        if not cv2.imwrite(str(output_path), out):
            raise RuntimeError(f"failed to write composite frame {output_path}")

        vis_filename = None
        if frames_vis_dir is not None:
            vis = np.concatenate([inpainted_rgb, robot_rgb, out], axis=1)
            vis_filename = VIS_FILENAME_PATTERN.format(index)
            if not cv2.imwrite(str(frames_vis_dir / vis_filename), vis):
                raise RuntimeError(f"failed to write composite vis frame {index}")

        entries.append(
            {
                "index": index,
                "timestamp_sec": inpaint_entry.timestamp_sec,
                "inpainted_filename": inpaint_entry.inpainted_filename,
                "robot_rgb_filename": camera_entry.rgb_filename,
                "robot_depth_filename": camera_entry.depth_filename,
                "output_filename": output_filename,
                "vis_filename": vis_filename,
                "arm_pixels": n_arm,
                "visible_pixels": n_visible,
                "visible_fraction": round(visible_fraction, 4),
                "calibration": (
                    {
                        "slope": calibration.slope,
                        "intercept": calibration.intercept,
                        "bg_slope": calibration.bg_slope,
                        "bg_intercept": calibration.bg_intercept,
                        "n_arm": calibration.n_arm,
                        "n_bg": calibration.n_bg,
                    }
                    if calibration is not None and calibration.valid
                    else None
                ),
            }
        )
        if position % 10 == 0 or position == len(indexes) - 1:
            logger.info(
                "[composite] frame {}/{} (idx {}): {}/{} arm px visible",
                position + 1,
                len(indexes),
                index,
                n_visible,
                n_arm,
            )

    manifest = {
        "schema_version": "1.0",
        "stage": "composite",
        "source_inpainted_json": str(args.inpainted_json.expanduser().resolve()),
        "source_depth_json": str(args.depth_json.expanduser().resolve()),
        "source_calibration_depth_json": (
            str(args.calibration_depth_json.expanduser().resolve())
            if args.calibration_depth_json is not None
            else None
        ),
        "source_masks_json": (
            str(args.masks_json.expanduser().resolve())
            if args.masks_json is not None
            else None
        ),
        "source_camera_json": str(args.camera_json.expanduser().resolve()),
        "fps": fps,
        "width": camera.width,
        "height": camera.height,
        "frame_format": "png",
        "frame_count": len(entries),
        "depth_matching_enabled": depth_matching,
        "frames_dir": str(frames_dir),
        "frames_vis_dir": str(frames_vis_dir) if frames_vis_dir is not None else None,
        "video_filename": VIDEO_FILENAME if args.video else None,
        "entries": entries,
    }
    composite_json_path = stage_dir / "composite.json"
    config_json_path = stage_dir / "config.json"
    _write_json(composite_json_path, manifest)
    _write_json(config_json_path, _config_dict(args, entries))

    video_path: Path | None = None
    if args.video:
        video_path = stage_dir / VIDEO_FILENAME
        _write_video(frames_dir, video_path, fps)

    logger.info(
        "[composite] complete: {} frames, depth matching {}, output {}",
        len(entries),
        "on" if depth_matching else "off",
        stage_dir,
    )
    return CompositeOutputs(
        clip_root=clip_root,
        stage_dir=stage_dir,
        frames_dir=frames_dir,
        frames_vis_dir=frames_vis_dir,
        composite_json_path=composite_json_path,
        config_json_path=config_json_path,
        video_path=video_path,
        entries=entries,
    )
