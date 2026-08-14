"""Video-mode (frame-by-frame) DA3 depth estimation.

Reads the ``process`` stage's ``frames.json``, estimates a dense depth map for
every frame with Depth Anything 3 (the model is loaded once and reused), and
writes a per-frame depth array, a single aggregate file (``npz``, ``pkl``, or
``json``), and a ``depth.json`` manifest. When ``vis`` is enabled a colorized
depth-map image is written for every frame too.

Output layout mirrors the ``process`` / ``segment`` stages:

.. code-block:: text

    outputs/<clip>/depth/
    ├── config.json     # effective run config (same style as process)
    ├── depth.json      # per-frame depth manifest (index / paths / depth stats)
    ├── depth.npz       # single aggregate file (all depth + intrinsics + timestamps)
    ├── depths/         # per-frame depth arrays (000000.npy, ...)
    └── depths_vis/     # only when vis=True (000000.png, ...)
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from loguru import logger

from depth import __version__
from depth.da3 import Da3Predictor
from depth.frames import FrameManifest, load_frame_manifest

DEPTH_FILENAME_PATTERN = "{:06d}.npy"
VIS_FILENAME_PATTERN = "{:06d}.png"

AggregateFormat = Literal["npz", "pkl", "json"]


@dataclass
class Da3Args:
    """Arguments for the DA3 depth predictor (checkpoint, device, resolution)."""

    model_path: Path | None = (
        Path(__file__).parents[2] / "ckpts/DA3NESTED-GIANT-LARGE-1.1"
    )
    """Path to the Depth Anything 3 checkpoint (required to run inference)."""
    device: str = "auto"
    """Torch device: ``auto``, ``cuda[:N]``, or ``cpu``."""
    process_res: int = 504
    """Longer-side resolution used internally by DA3."""
    overwrite: bool = True
    """Clear existing depth outputs and recompute. With it off, prior per-frame
    outputs are reused (idempotent re-runs)."""


@dataclass
class DepthVideoArgs:
    """Arguments for frame-by-frame depth estimation of a whole video."""

    frames_json: Path | None = None
    """Path to the ``process`` stage's ``frames.json`` (required for video mode)."""

    output_root: Path = Path(__file__).parents[2] / "outputs"
    """Root under which ``<clip_stem>/depth/`` is created."""

    vis: bool = True
    """Write a colorized depth-map image for every processed frame."""

    max_frames: int | None = None
    """Limit the number of frames processed (None = all frames in the manifest)."""

    aggregate_format: AggregateFormat = "npz"
    """Single aggregate file format: ``npz`` (default), ``pkl``, or ``json``."""

    da3: Da3Args = field(default_factory=Da3Args)
    """DA3 depth settings (checkpoint, device, resolution)."""


@dataclass(frozen=True)
class DepthEntry:
    """Per-frame depth record written into ``depth.json``."""

    index: int
    frame_filename: str
    timestamp_sec: float | None
    depth_filename: str | None
    vis_filename: str | None
    height: int | None
    width: int | None
    depth_min: float | None
    depth_max: float | None
    depth_mean: float | None
    intrinsics: list[list[float]] | None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "frame_filename": self.frame_filename,
            "timestamp_sec": self.timestamp_sec,
            "depth_filename": self.depth_filename,
            "vis_filename": self.vis_filename,
            "height": self.height,
            "width": self.width,
            "depth_min": self.depth_min,
            "depth_max": self.depth_max,
            "depth_mean": self.depth_mean,
            "intrinsics": self.intrinsics,
        }


@dataclass(frozen=True)
class DepthVideoOutputs:
    """Everything produced by one video depth run."""

    clip_root: Path
    stage_dir: Path
    depths_dir: Path
    depths_vis_dir: Path | None
    aggregate_path: Path
    depth_json_path: Path
    config_json_path: Path
    entries: list[DepthEntry]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _finite_values(depth: np.ndarray) -> np.ndarray:
    return depth[np.isfinite(depth)]


def _depth_stats(depth: np.ndarray) -> tuple[float | None, float | None, float | None]:
    finite = _finite_values(depth)
    if finite.size == 0:
        return None, None, None
    return (
        float(finite.min()),
        float(finite.max()),
        float(finite.mean()),
    )


def save_depth_vis(depth: np.ndarray, output_path: Path, overwrite: bool) -> None:
    """Write a colorized (normalized + inferno colormap) depth-map image."""

    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    finite = _finite_values(depth)
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-6:
        hi = lo + 1.0

    norm = np.clip((depth.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    if not cv2.imwrite(str(output_path), colored):
        raise RuntimeError(f"Failed to write depth visualization: {output_path}")


def _encode_array(array: np.ndarray) -> str:
    """Encode an ndarray as base64 of a gzip-compressed ``.npy`` byte stream."""

    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(gzip.compress(buffer.getvalue())).decode("ascii")


def _aggregate_dict(
    entries: list[DepthEntry],
    depths_dir: Path,
) -> dict:
    """The single aggregate payload, shared by the npz/pkl/json writers."""

    depth_arrays = [
        np.load(depths_dir / entry.depth_filename).astype(np.float32)
        for entry in entries
        if entry.depth_filename is not None
    ]
    intrinsics = np.stack(
        [
            np.asarray(entry.intrinsics, dtype=np.float32).reshape(3, 3)
            for entry in entries
            if entry.intrinsics is not None
        ],
        axis=0,
    )
    timestamps = np.asarray(
        [
            entry.timestamp_sec if entry.timestamp_sec is not None else 0.0
            for entry in entries
        ],
        dtype=np.float64,
    )
    return {
        "depth": np.stack(depth_arrays, axis=0),
        "intrinsics": intrinsics,
        "timestamps": timestamps,
        "frames": [entry.frame_filename for entry in entries],
    }


def _write_aggregate(
    path: Path,
    aggregate_format: AggregateFormat,
    entries: list[DepthEntry],
    depths_dir: Path,
) -> None:
    """Write the single aggregate file (one of npz / pkl / json)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = _aggregate_dict(entries, depths_dir)

    if aggregate_format == "npz":
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                depth=data["depth"],
                intrinsics=data["intrinsics"],
                timestamps=data["timestamps"],
            )
    elif aggregate_format == "pkl":
        with open(tmp, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    else:  # json
        payload = {
            "encoding": "base64-gzip-npy",
            "frame_count": len(entries),
            "frames": data["frames"],
            "timestamps": data["timestamps"].tolist(),
            "depth": _encode_array(data["depth"]),
            "intrinsics": _encode_array(data["intrinsics"]),
        }
        tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    tmp.replace(path)


def _load_existing_entries(depth_json: Path) -> dict[int, DepthEntry]:
    """Reuse previously written depth entries (idempotent non-overwrite runs)."""

    if not depth_json.exists():
        return {}
    try:
        data = json.loads(depth_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    existing: dict[int, DepthEntry] = {}
    for raw in data.get("entries", []):
        try:
            index = int(raw["index"])
        except (KeyError, TypeError, ValueError):
            continue
        intrinsics_raw = raw.get("intrinsics")
        intrinsics = None
        if isinstance(intrinsics_raw, list):
            try:
                intrinsics = [list(map(float, row)) for row in intrinsics_raw]
            except (TypeError, ValueError):
                intrinsics = None
        existing[index] = DepthEntry(
            index=index,
            frame_filename=str(raw.get("frame_filename", "")),
            timestamp_sec=raw.get("timestamp_sec"),
            depth_filename=raw.get("depth_filename"),
            vis_filename=raw.get("vis_filename"),
            height=raw.get("height"),
            width=raw.get("width"),
            depth_min=raw.get("depth_min"),
            depth_max=raw.get("depth_max"),
            depth_mean=raw.get("depth_mean"),
            intrinsics=intrinsics,
        )
    return existing


def _depth_manifest_dict(
    manifest: FrameManifest,
    args: DepthVideoArgs,
    depths_dir: Path,
    depths_vis_dir: Path | None,
    entries: list[DepthEntry],
) -> dict:
    return {
        "source_frames_json": str(args.frames_json.expanduser().resolve()),
        "source_video": manifest.source_video,
        "fps": manifest.fps,
        "width": manifest.width,
        "height": manifest.height,
        "frame_format": manifest.format,
        "frame_count": manifest.frame_count,
        "processed_count": len(entries),
        "depth_format": "npy",
        "depths_dir": str(depths_dir),
        "depths_vis_dir": str(depths_vis_dir) if depths_vis_dir is not None else None,
        "aggregate_format": args.aggregate_format,
        "vis_enabled": args.vis,
        "entries": [entry.to_dict() for entry in entries],
    }


def _config_dict(args: DepthVideoArgs, manifest: FrameManifest) -> dict:
    model_path = (
        str(args.da3.model_path.expanduser())
        if args.da3.model_path is not None
        else None
    )
    return {
        "package": {"name": "depth", "version": __version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "frames_json": str(args.frames_json.expanduser().resolve()),
            "source_video": manifest.source_video,
            "fps": manifest.fps,
            "width": manifest.width,
            "height": manifest.height,
            "frame_format": manifest.format,
            "frame_count": manifest.frame_count,
        },
        "depth": {
            "model_path": model_path,
            "device": args.da3.device,
            "process_res": args.da3.process_res,
            "overwrite": args.da3.overwrite,
            "vis": args.vis,
            "max_frames": args.max_frames,
            "aggregate_format": args.aggregate_format,
        },
        "software": {},
    }


def run_video_depth(
    args: DepthVideoArgs,
    predictor: Da3Predictor | None = None,
) -> DepthVideoOutputs:
    """Estimate depth for every frame of a process stage output, frame by frame.

    ``args.frames_json`` is read from the ``process`` stage. The DA3 model is
    loaded once and reused for all frames. When ``predictor`` is provided (e.g.
    a test double), it must expose ``predict_depth_arrays(image_path,
    process_res) -> (depth, intrinsics)``.
    """
    if args.max_frames is not None and args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0.")
    if args.aggregate_format not in ("npz", "pkl", "json"):
        raise ValueError(
            f"--aggregate-format must be 'npz', 'pkl', or 'json', got {args.aggregate_format}"
        )
    if args.frames_json is None:
        raise ValueError("--video.frames-json is required for video mode.")

    frames_json = args.frames_json.expanduser().resolve()
    manifest = load_frame_manifest(frames_json)

    clip_stem = frames_json.parent.parent.name  # outputs/<clip>/process/frames.json
    clip_root = args.output_root.expanduser().resolve() / clip_stem
    stage_dir = clip_root / "depth"
    depths_dir = stage_dir / "depths"
    depths_vis_dir = stage_dir / "depths_vis" if args.vis else None
    aggregate_path = stage_dir / f"depth.{args.aggregate_format}"

    if args.da3.overwrite:
        if depths_dir.exists():
            shutil.rmtree(depths_dir)
        if depths_vis_dir is not None and depths_vis_dir.exists():
            shutil.rmtree(depths_vis_dir)
        for ext in ("npz", "pkl", "json"):
            stale = stage_dir / f"depth.{ext}"
            if stale.exists():
                stale.unlink()
    depths_dir.mkdir(parents=True, exist_ok=True)
    if depths_vis_dir is not None:
        depths_vis_dir.mkdir(parents=True, exist_ok=True)

    selected = manifest.entries
    if args.max_frames is not None:
        selected = selected[: args.max_frames]

    active_predictor = predictor or Da3Predictor(
        model_path=args.da3.model_path,
        device=args.da3.device,
    )
    existing = _load_existing_entries(stage_dir / "depth.json")

    entries: list[DepthEntry] = []
    for frame in selected:
        depth_filename = DEPTH_FILENAME_PATTERN.format(frame.index)
        vis_filename = (
            VIS_FILENAME_PATTERN.format(frame.index)
            if depths_vis_dir is not None
            else None
        )
        depth_path = depths_dir / depth_filename
        vis_path = (
            depths_vis_dir / vis_filename
            if depths_vis_dir is not None and vis_filename is not None
            else None
        )

        prior = existing.get(frame.index)
        if (
            not args.da3.overwrite
            and depth_path.exists()
            and prior is not None
            and (vis_path is None or vis_path.exists())
        ):
            entries.append(prior)
            continue

        depth, intrinsics = active_predictor.predict_depth_arrays(
            image_path=frame.path,
            process_res=args.da3.process_res,
        )
        np.save(depth_path, depth.astype(np.float32))
        if vis_path is not None:
            save_depth_vis(depth, vis_path, overwrite=args.da3.overwrite)

        depth_min, depth_max, depth_mean = _depth_stats(depth)
        entries.append(
            DepthEntry(
                index=frame.index,
                frame_filename=frame.filename,
                timestamp_sec=frame.timestamp_sec,
                depth_filename=depth_filename,
                vis_filename=vis_filename,
                height=int(depth.shape[0]),
                width=int(depth.shape[1]),
                depth_min=depth_min,
                depth_max=depth_max,
                depth_mean=depth_mean,
                intrinsics=intrinsics.astype(float).tolist(),
            )
        )

    if not entries:
        raise RuntimeError("No depth entries produced (empty frame manifest).")

    _write_json(
        stage_dir / "depth.json",
        _depth_manifest_dict(manifest, args, depths_dir, depths_vis_dir, entries),
    )
    _write_json(stage_dir / "config.json", _config_dict(args, manifest))
    _write_aggregate(aggregate_path, args.aggregate_format, entries, depths_dir)

    logger.info(
        "[depth] Done: processed={} vis={} aggregate={} out={}",
        len(entries),
        args.vis,
        aggregate_path.name,
        stage_dir,
    )

    return DepthVideoOutputs(
        clip_root=clip_root,
        stage_dir=stage_dir,
        depths_dir=depths_dir,
        depths_vis_dir=depths_vis_dir,
        aggregate_path=aggregate_path,
        depth_json_path=stage_dir / "depth.json",
        config_json_path=stage_dir / "config.json",
        entries=entries,
    )
