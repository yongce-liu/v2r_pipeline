"""Source-type dispatch: map an IK config ``source_type`` to a frame loader.

The retarget CLI reads the IK config, pops ``source_type``, and hands the raw
human calibration file to the loader registered for that type. Each loader
decodes the file into an iterator of frames; a frame is any object exposing the
five fields :class:`RobotRetargeter` consumes (see :class:`Frame`).

New data sources register a loader here (or monkeypatch
:data:`SOURCE_LOADERS`) and the CLI picks it up without further changes:

    SOURCE_LOADERS["smpl"] = _smpl_frames
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np


@dataclass
class Frame:
    """Minimal human-frame contract consumed by :class:`RobotRetargeter`.

    ``human_data`` maps body/hand names to ``[pos(3,), quat_wxyz(4,)]``;
    ``hand_points`` are per-side finger keypoint positions. Loaders may yield
    richer frame objects (e.g. :class:`EgoDexFrame`) as long as they provide
    these fields; ``timestamp_s`` is optional metadata used for output.
    """

    human_data: dict[str, list[np.ndarray]]
    hand_points: dict[str, np.ndarray]
    hand_confidences: dict[str, float]
    dex_hand_points: dict[str, np.ndarray]
    dex_hand_rotations: dict[str, np.ndarray]
    frame_index: int = 0
    timestamp_s: float = 0.0


LoaderFn = Callable[..., Iterator]


def _egodex_frames(
    input_path: Path,
    ik_config: dict,
    *,
    confidence_threshold: float = 0.5,
) -> Iterator:
    """Decode a single EgoDex hdf5 episode into frame objects."""
    from retarget.egodex.dataloader import (
        EgoDexDataLoader,
        loader_targets_from_ik_config,
    )

    body_names, hand_point_names, mano_keypoint_names = loader_targets_from_ik_config(
        ik_config
    )
    loader = EgoDexDataLoader(
        Path(input_path).parent,
        confidence_threshold=confidence_threshold,
        body_names=body_names,
        hand_point_names=hand_point_names,
        mano_keypoint_names=mano_keypoint_names,
    )
    episode = loader.episode_from_path(input_path)
    with loader.reader(episode) as reader:
        yield from reader.iter_frames()


SOURCE_LOADERS: dict[str, LoaderFn] = {
    "egodex": _egodex_frames,
}


def known_source_types() -> tuple[str, ...]:
    """Registered ``source_type`` values the CLI can dispatch on."""
    return tuple(sorted(SOURCE_LOADERS))


def iter_source_frames(
    source_type: str,
    input_path: Path,
    ik_config: dict,
    *,
    confidence_threshold: float = 0.5,
) -> Iterator:
    """Yield frames decoded from ``input_path`` by the ``source_type`` loader.

    ``ik_config`` is the parsed config with ``source_type`` already popped.
    """
    loader = SOURCE_LOADERS.get(source_type)
    if loader is None:
        raise ValueError(
            f"Unknown source_type {source_type!r}; known types: {known_source_types()}"
        )
    yield from loader(
        input_path,
        ik_config,
        confidence_threshold=confidence_threshold,
    )
