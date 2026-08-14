"""EgoDex HDF5 discovery and frame decoding."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from scipy.spatial.transform import Rotation as R

ARKIT_TO_GMR = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

AVP_WRIST_TO_DEX_OPT_ROTATIONS = {
    "left": ARKIT_TO_GMR
    @ R.from_euler("xyz", [0.0, np.pi, 0.0]).as_matrix()
    @ R.from_euler("xyz", [-np.pi / 2.0, 0.0, np.pi / 2.0]).as_matrix(),
    "right": ARKIT_TO_GMR
    @ np.eye(3, dtype=np.float64)
    @ R.from_euler("xyz", [np.pi / 2.0, 0.0, -np.pi / 2.0]).as_matrix(),
}


@dataclass(frozen=True)
class EgoDexEpisodeInfo:
    hdf5_path: Path


@dataclass
class EgoDexFrame:
    frame_index: int
    timestamp_s: float
    human_data: dict[str, list[np.ndarray]]
    hand_points: dict[str, np.ndarray]
    hand_confidences: dict[str, float]
    hand_rotations: dict[str, np.ndarray]
    raw_hand_rotations: dict[str, np.ndarray]
    dex_hand_points: dict[str, np.ndarray]
    dex_hand_rotations: dict[str, np.ndarray]


class EgoDexDataLoader:
    """Discover EgoDex episodes and hand out per-episode readers."""

    def __init__(
        self,
        src_dir: str | Path,
        *,
        confidence_threshold: float = 0.5,
        body_names: Iterable[str] | None = None,
        hand_point_names: Iterable[str] | None = None,
        mano_keypoint_names: Iterable[str] | None = None,
    ) -> None:
        self.src_dir = Path(src_dir)
        self.confidence_threshold = confidence_threshold
        self.body_names = tuple(dict.fromkeys(body_names or ()))
        self.hand_point_names = tuple(dict.fromkeys(hand_point_names or ()))
        self.mano_keypoint_names = tuple(dict.fromkeys(mano_keypoint_names or ()))
        if self.mano_keypoint_names and len(self.mano_keypoint_names) != 21:
            raise ValueError(
                "mano_keypoint_names must have exactly 21 entries, got "
                f"{len(self.mano_keypoint_names)}"
            )

    def reader(
        self, info: EgoDexEpisodeInfo, fps: float = 30.0
    ) -> "EgoDexEpisodeReader":
        return EgoDexEpisodeReader(
            info,
            fps=fps,
            confidence_threshold=self.confidence_threshold,
            body_names=self.body_names,
            hand_point_names=self.hand_point_names,
            mano_keypoint_names=self.mano_keypoint_names,
        )

    def episode_from_path(self, hdf5_path: str | Path) -> EgoDexEpisodeInfo:
        """Wrap a single episode hdf5 path as :class:`EgoDexEpisodeInfo`."""
        hdf5_path = Path(hdf5_path)
        if not hdf5_path.is_file():
            raise FileNotFoundError(f"EgoDex hdf5 not found: {hdf5_path}")
        return EgoDexEpisodeInfo(hdf5_path=hdf5_path)


class EgoDexEpisodeReader:
    def __init__(
        self,
        info: EgoDexEpisodeInfo,
        fps: float = 30.0,
        confidence_threshold: float = 0.5,
        body_names: Iterable[str] | None = None,
        hand_point_names: Iterable[str] | None = None,
        mano_keypoint_names: Iterable[str] | None = None,
    ) -> None:
        self.info = info
        self.fps = fps
        self.confidence_threshold = confidence_threshold
        self.body_names = tuple(dict.fromkeys(body_names or ()))
        self.hand_point_names = tuple(dict.fromkeys(hand_point_names or ()))
        self.mano_keypoint_names = tuple(dict.fromkeys(mano_keypoint_names or ()))
        if self.mano_keypoint_names and len(self.mano_keypoint_names) != 21:
            raise ValueError(
                "mano_keypoint_names must have exactly 21 entries, got "
                f"{len(self.mano_keypoint_names)}"
            )
        self._h5 = None
        self._datasets: dict[str, object] = {}
        # First frame seeds the history from its own real transforms.
        self._previous_human: dict[str, list[np.ndarray]] = {}
        self._previous_hand_rotations: dict[str, np.ndarray] = {}

    def __enter__(self) -> "EgoDexEpisodeReader":
        import h5py

        self._h5 = h5py.File(self.info.hdf5_path, "r")
        self._datasets = self._resolve_datasets()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None

    @property
    def frame_count(self) -> int:
        if not self._datasets:
            return 0
        return min(
            len(dataset) for dataset in self._datasets.values() if len(dataset) > 0
        )

    def iter_frames(self) -> Iterator[EgoDexFrame]:
        for frame_index in range(self.frame_count):
            human_data = self._read_upper_body(frame_index)
            hand_points, hand_confidences, dex_hand_points = self._read_hand_points(
                frame_index
            )
            hand_rotations = self._estimate_hand_rotations(hand_points)
            raw_hand_rotations, dex_hand_rotations = self._read_raw_hand_rotations(
                frame_index
            )
            for side, rotation in hand_rotations.items():
                hand_key = f"{side}Hand"
                if hand_key in human_data:
                    quat = R.from_matrix(rotation).as_quat(scalar_first=True)
                    human_data[hand_key][1] = quat.astype(np.float64)
            yield EgoDexFrame(
                frame_index=frame_index,
                timestamp_s=frame_index / self.fps,
                human_data=human_data,
                hand_points=hand_points,
                hand_confidences=hand_confidences,
                hand_rotations=hand_rotations,
                raw_hand_rotations=raw_hand_rotations,
                dex_hand_points=dex_hand_points,
                dex_hand_rotations=dex_hand_rotations,
            )

    def _require_h5(self):
        if self._h5 is None:
            raise RuntimeError(
                "EgoDexEpisodeReader must be opened with a context manager"
            )
        return self._h5

    def _resolve_datasets(self) -> dict[str, object]:
        h5 = self._require_h5()
        datasets: dict[str, object] = {}
        for name in self.body_names:
            path = f"transforms/{name}"
            if path in h5:
                datasets[name] = h5[path]
        for point_name in self.hand_point_names:
            for key, transform_name in self._hand_transform_names(point_name):
                path = f"transforms/{transform_name}"
                if path in h5:
                    datasets[key] = h5[path]
        if not datasets and (self.body_names or self.hand_point_names):
            raise RuntimeError(
                f"No EgoDex transform datasets found in {self.info.hdf5_path}"
            )
        return datasets

    def _read_upper_body(self, frame_index: int) -> dict[str, list[np.ndarray]]:
        human_data: dict[str, list[np.ndarray]] = {}
        for name in self.body_names:
            dataset = self._datasets.get(name)
            confidence = self._read_confidence(name, frame_index)
            trusted = dataset is not None and confidence >= self.confidence_threshold
            if not trusted:
                # Low confidence / missing: reuse previous frame, else fall back to
                # this frame's data to seed the history.
                previous = self._previous_human.get(name)
                if previous is not None:
                    human_data[name] = [previous[0].copy(), previous[1].copy()]
                    continue
                if dataset is None:
                    continue
            pos, quat = self._convert_transform(
                np.asarray(dataset[frame_index], dtype=np.float64)
            )
            human_data[name] = [pos, quat]
            self._previous_human[name] = [pos.copy(), quat.copy()]
        return human_data

    def _read_hand_points(
        self, frame_index: int
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
        """Read per-side finger keypoints as 3D world-frame positions.

        Only the translation component of each transform is kept. Confidences are
        always recorded so the retargeter can fall back to the previous frame.
        """
        points: dict[str, np.ndarray] = {}
        dex_points: dict[str, np.ndarray] = {}
        confidences: dict[str, float] = {}
        for name, dataset in self._datasets.items():
            if name in self.body_names:
                continue
            confidence = self._read_confidence(name, frame_index)
            confidences[name] = confidence
            pos, _quat = self._convert_transform(
                np.asarray(dataset[frame_index], dtype=np.float64)
            )
            dex_points[name] = pos
            if confidence < self.confidence_threshold:
                continue
            points[name] = pos
        return points, confidences, dex_points

    def _estimate_hand_rotations(
        self, hand_points: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        rotations: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            if not self.mano_keypoint_names:
                continue
            keypoints = np.full((21, 3), np.nan, dtype=np.float64)
            for index, point_name in enumerate(self.mano_keypoint_names):
                point = hand_points.get(f"{side}_{point_name}")
                if point is not None:
                    keypoints[index] = point
            rotation = estimate_mano_hand_frame(keypoints)
            if rotation is not None:
                self._previous_hand_rotations[side] = rotation.copy()
                rotations[side] = rotation
                continue
            previous = self._previous_hand_rotations.get(side)
            if previous is not None:
                rotations[side] = previous.copy()
        return rotations

    def _read_raw_hand_rotations(
        self, frame_index: int
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        rotations: dict[str, np.ndarray] = {}
        dex_rotations: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            dataset = self._datasets.get(f"{side}_Hand")
            if dataset is None:
                continue
            transform = np.asarray(dataset[frame_index], dtype=np.float64)
            _pos, quat = self._convert_transform(transform)
            rotation = R.from_quat(quat, scalar_first=True).as_matrix()
            rotations[side] = rotation
            dex_rotations[side] = rotation @ AVP_WRIST_TO_DEX_OPT_ROTATIONS[side]
        return rotations, dex_rotations

    def _convert_transform(
        self, transform: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # EgoDex stores poses in a per-episode ARKit world frame. Keep that global
        # origin and only convert axes to the GMR z-up right-handed convention.
        pos = ARKIT_TO_GMR @ transform[:3, 3]
        rot_matrix = ARKIT_TO_GMR @ transform[:3, :3] @ ARKIT_TO_GMR.T
        quat_xyzw = R.from_matrix(rot_matrix).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return pos.astype(np.float64), quat_wxyz.astype(np.float64)

    def _egodex_name(self, name: str) -> str:
        """Map a human key back to its exact EgoDex joint name.

        Upper-body keys already are EgoDex names; hand-point names come from
        the IK config.
        """
        for point_name in self.hand_point_names:
            for key, transform_name in self._hand_transform_names(point_name):
                if key == name:
                    return transform_name
        return name

    def _read_confidence(self, name: str, frame_index: int) -> float:
        h5 = self._require_h5()
        egodex_name = self._egodex_name(name)
        path = f"confidences/{egodex_name}"
        if path not in h5:
            return 1.0
        value = np.asarray(h5[path][frame_index]).reshape(-1)[0]
        return float(value)

    @staticmethod
    def _hand_transform_names(point_name: str) -> tuple[tuple[str, str], ...]:
        if _has_side_prefix(point_name):
            return ((point_name, point_name),)
        return tuple(
            (f"{side}_{point_name}", f"{side}{point_name}")
            for side in ("left", "right")
        )


def loader_targets_from_ik_config(
    ik_config: dict,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Derive EgoDex loader targets from the IK and dex-hand config.

    IK tables define which human body names GMR consumes. Dex hand config can
    define the 21 MANO keypoint names consumed by dex-retargeting.
    """
    body_names: list[str] = []
    hand_point_names: list[str] = []

    def add_body(name: str | None) -> None:
        if name and name not in body_names:
            body_names.append(name)

    def add_hand_point(name: str | None) -> None:
        if name and name not in hand_point_names:
            hand_point_names.append(name)

    add_body(ik_config.get("human_root_name"))
    for table_key in ("ik_match_table1", "ik_match_table2"):
        for entry in ik_config.get(table_key, {}).values():
            if len(entry) < 3:
                continue
            human_name, pos_weight, rot_weight = entry[:3]
            if pos_weight == 0 and rot_weight == 0:
                continue
            if _is_hand_point_name(human_name):
                add_hand_point(human_name)
            else:
                add_body(human_name)

    dex_cfg = ik_config.get("dex_hand_config")
    mano_keypoint_names = tuple((dex_cfg or {}).get("mano_keypoint_names") or ())
    if dex_cfg is not None:
        for name in mano_keypoint_names:
            add_hand_point(name)

    return tuple(body_names), tuple(hand_point_names), mano_keypoint_names


def _is_hand_point_name(name: str) -> bool:
    if not _has_side_prefix(name):
        return False
    return "Finger" in name or "Thumb" in name


def _has_side_prefix(name: str) -> bool:
    return name.startswith("left") or name.startswith("right")


def estimate_mano_hand_frame(keypoints: np.ndarray) -> np.ndarray | None:
    """Estimate a MANO-style wrist frame from 21 hand keypoints.

    Columns follow the convention used by dex-retargeting examples:
    x is roughly middle-finger-to-wrist, y is palm normal, z points from middle
    toward index.
    """
    if keypoints.shape != (21, 3):
        raise ValueError(f"Expected 21x3 keypoints, got {keypoints.shape}")
    points = keypoints[[0, 5, 9], :]
    if not np.isfinite(points).all():
        return None
    x_vector = points[0] - points[2]
    centered = points - np.mean(points, axis=0, keepdims=True)
    try:
        _u, _s, vh = np.linalg.svd(centered)
    except np.linalg.LinAlgError:
        return None
    normal = vh[2, :]
    x_axis = x_vector - np.sum(x_vector * normal) * normal
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        return None
    x_axis /= x_norm
    z_axis = np.cross(x_axis, normal)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        return None
    z_axis /= z_norm
    if np.sum(z_axis * (centered[1] - centered[2])) < 0:
        normal *= -1
        z_axis *= -1
    frame = np.stack([x_axis, normal, z_axis], axis=1)
    if np.linalg.det(frame) < 0:
        frame[:, 1] *= -1
    return frame.astype(np.float64)
