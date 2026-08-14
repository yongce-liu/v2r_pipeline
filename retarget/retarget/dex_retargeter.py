"""Dataset-agnostic hand retargeting from MANO finger keypoints to hand joints."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

# Passive Inspire joints are filled from active drivers after optimization.
_MIMIC_RULES: tuple[tuple[str, str, float], ...] = (
    # Inspire FTP.
    ("thumb_3_joint", "thumb_2_joint", 0.8024),
    ("thumb_4_joint", "thumb_3_joint", 0.9487),
    ("index_2_joint", "index_1_joint", 1.0843),
    ("middle_2_joint", "middle_1_joint", 1.0843),
    ("ring_2_joint", "ring_1_joint", 1.0843),
    ("little_2_joint", "little_1_joint", 1.0843),
    # Inspire DFQ.
    ("thumb_intermediate_joint", "thumb_proximal_pitch_joint", 1.6),
    ("thumb_distal_joint", "thumb_proximal_pitch_joint", 2.4),
    ("index_intermediate_joint", "index_proximal_joint", 1.0),
    ("middle_intermediate_joint", "middle_proximal_joint", 1.0),
    ("ring_intermediate_joint", "ring_proximal_joint", 1.0),
    ("pinky_intermediate_joint", "pinky_proximal_joint", 1.0),
)

_MANO_FINGER_CHAINS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)


class DexHandRetargeter:
    """Retarget MANO-style finger keypoints through dex-retargeting."""

    def __init__(
        self,
        urdf_dir: str | Path,
        mano_keypoint_names: Sequence[str],
        wrist_rotations: Mapping[str, np.ndarray] | None = None,
        robot_name: str = "inspire",
        retargeting_type: str = "position",
        confidence_threshold: float = 0.5,
        hand_names: tuple[str, ...] = ("left", "right"),
        config_paths: dict[str, str | Path] | None = None,
        output_joint_prefixes: dict[str, str] | None = None,
        valid_keypoint_names: Sequence[str] | None = None,
    ) -> None:
        self.urdf_dir = Path(urdf_dir)
        self.robot_name = robot_name
        self.retargeting_type = retargeting_type
        self.confidence_threshold = confidence_threshold
        self.hand_names = hand_names
        self.mano_keypoint_names = tuple(mano_keypoint_names)
        if len(self.mano_keypoint_names) != 21:
            raise ValueError(
                "mano_keypoint_names must have exactly 21 entries, got "
                f"{len(self.mano_keypoint_names)}"
            )
        self.wrist_rotations = {
            side: np.asarray(rotation, dtype=np.float64)
            for side, rotation in (wrist_rotations or {}).items()
        }
        self.config_paths = (
            {side: Path(path) for side, path in config_paths.items()}
            if config_paths
            else {}
        )
        self.output_joint_prefixes = dict(output_joint_prefixes or {})
        self.valid_keypoint_names = (
            frozenset(valid_keypoint_names)
            if valid_keypoint_names is not None
            else None
        )
        self._retargeters: dict[str, object] = {}
        self._finger_joint_names: dict[str, list[str]] = {}
        self._finger_qpos_slice: dict[str, np.ndarray] = {}
        self._mimic_rules: dict[str, list[tuple[str, str, float]]] = {}
        self._fixed_joint_count: dict[str, int] = {}
        self._output_joint_names: dict[str, list[str]] = {}
        self._previous_keypoints: dict[str, np.ndarray] = {}
        self._previous_hand_rotations: dict[str, np.ndarray] = {}
        self._previous_ref: dict[str, np.ndarray] = {}
        self._build_retargeters()

    def _build_retargeters(self) -> None:
        from dex_retargeting.constants import (
            HandType,
            RetargetingType,
            RobotName,
            get_default_config_path,
        )
        from dex_retargeting.retargeting_config import RetargetingConfig

        if not self.urdf_dir.exists():
            raise RuntimeError(
                f"Dex hand URDF directory does not exist: {self.urdf_dir}"
            )
        RetargetingConfig.set_default_urdf_dir(str(self.urdf_dir))

        robot_name = RobotName[self.robot_name]
        retargeting_type = RetargetingType[self.retargeting_type]
        hand_type_map = {"left": HandType.left, "right": HandType.right}
        for side in self.hand_names:
            hand_type = hand_type_map[side]
            custom = self.config_paths.get(side)
            if custom is not None:
                config_path = Path(custom)
                if not config_path.is_file():
                    raise RuntimeError(
                        f"Custom dex-retargeting config not found for {side} hand: "
                        f"{config_path}"
                    )
            else:
                config_path = get_default_config_path(
                    robot_name, retargeting_type, hand_type
                )
            retarget = RetargetingConfig.load_from_file(config_path).build()
            opt = retarget.optimizer
            all_dof_names = list(opt.robot.dof_joint_names)
            finger_names = [n for n in all_dof_names if not n.startswith("dummy_")]
            fixed_names = list(opt.fixed_joint_names)
            finger_indices = np.array(
                [all_dof_names.index(name) for name in finger_names], dtype=np.int64
            )
            self._retargeters[side] = retarget
            self._finger_joint_names[side] = finger_names
            self._finger_qpos_slice[side] = finger_indices
            self._fixed_joint_count[side] = len(fixed_names)

            active_set = set(finger_names)
            prefixes = _joint_prefix_candidates(side, finger_names)
            side_rules: list[tuple[str, str, float]] = []
            resolved_passive: set[str] = set()
            for passive_suffix, driver_suffix, mult in _MIMIC_RULES:
                passive = _resolve_joint_name(passive_suffix, prefixes, active_set)
                driver = _resolve_joint_name(
                    driver_suffix, prefixes, active_set | resolved_passive
                )
                if passive is None or driver is None:
                    continue
                if passive not in active_set:
                    continue
                if driver in active_set or driver in resolved_passive:
                    side_rules.append((passive, driver, mult))
                    resolved_passive.add(passive)
            self._mimic_rules[side] = side_rules
            self._output_joint_names[side] = [
                _apply_output_prefix(name, self.output_joint_prefixes.get(side, ""))
                for name in finger_names
            ]

    @property
    def joint_names(self) -> list[str]:
        names: list[str] = []
        for side in self.hand_names:
            names.extend(self._output_joint_names[side])
        return names

    def retarget(
        self,
        points: Mapping[str, np.ndarray],
        confidences: Mapping[str, float] | None = None,
        hand_rotations: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Retarget one named-keypoint frame to hand joint angles."""
        confidences = confidences or {}
        hand_rotations = hand_rotations or {}
        values: dict[str, float] = {}
        for side in self.hand_names:
            values.update(
                self._retarget_side(side, points, confidences, hand_rotations)
            )
        return values

    def _retarget_side(
        self,
        side: str,
        points: Mapping[str, np.ndarray],
        confidences: Mapping[str, float],
        hand_rotations: Mapping[str, np.ndarray],
    ) -> dict[str, float]:
        retarget = self._retargeters[side]
        indices = retarget.optimizer.target_link_human_indices
        keypoints = self._build_mano_keypoints(
            side, points, confidences, hand_rotations
        )
        # Arm retargeting sets the wrist; fingers use wrist-local geometry.
        if indices.ndim == 1:
            ref_value = keypoints[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = keypoints[task_indices, :] - keypoints[origin_indices, :]
        if not np.any(ref_value) and side in self._previous_ref:
            ref_value = self._previous_ref[side]
        else:
            self._previous_ref[side] = ref_value
        fixed_qpos = np.zeros(self._fixed_joint_count[side], dtype=np.float64)
        qpos = retarget.retarget(ref_value, fixed_qpos)
        finger_qpos = qpos[self._finger_qpos_slice[side]]
        result = {
            name: float(value)
            for name, value in zip(self._output_joint_names[side], finger_qpos)
        }
        if retarget.optimizer.adaptor is None:
            for passive, driver, mult in self._mimic_rules[side]:
                output_passive = _apply_output_prefix(
                    passive, self.output_joint_prefixes.get(side, "")
                )
                output_driver = _apply_output_prefix(
                    driver, self.output_joint_prefixes.get(side, "")
                )
                result[output_passive] = result.get(output_driver, 0.0) * mult
        return result

    def _build_mano_keypoints(
        self,
        side: str,
        points: Mapping[str, np.ndarray],
        confidences: Mapping[str, float],
        hand_rotations: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        keypoints = np.zeros((21, 3), dtype=np.float64)
        valid = np.zeros(21, dtype=bool)
        for mano_index, point_name in enumerate(self.mano_keypoint_names):
            if (
                self.valid_keypoint_names is not None
                and point_name not in self.valid_keypoint_names
            ):
                continue
            key = f"{side}_{point_name}"
            if confidences.get(key, 1.0) < self.confidence_threshold:
                continue
            point = points.get(key)
            if point is not None and not np.isnan(point).any():
                keypoints[mano_index] = point
                valid[mano_index] = True
        previous = self._previous_keypoints.get(side)
        if not valid[0]:
            return previous.copy() if previous is not None else keypoints

        self._fill_missing_keypoints_from_finger_chain(keypoints, valid)
        wrist = keypoints[0].copy()
        rotation = hand_rotations.get(side)
        if rotation is None:
            rotation = self._previous_hand_rotations.get(side)
        if rotation is None:
            rotation = np.eye(3, dtype=np.float64)
        else:
            rotation = np.asarray(rotation, dtype=np.float64)
            self._previous_hand_rotations[side] = rotation.copy()
        local = (keypoints - wrist) @ rotation
        post_rotation = self.wrist_rotations.get(side)
        if post_rotation is not None:
            local = local @ post_rotation
        if previous is not None:
            local[~valid] = previous[~valid]
        else:
            local[~valid] = 0.0
        self._previous_keypoints[side] = local.copy()
        return local

    @staticmethod
    def _fill_missing_keypoints_from_finger_chain(
        keypoints: np.ndarray, valid: np.ndarray
    ) -> None:
        """Seed missing first-frame joints from neighboring joints."""
        for chain in _MANO_FINGER_CHAINS:
            for index in chain:
                if valid[index]:
                    continue
                candidates = [candidate for candidate in chain if valid[candidate]]
                if not candidates:
                    continue
                nearest = min(candidates, key=lambda candidate: abs(candidate - index))
                keypoints[index] = keypoints[nearest]
                valid[index] = True


def _joint_prefix_candidates(side: str, joint_names: list[str]) -> tuple[str, ...]:
    side_prefix = f"{side}_"
    short_prefix = "L_" if side == "left" else "R_"
    candidates = [side_prefix, short_prefix, ""]
    present = []
    for prefix in candidates:
        if prefix and any(name.startswith(prefix) for name in joint_names):
            present.append(prefix)
    present.append("")
    return tuple(dict.fromkeys(present))


def _resolve_joint_name(
    suffix: str, prefixes: tuple[str, ...], available: set[str]
) -> str | None:
    for prefix in prefixes:
        name = f"{prefix}{suffix}"
        if name in available:
            return name
    return None


def _apply_output_prefix(name: str, prefix: str) -> str:
    if not prefix or name.startswith(prefix):
        return name
    return f"{prefix}{name}"
