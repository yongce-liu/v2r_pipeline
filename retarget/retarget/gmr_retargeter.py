"""Dataset-agnostic GMR retargeting wrapper."""

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting as GMR

DEFAULT_SRC_HUMAN = "human_upper_body"
DEFAULT_TGT_ROBOT = "robot_target"


@dataclass
class GMRRetargeter:
    gmr: GMR
    action_names: list[str]
    qpos_indices: np.ndarray
    ik_config: dict

    @property
    def model(self) -> mj.MjModel:
        return self.gmr.model

    def retarget(self, human_data: dict[str, list[np.ndarray]]) -> np.ndarray:
        return self.gmr.retarget(human_data, offset_to_ground=False)


def build_gmr_retargeter(
    robot_xml: Path,
    ik_config: Path,
    *,
    robot_base: str = "pelvis",
    src_human: str = DEFAULT_SRC_HUMAN,
    tgt_robot: str = DEFAULT_TGT_ROBOT,
    verbose: bool = False,
) -> GMRRetargeter:
    ik_config_data = load_ik_config(ik_config)
    _register_gmr_paths(robot_xml, ik_config, robot_base, src_human, tgt_robot)
    retargeter = GMR(
        src_human=src_human,
        tgt_robot=tgt_robot,
        verbose=verbose,
        use_velocity_limit=True,
    )
    action_names, qpos_indices = actuator_joint_mapping(retargeter.model)
    return GMRRetargeter(
        gmr=retargeter,
        action_names=action_names,
        qpos_indices=qpos_indices,
        ik_config=ik_config_data,
    )


def load_ik_config(ik_config: Path) -> dict:
    with open(ik_config) as f:
        return json.load(f)


def required_gmr_hand_points(ik_config: dict) -> tuple[str, ...]:
    names: set[str] = set()
    for table_key in ("ik_match_table1", "ik_match_table2"):
        for entry in ik_config.get(table_key, {}).values():
            if len(entry) < 3:
                continue
            body_name, pos_weight, rot_weight = entry[:3]
            if (pos_weight != 0 or rot_weight != 0) and _is_hand_point(body_name):
                names.add(body_name)
    return tuple(sorted(names))


def merge_gmr_hand_points(
    human_data: dict[str, list[np.ndarray]],
    hand_points: dict[str, np.ndarray],
    required_points: tuple[str, ...],
) -> dict[str, list[np.ndarray]]:
    """Add hand keypoints as position-only GMR targets."""
    merged = {
        key: [np.asarray(value[0]).copy(), np.asarray(value[1]).copy()]
        for key, value in human_data.items()
    }
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for point_name in required_points:
        point_array = np.asarray(hand_points[point_name], dtype=np.float64).copy()
        merged[point_name] = [point_array, identity_quat.copy()]
    return merged


def _is_hand_point(name: str) -> bool:
    if not (name.startswith("left") or name.startswith("right")):
        return False
    return "Finger" in name or "Thumb" in name


def actuator_joint_mapping(model: mj.MjModel) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    qpos_indices: list[int] = []
    for actuator_id in range(model.nu):
        actuator_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, actuator_name)
        if joint_id < 0:
            raise RuntimeError(
                f"Actuator {actuator_name} does not map to a joint with the same name"
            )
        names.append(actuator_name)
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
    return names, np.asarray(qpos_indices, dtype=np.int64)


def _register_gmr_paths(
    robot_xml: Path,
    ik_config: Path,
    robot_base: str = "pelvis",
    src_human: str = DEFAULT_SRC_HUMAN,
    tgt_robot: str = DEFAULT_TGT_ROBOT,
) -> None:
    from general_motion_retargeting import params as gmr_params

    robot_xml = Path(robot_xml)
    ik_config = Path(ik_config)
    if not robot_xml.is_file():
        raise FileNotFoundError(f"robot_xml not found: {robot_xml}")
    if not ik_config.is_file():
        raise FileNotFoundError(f"ik_config not found: {ik_config}")

    gmr_params.ROBOT_XML_DICT[tgt_robot] = robot_xml
    gmr_params.ROBOT_BASE_DICT[tgt_robot] = robot_base
    gmr_params.VIEWER_CAM_DISTANCE_DICT[tgt_robot] = 2.0
    gmr_params.IK_CONFIG_DICT.setdefault(src_human, {})[tgt_robot] = ik_config
