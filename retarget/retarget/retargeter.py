"""Dataset-agnostic frame to robot qpos/action retargeting."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retarget import GMR_SRC_HUMAN, GMR_TGT_ROBOT
from retarget.dex_retargeter import DexHandRetargeter
from retarget.gmr_retargeter import (
    GMRRetargeter,
    build_gmr_retargeter,
    merge_gmr_hand_points,
    required_gmr_hand_points,
)


@dataclass
class RetargetResult:
    qpos: np.ndarray
    action: np.ndarray


def build_dex_retargeter(dex_cfg: dict, fallback_threshold: float) -> DexHandRetargeter:
    if "urdf_dir" not in dex_cfg:
        raise ValueError("dex_hand_config must define urdf_dir")
    urdf_dir = Path(dex_cfg["urdf_dir"])
    hand_names = tuple(dex_cfg.get("hand_names", ("left", "right")))
    threshold = float(dex_cfg.get("confidence_threshold", fallback_threshold))
    raw_config_paths = dex_cfg.get("config_paths") or {}
    config_paths = {side: Path(path) for side, path in raw_config_paths.items()}
    post_rotations = dex_cfg.get("post_rotations")
    mano_keypoint_names = tuple(dex_cfg.get("mano_keypoint_names") or ())
    wrist_rotations = (
        {
            side: np.asarray(rotation, dtype=np.float64)
            for side, rotation in post_rotations.items()
        }
        if post_rotations
        else None
    )
    return DexHandRetargeter(
        urdf_dir=urdf_dir,
        mano_keypoint_names=mano_keypoint_names,
        wrist_rotations=wrist_rotations,
        robot_name=dex_cfg.get("robot_name", "inspire"),
        retargeting_type=dex_cfg.get("retargeting_type", "position"),
        confidence_threshold=threshold,
        hand_names=hand_names,
        config_paths=config_paths or None,
        output_joint_prefixes=dex_cfg.get("output_joint_prefixes"),
        valid_keypoint_names=dex_cfg.get("valid_keypoint_names"),
    )


class RobotRetargeter:
    def __init__(
        self,
        robot_xml: Path,
        ik_config: Path,
        *,
        confidence_threshold: float = 0.5,
        verbose_gmr: bool = False,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.gmr_retargeter: GMRRetargeter = build_gmr_retargeter(
            robot_xml,
            ik_config,
            src_human=GMR_SRC_HUMAN,
            tgt_robot=GMR_TGT_ROBOT,
            verbose=verbose_gmr,
        )
        self.ik_config = self.gmr_retargeter.ik_config

        self.dex_hand_config = self.ik_config.get("dex_hand_config")
        self.use_dex = self.dex_hand_config is not None
        self.gmr_hand_points = (
            () if self.use_dex else required_gmr_hand_points(self.ik_config)
        )
        self.use_gmr_hands = bool(self.gmr_hand_points)

        action_names = list(self.gmr_retargeter.action_names)
        qpos_indices = self.gmr_retargeter.qpos_indices

        self.dex_retargeter: DexHandRetargeter | None = None
        self.dex_joint_names: list[str] = []
        self.dex_qpos_indices: np.ndarray | None = None
        if self.use_dex:
            self.dex_retargeter = build_dex_retargeter(
                self.dex_hand_config, confidence_threshold
            )
            self.dex_joint_names = list(self.dex_retargeter.joint_names)
            self.dex_qpos_indices = self._dex_qpos_indices()
            dex_joint_set = set(self.dex_joint_names)
            keep = [
                i for i, name in enumerate(action_names) if name not in dex_joint_set
            ]
            action_names = [action_names[i] for i in keep] + self.dex_joint_names
            qpos_indices = qpos_indices[keep]

        self.action_names = action_names
        self.qpos_indices = qpos_indices

    def _dex_qpos_indices(self) -> np.ndarray:
        """qpos columns of the dex hand joints, so hands can be folded into qpos.

        The retargeted ``qpos`` is GMR's whole-model pose, which solves only the
        body/arm task frames and leaves the finger joints untouched. For the
        saved ``qpos`` to be a complete robot pose the dex hand values must be
        written back in here, so every dex joint must exist in the robot model.
        """
        import mujoco as mj

        model = self.model
        name_to_qposadr = {
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id): int(
                model.jnt_qposadr[joint_id]
            )
            for joint_id in range(model.njnt)
        }
        missing = [name for name in self.dex_joint_names if name not in name_to_qposadr]
        if missing:
            raise RuntimeError(
                "Dex hand joints are missing from the robot model, so qpos cannot "
                f"carry the hand pose: {missing}. Add these joints to the MJCF "
                "or drop dex_hand_config from the IK config."
            )
        return np.asarray(
            [name_to_qposadr[name] for name in self.dex_joint_names], dtype=np.int64
        )

    @property
    def gmr(self):
        return self.gmr_retargeter.gmr

    @property
    def model(self):
        return self.gmr_retargeter.model

    @property
    def hand_backend(self) -> str:
        if self.use_dex:
            robot_name = self.dex_hand_config.get("robot_name", "inspire")
            return f"dex_retargeting ({robot_name})"
        return "gmr direct" if self.use_gmr_hands else "body only"

    def retarget(self, frame) -> RetargetResult:
        human_data = frame.human_data
        if self.use_gmr_hands:
            human_data = merge_gmr_hand_points(
                human_data,
                frame.hand_points,
                self.gmr_hand_points,
            )
        qpos = self.gmr_retargeter.retarget(human_data)

        if self.use_dex:
            if self.dex_retargeter is None or self.dex_qpos_indices is None:
                raise RuntimeError("Dex backend is enabled but was not initialized")
            arm_action = qpos[self.qpos_indices]
            dex_values = self.dex_retargeter.retarget(
                frame.dex_hand_points,
                frame.hand_confidences,
                frame.dex_hand_rotations,
            )
            hand_action = np.array(
                [dex_values[name] for name in self.dex_joint_names], dtype=np.float64
            )
            # Fold the dex hand joints into qpos so the saved qpos is the complete
            # robot pose (hands included), not just GMR's body/arm IK result.
            qpos[self.dex_qpos_indices] = hand_action
            action = np.concatenate([arm_action, hand_action]).astype(
                np.float32, copy=False
            )
        else:
            action = qpos[self.qpos_indices].astype(np.float32, copy=False)

        return RetargetResult(qpos=qpos, action=action)
