"""Dataset-agnostic frame to robot qpos/action retargeting.

The retarget step consumes human position/pose data (EgoDex HDF5 today, a
unified preprocessed format later) and produces robot joint qpos/action.

This package only depends on GMR, dex-retargeting, mujoco and numpy — it has no
dataset-loading, video, or torch dependencies, so it runs in an isolated uv
environment.
"""

# Temporary GMR registry names injected by this package.
GMR_SRC_HUMAN = "ego2robo_human"
GMR_TGT_ROBOT = "ego2robo_robot"

from retarget.dex_retargeter import DexHandRetargeter
from retarget.gmr_retargeter import (
    GMRRetargeter,
    build_gmr_retargeter,
    load_ik_config,
    merge_gmr_hand_points,
    required_gmr_hand_points,
)
from retarget.retargeter import (
    RetargetResult,
    RobotRetargeter,
    build_dex_retargeter,
)

__all__ = [
    "DexHandRetargeter",
    "GMRRetargeter",
    "GMR_SRC_HUMAN",
    "GMR_TGT_ROBOT",
    "RetargetResult",
    "RobotRetargeter",
    "build_dex_retargeter",
    "build_gmr_retargeter",
    "load_ik_config",
    "merge_gmr_hand_points",
    "required_gmr_hand_points",
]
