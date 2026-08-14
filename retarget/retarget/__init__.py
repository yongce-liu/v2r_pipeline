"""Dataset-agnostic frame to robot qpos/action retargeting.

The retarget step consumes human position/pose data (EgoDex HDF5 today, a
unified preprocessed format later) and produces robot joint qpos/action.

This package only depends on GMR, dex-retargeting, mujoco and numpy — it has no
dataset-loading, video, or torch dependencies, so it runs in an isolated uv
environment.
"""

from retarget.dex_retargeter import DexHandRetargeter
from retarget.gmr_retargeter import (
    GMRRetargeter,
    build_gmr_retargeter,
    load_ik_config,
    merge_gmr_hand_points,
    required_gmr_hand_points,
)
from retarget.loaders import (
    SOURCE_LOADERS,
    Frame,
    iter_source_frames,
    known_source_types,
)
from retarget.retargeter import (
    RetargetResult,
    RobotRetargeter,
    build_dex_retargeter,
)
from retarget.visualize import (
    launch_viewer,
    load_playback,
    render_video,
)

# Temporary GMR registry names injected by this package.
# Defined before the module imports below: ``retarget.retargeter`` does
# ``from retarget import GMR_SRC_HUMAN, GMR_TGT_ROBOT`` at import time.
GMR_SRC_HUMAN = "ego2robo_human"
GMR_TGT_ROBOT = "ego2robo_robot"

__all__ = [
    "DexHandRetargeter",
    "Frame",
    "GMRRetargeter",
    "GMR_SRC_HUMAN",
    "GMR_TGT_ROBOT",
    "RetargetResult",
    "RobotRetargeter",
    "SOURCE_LOADERS",
    "build_dex_retargeter",
    "build_gmr_retargeter",
    "iter_source_frames",
    "known_source_types",
    "load_ik_config",
    "merge_gmr_hand_points",
    "required_gmr_hand_points",
    "launch_viewer",
    "load_playback",
    "render_video",
]
