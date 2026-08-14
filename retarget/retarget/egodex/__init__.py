"""Self-contained EgoDex hdf5 -> robot trajectory retargeting.

The retarget step consumes an EgoDex episode hdf5 (``transforms/`` +
``confidences/``) plus the robot IK config and MJCF, and produces the robot
qpos/action trajectory. The video-to-LeRobot conversion is out of scope for this
step; this package only depends on the retarget venv (numpy, scipy, h5py, and
the generic retargeter).
"""

from retarget.egodex.dataloader import (
    EgoDexDataLoader,
    EgoDexEpisodeInfo,
    loader_targets_from_ik_config,
)

__all__ = [
    "EgoDexDataLoader",
    "EgoDexEpisodeInfo",
    "loader_targets_from_ik_config",
]
