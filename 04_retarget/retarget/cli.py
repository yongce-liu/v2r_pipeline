"""CLI for the retarget step: human position/pose -> robot qpos/action.

The retarget step consumes a frame's human data (body keypoints + hand
keypoints) and produces the robot joint ``qpos``/``action`` arrays. The frame
data structure is currently consumed from the EgoDex/Nuwa dataloaders; a later
unified preprocessed format will provide the same fields.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RetargetArgs:
    """Inputs for one retarget run."""

    robot_xml: Path = Path(
        "assets/unitree_g1_mjcf/g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml"
    )
    """MJCF robot model used by GMR."""

    ik_config: Path = Path("ego2robo/ego2robo/egodex/configs/g1_inspire_dfq.json")
    """GMR IK config defining the human-to-robot matching tables."""

    confidence_threshold: float = 0.5
    """Keypoint confidence below which previous-frame values are used."""

    output: Path = Path("outputs/retarget/trajectory.npz")
    """Output npz holding qpos and action arrays."""


@dataclass
class Frame:
    """Minimal human-frame contract consumed by :class:`RobotRetargeter`."""

    human_data: dict[str, list[np.ndarray]]
    hand_points: dict[str, np.ndarray]
    hand_confidences: dict[str, float]
    dex_hand_points: dict[str, np.ndarray]
    dex_hand_rotations: dict[str, np.ndarray]


def _build_retargeter(args: RetargetArgs):
    from retarget import RobotRetargeter

    return RobotRetargeter(
        args.robot_xml,
        args.ik_config,
        confidence_threshold=args.confidence_threshold,
    )


def _required_body_names(ik_config_path: Path) -> list[str]:
    """Body names GMR consumes, derived from the IK config's matching tables."""
    from retarget import load_ik_config

    config = load_ik_config(ik_config_path)
    names: list[str] = []
    for table_key in ("ik_match_table1", "ik_match_table2"):
        for entry in config.get(table_key, {}).values():
            if len(entry) < 3:
                continue
            human_name, pos_weight, rot_weight = entry[:3]
            if pos_weight != 0 or rot_weight != 0:
                names.append(human_name)
    return list(dict.fromkeys(names))


def retarget_frames(
    args: RetargetArgs,
    frames: list[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Retarget a list of human frames to robot qpos/action arrays."""
    retargeter = _build_retargeter(args)
    qpos_all: list[np.ndarray] = []
    action_all: list[np.ndarray] = []
    for frame in frames:
        result = retargeter.retarget(frame)
        qpos_all.append(result.qpos)
        action_all.append(result.action)
    return np.stack(qpos_all), np.stack(action_all)


def main(args: RetargetArgs | None = None) -> None:
    import tyro

    if args is None:
        args = tyro.cli(RetargetArgs)

    retargeter = _build_retargeter(args)
    print(f"Retargeting with hand backend: {retargeter.hand_backend}")

    # Placeholder frame source. Real frames come from a dataloader that
    # produces the fields above; this keeps the CLI runnable end-to-end.
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    body_names = _required_body_names(args.ik_config)
    human = {n: [np.array([0.0, 0.0, 1.0]), identity.copy()] for n in body_names}
    hand_points: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        hand_points[f"{side}_Hand"] = np.array([0.0, 0.0, 1.1], dtype=np.float64)
    frames = [
        Frame(
            human_data=human,
            hand_points=hand_points,
            hand_confidences={key: 1.0 for key in hand_points},
            dex_hand_points=hand_points,
            dex_hand_rotations={side: np.eye(3) for side in ("left", "right")},
        )
    ]
    qpos, action = retarget_frames(args, frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        qpos=qpos.astype(np.float64),
        action=action.astype(np.float32),
    )
    print(f"Wrote {len(qpos)} frame(s), action shape {action.shape} -> {args.output}")


if __name__ == "__main__":
    main()
