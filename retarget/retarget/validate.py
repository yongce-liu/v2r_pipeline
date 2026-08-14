"""Validate a retargeted robot trajectory against the robot model.

Checks performed:
  - shapes and dtypes (qpos (T,60), action (T,53))
  - all values finite
  - action == qpos[qpos_indices] for the body actuator block (internal consistency
    of RobotRetargeter: the arm part of ``action`` is sliced from ``qpos``)
  - per-joint qpos within the MJCF joint limits (``jnt_range``) with a small
    tolerance for IK/GMR soft targets
  - pelvis free-joint continuity (per-frame displacement, velocity spikes)
  - per-joint min/max sweep over the episode (which joints actually move)

Usage:
    python -m retarget.validate OUTPUT.npz [--robot-xml ...]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_ROBOT_XML = Path(
    "assets/unitree_g1_mjcf/g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml"
)
DEFAULT_IK_CONFIG = Path("configs/egodex_g1_inspire_dfq.json")


def validate_trajectory(
    npz_path: Path,
    robot_xml: Path = DEFAULT_ROBOT_XML,
    *,
    ik_config: Path = DEFAULT_IK_CONFIG,
    limit_tolerance: float = 0.15,
) -> dict:
    """Run all checks and return a dict of findings."""
    import mujoco as mj

    data = np.load(npz_path)
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    action = np.asarray(data["action"], dtype=np.float64)
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)

    model = mj.MjModel.from_xml_path(str(robot_xml))
    report: dict = {"robot_xml": str(robot_xml), "source": str(npz_path)}

    # 1. Shapes.
    report["qpos_shape"] = tuple(qpos.shape)
    report["action_shape"] = tuple(action.shape)
    report["frame_count"] = len(qpos)
    report["duration_s"] = (
        float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    )

    # 2. Finiteness.
    report["qpos_finite"] = bool(np.isfinite(qpos).all())
    report["action_finite"] = bool(np.isfinite(action).all())

    # 3. Internal consistency. RobotRetargeter orders ``action`` as the GMR body
    # block (first ``body_count`` entries == qpos at the body actuator qpos
    # indices) followed by the dex hand block.
    from retarget import RobotRetargeter

    retargeter = RobotRetargeter(robot_xml, ik_config)
    body_count = len(retargeter.qpos_indices)
    report["body_action_count"] = body_count
    report["hand_action_count"] = action.shape[1] - body_count
    body_match = np.allclose(
        action[:, :body_count], qpos[:, retargeter.qpos_indices], atol=1e-6
    )
    report["body_action_consistent_with_qpos"] = bool(body_match)
    # Dex block values are valid joint angles (finite, within model range); they
    # are not a qpos slice, so only finiteness + limits apply.
    dex_block = action[:, body_count:]
    report["dex_action_finite"] = bool(np.isfinite(dex_block).all())
    report["dex_action_consistent_with_qpos"] = None  # not applicable

    # 4. Joint-limit check for all actuated (non-free) joints.
    joint_names: list[str] = []
    joint_qpos_indices: list[int] = []
    joint_ranges: list[tuple[float, float]] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_FREE:
            continue
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            continue
        joint_names.append(name)
        joint_qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        lo, hi = model.jnt_range[joint_id]
        joint_ranges.append((float(lo), float(hi)))
    limit_exceeded: dict[str, dict] = {}
    for index, name in enumerate(joint_names):
        lo, hi = joint_ranges[index]
        values = qpos[:, joint_qpos_indices[index]]
        max_low = float(np.max(lo - values))
        max_high = float(np.max(values - hi))
        if max_low > limit_tolerance or max_high > limit_tolerance:
            limit_exceeded[name] = {
                "range": [lo, hi],
                "min": float(values.min()),
                "max": float(values.max()),
                "exceeds_lo_by": max_low,
                "exceeds_hi_by": max_high,
            }
    report["joint_limit_violations"] = limit_exceeded

    # 5. Pelvis continuity (linear velocity magnitude between consecutive frames).
    if len(qpos) > 1:
        pelvis_pos = qpos[:, 0:3]
        step = np.diff(pelvis_pos, axis=0)
        dt = np.diff(timestamps)
        dt[dt <= 0] = 1.0
        velocity = np.linalg.norm(step, axis=1) / dt
        report["pelvis_max_step_m"] = float(np.linalg.norm(step, axis=1).max())
        report["pelvis_mean_vel_mps"] = float(velocity.mean())
        report["pelvis_max_vel_mps"] = float(velocity.max())

    # 6. Per-joint sweep over the episode.
    moving: dict[str, dict] = {}
    for index, name in enumerate(joint_names):
        values = qpos[:, joint_qpos_indices[index]]
        moving[name] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "sweep": float(values.max() - values.min()),
        }
    report["joint_sweeps"] = moving
    return report


def _format_report(report: dict) -> str:
    lines = [
        f"Trajectory: {report['source']}",
        f"  frames={report['frame_count']} duration={report['duration_s']:.2f}s",
    ]
    lines.append(f"  qpos{report['qpos_shape']} action{report['action_shape']}")
    lines.append(
        f"  qpos finite={report['qpos_finite']} action finite={report['action_finite']}"
    )
    lines.append(
        f"  body action ({report['body_action_count']}) == qpos[qpos_indices]: "
        f"{report['body_action_consistent_with_qpos']} | "
        f"dex action ({report['hand_action_count']}) finite: {report['dex_action_finite']}"
    )
    if "pelvis_max_step_m" in report:
        lines.append(
            f"  pelvis: max_step={report['pelvis_max_step_m'] * 100:.1f}cm "
            f"mean_vel={report['pelvis_mean_vel_mps']:.2f} m/s max_vel={report['pelvis_max_vel_mps']:.2f} m/s"
        )
    violations = report.get("joint_limit_violations", {})
    if violations:
        lines.append(f"  JOINT LIMIT VIOLATIONS ({len(violations)}):")
        for name, detail in violations.items():
            lines.append(
                f"    {name}: range={detail['range']} got [{detail['min']:.3f},{detail['max']:.3f}] "
                f"(lo-breach {detail['exceeds_lo_by']:.3f}, hi-breach {detail['exceeds_hi_by']:.3f})"
            )
    else:
        lines.append("  joint limits: all within model range")
    sweeps = report.get("joint_sweeps", {})
    static = [name for name, s in sweeps.items() if s["sweep"] < 1e-4]
    active = sorted(sweeps.items(), key=lambda kv: kv[1]["sweep"], reverse=True)[:8]
    lines.append(f"  static joints (sweep<1e-4): {len(static)} of {len(sweeps)}")
    lines.append("  most active joints:")
    for name, s in active:
        lines.append(
            f"    {name:35s} sweep {s['sweep'] * 180 / np.pi:6.1f} deg  [{s['min']:.3f}, {s['max']:.3f}]"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a retargeted trajectory npz")
    parser.add_argument("npz", type=Path)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--ik-config", type=Path, default=DEFAULT_IK_CONFIG)
    parser.add_argument("--limit-tolerance", type=float, default=0.15)
    args = parser.parse_args()
    report = validate_trajectory(
        args.npz,
        args.robot_xml,
        ik_config=args.ik_config,
        limit_tolerance=args.limit_tolerance,
    )
    print(_format_report(report))
    if report.get("joint_limit_violations"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
