#!/usr/bin/env python3
"""
Visualize a UR5 + Robotiq UMI joint-action sequence with the cuRobo URDF.

Examples
--------
Paste a 7-D sequence directly:

python scripts/visualize_ur5_robotiq_joint_sequence.py \
  --joint-seq "[[0.03,-1.38,-1.61,-1.69,1.56,0.06,0.084],[0.034,-1.40,-1.61,-1.67,1.56,0.07,0.083]]"

Load from file:

python scripts/visualize_ur5_robotiq_joint_sequence.py \
  --joint-seq-file /path/to/joints.npy

The script accepts:
- shape (T, 7): [6 arm joints, gripper_width_in_meters]
- shape (T, 12): full cuRobo c-space joint vector
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation as R
from yourdfpy import URDF

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from curobo.util_file import get_assets_path, get_robot_configs_path, join_path, load_yaml


# Mirror the joint convention used by eval_real.py for UR5 + Robotiq.
CUROBO_CSPACE_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "left_outer_knuckle_joint",
    "left_inner_knuckle_joint",
    "left_inner_finger_joint",
    "right_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_inner_finger_joint",
]

ARM_JOINT_NAMES = CUROBO_CSPACE_JOINT_NAMES[:6]
JOINT_NAME_TO_IDX = {name: idx for idx, name in enumerate(CUROBO_CSPACE_JOINT_NAMES)}

ARM_LINK_CHAIN = [
    "base_link_inertia",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "robotiq_arg2f_base_link",
]

LEFT_FINGER_CHAIN = [
    "robotiq_arg2f_base_link",
    "left_outer_knuckle",
    "left_outer_finger",
    "left_inner_finger",
    "left_inner_finger_pad",
]

RIGHT_FINGER_CHAIN = [
    "robotiq_arg2f_base_link",
    "right_outer_knuckle",
    "right_outer_finger",
    "right_inner_finger",
    "right_inner_finger_pad",
]

EEF_LINK_NAME = "eef"


def _robotiq_width_to_joint_angles(
    gripper_width: np.ndarray,
    max_width: float = 0.085,
    outer_knuckle_max: float = 0.81,
    inner_knuckle_max: float = 0.8757,
) -> np.ndarray:
    width = np.asarray(gripper_width, dtype=np.float32)
    width = np.clip(width, 0.0, max_width)
    close_ratio = 1.0 - (width / max_width)

    outer_knuckle = close_ratio * outer_knuckle_max
    inner_knuckle = close_ratio * inner_knuckle_max
    inner_finger = -inner_knuckle

    return np.stack(
        [
            outer_knuckle,
            inner_knuckle,
            inner_finger,
            outer_knuckle,
            inner_knuckle,
            inner_finger,
        ],
        axis=-1,
    ).astype(np.float32)


def _robotiq_joint_angles_to_width(
    full_joint_seq: np.ndarray,
    max_width: float = 0.085,
    outer_knuckle_max: float = 0.81,
) -> np.ndarray:
    outer_knuckle = np.asarray(
        full_joint_seq[:, JOINT_NAME_TO_IDX["left_outer_knuckle_joint"]],
        dtype=np.float32,
    )
    close_ratio = np.clip(outer_knuckle / outer_knuckle_max, 0.0, 1.0)
    return (1.0 - close_ratio) * max_width


def _parse_joint_sequence_text(raw_text: str) -> np.ndarray:
    text = raw_text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            # Support NumPy-style pasted arrays where elements are separated by
            # whitespace instead of commas, e.g. "[[1 2 3]\n [4 5 6]]".
            normalized = re.sub(r"(?<=\d)\s+(?=[-\d])", ", ", text)
            normalized = re.sub(r"\]\s+\[", "], [", normalized)
            data = ast.literal_eval(normalized)
    return np.asarray(data, dtype=np.float32)


def _load_joint_sequence(path: Optional[str], inline_text: Optional[str]) -> np.ndarray:
    if inline_text is not None:
        return _parse_joint_sequence_text(inline_text)

    if path is None:
        raise ValueError("Please provide either --joint-seq or --joint-seq-file.")

    input_path = Path(path).expanduser()
    suffix = input_path.suffix.lower()
    if suffix == ".npy":
        data = np.load(input_path)
    elif suffix == ".npz":
        archive = np.load(input_path)
        if len(archive.files) != 1:
            raise ValueError(
                f"{input_path} contains multiple arrays {archive.files}; please keep only one."
            )
        data = archive[archive.files[0]]
    else:
        data = _parse_joint_sequence_text(input_path.read_text())
    return np.asarray(data, dtype=np.float32)


def _load_robot() -> URDF:
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "ur5_robotiq_umi.yml"))["robot_cfg"]
    urdf_rel = robot_cfg["kinematics"]["urdf_path"]
    urdf_path = join_path(get_assets_path(), urdf_rel)
    return URDF.load(urdf_path)


def _normalize_joint_sequence(joint_seq: np.ndarray) -> np.ndarray:
    joint_seq = np.asarray(joint_seq, dtype=np.float32)
    if joint_seq.ndim == 1:
        joint_seq = joint_seq[None, :]
    if joint_seq.ndim != 2:
        raise ValueError(f"Expected a 2-D joint array, got shape {joint_seq.shape}.")

    dim = joint_seq.shape[1]
    if dim == 7:
        arm = joint_seq[:, :6]
        gripper_width = joint_seq[:, 6]
        gripper = _robotiq_width_to_joint_angles(gripper_width)
        joint_seq = np.concatenate([arm, gripper], axis=-1)
    elif dim != 12:
        raise ValueError(
            f"Expected shape (T, 7) or (T, 12), got {joint_seq.shape}."
        )

    joint_dict_seq = []
    for frame in joint_seq:
        joint_dict_seq.append(dict(zip(CUROBO_CSPACE_JOINT_NAMES, frame.tolist())))
    return joint_seq, joint_dict_seq


def _get_link_position(robot: URDF, link_name: str) -> np.ndarray:
    transform = robot.get_transform(link_name)
    return transform[:3, 3].copy()


def _transform_to_pose(mat: np.ndarray) -> np.ndarray:
    pose = np.zeros((6,), dtype=np.float32)
    pose[:3] = mat[:3, 3]
    pose[3:] = R.from_matrix(mat[:3, :3]).as_rotvec().astype(np.float32)
    return pose


def _collect_chain_positions(robot: URDF, link_names: Iterable[str]) -> np.ndarray:
    return np.asarray([_get_link_position(robot, name) for name in link_names], dtype=np.float32)


def _compute_workspace_limits(robot: URDF, joint_dict_seq: Iterable[Dict[str, float]]) -> np.ndarray:
    all_points = []
    for joint_dict in joint_dict_seq:
        robot.update_cfg(joint_dict)
        all_points.append(_collect_chain_positions(robot, ARM_LINK_CHAIN))
        all_points.append(_collect_chain_positions(robot, LEFT_FINGER_CHAIN))
        all_points.append(_collect_chain_positions(robot, RIGHT_FINGER_CHAIN))

    stacked = np.concatenate(all_points, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = 0.5 * (mins + maxs)
    extent = float(np.max(maxs - mins))
    extent = max(extent, 0.6)
    half = 0.6 * extent
    return np.stack([center - half, center + half], axis=0)


def _maybe_make_writer(output_path: Optional[str], fps: float):
    if output_path is None:
        return None
    ext = Path(output_path).suffix.lower()
    if ext == ".gif":
        return animation.PillowWriter(fps=fps)
    if ext == ".mp4":
        return animation.FFMpegWriter(fps=fps)
    raise ValueError("Only .gif and .mp4 are supported for --save.")


def _compute_eef_pose_sequence(
    robot: URDF,
    joint_seq_full: np.ndarray,
    joint_dict_seq: Iterable[Dict[str, float]],
) -> np.ndarray:
    pose_seq = []
    for joint_dict in joint_dict_seq:
        robot.update_cfg(joint_dict)
        eef_transform = robot.get_transform(EEF_LINK_NAME)
        pose_seq.append(_transform_to_pose(eef_transform))
    return np.asarray(pose_seq, dtype=np.float32)


def visualize_joint_sequence(
    joint_seq: np.ndarray,
    fps: float = 10.0,
    repeat: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    robot = _load_robot()
    joint_seq_full, joint_dict_seq = _normalize_joint_sequence(joint_seq)
    limits = _compute_workspace_limits(robot, joint_dict_seq)

    fig = plt.figure(figsize=(14, 7))
    grid = GridSpec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0], figure=fig)
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_arm = fig.add_subplot(grid[0, 1])
    ax_gripper = fig.add_subplot(grid[1, 1])

    arm_traj = joint_seq_full[:, :6]
    gripper_width = None
    if joint_seq.shape[1] == 7:
        gripper_width = np.asarray(joint_seq[:, 6], dtype=np.float32)
    else:
        gripper_width = _robotiq_joint_angles_to_width(joint_seq_full)

    ax3d.set_title("UR5 Robotiq UMI Joint Motion")
    ax3d.view_init(elev=24, azim=42)
    ax3d.set_box_aspect((1.0, 1.0, 1.0))
    ax3d.set_xlim(limits[0, 0], limits[1, 0])
    ax3d.set_ylim(limits[0, 1], limits[1, 1])
    ax3d.set_zlim(limits[0, 2], limits[1, 2])
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")

    arm_line, = ax3d.plot([], [], [], "-o", linewidth=3, markersize=5, color="#1f77b4")
    left_line, = ax3d.plot([], [], [], "-o", linewidth=2, markersize=4, color="#ff7f0e")
    right_line, = ax3d.plot([], [], [], "-o", linewidth=2, markersize=4, color="#2ca02c")
    eef_point = ax3d.scatter([], [], [], s=60, color="#d62728")
    frame_text = ax3d.text2D(0.02, 0.96, "", transform=ax3d.transAxes)

    ts = np.arange(len(joint_seq_full), dtype=np.float32) / fps
    arm_cursor = []
    for joint_idx, joint_name in enumerate(ARM_JOINT_NAMES):
        line, = ax_arm.plot(ts, arm_traj[:, joint_idx], label=joint_name)
        cursor = ax_arm.axvline(ts[0], color=line.get_color(), alpha=0.25, linewidth=1)
        arm_cursor.append(cursor)
    ax_arm.set_title("Arm Joint Angles")
    ax_arm.set_xlabel("time (s)")
    ax_arm.set_ylabel("rad")
    ax_arm.grid(True, alpha=0.3)
    ax_arm.legend(loc="upper right", fontsize=8)

    ax_gripper.plot(ts, gripper_width, color="#9467bd", linewidth=2)
    grip_cursor = ax_gripper.axvline(ts[0], color="#9467bd", alpha=0.35, linewidth=1.5)
    ax_gripper.set_title("Gripper Width")
    ax_gripper.set_xlabel("time (s)")
    ax_gripper.set_ylabel("m")
    ax_gripper.grid(True, alpha=0.3)

    def _update(frame_idx: int):
        joint_dict = joint_dict_seq[frame_idx]
        robot.update_cfg(joint_dict)

        arm_points = _collect_chain_positions(robot, ARM_LINK_CHAIN)
        left_points = _collect_chain_positions(robot, LEFT_FINGER_CHAIN)
        right_points = _collect_chain_positions(robot, RIGHT_FINGER_CHAIN)

        arm_line.set_data(arm_points[:, 0], arm_points[:, 1])
        arm_line.set_3d_properties(arm_points[:, 2])

        left_line.set_data(left_points[:, 0], left_points[:, 1])
        left_line.set_3d_properties(left_points[:, 2])

        right_line.set_data(right_points[:, 0], right_points[:, 1])
        right_line.set_3d_properties(right_points[:, 2])

        eef = arm_points[-1]
        eef_point._offsets3d = ([eef[0]], [eef[1]], [eef[2]])
        frame_text.set_text(
            f"frame {frame_idx + 1}/{len(joint_dict_seq)}\n"
            f"gripper = {gripper_width[frame_idx]:.4f} m"
        )

        current_t = ts[frame_idx]
        for cursor in arm_cursor:
            cursor.set_xdata([current_t, current_t])
        grip_cursor.set_xdata([current_t, current_t])

        return arm_line, left_line, right_line, eef_point, frame_text, *arm_cursor, grip_cursor

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(joint_dict_seq),
        interval=1000.0 / fps,
        blit=False,
        repeat=repeat,
    )

    writer = _maybe_make_writer(save_path, fps)
    if writer is not None:
        ani.save(save_path, writer=writer, dpi=120)
        print(f"Saved animation to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_joint_sequence_as_eef(
    joint_seq: np.ndarray,
    fps: float = 10.0,
    repeat: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    from visualize_ur5_robotiq_eef_sequence import visualize_eef_sequence

    robot = _load_robot()
    joint_seq_full, joint_dict_seq = _normalize_joint_sequence(joint_seq)
    pose_seq = _compute_eef_pose_sequence(robot, joint_seq_full, joint_dict_seq)
    if joint_seq.shape[1] == 7:
        gripper_width_seq = np.asarray(joint_seq[:, 6], dtype=np.float32)
    else:
        gripper_width_seq = _robotiq_joint_angles_to_width(joint_seq_full)

    visualize_eef_sequence(
        pose_seq=pose_seq,
        gripper_width_seq=gripper_width_seq,
        fps=fps,
        save_path=save_path,
        show=show,
        repeat=repeat,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joint-seq",
        type=str,
        default=None,
        help="Inline joint sequence as JSON or Python literal.",
    )
    parser.add_argument(
        "--joint-seq-file",
        type=str,
        default=None,
        help="Path to a .npy/.npz/.json/.txt file containing the joint sequence.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Playback FPS.")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output path (.gif or .mp4) for saving the animation.",
    )
    parser.add_argument(
        "--eef-only",
        action="store_true",
        help="Visualize only the end-effector trajectory in the same style as the EEF script.",
    )
    parser.add_argument(
        "--no-repeat",
        action="store_true",
        help="Play the animation once instead of looping.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window; useful with --save.",
    )
    args = parser.parse_args()

    if args.joint_seq is None and args.joint_seq_file is None:
        parser.error("Please provide either --joint-seq or --joint-seq-file.")

    joint_seq = _load_joint_sequence(args.joint_seq_file, args.joint_seq)
    if args.eef_only:
        visualize_joint_sequence_as_eef(
            joint_seq=joint_seq,
            fps=args.fps,
            repeat=not args.no_repeat,
            save_path=args.save,
            show=not args.no_show,
        )
    else:
        visualize_joint_sequence(
            joint_seq=joint_seq,
            fps=args.fps,
            repeat=not args.no_repeat,
            save_path=args.save,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
