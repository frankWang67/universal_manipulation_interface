#!/usr/bin/env python3
"""
Visualize a UR5 + Robotiq UMI end-effector pose sequence directly, without IK.

Supported inputs
----------------
- (T, 6):  [x, y, z, rx, ry, rz] where r* is axis-angle / rotvec.
- (T, 7):  [x, y, z, rx, ry, rz, gripper_width]
- (T, 9):  UMI pose10d = [x, y, z, rot6d(6)]
- (T, 10): UMI pose10d + gripper_width

The visualizer draws:
- the 3D end-effector trajectory,
- the current end-effector local axes,
- position / rotation / gripper time curves.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from umi.common.pose_util import mat_to_pose, pose10d_to_mat

from visualize_ur5_robotiq_joint_sequence import _load_joint_sequence


def _normalize_eef_sequence(eef_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eef_seq = np.asarray(eef_seq, dtype=np.float32)
    if eef_seq.ndim == 1:
        eef_seq = eef_seq[None, :]
    if eef_seq.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {eef_seq.shape}.")

    dim = eef_seq.shape[1]
    if dim == 6:
        pose_seq = eef_seq
        gripper = np.full((len(eef_seq),), 0.085, dtype=np.float32)
    elif dim == 7:
        pose_seq = eef_seq[:, :6]
        gripper = eef_seq[:, 6].astype(np.float32)
    elif dim == 9:
        pose_seq = mat_to_pose(pose10d_to_mat(eef_seq))
        gripper = np.full((len(eef_seq),), 0.085, dtype=np.float32)
    elif dim == 10:
        pose_seq = mat_to_pose(pose10d_to_mat(eef_seq[:, :9]))
        gripper = eef_seq[:, 9].astype(np.float32)
    else:
        raise ValueError(
            f"Expected shape (T,6), (T,7), (T,9), or (T,10), got {eef_seq.shape}."
        )

    return np.asarray(pose_seq, dtype=np.float32), np.asarray(gripper, dtype=np.float32)


def _set_equal_3d_axes(ax, points: np.ndarray, margin: float = 0.08) -> float:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    span = max(span, 0.2)
    half = 0.5 * (span + margin)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    return span


def _maybe_make_writer(output_path: Optional[str], fps: float):
    if output_path is None:
        return None
    ext = Path(output_path).suffix.lower()
    if ext == ".gif":
        return animation.PillowWriter(fps=fps)
    if ext == ".mp4":
        return animation.FFMpegWriter(fps=fps)
    raise ValueError("Only .gif and .mp4 are supported for --save.")


def visualize_eef_sequence(
    pose_seq: np.ndarray,
    gripper_width_seq: np.ndarray,
    fps: float = 10.0,
    save_path: Optional[str] = None,
    show: bool = True,
    repeat: bool = True,
) -> None:
    pos_seq = pose_seq[:, :3]
    rot_seq = pose_seq[:, 3:]
    rot_mats = R.from_rotvec(rot_seq).as_matrix()

    fig = plt.figure(figsize=(14, 8))
    grid = GridSpec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0], figure=fig)
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_pos = fig.add_subplot(grid[0, 1])
    ax_aux = fig.add_subplot(grid[1, 1])

    ax3d.set_title("EEF Trajectory")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    span = _set_equal_3d_axes(ax3d, pos_seq)
    ax3d.view_init(elev=28, azim=40)

    axis_len = max(0.03, 0.12 * span)

    ax3d.plot(pos_seq[:, 0], pos_seq[:, 1], pos_seq[:, 2], "--", color="#9ecae1", linewidth=2)
    past_line, = ax3d.plot([], [], [], "-", color="#1f77b4", linewidth=3)
    current_point = ax3d.scatter([], [], [], s=65, color="#d62728")
    axis_x, = ax3d.plot([], [], [], color="#d62728", linewidth=2)
    axis_y, = ax3d.plot([], [], [], color="#2ca02c", linewidth=2)
    axis_z, = ax3d.plot([], [], [], color="#ffbf00", linewidth=2)
    info_text = ax3d.text2D(0.02, 0.96, "", transform=ax3d.transAxes)

    ts = np.arange(len(pose_seq), dtype=np.float32) / fps
    pos_labels = ["x", "y", "z"]
    pos_colors = ["#d62728", "#2ca02c", "#1f77b4"]
    pos_cursors = []
    for idx, (label, color) in enumerate(zip(pos_labels, pos_colors)):
        ax_pos.plot(ts, pos_seq[:, idx], label=label, color=color)
        pos_cursors.append(ax_pos.axvline(ts[0], color=color, alpha=0.2, linewidth=1))
    ax_pos.set_title("Position")
    ax_pos.set_xlabel("time (s)")
    ax_pos.set_ylabel("m")
    ax_pos.grid(True, alpha=0.3)
    ax_pos.legend(loc="upper right")

    rot_norm = np.linalg.norm(rot_seq, axis=-1)
    ax_aux.plot(ts, rot_norm, color="#9467bd", linewidth=2, label="|rotvec|")
    ax_aux.plot(ts, gripper_width_seq, color="#ff7f0e", linewidth=2, label="gripper")
    rot_cursor = ax_aux.axvline(ts[0], color="#9467bd", alpha=0.25, linewidth=1)
    grip_cursor = ax_aux.axvline(ts[0], color="#ff7f0e", alpha=0.25, linewidth=1)
    ax_aux.set_title("Rotation / Gripper")
    ax_aux.set_xlabel("time (s)")
    ax_aux.grid(True, alpha=0.3)
    ax_aux.legend(loc="upper right")

    def _update(frame_idx: int):
        pos = pos_seq[frame_idx]
        rot = rot_mats[frame_idx]

        past = pos_seq[: frame_idx + 1]
        past_line.set_data(past[:, 0], past[:, 1])
        past_line.set_3d_properties(past[:, 2])

        current_point._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        x_end = pos + axis_len * rot[:, 0]
        y_end = pos + axis_len * rot[:, 1]
        z_end = pos + axis_len * rot[:, 2]
        axis_x.set_data([pos[0], x_end[0]], [pos[1], x_end[1]])
        axis_x.set_3d_properties([pos[2], x_end[2]])
        axis_y.set_data([pos[0], y_end[0]], [pos[1], y_end[1]])
        axis_y.set_3d_properties([pos[2], y_end[2]])
        axis_z.set_data([pos[0], z_end[0]], [pos[1], z_end[1]])
        axis_z.set_3d_properties([pos[2], z_end[2]])

        info_text.set_text(
            f"frame {frame_idx + 1}/{len(pose_seq)}\n"
            f"pos = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n"
            f"gripper = {gripper_width_seq[frame_idx]:.4f} m"
        )

        t = ts[frame_idx]
        for cursor in pos_cursors:
            cursor.set_xdata([t, t])
        rot_cursor.set_xdata([t, t])
        grip_cursor.set_xdata([t, t])

        return (
            past_line,
            current_point,
            axis_x,
            axis_y,
            axis_z,
            info_text,
            *pos_cursors,
            rot_cursor,
            grip_cursor,
        )

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(pose_seq),
        interval=1000.0 / fps,
        repeat=repeat,
        blit=False,
    )

    writer = _maybe_make_writer(save_path, fps)
    if writer is not None:
        ani.save(save_path, writer=writer, dpi=120)
        print(f"Saved EEF animation to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eef-seq", type=str, default=None, help="Inline EEF sequence.")
    parser.add_argument(
        "--eef-seq-file",
        type=str,
        default=None,
        help="Path to a .npy/.npz/.json/.txt file containing the EEF sequence.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Playback FPS.")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output path (.gif/.mp4) for the EEF animation.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window.")
    parser.add_argument("--no-repeat", action="store_true", help="Play once instead of looping.")
    args = parser.parse_args()

    if args.eef_seq is None and args.eef_seq_file is None:
        parser.error("Please provide either --eef-seq or --eef-seq-file.")

    raw_seq = _load_joint_sequence(args.eef_seq_file, args.eef_seq)
    pose_seq, gripper_width_seq = _normalize_eef_sequence(raw_seq)
    visualize_eef_sequence(
        pose_seq=pose_seq,
        gripper_width_seq=gripper_width_seq,
        fps=args.fps,
        save_path=args.save,
        show=not args.no_show,
        repeat=not args.no_repeat,
    )


if __name__ == "__main__":
    main()
