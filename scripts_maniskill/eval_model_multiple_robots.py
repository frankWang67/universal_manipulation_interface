import os
import sys
import subprocess
import re
from argparse import ArgumentParser
from datetime import datetime

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "eval_model.py")

parser = ArgumentParser(description="Multi-Robot Policy Model Evaluation")
parser.add_argument("--input", "-i", type=str, required=True, help="Path to checkpoint and experiment results")
parser.add_argument("--ckpt-filename", "-f", type=str, required=True, help="Checkpoint filename within the experiment folder")
parser.add_argument("--env-id", "-e", type=str, required=True, help="Environment ID")
parser.add_argument("--sim-backend", "-s", type=str, default="physx_cpu", help="Simulation backend for ManiSkill env")
parser.add_argument("--control-mode", "-c", type=str, default="pd_ee_pose", help="Control mode for ManiSkill env")
parser.add_argument("--num-env", "-n", type=int, default=10, help="Number of parallel environments")
parser.add_argument("--num-eval-episodes", "-ne", type=int, default=100, help="Number of evaluation episodes")
parser.add_argument("--obs-mode", "-o", type=str, default="rgb", help="Observation mode for ManiSkill env")
parser.add_argument("--render-mode", "-rm", type=str, default="all", help="Render mode for ManiSkill env")
parser.add_argument("--steps-per-inference", "-si", type=int, default=8, help="Number of predicted actions to execute per policy call. Use 0 to execute the checkpoint action horizon.")
parser.add_argument("--max-episode-steps", "-mes", type=int, default=500, help="Max episode steps for evaluation")
parser.add_argument("--joint-space", action="store_true", help="Whether the policy is a joint space diffusion policy")
parser.add_argument("--harder", action="store_true", help="Whether to evaluate on harder version of the environment")
parser.add_argument("--joint-space-guidance", action="store_true", help="Whether to add guidance for the policy during evaluation")
parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale for joint-space guidance")
parser.add_argument("--guidance-safety-margin", type=float, default=0.05, help="Safety margin for joint-space guidance")
parser.add_argument("--guidance-grad-clip", type=float, default=0.1, help="Per-step joint-space guidance clip")
parser.add_argument("--guidance-task-rot-weight", type=float, default=0.1, help="Rotation weight in the joint-space CBF task metric")
parser.add_argument("--guidance-use-clean-sample", action="store_true", help="Apply guidance on an estimated clean x0 joint sample")
parser.add_argument("--guidance-apply-last-step-only", action="store_true", help="Apply guidance only at the final denoising step")
args = parser.parse_args()

# ================= 配置区域 =================
tasks = [
    {
        "name": "panda",
        "robot_uid": "panda_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "xarm6",
        "robot_uid": "xarm6_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "xarm7",
        "robot_uid": "xarm7_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "ur5",
        "robot_uid": "ur5_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "floating_robotiq",
        "robot_uid": "floating_robotiq_2f_85_gripper_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "iiwa7",
        "robot_uid": "iiwa7_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "gen3_6dof",
        "robot_uid": "gen3_6dof_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "gen3_7dof",
        "robot_uid": "gen3_7dof_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "rizon4",
        "robot_uid": "rizon4_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "sawyer",
        "robot_uid": "sawyer_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
]

# ===========================================


def build_command(task):
    cmd = f"python {task['script_path']}"
    cmd += f" --input {args.input}"
    cmd += f" --ckpt_filename {args.ckpt_filename}"
    cmd += f" --env_id {args.env_id}"
    cmd += f" --robot_uids {task['robot_uid']}"
    cmd += f" --sim_backend {args.sim_backend}"
    cmd += f" --control_mode {args.control_mode}"
    cmd += f" --num_env {args.num_env}"
    cmd += f" --num_eval_episodes {args.num_eval_episodes}"
    cmd += f" --obs_mode {args.obs_mode}"
    cmd += f" --render_mode {args.render_mode}"
    cmd += f" --steps_per_inference {args.steps_per_inference}"
    cmd += f" --max_episode_steps {args.max_episode_steps}"
    if args.joint_space:
        cmd += " --joint_space"
    if args.harder:
        cmd += " --harder"
    if args.joint_space_guidance:
        cmd += " --joint_space_guidance"
        cmd += f" --guidance_scale {args.guidance_scale}"
        cmd += f" --guidance_safety_margin {args.guidance_safety_margin}"
        cmd += f" --guidance_grad_clip {args.guidance_grad_clip}"
        cmd += f" --guidance_task_rot_weight {args.guidance_task_rot_weight}"
        if args.guidance_use_clean_sample:
            cmd += " --guidance_use_clean_sample"
        if args.guidance_apply_last_step_only:
            cmd += " --guidance_apply_last_step_only"
    return cmd


def get_results_path(task):
    """Replicate the log dir logic from eval_model.py."""
    subdir = ""
    if args.joint_space_guidance:
        subdir = "joint_space_whole_body_guidance"
    elif args.harder:
        subdir = "without_guidance"
    if subdir:
        return os.path.join(args.input, "eval_results", task["robot_uid"], subdir, "eval_results.txt")
    return os.path.join(args.input, "eval_results", task["robot_uid"], "eval_results.txt")


def parse_results(filepath):
    """Parse eval_results.txt to dict.  Returns empty dict if file missing."""
    if not os.path.exists(filepath):
        return {}
    results = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^([a-z_]+):\s+(.+)$", line)
            if match:
                key = match.group(1)
                val = match.group(2)
                if val == "N/A":
                    results[key] = None
                else:
                    try:
                        results[key] = float(val)
                    except ValueError:
                        results[key] = val
    return results


def run_command(cmd):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        sys.exit(1)


def format_pct(val):
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def format_collision(val, std_val=None):
    if val is None:
        return "N/A"
    if std_val is not None:
        return f"{val:.1f}±{std_val:.1f}"
    return f"{val:.1f}"


def format_reward(val, std_val=None):
    if val is None:
        return "N/A"
    if std_val is not None:
        return f"{val:.3f}±{std_val:.3f}"
    return f"{val:.3f}"


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_cmd = "python " + " ".join(sys.argv)
    show_collisions = args.harder or args.joint_space_guidance

    # Collect results from each robot
    rows = []
    for task in tasks:
        print(f"\n=== Evaluating Robot: {task['name']} ===")
        cmd = build_command(task)
        run_command(cmd)

        results_path = get_results_path(task)
        results = parse_results(results_path)
        rows.append({
            "name": task["name"],
            "success_once": results.get("success_once_rate"),
            "success_at_end": results.get("success_at_end_rate"),
            "collision_mean": results.get("avg_collision_per_episode"),
            "collision_std": results.get("std_collision_per_episode"),
            "max_reward_mean": results.get("avg_max_reward_per_episode"),
            "max_reward_std": results.get("std_max_reward_per_episode"),
            "fail_in_never_success": results.get("fail_rate_among_non_success"),
        })
        print(f"  Parsed results: {rows[-1]}")

    # Build markdown table
    header_cols = ["Robot", "Success Once", "Success At End"]
    if show_collisions:
        header_cols.append("Collisions/Ep")
    header_cols.append("Max Reward/Ep")
    header_cols.append("Fail@NeverSuccess")
    header = "| " + " | ".join(header_cols) + " |"
    sep = "|" + "|".join([":-:" for _ in header_cols]) + "|"

    data_rows = []
    for r in rows:
        cols = [r["name"], format_pct(r["success_once"]), format_pct(r["success_at_end"])]
        if show_collisions:
            cols.append(format_collision(r["collision_mean"], r["collision_std"]))
        cols.append(format_reward(r["max_reward_mean"], r["max_reward_std"]))
        cols.append(format_pct(r["fail_in_never_success"]))
        data_rows.append("| " + " | ".join(cols) + " |")

    # Average row
    if len(rows) > 1:
        valid_so = [r["success_once"] for r in rows if r["success_once"] is not None]
        valid_sae = [r["success_at_end"] for r in rows if r["success_at_end"] is not None]
        avg_cols = ["**Average**"]
        avg_cols.append(format_pct(sum(valid_so) / len(valid_so)) if valid_so else "N/A")
        avg_cols.append(format_pct(sum(valid_sae) / len(valid_sae)) if valid_sae else "N/A")
        if show_collisions:
            valid_cm = [r["collision_mean"] for r in rows if r["collision_mean"] is not None]
            valid_cs = [r["collision_std"] for r in rows if r["collision_std"] is not None]
            avg_cols.append(format_collision(
                sum(valid_cm) / len(valid_cm) if valid_cm else None,
                sum(valid_cs) / len(valid_cs) if valid_cs else None))
        valid_mr = [r["max_reward_mean"] for r in rows if r["max_reward_mean"] is not None]
        valid_mrs = [r["max_reward_std"] for r in rows if r["max_reward_std"] is not None]
        avg_cols.append(format_reward(
            sum(valid_mr) / len(valid_mr) if valid_mr else None,
            sum(valid_mrs) / len(valid_mrs) if valid_mrs else None))
        valid_fn = [r["fail_in_never_success"] for r in rows if r["fail_in_never_success"] is not None]
        avg_cols.append(format_pct(sum(valid_fn) / len(valid_fn)) if valid_fn else "N/A")
        data_rows.append("| " + " | ".join(avg_cols) + " |")

    results_dir = os.path.join(args.input, "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    md_path = os.path.join(results_dir, "results.md")

    block = f"\n## {timestamp}\n\n"
    block += f"Command: `{full_cmd}`\n\n"
    block += header + "\n" + sep + "\n" + "\n".join(data_rows) + "\n"

    # Check if file already has content (append), otherwise create with header
    if os.path.exists(md_path):
        with open(md_path, "a") as f:
            f.write(block)
    else:
        with open(md_path, "w") as f:
            f.write("# Evaluation Results\n")
            f.write(block)

    print(f"\nResults appended to {md_path}")


if __name__ == "__main__":
    main()
