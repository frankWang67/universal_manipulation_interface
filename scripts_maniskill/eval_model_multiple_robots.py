import os
import sys
import subprocess
from argparse import ArgumentParser

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
parser.add_argument(
    "--steps-per-inference",
    "-si",
    type=int,
    default=0,
    help="Number of predicted actions to execute per policy call. Use 0 to execute the checkpoint action horizon.",
)
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
# 在这里定义你想采集的所有机器人配置
# 注意：请确保 script 路径和 robot_uid 是正确的
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
]

# ===========================================

def run_command(cmd):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        sys.exit(1)

def main():
    for task in tasks:
        print(f"\n=== Evaluating Robot: {task['name']} ===")
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
        run_command(cmd)

if __name__ == "__main__":
    main()
