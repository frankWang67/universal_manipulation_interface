# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*CUDA reports that you have.*")

# %%
import click
import dill
import hydra
import numpy as np
import torch
import pytorch_kinematics as pk
from omegaconf import OmegaConf
from omegaconf import open_dict
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import (
    get_real_umi_obs_dict,
)
from curobo.util_file import get_robot_configs_path, get_assets_path, join_path, load_yaml
from scripts_maniskill.utils import (
    make_eval_envs, 
    maniskill_to_umi_env_obs, 
    get_maniskill_umi_action, 
    get_maniskill_joint_action,
    evaluate,
)

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _infer_robot_kinematic_args(robot_cfg_name: str):
    robot_cfg_dict = load_yaml(join_path(get_robot_configs_path(), robot_cfg_name))["robot_cfg"]
    kin_cfg = robot_cfg_dict.get("kinematics", {})
    urdf_rel = kin_cfg.get("urdf_path")
    if urdf_rel is None:
        return None, kin_cfg.get("ee_link", "eef"), None

    robot_urdf_path = join_path(get_assets_path(), urdf_rel)
    ee_link_name = kin_cfg.get("ee_link", "eef")

    arm_dof = None
    try:
        with open(robot_urdf_path, "r") as f:
            urdf_str = f.read()
        try:
            chain = pk.build_serial_chain_from_urdf(urdf_str, ee_link_name)
        except ValueError:
            chain = pk.build_serial_chain_from_urdf(urdf_str.encode("utf-8"), ee_link_name)
        arm_dof = int(chain.n_joints)
    except Exception:
        arm_dof = None

    return robot_urdf_path, ee_link_name, arm_dof

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint and experiment results')
@click.option('--ckpt_filename', '-f', required=True, help='Checkpoint filename within the experiment folder')
@click.option('--env_id', '-e', required=True, help='ManiSkill environment id')
@click.option('--robot_uids', '-r', required=True, help='Robot UIDs in ManiSkill env')
@click.option('--sim_backend', '-s', default='physx_cpu', help='Simulation backend for ManiSkill env')
@click.option('--control_mode', '-c', default='pd_ee_pose', help='Control mode for ManiSkill env')
@click.option('--num_env', '-n', default=10, type=int, help='Number of parallel environments')
@click.option('--num_eval_episodes', '-ne', default=100, type=int, help='Number of evaluation episodes')
@click.option('--obs_mode', '-o', default='rgb', help='Observation mode for ManiSkill env')
@click.option('--render_mode', '-rm', default='all', help='Render mode for ManiSkill env')
@click.option('--steps_per_inference', '-si', default=0, type=int, help="Number of predicted actions to execute per policy call. Use 0 to execute cfg.task.action_horizon.")
@click.option('--max_episode_steps', '-mes', default=500, type=int, help="Max episode steps for evaluation.")
@click.option('--harder', is_flag=True, help="Whether to use harder environment setting with more obstacles and narrower workspace.")
@click.option('--add_guidance', is_flag=True, help="Whether to add obstacle avoidance guidance during inference.")
@click.option('--joint_space', is_flag=True, help="Whether the policy is a joint space diffusion policy.")
@click.option('--joint_space_guidance', is_flag=True, help="Whether to use joint-space policy with whole-body collision guidance.")
@click.option('--guidance_scale', default=1.0, type=float, help="Guidance scale for joint-space whole-body collision guidance.")
@click.option('--guidance_safety_margin', default=0.05, type=float, help="Safety margin (meters) used in collision guidance loss.")
@click.option('--guidance_activation_distance', default=1.0, type=float, help="Activation distance (meters) for cuRobo SDF query.")
@click.option('--guidance_grad_clip', default=0.1, type=float, help="Per-step gradient clip for joint-space guidance.")
@click.option('--guidance_method', default='cbf', type=str, help="Guidance method to use, one of ['cbf', 'gd']")
@click.option('--guidance_sdf_agg', default='topk', type=str, help="SDF aggregation over robot spheres: one of ['max', 'topk'] (legacy 'softmax' is treated as 'topk').")
@click.option('--guidance_sdf_softmax_temp', default=20.0, type=float, help="Temperature for top-k normalized smooth-max aggregation.")
@click.option('--guidance_sdf_topk', default=4, type=int, help="Number of top spheres used by top-k SDF aggregation.")
@click.option('--guidance_task_pos_weight', default=1.0, type=float, help="Position weight in the joint-space CBF task metric.")
@click.option('--guidance_task_rot_weight', default=0.1, type=float, help="Rotation weight in the joint-space CBF task metric.")
@click.option('--guidance_use_clean_sample', is_flag=True, help="Apply joint-space guidance on an estimated clean x0 joint sample.")
@click.option('--guidance_apply_last_step_only', is_flag=True, help="Apply joint-space guidance only at the final denoising step.")
@click.option('--ik_refine_last_step', is_flag=True, help="Run a cuRobo IK projection at the last denoising step for joint-space policies.")
@click.option('--cartesian_delta_mode', default='geometric', type=click.Choice(['geometric', 'se3_delta']), help="Pose residual used by joint-space policies.")
def main(
    input, 
    ckpt_filename, 
    env_id, 
    robot_uids, 
    sim_backend, 
    control_mode, 
    num_env, 
    num_eval_episodes, 
    obs_mode, 
    render_mode, 
    steps_per_inference, 
    max_episode_steps,
    harder,
    add_guidance,
    joint_space,
    joint_space_guidance,
    guidance_scale,
    guidance_safety_margin,
    guidance_activation_distance,
    guidance_grad_clip,
    guidance_method,
    guidance_sdf_agg,
    guidance_sdf_softmax_temp,
    guidance_sdf_topk,
    guidance_task_pos_weight,
    guidance_task_rot_weight,
    guidance_use_clean_sample,
    guidance_apply_last_step_only,
    ik_refine_last_step,
    cartesian_delta_mode,
):
    # load checkpoint
    exp_path = input
    if os.path.isfile(input):
        ckpt_path = input
        exp_path = os.path.dirname(os.path.dirname(ckpt_path))
    else:
        ckpt_path = os.path.join(exp_path, 'ckpt', ckpt_filename)
        if not (ckpt_path.endswith('.ckpt') or ckpt_path.endswith('.pth')):
            ckpt_path_ckpt = ckpt_path + '.ckpt'
            ckpt_path_pth = ckpt_path + '.pth'
            if os.path.exists(ckpt_path_ckpt):
                ckpt_path = ckpt_path_ckpt
            elif os.path.exists(ckpt_path_pth):
                ckpt_path = ckpt_path_pth
            else:
                ckpt_path = ckpt_path_ckpt
    assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} does not exist."
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    if steps_per_inference <= 0:
        steps_per_inference = int(cfg.task.action_horizon)

    robot_cfg_name_map = {
        'panda_robotiq_wristcam': 'panda_robotiq_wristcam.yml',
        'ur5_robotiq_wristcam': 'ur5_robotiq_wristcam.yml',
        'xarm6_robotiq_wristcam': 'xarm6_robotiq_wristcam.yml',
        'xarm7_robotiq_wristcam': 'xarm7_robotiq_wristcam.yml',
        'floating_robotiq_2f_85_gripper_wristcam': 'floating_robotiq_wristcam.yml',
    }

    if joint_space_guidance:
        cfg.policy._target_ = (
            'diffusion_policy.policy.diffusion_unet_timm_policy_joint_space_with_guidance.'
            'DiffusionUnetTimmPolicyJointSpaceWithGuidance'
        )
        # cfg.policy._target_ = (
        #     'diffusion_policy.policy.diffusion_unet_timm_policy_joint_space_clean_sample_with_guidance.'
        #     'DiffusionUnetTimmPolicyJointSpaceCleanSampleWithGuidance'
        # )
        robot_cfg_name = robot_cfg_name_map.get(robot_uids, f'{robot_uids}.yml')
        robot_urdf_path, ee_link_name, arm_dof = _infer_robot_kinematic_args(robot_cfg_name)
        with open_dict(cfg.policy):
            cfg.policy.robot_uid = robot_uids
            cfg.policy.robot_cfg_name = robot_cfg_name
            if robot_urdf_path is not None:
                cfg.policy.robot_urdf_path = robot_urdf_path
            cfg.policy.ee_link_name = ee_link_name
            if arm_dof is not None:
                cfg.policy.arm_dof = arm_dof
            cfg.policy.guidance_scale = guidance_scale
            cfg.policy.guidance_safety_margin = guidance_safety_margin
            cfg.policy.guidance_activation_distance = guidance_activation_distance
            cfg.policy.guidance_grad_clip = guidance_grad_clip
            cfg.policy.guidance_method = guidance_method
            cfg.policy.guidance_sdf_agg = guidance_sdf_agg
            cfg.policy.guidance_sdf_softmax_temp = guidance_sdf_softmax_temp
            cfg.policy.guidance_sdf_topk = guidance_sdf_topk
            cfg.policy.guidance_task_pos_weight = guidance_task_pos_weight
            cfg.policy.guidance_task_rot_weight = guidance_task_rot_weight
            cfg.policy.guidance_use_clean_sample = guidance_use_clean_sample
            cfg.policy.guidance_apply_last_step_only = guidance_apply_last_step_only
            cfg.policy.ik_refine_last_step = ik_refine_last_step
            cfg.policy.cartesian_delta_mode = cartesian_delta_mode
    elif add_guidance:
        cfg.policy._target_ = "diffusion_policy.policy.diffusion_unet_timm_policy_with_guidance.DiffusionUnetTimmPolicyWithGuidance"
    elif joint_space:
        cfg.policy._target_ = "diffusion_policy.policy.diffusion_unet_timm_policy_joint_space.DiffusionUnetTimmPolicyJointSpace"
        # cfg.policy._target_ = "diffusion_policy.policy.diffusion_unet_timm_policy_joint_space_clean_sample.DiffusionUnetTimmPolicyJointSpaceCleanSample"
        robot_cfg_name = robot_cfg_name_map.get(robot_uids, f'{robot_uids}.yml')
        robot_urdf_path, ee_link_name, arm_dof = _infer_robot_kinematic_args(robot_cfg_name)
        with open_dict(cfg.policy):
            cfg.policy.robot_cfg_name = robot_cfg_name
            if robot_urdf_path is not None:
                cfg.policy.robot_urdf_path = robot_urdf_path
            cfg.policy.ee_link_name = ee_link_name
            if arm_dof is not None:
                cfg.policy.arm_dof = arm_dof
            cfg.policy.ik_refine_last_step = ik_refine_last_step
            cfg.policy.cartesian_delta_mode = cartesian_delta_mode
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    video_dir = os.path.join(exp_path, 'eval_results', robot_uids)
    if joint_space_guidance:
        video_dir = os.path.join(video_dir, 'joint_space_whole_body_guidance')
    elif add_guidance:
        video_dir = os.path.join(video_dir, 'guided_diffusion')
    elif harder:
        video_dir = os.path.join(video_dir, 'without_guidance')
    video_dir = os.path.join(video_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    env_kwargs = dict(
        robot_uids=robot_uids,
        control_mode=control_mode,
        obs_mode=obs_mode,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        sensor_configs=dict(shader_pack="default"),
        reward_mode="normalized_dense",
    )
    if harder or add_guidance or joint_space_guidance:
        env_kwargs['harder'] = True
    track_collisions = harder or joint_space_guidance
    other_kwargs = dict(obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon)
    env = make_eval_envs(
        env_id,
        num_env,
        sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=video_dir,
        wrappers=[FlattenRGBDObservationWrapper],
        track_collisions=track_collisions,
    )

    # creating model
    # have to be done after fork to prevent 
    # duplicating CUDA context with ffmpeg nvenc
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.num_inference_steps = 16 # DDIM inference iterations
    obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr
    print('obs_pose_rep', obs_pose_rep)
    print('action_pose_repr', action_pose_repr)

    device = torch.device('cuda')
    policy.eval().to(device)

    print("Warming up policy inference")
    obs, info = env.reset()
    obs = maniskill_to_umi_env_obs(obs)
    with torch.no_grad():
        policy.reset()
        # [新增] 获取当前位姿作为临时的 start_pose 用于预热
        episode_start_pose = [np.concatenate([
            obs['robot0_eef_pos'][:, -1], 
            obs['robot0_eef_rot_axis_angle'][:, -1]
        ], axis=-1)]
        obs_dict_np = get_real_umi_obs_dict(
            env_obs=obs, shape_meta=cfg.task.shape_meta, 
            obs_pose_repr=obs_pose_rep,
            episode_start_pose=episode_start_pose,
            batched=True,
        )
        obs_dict = dict_apply(obs_dict_np, 
            lambda x: torch.from_numpy(x).to(device))
        if add_guidance or joint_space or joint_space_guidance:
            episode_start_pose_tensor = torch.from_numpy(episode_start_pose[0]).to(device)
        else:
            episode_start_pose_tensor = None
        result = policy.predict_action(obs_dict, env_batched=False, episode_start_pose=episode_start_pose_tensor)
        if control_mode.startswith("pd_joint"):
            action = result["joint_action_pred"].detach().to("cpu").numpy()
            single_action_space = getattr(env, "single_action_space", env.action_space)
            action = get_maniskill_joint_action(action, action_space=single_action_space)
            assert action.shape[-1] == single_action_space.shape[-1]
        else:
            action = result['action_pred'].detach().to('cpu').numpy()
            assert action.shape[-1] == 10
            action = get_maniskill_umi_action(action, obs, action_pose_repr, batched=True)
            single_action_space = getattr(env, "single_action_space", env.action_space)
            assert action.shape[-1] == single_action_space.shape[-1]
        del result

    print('Ready! Start evaluation.')
    eval_metrics = evaluate(
        num_eval_episodes, cfg, policy, env, steps_per_inference,
        (add_guidance or joint_space_guidance),
        (joint_space or joint_space_guidance),
        device,
        control_mode=control_mode,
    )
    print("Evaluation results over {} episodes:".format(num_eval_episodes))
    success_once_rate = float(np.mean(eval_metrics["success_once"]))
    success_at_end_rate = float(np.mean(eval_metrics["success_at_end"]))
    print(f"{success_once_rate=}")
    print(f"{success_at_end_rate=}")

    # ---- collisions ----
    if "collision_count" in eval_metrics:
        avg_collision = float(np.mean(eval_metrics["collision_count"]))
        std_collision = float(np.std(eval_metrics["collision_count"]))
        print(f"avg collisions/ep: {avg_collision:.2f}  std: {std_collision:.2f}")
    else:
        avg_collision = None
        std_collision = None

    # ---- max reward ----
    if "max_reward" in eval_metrics:
        avg_max_reward = float(np.mean(eval_metrics["max_reward"]))
        std_max_reward = float(np.std(eval_metrics["max_reward"]))
        print(f"avg max reward/ep: {avg_max_reward:.4f}  std: {std_max_reward:.4f}")
    else:
        avg_max_reward = None
        std_max_reward = None

    # ---- fail rate among non-success episodes ----
    if "fail_once" in eval_metrics:
        success_once = eval_metrics["success_once"]
        fail_once = eval_metrics["fail_once"]
        non_success_mask = success_once == 0
        if non_success_mask.sum() > 0:
            fail_rate_among_non_success = float(fail_once[non_success_mask].mean())
        else:
            fail_rate_among_non_success = -1.0  # all episodes succeeded
        print(f"fail rate among non-success episodes: {fail_rate_among_non_success}")
    else:
        fail_rate_among_non_success = None

    log_dir = os.path.join(exp_path, 'eval_results', robot_uids)
    if joint_space_guidance:
        log_dir = os.path.join(log_dir, 'joint_space_whole_body_guidance')
    elif add_guidance:
        log_dir = os.path.join(log_dir, 'guided_diffusion')
    elif harder:
        log_dir = os.path.join(log_dir, 'without_guidance')
    log_filename = os.path.join(log_dir, 'eval_results.txt')
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    with open(log_filename, "w") as f:
        f.write(f"success_once_rate: {success_once_rate}\n")
        f.write(f"success_at_end_rate: {success_at_end_rate}\n")

        if avg_collision is not None:
            f.write(f"avg_collision_per_episode: {avg_collision}\n")
            f.write(f"std_collision_per_episode: {std_collision}\n")
        else:
            f.write("avg_collision_per_episode: N/A\n")
            f.write("std_collision_per_episode: N/A\n")

        if avg_max_reward is not None:
            f.write(f"avg_max_reward_per_episode: {avg_max_reward}\n")
            f.write(f"std_max_reward_per_episode: {std_max_reward}\n")
        else:
            f.write("avg_max_reward_per_episode: N/A\n")
            f.write("std_max_reward_per_episode: N/A\n")

        if fail_rate_among_non_success is not None:
            f.write(f"fail_rate_among_non_success: {fail_rate_among_non_success}\n")
        else:
            f.write("fail_rate_among_non_success: N/A\n")

# %%
if __name__ == '__main__':
    main()
