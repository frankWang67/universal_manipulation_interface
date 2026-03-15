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
from omegaconf import OmegaConf
from omegaconf import open_dict
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import (
    get_real_umi_obs_dict,
)
from scripts_maniskill.utils import (
    make_eval_envs, 
    maniskill_to_umi_env_obs, 
    get_maniskill_umi_action, 
    evaluate,
)

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

OmegaConf.register_new_resolver("eval", eval, replace=True)

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
@click.option('--steps_per_inference', '-si', default=8, type=int, help="Action horizon for inference.")
@click.option('--max_episode_steps', '-mes', default=500, type=int, help="Max episode steps for evaluation.")
@click.option('--harder', is_flag=True, help="Whether to use harder environment setting with more obstacles and narrower workspace.")
@click.option('--add_guidance', is_flag=True, help="Whether to add obstacle avoidance guidance during inference.")
@click.option('--joint_space', is_flag=True, help="Whether the policy is a joint space diffusion policy.")
@click.option('--joint_space_guidance', is_flag=True, help="Whether to use joint-space policy with whole-body collision guidance.")
@click.option('--guidance_scale', default=1.0, type=float, help="Guidance scale for joint-space whole-body collision guidance.")
@click.option('--guidance_safety_margin', default=0.05, type=float, help="Safety margin (meters) used in collision guidance loss.")
@click.option('--guidance_activation_distance', default=10.0, type=float, help="Activation distance (meters) for cuRobo SDF query.")
@click.option('--guidance_grad_clip', default=0.25, type=float, help="Per-step gradient clip for joint-space guidance.")
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
        with open_dict(cfg.policy):
            cfg.policy.robot_uid = robot_uids
            cfg.policy.robot_cfg_name = robot_cfg_name_map.get(robot_uids, f'{robot_uids}.yml')
            cfg.policy.guidance_scale = guidance_scale
            cfg.policy.guidance_safety_margin = guidance_safety_margin
            cfg.policy.guidance_activation_distance = guidance_activation_distance
            cfg.policy.guidance_grad_clip = guidance_grad_clip
    elif add_guidance:
        cfg.policy._target_ = "diffusion_policy.policy.diffusion_unet_timm_policy_with_guidance.DiffusionUnetTimmPolicyWithGuidance"
    elif joint_space:
        cfg.policy._target_ = "diffusion_policy.policy.diffusion_unet_timm_policy_joint_space.DiffusionUnetTimmPolicyJointSpace"
        with open_dict(cfg.policy):
            cfg.policy.robot_cfg_name = robot_cfg_name_map.get(robot_uids, f'{robot_uids}.yml')
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
    )
    if harder or add_guidance or joint_space_guidance:
        env_kwargs['harder'] = True
    other_kwargs = dict(obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon)
    env = make_eval_envs(
        env_id,
        num_env,
        sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=video_dir,
        wrappers=[FlattenRGBDObservationWrapper],
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
        action = result['action_pred'].detach().to('cpu').numpy()
        assert action.shape[-1] == 10
        action = get_maniskill_umi_action(action, obs, action_pose_repr, batched=True)
        assert action.shape[-1] == 8
        del result

    print('Ready! Start evaluation.')
    eval_metrics = evaluate(
        num_eval_episodes, cfg, policy, env, steps_per_inference,
        (add_guidance or joint_space_guidance),
        (joint_space or joint_space_guidance),
        device,
    )
    print("Evaluation results over {} episodes:".format(num_eval_episodes))
    success_once_rate = np.mean(eval_metrics["success_once"])
    success_at_end_rate = np.mean(eval_metrics["success_at_end"])
    print(f"{success_once_rate=}")
    print(f"{success_at_end_rate=}")

    log_dir = os.path.join(exp_path, 'eval_results', robot_uids)
    if joint_space_guidance:
        log_dir = os.path.join(log_dir, 'joint_space_whole_body_guidance')
    elif add_guidance:
        log_dir = os.path.join(log_dir, 'guided_diffusion')
    elif harder:
        log_dir = os.path.join(log_dir, 'without_guidance')
    log_filename = os.path.join(log_dir, 'eval_results.txt')
    with open(log_filename, "a") as f:
        f.write(f"Success Once Rate: {success_once_rate}\n")
        f.write(f"Success At End Rate: {success_at_end_rate}\n")

# %%
if __name__ == '__main__':
    main()