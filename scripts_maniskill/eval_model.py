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
):
    # load checkpoint
    exp_path = input
    ckpt_path = os.path.join(exp_path, 'ckpt', ckpt_filename)
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = ckpt_path + '.ckpt'
    assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} does not exist."
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    if add_guidance:
        cfg.policy._target_ = "diffusion_policy.policy.diffusion_unet_timm_policy_with_guidance.DiffusionUnetTimmPolicyWithGuidance"
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    video_dir = os.path.join(exp_path, 'eval_results', robot_uids)
    if add_guidance:
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
    if harder or add_guidance:
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
        if add_guidance:
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
        num_eval_episodes, cfg, policy, env, steps_per_inference, add_guidance, device, 
    )
    print("Evaluation results over {} episodes:".format(num_eval_episodes))
    success_once_rate = np.mean(eval_metrics["success_once"])
    success_at_end_rate = np.mean(eval_metrics["success_at_end"])
    print(f"{success_once_rate=}")
    print(f"{success_at_end_rate=}")

    log_dir = os.path.join(exp_path, 'eval_results', robot_uids)
    if add_guidance:
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