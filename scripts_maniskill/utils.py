import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from typing import Optional, Dict
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

from umi.common.pose_util import (
    pose_to_mat, 
    pose10d_to_mat,
    mat_to_pose,
    mat_to_euler_pose, 
    mat_to_quat_pose,
)
from umi.real_world.real_inference_util import get_real_umi_obs_dict
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep, batched_convert_pose_mat_rep

from mani_skill.utils import gym_utils, common
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

class AgentPoseWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    @property
    def world_to_robot_matrix(self):
        T_from_world_to_robot = self.unwrapped.agent.get_state()["robot_root_pose"].inv().to_transformation_matrix()
        return T_from_world_to_robot

def make_eval_envs(
    env_id,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    other_kwargs: dict,
    video_dir: Optional[str] = None,
    wrappers: list[gym.Wrapper] = [],
):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    if sim_backend == "physx_cpu":

        def cpu_make_env(
            env_id, seed, video_dir=None, env_kwargs=dict(), other_kwargs=dict()
        ):
            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir:
                    env = RecordEpisode(
                        env,
                        output_dir=video_dir,
                        save_trajectory=False,
                        info_on_video=True,
                        source_type="diffusion_policy",
                        source_desc="diffusion_policy evaluation rollout",
                    )
                env = AgentPoseWrapper(env)
                env.action_space.seed(seed)
                env.observation_space.seed(seed)

                if np.any(np.isnan(env.action_space.low)):
                    env.action_space.low[np.isnan(env.action_space.low)] = -np.inf
                if np.any(np.isnan(env.action_space.high)):
                    env.action_space.high[np.isnan(env.action_space.high)] = np.inf

                return env

            return thunk

        vector_cls = (
            gym.vector.SyncVectorEnv
            if num_envs == 1
            else lambda x: gym.vector.AsyncVectorEnv(x, context="forkserver")
        )
        env = vector_cls(
            [
                cpu_make_env(
                    env_id,
                    seed,
                    video_dir if seed == 0 else None,
                    env_kwargs,
                    other_kwargs,
                )
                for seed in range(num_envs)
            ]
        )
    else:
        env = gym.make(
            env_id,
            num_envs=num_envs,
            sim_backend=sim_backend,
            reconfiguration_freq=1,
            **env_kwargs
        )
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
        if video_dir:
            env = RecordEpisode(
                env,
                output_dir=video_dir,
                save_trajectory=False,
                save_video=True,
                source_type="diffusion_policy",
                source_desc="diffusion_policy evaluation rollout",
                max_steps_per_video=max_episode_steps,
            )
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env

def maniskill_to_umi_env_obs(
    maniskill_obs: dict, 
    robot_prefix: str = "robot0",
    camera_prefix: str = "camera0",
    maniskill_tcp_key: str = "state",
    maniskill_img_key: str = "rgb",
) -> dict:
    """
    将 ManiSkill 的单步观测转换为 UMI get_real_umi_obs_dict 需要的 env_obs 格式。
    
    Args:
        maniskill_obs: env.step() 返回的原始观测字典
        camera_mapping: UMI 配置中的 key (如 'camera0_rgb') 到 ManiSkill 相机名的映射
        robot_prefix: UMI 配置中的机器人前缀
        maniskill_tcp_key: ManiSkill 观测中存放 TCP 位姿的 key (通常在 'extra' 或 'agent' 下)
    """
    env_obs = {}

    # -------------------------------------------------------------------------
    # 1. 图像处理 (Image Processing)
    # -------------------------------------------------------------------------
    # ManiSkill 结构: obs['rgb'] -> [B, T, H, W, C]
    # UMI 需求: env_obs['camera0_rgb'] -> [B, T, H, W, C]
    
    img = maniskill_obs.get(maniskill_img_key, None)
    if img is None:
        raise ValueError(f"无法在 ManiSkill obs 中找到图像数据 (key: {maniskill_img_key})。请检查环境配置或 obs_mode。")
    env_obs[f'{camera_prefix}_rgb'] = img

    # -------------------------------------------------------------------------
    # 2. 机器人状态处理 (Proprioception)
    # -------------------------------------------------------------------------
    # ManiSkill 结构: obs['state'] -> [B, T, 7] (pos(3) + rot(4) + gripper(1))
    # UMI 需求: 'robot0_eef_pos' (B, T, 3), 'robot0_eef_rot_axis_angle' (B, T, 3)
    
    # 获取 ManiSkill 的 TCP Pose
    # 使用修改后的 FlattenRGBDObservationWrapper 返回的 observation
    tcp_pose = maniskill_obs[maniskill_tcp_key]
    if tcp_pose is None:
        raise ValueError(f"无法在 ManiSkill obs 中找到 TCP Pose (key: {maniskill_tcp_key})。请检查环境配置或 obs_mode。")

    # 分离位置和旋转
    pos = tcp_pose[..., :3] # [B, T, 3]
    axis_angle = tcp_pose[..., 3:6] # [B, T, 3] (assuming x, y, z)
    gripper_width = tcp_pose[..., 6:7]  # [B, T, 1]

    env_obs[f'{robot_prefix}_eef_pos'] = pos
    env_obs[f'{robot_prefix}_eef_rot_axis_angle'] = axis_angle

    # -------------------------------------------------------------------------
    # 3. 夹爪状态 (Gripper)
    # -------------------------------------------------------------------------
    env_obs[f'{robot_prefix}_gripper_width'] = gripper_width
    
    return env_obs

def get_maniskill_umi_action(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray], 
        action_pose_repr: str='abs',
        batched: bool=False,
    ):

    n_robots = int(action.shape[-1] // 10)
    env_action = list()
    for robot_idx in range(n_robots):
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            env_obs[f'robot{robot_idx}_eef_pos'][:, -1],
            env_obs[f'robot{robot_idx}_eef_rot_axis_angle'][:, -1]
        ], axis=-1))

        start = robot_idx * 10
        action_pose10d = action[..., start:start+9]
        action_grip = action[..., start+9:start+10]
        if batched:
            batch_size = action.shape[0]
            time_horizon = action.shape[1]
            action_pose10d = action_pose10d.reshape(batch_size * time_horizon, -1)
        action_pose_mat = pose10d_to_mat(action_pose10d)
        if batched:
            action_pose_mat = action_pose_mat.reshape(batch_size, time_horizon, 4, 4)

        # solve relative action
        convert_pose_mat_rep_func = batched_convert_pose_mat_rep if batched else convert_pose_mat_rep
        action_mat = convert_pose_mat_rep_func(
            action_pose_mat, 
            base_pose_mat=pose_mat,
            pose_rep=action_pose_repr,
            backward=True
        )

        # convert action to pose
        if batched:
            action_mat = action_mat.reshape(batch_size * time_horizon, 4, 4)
        # ===== CHANGE FROM EULER TO QUATERNION =====
        # action_pose = mat_to_euler_pose(action_mat)
        action_pose = mat_to_quat_pose(action_mat)
        # ===========================================
        if batched:
            action_pose = action_pose.reshape(batch_size, time_horizon, -1)
        env_action.append(action_pose)
        env_action.append(action_grip)

    env_action = np.concatenate(env_action, axis=-1)
    return env_action

def evaluate(n: int, cfg, policy, eval_envs, steps_per_inference, device, progress_bar: bool = True):
    assert steps_per_inference >= 1 and steps_per_inference <= cfg.task.action_horizon, \
        f"steps_per_inference should be in [1, {cfg.task.action_horizon}], but got {steps_per_inference}"
    obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr

    if progress_bar:
        pbar = tqdm(total=n)
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        while eps_count < n:
            # obs = common.to_tensor(obs, device)
            obs = maniskill_to_umi_env_obs(obs)
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
            result = policy.predict_action(obs_dict, env_batched=False)
            raw_action = result['action_pred'].detach().to('cpu').numpy()
            action_seq = get_maniskill_umi_action(raw_action, obs, action_pose_repr, batched=True)

            for i in range(steps_per_inference):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break

            if truncated.any():
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                eps_count += eval_envs.num_envs
                if progress_bar:
                    pbar.update(eval_envs.num_envs)

    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics