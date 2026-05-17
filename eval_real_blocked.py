"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import av
import click
import cv2
import yaml
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf, open_dict
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
# from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.real_world.keyboard_spacemouse_shared_memory import KeyboardSpacemouse as Spacemouse
from umi.common.pose_util import pose_to_mat, mat_to_pose

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _get_robot_cfg_name(robot_type: str) -> str:
    robot_type = str(robot_type).lower()
    robot_cfg_name_map = {
        'ur5': 'ur5_robotiq_umi.yml',
        'franka': 'panda_robotiq_umi.yml',
    }
    if robot_type not in robot_cfg_name_map:
        raise KeyError(f"Unsupported robot_type for joint-space policy adaptation: {robot_type}")
    return robot_cfg_name_map[robot_type]


def _robotiq_width_to_joint_angles(
    gripper_width,
    max_width: float = 0.085,
    outer_knuckle_max: float = 0.81,
    inner_knuckle_max: float = 0.8757,
):
    """
    Convert parallel jaw opening width (meters) into the 6-D Robotiq joint state
    expected by `ur5_robotiq_umi.yml`:
    [left_outer_knuckle, left_inner_knuckle, left_inner_finger,
     right_outer_knuckle, right_inner_knuckle, right_inner_finger].

    The 2F-85 URDF uses one scalar closure state replicated across the finger
    joints, with `inner_finger` rotating in the opposite direction.
    """
    width = np.asarray(gripper_width, dtype=np.float32)
    width = np.clip(width, 0.0, max_width)
    close_ratio = 1.0 - (width / max_width)

    outer_knuckle = close_ratio * outer_knuckle_max
    inner_knuckle = close_ratio * inner_knuckle_max
    inner_finger = -inner_knuckle

    return np.stack([
        outer_knuckle,
        inner_knuckle,
        inner_finger,
        outer_knuckle,
        inner_knuckle,
        inner_finger,
    ], axis=-1).astype(np.float32)


def _build_policy_current_joint_angles(
    arm_joint_angles,
    gripper_width,
    robot_cfg_name: str,
):
    arm_joint_angles = np.asarray(arm_joint_angles, dtype=np.float32)
    if robot_cfg_name == 'ur5_robotiq_umi.yml':
        gripper_joint_angles = _robotiq_width_to_joint_angles(gripper_width)
        return np.concatenate([arm_joint_angles, gripper_joint_angles], axis=-1)
    return arm_joint_angles


def _get_policy_current_joint_angles_from_env(
    env,
    obs,
    robot_cfg_name: str,
    source: str,
):
    aligned_joint_pos = np.asarray(obs['robot0_joint_pos'][-1], dtype=np.float32)
    aligned_gripper_width = float(
        np.asarray(obs['robot0_gripper_width'][-1]).squeeze()
    )

    latest_robot_state = env.get_robot_state()[0]
    latest_gripper_state = env.get_gripper_state()[0]
    latest_joint_pos = np.asarray(latest_robot_state['ActualQ'], dtype=np.float32)
    target_joint_pos = np.asarray(latest_robot_state['TargetQ'], dtype=np.float32)
    latest_gripper_width = float(latest_gripper_state['gripper_position'])

    if source == 'latest_actual':
        seed_joint_pos = latest_joint_pos
        seed_gripper_width = latest_gripper_width
    elif source == 'aligned_obs':
        seed_joint_pos = aligned_joint_pos
        seed_gripper_width = aligned_gripper_width
    else:
        raise ValueError(f"Unknown joint seed source: {source}")

    current_joint_angles_np = _build_policy_current_joint_angles(
        arm_joint_angles=seed_joint_pos,
        gripper_width=seed_gripper_width,
        robot_cfg_name=robot_cfg_name,
    )
    debug = {
        'aligned_joint_pos': aligned_joint_pos,
        'latest_joint_pos': latest_joint_pos,
        'target_joint_pos': target_joint_pos,
        'aligned_gripper_width': aligned_gripper_width,
        'latest_gripper_width': latest_gripper_width,
    }
    return current_joint_angles_np, debug


def _get_current_action_base_pose(obs, robot_id: int = 0) -> np.ndarray:
    return np.concatenate([
        obs[f'robot{robot_id}_eef_pos'][-1],
        obs[f'robot{robot_id}_eef_rot_axis_angle'][-1],
    ], axis=-1).astype(np.float32)

def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta

def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal
                
                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='0')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_swap', is_flag=True, default=False)
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
@click.option('--max_joint_speed', default=0.6, type=float, help="Joint-space waypoint speed limit in rad/s.")
@click.option('--blocked_action_lead_time', default=0.25, type=float, help="Lead time before the first blocking chunk waypoint, after latency compensation, in seconds.")
@click.option('--blocked_settle_time', default=0.5, type=float, help="How long robot/gripper velocities must stay low before taking the next observation.")
@click.option('--blocked_joint_vel_threshold', default=0.01, type=float, help="Joint velocity threshold in rad/s used to decide the robot is static.")
@click.option('--blocked_gripper_vel_threshold', default=0.002, type=float, help="Gripper velocity threshold in m/s used to decide the gripper is static.")
@click.option('--joint_seed_source', default='latest_actual', type=click.Choice(['latest_actual', 'aligned_obs']), help="Joint state used as current_joint_angles for joint-space policy inference.")
@click.option('--blocked_action_start_idx', default=0, type=int, help="Drop this many leading actions from each blocking chunk before execution.")
def main(input, output, robot_config, 
    match_dataset, match_episode, match_camera,
    camera_reorder,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency, 
    no_mirror, sim_fov, camera_intrinsics, mirror_swap,
    joint_space, joint_space_guidance,
    guidance_scale, guidance_safety_margin, guidance_activation_distance,
    guidance_grad_clip, guidance_method, guidance_sdf_agg,
    guidance_sdf_softmax_temp, guidance_sdf_topk,
    guidance_task_pos_weight, guidance_task_rot_weight,
    guidance_use_clean_sample, guidance_apply_last_step_only,
    ik_refine_last_step, cartesian_delta_mode, max_joint_speed,
    blocked_action_lead_time, blocked_settle_time,
    blocked_joint_vel_threshold, blocked_gripper_vel_threshold,
    joint_seed_source, blocked_action_start_idx):
    max_gripper_width = 0.09
    gripper_speed = 0.2
    
    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    
    # load left-right robot relative transform
    tx_left_right = np.array(robot_config_data['tx_left_right'])
    tx_robot1_robot0 = tx_left_right
    
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    policy_target = str(getattr(cfg.policy, '_target_', ''))
    joint_space_guidance = joint_space_guidance or ('joint_space_with_guidance' in policy_target)
    joint_space = joint_space or (
        ('joint_space' in policy_target) and ('joint_space_with_guidance' not in policy_target)
    )
    if joint_space and joint_space_guidance:
        raise ValueError("Please enable at most one of --joint_space and --joint_space_guidance.")

    if joint_space or joint_space_guidance:
        if len(robots_config) != 1:
            raise NotImplementedError(
                "Joint-space real-world evaluation currently supports a single robot only."
            )
        if str(robots_config[0]['robot_type']).lower().startswith('franka'):
            raise NotImplementedError(
                "Joint-space real-world evaluation is currently implemented for the UR RTDE controller path, not Franka."
            )
        robot_cfg_name = _get_robot_cfg_name(robots_config[0]['robot_type'])
        if joint_space_guidance:
            cfg.policy._target_ = (
                'diffusion_policy.policy.diffusion_unet_timm_policy_joint_space_with_guidance.'
                'DiffusionUnetTimmPolicyJointSpaceWithGuidance'
            )
        else:
            cfg.policy._target_ = (
                'diffusion_policy.policy.diffusion_unet_timm_policy_joint_space.'
                'DiffusionUnetTimmPolicyJointSpace'
            )
        with open_dict(cfg.policy):
            # cfg.policy.robot_uid = str(robots_config[0]['robot_type'])
            cfg.policy.robot_cfg_name = robot_cfg_name
            cfg.policy.ik_refine_last_step = ik_refine_last_step
            cfg.policy.cartesian_delta_mode = cartesian_delta_mode
            if joint_space_guidance:
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
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)
    print("obs_keys:", list(cfg.task.shape_meta.obs.keys()))

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm, \
            KeystrokeCounter() as key_counter, \
            BimanualUmiEnv(
                output_dir=output,
                robots_config=robots_config,
                grippers_config=grippers_config,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                # latency
                camera_obs_latency=0.165,
                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_swap=mirror_swap,
                # action
                max_pos_speed=2.0,
                max_rot_speed=6.0,
                max_joint_speed=max_joint_speed,
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            # load match_dataset
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break

                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

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
            obs = env.get_obs()
            episode_start_pose = list()
            for robot_id in range(len(robots_config)):
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=episode_start_pose)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                current_joint_angles = None
                if joint_space or joint_space_guidance:
                    current_joint_angles_np, _ = _get_policy_current_joint_angles_from_env(
                        env=env,
                        obs=obs,
                        robot_cfg_name=robot_cfg_name,
                        source=joint_seed_source,
                    )
                    current_joint_angles = torch.from_numpy(
                        current_joint_angles_np
                    ).unsqueeze(0).to(device)
                result = policy.predict_action(
                    obs_dict,
                    chunk_start_pose=torch.from_numpy(
                        _get_current_action_base_pose(obs, robot_id=0)
                    ).unsqueeze(0).to(device) if (joint_space or joint_space_guidance) else None,
                    current_joint_angles=current_joint_angles,
                )
                if joint_space or joint_space_guidance:
                    action = result['joint_action_pred'][0].detach().to('cpu').numpy()
                    assert action.shape[-1] == current_joint_angles.shape[-1] - 5 # arm_dof + 6 (robotiq finger joints) -> arm_dof + gripper_width
                else:
                    action = result['action_pred'][0].detach().to('cpu').numpy()
                    assert action.shape[-1] == 10 * len(robots_config)
                    action = get_real_umi_action(action, obs, action_pose_repr)
                    assert action.shape[-1] == 7 * len(robots_config)
                del result

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                robot_states = env.get_robot_state()
                target_pose = np.stack([rs['TargetTCPPose'] for rs in robot_states])

                gripper_states = env.get_gripper_state()
                gripper_target_pos = np.asarray([gs['gripper_position'] for gs in gripper_states])
                
                control_robot_idx_list = [0]

                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera{match_camera}_rgb'][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = (vis_img + match_img) / 2
                    obs_left_img = obs['camera0_rgb'][-1]
                    obs_right_img = obs['camera0_rgb'][-1]
                    vis_img = np.concatenate([obs_left_img, obs_right_img, vis_img], axis=1)
                    
                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0,0,0)
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    _ = cv2.pollKey()
                    press_events = key_counter.get_press_events()
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            # Exit human control loop
                            # hand control over to the policy
                            start_policy = True
                        elif key_stroke == KeyCode(char='e'):
                            # Next episode
                            if match_episode is not None:
                                match_episode = min(match_episode + 1, env.replay_buffer.n_episodes-1)
                        elif key_stroke == KeyCode(char='w'):
                            # Prev episode
                            if match_episode is not None:
                                match_episode = max(match_episode - 1, 0)
                        elif key_stroke == KeyCode(char='m'):
                            # move the robot
                            duration = 3.0
                            ep = match_replay_buffer.get_episode(match_episode_id)

                            for robot_idx in range(1):
                                pos = ep[f'robot{robot_idx}_eef_pos'][0]
                                rot = ep[f'robot{robot_idx}_eef_rot_axis_angle'][0]
                                grip = ep[f'robot{robot_idx}_gripper_width'][0]
                                pose = np.concatenate([pos, rot])
                                env.robots[robot_idx].servoL(pose, duration=duration)
                                env.grippers[robot_idx].schedule_waypoint(grip, target_time=time.time() + duration)
                                target_pose[robot_idx] = pose
                                gripper_target_pos[robot_idx] = grip
                            time.sleep(duration)

                        elif key_stroke == Key.backspace:
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                        elif key_stroke == KeyCode(char='a'):
                            control_robot_idx_list = list(range(target_pose.shape[0]))
                        elif key_stroke == KeyCode(char='1'):
                            control_robot_idx_list = [0]
                        elif key_stroke == KeyCode(char='2'):
                            control_robot_idx_list = [1]

                    if start_policy:
                        break

                    precise_wait(t_sample)
                    # get teleop command
                    sm_state = sm.get_motion_state_transformed()
                    # print(sm_state)
                    dpos = sm_state[:3] * (0.1 / frequency)
                    drot_xyz = sm_state[3:] * (0.3 / frequency)

                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    for robot_idx in control_robot_idx_list:
                        target_pose[robot_idx, :3] += dpos
                        target_pose[robot_idx, 3:] = (drot * st.Rotation.from_rotvec(
                            target_pose[robot_idx, 3:])).as_rotvec()

                    dpos = 0
                    if sm.is_button_pressed(0):
                        # close gripper
                        dpos = -gripper_speed / frequency
                    if sm.is_button_pressed(1):
                        dpos = gripper_speed / frequency
                    for robot_idx in control_robot_idx_list:
                        gripper_target_pos[robot_idx] = np.clip(gripper_target_pos[robot_idx] + dpos, 0, max_gripper_width)

                    # solve collision with table
                    for robot_idx in control_robot_idx_list:
                        solve_table_collision(
                            ee_pose=target_pose[robot_idx],
                            gripper_width=gripper_target_pos[robot_idx],
                            height_threshold=robots_config[robot_idx]['height_threshold'])
                    
                    # solve collison between two robots
                    solve_sphere_collision(
                        ee_poses=target_pose,
                        robots_config=robots_config
                    )

                    action = np.zeros((7 * target_pose.shape[0],))

                    for robot_idx in range(target_pose.shape[0]):
                        action[7 * robot_idx + 0: 7 * robot_idx + 6] = target_pose[robot_idx]
                        action[7 * robot_idx + 6] = gripper_target_pos[robot_idx]


                    # execute teleop command
                    env.exec_actions(
                        actions=[action], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        compensate_latency=False)
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # get current pose
                    obs = env.get_obs()
                    episode_start_pose = list()
                    for robot_id in range(len(robots_config)):
                        pose = np.concatenate([
                            obs[f'robot{robot_id}_eef_pos'],
                            obs[f'robot{robot_id}_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        episode_start_pose.append(pose)

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None
                    prev_joint_action_end = None
                    while True:
                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep,
                                tx_robot1_robot0=tx_robot1_robot0,
                                episode_start_pose=episode_start_pose)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            current_joint_angles = None
                            joint_seed_debug = None
                            if joint_space or joint_space_guidance:
                                action_base_pose = _get_current_action_base_pose(obs, robot_id=0)
                                current_joint_angles_np, joint_seed_debug = _get_policy_current_joint_angles_from_env(
                                    env=env,
                                    obs=obs,
                                    robot_cfg_name=robot_cfg_name,
                                    source=joint_seed_source,
                                )
                                current_joint_angles = torch.from_numpy(
                                    current_joint_angles_np
                                ).unsqueeze(0).to(device)
                            result = policy.predict_action(
                                obs_dict,
                                chunk_start_pose=torch.from_numpy(
                                    action_base_pose
                                ).unsqueeze(0).to(device) if (joint_space or joint_space_guidance) else None,
                                current_joint_angles=current_joint_angles,
                            )
                            if joint_space or joint_space_guidance:
                                action = result['joint_action_pred'][0].detach().to('cpu').numpy()
                                action_base_delta = action_base_pose - np.asarray(episode_start_pose[0], dtype=np.float32)
                                prev_end_msg = "prev_chunk=none"
                                if prev_joint_action_end is not None:
                                    first_delta = action[0, :6] - prev_joint_action_end
                                    last_delta = action[-1, :6] - prev_joint_action_end
                                    prev_end_msg = (
                                        f"new_first_vs_prev_end_norm={np.linalg.norm(first_delta):.6f}rad, "
                                        f"new_first_vs_prev_end_max={np.max(np.abs(first_delta)):.6f}rad, "
                                        f"new_last_vs_prev_end_norm={np.linalg.norm(last_delta):.6f}rad"
                                    )
                                print(
                                    "[JOINT OBS DEBUG] "
                                    f"seed_source={joint_seed_source}, "
                                    f"aligned_vs_actual_norm={np.linalg.norm(joint_seed_debug['aligned_joint_pos'] - joint_seed_debug['latest_joint_pos']):.6f}rad, "
                                    f"aligned_vs_actual_max={np.max(np.abs(joint_seed_debug['aligned_joint_pos'] - joint_seed_debug['latest_joint_pos'])):.6f}rad, "
                                    f"target_vs_actual_norm={np.linalg.norm(joint_seed_debug['target_joint_pos'] - joint_seed_debug['latest_joint_pos']):.6f}rad, "
                                    f"action_base_vs_episode_start_norm={np.linalg.norm(action_base_delta):.6f}, "
                                    f"new_first_vs_seed_norm={np.linalg.norm(action[0, :6] - current_joint_angles_np[:6]):.6f}rad, "
                                    f"new_last_vs_seed_norm={np.linalg.norm(action[-1, :6] - current_joint_angles_np[:6]):.6f}rad, "
                                    f"{prev_end_msg}"
                                )
                            else:
                                raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                                action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            print('Inference latency:', time.time() - s)
                        
                        # convert policy action to env actions
                        this_target_poses = action
                        action_mode = 'eef'
                        if blocked_action_start_idx > 0:
                            if blocked_action_start_idx >= len(this_target_poses):
                                raise ValueError(
                                    f"blocked_action_start_idx={blocked_action_start_idx} "
                                    f"must be smaller than chunk length={len(this_target_poses)}."
                                )
                            this_target_poses = this_target_poses[blocked_action_start_idx:]
                            print(
                                f"Blocking debug dropped leading {blocked_action_start_idx} "
                                f"actions; executing {len(this_target_poses)} actions."
                            )
                        if joint_space or joint_space_guidance:
                            action_mode = 'joint'
                            if prev_joint_action_end is not None:
                                exec_first_delta = (
                                    this_target_poses[0, :6] - prev_joint_action_end
                                )
                                print(
                                    "[JOINT EXEC DEBUG] "
                                    f"exec_first_idx={blocked_action_start_idx}, "
                                    f"exec_first_vs_prev_end_norm={np.linalg.norm(exec_first_delta):.6f}rad, "
                                    f"exec_first_vs_prev_end_max={np.max(np.abs(exec_first_delta)):.6f}rad"
                                )
                        else:
                            assert this_target_poses.shape[1] == len(robots_config) * 7
                            for target_pose in this_target_poses:
                                for robot_idx in range(len(robots_config)):
                                    solve_table_collision(
                                        ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                                        gripper_width=target_pose[robot_idx * 7 + 6],
                                        height_threshold=robots_config[robot_idx]['height_threshold']
                                    )
                                
                                # solve collison between two robots
                                solve_sphere_collision(
                                    ee_poses=target_pose.reshape([len(robots_config), -1]),
                                    robots_config=robots_config
                                )

                        # In this blocked debug variant, do not align execution
                        # timestamps to the stale observation time. Re-time the
                        # whole chunk from "now", execute it, wait until it is
                        # finished and settled, then take the next observation.
                        max_robot_action_latency = max(
                            rc['robot_action_latency'] for rc in robots_config
                        )
                        max_gripper_action_latency = max(
                            gc['gripper_action_latency'] for gc in grippers_config
                        )
                        max_action_latency = max(
                            max_robot_action_latency,
                            max_gripper_action_latency
                        )
                        action_start_time = (
                            time.time()
                            + max_action_latency
                            + blocked_action_lead_time
                        )
                        action_timestamps = (
                            np.arange(len(this_target_poses), dtype=np.float64) * dt
                            + action_start_time
                        )
                        nominal_chunk_end_time = action_timestamps[-1]

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True,
                            action_mode=action_mode
                        )
                        if joint_space or joint_space_guidance:
                            prev_joint_action_end = np.asarray(this_target_poses[-1, :6]).copy()
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        obs_left_img = obs['camera0_rgb'][-1]
                        obs_right_img = obs['camera0_rgb'][-1]
                        vis_img = np.concatenate([obs_left_img, obs_right_img], axis=1)
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])

                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode
                                # Hand control back to human
                                print('Stopped.')
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            break

                        # Wait for execution to finish and settle before the
                        # next observation/inference cycle. This is velocity-
                        # based because the joint controller can internally
                        # extend timing when max_joint_speed is active.
                        static_since = None
                        while True:
                            _ = cv2.pollKey()
                            press_events = key_counter.get_press_events()
                            for key_stroke in press_events:
                                if key_stroke == KeyCode(char='s'):
                                    print('Stopped.')
                                    stop_episode = True
                                elif key_stroke == KeyCode(char='q'):
                                    env.end_episode()
                                    exit(0)
                            if stop_episode:
                                break

                            robot_states = env.get_robot_state()
                            max_joint_vel = max(
                                np.max(np.abs(np.asarray(rs['ActualQd'])))
                                for rs in robot_states
                            )
                            gripper_states = env.get_gripper_state()
                            max_gripper_vel = max(
                                abs(float(gs.get('gripper_velocity', 0.0)))
                                for gs in gripper_states
                            )
                            now = time.time()
                            is_static = (
                                (now >= nominal_chunk_end_time)
                                and (max_joint_vel <= blocked_joint_vel_threshold)
                                and (max_gripper_vel <= blocked_gripper_vel_threshold)
                            )
                            if is_static:
                                if static_since is None:
                                    static_since = now
                                elif now - static_since >= blocked_settle_time:
                                    print(
                                        "Blocking chunk settled: "
                                        f"max_joint_vel={max_joint_vel:.6f}rad/s, "
                                        f"max_gripper_vel={max_gripper_vel:.6f}m/s"
                                    )
                                    break
                            else:
                                static_since = None
                            precise_wait(now + 0.05, time_func=time.time)
                        if stop_episode:
                            env.end_episode()
                            break

                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
