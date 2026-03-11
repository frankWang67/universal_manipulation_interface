from typing import Dict, Optional, Tuple

import torch
import pytorch_kinematics as pk

from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.math import Pose
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy

from mani_skill.utils.geometry.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d as matrix_to_rot6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix as rot6d_to_matrix,
)

def pose9d_to_mat(pose9d: torch.Tensor) -> torch.Tensor:
    pos = pose9d[..., :3]
    rot6d = pose9d[..., 3:]
    rot = rot6d_to_matrix(rot6d)
    out = torch.zeros((*pose9d.shape[:-1], 4, 4), device=pose9d.device, dtype=pose9d.dtype)
    out[..., :3, :3] = rot
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1.0
    return out


class DiffusionUnetTimmPolicyJointSpace(DiffusionUnetTimmPolicy):
    """
    Joint-space inference wrapper for a Cartesian-space trained diffusion model.

    The denoising model remains unchanged (predicts Cartesian noise in normalized
    action space). Inference state is lifted into joint space by:
      1) Cartesian noisy sample -> (unnorm + rel->abs) -> IK to initialize q_t
      2) q_t -> FK -> relative Cartesian -> normalize -> model epsilon prediction
      3) epsilon_cart(norm) -> epsilon_cart(physical) -> 6D twist
      4) 6D twist -> joint noise via damped Jacobian pseudo-inverse
      5) DDPM update executed in joint space.
    """

    def __init__(
        self,
        *args,
        robot_cfg_name: str = "panda_robotiq_wristcam.yml",
        robot_urdf_path: str = "/home/wshf/curobo/src/curobo/content/assets/robot/robotiq_gripper_robots/panda/panda_robotiq_wristcam.urdf",
        ee_link_name: str = "eef",
        arm_dof: int = 7,
        ik_num_seeds: int = 20,
        jacobian_damping: float = 0.01,
        ik_refine_each_step: bool = False,
        ik_position_threshold: float = 5e-4,
        ik_rotation_threshold: float = 5e-3,
        init_noise_scale: float = 0.3,
        max_dq_per_step: float = 0.5,
        ik_refine_last_step: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.robot_cfg_name = robot_cfg_name
        self.robot_urdf_path = robot_urdf_path
        self.ee_link_name = ee_link_name
        self.arm_dof = int(arm_dof)
        self.ik_num_seeds = int(ik_num_seeds)
        self.jacobian_damping = float(jacobian_damping)
        self.ik_refine_each_step = bool(ik_refine_each_step)
        self.ik_position_threshold = float(ik_position_threshold)
        self.ik_rotation_threshold = float(ik_rotation_threshold)
        self.init_noise_scale = float(init_noise_scale)
        self.max_dq_per_step = float(max_dq_per_step)
        self.ik_refine_last_step = bool(ik_refine_last_step)

        self._ik_solver = None
        self._kin_model = None
        self._pk_chain = None
        self._robot_dof = None
        self._last_joint_traj = None

    # ===========================
    # Kinematics initialization
    # ===========================
    def _ensure_kinematics(self, device: torch.device):
        if self._ik_solver is not None:
            return

        tensor_args = TensorDeviceType(device=device)
        robot_cfg_dict = load_yaml(join_path(get_robot_configs_path(), self.robot_cfg_name))["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(robot_cfg_dict, tensor_args)

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=self.ik_rotation_threshold,
            position_threshold=self.ik_position_threshold,
            num_seeds=self.ik_num_seeds,
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=tensor_args,
            use_cuda_graph=False,
        )
        self._ik_solver = IKSolver(ik_config)
        self._kin_model = self._ik_solver.kinematics
        self._robot_dof = int(self._kin_model.get_dof())

        with open(self.robot_urdf_path, "r") as f:
            urdf_str = f.read()
        self._pk_chain = pk.build_serial_chain_from_urdf(urdf_str, self.ee_link_name)
        self._pk_chain = self._pk_chain.to(dtype=torch.float32, device=device)

    @staticmethod
    def _inv_se3(mat: torch.Tensor) -> torch.Tensor:
        rot = mat[..., :3, :3]
        pos = mat[..., :3, 3]
        rot_t = rot.transpose(-2, -1)
        out = torch.zeros_like(mat)
        out[..., :3, :3] = rot_t
        out[..., :3, 3] = -(rot_t @ pos.unsqueeze(-1)).squeeze(-1)
        out[..., 3, 3] = 1.0
        return out

    # ===========================
    # Pose conversions
    # ===========================
    def _relative_pose9_to_absolute(
        self,
        rel_pose9: torch.Tensor,
        episode_start_pose: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        rel_pose9: (B,T,9), relative to episode start EE frame.
        episode_start_pose: (B,6), [xyz, axis-angle].
        returns abs_pos (B,T,3), abs_rot (B,T,3,3)
        """
        base_pos = episode_start_pose[:, :3]
        base_rot = axis_angle_to_matrix(episode_start_pose[:, 3:6])

        rel_pos = rel_pose9[..., :3]
        rel_rot = rot6d_to_matrix(rel_pose9[..., 3:])

        abs_pos = (base_rot.unsqueeze(1) @ rel_pos.unsqueeze(-1)).squeeze(-1) + base_pos.unsqueeze(1)
        abs_rot = base_rot.unsqueeze(1) @ rel_rot
        return abs_pos, abs_rot

    def _absolute_pose_to_relative9(
        self,
        abs_pos: torch.Tensor,
        abs_rot: torch.Tensor,
        episode_start_pose: torch.Tensor,
    ) -> torch.Tensor:
        base_pos = episode_start_pose[:, :3]
        base_rot = axis_angle_to_matrix(episode_start_pose[:, 3:6])
        base_rot_t = base_rot.transpose(-2, -1)

        rel_pos = (base_rot_t.unsqueeze(1) @ (abs_pos - base_pos.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rel_rot = base_rot_t.unsqueeze(1) @ abs_rot
        rel_rot6d = matrix_to_rot6d(rel_rot.reshape(-1, 3, 3)).reshape(*rel_rot.shape[:2], 6)
        return torch.cat([rel_pos, rel_rot6d], dim=-1)

    # ===========================
    # FK / IK / Jacobian helpers
    # ===========================
    def _solve_start_joint_from_pose(self, episode_start_pose: torch.Tensor) -> torch.Tensor:
        B = episode_start_pose.shape[0]
        start_pos = episode_start_pose[:, :3]
        start_rot = axis_angle_to_matrix(episode_start_pose[:, 3:6])
        start_quat = matrix_to_quaternion(start_rot)

        seed = torch.zeros((B, self._robot_dof), device=episode_start_pose.device, dtype=episode_start_pose.dtype)
        goal_pose = Pose(position=start_pos, quaternion=start_quat)
        with torch.enable_grad():
            ik_result = self._ik_solver.solve_batch(goal_pose, retract_config=seed)
        q = ik_result.solution.squeeze(1)
        if hasattr(ik_result, "success"):
            succ = ik_result.success
            while succ.ndim > 1:
                succ = succ.squeeze(-1)
            q = torch.where(succ.unsqueeze(-1), q, seed)
        return q

    def _ik_from_absolute(
        self,
        abs_pos: torch.Tensor,
        abs_rot: torch.Tensor,
        seed_q: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = abs_pos.shape
        pos_flat = abs_pos.reshape(-1, 3)
        quat_flat = matrix_to_quaternion(abs_rot.reshape(-1, 3, 3))
        seed_flat = seed_q.reshape(-1, self._robot_dof).contiguous()

        goal_pose = Pose(position=pos_flat, quaternion=quat_flat)
        with torch.enable_grad():
            ik_result = self._ik_solver.solve_batch(goal_pose, retract_config=seed_flat)
        q_flat = ik_result.solution.squeeze(1)

        if hasattr(ik_result, "success"):
            succ = ik_result.success
            while succ.ndim > 1:
                succ = succ.squeeze(-1)
            q_flat = torch.where(succ.unsqueeze(-1), q_flat, seed_flat)

        return q_flat.reshape(B, T, self._robot_dof)

    def _fk_to_absolute(self, q_arm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = q_arm.shape
        q_flat = q_arm.reshape(-1, q_arm.shape[-1]).contiguous()
        kin_state = self._kin_model.get_state(q_flat)

        if hasattr(kin_state, "ee_position"):
            pos_flat = kin_state.ee_position
        else:
            pos_flat = kin_state.ee_pose.position

        if hasattr(kin_state, "ee_quaternion"):
            quat_flat = kin_state.ee_quaternion
        else:
            quat_flat = kin_state.ee_pose.quaternion

        rot_flat = quaternion_to_matrix(quat_flat)
        pos = pos_flat.reshape(B, T, 3)
        rot = rot_flat.reshape(B, T, 3, 3)
        return pos, rot

    def _jacobian(self, q_arm: torch.Tensor) -> torch.Tensor:
        qf = q_arm.reshape(-1, q_arm.shape[-1])
        n = qf.shape[0]

        arm_dof = min(self.arm_dof, qf.shape[-1])
        q_for_jac = qf[:, :arm_dof].detach().to(dtype=torch.float32)
        j_arm = self._pk_chain.jacobian(q_for_jac).to(device=qf.device, dtype=qf.dtype)

        if qf.shape[-1] > arm_dof:
            pad = torch.zeros((n, 6, qf.shape[-1] - arm_dof), device=qf.device, dtype=qf.dtype)
            j = torch.cat([j_arm, pad], dim=-1)
        else:
            j = j_arm
        return j

    def _dls_pinv_map(self, twist: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """
        twist: (N,6), jacobian: (N,6,D) -> dq: (N,D)
        Damped least-squares pseudo-inverse: dq = J^T (J J^T + λ I)^{-1} twist
        Uses torch.linalg.solve instead of explicit inverse for stability.
        """
        j_t = jacobian.transpose(-2, -1)
        jjt = jacobian @ j_t
        eye = torch.eye(6, device=jacobian.device, dtype=jacobian.dtype).unsqueeze(0)
        A = jjt + self.jacobian_damping * eye
        y = torch.linalg.solve(A, twist.unsqueeze(-1))  # (N,6,1)
        dq = (j_t @ y).squeeze(-1)  # (N,D)
        return dq

    def _absolute_pose_delta_to_twist6(
        self,
        abs_pose9_curr: torch.Tensor,
        abs_pose9_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute spatial/world-frame 6D twist (vx, vy, vz, wx, wy, wz)
        from current absolute pose to target absolute pose.
        """
        b, t, _ = abs_pose9_curr.shape
        cur_flat = abs_pose9_curr.reshape(-1, 9)
        tgt_flat = abs_pose9_tgt.reshape(-1, 9)

        t_cur = pose9d_to_mat(cur_flat)
        t_tgt = pose9d_to_mat(tgt_flat)
        t_delta = t_tgt @ self._inv_se3(t_cur)

        dpos = t_delta[:, :3, 3]
        drot = matrix_to_axis_angle(t_delta[:, :3, :3])
        return torch.cat([dpos, drot], dim=-1).reshape(b, t, 6)

    # ===========================
    # Inference
    # ===========================
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        episode_start_pose: Optional[torch.Tensor] = None,
        obstacle_info=[],
        current_joint_angles: Optional[torch.Tensor] = None,
        return_debug: bool = False,
        **kwargs,
    ):
        if episode_start_pose is None:
            raise ValueError("episode_start_pose is required for joint-space inference.")

        self._ensure_kinematics(condition_data.device)

        model = self.model
        scheduler = self.noise_scheduler
        bsz = condition_data.shape[0]
        horizon = condition_data.shape[1]

        # ── 1) Obtain starting joint configuration ──────────────────────
        if current_joint_angles is None:
            q_start = self._solve_start_joint_from_pose(episode_start_pose)
        else:
            q_start = current_joint_angles[..., : self._robot_dof].to(
                device=condition_data.device, dtype=condition_data.dtype
            )
        q_seed = q_start.unsqueeze(1).expand(bsz, horizon, self._robot_dof)

        # ── 2) Initialize noisy joint trajectory ────────────────────────
        # Generate normalized Cartesian noise (same as vanilla DDPM).
        trajectory_cart_n = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        trajectory_cart_n[condition_mask] = condition_data[condition_mask]

        trajectory_cart = self.normalizer["action"].unnormalize(trajectory_cart_n)
        grip = trajectory_cart[..., 9:10]

        # Direct joint noise: small σ keeps joints in the well-
        # conditioned neighborhood of q_seed where the Jacobian
        # linearization is accurate.
        q_arm = q_seed + torch.randn(
            q_seed.shape, device=q_seed.device, dtype=q_seed.dtype,
        ) * self.init_noise_scale
        q_traj = torch.cat([q_arm, grip], dim=-1)

        # ── 3) Precompute conditioned joint targets (for inpainting) ────
        cond_step_mask = condition_mask.any(dim=-1)
        q_cond = None
        if torch.any(cond_step_mask):
            cond_cart = self.normalizer["action"].unnormalize(condition_data)
            cond_abs_pos, cond_abs_rot = self._relative_pose9_to_absolute(
                cond_cart[..., :9], episode_start_pose
            )
            q_cond_arm = self._ik_from_absolute(cond_abs_pos, cond_abs_rot, q_seed)
            q_cond = torch.cat([q_cond_arm, cond_cart[..., 9:10]], dim=-1)
            q_traj[cond_step_mask] = q_cond[cond_step_mask]

        # ── 4) Denoising in joint space ─────────────────────────────────
        scheduler.set_timesteps(self.num_inference_steps)

        debug = {"step_cart_l2": []}
        timesteps = list(scheduler.timesteps)
        n_steps = len(timesteps)

        for idx, t in enumerate(timesteps):
            if q_cond is not None:
                q_traj[cond_step_mask] = q_cond[cond_step_mask]

            # ── FK once → build both absolute and relative Cartesian ────
            q_arm_curr = q_traj[..., : self._robot_dof]
            grip_curr = q_traj[..., self._robot_dof : self._robot_dof + 1]
            abs_pos_curr, abs_rot_curr = self._fk_to_absolute(q_arm_curr)

            rel_pose9_curr = self._absolute_pose_to_relative9(
                abs_pos_curr, abs_rot_curr, episode_start_pose
            )
            cart_phys_curr = torch.cat([rel_pose9_curr, grip_curr], dim=-1)
            cart_n_curr = self.normalizer["action"].normalize(cart_phys_curr)
            cart_n_curr[condition_mask] = condition_data[condition_mask]

            # ── Model predicts epsilon in normalized Cartesian space ────
            eps_cart_n = model(
                cart_n_curr, t, local_cond=local_cond, global_cond=global_cond
            )

            # ── Scheduler step in normalized Cartesian space ────────────
            cart_n_prev_tgt = scheduler.step(
                eps_cart_n, t, cart_n_curr, generator=generator, **kwargs,
            ).prev_sample
            cart_n_prev_tgt[condition_mask] = condition_data[condition_mask]
            cart_phys_prev_tgt = self.normalizer["action"].unnormalize(
                cart_n_prev_tgt
            )

            # ── Target absolute pose ────────────────────────────────────
            abs_pos_tgt, abs_rot_tgt = self._relative_pose9_to_absolute(
                cart_phys_prev_tgt[..., :9], episode_start_pose,
            )

            # ── World-frame twist (current → target) ────────────────────
            abs_rot6d_curr = matrix_to_rot6d(
                abs_rot_curr.reshape(-1, 3, 3)
            ).reshape(bsz, horizon, 6)
            abs_pose9_curr_9d = torch.cat(
                [abs_pos_curr, abs_rot6d_curr], dim=-1
            )

            abs_rot6d_tgt = matrix_to_rot6d(
                abs_rot_tgt.reshape(-1, 3, 3)
            ).reshape(bsz, horizon, 6)
            abs_pose9_tgt = torch.cat(
                [abs_pos_tgt, abs_rot6d_tgt], dim=-1
            )

            twist6 = self._absolute_pose_delta_to_twist6(
                abs_pose9_curr_9d, abs_pose9_tgt
            )

            # ── Clamped Jacobian step + reuse sub-iterations ─────────────
            jac = self._jacobian(q_arm_curr)
            dq = self._dls_pinv_map(
                twist6.reshape(-1, 6), jac,
            ).reshape(bsz, horizon, self._robot_dof)
            if self.max_dq_per_step > 0:
                dq = torch.clamp(
                    dq, -self.max_dq_per_step, self.max_dq_per_step
                )
            q_arm_new = q_arm_curr + dq

            # Reuse sub-iterations: same Jacobian, refresh FK + twist
            for _ in range(2):
                abs_pos_ri, abs_rot_ri = self._fk_to_absolute(q_arm_new)
                abs_rot6d_ri = matrix_to_rot6d(
                    abs_rot_ri.reshape(-1, 3, 3)
                ).reshape(bsz, horizon, 6)
                abs_pose9_ri = torch.cat(
                    [abs_pos_ri, abs_rot6d_ri], dim=-1
                )
                twist_ri = self._absolute_pose_delta_to_twist6(
                    abs_pose9_ri, abs_pose9_tgt
                )
                dq_ri = self._dls_pinv_map(
                    twist_ri.reshape(-1, 6), jac,
                ).reshape(bsz, horizon, self._robot_dof)
                if self.max_dq_per_step > 0:
                    dq_ri = torch.clamp(
                        dq_ri, -self.max_dq_per_step, self.max_dq_per_step
                    )
                q_arm_new = q_arm_new + dq_ri

            # ── Optional cuRobo IK refine ───────────────────────────────
            is_last = (idx == n_steps - 1)
            if self.ik_refine_each_step or (is_last and self.ik_refine_last_step):
                q_arm_new = self._ik_from_absolute(
                    abs_pos_tgt, abs_rot_tgt, q_arm_new
                )

            # ── Joint update; gripper follows Cartesian target ──────────
            q_traj = torch.cat([
                q_arm_new,
                cart_phys_prev_tgt[..., 9:10],
            ], dim=-1)

            if return_debug:
                q_arm_dbg = q_traj[..., : self._robot_dof]
                abs_p, abs_r = self._fk_to_absolute(q_arm_dbg)
                rel9 = self._absolute_pose_to_relative9(
                    abs_p, abs_r, episode_start_pose
                )
                cart_phys_after = torch.cat(
                    [rel9, q_traj[..., self._robot_dof : self._robot_dof + 1]],
                    dim=-1,
                )
                debug["step_cart_l2"].append(
                    torch.linalg.norm(
                        (cart_phys_after - cart_phys_prev_tgt).reshape(bsz, -1),
                        dim=-1,
                    ).detach().cpu()
                )

        if q_cond is not None:
            q_traj[cond_step_mask] = q_cond[cond_step_mask]

        self._last_joint_traj = q_traj.detach()

        # ── Return normalized Cartesian action (keeps external API unchanged)
        q_arm_final = q_traj[..., : self._robot_dof]
        grip_final = q_traj[..., self._robot_dof : self._robot_dof + 1]
        abs_p_f, abs_r_f = self._fk_to_absolute(q_arm_final)
        rel9_f = self._absolute_pose_to_relative9(abs_p_f, abs_r_f, episode_start_pose)
        cart_final = torch.cat([rel9_f, grip_final], dim=-1)
        cart_final_n = self.normalizer["action"].normalize(cart_final)
        cart_final_n[condition_mask] = condition_data[condition_mask]
        if return_debug:
            return cart_final_n, q_traj, debug
        return cart_final_n

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        fixed_action_prefix: torch.Tensor = None,
        env_batched=False,
        episode_start_pose: torch.Tensor = None,
        obstacle_info=[],
        current_joint_angles: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict

        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        if env_batched:
            env_batch_size = next(iter(nobs.values())).shape[1]
            nobs = dict_apply(nobs, lambda x: x.reshape(B * env_batch_size, *x.shape[2:]))
        global_cond = self.obs_encoder(nobs)

        if env_batched:
            cond_data = torch.zeros(
                size=(B * env_batch_size, self.action_horizon, self.action_dim),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            cond_data = torch.zeros(
                size=(B, self.action_horizon, self.action_dim),
                device=self.device,
                dtype=self.dtype,
            )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer["action"].normalize(cond_data)

        if episode_start_pose is None:
            raise ValueError("episode_start_pose must be provided for joint-space policy inference.")

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            episode_start_pose=episode_start_pose,
            obstacle_info=obstacle_info,
            current_joint_angles=current_joint_angles,
            **self.kwargs,
        )

        if env_batched:
            assert nsample.shape == (B * env_batch_size, self.action_horizon, self.action_dim)
        else:
            assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer["action"].unnormalize(nsample)
        if env_batched:
            action_pred = action_pred.reshape(B, env_batch_size, self.action_horizon, self.action_dim)

        return {
            "action": action_pred,
            "action_pred": action_pred,
        }
