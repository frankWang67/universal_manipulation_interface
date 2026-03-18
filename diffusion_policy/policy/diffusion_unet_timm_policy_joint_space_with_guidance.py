from typing import Dict, Optional, Tuple, List, Any
import os
import time

import torch
import torch.nn.functional as F
import pytorch_kinematics as pk

from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.math import Pose
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionQueryBuffer, WorldCollisionConfig
from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.types import WorldConfig, Cuboid
from curobo.util_file import get_robot_configs_path, get_assets_path, join_path, load_yaml

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.guided_diffusion_util import (
    get_guidance_strength,
    flatten_obstacle_info,
    get_pred_x0,
)
from diffusion_policy.policy.diffusion_unet_timm_policy_joint_space import DiffusionUnetTimmPolicyJointSpace

from mani_skill.utils.geometry.rotation_conversions import matrix_to_rotation_6d as matrix_to_rot6d


class DiffusionUnetTimmPolicyJointSpaceWithGuidance(DiffusionUnetTimmPolicyJointSpace):
    """
    Joint-space diffusion with whole-body collision guidance.

    Guidance is applied in joint space after each reverse DDPM step:
        q_{t-1}^{guided} = q_{t-1} - gamma_t * dL_col/dq

    where L_col is computed from cuRobo ESDF queried at robot body collision spheres.
    """

    def __init__(
        self,
        *args,
        robot_uid: Optional[str] = None,
        robot_cfg_name: Optional[str] = None,
        robot_urdf_path: Optional[str] = None,
        ee_link_name: Optional[str] = None,
        arm_dof: int = -1,
        guidance_scale: float = 0.01,
        guidance_safety_margin: float = 0.02,
        guidance_activation_distance: float = 0.02,
        guidance_grad_clip: float = 1.0,
        guidance_loss_power: float = 2.0,
        guidance_use_schedule: bool = True,
        guidance_apply_last_step_only: bool = False,
        guidance_steps_per_denoise: int = 1,
        guidance_use_clean_sample: bool = True,
        **kwargs,
    ):
        if robot_cfg_name is None:
            robot_cfg_name = self._infer_robot_cfg_name(robot_uid)

        super().__init__(
            *args,
            robot_cfg_name=robot_cfg_name,
            robot_urdf_path=robot_urdf_path,
            ee_link_name=ee_link_name if ee_link_name is not None else "eef",
            arm_dof=arm_dof,
            **kwargs,
        )

        self.robot_uid = robot_uid
        self.guidance_scale = float(guidance_scale)
        self.guidance_safety_margin = float(guidance_safety_margin)
        self.guidance_activation_distance = float(guidance_activation_distance)
        self.guidance_grad_clip = float(guidance_grad_clip)
        self.guidance_loss_power = float(guidance_loss_power)
        self.guidance_use_schedule = bool(guidance_use_schedule)
        self.guidance_apply_last_step_only = bool(guidance_apply_last_step_only)
        self.guidance_steps_per_denoise = int(max(guidance_steps_per_denoise, 1))
        self.guidance_use_clean_sample = bool(guidance_use_clean_sample)

        self._world_collision = None
        self._coll_query_buffer = None
        self._coll_weight = None
        self._coll_activation_distance = None
        self._cached_world_key = None

    @staticmethod
    def _infer_robot_cfg_name(robot_uid: Optional[str]) -> str:
        if robot_uid is None:
            return "panda_robotiq_wristcam.yml"

        mapping = {
            "panda_robotiq_wristcam": "panda_robotiq_wristcam.yml",
            "ur5_robotiq_wristcam": "ur5_robotiq_wristcam.yml",
            "xarm6_robotiq_wristcam": "xarm6_robotiq_wristcam.yml",
            "xarm7_robotiq_wristcam": "xarm7_robotiq_wristcam.yml",
            "floating_robotiq_2f_85_gripper_wristcam": "floating_robotiq_wristcam.yml",
            "floating_robotiq_wristcam": "floating_robotiq_wristcam.yml",
        }
        if robot_uid in mapping:
            return mapping[robot_uid]

        candidate = f"{robot_uid}.yml"
        cfg_path = os.path.join(get_robot_configs_path(), candidate)
        if os.path.exists(cfg_path):
            return candidate
        raise ValueError(f"Cannot infer cuRobo robot config for robot_uid={robot_uid}")

    def _ensure_kinematics(self, device: torch.device):
        if self._ik_solver is not None:
            return

        self._tensor_args = TensorDeviceType(device=device)
        robot_cfg_dict = load_yaml(join_path(get_robot_configs_path(), self.robot_cfg_name))["robot_cfg"]

        kin_cfg = robot_cfg_dict.get("kinematics", {})
        if self.ee_link_name is None:
            self.ee_link_name = kin_cfg.get("ee_link", "eef")

        if self.robot_urdf_path is None:
            urdf_rel = kin_cfg.get("urdf_path")
            if urdf_rel is None:
                raise ValueError(f"urdf_path missing in robot config {self.robot_cfg_name}")
            self.robot_urdf_path = join_path(get_assets_path(), urdf_rel)

        robot_cfg = RobotConfig.from_dict(robot_cfg_dict, self._tensor_args)

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=self.ik_rotation_threshold,
            position_threshold=self.ik_position_threshold,
            num_seeds=self.ik_num_seeds,
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=self._tensor_args,
            use_cuda_graph=False,
        )
        self._ik_solver = IKSolver(ik_config)
        self._kin_model = self._ik_solver.kinematics
        self._robot_dof = int(self._kin_model.get_dof())

        with open(self.robot_urdf_path, "r") as f:
            urdf_str = f.read()
        try:
            self._pk_chain = pk.build_serial_chain_from_urdf(urdf_str, self.ee_link_name)
        except ValueError:
            self._pk_chain = pk.build_serial_chain_from_urdf(
                urdf_str.encode("utf-8"), self.ee_link_name
            )
        self._pk_chain = self._pk_chain.to(dtype=torch.float32, device=device)

        if self.arm_dof <= 0:
            self.arm_dof = int(self._pk_chain.n_joints)

    def _build_world_collision(
        self,
        obstacle_info: Any,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if obstacle_info is None:
            obstacles = []
        else:
            obstacles = flatten_obstacle_info(obstacle_info)
        if len(obstacles) == 0:
            self._world_collision = None
            self._coll_query_buffer = None
            self._cached_world_key = None
            return

        # Build world from the first environment's obstacles;
        # query remains batched over trajectory spheres.
        world_cuboids = []
        world_key_items = []
        for i, obs in enumerate(obstacles):
            center = obs["center"]
            quat = obs["quat"]
            extent = obs["extent"]

            if center.ndim == 2:
                c = center[0]
            else:
                c = center
            if quat.ndim == 2:
                q = quat[0]
            else:
                q = quat
            if extent.ndim == 2:
                e = extent[0]
            else:
                e = extent

            c = c.to(device=device, dtype=dtype)
            q = q.to(device=device, dtype=dtype)
            e = e.to(device=device, dtype=dtype)
            world_key_items.append(torch.cat([c, q, e], dim=0))

            world_cuboids.append(
                Cuboid(
                    name=f"obs_{i}",
                    pose=[
                        float(c[0].item()),
                        float(c[1].item()),
                        float(c[2].item()),
                        float(q[0].item()),
                        float(q[1].item()),
                        float(q[2].item()),
                        float(q[3].item()),
                    ],
                    dims=[
                        float(2.0 * e[0].item()),
                        float(2.0 * e[1].item()),
                        float(2.0 * e[2].item()),
                    ],
                )
            )

        world_key = torch.cat(world_key_items, dim=0)
        if (
            self._cached_world_key is not None
            and self._cached_world_key.shape == world_key.shape
            and torch.allclose(self._cached_world_key, world_key, atol=1e-6, rtol=0.0)
        ):
            return

        world_config = WorldConfig(cuboid=world_cuboids)
        world_coll_config = WorldCollisionConfig(
            tensor_args=self._tensor_args,
            world_model=world_config,
        )
        self._world_collision = create_collision_checker(world_coll_config)
        self._coll_query_buffer = CollisionQueryBuffer()
        self._coll_weight = self._tensor_args.to_device(torch.tensor([1.0], dtype=dtype))
        self._coll_activation_distance = self._tensor_args.to_device(
            torch.tensor([self.guidance_activation_distance], dtype=dtype)
        )
        self._cached_world_key = world_key.detach().clone()

    def _collision_penalty(self, dist: torch.Tensor) -> torch.Tensor:
        penetration = dist + self.guidance_safety_margin
        penalty = F.relu(penetration)
        if self.guidance_loss_power != 1.0:
            penalty = penalty.pow(self.guidance_loss_power)
        return penalty

    def _guidance_scale_at(self, idx: int, n_steps: int, t: torch.Tensor, dtype, device):
        scale = torch.tensor(self.guidance_scale, device=device, dtype=dtype)
        if not self.guidance_use_schedule:
            return scale

        # Guidance should be stronger at late denoising steps, when samples are
        # close to the data manifold and less likely to be washed out.
        # get_guidance_strength() is monotonic in k where small k -> strong,
        # thus we map idx (0->start) to k (large->start, small->end).
        if n_steps <= 1:
            k = torch.tensor(0.0, device=device, dtype=dtype)
            n = torch.tensor(1.0, device=device, dtype=dtype)
        else:
            k = torch.tensor(float((n_steps - 1) - idx), device=device, dtype=dtype)
            n = torch.tensor(float(n_steps - 1), device=device, dtype=dtype)
        return scale * get_guidance_strength(k, n)

    def _compute_collision_grad(self, q_arm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._world_collision is None:
            z = torch.zeros_like(q_arm)
            return z, torch.zeros((), device=q_arm.device, dtype=q_arm.dtype)

        B, T, D = q_arm.shape
        with torch.enable_grad():
            q_req = q_arm.detach().clone().requires_grad_(True)
            q_flat = q_req.reshape(B * T, D).contiguous()
            kin_state = self._kin_model.get_state(q_flat)
            spheres = kin_state.link_spheres_tensor.reshape(B, T, -1, 4)

            self._coll_query_buffer.update_buffer_shape(
                spheres.shape,
                self._tensor_args,
                self._world_collision.collision_types,
            )
            dist = self._world_collision.get_sphere_distance(
                spheres,
                self._coll_query_buffer,
                self._coll_weight,
                self._coll_activation_distance,
                return_loss=False,
                compute_esdf=True,
            )
            penalty = self._collision_penalty(dist)
            # Avoid gradient dilution from averaging over all spheres and time.
            # Sum over horizon/spheres, then average over batch.
            loss = penalty.reshape(B, -1).sum(dim=-1).mean()
            grad = torch.autograd.grad(loss, q_req, allow_unused=True)[0]
            if grad is None:
                grad = torch.zeros_like(q_req)
            return grad.detach(), loss.detach()

    def _estimate_clean_joint_from_cartesian(
        self,
        q_arm_ref: torch.Tensor,
        cart_phys_clean: torch.Tensor,
        episode_start_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate clean joint sample using one Jacobian linearization step.
        This keeps one-step efficiency while avoiding guidance on very noisy q.
        """
        bsz, horizon, _ = q_arm_ref.shape

        # current absolute pose from reference joint trajectory
        abs_pos_ref, abs_rot_ref = self._fk_to_absolute(q_arm_ref)
        abs_rot6d_ref = matrix_to_rot6d(abs_rot_ref.reshape(-1, 3, 3)).reshape(bsz, horizon, 6)
        abs_pose9_ref = torch.cat([abs_pos_ref, abs_rot6d_ref], dim=-1)

        # target absolute pose from predicted clean Cartesian sample
        abs_pos_clean, abs_rot_clean = self._relative_pose9_to_absolute(
            cart_phys_clean[..., :9], episode_start_pose
        )
        abs_rot6d_clean = matrix_to_rot6d(abs_rot_clean.reshape(-1, 3, 3)).reshape(bsz, horizon, 6)
        abs_pose9_clean = torch.cat([abs_pos_clean, abs_rot6d_clean], dim=-1)

        twist_to_clean = self._absolute_pose_delta_to_twist6(abs_pose9_ref, abs_pose9_clean)
        jac_ref = self._jacobian(q_arm_ref)
        dq_clean = self._dls_pinv_map(
            twist_to_clean.reshape(-1, 6), jac_ref
        ).reshape(bsz, horizon, self._robot_dof)
        if self.max_dq_per_step > 0:
            dq_clean = torch.clamp(dq_clean, -self.max_dq_per_step, self.max_dq_per_step)
        q_arm_clean = q_arm_ref + dq_clean
        return q_arm_clean

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        episode_start_pose: Optional[torch.Tensor] = None,
        obstacle_info=None,
        current_joint_angles: Optional[torch.Tensor] = None,
        return_debug: bool = False,
        **kwargs,
    ):
        if episode_start_pose is None:
            raise ValueError("episode_start_pose is required for joint-space inference.")

        self._ensure_kinematics(condition_data.device)
        self._build_world_collision(
            obstacle_info=obstacle_info,
            device=condition_data.device,
            dtype=condition_data.dtype,
        )

        model = self.model
        scheduler = self.noise_scheduler
        bsz = condition_data.shape[0]
        horizon = condition_data.shape[1]

        timing = {
            "guidance_total": 0.0,
            "guidance_grad": 0.0,
            "guidance_apply": 0.0,
        }

        if current_joint_angles is None:
            q_start = self._solve_start_joint_from_pose(episode_start_pose)
        else:
            q_start = current_joint_angles[..., : self._robot_dof].to(
                device=condition_data.device, dtype=condition_data.dtype
            )
        q_seed = q_start.unsqueeze(1).expand(bsz, horizon, self._robot_dof)

        trajectory_cart_n = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        trajectory_cart_n[condition_mask] = condition_data[condition_mask]

        trajectory_cart = self.normalizer["action"].unnormalize(trajectory_cart_n)
        grip = trajectory_cart[..., 9:10]

        if self.noise_init_mode == "jacobian_projected":
            jac_init = self._jacobian(q_seed)
            eps_twist = torch.randn(
                (bsz * horizon, 6),
                device=q_seed.device,
                dtype=q_seed.dtype,
                generator=generator,
            )
            dq_init = self._dls_pinv_map(eps_twist, jac_init)
            dq_init = dq_init.reshape(bsz, horizon, self._robot_dof)
            if self.max_dq_per_step > 0:
                dq_init = torch.clamp(dq_init, -self.max_dq_per_step, self.max_dq_per_step)
            q_arm = q_seed + dq_init * self.jac_noise_alpha
        elif self.noise_init_mode == "jacobian_diagonal":
            jac_init = self._jacobian(q_seed)
            jt = jac_init.transpose(-2, -1)
            jjt = jac_init @ jt
            eye6 = torch.eye(6, device=jac_init.device, dtype=jac_init.dtype).unsqueeze(0)
            Jpinv = torch.linalg.solve(jjt + self.jacobian_damping * eye6, jac_init).transpose(-2, -1)
            Sigma_q = Jpinv @ Jpinv.transpose(-2, -1)
            per_joint_std = torch.sqrt(torch.clamp(torch.diagonal(Sigma_q, dim1=-2, dim2=-1), min=1e-8))
            mean_std = per_joint_std.mean(dim=-1, keepdim=True)
            per_joint_std = per_joint_std / mean_std * self.jac_noise_alpha
            per_joint_std = per_joint_std.reshape(bsz, horizon, self._robot_dof)
            q_arm = q_seed + torch.randn(
                q_seed.shape,
                device=q_seed.device,
                dtype=q_seed.dtype,
                generator=generator,
            ) * per_joint_std
        else:
            q_arm = q_seed + torch.randn(
                q_seed.shape,
                device=q_seed.device,
                dtype=q_seed.dtype,
                generator=generator,
            ) * self.init_noise_scale

        q_traj = torch.cat([q_arm, grip], dim=-1)

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

        scheduler.set_timesteps(self.num_inference_steps)
        debug = {
            "step_cart_l2": [],
            "guidance_loss": [],
            "guidance_grad_norm": [],
            "clean_sample_cart_err_before": [],
            "clean_sample_cart_err_after": [],
        }
        timesteps = list(scheduler.timesteps)
        n_steps = len(timesteps)

        for idx, t in enumerate(timesteps):
            if q_cond is not None:
                q_traj[cond_step_mask] = q_cond[cond_step_mask]

            q_arm_curr = q_traj[..., : self._robot_dof]
            grip_curr = q_traj[..., self._robot_dof : self._robot_dof + 1]
            abs_pos_curr, abs_rot_curr = self._fk_to_absolute(q_arm_curr)

            rel_pose9_curr = self._absolute_pose_to_relative9(abs_pos_curr, abs_rot_curr, episode_start_pose)
            cart_phys_curr = torch.cat([rel_pose9_curr, grip_curr], dim=-1)
            cart_n_curr = self.normalizer["action"].normalize(cart_phys_curr)
            cart_n_curr[condition_mask] = condition_data[condition_mask]

            eps_cart_n = model(cart_n_curr, t, local_cond=local_cond, global_cond=global_cond)

            # Predicted clean Cartesian sample x0 in normalized action space.
            pred_type = self.noise_scheduler.config.prediction_type
            if pred_type == "epsilon":
                alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
                alpha_prod_t = alpha_prod_t.to(device=cart_n_curr.device, dtype=cart_n_curr.dtype)
                cart_n_clean = get_pred_x0(eps_cart_n, cart_n_curr, alpha_prod_t)
            elif pred_type == "sample":
                cart_n_clean = eps_cart_n
            else:
                cart_n_clean = cart_n_curr
            cart_n_clean[condition_mask] = condition_data[condition_mask]
            cart_phys_clean = self.normalizer["action"].unnormalize(cart_n_clean)

            cart_n_prev_tgt = scheduler.step(
                eps_cart_n, t, cart_n_curr, generator=generator, **kwargs
            ).prev_sample
            cart_n_prev_tgt[condition_mask] = condition_data[condition_mask]
            cart_phys_prev_tgt = self.normalizer["action"].unnormalize(cart_n_prev_tgt)

            abs_pos_tgt, abs_rot_tgt = self._relative_pose9_to_absolute(
                cart_phys_prev_tgt[..., :9], episode_start_pose
            )

            abs_rot6d_curr = matrix_to_rot6d(abs_rot_curr.reshape(-1, 3, 3)).reshape(bsz, horizon, 6)
            abs_pose9_curr_9d = torch.cat([abs_pos_curr, abs_rot6d_curr], dim=-1)

            abs_rot6d_tgt = matrix_to_rot6d(abs_rot_tgt.reshape(-1, 3, 3)).reshape(bsz, horizon, 6)
            abs_pose9_tgt = torch.cat([abs_pos_tgt, abs_rot6d_tgt], dim=-1)

            twist6 = self._absolute_pose_delta_to_twist6(abs_pose9_curr_9d, abs_pose9_tgt)

            jac = self._jacobian(q_arm_curr)
            dq = self._dls_pinv_map(twist6.reshape(-1, 6), jac).reshape(bsz, horizon, self._robot_dof)
            if self.max_dq_per_step > 0:
                dq = torch.clamp(dq, -self.max_dq_per_step, self.max_dq_per_step)
            q_arm_new = q_arm_curr + dq

            for _ in range(2):
                abs_pos_ri, abs_rot_ri = self._fk_to_absolute(q_arm_new)
                abs_rot6d_ri = matrix_to_rot6d(abs_rot_ri.reshape(-1, 3, 3)).reshape(bsz, horizon, 6)
                abs_pose9_ri = torch.cat([abs_pos_ri, abs_rot6d_ri], dim=-1)
                twist_ri = self._absolute_pose_delta_to_twist6(abs_pose9_ri, abs_pose9_tgt)
                dq_ri = self._dls_pinv_map(twist_ri.reshape(-1, 6), jac).reshape(bsz, horizon, self._robot_dof)
                if self.max_dq_per_step > 0:
                    dq_ri = torch.clamp(dq_ri, -self.max_dq_per_step, self.max_dq_per_step)
                q_arm_new = q_arm_new + dq_ri

            is_last = idx == (n_steps - 1)
            if self.ik_refine_each_step or (is_last and self.ik_refine_last_step):
                q_arm_new = self._ik_from_absolute(abs_pos_tgt, abs_rot_tgt, q_arm_new)

            apply_guidance = (self._world_collision is not None)
            if self.guidance_apply_last_step_only and not is_last:
                apply_guidance = False

            if apply_guidance:
                t0 = time.perf_counter()

                if self.guidance_use_clean_sample:
                    q_guidance_state = self._estimate_clean_joint_from_cartesian(
                        q_arm_ref=q_arm_new,
                        cart_phys_clean=cart_phys_clean,
                        episode_start_pose=episode_start_pose,
                    )
                else:
                    q_guidance_state = q_arm_new

                for _ in range(self.guidance_steps_per_denoise):
                    tg = time.perf_counter()
                    grad, guide_loss = self._compute_collision_grad(q_guidance_state)
                    timing["guidance_grad"] += time.perf_counter() - tg

                    # Do not perturb conditioned prefix steps.
                    if torch.any(cond_step_mask):
                        grad = grad.clone()
                        grad[cond_step_mask] = 0.0

                    ta = time.perf_counter()
                    if self.guidance_grad_clip > 0:
                        grad = torch.clamp(
                            grad,
                            min=-self.guidance_grad_clip,
                            max=self.guidance_grad_clip,
                        )

                    gamma = self._guidance_scale_at(
                        idx=idx,
                        n_steps=n_steps,
                        t=t,
                        dtype=q_arm_new.dtype,
                        device=q_arm_new.device,
                    )
                    q_arm_new = q_arm_new - gamma * grad
                    # Keep guidance linearization state synchronized.
                    q_guidance_state = q_guidance_state - gamma * grad
                    timing["guidance_apply"] += time.perf_counter() - ta

                    if return_debug:
                        debug["guidance_loss"].append(guide_loss.detach().cpu())
                        debug["guidance_grad_norm"].append(
                            torch.linalg.norm(grad.reshape(bsz, -1), dim=-1).detach().cpu()
                        )
                timing["guidance_total"] += time.perf_counter() - t0

                if return_debug and self.guidance_use_clean_sample:
                    abs_p_before, abs_r_before = self._fk_to_absolute(q_arm_new + gamma * grad)
                    rel9_before = self._absolute_pose_to_relative9(abs_p_before, abs_r_before, episode_start_pose)
                    cart_before = torch.cat([rel9_before, cart_phys_prev_tgt[..., 9:10]], dim=-1)
                    err_before = torch.linalg.norm(
                        (cart_before[..., :9] - cart_phys_clean[..., :9]).reshape(bsz, -1), dim=-1
                    )

                    abs_p_after, abs_r_after = self._fk_to_absolute(q_arm_new)
                    rel9_after = self._absolute_pose_to_relative9(abs_p_after, abs_r_after, episode_start_pose)
                    cart_after = torch.cat([rel9_after, cart_phys_prev_tgt[..., 9:10]], dim=-1)
                    err_after = torch.linalg.norm(
                        (cart_after[..., :9] - cart_phys_clean[..., :9]).reshape(bsz, -1), dim=-1
                    )
                    debug["clean_sample_cart_err_before"].append(err_before.detach().cpu())
                    debug["clean_sample_cart_err_after"].append(err_after.detach().cpu())

            q_traj = torch.cat([q_arm_new, cart_phys_prev_tgt[..., 9:10]], dim=-1)

            if return_debug:
                q_arm_dbg = q_traj[..., : self._robot_dof]
                abs_p, abs_r = self._fk_to_absolute(q_arm_dbg)
                rel9 = self._absolute_pose_to_relative9(abs_p, abs_r, episode_start_pose)
                cart_phys_after = torch.cat(
                    [rel9, q_traj[..., self._robot_dof : self._robot_dof + 1]], dim=-1
                )
                debug["step_cart_l2"].append(
                    torch.linalg.norm(
                        (cart_phys_after - cart_phys_prev_tgt).reshape(bsz, -1), dim=-1
                    ).detach().cpu()
                )

        if q_cond is not None:
            q_traj[cond_step_mask] = q_cond[cond_step_mask]

        self._last_joint_traj = q_traj.detach()

        q_arm_final = q_traj[..., : self._robot_dof]
        grip_final = q_traj[..., self._robot_dof : self._robot_dof + 1]
        abs_p_f, abs_r_f = self._fk_to_absolute(q_arm_final)
        rel9_f = self._absolute_pose_to_relative9(abs_p_f, abs_r_f, episode_start_pose)
        cart_final = torch.cat([rel9_f, grip_final], dim=-1)
        cart_final_n = self.normalizer["action"].normalize(cart_final)
        cart_final_n[condition_mask] = condition_data[condition_mask]

        if return_debug:
            debug["timing_guidance"] = timing
            return cart_final_n, q_traj, debug
        return cart_final_n

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        fixed_action_prefix: torch.Tensor = None,
        env_batched=False,
        episode_start_pose: torch.Tensor = None,
        obstacle_info=None,
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
