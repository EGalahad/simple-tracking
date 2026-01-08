import torch
from typing import Sequence

from active_adaptation.envs.mdp.commands.base import Command
from mjlab.utils.lab_api.math import (
    sample_uniform,
    quat_apply_inverse,
    quat_apply,
    wrap_to_pi,
)


class LocomotionCommand(Command):
    def __init__(
        self,
        env,
        linvel_x_range=(-1.0, 1.0),
        linvel_y_range=(-1.0, 1.0),
        angvel_range=(-1.0, 1.0),
        yaw_stiffness_range=(0.5, 0.6),
        use_stiffness_ratio: float = 0.5,
        resample_interval: int = 300,
        resample_prob: float = 0.75,
        stand_prob: float = 0.05,
        motion_path: str | Sequence[str] | None = None,
        lift_height: float = 0.0,
    ):
        super().__init__(env)

        # Store parameters
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.angvel_range = angvel_range
        self.yaw_stiffness_range = yaw_stiffness_range
        self.use_stiffness_ratio = use_stiffness_ratio
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob
        self.stand_prob = stand_prob

        with torch.device(self.device):
            # Command state variables
            self.target_yaw = torch.zeros(self.num_envs, 1)
            self.yaw_stiffness = torch.zeros(self.num_envs, 1)
            self.use_stiffness = torch.zeros(self.num_envs, 1, dtype=bool)
            self.fixed_yaw_speed = torch.zeros(self.num_envs, 1)

            self.command_linvel = torch.zeros(self.num_envs, 3)
            self.next_command_linvel = torch.zeros(self.num_envs, 3)
            self.command_angvel = torch.zeros(self.num_envs, 1)

            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

        # motion
        if motion_path is not None:
            from active_adaptation.utils.motion import MotionDataset

            self.dataset = MotionDataset.create_from_path(
                motion_path, target_fps=int(1 / self.env.step_dt)
            ).to(self.device)
            self.root_idx = self.dataset.body_names.index("pelvis")
            self.joint_indices = [
                self.dataset.joint_names.index(jnt_name)
                for jnt_name in self.asset.joint_names
            ]
            self.lift_height = lift_height
        else:
            self.dataset = None

    def sample_init(self, env_ids):
        if self.dataset is not None:
            # Sample initial motion poses from motion dataset
            from active_adaptation.utils.motion import MotionData

            indices = torch.randint(0, self.dataset.num_steps, (len(env_ids),))
            motion_data: MotionData = self.dataset.data[indices]

            joint_pos = motion_data.joint_pos[:, self.joint_indices]
            joint_vel = motion_data.joint_vel[:, self.joint_indices]
            self.asset.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
            self.asset.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)

            root_link_pos_w = (
                motion_data.body_pos_w[:, self.root_idx]
                + self.env.scene.env_origins[env_ids]
            )
            root_link_quat_w = motion_data.body_quat_w[:, self.root_idx]
            root_com_lin_vel_w = motion_data.body_lin_vel_w[:, self.root_idx]
            root_com_ang_vel_w = motion_data.body_ang_vel_w[:, self.root_idx]
            root_link_pos_w[..., 2] += self.lift_height

            root_lin_vel_b = quat_apply_inverse(root_link_quat_w, root_com_lin_vel_w)
            root_ang_vel_b = quat_apply_inverse(root_link_quat_w, root_com_ang_vel_w)

            root_lin_vel_b[..., 0].clamp_(
                min=self.linvel_x_range[0], max=self.linvel_x_range[1]
            )
            root_lin_vel_b[..., 1].clamp_(
                min=self.linvel_y_range[0], max=self.linvel_y_range[1]
            )
            root_lin_vel_b[..., 2].zero_()

            root_ang_vel_b[..., 0].zero_()
            root_ang_vel_b[..., 1].zero_()
            root_ang_vel_b[..., 2].clamp_(
                min=self.angvel_range[0], max=self.angvel_range[1]
            )

            root_com_lin_vel_w = quat_apply(root_link_quat_w, root_lin_vel_b)
            root_com_ang_vel_w = quat_apply(root_link_quat_w, root_ang_vel_b)

            root_state_w = torch.cat(
                [
                    root_link_pos_w,
                    root_link_quat_w,
                    root_com_lin_vel_w,
                    root_com_ang_vel_w,
                ],
                dim=-1,
            )
            self.asset.write_root_state_to_sim(root_state_w, env_ids=env_ids)

            # set command to match current state
            self.command_linvel[env_ids] = root_lin_vel_b
            self.next_command_linvel[env_ids] = root_lin_vel_b

            self.target_yaw[env_ids] = self.asset.data.heading_w[env_ids, None]
            self.fixed_yaw_speed[env_ids] = root_ang_vel_b[:, 2:3]
            self.command_angvel[env_ids] = root_ang_vel_b[:, 2:3]

            self.is_standing_env[env_ids] = (
                root_lin_vel_b.norm(dim=-1, keepdim=True) < 0.1
            ) & (root_ang_vel_b[:, 2:3].abs() < 0.1)

            return None
        else:
            self.command_linvel[env_ids] = 0.0
            self.next_command_linvel[env_ids] = 0.0

            self.target_yaw[env_ids] = self.asset.data.heading_w[env_ids, None]
            self.fixed_yaw_speed[env_ids] = 0.0
            self.command_angvel[env_ids] = 0.0

            self.is_standing_env[env_ids] = True

            joint_pos = self.asset.data.default_joint_pos[env_ids]
            joint_vel = self.asset.data.default_joint_vel[env_ids]
            self.asset.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
            self.asset.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)

            return super().sample_init(env_ids)

    def update(self):
        # Smoothly interpolate to next command
        self.command_linvel.lerp_(self.next_command_linvel, 0.1)

        # Compute angular velocity command using yaw stiffness
        body_heading_w = self.asset.data.heading_w.unsqueeze(1)
        yaw_diff = wrap_to_pi(self.target_yaw - body_heading_w)
        command_yaw_speed = torch.clamp(
            self.yaw_stiffness * yaw_diff,
            min=self.angvel_range[0],
            max=self.angvel_range[1],
        )

        self.command_angvel = torch.where(
            self.use_stiffness,
            command_yaw_speed,
            self.fixed_yaw_speed,
        )

        # Check if we should resample commands
        interval_reached = (
            self.env.episode_length_buf + 1
        ) % self.resample_interval == 0
        resample_mask = interval_reached & (
            (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
            | self.is_standing_env.squeeze(1)
        )

        if resample_mask.any():
            resample_env_ids = resample_mask.nonzero().squeeze(-1)
            self.sample_command(resample_env_ids)

    def sample_command(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        stand_mask = torch.rand(len(env_ids), 1, device=self.device) < self.stand_prob

        next_command_linvel = torch.zeros(len(env_ids), 3, device=self.device)
        next_command_linvel[:, 0].uniform_(*self.linvel_x_range)
        next_command_linvel[:, 1].uniform_(*self.linvel_y_range)

        # Apply standing probability
        speed = next_command_linvel.norm(dim=-1, keepdim=True)
        valid = ~((speed < 0.10) | stand_mask)
        self.next_command_linvel[env_ids] = next_command_linvel * valid

        self.target_yaw[env_ids] = torch.empty(
            len(env_ids), 1, device=self.device
        ).uniform_(-torch.pi, torch.pi)

        shape = (len(env_ids), 1)
        self.yaw_stiffness[env_ids] = sample_uniform(
            *self.yaw_stiffness_range, shape, self.device
        )
        use_stiffness = torch.rand(shape, device=self.device) < self.use_stiffness_ratio
        fixed_yaw_speed = sample_uniform(*self.angvel_range, shape, self.device)

        self.use_stiffness[env_ids] = use_stiffness & valid
        self.fixed_yaw_speed[env_ids] = fixed_yaw_speed * valid

        self.is_standing_env[env_ids] = ~valid

    def debug_draw(self):
        return
