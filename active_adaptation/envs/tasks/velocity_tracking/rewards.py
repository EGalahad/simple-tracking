import torch

from active_adaptation.envs.mdp.rewards.base import Reward as BaseReward
from active_adaptation.envs.tasks.velocity_tracking.command import LocomotionCommand
from mjlab.utils.lab_api.math import quat_apply, yaw_quat


LocomotionReward = BaseReward[LocomotionCommand]


class track_lin_vel(LocomotionReward):
    """Reward for tracking linear velocity commands."""

    def __init__(self, sigma: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

    def compute(self):
        robot_linvel_w = self.command_manager.asset.data.root_com_lin_vel_w
        robot_quat_w = self.command_manager.asset.data.root_link_quat_w

        # Transform command velocity to world frame.
        command_linvel_w = quat_apply(
            yaw_quat(robot_quat_w), self.command_manager.command_linvel
        )

        # Compute tracking error.
        linvel_error = (robot_linvel_w - command_linvel_w).norm(dim=-1)
        return torch.exp(-linvel_error / self.sigma).unsqueeze(-1)


class track_ang_vel(LocomotionReward):
    """Reward for tracking angular velocity commands."""

    def __init__(self, sigma: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

    def compute(self):
        robot_angvel_w = self.command_manager.asset.data.root_com_ang_vel_w[:, 2:3]
        command_angvel = self.command_manager.command_angvel

        # Compute tracking error.
        angvel_error = (robot_angvel_w - command_angvel).abs()
        return torch.exp(-angvel_error / self.sigma)


class is_standing_env(LocomotionReward):
    """Check if the robot is standing based on linear and angular velocity."""

    def compute(self):
        return self.command_manager.is_standing_env.float()
