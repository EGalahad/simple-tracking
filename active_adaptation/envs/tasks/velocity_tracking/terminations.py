import torch

from active_adaptation.envs.mdp.terminations.base import Termination as BaseTermination
from active_adaptation.envs.tasks.velocity_tracking.command import LocomotionCommand
from mjlab.utils.lab_api.math import quat_apply, yaw_quat


LocomotionTermination = BaseTermination[LocomotionCommand]


class cum_lin_vel_error(LocomotionTermination):
    """Termination based on cumulative linear velocity error."""

    def __init__(self, threshold: float = 0.2, decay=0.99, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.decay = decay
        self.cum_error = torch.zeros(self.num_envs, 3, device=self.device)

    def reset(self, env_ids):
        self.cum_error[env_ids] = 0.0

    def update(self):
        # Compute current error.
        robot_linvel_w = self.command_manager.asset.data.root_com_lin_vel_w
        command_linvel_w = quat_apply(
            yaw_quat(self.command_manager.asset.data.root_link_quat_w),
            self.command_manager.command_linvel,
        )

        linvel_error = robot_linvel_w - command_linvel_w
        self.cum_error.mul_(self.decay).add_(linvel_error * self.env.step_dt)

    def __call__(self):
        exceeded = self.cum_error.norm(dim=1) > self.threshold
        return exceeded.unsqueeze(-1)


class cum_ang_vel_error(LocomotionTermination):
    """Termination based on cumulative angular velocity error."""

    def __init__(self, threshold: float = 0.4, decay=0.99, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.decay = decay
        self.cum_error = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids):
        self.cum_error[env_ids] = 0.0

    def update(self):
        # Compute current error.
        robot_angvel_w = self.command_manager.asset.data.root_com_ang_vel_w[:, 2:3]
        command_angvel = self.command_manager.command_angvel

        angvel_error = (robot_angvel_w - command_angvel).squeeze(-1)
        self.cum_error.mul_(self.decay).add_(angvel_error * self.env.step_dt)

    def __call__(self):
        exceeded = self.cum_error.abs() > self.threshold
        return exceeded.unsqueeze(-1)
