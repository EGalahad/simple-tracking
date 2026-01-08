from active_adaptation.envs.mdp.observations.base import Observation as BaseObservation
from active_adaptation.envs.tasks.velocity_tracking.command import LocomotionCommand


LocomotionObservation = BaseObservation[LocomotionCommand]


class command_lin_vel_b(LocomotionObservation):
    """Linear velocity command in robot body frame."""

    def compute(self):
        return self.command_manager.command_linvel[:, :2]


class command_ang_vel_b(LocomotionObservation):
    """Angular velocity command in robot body frame."""

    def compute(self):
        return self.command_manager.command_angvel
