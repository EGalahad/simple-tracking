import abc
from typing import Generic, Tuple, TYPE_CHECKING, TypeVar

import torch

from active_adaptation.envs.mdp.utils.registry import RegistryMixin

if TYPE_CHECKING:
    from active_adaptation.envs.base import _Env
    from active_adaptation.envs.mdp.commands.base import Command

CT = TypeVar("CT", bound="Command")


class Observation(Generic[CT], RegistryMixin):
    """
    Base class for all observations.
    """

    def __init__(self, env):
        self.env: _Env = env
        self.command_manager: CT = env.command_manager

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = self.compute()
        return tensor

    def startup(self):
        """Called once upon initialization of the environment."""
        pass

    def post_step(self, substep: int):
        """Called after each physics substep."""
        pass

    def update(self):
        """Called after all physics substeps are completed."""
        pass

    def reset(self, env_ids: torch.Tensor):
        """Called after episode termination."""

    def debug_draw(self):
        """Called at each step **after** simulation, if GUI is enabled."""
        pass
