import abc
from typing import Generic, TYPE_CHECKING, TypeVar

import torch

from active_adaptation.envs.mdp.utils.registry import RegistryMixin

if TYPE_CHECKING:
    from active_adaptation.envs.base import _Env
    from active_adaptation.envs.mdp.commands.base import Command

CT = TypeVar("CT", bound="Command")


class Reward(Generic[CT], RegistryMixin):
    def __init__(
        self,
        env,
        weight: float,
        enabled: bool = True,
    ):
        self.env: _Env = env
        self.command_manager: CT = env.command_manager
        self.weight = weight
        self.enabled = enabled

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    def step(self, substep: int):
        pass

    def post_step(self, substep: int):
        pass

    def update(self):
        pass

    def reset(self, env_ids: torch.Tensor):
        pass

    def __call__(self) -> torch.Tensor:
        result = self.compute()
        if isinstance(result, torch.Tensor):
            rew, count = result, result.numel()
        elif isinstance(result, tuple):
            rew, is_active = result
            rew = rew * is_active.float()
            count = is_active.sum().item()
        return self.weight * rew, count

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def debug_draw(self):
        pass
