import abc
from typing import Generic, TYPE_CHECKING, TypeVar

from active_adaptation.envs.mdp.utils.registry import RegistryMixin

if TYPE_CHECKING:
    from active_adaptation.envs.base import _Env
    from active_adaptation.envs.mdp.commands.base import Command

CT = TypeVar("CT", bound="Command")


class Termination(Generic[CT], RegistryMixin):
    def __init__(self, env, **kwargs):
        if kwargs:
            print("Warning: Unused kwargs in Termination:", kwargs)
            breakpoint()
        super().__init__(**kwargs)
        self.env: _Env = env
        self.command_manager: CT = env.command_manager

    def update(self):
        pass

    def reset(self, env_ids):
        pass

    @abc.abstractmethod
    def __call__(self) -> "torch.Tensor":
        raise NotImplementedError

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device
