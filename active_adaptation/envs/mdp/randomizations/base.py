from typing import Generic, TYPE_CHECKING, TypeVar

from active_adaptation.envs.mdp.utils.registry import RegistryMixin

if TYPE_CHECKING:
    from active_adaptation.envs.base import _Env
    from active_adaptation.envs.mdp.commands.base import Command

CT = TypeVar("CT", bound="Command")


class Randomization(Generic[CT], RegistryMixin):
    def __init__(self, env):
        self.env: _Env = env
        self.command_manager: CT = env.command_manager

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    def startup(self):
        pass

    def reset(self, env_ids):
        pass

    def step(self, substep):
        pass

    def update(self):
        pass

    def debug_draw(self):
        pass
