import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation
    from active_adaptation.envs.base import _Env


class ActionManager:
    action_dim: int

    def __init__(self, env):
        self.env: _Env = env
        self.asset: Articulation = self.env.scene["robot"]
        self.action_buf: torch.Tensor

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        pass

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device
