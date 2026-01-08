import torch
import torch.nn as nn
from typing import Dict, Sequence
from copy import deepcopy


class VecNorm(nn.Module):
    """Simple running normalization for observations.

    Keeps track of running mean and variance to normalize observations on-the-fly.

    Args:
        obs_keys: List of observation keys to normalize
        decay: Decay rate for running statistics (default: 0.99)
        eps: Small constant for numerical stability (default: 1e-4)
        device: Device to store statistics on
    """

    def __init__(
        self,
        obs_keys: Sequence[str],
        decay: float = 0.9999,
        eps: float = 1e-4,
        device: str = "cpu",
    ):
        super().__init__()
        self.obs_keys = list(obs_keys)
        self.decay = decay
        self.eps = eps
        self.device = device

        # Running statistics as nn.Parameters
        self.sum: Dict[str, nn.Parameter] = nn.ParameterDict({})
        self.ssq: Dict[str, nn.Parameter] = nn.ParameterDict({})
        self.cnt: Dict[str, nn.Parameter] = nn.ParameterDict({})
        self.mean: Dict[str, nn.Parameter] = nn.ParameterDict({})
        self.var: Dict[str, nn.Parameter] = nn.ParameterDict({})

        self.initialized = False
        self.frozen = False

    def init_stats(self, obs_dict: Dict[str, torch.Tensor]):
        """Initialize running statistics based on observation shapes."""
        for key in self.obs_keys:
            if key in obs_dict:
                shape = obs_dict[key].shape[1:]  # Remove batch dimension
                self.sum[key] = nn.Parameter(
                    torch.zeros(shape, device=self.device), requires_grad=False
                )
                self.ssq[key] = nn.Parameter(
                    torch.zeros(shape, device=self.device), requires_grad=False
                )
                self.cnt[key] = nn.Parameter(
                    torch.zeros(1, device=self.device), requires_grad=False
                )
                self.mean[key] = nn.Parameter(
                    torch.zeros(shape, device=self.device), requires_grad=False
                )
                self.var[key] = nn.Parameter(
                    torch.ones(shape, device=self.device), requires_grad=False
                )
        self.initialized = True

    def update(self, obs_dict: Dict[str, torch.Tensor]):
        """Update running statistics with new observations."""
        if not self.initialized:
            self.init_stats(obs_dict)

        if self.frozen:
            return

        for key in self.obs_keys:
            if key not in obs_dict:
                continue

            x = obs_dict[key]
            sum_x = x.sum(dim=0)
            ssq_x = (x**2).sum(dim=0)
            cnt_x = x.shape[0]

            self.sum[key] = self.sum[key] * self.decay + sum_x
            self.ssq[key] = self.ssq[key] * self.decay + ssq_x
            self.cnt[key] = self.cnt[key] * self.decay + cnt_x

            self.mean[key] = self.sum[key] / self.cnt[key]
            self.var[key] = (self.ssq[key] / self.cnt[key] - self.mean[key] ** 2).clamp(
                min=self.eps
            )

    def normalize(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize observations using running statistics."""
        if not self.initialized:
            self.init_stats(obs_dict)
            return obs_dict.copy()

        normalized = obs_dict.copy()
        for key in self.obs_keys:
            normalized[key] = (normalized[key] - self.mean[key]) / self.var[key].sqrt()

        return normalized

    def denormalize(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Denormalize observations using running statistics."""
        if not self.initialized:
            self.init_stats(obs_dict)
            return obs_dict.copy()

        denormalized = obs_dict.copy()
        for key in self.obs_keys:
            denormalized[key] = (
                denormalized[key] * self.var[key].sqrt() + self.mean[key]
            )

        return denormalized

    def freeze(self):
        """Freeze running statistics updates."""
        self.frozen = True

    def unfreeze(self):
        """Unfreeze running statistics updates."""
        self.frozen = False

    def get_stats(self):
        """Get current running statistics."""
        return {
            "sum": {key: param.data.clone() for key, param in self.sum.items()},
            "ssq": {key: param.data.clone() for key, param in self.ssq.items()},
            "cnt": {key: param.data.clone() for key, param in self.cnt.items()},
            "mean": {key: param.data.clone() for key, param in self.mean.items()},
            "var": {key: param.data.clone() for key, param in self.var.items()},
        }

    def load_stats(self, stats):
        """Load running statistics."""
        for key in stats["sum"]:
            self.sum[key] = nn.Parameter(
                deepcopy(stats["sum"][key]), requires_grad=False
            )
            self.ssq[key] = nn.Parameter(
                deepcopy(stats["ssq"][key]), requires_grad=False
            )
            self.cnt[key] = nn.Parameter(
                deepcopy(stats["cnt"][key]), requires_grad=False
            )
            self.mean[key] = nn.Parameter(
                deepcopy(stats["mean"][key]), requires_grad=False
            )
            self.var[key] = nn.Parameter(
                deepcopy(stats["var"][key]), requires_grad=False
            )
        self.initialized = True
