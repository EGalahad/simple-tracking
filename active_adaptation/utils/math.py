# This file contains additional math utilities
# that are not covered by IsaacLab

import torch
import torch.distributions as D


class MultiUniform(D.Distribution):
    """
    A distribution over the union of multiple disjoint intervals.
    """

    def __init__(self, ranges: torch.Tensor):
        batch_shape = ranges.shape[:-2]
        if not ranges[..., 0].le(ranges[..., 1]).all():
            raise ValueError("Ranges must be non-empty and ordered.")
        super().__init__(batch_shape, validate_args=False)
        self.ranges = ranges
        self.ranges_len = ranges.diff(dim=-1).squeeze(1)
        self.total_len = self.ranges_len.sum(-1)
        self.starts = torch.zeros_like(ranges[..., 0])
        self.starts[..., 1:] = self.ranges_len.cumsum(-1)[..., :-1]

    def sample(self, sample_shape: torch.Size = ()) -> torch.Tensor:
        sample_shape = torch.Size(sample_shape)
        shape = sample_shape + self.batch_shape
        uniform = torch.rand(shape, device=self.ranges.device) * self.total_len
        i = torch.searchsorted(self.starts, uniform) - 1
        return self.ranges[i, 0] + uniform - self.starts[i]


class EMA:
    """
    Exponential Moving Average.

    Args:
        x: The tensor to compute the EMA of.
        gammas: The decay rates. Can be a single float or a list of floats.

    Example:
        >>> ema = EMA(x, gammas=[0.9, 0.99])
        >>> ema.update(x)
        >>> ema.ema
    """

    def __init__(self, x: torch.Tensor, gammas):
        self.gammas = torch.tensor(gammas, device=x.device)
        shape = (x.shape[0], len(self.gammas), *x.shape[1:])
        self.sum = torch.zeros(shape, device=x.device)
        shape = (x.shape[0], len(self.gammas), 1)
        self.cnt = torch.zeros(shape, device=x.device)

    def reset(self, env_ids: torch.Tensor):
        self.sum[env_ids] = 0.0
        self.cnt[env_ids] = 0.0

    def update(self, x: torch.Tensor):
        self.sum.mul_(self.gammas.unsqueeze(-1)).add_(x.unsqueeze(1))
        self.cnt.mul_(self.gammas.unsqueeze(-1)).add_(1.0)
        self.ema = self.sum / self.cnt
        return self.ema
