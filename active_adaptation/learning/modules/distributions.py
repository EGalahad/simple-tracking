import torch
import torch.distributions as D
from torch.distributions import constraints

D.Distribution.set_default_validate_args(False)


class IndependentNormal(D.Independent):
    dist_keys = ["loc", "scale"]
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = D.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def deterministic_sample(self):
        return self.base_dist.mean


class IndependentBeta(D.Independent):
    dist_keys = ["alpha", "beta"]

    def __init__(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        min: float | torch.Tensor = 0.0,
        max: float | torch.Tensor = 1.0,
        event_dims: int = 1,
    ):
        self.min = torch.as_tensor(min, device=alpha.device).broadcast_to(alpha.shape)
        self.max = torch.as_tensor(max, device=alpha.device).broadcast_to(alpha.shape)
        self.scale = self.max - self.min
        self.eps = torch.finfo(alpha.dtype).eps
        base_dist = D.Beta(alpha, beta)
        super().__init__(base_dist, event_dims)

    def sample(self, sample_shape: torch.Size = torch.Size()):
        return super().sample(sample_shape) * self.scale + self.min

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        return super().rsample(sample_shape) * self.scale + self.min

    def log_prob(self, value: torch.Tensor):
        return super().log_prob(
            ((value - self.min) / self.scale).clamp(self.eps, 1.0 - self.eps)
        )
