import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import warnings
import functools
import torch.utils._pytree as pytree
import copy
import numpy as np

from torchrl.data import CompositeSpec, TensorSpec, Unbounded
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import TensorDictPrimer
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union, List
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *

torch.set_float32_matmul_precision("high")

REF_JPOS_KEY = "ref_joint_pos_"
AMP_KEY = "amp_obs_"


@dataclass
class AMPConfig:
    motion_data_path: List[str] = (
        r"data/motion/AMASS/ACCAD/Female1Walking_c3d-z=-0.1/.*",
    )
    lr: float = 1e-5
    weight_decay: float = 1e-3
    reward_scale: float = 1.0
    gan_type: str = "lsgan"  # "gan", "wgan", "lsgan"
    eta_wgan: float = 0.3

    amp_normalizer_source: List[str] = (
        "expert",
    )  # ("expert", "policy", "policy_replay")

    grad_pen_weight: float = 10.0
    grad_pen_target_norm: float = 0.0
    grad_pen_source: List[str] = (
        "interpolated",
    )  # "expert", "policy", "policy_replay", "interpolated"

    replay_buffer_iters: int = 16
    random_replace: bool = False


# From https://github.com/ami-iit/amp-rsl-rl/blob/main/amp_rsl_rl/networks/discriminator.py for computing gradient penalty
# BCELoss: use expert, grad norm target = 0
# WGAN: use interpolated, grad norm target = 1
# LSGAN: not sure, from amp for hardware is interpolated, grad norm target = 0


@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_amp.PPOAMP"
    name: str = "ppo_amp"
    train_every: int = 32
    ppo_epochs: int = 3
    num_minibatches: int = 8
    clip_param: float = 0.2
    enable_residual_distillation: bool = True

    # lr linear schedule or adaptive lr
    lr_start: float = 3e-4
    lr_end: float = 1e-4
    lr_decay_iters: int = 500

    desired_kl: float | None = 0.01  # None

    # entropy coef schedule
    entropy_coef_start: float = 0.001
    entropy_coef_end: float = 0.001
    entropy_decay_iters: int = 1000

    init_noise_scale: float = 1.0
    load_noise_scale: float | None = 0.5

    clip_neg_reward: bool = False

    normalize_before_sum: bool = True

    layer_norm: Union[str, None] = "before"
    value_norm: bool = True

    latent_dim: int = 256

    max_grad_norm: float = 1.0

    clip_adv: float | None = None
    phase: str = "train"
    vecnorm: Union[str, None] = None
    checkpoint_path: Union[str, None] = None
    in_keys: List[str] = (CMD_KEY, OBS_KEY, OBS_PRIV_KEY, AMP_KEY)

    amp: AMPConfig = field(default_factory=AMPConfig)


cs = ConfigStore.instance()
cs.store(
    "ppo_amp_train",
    node=PPOConfig(
        phase="train", vecnorm="train", entropy_coef_start=0.001, entropy_coef_end=0.001
    ),
    group="algo",
)
cs.store(
    "ppo_amp_adapt",
    node=PPOConfig(
        phase="adapt", vecnorm="eval", entropy_coef_start=0.00, entropy_coef_end=0.00
    ),
    group="algo",
)
cs.store(
    "ppo_amp_finetune",
    node=PPOConfig(
        phase="finetune",
        vecnorm="eval",
        entropy_coef_start=0.001,
        entropy_coef_end=0.001,
    ),
    group="algo",
)


class PPOAMP(TensorDictModuleBase):
    train_in_keys = [
        CMD_KEY,
        OBS_KEY,
        OBS_PRIV_KEY,
        ACTION_KEY,
        "adv",
        "ret",
        "is_init",
        "sample_log_prob",
        "step_count",
    ]

    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
        env,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        assert self.cfg.phase in ["train", "adapt", "finetune"]

        self.entropy_coef = self.cfg.entropy_coef_start
        self.desired_kl = cfg.desired_kl
        self.clip_param = self.cfg.clip_param

        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.adapt_loss_fn = nn.MSELoss(reduction="none")
        self.rec_loss = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)

        self.reward_groups = list(env.cfg.reward.keys()) + ["amp"]
        num_reward_groups = len(self.reward_groups)
        self.reward_scales = torch.ones(num_reward_groups, device=self.device)
        self.reward_scales[-1] = self.cfg.amp.reward_scale
        self.reward_scales /= self.reward_scales.sum()

        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=num_reward_groups).to(self.device)

        self.action_dim = action_spec.shape[-1]
        self.joint_names = env.action_manager.joint_names

        fake_input = observation_spec.zero()

        latent_dim = self.cfg.latent_dim
        self.encoder_priv = Seq(
            Mod(
                nn.Sequential(make_mlp([latent_dim]), nn.LazyLinear(latent_dim)),
                [OBS_PRIV_KEY],
                ["priv_feature"],
            ),
        ).to(self.device)

        if observation_spec.get("command_", None) is not None:
            global CMD_KEY
            CMD_KEY = "command_"

        object.__setattr__(self, "env", env)

        self.adapt_module = Mod(
            nn.Sequential(make_mlp([512, 256]), nn.LazyLinear(latent_dim)),
            [OBS_KEY],
            ["priv_pred"],
        ).to(self.device)

        if cfg.phase == "train" and cfg.enable_residual_distillation:
            assert (
                REF_JPOS_KEY in observation_spec
            ), f"{REF_JPOS_KEY} should be in observation_spec"

            class RefJointPos(nn.Module):
                def forward(self, ref_jpos, action):
                    return (ref_jpos + action,)

            residual_module_cls = RefJointPos
        else:

            class DummyRefJointPos(nn.Module):
                def forward(self, ref_jpos, action):
                    return action

            residual_module_cls = DummyRefJointPos
        in_keys = [REF_JPOS_KEY, "loc"]
        out_keys = ["loc"]
        residual_module = Mod(residual_module_cls(), in_keys, out_keys)

        def build_actor(
            in_keys: List[str], dist_cls, dist_keys, residual_module=None
        ) -> ProbabilisticActor:
            actor_modules = [
                CatTensors(in_keys, "_actor_inp", del_keys=False, sort=False),
                Mod(make_mlp([512, 256, 256]), ["_actor_inp"], ["_actor_feature"]),
                Mod(
                    Actor(
                        self.action_dim,
                        init_noise_scale=self.cfg.init_noise_scale,
                        load_noise_scale=self.cfg.load_noise_scale,
                    ),
                    ["_actor_feature"],
                    dist_keys,
                ),
            ]
            if residual_module is not None:
                actor_modules.append(residual_module)
            actor_module = Seq(*actor_modules)
            actor = ProbabilisticActor(
                module=actor_module,
                in_keys=dist_keys,
                out_keys=[ACTION_KEY],
                distribution_class=dist_cls,
                return_log_prob=True,
            ).to(self.device)
            return actor

        self.dist_cls = IndependentNormal
        self.dist_keys = IndependentNormal.dist_keys

        in_keys = [CMD_KEY, OBS_KEY, "priv_feature"]
        self.actor = build_actor(
            in_keys, self.dist_cls, self.dist_keys, residual_module=residual_module
        )
        in_keys = [CMD_KEY, OBS_KEY, "priv_pred"]
        self.actor_adapt = build_actor(in_keys, self.dist_cls, self.dist_keys)

        _critic = nn.Sequential(
            make_mlp([512, 256, 128]), nn.LazyLinear(num_reward_groups)
        )
        self.critic = Seq(
            CatTensors(
                [CMD_KEY, OBS_KEY, OBS_PRIV_KEY, AMP_KEY],
                "_critic_input",
                del_keys=False,
            ),
            Mod(_critic, ["_critic_input"], ["state_value"]),
        ).to(self.device)

        _discriminator = nn.Sequential(make_mlp([512, 512, 512]), nn.LazyLinear(1))
        self.amp_discriminator = _discriminator.to(self.device)

        self.encoder_priv(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)
        self.adapt_module(fake_input)
        self.actor_adapt(fake_input)
        self.amp_discriminator(fake_input[AMP_KEY])

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)

        self.apply(init_)
        self.adapt_ema = copy.deepcopy(self.adapt_module).requires_grad_(False)

        self.lr = cfg.lr_start
        if self.cfg.phase == "train":
            policy_params = [
                {"params": self.actor.parameters()},
                {"params": self.encoder_priv.parameters()},
            ]
        else:
            policy_params = [
                {"params": self.actor_adapt.parameters()},
            ]

        self.opt_policy = torch.optim.Adam(
            policy_params,
            lr=self.lr,
        )
        self.opt_critic = torch.optim.Adam(
            [
                {"params": self.critic.parameters()},
            ],
            lr=self.lr,
        )

        self.opt_adapt = torch.optim.Adam(
            [
                {"params": self.adapt_module.parameters()},
            ],
            lr=self.lr,
        )

        self.opt_discriminator = torch.optim.Adam(
            [
                {"params": self.amp_discriminator.parameters()},
            ],
            lr=cfg.amp.lr,
            weight_decay=cfg.amp.weight_decay,
        )
        if cfg.phase == "train" and cfg.enable_residual_distillation:
            self.opt_adapt_actor = torch.optim.Adam(
                [
                    {"params": self.actor_adapt.parameters()},
                ],
                lr=self.lr,
            )

        # setup AMP observation buffer
        from active_adaptation.utils.motion import MotionDataset
        from active_adaptation.learning.utils.amp_obs_buf import AMPObsBuffer

        motion_data_path = cfg.amp.motion_data_path
        motion_dataset = MotionDataset.create_from_path(motion_data_path).to(
            self.device
        )
        obs_cfg = self.env.cfg.observation[AMP_KEY]
        self.amp_obs_buf = AMPObsBuffer(motion_dataset, obs_cfg)
        # self.amp_obs_buf.export("amp_obs/expert")
        # breakpoint()

        # setup AMP normalizer
        from active_adaptation.learning.utils.vecnorm import VecNorm

        self.amp_normalizer = VecNorm([AMP_KEY], device=self.device)
        self.amp_normalizer.init_stats(fake_input)

        def amp_logits(amp_obs: torch.Tensor):
            amp_obs = self.amp_normalizer.normalize({AMP_KEY: amp_obs})[AMP_KEY]
            return self.amp_discriminator(amp_obs)

        self.amp_logits = amp_logits

        # setup AMP replay buffer
        if cfg.amp.replay_buffer_iters > 0:
            from active_adaptation.learning.utils.replay_buffer import ReplayBuffer

            replay_buffer_capacity = (
                cfg.amp.replay_buffer_iters * env.num_envs * cfg.train_every
            )
            self.amp_replay_buffer = ReplayBuffer(
                capacity=replay_buffer_capacity, device=self.device
            )
        else:
            self.amp_replay_buffer = None
        self.num_updates = 0

    def make_tensordict_primer(self):
        return TensorDictPrimer({}, reset_key="done")

    def get_rollout_policy(self, mode: str = "train"):
        modules = []

        if self.cfg.phase == "train":
            modules.append(self.encoder_priv)
            modules.append(self.actor)
            modules.append(self.adapt_module)
        elif self.cfg.phase == "adapt":
            modules.append(self.adapt_module)
            modules.append(self.actor_adapt)
        elif self.cfg.phase == "finetune":
            modules.append(self.adapt_ema)
            modules.append(self.actor_adapt)

        out_keys = ["sample_log_prob", "action"] + self.dist_keys
        if self.cfg.phase == "finetune":
            out_keys.append("priv_pred")
        policy = Seq(*modules, selected_out_keys=out_keys)
        return policy

    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.exclude("stats")
        info = {}
        if self.cfg.phase == "train":
            info.update(self.train_policy(tensordict.copy()))
            info.update(self.train_adapt(tensordict.copy()))
        elif self.cfg.phase == "adapt":
            info.update(self.train_adapt(tensordict.copy()))
        elif self.cfg.phase == "finetune":
            info.update(self.train_policy(tensordict.copy()))
            info.update(self.train_adapt(tensordict.copy()))
        self.num_updates += 1

        actor = self.actor if self.cfg.phase == "train" else self.actor_adapt
        action_std = actor.module[0][2].module.actor_std.detach()
        for joint_name, std in zip(self.joint_names, action_std):
            info[f"actor_std/{joint_name}"] = std
        info["actor_std/mean"] = action_std.mean()
        return info

    def train_policy(self, tensordict: TensorDict):
        infos = []
        self._compute_advantage(
            tensordict, self.critic, "adv", "ret", update_value_norm=True
        )

        if self.amp_replay_buffer is not None:
            self.amp_replay_buffer.insert(
                **tensordict.select(AMP_KEY).flatten(),
                random_replace=self.cfg.amp.random_replace,
            )

        # entropy coef schedule
        current_iter = self.env.current_iter
        entropy_progress = float(
            np.clip(current_iter / self.cfg.entropy_decay_iters, 0.0, 1.0)
        )
        self.entropy_coef = (
            self.cfg.entropy_coef_start
            + (self.cfg.entropy_coef_end - self.cfg.entropy_coef_start)
            * entropy_progress
        )

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                info = {}
                info.update(self._update_ppo(minibatch))
                info.update(self._update_amp(minibatch))
                infos.append(info)

                if self.desired_kl is not None:  # adaptive learning rate
                    kl = infos[-1]["actor/kl"]
                    if kl > self.desired_kl * 2.0:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif kl < self.desired_kl / 2.0 and kl > 0.0:
                        self.lr = min(1e-2, self.lr * 1.5)
                else:  # use manual linear schedule
                    lr_progress = float(
                        np.clip(current_iter / self.cfg.lr_decay_iters, 0.0, 1.0)
                    )
                    self.lr = (
                        self.cfg.lr_start
                        + (self.cfg.lr_end - self.cfg.lr_start) * lr_progress
                    )

                for param_group in self.opt_policy.param_groups:
                    param_group["lr"] = self.lr

        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.lr
        infos["actor/entropy_coef"] = self.entropy_coef

        ret = tensordict["ret"]
        ret_mean = ret.mean(dim=(0, 1))
        ret_std = ret.std(dim=(0, 1))
        for i, group_name in enumerate(self.reward_groups):
            infos[f"critic/{group_name}.ret_mean"] = ret_mean[i].item()
            infos[f"critic/{group_name}.ret_std"] = ret_std[i].item()
            infos[f"critic/{group_name}.neg_rew_ratio"] = (
                (tensordict[REWARD_KEY][:, :, i] <= 0.0).float().mean().item()
            )
        return dict(sorted(infos.items()))

    def train_adapt(self, tensordict: TensorDict):
        infos = []

        with torch.no_grad():
            self.encoder_priv(tensordict)

        for epoch in range(2):
            for minibatch in make_batch(
                tensordict, self.cfg.num_minibatches, self.cfg.train_every
            ):
                self.adapt_module(minibatch)
                priv_loss = self.adapt_loss_fn(
                    minibatch["priv_pred"], minibatch["priv_feature"]
                )
                priv_loss = (priv_loss * (~minibatch["is_init"])).mean()
                self.opt_adapt.zero_grad()
                priv_loss.backward()
                self.opt_adapt.step()
                info = {}
                info["adapt/priv_loss"] = priv_loss

                if self.cfg.phase == "train" and self.cfg.enable_residual_distillation:
                    # residual action distillation
                    with torch.no_grad():
                        dist_teacher = self.actor.get_dist(minibatch)
                        action_teacher = dist_teacher.mean

                    minibatch["priv_pred"] = minibatch["priv_feature"]
                    dist_student = self.actor_adapt.get_dist(minibatch)
                    action_student = dist_student.mean

                    adapt_loss = (action_student - action_teacher).square().mean()

                    self.opt_adapt_actor.zero_grad()
                    adapt_loss.backward()
                    self.opt_adapt_actor.step()
                    info["adapt/adapt_loss"] = adapt_loss

                infos.append(TensorDict(info, []))

        soft_copy_(self.adapt_module, self.adapt_ema, 0.04)

        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        return infos

    @torch.no_grad()
    def _compute_advantage(
        self,
        tensordict: TensorDict,
        critic: Mod,
        adv_key: str = "adv",
        ret_key: str = "ret",
        update_value_norm: bool = True,
    ):
        # with tensordict.view(-1) as tensordict_flat:
        #     critic(tensordict_flat)
        #     critic(tensordict_flat["next"])
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                critic(tensordict_flat)
                critic(tensordict_flat["next"])

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY]
        if self.cfg.clip_neg_reward:
            rewards = rewards.clamp_min(0.0)

        # compute AMP reward
        amp_logits = self.amp_logits(tensordict[AMP_KEY])
        if self.cfg.amp.gan_type == "gan":
            s = torch.sigmoid(amp_logits).clamp(1e-5, 1 - 1e-5)
            amp_reward = s.log() - (1 - s).log()
        elif self.cfg.amp.gan_type == "wgan":
            amp_reward = torch.tanh(amp_logits * self.cfg.amp.eta_wgan).exp()
        elif self.cfg.amp.gan_type == "lsgan":
            amp_reward = torch.clamp(
                1 - 0.25 * (1 - amp_logits).square(), min=0.0, max=1.0
            )

        rewards = torch.concat([rewards, amp_reward], dim=-1)
        tensordict.set(REWARD_KEY, rewards)

        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)

        # Compute and normalize the advantages
        # [num_steps, num_envs, num_reward_groups]
        if self.cfg.normalize_before_sum:  # normalize, scale, sum
            adv_norm = (adv - adv.mean(dim=(0, 1))) / (adv.std(dim=(0, 1)) + 1e-8)
            adv_norm *= self.reward_scales
            # [num_steps, num_envs, num_reward_groups]
            adv_norm_sum = adv_norm.sum(dim=2, keepdim=True)
            # [num_steps, num_envs, 1]
            adv_final = adv_norm_sum
        else:  # scale, sum, normalize
            adv *= self.reward_scales
            adv_sum = adv.sum(dim=2, keepdim=True)
            # [num_steps, num_envs, 1]
            adv_sum_norm = (adv_sum - adv_sum.mean(dim=(0, 1))) / (
                adv_sum.std(dim=(0, 1)) + 1e-8
            )
            # [num_steps, num_envs, 1]
            adv_final = adv_sum_norm

        if update_value_norm:
            self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv_final)
        # shape: (N, T, 1)
        tensordict.set(ret_key, ret)
        tensordict["adv_before_norm"] = adv
        # shape: (N, T, num_reward_groups)
        return tensordict

    # @torch.compile
    def _update_ppo(self, tensordict: TensorDict):
        dist_kwargs_old = tensordict.select(*self.dist_keys)

        if self.cfg.phase == "train":
            self.encoder_priv(tensordict)
            actor = self.actor
        else:
            actor = self.actor_adapt

        dist: D.Independent = actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        if self.cfg.phase == "train":
            valid = tensordict["step_count"] > 1
        else:
            valid = tensordict["step_count"] > 5
        valid = valid.squeeze(-1)

        adv = tensordict["adv"]
        log_ratio = (log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        policy_loss = -(torch.min(surr1, surr2)[valid]).mean()
        entropy_loss = -self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = value_loss[valid].mean(dim=0)

        loss = policy_loss + entropy_loss + value_loss.mean()

        self.opt_policy.zero_grad()
        self.opt_critic.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            actor.parameters(), self.cfg.max_grad_norm
        )
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.cfg.max_grad_norm
        )
        if self.cfg.phase == "train":
            priv_grad_norm = nn.utils.clip_grad_norm_(
                self.encoder_priv.parameters(), self.cfg.max_grad_norm
            )
        else:
            priv_grad_norm = torch.zeros(1)
        self.opt_policy.step()
        self.opt_critic.step()

        with torch.no_grad():
            explained_var = 1 - value_loss / b_returns[valid].var(dim=0)
            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            # loc, scale = dist.loc, dist.scale
            # kl = torch.sum(
            #     torch.log(scale) - torch.log(scale_old)
            #     + (torch.square(scale_old) + torch.square(loc_old - loc)) / (2.0 * torch.square(scale))
            #     - 0.5,
            #     dim=-1,
            # ).mean()
            dist_old = self.dist_cls(**dist_kwargs_old)
            kl = D.kl_divergence(dist_old, dist).mean()

        info = {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/mean_std": tensordict["scale"].detach().mean(),
            "actor/grad_norm": actor_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "actor/kl": kl,
            "actor/priv_grad_norm": priv_grad_norm,
            "actor/approx_kl": ((ratio - 1) - log_ratio).mean(),
            "critic/grad_norm": critic_grad_norm,
        }
        for i, group_name in enumerate(self.reward_groups):
            info[f"critic/{group_name}.explained_var"] = explained_var[i]
            info[f"critic/{group_name}.value_loss"] = value_loss[i].detach()
        return info

    def _update_amp(self, tensordict: TensorDict):
        minibatch_size = tensordict.shape[0]

        policy_amp_obs = tensordict[AMP_KEY]
        expert_amp_obs = self.amp_obs_buf.sample(minibatch_size)
        if self.amp_replay_buffer is not None:
            policy_amp_obs_replay = self.amp_replay_buffer.sample(minibatch_size)[
                AMP_KEY
            ]

        if "policy" in self.cfg.amp.amp_normalizer_source:
            self.amp_normalizer.update({AMP_KEY: policy_amp_obs})
        if "expert" in self.cfg.amp.amp_normalizer_source:
            self.amp_normalizer.update({AMP_KEY: expert_amp_obs})
        if "policy_replay" in self.cfg.amp.amp_normalizer_source:
            self.amp_normalizer.update({AMP_KEY: policy_amp_obs_replay})

        # compute discriminator loss
        expert_logits = self.amp_logits(expert_amp_obs)
        policy_logits = self.amp_logits(policy_amp_obs)

        if self.cfg.amp.gan_type == "gan":
            expert_loss = -torch.nn.functional.logsigmoid(expert_logits)
            policy_loss = torch.nn.functional.softplus(-policy_logits)
            discriminator_loss = expert_loss + policy_loss
        elif self.cfg.amp.gan_type == "wgan":
            expert_loss = -torch.tanh(expert_logits * self.cfg.amp.eta_wgan)
            policy_loss = torch.tanh(policy_logits * self.cfg.amp.eta_wgan)
            discriminator_loss = expert_loss + policy_loss
        elif self.cfg.amp.gan_type == "lsgan":
            expert_loss = (expert_logits - 1) ** 2
            policy_loss = (policy_logits + 1) ** 2
            discriminator_loss = expert_loss + policy_loss

        if self.amp_replay_buffer is not None:
            policy_logits_replay = self.amp_logits(policy_amp_obs_replay)
            if self.cfg.amp.gan_type == "gan":
                policy_loss_replay = torch.nn.functional.softplus(-policy_logits_replay)
                discriminator_loss += policy_loss_replay
            if self.cfg.amp.gan_type == "wgan":
                policy_loss_replay = torch.tanh(
                    policy_logits_replay * self.cfg.amp.eta_wgan
                )
                discriminator_loss += policy_loss_replay
            if self.cfg.amp.gan_type == "lsgan":
                policy_loss_replay = (policy_logits_replay + 1) ** 2
                discriminator_loss += policy_loss_replay
        discriminator_loss = discriminator_loss.mean()

        amp_grad_pen_inputs = []
        if "expert" in self.cfg.amp.grad_pen_source:
            amp_grad_pen_inputs.append(expert_amp_obs)
        if "policy" in self.cfg.amp.grad_pen_source:
            amp_grad_pen_inputs.append(policy_amp_obs)
        if "policy_replay" in self.cfg.amp.grad_pen_source:
            amp_grad_pen_inputs.append(policy_amp_obs_replay)
        if "interpolated" in self.cfg.amp.grad_pen_source:
            alpha = torch.rand(minibatch_size, 1, device=self.device)
            amp_grad_pen_inputs.append(
                alpha * expert_amp_obs + (1 - alpha) * policy_amp_obs
            )
        amp_grad_pen_input = torch.cat(amp_grad_pen_inputs, dim=0).requires_grad_(True)

        # compute gradient penalty
        grad_pen_score = self.amp_logits(amp_grad_pen_input)
        ones = torch.ones_like(grad_pen_score)
        grad = torch.autograd.grad(
            outputs=grad_pen_score,
            inputs=amp_grad_pen_input,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # enforce grad norm approaches 0/1
        disc_grad_norm = grad.norm(p=2, dim=1)
        gradient_penalty = (
            (disc_grad_norm - self.cfg.amp.grad_pen_target_norm).pow(2).mean()
        )

        loss = discriminator_loss + self.cfg.amp.grad_pen_weight * gradient_penalty
        self.opt_discriminator.zero_grad()
        loss.backward()
        self.opt_discriminator.step()

        # TODO: compute classification acc
        info = {
            "amp/discriminator_loss": discriminator_loss.detach(),
            "amp/grad_norm": disc_grad_norm.mean().detach(),
            "amp/gradient_penalty": gradient_penalty.detach(),
            "amp/expert_logit_mean": expert_logits.mean().detach(),
            "amp/policy_logit_mean": policy_logits.mean().detach(),
        }
        return info

    def state_dict(self):
        if self.cfg.phase == "train" and not self.cfg.enable_residual_distillation:
            hard_copy_(self.actor, self.actor_adapt)

        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        state_dict["last_phase"] = self.cfg.phase
        state_dict["last_iter"] = self.env.current_iter
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _state_dict = state_dict.get(name, {})
            try:
                module.load_state_dict(_state_dict, strict=strict)
                succeed_keys.append(name)
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")

        self.env.set_progress(state_dict.get("last_iter", 0))

        return failed_keys
