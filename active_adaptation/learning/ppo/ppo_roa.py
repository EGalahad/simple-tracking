import torch
import torch.nn as nn
import torch.distributions as D
import warnings
import torch.utils._pytree as pytree
import copy

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import TensorDictPrimer
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union, List
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *

torch.set_float32_matmul_precision("high")

OBJECT_KEY = "object"
DEPTH_KEY = "depth"
REF_JPOS_KEY = "ref_joint_pos_"
PRIV_FEATURE_KEY = "priv_feature"
PRIV_PRED_KEY = "priv_pred"


@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_roa.PPOROA"
    name: str = "ppo_roa"
    train_every: int = 32
    ppo_epochs: int = 3
    num_minibatches: int = 8
    clip_param: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.95

    residual_action: bool = True

    # lr
    lr: float = 3e-4
    desired_kl: float | None = 0.01  # None

    # exploration
    entropy_coef: float = 0.004
    init_noise_scale: float = 1.0
    load_noise_scale: float | None = 0.5

    clip_neg_reward: bool = False
    normalize_before_sum: bool = False

    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    latent_dim: int = 256

    max_grad_norm: float = 1.0

    clip_adv: float | None = None
    phase: str = "train"
    vecnorm: Union[str, None] = None
    checkpoint_path: Union[str, None] = None
    in_keys: List[str] = (CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY)


cs = ConfigStore.instance()
cs.store(
    "ppo_roa_train",
    node=PPOConfig(phase="train", vecnorm="train", entropy_coef=0.004),
    group="algo",
)
cs.store(
    "ppo_roa_adapt",
    node=PPOConfig(phase="adapt", vecnorm="eval", entropy_coef=0.00),
    group="algo",
)
cs.store(
    "ppo_roa_finetune",
    node=PPOConfig(phase="finetune", vecnorm="eval", entropy_coef=0.004),
    group="algo",
)
cs.store(
    "ppo_roa_train_est",
    node=PPOConfig(
        phase="train_est",
        vecnorm="eval",
        entropy_coef=0.00,
        in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, DEPTH_KEY),
    ),
    group="algo",
)
cs.store(
    "ppo_roa_adapt_est",
    node=PPOConfig(
        phase="adapt_est",
        vecnorm="eval",
        entropy_coef=0.00,
        in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, DEPTH_KEY),
    ),
    group="algo",
)


class PPOROA(TensorDictModuleBase):
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
        assert self.cfg.phase in [
            "train",
            "adapt",
            "finetune",
            "train_est",
            "adapt_est",
        ]

        self.desired_kl = cfg.desired_kl
        self.clip_param = self.cfg.clip_param

        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.adapt_loss_fn = nn.MSELoss(reduction="none")
        self.rec_loss = nn.MSELoss(reduction="none")
        self.gae = GAE(gamma=self.cfg.gamma, lmbda=self.cfg.lmbda)
        self.reward_groups = list(env.cfg.reward.keys())
        num_reward_groups = len(self.reward_groups)
        self.reward_scales = torch.ones(num_reward_groups, device=self.device)
        self.reward_scales /= self.reward_scales.sum()
        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=num_reward_groups).to(self.device)
        object.__setattr__(self, "env", env)

        self.action_dim = action_spec.shape[-1]
        self.joint_names = env.action_manager.joint_names

        fake_input = observation_spec.zero()

        if observation_spec.get("command_", None) is not None:
            global CMD_KEY
            CMD_KEY = "command_"

        # build encoder, adapt module, critic
        encoder_priv_in_keys = [OBS_PRIV_KEY]
        adapt_module_in_keys = [OBS_KEY, CMD_KEY]
        critic_in_keys = [OBS_PRIV_KEY, OBS_KEY, CMD_KEY]
        if observation_spec.get(OBJECT_KEY, None) is not None:
            encoder_priv_in_keys.append(OBJECT_KEY)
            adapt_module_in_keys.append(OBJECT_KEY)
            critic_in_keys.append(OBJECT_KEY)

        latent_dim = self.cfg.latent_dim
        self.encoder_priv = Seq(
            CatTensors(
                encoder_priv_in_keys, "_encoder_priv_inp", del_keys=False, sort=False
            ),
            Mod(
                nn.Sequential(
                    make_mlp([latent_dim], norm=self.cfg.layer_norm),
                    nn.LazyLinear(latent_dim),
                ),
                "_encoder_priv_inp",
                PRIV_FEATURE_KEY,
            ),
            selected_out_keys=[PRIV_FEATURE_KEY],
        ).to(self.device)

        self.adapt_module = Seq(
            CatTensors(adapt_module_in_keys, "_adapt_inp", del_keys=False, sort=False),
            Mod(
                nn.Sequential(
                    make_mlp([latent_dim, latent_dim], norm=self.cfg.layer_norm),
                    nn.LazyLinear(latent_dim),
                ),
                "_adapt_inp",
                [PRIV_PRED_KEY],
            ),
            selected_out_keys=[PRIV_PRED_KEY],
        ).to(self.device)

        # build actor
        if cfg.phase == "train" and cfg.residual_action:
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
                Mod(
                    make_mlp([512, 256, 256], norm=self.cfg.layer_norm),
                    ["_actor_inp"],
                    ["_actor_feature"],
                ),
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

        in_keys = [CMD_KEY, OBS_KEY, PRIV_FEATURE_KEY]
        self.actor = build_actor(
            in_keys, self.dist_cls, self.dist_keys, residual_module=residual_module
        )
        if cfg.phase == "adapt_est":
            in_keys = [CMD_KEY, OBS_KEY, "priv_est"]
        else:
            in_keys = [CMD_KEY, OBS_KEY, PRIV_PRED_KEY]
        self.actor_adapt = build_actor(in_keys, self.dist_cls, self.dist_keys)

        # build critic
        _critic = nn.Sequential(
            make_mlp([512, 256, 128], norm=self.cfg.layer_norm),
            nn.LazyLinear(num_reward_groups),
        )
        self.critic = Seq(
            CatTensors(critic_in_keys, "_critic_input", del_keys=False),
            Mod(_critic, ["_critic_input"], ["state_value"]),
        ).to(self.device)

        # build estimator
        if self.cfg.phase in ["train_est", "adapt_est"]:
            assert (
                OBJECT_KEY in observation_spec
            ), f"{OBJECT_KEY} obs needed for estimator"
            assert (
                DEPTH_KEY in observation_spec
            ), f"{DEPTH_KEY} obs needed for estimator"

            mlp = make_mlp([latent_dim], norm=self.cfg.layer_norm)
            cnn = nn.Sequential(
                make_conv(num_channels=[8, 8, 8], activation=nn.Mish, kernel_sizes=5),
                nn.LazyLinear(64),
                nn.LayerNorm(64),
            )
            back_bone = make_mlp([latent_dim, latent_dim], norm=self.cfg.layer_norm)
            modules = [
                CatTensors(
                    [OBS_KEY, CMD_KEY], "_estimator_mlp_inp", del_keys=False, sort=False
                ),
                Mod(mlp, "_estimator_mlp_inp", ["_mlp"]),
                Mod(cnn, [DEPTH_KEY], ["_cnn"]),
                CatTensors(["_mlp", "_cnn"], "_estimator_inp", del_keys=False),
                Mod(back_bone, "_estimator_inp", "priv_est"),
            ]
            self.estimator = Seq(*modules, selected_out_keys=["priv_est"]).to(
                self.device
            )

            # mlp = make_mlp([latent_dim])
            # cnn = nn.Sequential(
            #     make_conv(
            #         num_channels=[8, 8, 8],
            #         activation=nn.Mish,
            #         kernel_sizes=5
            #     ),
            #     nn.LazyLinear(64),
            #     nn.LayerNorm(64)
            # )
            # back_bone = make_mlp([latent_dim, latent_dim])
            # mlp_out = mlp(observation)
            # cnn_out = cnn(depth)
            # feature_est = back_bone(torch.cat([mlp_out, cnn_out], dim=-1))

            self.dr_estimator = Mod(mlp, [PRIV_PRED_KEY], ["dr_pred"]).to(self.device)

        self.encoder_priv(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)
        self.adapt_module(fake_input)
        if self.cfg.phase in ["train_est", "adapt_est"]:
            self.estimator(fake_input)
        self.actor_adapt(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)

        self.apply(init_)
        self.adapt_ema = copy.deepcopy(self.adapt_module).requires_grad_(False)

        self.lr_policy = cfg.lr
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
            lr=self.lr_policy,
        )
        self.opt_critic = torch.optim.Adam(
            [
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr,
        )

        self.opt_adapt = torch.optim.Adam(
            [
                {"params": self.adapt_module.parameters()},
            ],
            lr=cfg.lr,
        )
        if cfg.phase == "train" and cfg.residual_action:
            self.opt_adapt_actor = torch.optim.Adam(
                [
                    {"params": self.actor_adapt.parameters()},
                ],
                lr=cfg.lr,
            )
        if cfg.phase in ["train_est", "adapt_est"]:
            self.opt_estimator = torch.optim.Adam(
                [
                    {"params": self.estimator.parameters()},
                ],
                lr=cfg.lr,
            )
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
        elif self.cfg.phase == "train_est":
            modules.append(self.adapt_ema)
            modules.append(self.actor_adapt)
        elif self.cfg.phase == "adapt_est":
            modules.append(self.estimator)
            modules.append(self.actor_adapt)

        out_keys = ["sample_log_prob", "action"] + self.dist_keys
        if self.cfg.phase == "finetune":
            out_keys.append(PRIV_PRED_KEY)
        if self.cfg.phase == "adapt_est":
            out_keys.append("priv_est")

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
        elif self.cfg.phase == "train_est":
            info.update(self.train_estimator(tensordict.copy()))
        elif self.cfg.phase == "adapt_est":
            info.update(self.train_policy(tensordict.copy()))
            info.update(self.train_estimator(tensordict.copy()))

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

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                info = {}
                info.update(self._update_ppo(minibatch))
                infos.append(info)

                if self.desired_kl is not None:  # adaptive learning rate
                    kl = infos[-1]["actor/kl"]
                    if kl > self.desired_kl * 2.0:
                        self.lr_policy = max(1e-5, self.lr_policy / 1.5)
                    elif kl < self.desired_kl / 2.0 and kl > 0.0:
                        self.lr_policy = min(1e-2, self.lr_policy * 1.5)

                for param_group in self.opt_policy.param_groups:
                    param_group["lr"] = self.lr_policy

        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.lr_policy
        infos["actor/entropy_coef"] = self.cfg.entropy_coef

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
                    minibatch[PRIV_PRED_KEY], minibatch[PRIV_FEATURE_KEY]
                )
                priv_loss = (priv_loss * (~minibatch["is_init"])).mean()
                self.opt_adapt.zero_grad()
                priv_loss.backward()
                opt_adapt_grad_norm = nn.utils.clip_grad_norm_(
                    self.adapt_module.parameters(), self.cfg.max_grad_norm
                )
                self.opt_adapt.step()
                info = {}
                info["adapt/priv_loss"] = priv_loss
                info["adapt/grad_norm"] = opt_adapt_grad_norm
                info["adapt/priv_feature_norm"] = (
                    minibatch[PRIV_FEATURE_KEY].norm(p=2, dim=-1).mean()
                )
                info["adapt/priv_pred_norm"] = (
                    minibatch[PRIV_PRED_KEY].norm(p=2, dim=-1).mean()
                )

                if self.cfg.phase == "train" and self.cfg.residual_action:
                    # residual action distillation
                    with torch.no_grad():
                        dist_teacher = self.actor.get_dist(minibatch)

                    minibatch[PRIV_PRED_KEY] = minibatch[PRIV_FEATURE_KEY].detach()
                    dist_student = self.actor_adapt.get_dist(minibatch)

                    adapt_loss = (dist_teacher.mean - dist_student.mean).square().mean()

                    self.opt_adapt_actor.zero_grad()
                    adapt_loss.backward()
                    self.opt_adapt_actor.step()
                    info["adapt/adapt_loss"] = adapt_loss

                infos.append(TensorDict(info, []))

        soft_copy_(self.adapt_module, self.adapt_ema, 0.04)

        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        return infos

    def train_estimator(self, tensordict: TensorDict):
        infos = []

        with torch.no_grad():
            self.adapt_ema(tensordict)

        for epoch in range(2):
            for minibatch in make_batch(
                tensordict, self.cfg.num_minibatches, self.cfg.train_every
            ):
                # minibatch: shape (num_envs / num_minibatches, train_every, ...)
                self.estimator(minibatch)
                est_loss = self.adapt_loss_fn(
                    minibatch["priv_est"], minibatch[PRIV_PRED_KEY]
                )
                est_loss = (est_loss * (~minibatch["is_init"])).mean()
                self.opt_estimator.zero_grad()
                est_loss.backward()
                opt_estimator_grad_norm = nn.utils.clip_grad_norm_(
                    self.estimator.parameters(), self.cfg.max_grad_norm
                )
                self.opt_estimator.step()

                info = {}
                info["estimator/est_loss"] = est_loss
                info["estimator/grad_norm"] = opt_estimator_grad_norm
                info["estimator/priv_est_norm"] = (
                    minibatch["priv_est"].norm(p=2, dim=-1).mean()
                )
                info["estimator/priv_pred_norm"] = (
                    minibatch[PRIV_PRED_KEY].norm(p=2, dim=-1).mean()
                )
                infos.append(TensorDict(info, []))
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
        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)

        # Compute and normalize the advantages
        # [num_steps, num_envs, num_reward_groups]
        if self.cfg.normalize_before_sum:  # normalize, scale, sum
            adv_norm = (adv - adv.mean(dim=(0, 1))) / (adv.std(dim=(0, 1)) + 0.01)
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
        elif self.cfg.phase == "finetune":
            actor = self.actor_adapt
        elif self.cfg.phase == "adapt_est":
            actor = self.actor_adapt
        else:
            raise ValueError(f"Invalid phase: {self.cfg.phase}")

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
        entropy_loss = -self.cfg.entropy_coef * entropy

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

    def state_dict(self):
        if self.cfg.phase == "train":
            if not self.cfg.residual_action:
                hard_copy_(self.actor, self.actor_adapt)
            else:
                actor_std = self.actor.module[0][2].module.actor_std
                actor_adapt_std = self.actor_adapt.module[0][2].module.actor_std
                actor_adapt_std.data.copy_(actor_std.data)

        if self.cfg.phase in ["train", "adapt"]:
            hard_copy_(self.adapt_module, self.adapt_ema)

        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        state_dict["last_phase"] = self.cfg.phase
        state_dict["last_iter"] = self.env.current_iter
        state_dict["lr_policy"] = self.lr_policy
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
        lr_policy = state_dict.get("lr_policy", None)
        if lr_policy is not None:
            self.lr_policy = lr_policy
            for param_group in self.opt_policy.param_groups:
                param_group["lr"] = self.lr_policy

        return failed_keys
