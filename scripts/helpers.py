import torch
import torch.nn as nn
import hydra
import numpy as np
import time
import wandb
import logging
import os
import datetime

from typing import Sequence, List, Tuple, TYPE_CHECKING
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModuleBase as ModBase
from torchrl.envs.transforms import VecNorm

from termcolor import colored
from collections import OrderedDict
import imageio
from omegaconf import OmegaConf, DictConfig
import active_adaptation.learning
from active_adaptation.utils.wandb import parse_checkpoint_path
import active_adaptation

if TYPE_CHECKING:
    from active_adaptation.envs.base import _Env


class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1


class ObsNorm(ModBase):
    def __init__(self, in_keys, out_keys, locs, scales):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys

        self.loc = nn.ParameterDict({k: nn.Parameter(locs[k]) for k in in_keys})
        self.scale = nn.ParameterDict({k: nn.Parameter(scales[k]) for k in out_keys})
        self.requires_grad_(False)

    def forward(self, tensordict: TensorDictBase):
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            obs = tensordict.get(in_key, None)
            if obs is not None:
                loc = self.loc[in_key]
                scale = self.scale[out_key]
                tensordict.set(out_key, (obs - loc) / scale)
        return tensordict

    @classmethod
    def from_vecnorm(cls, vecnorm: VecNorm, keys):
        in_keys = []
        out_keys = []
        for in_key, out_key in zip(vecnorm.in_keys, vecnorm.out_keys):
            if in_key in keys:
                in_keys.append(in_key)
                out_keys.append(out_key)
        return cls(
            in_keys=in_keys, out_keys=out_keys, locs=vecnorm.loc, scales=vecnorm.scale
        )


class ObsOODDetector(ModBase):
    def __init__(self, in_keys, sigma=5.0):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = [("next", f"{k}_ood_ratio") for k in in_keys] + [
            ("next", k) for k in in_keys
        ]
        self.sigma = sigma

    def forward(self, tensordict: TensorDictBase):
        for in_key in self.in_keys:
            obs = tensordict.get(in_key, None)
            if obs is not None:
                ood_ratio = (
                    (obs.abs() > self.sigma).float().mean(dim=tuple(range(1, obs.ndim)))
                )
                tensordict.set(("next", f"{in_key}_ood_ratio"), ood_ratio)
                tensordict.set(("next", in_key), obs)
        return tensordict


class EpisodeStats:
    def __init__(self, in_keys: Sequence[str], device: torch.device):
        self.in_keys = in_keys
        self.device = device
        self._stats = TensorDict(
            {key: torch.tensor([0.0], device=device) for key in in_keys}, [1]
        )
        self._episodes = torch.tensor(0, device=device)

    def add(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_tensordict = tensordict["next"]
        done = next_tensordict["done"]
        if done.any():
            done = done.squeeze(-1)
            next_tensordict = next_tensordict.select(*self.in_keys)
            self._stats = self._stats + next_tensordict[done].sum(dim=0)
            self._episodes += done.sum()
        return len(self)

    def pop(self):
        stats = self._stats / self._episodes
        self._stats.zero_()
        self._episodes.zero_()
        return stats.cpu()

    def __len__(self):
        return self._episodes.item()


def make_env_policy(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    from active_adaptation.envs import SimpleEnv
    from torchrl.envs.transforms import (
        TransformedEnv,
        Compose,
        InitTracker,
        VecNorm,
        StepCounter,
    )

    policy_in_keys = cfg.algo.get("in_keys", ["policy", "priv"])

    for obs_group_key in list(cfg.task.observation.keys()):
        if obs_group_key not in policy_in_keys and not obs_group_key.endswith("_"):
            cfg.task.observation.pop(obs_group_key)
            print(
                colored(
                    f"Discard obs group {obs_group_key} as it is not used.", "yellow"
                )
            )

    base_env = SimpleEnv(cfg.task)

    checkpoint_path = parse_checkpoint_path(cfg.checkpoint_path)
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, weights_only=False)
    else:
        state_dict = {}

    obs_keys = [
        key
        for key, spec in base_env.observation_spec.items(True, True)
        if not (spec.dtype == bool or key.endswith("_"))
    ]
    transform = Compose(InitTracker(), StepCounter())

    assert cfg.vecnorm in ("train", "eval", None)
    print(colored(f"[Info]: create VecNorm for keys: {obs_keys}", "green"))
    vecnorm = VecNorm(obs_keys, decay=0.9999)
    vecnorm(base_env.fake_tensordict())

    if "vecnorm" in state_dict.keys():
        print(colored("[Info]: Load VecNorm from checkpoint.", "green"))
        vecnorm.load_state_dict(state_dict["vecnorm"])
    if cfg.vecnorm == "train":
        print(colored("[Info]: Updating obervation normalizer.", "green"))
        transform.append(vecnorm)
    elif cfg.vecnorm == "eval":
        print(colored("[Info]: Not updating obervation normalizer.", "green"))
        transform.append(vecnorm.to_observation_norm())
    elif cfg.vecnorm is not None:
        raise ValueError

    env = TransformedEnv(base_env, transform)
    env.set_seed(cfg.seed)

    # setup policy
    policy_cls = hydra.utils.get_class(cfg.algo._target_)
    active_adaptation.print(f"Creating policy {policy_cls} on device {base_env.device}")
    policy: ModBase = policy_cls(
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device,
        env=env,
    )

    if "policy" in state_dict.keys():
        print(colored("[Info]: Load policy from checkpoint.", "green"))
        policy.load_state_dict(state_dict["policy"])

    if hasattr(policy, "make_tensordict_primer"):
        primer = policy.make_tensordict_primer()
        print(colored(f"[Info]: Add TensorDictPrimer {primer}.", "green"))
        transform.append(primer)
        env = TransformedEnv(env.base_env, transform)
    env: _Env

    return env, policy, vecnorm


from torchrl.envs import TransformedEnv, ExplorationType, set_exploration_type
from tqdm import tqdm


@torch.inference_mode()
def evaluate(
    env: TransformedEnv,
    policy: torch.nn.Module,
    seed: int = 0,
    exploration_type: ExplorationType = ExplorationType.MODE,
    # exploration_type: ExplorationType=ExplorationType.RANDOM,
    render=False,
    render_mode="rgb_array",
    keys=[("next", "stats")],
    policy_keys=[],
):
    """
    Evaluate the policy on the environment, selecting `keys` from the trajectory.
    If `render` is True, record and save the video.
    """
    keys = ["ref_motion_phase_", "step_count"]
    keys = set(keys)
    keys.add(("next", "done"))
    keys.add(("next", "stats"))

    env.base_env.eval()
    env.eval()
    env.set_seed(seed)

    tensordict_ = env.reset()
    trajs = []
    frames = []
    policy_trajs = []

    inference_time = []
    torch.compiler.cudagraph_mark_step_begin()
    with set_exploration_type(exploration_type):
        for i in tqdm(range(env.max_episode_length), miniters=10):
            s = time.perf_counter()
            tensordict_ = policy(tensordict_)
            e = time.perf_counter()
            inference_time.append(e - s)

            policy_trajs.append(tensordict_.select(*policy_keys, strict=False).cpu())
            tensordict, tensordict_ = env.step_and_maybe_reset(tensordict_)
            trajs.append(tensordict.select(*keys, strict=False).cpu())

            if render:
                frames.append(env.render(mode=render_mode))
    inference_time = np.mean(inference_time[5:])
    print(f"Average inference time: {inference_time:.4f} s")

    policy_trajs: TensorDictBase = torch.stack(policy_trajs, dim=1)
    trajs: TensorDictBase = torch.stack(trajs, dim=1)
    done = trajs.get(("next", "done"))
    episode_cnt = len(done.nonzero())
    first_done = torch.argmax(done.long(), dim=1).cpu()

    def take_first_episode(tensor: torch.Tensor):
        indices = first_done.reshape(first_done.shape + (1,) * (tensor.ndim - 2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    info = {}
    stats = {}
    episode_len = take_first_episode(trajs["next", "stats", "episode_len"])
    # shape: (num_envs,)
    for k, v in trajs["next", "stats"].items(True, True):
        v = take_first_episode(v)
        if k == "episode_len" or k == "success":
            pass
        else:
            v = v.float() / episode_len.float()

        key = "eval/" + ("/".join(k) if isinstance(k, tuple) else k)
        stats[key] = v
        info[key] = torch.mean(v.float()).item()
        info[key + "_std"] = torch.std(v.float()).item()

    # log video
    if len(frames):
        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        video_array = np.stack(frames)
        frames.clear()
        video_path = os.path.join(
            os.path.dirname(__file__), f"recording-{time_str}.mp4"
        )
        imageio.mimwrite(
            video_path, video_array, fps=int(1 / env.step_dt), codec="libx264"
        )

    info["episode_cnt"] = episode_cnt
    return dict(sorted(info.items())), trajs, stats, policy_trajs


def extract_episodes(trajs: TensorDictBase) -> List[TensorDictBase]:
    """
    将一个包含多个环境和时间步的批次化 TensorDict 分割成一个列表,
    其中每个元素都是一个独立的、完整的 episode。

    Args:
        trajs (TensorDictBase): 一个形状为 (N, T, ...) 的 TensorDict,
            其中 N 是环境数量, T 是时间步数。
            这个 TensorDict 必须包含键 ("next", "done")。

    Returns:
        List[TensorDictBase]: 一个 TensorDict 的列表。列表中的每个 TensorDict
            代表一个完整的 episode,其形状为 (t, ...), t 是该 episode 的长度。
    """
    # 验证输入 TensorDict 的维度是否正确 (N, T)
    if trajs.batch_dims != 2:
        raise ValueError(
            f"输入的 trajs 应该有两个批次维度 (N, T), 但得到了 {trajs.batch_dims} 个。"
        )

    # 获取 done 信号, 形状为 (N, T)
    # 使用 .squeeze() 以防 done 信号的形状是 (N, T, 1)
    dones = trajs.get(("next", "done")).squeeze(-1)
    if dones.ndim != 2:
        raise ValueError(
            f"期望 ('next', 'done') 是一个二维张量, 但其形状为 {dones.shape}"
        )

    N, T = dones.shape

    all_episodes = []

    # 遍历每一个环境
    for i in range(N):
        # 找到当前环境中所有 done=True 的时间步索引
        # torch.where 返回一个元组, 我们需要第一个元素
        done_indices = torch.where(dones[i])[0]

        start_idx = 0
        # 遍历这些结束点, 切分出每一个 episode
        for end_idx in done_indices:
            # 切片是左闭右开, 所以我们需要 end_idx + 1 来包含结束的那一帧
            episode = trajs[i, start_idx : end_idx + 1]
            all_episodes.append(episode)

            # 更新下一个 episode 的起始点
            start_idx = end_idx + 1

    return all_episodes


def evaluate_track(trajs: TensorDictBase) -> Tuple[TensorDictBase, TensorDictBase]:
    trajs_tracking_info = trajs.select(
        "ref_motion_phase_", "step_count", ("next", "done")
    )
    episodes = extract_episodes(trajs_tracking_info)
    init_ref_motion_phase = []
    final_step_count = []
    for episode in episodes:
        init_ref_motion_phase.append(episode["ref_motion_phase_"][1].item())
        final_step_count.append(episode["step_count"][-1].item())

    # use matplotlib to plot the init_ref_motion_phase and final_step_count
    # use scatter plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.scatter(init_ref_motion_phase, final_step_count, alpha=0.5)
    plt.xlabel("Initial Reference Motion Phase")
    plt.ylabel("Final Step Count")
    plt.grid()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.dirname(__file__), "init_ref_motion_phase_vs_final_step_count.png"
        )
    )
    plt.close()

    breakpoint()


def plot_obs_histogram(
    trajs: TensorDictBase,
):
    trajs_obs: TensorDictBase = trajs.flatten(0, 1).select("command", "policy")
    policy_obs_np = trajs_obs.numpy()["policy"]
    # use matplotlib to plot the histgram of each dimension of trajs_obs_np
    num_cols = 15
    num_rows = (policy_obs_np.shape[-1] + num_cols - 1) // num_cols
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    for i in range(policy_obs_np.shape[-1]):
        ax = axes[i // num_cols, i % num_cols]
        ax.hist(policy_obs_np[:, i], bins=50)
        # plot mean for this dimension
        mean = np.mean(policy_obs_np[:, i])
        std = np.std(policy_obs_np[:, i])
        ax.axvline(mean, color="red", linestyle="dashed", linewidth=1)
        ax.axvline(mean + std, color="green", linestyle="dashed", linewidth=1)
        ax.axvline(mean - std, color="green", linestyle="dashed", linewidth=1)
        ax.set_title(f"Dim {i}")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "trajs_obs_hist.png"))
    plt.close()
