import torch
import hydra
import numpy as np
import einops
import time
import sys
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

import wandb
import logging
from tqdm import tqdm
from scripts.helpers import make_env_policy, evaluate

import os
import datetime
import termcolor


@hydra.main(config_path="../cfg", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    env, agent, vecnorm = make_env_policy(cfg)

    keys = [
        ("next", "stats"),
        ("next", "done"),
        ("next", "reward"),
        "value_obs",
        "value_priv",
        "value_adapt",
        "context_expert",
        "context_scale",
        "context_adapt",
        "context_adapt_scale",
        "action_kl",
    ]

    policy_keys = ["dr_", "dr_pred"]

    policy_eval = agent.get_rollout_policy("eval")
    render_mode = cfg.get("render_mode", "rgb_array")
    info, trajs, stats, policy_trajs = evaluate(
        env,
        policy_eval,
        render=cfg.eval_render,
        render_mode=render_mode,
        seed=cfg.seed,
        keys=keys,
        policy_keys=policy_keys,
    )

    # print(termcolor.colored(trajs, "light_yellow"))
    # time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    # path = os.path.join(os.path.dirname(__file__), f"trajs-{time_str}.pt")
    # torch.save(trajs, path)

    torch.save(
        policy_trajs, os.path.join(os.path.dirname(__file__), f"policy_trajs.pt")
    )

    # path = os.path.join(os.path.dirname(__file__), f"stats-{time_str}.pt")
    # torch.save(stats, path)

    info["task"] = cfg.task.name
    info["algo"] = cfg.algo.name
    info["checkpoint_path"] = cfg.checkpoint_path
    info["argv"] = sys.argv
    print(OmegaConf.to_yaml(info))

    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    dir_path = os.path.join(os.path.dirname(__file__), f"eval", cfg.task.name)
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"{cfg.task.name}-{time_str}.yaml")
    with open(path, "w") as f:
        OmegaConf.save(info, f)
    print(termcolor.colored(f"Saved results to: {path}", "green"))

    env.close()


if __name__ == "__main__":
    main()
