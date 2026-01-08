import torch
import hydra
import numpy as np
import einops
import time
import sys
from tqdm import tqdm
from omegaconf import OmegaConf

import wandb
import logging
from tqdm import tqdm
from scripts.helpers import make_env_policy, evaluate

import os
import datetime
import termcolor


@hydra.main(config_path="../cfg", config_name="render", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    env, agent, vecnorm = make_env_policy(cfg)

    policy_eval = agent.get_rollout_policy("eval")
    evaluate(
        env,
        policy_eval,
        render=cfg.eval_render,
        render_mode=cfg.render_mode,
        seed=cfg.seed,
    )
    env.close()


if __name__ == "__main__":
    main()
