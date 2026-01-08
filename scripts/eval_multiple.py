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

    # --- 1. Parse Checkpoint Path ---
    base_checkpoint_path = cfg.checkpoint_path
    cfg.checkpoint_path = None
    checkpoint_steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
    # checkpoint_steps = [200, 600, 1000]

    env, agent, vecnorm = make_env_policy(cfg)

    # --- 3. Evaluation Loop for Each Checkpoint ---
    all_results = {}

    for step in tqdm(checkpoint_steps, desc="Evaluating Checkpoints"):
        print(
            termcolor.colored(
                f"\n===== Evaluating Checkpoint Step: {step} =====", "cyan"
            )
        )

        # # Construct the full path for the current checkpoint
        # current_checkpoint_path = f"{base_checkpoint_path}:{step}"

        # Download and load the checkpoint from wandb
        try:
            # Use a temporary name for the downloaded file
            wandb_run = wandb.Api().run(base_checkpoint_path.replace("run:", ""))
            file = wandb_run.file(f"checkpoint_{step}.pt")

            # Use a temporary directory for downloading
            temp_dir = "temp_checkpoints"
            os.makedirs(temp_dir, exist_ok=True)
            checkpoint_file = file.download(root=temp_dir, replace=True)
            state_dict = torch.load(
                checkpoint_file.name, map_location=env.device, weights_only=False
            )
            checkpoint_file.close()  # Close the file handle
            print(
                termcolor.colored(
                    f"Successfully loaded checkpoint {step} from wandb.", "green"
                )
            )

            # Load the state dict into the policy
            agent.load_state_dict(state_dict["policy"])
            vecnorm.load_state_dict(state_dict["vecnorm"])
            new_observation_norms = vecnorm.to_observation_norm().transforms

            from torchrl.envs.transforms import Compose, ObservationNorm

            new_transforms_list = []
            for transform in env.transform:
                if not isinstance(transform, ObservationNorm):
                    new_transforms_list.append(transform.clone())
            new_transforms_list.extend(new_observation_norms)
            env.transform = Compose(*new_transforms_list)

        except Exception as e:
            print(
                termcolor.colored(
                    f"Failed to download or load checkpoint {step}. Error: {e}", "red"
                )
            )
            continue

        # Define keys for data collection during rollout
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

        # Get the evaluation policy and run the evaluation
        policy_eval = agent.get_rollout_policy("eval")
        render_mode = cfg.get("render_mode", "rgb_array")

        # We can disable rendering for multiple evaluations to speed it up
        info, trajs, stats, policy_trajs = evaluate(
            env,
            policy_eval,
            render=cfg.eval_render,
            render_mode=render_mode,
            seed=cfg.seed,
            keys=keys,
            policy_keys=policy_keys,
        )

        # Store info for this policy
        info["task"] = cfg.task.name
        info["algo"] = cfg.algo.name
        info["checkpoint_step"] = step

        all_results[f"checkpoint_{step}"] = info
        print(termcolor.colored(f"--- Results for step {step} ---", "yellow"))
        print(OmegaConf.to_yaml(info))

    # --- 4. Print and Save All Collected Info ---
    print(termcolor.colored("\n\n===== All Evaluation Results =====", "magenta"))
    # Convert to a dict for clean YAML output
    final_output = OmegaConf.create(all_results)
    print(OmegaConf.to_yaml(final_output))

    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    dir_path = os.path.join(os.path.dirname(__file__), "eval_multiple", cfg.task.name)
    os.makedirs(dir_path, exist_ok=True)

    # Extract run ID for a more descriptive filename
    run_id = base_checkpoint_path.split("/")[-1]
    path = os.path.join(dir_path, f"{cfg.task.name}-{run_id}-{time_str}.yaml")

    with open(path, "w") as f:
        OmegaConf.save(config=final_output, f=f)
    print(termcolor.colored(f"Saved all results to: {path}", "green"))

    # --- 5. Cleanup ---
    env.close()


if __name__ == "__main__":
    main()
