import datetime
import logging
import os

import wandb
from omegaconf import OmegaConf
from typing import Union


def dict_flatten(a: dict, delim="."):
    """Flatten a dict recursively.
    Examples:
        >>> a = {
                "a": 1,
                "b":{
                    "c": 3,
                    "d": 4,
                    "e": {
                        "f": 5
                    }
                }
            }
        >>> dict_flatten(a)
        {'a': 1, 'b.c': 3, 'b.d': 4, 'b.e.f': 5}
    """
    result = {}
    for k, v in a.items():
        if isinstance(v, dict):
            result.update({k + delim + kk: vv for kk, vv in dict_flatten(v).items()})
        else:
            result[k] = v
    return result


def init_wandb(cfg):
    """Initialize WandB.

    If only `run_id` is given, resume from the run specified by `run_id`.
    If only `run_path` is given, start a new run from that specified by `run_path`,
        possibly restoring trained models.

    Otherwise, start a fresh new run.

    """
    wandb_cfg = cfg.wandb
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    run_name = f"{wandb_cfg.run_name}/{time_str}"
    kwargs = dict(
        project=wandb_cfg.project,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        name=run_name,
        mode=wandb_cfg.mode,
        tags=wandb_cfg.tags,
    )
    if wandb_cfg.run_id is not None:
        kwargs["id"] = wandb_cfg.run_id
        kwargs["resume"] = "must"
    else:
        kwargs["id"] = wandb.util.generate_id()
    run = wandb.init(**kwargs)
    cfg_dict = dict_flatten(OmegaConf.to_container(cfg))
    run.config.update(cfg_dict)
    return run


def parse_checkpoint_path(path: str = None):
    """
    Parse a checkpoint path from local or wandb.
    If `path` is of the form `run:<wandb_run_id>`, it will be downloaded from wandb.
    If `path` is of the form `run:<wandb_run_id>:<checkpoint_num>`, it will download the specific checkpoint.

    Args:
        path (str or None): Path to a checkpoint.

    Returns:
        str: Path to the checkpoint.
    """
    if path is None:
        return None

    if path.startswith("run:"):
        # Parse the run path and optional checkpoint number
        parts = path[4:].split(":")
        run_path = parts[0]
        target_checkpoint_num = parts[1] if len(parts) > 1 else None

        api = wandb.Api()
        run = api.run(run_path)
        root = os.path.join(os.path.dirname(__file__), "wandb", run.name)
        os.makedirs(root, exist_ok=True)

        checkpoints = []
        for file in run.files():
            print(file.name)
            if "checkpoint" in file.name:
                checkpoints.append(file)
            elif file.name == "files/cfg.yaml":
                file.download(root, replace=True)

        def sort_by_time(file):
            number_str = file.name[:-3].split("_")[-1]
            if number_str == "final":
                return 100000
            else:
                return int(number_str)

        checkpoints.sort(key=sort_by_time)

        # If a specific checkpoint number is requested, find it
        if target_checkpoint_num is not None:
            target_checkpoint = None
            for checkpoint in checkpoints:
                number_str = checkpoint.name[:-3].split("_")[-1]
                if number_str == target_checkpoint_num:
                    target_checkpoint = checkpoint
                    break

            if target_checkpoint is None:
                available_nums = [f.name[:-3].split("_")[-1] for f in checkpoints]
                raise ValueError(
                    f"Checkpoint {target_checkpoint_num} not found. Available checkpoints: {available_nums}"
                )

            checkpoint = target_checkpoint
        else:
            # Use the latest checkpoint (existing behavior)
            checkpoint = checkpoints[-1]

        path = os.path.join(root, checkpoint.name)
        print(f"Downloading checkpoint to {path}")
        checkpoint.download(root, replace=True)
    return path
