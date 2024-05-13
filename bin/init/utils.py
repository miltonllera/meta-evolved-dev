import os.path as osp
import logging
import yaml
from datetime import datetime
from hydra.core.hydra_config import HydraConfig

import numpy as np
import jax.random as jr

# from rich import get_console, table


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical"
    )

    return logger


def seed_from_timestamp():
    return int(datetime.now().timestamp())


def seed_everything(seed):
    if seed is None:
        seed = seed_from_timestamp()
    rng = np.random.default_rng(seed)
    return seed, jr.PRNGKey(seed), rng


def save_seed_to_config(seed):
    output_dir = HydraConfig.get()['runtime']['output_dir']
    cfg_file = osp.join(output_dir, ".hydra", "config.yaml")
    with open(cfg_file, "r+") as stream:
        cfg = yaml.load(stream, yaml.FullLoader)
        cfg['seed'] = seed
        stream.seek(0)
        stream.write(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        stream.truncate()
