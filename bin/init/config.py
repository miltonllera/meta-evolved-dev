import os
import os.path as osp
# import logging
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import jax.random as jr
import equinox as eqx
import hydra
from hydra.utils import get_class, get_method
from omegaconf import DictConfig, OmegaConf
from jaxtyping import PyTree

# from bin.extra.analysis import Analysis
# from bin.extra.visualization import Visualzation
from src.task.base import Task
from src.trainer.base import Trainer
from src.trainer.callback import Callback
from src.trainer.logger import Logger
from .utils import get_logger, seed_from_timestamp


_log = get_logger(__name__)


INSTANTIATED_RUN_MODULES = Tuple[Trainer, eqx.Module]


# Always convert to base python classes
instantiate = partial(hydra.utils.instantiate, _convert_="partial")

# Add custom resolvers

# Note: it should be possible to do this with the 'eval' resolver, but I get an error
OmegaConf.register_new_resolver(name="sum", resolver=lambda x, y: x + y)
OmegaConf.register_new_resolver(name="prod", resolver=lambda x, y: x * y)
OmegaConf.register_new_resolver(name="get_cls", resolver=lambda cls: get_class(cls))
OmegaConf.register_new_resolver(name="get_fn", resolver=lambda fn: get_method(fn))

# Equinox modules need a random key at initialization. We'll use a resolver to provide them.
def get_key_array():
    rand = np.random.randint(0, 2 ** 32 - 1)
    return jr.PRNGKey(rand)

OmegaConf.register_new_resolver(name="prng_key", resolver=get_key_array)
OmegaConf.register_new_resolver(name="seed_from_timestamp", resolver=seed_from_timestamp)


#--------------------------------------- Runs -----------------------------------------------

def instantiate_run(cfg) -> INSTANTIATED_RUN_MODULES:
    # instantiate model first so that prng_key value can be recoverd from experiment seed.
    model = instantiate_model(cfg.model)
    if cfg.ckpt_path is not None:
        weights = load_model_weights(model, *osp.split(cfg.ckpt_path))
        model = eqx.combine(weights, model)

    task = instantiate_task(cfg.task)
    callbacks = instantiate_callbacks(cfg.callbacks)
    loggers = instantiate_loggers(cfg.logger)
    trainer = instantiate_trainer(cfg.trainer, task=task, callbacks=callbacks, logger=loggers)
    return trainer, model


def instantiate_task(task_cfg) -> Task:
    _log.info(
        f"Initializing task <{task_cfg._target_}> ..."
    )
    datamodule: Task = instantiate(task_cfg)
    return datamodule


def instantiate_model(model_cfg) -> eqx.Module:
    _log.info(f"Initializing model <{model_cfg._target_}>...")
    model: eqx.Module = instantiate(model_cfg)
    return model


def instantiate_trainer(
    trainer_cfg: DictConfig,
    task: Task,
    callbacks: List[Callback],
    logger: List[Logger],
) -> Trainer:
    if "strategy" in trainer_cfg:
        _log.info(f"Initializing trainer <{trainer_cfg._target_}> "
            f"with <{trainer_cfg.strategy.strategy}> evolutionary strategy...")

    # use partial to avoid hydra converting PyTrees to dicts
    trainer: Trainer = instantiate(trainer_cfg, _partial_=True)(
        task=task, callbacks=callbacks, logger=logger
    )

    return trainer


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        _log.warning("No callback configs found! Skipping...")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            _log.info(f"Initializing callback <{cb_conf._target_}>...")
            callbacks.append(instantiate(cb_conf))
    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        _log.warning("No logger configs found! Make sure you are not debugging.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            _log.info(f"Initializing logger <{lg_conf._target_}>...")
            logger.append(instantiate(lg_conf))

    return logger


#----------------------------- Analysis ---------------------------------------

def instantiate_analysis(cfg: DictConfig):
    _log.info(f"Initializing analysis module...")

    trainer, best_model = load_run(cfg.run_path, cfg.overrides, cfg.checkpoint_file)
    analyses = instantiate(cfg.analysis)

    return analyses, trainer, best_model

    # metrics: Dict[str, Metric] = instantiate_metrics(cfg.metrics)

    # visualizations = instantiate_visualizations(cfg.visualizations)

    # logger = instantiate_loggers(cfg.logger)[0]

    # analysis_module = Analysis(
    #     datamodule=datamodule,
    #     model=model,
    #     trainer=trainer,
    #     metrics=metrics,
    #     visualizations=visualizations,
    #     logger=logger,
    # )


def load_run(
    run_path: str,
    overrides: DictConfig,
    checkpoint_file: Optional[str] = None,
) -> INSTANTIATED_RUN_MODULES:
    _log.info(f"Loading run found at <{run_path}>...")

    run_cfg = load_cfg(run_path)
    run_cfg.callbacks = None
    run_cfg.logger = None
    run_cfg.merge_with(overrides)

    trainer, model = instantiate_run(run_cfg)

    weights = load_model_weights(model, run_path, checkpoint_file)

    best_model = eqx.combine(weights, model)

    return trainer, best_model


def load_cfg(run_path: str) -> DictConfig:
    assert osp.isdir(run_path), f"Run log directory {run_path} does not exist"

    config_path = osp.join(run_path, ".hydra", "hydra.yaml")
    overrides_path = osp.join(run_path, ".hydra", "overrides.yaml")

    loaded_config = OmegaConf.load(config_path).hydra.job.config_name
    overrides = OmegaConf.load(overrides_path)

    return hydra.compose(loaded_config, overrides=overrides)


def load_model_weights(model: PyTree, save_folder: str, checkpoint_file: Optional[str] = None):
    if checkpoint_file is None:
        save_file = find_best_by_epoch(osp.join(save_folder, "checkpoints"))
    else:
        save_file = osp.join(save_folder, "checkpoints", checkpoint_file)

    # only keep the learned weights of the model in case we want to override other hyper-parameters
    return eqx.partition(eqx.tree_deserialise_leaves(save_file, model), eqx.is_array)[0]


def find_best_by_epoch(checkpoint_folder: str) -> str:
    ckpt_files = os.listdir(checkpoint_folder)  # list of strings

    # checkpoint format is '{best/periodic}_ckpt-iteration_{int}.eqx'
    def is_epoch_ckpt(f: str):
        return f[-10:-4].isdigit()

    last_epoch = max([f[-10:-4] for f in ckpt_files if is_epoch_ckpt(f)])

    best_ckpt = osp.join(checkpoint_folder, f"best_ckpt-iteration_{last_epoch}.eqx")
    if osp.isfile(best_ckpt):
        return best_ckpt

    last_ckpt = osp.join(checkpoint_folder, f"periodic_ckpt-iteration_{last_epoch}.eqx")
    if not osp.isfile(last_ckpt):
        raise RuntimeError(
            "Neither a default best or last checkpoint path could be found. Specifiy them manually"
        )

    return last_ckpt
