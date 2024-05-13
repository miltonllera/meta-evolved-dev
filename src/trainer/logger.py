import os.path as osp
from abc import ABC, abstractmethod
from typing import Iterable, List
from jaxtyping import Float, Array

import wandb
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

from .callback import Callback


class Logger(Callback, ABC):
    def __init__(self) -> None:
        self.owner = None

    @abstractmethod
    def log_scalar(self, key: str, value: List, step):
        raise NotImplementedError

    @abstractmethod
    def log_dict(self, dict_to_log, step):
        raise NotImplementedError

    @abstractmethod
    def save_artifact(self, name, artifact):
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        raise NotImplementedError


class TensorboardLogger(Logger):
    """
    Wrapper around the TensorBoard SummaryWriter class.
    """
    def __init__(self, log_dir: str) -> None:
        super().__init__()
        self.log_dir = log_dir

    def init(self):
        self._summary_writer = SummaryWriter(self.log_dir)

    def finalize(self):
        self._summary_writer.flush()
        self._summary_writer.close()

    def train_iter_end(self, iter, log_dict, _):
        self.log_dict(log_dict, iter)

    def train_end(self, *_):
        self.finalize()

    def validation_end(self, iter, log_dict, _):
        self.log_dict(log_dict, iter)
        self.finalize()

    def test_end(self, iter, log_dict, _):
        self.log_dict(log_dict, iter)

    def log_scalar(self, key, value, step):
        if not isinstance(value, list):
            value = [value]
            step = [step]

        assert len(value) == len(step)

        for v,s in zip(value, step):
            if isinstance(value, Array):
                value = value.item()
            self._summary_writer.add_scalar(key, v, s)

    def log_dict(self, dict_to_log, step):
        for k, values in dict_to_log.items():
            if isinstance(values, Iterable):
                for v, s in zip(values, step):
                    self._summary_writer.add_scalar(k, v, s)
            else:
                self._summary_writer.add_scalar(k, values, step)

    def save_artifact(self, name, artifact):
        if isinstance(artifact, plt.Figure):
            self._summary_writer.add_figure(name, artifact)
        elif isinstance(artifact, Float):
            self._summary_writer.add_image(name, artifact)
        else:
            raise ValueError(f"Unrecognized type {type(artifact)} for artifact value")


class WandBLogger(Logger):
    def __init__(self,
        project: str,
        name: str,
        notes: str,
        tags: List[str],
        run_folder: str,
        log_artifacts: bool = False,
		verbose: bool=False
    ):
        self.project = project
        self.name = name
        self.notes = notes
        self.tags = tags
        self.run_folder = run_folder
        self.log_artifacts = log_artifacts
        self.verbose = verbose
        self._run = None

    def log_dict(self, iter, log_dict):
        if self._run is not None:
            self._run.log(log_dict, iter)

    def init(self):
        self._run = wandb.init(
            project=self.project,
            name=self.name,
            dir=self.run_folder,
            notes=self.notes,
            tags=self.tags,
        )
        self._config_artifact = wandb.Artifact(f"run_{self._run.id}_config", type="config")  # type: ignore
        self._config_artifact.add_dir(local_path=osp.join(self.run_folder, ".hydra"))

        if self.log_artifacts:
            self._model_checkpoint_artifact = wandb.Artifact(
                f"run_{self._run.id}_checkpoints", type="model"  # type: ignore
            )
            self._model_checkpoint_artifact.add_dir(
                local_path=osp.join(self.run_folder, "checkpoints")
            )

    def finalize(self):
        if self._run is not None:
            if self.log_artifacts:
                self._run.log_artifact(self._model_checkpoint_artifact)
            self._run.log_artifact(self._config_artifact)
            self._run.finish()

    def train_iter_end(self, iter, log_dict, _):
        self.log_dict(iter, log_dict)

    def validation_end(self, iter, log_dict, _):
        self.log_dict(iter, log_dict)

    def test_iter_end(self, iter, log_dict, _):
        self.log_dict(iter, log_dict)

    def log_scalar(self, key: str, value: List, step):
        pass

    def save_artifact(self, name, artifact):
        pass
