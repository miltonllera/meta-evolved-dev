import os
import os.path as osp
import logging
from collections import deque
from pathlib import Path
from typing import Callable, List, Literal, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from src.utils import save_pytree, load_pytree
from src.trainer.utils import PriorityQueue


_logger = logging.getLogger(__name__)


CALLBACK_HOOKS = Literal[
    "train_iter_end",
    "train_end",
    "validation_iter_end",
    "validation_end",
    "test_iter_end",
    "test_end",
]


class Callback:
    """
    Callbacks implementing different functionality.

    The general idea is that the callbacks define what to log (fitness values, model parameters,
    etc.) and the attached logger instance defines where/how to save the information. This could
    be just saving it to disk or a more sophisticated storage such as WandB or TensorBoard.
    """
    def init(self):
        pass

    def train_start(self, *_):
        pass

    def train_iter_end(self, *_):
        pass

    def train_end(self, *_):
        pass

    def validation_start(self, *_):
        pass

    def validation_iter_end(self, *_):
        pass

    def validation_end(self, *_):
        pass

    def test_start(self, *_):
        pass

    def test_iter_end(self, *_):
        pass

    def test_end(self, *_):
        pass

    def finalize(self):
        pass

    # TODO: add handlers for different events?


class Checkpoint(Callback):
    def __init__(self, save_dir, file_template):
        super().__init__()
        self.save_dir = save_dir
        self.file_template = file_template

    def init(self):
        os.makedirs(self.save_dir, exist_ok=True)

    @property
    def best_state(self):
        raise NotImplementedError

    @property
    def last_state(self):
        raise NotImplementedError


class MonitorCheckpoint(Checkpoint):
    def __init__(
        self,
        save_dir: str,
        file_template: str = "best_ckpt-iteration_{iteration:06d}",
        k_best: int = 1,
        mode: str = 'min',
        monitor_key: Union[Callable, int, str, None] = None,
        run_on: CALLBACK_HOOKS = "validation_end",
        state_getter: Optional[Union[Callable, str, int]] = None,
    ) -> None:
        super().__init__(save_dir, file_template)

        if state_getter is not None and not isinstance(state_getter, Callable):
            getter = lambda x: x[state_getter]
        elif state_getter is None:
            getter = lambda x: x
        else:
            getter = state_getter

        self.k_best = k_best
        self.mode = mode
        self.monitor_key = monitor_key
        self.state_getter = getter
        self.run_on = run_on
        self._ckpts = PriorityQueue(k_best, [])
        self._state_template = None

    def has_improved(self, metric):
        if len(self._ckpts) < self.k_best:
            return True
        return metric > self._ckpts.lowest_priority

    @property
    def best_state(self):
        best_state_file = max(self._ckpts).item
        return load_pytree(self.save_dir, best_state_file, self._state_template)

    def train_start(self, model, initial_trainer_state):
        if self.state_getter is not None and hasattr(self.state_getter, 'init'):
            self.state_getter.init(model, initial_trainer_state)
        self._state_template = self.state_getter(initial_trainer_state)

        # use the initial state as a sentinel
        self.update_files(0, -np.inf, initial_trainer_state)

    def validation_end(self, iter, metric, state):
        if self.run_on == "validation_end":
            return self.update_checkpoints(iter, metric, state)

    def train_iter_end(self, iter, metric, state):
        if self.run_on == "train_iter_end":
            return self.update_checkpoints(iter, metric, state)

    def update_checkpoints(self, iter, metric, state):
        if self.monitor_key is not None:
            try:
                metric = metric[self.monitor_key]
            except KeyError as e:
                e.add_note(f"Available keys are {metric.keys()}")

        # because checkpoints are stored in a min heap we want the worst model to lowest priority
        priority = metric if self.mode == "max" else -metric
        if self.has_improved(priority):
            self.update_files(iter, priority, state)

    def update_files(self, iter, priority, state):
        state = self.state_getter(state)
        file = self.file_template.format(iteration=iter)
        to_delete = self._ckpts.push_and_pop((priority, file))

        save_pytree(state, self.save_dir, file)
        if to_delete is not None:
            path = Path(osp.join(self.save_dir, to_delete[1]) + ".eqx")
            path.unlink(True)


class PeriodicCheckpoint(Checkpoint):
    def __init__(
        self,
        save_dir: str,
        file_template: str = "periodic_ckpt-iteration_{iteration:06d}",
        checkpoint_freq: int = 1,
        max_checkpoints: Optional[int] = None,
        state_getter: Optional[Union[Callable, str, int]] = None,
    ) -> None:
        super().__init__(save_dir, file_template)

        if max_checkpoints is None:
            _logger.warn(
                "The maximum number of checkpoints for a PeriodicCheckpoint instance was not set. "
                "This can potentially result in large hard disk space usage for long training runs "
                "and/or large training states. Ensure that this is intended."
            )

        if state_getter is not None and not isinstance(state_getter, Callable):
            getter = lambda x: x[state_getter]
        elif state_getter is None:
            getter = lambda x: x
        else:
            getter = state_getter

        self.state_getter = getter
        self.file_template = file_template
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoints = max_checkpoints
        self._ckpt_files = deque()
        self._ckpt_state = None
        self._n_calls = 0

    @property
    def last_checpoint_state(self):
        return self._ckpt_state

    def train_start(self, model, initial_trainer_state):
        self._n_calls = 0

        if self.state_getter is not None and hasattr(self.state_getter, 'init'):
            self.state_getter.init(model, initial_trainer_state)
        self._state_template = self.state_getter(initial_trainer_state)

        # use the initial state as a sentinel
        self.update_checkpoints(0, initial_trainer_state)

    def train_iter_end(self, iter, metric, state) -> None:
        self._n_calls = self._n_calls + 1
        if self._n_calls % self.checkpoint_freq == 0:
            self.update_checkpoints(iter, state)

    def update_checkpoints(self, iter, state):
        state = self.state_getter(state)

        if self.max_checkpoints is not None and len(self._ckpt_files) == self.max_checkpoints:
            self.delete_oldest()

        new_file = self.file_template.format(iteration=iter)
        self._ckpt_files.append(new_file)
        save_pytree(state, self.save_dir, new_file)

    def delete_oldest(self):
        path = Path(osp.join(self.save_dir, self._ckpt_files.popleft()) + ".eqx")
        path.unlink(True)


class VisualizationCallback(Callback):
    """
    wrapper class around visualization functions
    """
    def __init__(
        self,
        visualization,
        save_dir: str,
        save_prefix: str = "",
        run_on: Union[Literal["all"], List[CALLBACK_HOOKS]] = "all",
    ):
        self.viz = visualization
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        self.run_on = run_on

        os.makedirs(save_dir, exist_ok=True)

    def train_iter_end(self, *args, **kwargs):
        if self.run_on == "all" or "train_iter_end" in self.run_on:
            self._plot(*args, **kwargs)

    def train_end(self, *args, **kwargs):
        if self.run_on == "all" or "train_end" in self.run_on:
            self._plot(*args, **kwargs)

    def validation_iter_end(self, *args, **kwargs):
        if self.run_on == "all" or "validation_iter_end" in self.run_on:
            self._plot(*args, **kwargs)

    def validation_end(self, *args, **kwargs):
        if self.run_on == "all" or "validation_end" in self.run_on:
            self._plot(*args, **kwargs)

    def test_iter_end(self, *args, **kwargs):
        if self.run_on == "all" or "test_iter_end" in self.run_on:
            self._plot(*args, **kwargs)

    def test_end(self, *args, **kwargs):
        if self.run_on == "all" or "test_end" in self.run_on:
            self._plot(*args, **kwargs)

    def _plot(self, *args, **kwargs):
        plot_names, figures = self.viz(*args, **kwargs)

        if not isinstance(plot_names, list):
            plot_names = [plot_names]
            figures = [figures]

        for name, fig in zip(plot_names, figures):
            if self.save_prefix == "":
                file_name = name
            else:
                file_name = f"{self.save_prefix}_{name}"

            save_file = osp.join(self.save_dir, file_name)
            fig.savefig(save_file, bbox_inches='tight', dpi=300)
            plt.close(fig)
