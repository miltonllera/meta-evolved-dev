from abc import abstractmethod, ABC
from typing import List, Optional
from logging import getLogger

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr
import equinox as eqx

from src.trainer.callback import Callback
from src.trainer.logger import Logger
from src.trainer.utils import aot_compilation
from src.task.base import Task
from src.utils import tree_stack


_logger = getLogger(__name__)


class Trainer(ABC):
    def __init__(
        self,
        task: Task,
        steps: int,
        val_steps: int,
        val_freq: Optional[int],
        loggers: Optional[List[Logger]],
        callbacks: Optional[List[Callback]],
        use_progress_bar: bool = True,
    ) -> None:
        if loggers is None:
            loggers = []

        if callbacks is None:
            callbacks = []

        if val_freq is None:
            val_freq = steps + 1

        if val_freq > steps:
            _logger.warn(
                "Validation frequency set to value greater than training steps or None which "
                "means no validation iterations will be executed for this run. If this was not"
                f"intended set val_freq to value smaller than {steps}."
            )

        self.task = task
        self.steps = steps
        self.val_steps = val_steps
        self.val_freq = val_freq
        self.loggers = loggers
        self.callbacks = callbacks
        self.use_progress_bar = use_progress_bar

    def run(self, model: eqx.Module, key: jax.Array) -> eqx.Module:
        self._run_callbacks("init")
        try:
            result = self._run(model, key)
        finally:
            self._run_callbacks("finalize")
        return result

    @abstractmethod
    def _run(self, model: eqx.Module, key: jax.Array) -> eqx.Module:
        raise NotImplementedError

    @abstractmethod
    def init(self, stage, fit_algorithm, state=None, *, key):
        raise NotImplementedError

    def format_log_dict(self, stage, metrics_dict):
        """
        This method just adds the stage from which the metrics where obtained to the key.
        """
        return {f"{stage}/{k}": v for (k, v) in metrics_dict.items()}

    def _fit_loop(self, model, fit_algorithm, train_step, val_step, *, key):
        init_key, key = jr.split(key)
        train_state = self.init("train", fit_algorithm, None, key=init_key)
        train_step, val_step = self._compile_step_fns(train_step, val_step, train_state)

        _logger.info("Training is starting...")
        self._run_callbacks('train_start', model, train_state)

        for i in range(self.steps):
            train_state, log_dict = train_step(train_state, i)

            log_dict = self.format_log_dict("train", log_dict)

            self._run_callbacks("train_iter_end", i + 1, log_dict, train_state)

            if (i + 1) % self.val_freq == 0 or (i + 1) == self.steps:
                key, val_key = jr.split(key)
                val_log_dict = self._val_loop(val_step, model, train_state, val_key)
                self._run_callbacks("validation_end", i + 1, val_log_dict, train_state)

        self._run_callbacks("train_end", self.steps, train_state)
        _logger.info("Training completed.")

        return train_state

    def _val_loop(self, val_step, model, trainer_state, key):
        val_state = self.init("val", None, trainer_state, key=key)
        self._run_callbacks('validation_start', model, val_state)

        accum_metrics = []
        for i in range(self.val_steps):
            val_state, metrics = val_step(val_state, i)

            log_dict = self.format_log_dict("val", metrics)
            accum_metrics.append(metrics)

            self._run_callbacks("validation_iter_end", i + 1, log_dict, val_state)

        # steps in a validation loop are averaged
        accum_metrics = jtu.tree_map(lambda x: jnp.mean(x, axis=0), tree_stack(accum_metrics))

        return self.format_log_dict("val", accum_metrics)

    # def _test_loop(self, model, test_step, trainer_state, *, key):
    #     _logger.info("Test started")

    #     test_state = self.init("test", model, None, trainer_state, key=key)
    #     test_step = aot_compilation(test_step, test_state)

    #     self.run_logger_and_callbacks("test_start", model, test_state)

    #     accumulated_metrics = []
    #     for i in range(self.val_steps):
    #         test_state, (metrics, extra_results) = test_step(test_state, i)

    #         log_dict = self.format_log_dict("test", metrics)
    #         accumulated_metrics.append(metrics)

    #         self.run_logger_and_callbacks("test_iter_end", i, log_dict, test_state, extra_results)

    #     accumulated_metrics = jtu.tree_map(
    #         lambda x: jnp.mean(x, axis=0), tree_stack(accumulated_metrics)
    #     )

    #     self.run_logger_and_callbacks("test_end", self.val_steps, accumulated_metrics, test_state)

    #     _logger.info("Test completed.")

    def _run_callbacks(self, hook_name: str, *args):
        if self.callbacks is not None:
            for c in self.callbacks:
                getattr(c, hook_name)(*args)

        if self.loggers is not None:
            for l in self.loggers:
                getattr(l, hook_name)(*args)

    def _compile_step_fns(self, train_step, val_step, train_init_state):
        """
        Performs ahead-of-time compilation of step functions to prevent recompilations.
        """
        # TODO: Add parameters to compilation if needed

        _logger.info("Compiling step functions...")

        dummy_val_state = self.init("val", None, train_init_state, key=jr.PRNGKey(0))

        train_step = aot_compilation(train_step, train_init_state)
        val_step = aot_compilation(val_step, dummy_val_state)

        # from src.utils import todotgraph
        # todotgraph(train_step.as_text()).view()
        # exit()

        _logger.info("Done.")

        return train_step, val_step
