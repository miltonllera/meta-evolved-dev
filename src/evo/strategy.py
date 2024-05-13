from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Union

import jax
import evosax as ex
from jaxtyping import Array, PyTree

from .dummy_es import DummyES
from .ga import DiscreteGA


Strategies = {
    'Dummy_ES': DummyES,
    'DiscreteGA': DiscreteGA,
    **ex.Strategies
}


class InstantiatedStrategy(NamedTuple):
    init: Callable[[jax.Array], ex.EvoState]
    ask: Callable[[jax.Array, ex.EvoState], PyTree]
    tell: Callable[[Union[Array, PyTree], Array, ex.EvoState], ex.EvoState]
    param_shaper: ex.ParameterReshaper


class Strategy:
    """
    Wrapper around an Evosax strategy.
    """
    def __init__(
        self,
        strategy: str,
        args: Dict[str, Any],
        strategy_params: Dict[str, Any],
    ) -> None:
        if strategy not in Strategies:
            raise ValueError(f"Unrecognized strategy {strategy}.")

        self.strategy = Strategies[strategy]
        self.args = args
        self.strategy_params = strategy_params

    @property
    def maximize(self):
        return self.args['maximize']

    def instantiate(self, model_params, trainable_filter=None):
        """
        Instatiate works by wrapping each of the relevant functions (i.e. initialize, ask and tell)
        of a Strategy. In this way we do not need to carry the strategy parameter structure around.
        """
        if trainable_filter is not None:
            trainable_params = jax.tree_map(
                lambda x, t: x if t else None,
                model_params,
                trainable_filter,
            )
        else:
            trainable_params = model_params

        # NOTE: we always use n_devices == 1 so the the leading axes has len(params) == popsize.
        # Partitioning of parameters across devices is taken care of by shard_map in the EvoTrainer class.
        strategy: ex.Strategy = self.strategy(
            pholder_params=trainable_params, n_devices=1, **self.args
        )

        strategy_params = strategy.default_params
        strategy_params = strategy_params.replace(  # type: ignore [reportGeneralTypeIssue]
           **self.strategy_params
        )

        return InstantiatedStrategy(
            init=partial(strategy.initialize, params=strategy_params),
            ask=partial(strategy.ask, params=strategy_params),
            tell=partial(strategy.tell, params=strategy_params),
            param_shaper=strategy.param_reshaper,
        )
