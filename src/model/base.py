from abc import ABC, abstractmethod

import jax
import equinox as eqx

from typing import Any
from typing_extensions import Self
from jaxtyping import PyTree


class FunctionalModel(eqx.Module, ABC):
    def partition(self):
        """
        Define how the model should partitioned between params and statics. By default all arrays
        are trainable, but more complex models my want to define things differently.
        """
        # arrays, statics = eqx.partition(self, eqx.is_array)
        # is_trainable = jax.tree_map(lambda x: False if x is None else True, arrays)
        # return Parameters(values=arrays, is_trainable=is_trainable), statics
        return eqx.partition(self, eqx.is_array)

    def parameters(self) -> PyTree:
        return self.partition()[0]

    def instantiate(self, params: PyTree) -> Self:
        return eqx.combine(params, self)

    @abstractmethod
    def __call__(self, inputs: PyTree, key: jax.Array) -> Any:
        raise NotImplementedError

    @abstractmethod
    def init(self, inputs, key) -> PyTree:
        raise NotImplementedError

    def set_inference(self, mode=True) -> Self:
        model = eqx.tree_at(lambda x: x.inference, self, mode)
        return eqx.tree_inference(model, mode)

    def set_train(self) -> Self:
        return self.set_inference(False)
