import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from typing import Callable


class GatedActivation(eqx.Module):
    n_inputs: int
    act_fn: Callable

    def __init__(self, n_inputs, act_fn=jnn.sigmoid):
        self.n_inputs = n_inputs
        self.act_fn = act_fn

    def __call__(self, inputs, key=None):
        sign_in, act_in = jnp.split(inputs, 2)
        return jnp.sign(sign_in) * self.act_fn(act_in)
