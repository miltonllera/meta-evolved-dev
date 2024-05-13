import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array


def dummy_mutation(x: Float[Array, "N S"], key: jax.Array, **kwargs):
    """
    Use this to "trick" QDax into no-mutating a solution. Useful when testing if a model is just
    using a chaotic strategy or actually making use of the DNA representation.
    """
    return x, key


def discrete_mutation(
    x: Float[Array, "N S"],
    key: jax.Array,
    mutation_prob: float,
    min_val: float,
    max_val: float,
    force_mutation: bool = True,
):
    keys = jr.split(key, len(x) + 1)
    mutations = jax.vmap(_discrete_mutation, in_axes=(0, 0, None, None, None, None))(
        keys[:-1], x, mutation_prob, min_val, max_val, force_mutation
    )
    return mutations, keys[-1]


def _discrete_mutation(
    key: jax.Array,
    x: Float[Array, "S"],
    mutation_prob: float,
    min_val: float,
    max_val: float,
    force_mutation: bool = True,
):
    keys = jr.split(key, len(x) + 1)
    values = jnp.arange(min_val, max_val + 1, 1.0)

    def _resample(key, to_replace):
        p = 1 / (len(values) - 1) * (to_replace != values)
        return jr.choice(key, values, p=p)

    to_mutate = jr.bernoulli(keys[0], mutation_prob, (len(x),))
    if force_mutation:
        force_mutation_idx = jr.choice(keys[0], len(x))
        to_mutate = to_mutate.at[force_mutation_idx].set(True)

    mutations = jax.vmap(_resample)(keys[1:], x)

    return jnp.where(to_mutate, mutations, x)
