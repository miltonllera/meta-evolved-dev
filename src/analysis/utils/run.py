import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import jax.lax as lax
import jax.tree_util as jtu
from typing import Iterable
from jaxtyping import PyTree

from src.utils import tree_select, tree_unstack, tree_dim_flatten


def select_and_unstack(outputs: Iterable[PyTree], indexes: Iterable[int]):
    indexed_outputs = tuple(tree_select(op, indexes) for op in outputs)
    return tuple(tree_unstack(io) for io in indexed_outputs)


def generate_outputs(model, task, key=None):
    if key is None:
        key = jr.PRNGKey(0)
    task_state = task.init("val", None, key=key)
    # we must use filter jit because model is a PyTree with non-jax arrays
    jit_predict = eqx.filter_jit(task.predict).lower(model, task_state, key=key).compile()
    return jit_predict(model, task_state, key=key)


def validate_model(model, task, key=None):
    if key is None:
        key = jr.PRNGKey(0)

    task_state = task.init("val", None, key=key)
    jit_validate = eqx.filter_jit(task.validate).lower(model,task_state, key=key).compile()
    return jit_validate(model, task_state, key=key)


def batched_eval(model, inputs, eval_fn, batch_size, key):
    eqx.filter_jit
    def eval_step(key, batch):
        key, carry = jr.split(key)
        outputs = jax.vmap(model)(batch, jr.split(key, batch_size))
        results = jax.vmap(eval_fn)(outputs)
        return carry, results

    n_iters = int(jnp.ceil(len(inputs) / batch_size))
    n_pad = batch_size - (len(inputs) % batch_size)
    dummy_pads = [(0, 0) for _ in range(len(inputs.shape) - 1)]

    padded_inputs = jnp.pad(inputs, pad_width=((0, n_pad), *dummy_pads))
    batched_inputs = padded_inputs.reshape(n_iters, batch_size, *inputs.shape[1:])

    results = lax.scan(eval_step, key, batched_inputs)[1]
    results = tree_dim_flatten(results, 0, 1)

    return jtu.tree_map(lambda x: x[:-n_pad], results)


def get_phenotype_from_genotype(lookup_genotype, genotypes, phenotypes):
    idx = jax.vmap(jnp.allclose, in_axes=(0, None))(genotypes, lookup_genotype)
    return phenotypes[idx][0]
