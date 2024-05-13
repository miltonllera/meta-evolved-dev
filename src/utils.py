import os.path as osp
from time import time

import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
from jax._src.lib import xla_client  # type: ignore
from graphviz import Source
from jaxtyping import Array, Float, PyTree


Tensor = Float[Array, "..."]


#------------------------------------------ Tree utils -------------------------------------------

def tree_select(tree, indexes, axis=0):
    indexes = jnp.asarray(indexes)
    return jtu.tree_map(lambda x: jnp.take(x, indexes, axis), tree)


def tree_shape(tree):
    return jtu.tree_map(lambda x: x.shape, tree)


def tree_stack(trees, axis=0):
    return jtu.tree_map(lambda *v: jnp.stack(v, axis=axis), *trees)


def tree_cat(trees, axis=0):
    return jtu.tree_map(lambda *v: jnp.concatenate(v, axis=axis), *trees)


def tree_unstack(tree, is_leaf=None):
    leaves, treedef = jtu.tree_flatten(tree, is_leaf=is_leaf)
    return tuple(treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True))


def tree_dim_unflatten(tree, dim, shape, is_leaf=None):
    unflatten_dim = lambda x: x.reshape(*x.shape[:dim], *shape, *x.shape[dim+1:])
    return jtu.tree_map(unflatten_dim, tree, is_leaf=is_leaf)


def tree_dim_flatten(tree, start_dim, end_dim=None, is_leaf=None):
    if end_dim is None:
        end_dim = start_dim + 1

    def flatten_dims(x):
        total_size = np.prod(x.shape[start_dim: end_dim+1])
        return x.reshape(*x.shape[:start_dim], total_size, *x.shape[end_dim + 1:])

    return jtu.tree_map(flatten_dims, tree, is_leaf=is_leaf)


#--------------------------------------------- I/O ------------------------------------------------

def save_pytree(model: PyTree, save_folder: str, save_name: str):
    save_file = osp.join(save_folder, f"{save_name}.eqx")
    eqx.tree_serialise_leaves(save_file, model)


def load_pytree(save_folder: str,  file_name: str, template: PyTree):
    save_file = osp.join(save_folder, f"{file_name}.eqx")
    return eqx.tree_deserialise_leaves(save_file, template)


#------------------------------------------ Debugging ---------------------------------------------

def todotgraph(x):
    dot_graph = xla_client._xla.hlo_module_to_dot_graph(xla_client._xla.hlo_module_from_text(x))
    s = Source(dot_graph, filename="step.gv", format="png")
    return s


class SnippetTimer:
    def __init__(self, snippet_description=None):
        self.start_time = 0
        self.snippet_description = snippet_description

    def __enter__(self):
        self.start_time = time()
        return self.start_time

    def __exit__(self, *args, **kwargs):
        total_time = time() - self.start_time
        if self.snippet_description is None:
            print(f"Total time taken: {total_time}")
        else:
            print(f"Total time taken for snippet '{self.snippet_description}': {total_time}")
