from dataclasses import dataclass, field
from heapq import heapify, heappush, heappop, heappushpop
from typing import Any, Union, Tuple

import jax
import jax.tree_util as jtu
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.shard_map import shard_map as shmap
from jax.experimental.mesh_utils import create_device_mesh


def get_spec_from_mask(shard_mask):
    return jtu.tree_map(lambda x: PartitionSpec("p") if x else PartitionSpec(), shard_mask)


def shard_over_gpus(func, in_sharding, out_sharding):
    devices = jax.devices()
    n_devices = len(devices)

    if n_devices == 1 or jax.device_count("gpu") == 0:
        return func

    mesh = Mesh(create_device_mesh((n_devices,)), axis_names=("p"))
    in_specs = get_spec_from_mask(in_sharding)
    out_specs = get_spec_from_mask(out_sharding)

    return shmap(func, mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)


def shv_map(func, in_axes, out_axes):
    """
    Convenience wrapper for a function that must be vmapped and shard-mapped --- for example
    evaluation of training parameters.

    For vmapping, this function just uses in_axes and out_axes as usual. It assumes that we wish
    to split the same inputs that we will vmap (as long as there is more than one GPU), and will
    thus compute the specs for the shard_map based on the (in/out)_axes values. Vmapped inputs will
    be split while the rest will be tiled.
    """
    vmap_fn = jax.vmap(func, in_axes, out_axes)

    to_shard = lambda x: x is not None
    int_and_none_leafs = lambda x: x is None or isinstance(x, int)
    in_sharding = jtu.tree_map(to_shard, in_axes, is_leaf=int_and_none_leafs)
    out_sharding = jtu.tree_map(to_shard, out_axes, is_leaf=int_and_none_leafs)

    return shard_over_gpus(vmap_fn, in_sharding, out_sharding)


def aot_compilation(func, inputs, jit_kwargs=None):
    if jit_kwargs is None:
        jit_kwargs = {}
    return jax.jit(func, **jit_kwargs).lower(inputs, 0).compile()


@dataclass(order=True)
class PriorityItem:
    priority: Union[float, int]
    item: Any = field(compare=False)


class PriorityQueue:
    def __init__(self, max_cap: int, items):
        self.max_cap = max_cap
        self.items = [PriorityItem(*i) for i in items]
        heapify(self.items)
        while len(self.items) > max_cap:
            heappop(self.items)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @property
    def lowest_priority(self):
        return self.items[0].priority

    def push_and_pop(self, item: Tuple[Union[float, int], Any]) -> Union[None, Tuple]:
        if len(self.items) == self.max_cap:
            value = heappushpop(self.items, PriorityItem(*item))
            value = value.priority, value.item
        else:
            heappush(self.items, PriorityItem(*item))
            value = None

        return value
