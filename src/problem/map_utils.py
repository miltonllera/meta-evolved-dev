import os
import logging
import multiprocessing as mp
from itertools import product
from collections import deque

import numpy as np
import jax.numpy as jnp

# from src.utils import SnippetTimer


_logger = logging.getLogger(__name__)


if 'N_EVALUATION_CPUS' in os.environ:
    N_CPUS = int(os.environ['N_EVALUATION_CPUS'])
    if N_CPUS >= mp.cpu_count():
        raise RuntimeError
else:
    N_CPUS = mp.cpu_count() - 1
_logger.info(f"Using {N_CPUS} CPUs for parallel map evaluation.")


MAX_VALUE = np.finfo(np.float32).max
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def batched_n_islands(int_maps):
    # Note that because the callbacks above are using vectorization, they will sync over all
    # vmaps applied. Thus there could be 1 or 2 leading dimenisions if we are either evaluating
    # a model or if this is called when running evolutionary training, respectively.
    if len(int_maps.shape) == 3:
        batch_shapes = int_maps.shape[:1]
    else:
        batch_shapes = int_maps.shape[:2]
        int_maps = int_maps.reshape((-1, *int_maps.shape[2:]))

    # with SnippetTimer("batched_n_islands"):
    with mp.Pool(N_CPUS) as pool:
        result = pool.imap(n_islands, int_maps, int(np.ceil(len(int_maps) / N_CPUS)))
        result = list(result)

    result = np.concatenate(result)

    return result.reshape(batch_shapes)[..., None]


def n_islands(int_map: np.ndarray, non_traversible_tiles=(0,)):
    h, w = int_map.shape

    int_map = ~np.isin(int_map, non_traversible_tiles)

    if np.all(int_map == 0):
        return np.asarray([np.inf], dtype=np.float32)

    visited = np.zeros_like(int_map, dtype=bool)

    def in_bounds(pos):
        return 0 <= pos[0] < h and 0 <= pos[1] < w

    def visit(i, j):
        to_visit = deque()

        visited[i, j] = True
        to_visit.append((i, j))

        while len(to_visit) > 0:
            i, j = to_visit.popleft()
            for di, dj in directions:
                nb = i + di, j + dj
                if in_bounds(nb) and int_map[nb] and not visited[nb]:
                    to_visit.append(nb)
                    visited[nb] = True

    n_components = 0
    for i, j in product(range(h), range(w)):
        if int_map[i, j] and not visited[i, j]:
            visit(i, j)
            n_components += 1

    return np.asarray([n_components], dtype=np.float32)


def batched_lsp(int_maps):
    # Note that because the callbacks above are using vectorization, they will sync over all
    # vmaps applied. Thus there could be 1 or 2 leading dimenisions if we are evaluating a model
    # if this is called as part of evolutionary training.
    if len(int_maps.shape) == 3:  # when analysing a model we do not have an extra leading dim
        batch_shapes = int_maps.shape[:1]
    else:
        batch_shapes = int_maps.shape[:2]
        int_maps = int_maps.reshape((-1, *int_maps.shape[2:]))

    # Use this to measure how long it takes to run the algorithm on one map. Note that because
    # batched_lsp dominates batched_n_islands (O(N^2) vs O(N)) we only need to measure this one to
    # get an upper bound on how much it costs to run these evaluations. Also note that running it
    # for a map of only valid tiles is the worst case scenario.

    # with SnippetTimer("batched_lsp"):
        # result = list(map(longest_shortest_path, np.ones_like(int_maps[:1])))
    # print(result)
    # input()

    # with SnippetTimer("batched_lsp"):
    with mp.Pool(N_CPUS) as pool:
        result = pool.imap(longest_shortest_path, int_maps, int(np.ceil(len(int_maps) / N_CPUS)))
        result = list(result)

    result = np.concatenate(result)

    return result.reshape(batch_shapes)[..., None]


def longest_shortest_path(int_map, non_traversible_tiles=(0,)):
    """
    Use scipy's graph algorithms to compute the longerst shortest path in a map.

    By default assumes that the empty tile is 0 and it's the only non-traversable tile. Then it
    creates an adjacency matrix from the integer map.
    """
    h, w = int_map.shape
    int_map = ~np.isin(int_map, non_traversible_tiles)

    def in_bounds(pos):
        return 0 <= pos[0] < h and 0 <= pos[1] < w

    def bfs_shortest_path(start_pos):

        to_visit = deque()
        to_visit.append(start_pos)

        visited = np.zeros_like(int_map, dtype=bool)
        visited[start_pos] = True

        min_dist = np.zeros_like(int_map, dtype=np.float32) + MAX_VALUE
        min_dist[start_pos] = 1

        while len(to_visit) > 0:
            i, j = to_visit.popleft()
            for di, dj, in directions:
                nb = i + di, j + dj
                if in_bounds(nb) and int_map[nb] and not visited[nb]:
                    min_dist[nb] = np.minimum(min_dist[nb], min_dist[i, j] + 1)
                    to_visit.append(nb)
                    visited[nb] = True

        return min_dist

    max_path = -MAX_VALUE

    for start_pos in product(range(h), range(w)):
        min_paths = bfs_shortest_path(start_pos)
        max_path = max((min_paths * (min_paths != MAX_VALUE)).max(), max_path)

    return np.asarray([max_path], dtype=np.float32)


def compute_simmetry(int_map, env_shape):
    """
    Compute an aggregate of both vertical and horizontal symmetry.
    """
    result = (
        compute_vertical_symmetry(int_map, env_shape) +
        compute_horizontal_symmetry(int_map, env_shape)
    ) / 2.0

    return result[None]


def compute_horizontal_symmetry(int_map, env_shape):
    """
    Function to get the horizontal symmetry of a level
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns a symmetry float value normalized to a range of 0.0 to 1.0
    """
    # max_val = env_shape[0] * env_shape[1] / 2  # for example 14*14/2=98
    m = 0

    if (int_map.shape[0] % 2) == 0:
        m = jnp.sum(
            int_map[: int_map.shape[0] // 2] == jnp.flip(int_map[int_map.shape[0] // 2 :], 0)
        )
    else:
        m = jnp.sum(
            int_map[: int_map.shape[0] // 2] == jnp.flip(int_map[int_map.shape[0] // 2 + 1 :], 0)
        )

    return m


def compute_vertical_symmetry(int_map, env_shape):
    """
    Function to get the vertical symmetry of a level
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns a symmetry float value normalized to a range of 0.0 to 1.0
    """
    # max_val = env_shape[0] * env_shape[1] / 2
    m = 0

    if (int_map.shape[1] % 2) == 0:
        m = jnp.sum(
            int_map[:, : int_map.shape[1] // 2] == jnp.flip(int_map[:, int_map.shape[1] // 2 :], 1)
        )
    else:
        m = jnp.sum(
            int_map[:, : int_map.shape[1] // 2] == jnp.flip(int_map[:, int_map.shape[1] // 2 + 1 :], 1)
        )

    return m
