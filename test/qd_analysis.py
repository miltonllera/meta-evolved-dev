import pyrootutils
import warnings
from functools import partial
from typing import NamedTuple

import jax.random as jr


warnings.filterwarnings("ignore")

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,  # add to system path
    dotenv=True,      # load environment variables .env file
    cwd=True,         # change cwd to root
)

# import hydra
import jax
import equinox as eqx
import equinox.nn as nn
# import matplotlib.pyplot as plt
# from matplotlib.animation import PillowWriter
from qdax.utils.metrics import default_qd_metrics

from src.task.dnaqd import QDSearchDNA
from src.problem.levelgen import SimpleLevelGeneration
from src.evo.qd import MAPElites, CMAOptEmitter
from src.model.dev import NCA, DNAGuidedDevModel
from src.nn.ca import IdentityAndSobelFilter, SliceOutput, MaxPoolAlive
from src.nn.dna import DNAControl, DNAContextEncoder, DNAIndependentSampler
from src.analysis.qd import _plot_2d_repertoire
from src.analysis.levelgen import plot_generated_levels
# from src.analysis.viz_utils import generate_gif
from src.utils import tree_unstack



model = DNAGuidedDevModel(
    dev=NCA(
        state_size=16,
        grid_size=(5, 5),
        dev_steps=(30, 40),
        update_prob=0.5,
        context_encoder=DNAContextEncoder(4, 8, 16, key=jr.key(5)),
        control_fn=DNAControl(16, 16, 16, key=jr.key(4)),
        alive_fn=MaxPoolAlive(alive_bit=3, alive_threshold=0.1),
        message_fn=IdentityAndSobelFilter(),
        update_fn=nn.Sequential(
            layers=[
                nn.Conv2d(in_channels=16 * 3, out_channels=32, kernel_size=1, key=jr.key(1)),
                nn.Lambda(jax.nn.relu),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, key=jr.key(2)),
            ],
        ),
        output_decoder=SliceOutput(
            dim=0,
            start_idx=0,
            end_idx=3,
            squashing_function=partial(jax.numpy.argmax, axis=0),
        ),
        output_dev_steps=True,
    ),
    dna_generator=DNAIndependentSampler(
        8, 4, jr.PRNGKey(3)
    ),
)

params, statics = eqx.partition(model, eqx.is_array)


qd_algorithm = MAPElites(
    CMAOptEmitter(
        10,
        32,
        0.1,
        num_descriptors = 2,
        num_centroids = 10,
        random_key= jr.PRNGKey(6),
    ),
    partial(default_qd_metrics, qd_offset=0.0),
)


# jaxpr = jax.make_jaxpr(qd_algorithm.init)
# print(jaxpr.jaxpr)


task = QDSearchDNA(
    SimpleLevelGeneration(5, 5),
    qd_algorithm=qd_algorithm,
    n_iters=3,
    popsize=10,
    n_centroids=10,
    n_centroid_samples=100,
)


class Trainer(NamedTuple):
    """
    Dummy wrapper to conform to plot function's signature
    """
    task: QDSearchDNA


if __name__ == "__main__":
    task_state = task.init("val", None, key=jr.PRNGKey(0))
    key = jr.PRNGKey(0)

    # we must use filter jit because model is a PyTree with non-jax arrays
    generate_outputs = eqx.filter_jit(task.predict).lower(model, task_state, key=key).compile()
    (dnas, outputs, states), mpe_states, (scores, metrics), _ = generate_outputs(model, task_state, key=key)

    # (iters, popsize, dna_size), (n_iters, popsize, out_shape), (iters, popsize, dev_steps, c, h, w)
    print(dnas.shape, outputs.shape, states[0].shape, states[1].shape)
    # print(mpe_states[0])

    # plot map elites-gif

    # trainer = Trainer(task)
    # plot_generated_levels(model, ((dnas, outputs, states), mpe_states, (scores, metrics)), trainer, '')
