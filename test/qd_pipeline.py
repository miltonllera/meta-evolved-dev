from functools import partial
import jax.random as jr
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,  # add to system path
    dotenv=True,      # load environment variables .env file
    cwd=True,         # change cwd to root
)

import jax
import equinox.nn as nn
from qdax.utils.metrics import default_qd_metrics
from src.trainer.evo import EvoTrainer
from src.evo.strategy import Strategy
from src.task.dnaqd import QDSearchDNA
from src.problem.levelgen import SimpleLevelGeneration
from src.evo.qd import MAPElites, CMAOptEmitter
from src.model.dev import NCA, DNAGuidedDevModel
from src.nn.ca import IdentityAndSobelFilter, SliceOutput, MaxPoolAlive
from src.nn.dna import DNAControl, DNAContextEncoder, DNAIndependentSampler
from src.trainer.callback import MonitorCheckpoint, PeriodicCheckpoint

# import sys
# import warnings
# import traceback


# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))

# warnings.showwarning = warn_with_traceback


model = DNAGuidedDevModel(
    dev=NCA(
        state_size=16,
        grid_size=(5, 5),
        dev_steps=(30, 40),
        update_prob=0.5,
        context_encoder=DNAContextEncoder(4, 8, 16, input_is_distribution=True, key=jr.PRNGKey(5)),
        control_fn=DNAControl(16, 16, 16, key=jr.PRNGKey(4)),
        alive_fn=MaxPoolAlive(alive_bit=3, alive_threshold=0.1),
        message_fn=IdentityAndSobelFilter(),
        update_fn=nn.Sequential(
            layers=[
                nn.Conv2d(in_channels=16 * 3, out_channels=32, kernel_size=1, key=jr.PRNGKey(1)),
                nn.Lambda(jax.nn.relu),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, key=jr.PRNGKey(2)),
            ],
        ),
        output_decoder=SliceOutput(
            dim=0,
            start_idx=0,
            end_idx=3,
            squashing_function=partial(jax.numpy.argmax, axis=0),
        ),
    ),
    dna_generator=DNAIndependentSampler(
        8, 4, key=jr.PRNGKey(3)
    ),
)


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
    n_iters=1,
    popsize=10,
    n_centroids=10,
    n_centroid_samples=100,
)

strategy = Strategy("CMA_ES", {'popsize': 5}, {})


callbacks = [
    MonitorCheckpoint(
        "test/temp",
        k_best=2,
        mode="max",
        monitor_key="val/fitness_mean"
    ),
    PeriodicCheckpoint(
        "test/temp",
        checkpoint_freq=1,
        max_checkpoints=None,  # save on every call
    )
]

trainer = EvoTrainer(task, strategy, 11, 2, 5, callbacks=callbacks)

trainer.run(model, jr.PRNGKey(0))
