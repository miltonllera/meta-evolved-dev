from typing import Callable, Dict, Tuple, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
from jaxtyping import PyTree, Float, Array
from qdax.types import Centroid
from qdax.utils.metrics import default_qd_metrics

from src.problem.base import QDProblem, GoalDirectedSearch
from src.problem.catgen import CategoricalQDProblem
from src.task.base import Task
from src.model.base import FunctionalModel
from src.nn.dna import DNAGenerator
from src.evo.qd import (
    CategoricalRepertoire,
    MapElitesRepertoire,
    MAPElites,
    OneShotMapElites,
    QDScoreAggregator,
    Metrics,
    qd_score_x_coverage,
    compute_cvt_centroids
)
from src.utils import tree_cat
# from src.analysis.qd import _plot_2d_repertoire


QD_ALGORITHM = Union[MAPElites, OneShotMapElites]


class QDSearchDNA(Task):
    """
    Search over possible DNA sequences that guide the rollout of a developmental model. Note that
    unlike other tasks, the models here are pairs of NCA + DNA generator. This can be any function
    that returns a sample of sequences, whether this is a fixed list or randomly generated.
    """
    def __init__(
        self,
        problem: QDProblem,
        qd_algorithm: QD_ALGORITHM,
        popsize: int,
        n_iters: int,
        n_centroids: int = 1000,
        n_centroid_samples: Optional[int] = None,
        score_aggregator: Optional[QDScoreAggregator] = None,
    ) -> None:
        if n_centroid_samples is None:
            n_centroid_samples = problem.descriptor_length * n_centroids

        if score_aggregator is None:
            score_aggregator = QDScoreAggregator(qd_score_x_coverage)

        self.problem = problem
        self.qd_algorithm = qd_algorithm
        self.n_iters = n_iters
        self.popsize = popsize
        self.n_centroids = n_centroids
        self.n_centroid_samples = n_centroid_samples  # type: ignore
        self.score_aggregator = score_aggregator

    @property
    def mode(self):
        return 'max'

    def init(self, stage, centroids, key):
        # pass none to get a new state regardless of stage
        if stage == "train" or centroids is None:
            centroids = self.init_centroids(key)
        return centroids

    def init_centroids(self, key):
        centroids, _ = compute_cvt_centroids(
            self.problem.descriptor_length, # type: ignore
            self.n_centroid_samples,
            self.n_centroids,
            self.problem.descriptor_min_val, # type: ignore
            self.problem.descriptor_max_val, # type: ignore
            key
        )

        return centroids

    # eqx.filter_jit is not required here as all inputs are jax arrays, jax.jit would suffice
    @eqx.filter_jit
    def overall_fitness(
        self,
        genotypes_and_phenotypes: Tuple[Array, PyTree],
        metrics: Dict[str, Float[Array, "..."]],
        repertoire: MapElitesRepertoire,
        key: jax.Array,
    ):
        """
        Compute the overall, per map fitness
        """
        # aggregated_qd_score: (n_iters,)
        qd_scores, extra_terms = self.score_aggregator(
            genotypes_and_phenotypes, metrics, repertoire
        )

        return qd_scores, extra_terms

    @eqx.filter_jit
    def eval(
        self,
        model_and_dna: Tuple[FunctionalModel, DNAGenerator],
        centroids: Centroid,
        key: jax.Array,
    ):
        (dnas, outputs, _), _, (_, metrics), ((repertoire, _, _), key) = self.predict(
            model_and_dna, centroids, key
        )
        fitnesses, _ = self.overall_fitness((dnas, outputs), metrics, repertoire, key)

        # NOTE: Since we are now returning vectors for both the fitnesses and the metrics,
        # we must take the final value of fitness explicity in this function. The average
        # variance over all iterations in the inner loop is thus not included in the fitness
        # term below, only the variance in the final interation (for consistency). I can
        # maintain consistency with previous training runs by using:
        #
        #   fitness = fitness[-1] + self.dna_variance_coefficient * (
        #            extra_terms['dna_variance'].mean() - extra_terms['dna_variance'][-1]
        #        )
        #
        # but for now I'll keep it like this.
        fitness = fitnesses[-1]

        return fitness, (centroids, dict(fitness=fitnesses, coverage=metrics['coverage']))

    @eqx.filter_jit
    def validate(
        self,
        model_and_dna: Tuple[FunctionalModel, DNAGenerator],
        centroids: Centroid,
        key: jax.Array,
    ):
        (dnas, outputs, _), _, (_, metrics), ((repertoire, _, _), key) = self.predict(
            model_and_dna, centroids, key
        ) # type: ignore
        fitness, extra_terms = self.overall_fitness((dnas, outputs), metrics, repertoire, key)

        # NOTE: See note in previous funcition, for validation I am just averaging over iterations.
        metrics['fitness'] = fitness
        metrics =  {**metrics, **extra_terms}

        metrics = jax.tree_map(jnp.mean, metrics)

        return metrics, centroids

    @eqx.filter_jit
    def predict(
        self,
        model_and_dna: Tuple[FunctionalModel, DNAGenerator],
        centroids: Centroid,
        key: jax.Array,
    ):
        """
        Compute outputs for a given model along the MAP-Elites sample path. The results of this
        function are returned without further processing to the caller for e.g. evaluation.

        Notice that outputs includes all intermediate states produced by the model, even though
        these are not used during either training or validation. However, since they could be used
        for model analysis it is best to return them and let the XLA compiler remove them from the
        computation graph during training.
        """
        model, dna_gen = model_and_dna

        dna_key, score_init_key, mpe_key = jr.split(key, 3)

        @jax.vmap
        def generate_from_dna(genotype, key):
            output, states = model(genotype, key)
            return self.problem(output), (output, states)

        init_dnas = dna_gen(self.popsize, key=dna_key).reshape(self.popsize, -1)
        init_scores, (init_out, init_dev) = generate_from_dna(
            init_dnas, jr.split(score_init_key, self.popsize)
        )

        init_mpe_state = self.qd_algorithm.init(init_dnas, centroids, init_scores, mpe_key)

        def step_fn(carry, _):
            mpe_state, key = carry
            eval_key, key = jr.split(key)

            dnas = self.qd_algorithm.ask(mpe_state)
            scores, outputs = generate_from_dna(dnas, jr.split(eval_key, self.popsize))
            mpe_state, metrics = self.qd_algorithm.tell(dnas, scores, mpe_state)  # type: ignore

            return (mpe_state, key), ((dnas, outputs[0], outputs[1]), mpe_state, (scores, metrics))

        final_state, (genotype_and_phenotypes, mpe_states, scores_and_metrics) = jax.lax.scan(
            step_fn,
            (init_mpe_state, key),
            jnp.arange(self.n_iters)
        )

        # prepend the intial state
        tree_expand_dim = lambda t: jtu.tree_map(lambda x: x[None], t)

        init_metrics = self.qd_algorithm._metrics_function(init_mpe_state[0])
        init_scores_and_metrics = init_scores, init_metrics
        init_gen_and_phen = init_dnas, init_out, init_dev

        mpe_states = tree_cat(
            [tree_expand_dim(init_mpe_state), mpe_states], axis=0
        )
        genotype_and_phenotypes = tree_cat(
            [tree_expand_dim(init_gen_and_phen), genotype_and_phenotypes], axis=0
        )
        scores_and_metrics = tree_cat(
            [tree_expand_dim(init_scores_and_metrics), scores_and_metrics], axis=0
        )

        return genotype_and_phenotypes, mpe_states, scores_and_metrics, final_state


class ZeroShotQDSearchDNA(QDSearchDNA):
    def __init__(
        self,
        problem: QDProblem,
        popsize: int,
        n_centroids: int = 1000,
        n_centroid_samples: Optional[int] = None,
        metrics_fn: Optional[Callable[[MapElitesRepertoire], Metrics]] = None,
        score_aggregator: Optional[QDScoreAggregator] = None,
    ):
        if metrics_fn is None:
            metrics_fn = lambda x: default_qd_metrics(x, 0)

        super().__init__(
            problem,
            OneShotMapElites(metrics_fn),  # type: ignore
            popsize,
            0,
            n_centroids,
            n_centroid_samples,
            score_aggregator,
        )

    @eqx.filter_jit
    def overall_fitness(
        self,
        genotypes_and_phenotypes: Tuple[Array, PyTree],
        metrics: Dict[str, Float[Array, "..."]],
        repertoire: MapElitesRepertoire,
        key: jax.Array,
    ):
        """
        Compute the overall, per map fitness
        """
        # aggregated_qd_score: (n_iters,)
        qd_scores, extra_terms = self.score_aggregator(
            genotypes_and_phenotypes, metrics, repertoire
        )

        return qd_scores, extra_terms

    @eqx.filter_jit
    def eval(
        self,
        model_and_dna: Tuple[FunctionalModel, DNAGenerator],
        centroids: Centroid,
        key: jax.Array,
    ):
        (dnas, outputs, _), _, (_, metrics), (repertoire, key) = self.predict(
            model_and_dna, centroids, key
        )
        fitnesses, _ = self.overall_fitness((dnas, outputs), metrics, repertoire, key)
        fitness = fitnesses[-1]

        return fitness, (centroids, dict(fitness=fitnesses, coverage=metrics['coverage']))

    @eqx.filter_jit
    def validate(
        self,
        model_and_dna: Tuple[FunctionalModel, DNAGenerator],
        centroids: Centroid,
        key: jax.Array,
    ):
        (dnas, outputs, _), _, (_, metrics), (repertoire, key) = self.predict(
            model_and_dna, centroids, key
        ) # type: ignore
        fitness, extra_terms = self.overall_fitness((dnas, outputs), metrics, repertoire, key)

        metrics['fitness'] = fitness
        metrics =  {**metrics, **extra_terms}

        metrics = jax.tree_map(jnp.mean, metrics)

        return metrics, centroids

    @eqx.filter_jit
    def predict(
        self,
        model_and_dna: Tuple[FunctionalModel, DNAGenerator],
        centroids: Centroid,
        key: jax.Array,
    ):
        """
        Compute outputs for a given model along the MAP-Elites sample path. The results of this
        function are returned without further processing to the caller for e.g. evaluation.

        Unlike the full QD task this is one shot, so we don't require an emitter.
        """
        model, dna_gen = model_and_dna

        dna_key, score_init_key, mpe_key = jr.split(key, 3)

        @jax.vmap
        def generate_from_dna(genotype, key):
            output, states = model(genotype, key)
            return self.problem(output), (output, states)

        dnas = dna_gen(self.popsize, key=dna_key).reshape(self.popsize, -1)
        scores, phenotypes = generate_from_dna(dnas, jr.split(score_init_key, self.popsize))

        mpe_state, metrics, _ = self.qd_algorithm.init(dnas, centroids, scores, mpe_key)

        final_state = mpe_state, key
        # all functions expect a leading inner-loop dimension, so we add it for consistency
        dnas, scores, phenotypes, mpe_state, metrics = jtu.tree_map(
            lambda x: x[None], (dnas, scores, phenotypes, mpe_state, metrics)
        )

        return (dnas, *phenotypes), mpe_state, (scores, metrics), final_state


class DNACategoryQD(QDSearchDNA):
    def __init__(
        self,
        problem: CategoricalQDProblem,
        qd_algorithm: MAPElites,
        popsize: int,
        n_iters: int,
        score_aggregator: Optional[QDScoreAggregator] = None,
    ) -> None:
        if score_aggregator is None:
            score_aggregator = QDScoreAggregator(qd_score_x_coverage)

        assert qd_algorithm.repertoire_cls == CategoricalRepertoire

        self.problem = problem
        self.qd_algorithm = qd_algorithm
        self.n_iters = n_iters
        self.popsize = popsize
        self.score_aggregator = score_aggregator

    @property
    def mode(self):
        return 'max'

    def init(self, stage, categories, key):
        # return the number of classes which for categorical QD problem is the descriptor length
        if stage == "train" or categories is None:
            categories = jnp.arange(self.problem.descriptor_max_val[0])
        return categories


#---------------------------------------- Goal DNA search -----------------------------------------

class DNATargetSearch(Task):
    def __init__(
        self,
        problem: GoalDirectedSearch,
    ) -> None:
        self.problem = problem

    @property
    def mode(self):
        return 'min'

    def init(self, stage, state, key):
        return None

    def eval(self, model, state: PyTree, key: jax.Array):
        _, (error, metrics) = self.predict(model, state, key)
        return error, (None,  {'error': error} | metrics)

    def validate(self, model, state: PyTree, key: jax.Array):
        _, (error, metrics) = self.predict(model, state, key)
        return {'error': error} | metrics , None

    def predict(
        self,
        model: FunctionalModel,
        state: PyTree,
        key: jax.Array,
    ):
        output, dev_states = model(key)
        error, metrics = self.problem(output)  # type: ignore
        return (model.dna, output, dev_states), (error, metrics)  # type: ignore
