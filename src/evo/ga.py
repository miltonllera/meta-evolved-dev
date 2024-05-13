from typing import Optional, Union, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import evosax as ex
import chex
from flax import struct
from qdax.core.emitters.mutation_operators import polynomial_crossover
from .operators import discrete_mutation


def sample_genotypes(key, genotypes, n_samples):
    idx = jr.choice(key, jnp.arange(len(genotypes)), shape=(n_samples,))
    return genotypes[idx]


@struct.dataclass
class EvoState:
    population: chex.Array
    fitness: chex.Array
    best_member: chex.Array
    best_fitness: float
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    clip_min: int = 0
    clip_max: int = 1


class DiscreteGA(ex.Strategy):
    """
    A dummy strategy that does not change the members of the population.
    """
    def __init__(
        self,
        popsize: int,
        elite_ratio: float = 0.5,
        min_val: float = 0.0,
        max_val: float = 1.0,
        variation_percentage: float = 0.5,
        cross_over_ratio: float = 0.5,
        mutation_prob: Optional[float] = None,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs,
    ):
        super().__init__(popsize, num_dims, pholder_params, mean_decay, n_devices, **fitness_kwargs)
        self.elite_ratio = elite_ratio
        self.min_val = min_val
        self.max_val = max_val
        self.variation_percentage = variation_percentage
        self.cross_over_ratio = cross_over_ratio
        self.mutation_prob = mutation_prob

    def initialize_strategy(self, rng: jax.Array, params: EvoParams) -> EvoState:
        """`initialize` the evolution strategy."""
        coeff = (1 if self.fitness_shaper.maximize else -1)
        init_x = jr.choice(
            rng, jnp.arange(self.min_val, self.max_val), (self.popsize, self.num_dims)  # type: ignore
        )

        return EvoState(
            population=init_x,
            fitness= coeff * jnp.inf * jnp.ones(len(init_x)),
            best_member=init_x[0],
            best_fitness= coeff * jnp.inf,
            gen_counter=0,
        )

    def ask_strategy(
        self,
        rng: jax.Array,
        state: EvoState,
        params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        n_variation = int(self.popsize * self.variation_percentage)
        n_mutation = self.popsize - n_variation

        if n_variation > 0:
            rng1, rng2, rng = jr.split(rng, 3)
            x1 = sample_genotypes(rng1, state.population, n_variation)
            x2 = sample_genotypes(rng2, state.population, n_variation)
            x_variation, _ = polynomial_crossover(x1, x2, rng, self.cross_over_ratio)

        if n_mutation > 0:
            rng1, rng2 = jr.split(rng)
            x1 = sample_genotypes(rng1, state.population, n_mutation)

            if self.mutation_prob is None:
                mutation_prob = 1 / x1.shape[1]
            else:
                mutation_prob = self.mutation_prob

            x_mutation, _ = discrete_mutation(x1, rng2, mutation_prob, self.min_val, self.max_val)

        if n_variation == 0:
            genotypes = x_mutation  # type: ignore
        elif n_mutation == 0:
            genotypes = x_variation  # type: ignore
        else:
            genotypes = jax.tree_util.tree_map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_variation,  # type: ignore
                x_mutation,  # type: ignore
            )

        return genotypes, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams
    ) -> EvoState:
        elite_pop = int(self.popsize * self.elite_ratio)
        n_new = self.popsize - elite_pop

        elite = state.population[:elite_pop]
        elite_fitness = state.fitness[:elite_pop]

        all_fitness = jnp.concatenate([state.fitness[elite_pop:], fitness])

        best_idx = jnp.argsort(all_fitness)
        if self.fitness_shaper.maximize:
            best_idx = best_idx[-n_new:]
        else:
            best_idx = best_idx[:n_new]

        new_fitness = all_fitness[best_idx]
        new_members = jnp.concatenate([state.population[elite_pop:], x])[best_idx]

        new_fitness = jnp.concatenate([elite_fitness, new_fitness])
        new_population = jnp.concatenate([elite, new_members])

        sort_idx = jnp.argsort(new_fitness)
        if self.fitness_shaper.maximize:
            sort_idx = sort_idx[::-1]

        new_fitness = new_fitness[sort_idx][:self.popsize]
        new_population = new_population[sort_idx][:self.popsize]

        return state.replace(population=new_population, fitness=new_fitness)

    @property
    def params_strategy(self):
        return EvoParams(self.min_val, self.max_val)
