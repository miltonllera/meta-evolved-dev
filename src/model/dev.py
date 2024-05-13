from abc import abstractmethod

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Callable, Iterable, Optional, Tuple, Union
from typing_extensions import Self
from jaxtyping import Array, Float, Int, PyTree

from src.model.base import FunctionalModel
from src.nn.dna import DNAGenerator
from src.nn.graph import Graph
from src.utils import Tensor
from bin.init.config import load_model_weights

State = Tuple[Float[Array, "..."], PyTree, Float[Array, "..."], Int, jax.Array]
# class State(namedtuple):
#     """
#     Describe the state of model computations across a developmental + policy evaluation run.
#     """
#     inputs: Float[Array, "..."]
#     node_states: PyTree
#     dev_steps: Int
#     rng_key: jax.Array


class DevelopmentalModel(FunctionalModel):
    """
    A generic developmental model which takes an input encoding (aka goal, "DNA", etc.) and
    produces an output by growing it over several steps.
    """
    dev_steps: Union[Int, Tuple[int, int]]
    context_encoder: Callable[[Tensor], Tensor]
    output_decoder: Callable[[PyTree, jax.Array], PyTree]
    inference: bool
    output_dev_steps: bool

    def __call__(
        self,
        inputs: Tensor,
        key: jax.Array,
        # state: Optional[State] = None
    ) -> Tuple[Tensor, PyTree]:
        return self.rollout(inputs, key)

    def rollout(
        self,
        inputs: Float[Array, "..."],
        key: jax.Array,
        # state: Optional[State] = None,  # use this to intervene on state values during analysis
    )-> Tuple[Tensor, PyTree]:
        if isinstance(self.dev_steps, (tuple, list)):
            max_dev_steps = self.dev_steps[1]
        else:
            max_dev_steps = self.dev_steps

        init_state = self.init(inputs, key)
        # if state is None:
        #     init_state = self.init(inputs, key)
        # else:
        #     init_state = state

        final_state, cell_states = jax.lax.scan(self.step, init_state, jnp.arange(max_dev_steps))

        output = self.output_decoder(final_state[1], key=final_state[-1])  # type: ignore

        return output, cell_states

    @abstractmethod
    def step(self, carry: State, _, *args) -> Tuple[State, Iterable[State]]:
        raise NotImplementedError

    def return_dev_states(self, mode: bool) -> Self:
        return eqx.tree_at(lambda x: x.output_dev_states, self, mode)

    def sample_generation_steps(self, key: jax.Array):
        if isinstance(self.dev_steps, tuple):
            steps = jax.lax.cond(
                self.inference,
                lambda: self.dev_steps[1],
                lambda: jr.choice(key, jnp.arange(*self.dev_steps)).squeeze(),  # type: ignore
            )
        else:
            steps = self.dev_steps
        return steps


#-------------------------------------------- NCA -------------------------------------------------

class NCA(DevelopmentalModel):
    """
    Neural Cellular Automata based on Mordvintsev et al. (2020) which supports using goal-directed
    generation as in Shudhakaran et al. (2022).

    This class assumes a grid like organization where cell states occupy the leading dimension of
    the vectors. This means that we can use convolution operations for the updates themselves and
    any function that rearranges the dimensions internally must reverse this process when returning
    the results back to the NCA class.
    """
    state_size: int
    grid_size: Tuple[int, int]
    alive_fn: Callable
    control_fn: Callable
    message_fn: Callable
    update_fn: Callable
    state_norm: Callable
    # hyperparams
    update_prob: float

    def __init__(
        self,
        state_size: int,
        grid_size: Tuple[int, int],
        dev_steps: int,
        context_encoder: Callable,
        alive_fn: Callable,
        control_fn: Callable,
        message_fn: Callable,
        update_fn: Callable,
        update_prob: float,
        output_decoder: Optional[Callable],
        state_norm: Optional[Callable] = None,
        output_dev_steps: bool = False
    ):
        if not update_prob > 0.0:
            raise ValueError("Update probability must be greater than 0")

        if isinstance(dev_steps, Iterable):
            dev_steps = tuple(dev_steps)  # type: ignore

        if output_decoder is None:
            output_decoder = lambda x, key: x

        if state_norm is None:
            state_norm = lambda x: x

        super().__init__(dev_steps, context_encoder, output_decoder, False, output_dev_steps)

        self.state_size = state_size
        self.grid_size = grid_size
        self.alive_fn = alive_fn
        self.control_fn = control_fn
        self.message_fn = message_fn
        self.update_fn = update_fn
        self.state_norm = state_norm
        self.update_prob = update_prob

    def init(self, inputs, key):
        H, W = self.grid_size

        key, init_key = jr.split(key)

        # TODO: Random initialization doesn't seem to work. Cells tend to die or their values diverge.
        # init_states = jnp.zeros((self.state_size, H, W))
        # # random initialization of cell
        # seed = (0.5 * jr.normal(init_key, (self.state_size,))).at[3].set(1)
        # init_states = init_states.at[:, H//2, W//2].set(seed)

        dna_seq_length = self.context_encoder.input_shape[0]
        init_states = jnp.zeros((self.state_size, H, W)).at[:, H//2, W//2].set(1.0)
        init_weights = jnp.zeros_like(init_states, shape=(dna_seq_length, H, W))
        n_dev_steps = self.sample_generation_steps(init_key)

        return (self.context_encoder(inputs), init_states, init_weights, n_dev_steps, key)

    def step(self, state: State, i: int) -> Tuple[State, Tensor]:
        dev_steps = state[-2]

        def update(state):
            # TODO: Currently not using the previous control weights, maybe they should be used
            context, old_states, _, dev_steps, key = state

            updt_key, ctx_key, carry_key = jr.split(key, 3)

            pre_alive_mask = self.alive_fn(old_states)

            flattened = old_states.reshape(self.state_size, -1).transpose(1, 0)
            ctrl_signal, ctrl_weights = self.control_fn(flattened, context, key=ctx_key)

            ctrl_signal = ctrl_signal.transpose(1, 0).reshape(-1, *self.grid_size)
            ctrl_weights = ctrl_weights.transpose(1, 0).reshape(-1, *self.grid_size)

            message_vectors = self.message_fn(old_states, ctrl_signal, pre_alive_mask)

            updates = self.update_fn(message_vectors)
            new_states = self.state_norm(
                old_states + updates * self.stochastic_update_mask(updt_key)
            )

            alive_mask = (self.alive_fn(new_states) & pre_alive_mask).astype(jnp.float32)
            new_states = new_states * alive_mask

            # We just need the final output for training, so we use this flag to disable the rest
            if self.output_dev_steps:
                outputs = (new_states, ctrl_weights * pre_alive_mask)
            else:
                outputs = None

            return (context, new_states, ctrl_weights, dev_steps, carry_key), outputs

        no_update = lambda state: (state, (state[1], state[2]) if self.output_dev_steps else None)

        # NOTE: in this case 'jax.cond' executes both branches during evaluation since the
        # functions are not dependent on the input. Passing i as a parameter could make it
        # short-circuit, but this does not work inside a vmap --- it will be converted into
        # a call to 'jax.select' which always executes both branches:
        #    https://github.com/google/jax/issues/3103, #issuecomment-1716019765
        return jax.lax.cond(i < dev_steps, update, no_update, state,)

    def stochastic_update_mask(self, key: jax.Array):
        if self.update_prob < 1.0:
            return jr.bernoulli(key, self.update_prob, (1, *self.grid_size))
        else:
            return jnp.ones((1, *self.grid_size))


#-------------------------------------------- DEV + DNA -------------------------------------------

class DNAGuidedDevModel(eqx.Module):
    dev: DevelopmentalModel
    dna_generator: DNAGenerator

    def __init__(self, dev: DevelopmentalModel, dna_generator: DNAGenerator):
        self.dev = dev
        self.dna_generator = dna_generator

    def instantiate(self, params):
        return DNAGuidedDevModel(
            self.dev.instantiate(params.dev),
            eqx.combine(params.dna_generator, self.dna_generator)
        )

    def partition(self):
        nca_params, nca_statics = self.dev.partition()
        dna_params, dna_statics = eqx.partition(self.dna_generator, eqx.is_array)
        return DNAGuidedDevModel(nca_params, dna_params), DNAGuidedDevModel(nca_statics, dna_statics)

    def __iter__(self):
        yield self.dev
        yield self.dna_generator


class TargetDevModel(eqx.Module):
    """
    Wrapper class to optimize the indirect encodings inputs to a DevelopmentalModel.

    This wrapper class enables optimizing a developmental model with the standard EvoTrainer, which
    expects a set of parameters to optimize. In this case, it is the inputs with respect to which
    we wisht to optimize instead of the NCA parameters. We thus define them as a parameter of the
    model and return them as the first output of the partition function. Instead inputs are fed in
    the instantiate methods which sets the optimized parameters of the model.
    """
    eval: Callable
    dna: Float[Array, "S"]

    def __init__(
        self,
        dna_shape: Tuple[int, ...],
        dev: DevelopmentalModel
    ):
        self.dev = dev
        self.dna = jnp.zeros(dna_shape)

    def __call__(self, key):
        return self.eval(self.dna, key)

    def __iter__(self):
        yield self.dna
        yield self.eval

    def partition(self):
        return self.dna, eqx.tree_at(lambda x: x.dna, self, None)

    def instantiate(self, dna):
        return eqx.tree_at(lambda x: x.dna, self, dna)

    @classmethod
    def load_from_dna_guided_dev_model(cls, dgd: DNAGuidedDevModel, path=None):
        # TODO: Load weights from path
        if path is not None:
            dgd_statics = dgd.partition()[1]
            dgd = eqx.combine(dgd_statics, load_model_weights(dgd_statics, path))
        return cls(dgd.dna_generator.dna_shape[:1], dgd.dev)


#---------------------------------------------- NDP ------------------------------------------------

class bNDP(DevelopmentalModel):
    """
    Baseline NDP model.

    This is Erwan's version of the NDP which uses a Graph Transformer Network to update the node
    embeddings and implements division at a local level --- each node decides if it wants to divide
    or not. It has been augmented with context and control functions which can be used to direct
    the NDA using a DNA. This control signal is fed only to the node updates, not the edge updates.
    """
    max_nodes: int
    node_features: int
    edge_features: int
    node_fn: Callable[[Graph, jax.Array], Graph]
    edge_fn: Callable[[Graph, jax.Array], Tensor]
    div_fn: Callable[[Tensor], Tensor]
    control_fn: Callable

    def __init__(
        self,
        node_features: int,
        max_nodes: int,
        dev_steps: int,
        context_encoder: Callable,
        control_fn: Callable,
        node_fn: Callable[[Graph, jax.Array], Graph],
        edge_fn: Callable[[Graph, jax.Array], Tensor],
        div_fn: Callable[[Tensor], Tensor],
        output_decoder: Optional[Callable],
        output_dev_steps: bool = False
    ):
        if isinstance(dev_steps, Iterable):
            dev_steps = tuple(dev_steps)  # type: ignore

        if output_decoder is None:
            output_decoder = lambda x, key: x

        super().__init__(dev_steps, context_encoder, output_decoder, False, output_dev_steps)

        self.node_features = node_features
        self.edge_features = 3
        self.max_nodes = max_nodes
        self.control_fn = control_fn
        self.node_fn = node_fn
        self.div_fn = div_fn
        self.edge_fn = edge_fn

    def init(self, inputs, key):
        key, init_key = jr.split(key)

        init_states = Graph(
            nodes=jnp.zeros((self.max_nodes, self.node_features)),
			adj=jnp.zeros((self.max_nodes, self.max_nodes)).astype(bool).at[0,0].set(True),
			edges=jnp.zeros((self.max_nodes, self.max_nodes, self.edge_features)),
			mask=jnp.zeros((self.max_nodes,)).at[0].set(1.).astype(bool)
        )

        dna_seq_length = self.context_encoder.input_shape[0]
        init_context_weights = jnp.zeros_like(
            init_states.nodes, shape=(self.max_nodes, dna_seq_length)
        )
        n_dev_steps = self.sample_generation_steps(init_key)

        return (self.context_encoder(inputs), init_states, init_context_weights, n_dev_steps, key)

    def step(self, state: State, i: int) -> Tuple[State, Iterable[State]]:
        dev_steps = state[-2]

        def update(state):
            # TODO: Currently not using the previous control weights, maybe they should be used
            context, old_graph, _, dev_steps, key = state
            pre_alive_mask = old_graph.mask

            ctx_key, updt_key, div_key, edge_updt_key, carry_key = jr.split(key, 5)

            # Compute the control signal for each node
            ctrl_signal, ctrl_weights = self.control_fn(old_graph.nodes, context, key=ctx_key)
            ctrl_signal = ctrl_signal * pre_alive_mask[:, None]

            # CHECK: May need to add normalization here
            graph = self.node_fn(
                old_graph._replace(
                    nodes=old_graph.nodes + ctrl_signal,
                    edges=jnp.concatenate([
                        old_graph.adj[..., None],
                        old_graph.adj.T[..., None],
                        jnp.identity(self.max_nodes)[..., None]
                    ], axis=-1
                )),
                key=updt_key  # type: ignore
            )

            # Compute the probability that each node will divide.
            ps_div = jax.vmap(self.div_fn)(graph.nodes).squeeze() * pre_alive_mask
            to_divide = jr.uniform(div_key, ps_div.shape) < ps_div

            # Update the mask and adjacency matrix to include the new nodes. Note that we have to
            # take into account if there is still capacity left when addding new nodes, which we
            # keep track of using the add_mask.
            pre_alive_count = pre_alive_mask.sum()

            # compute child_idx[parent_idx] = new_node_idx
            child_idx = (
                # broadcast the pre_alive_count to the parents that are dividing
                (pre_alive_count * to_divide) +
                # each new node is assigned an index starting from the already existing node count
                jnp.where(to_divide, jnp.cumsum(to_divide) - 1, 0)
            )

            post_alive_mask = jnp.arange(self.max_nodes) < (pre_alive_count + to_divide.sum())
            new_node_mask = post_alive_mask ^ pre_alive_mask  # bitwise xor

            # Set new nodes' incoming connection from their parents. segment_sum will return a
            # matrix of outgoing edges for each new node, so we need to transpose it.
            # new_adj = jax.ops.segment_sum(jnp.identity(self.max_nodes), child_idx, self.max_nodes)
            # post_adj = jnp.where(new_node_mask[None], new_adj.T.astype(bool), graph.adj)

            # NOTE: I find the use of segment_sum a bit hard to follow so I switched to straight
            # assingment of the adjacency values. I am also using the first node to dump dummy
            # edges in child_idx as these entries will never be set to True in new_node_mask.
            new_adj = jnp.zeros_like(graph.adj).at[jnp.arange(self.max_nodes), child_idx].set(True)
            post_adj = jnp.where(new_node_mask[None], new_adj, graph.adj)

            # Add and remove edges. First compute edge probabilities for each action and then
            # apply these by constructing generation and prune matrices.
            edge_fn_key, gen_key, prune_key = jr.split(edge_updt_key, 3)
            p_gen, p_prune = jnp.split(self.edge_fn(graph, edge_fn_key), (1,), axis=-1)

            # We can generate any potential edge that doesn't already exist. Note that this excludes
            # edges to/from newly created nodes. I can probably remove the bitwise_not.
            potential_edges = pre_alive_mask[None] * pre_alive_mask[:, None]
            p_gen = p_gen.squeeze() * potential_edges * ~graph.adj
            gen = jr.uniform(gen_key, p_gen.shape) < p_gen

            # we can remove edges that already exist, the ones to newly created nodes must be kept
            # so we use the original adjacency matrix.
            p_prune = p_prune.squeeze() * graph.adj
            prune = jr.uniform(prune_key, p_gen.shape) < p_prune

            # post_adj = (post_adj + gen - prune.astype(float)).astype(bool)
            post_adj = (post_adj | gen) ^ prune

            graph = graph._replace(adj=post_adj, mask=post_alive_mask)

            if self.output_dev_steps:
                outputs = (graph, ctrl_weights * pre_alive_mask[..., None])
            else:
                outputs = None

            return (context, graph, ctrl_weights, dev_steps, carry_key), outputs

        no_update = lambda state: (state, (state[1], state[2]) if self.output_dev_steps else None)

        return jax.lax.cond(i < dev_steps, update, no_update, state)


class StagedNDP(DevelopmentalModel):
    max_nodes: int
    node_features: int
    edge_features: int
    div_heads: int
    control_fn: Callable
    update_fn: Callable
    div_fn: Callable[[Tensor], Tensor]
    pos_fn: Callable[[Tensor], Tensor]
    # node_fn: Callable[[Graph, jax.Array], Graph]

    def __init__(
        self,
        node_features: int,
        max_nodes: int,
        dev_steps: int,
        div_heads: int,
        context_encoder: Callable,
        control_fn: Callable,
        update_fn: Callable,
        div_fn: Callable[[Tensor], Tensor],
        pos_fn: Callable[[Tensor], Tensor],
        # node_fn: Callable[[Graph, jax.Array], Graph],
        output_decoder: Optional[Callable],
        output_dev_steps: bool = False
    ):
        if isinstance(dev_steps, Iterable):
            dev_steps = tuple(dev_steps)  # type: ignore

        if output_decoder is None:
            output_decoder = lambda x, key: x

        super().__init__(dev_steps, context_encoder, output_decoder, False, output_dev_steps)

        self.node_features = node_features
        self.edge_features = 3
        self.div_heads = div_heads
        self.max_nodes = max_nodes
        self.control_fn = control_fn
        self.update_fn = update_fn
        # self.node_fn = node_fn
        self.div_fn = div_fn
        self.pos_fn = pos_fn

    def init(self, inputs, key):
        key, init_key = jr.split(key)

        context = self.context_encoder(inputs)
        # for this model, the state is the graph and the generator nodes
        prog_state = jr.uniform(
            init_key, (self.div_heads, self.node_features), minval=-0.1, maxval=0.1
        )

        net_state = (
            jnp.zeros((self.max_nodes + 1, self.node_features)),  # use one extra node to dump unused nodes
            jnp.zeros((self.max_nodes + 1, self.max_nodes, self.edge_features)),
        )

        dna_seq_length = self.context_encoder.input_shape[0]
        init_context_weights = jnp.zeros_like(
            net_state[0], shape=(self.div_heads, dna_seq_length)
        )
        n_dev_steps = self.sample_generation_steps(init_key)

        return (context, net_state, prog_state, init_context_weights, n_dev_steps, key)

    def step(self, state: State, i: int) -> Tuple[State, Iterable[State]]:
        dev_steps = state[-2]

        def update(state):
            context, (node_intrinsic, pre_alive_mask), prog_state, _, dev_steps, key = state

            ctx_key, div_key, carry_key = jr.split(key, 3)

            # Compute the control signal for each node
            ctrl_signal, ctrl_weights = self.control_fn(prog_state, context, key=ctx_key)
            new_prog_state = prog_state + self.update_fn(prog_state + ctrl_signal)

            div_vector = jax.vmap(self.div_fn)(new_prog_state)

            assert div_vector.shape[-1] == (self.node_features + 4)  # will be compiled out by jax
            div_bool, pos_xyz, new_intrinsic = jnp.split(div_vector, [1, 4], axis=-1)
            # NOTE: may want to further process node intrinsic here

            pre_alive_count = pre_alive_mask.sum()
            to_divide = jr.uniform(div_key, div_bool.shape) < jax.nn.sigmoid(div_bool)

            # set index of progenitors that won't divide to last position so they can be dumped
            # also do not exceed the total capacity of the system.
            new_node_idx = pre_alive_count + to_divide.cumsum() - 1
            new_node_idx = jnp.where(to_divide & (new_node_idx < self.max_nodes), new_node_idx, -1)

            # NOTE: We can test whether position embeddings work by making pos_fn return 0.
            new_intrinsic = jax.vmap(self.pos_fn)(pos_xyz) + new_intrinsic
            new_intrinsic = node_intrinsic.at[new_node_idx].set(new_intrinsic * to_divide)

            post_alive_mask = jnp.arange(self.max_nodes) < (pre_alive_count + to_divide.sum())

            if self.output_dev_steps:
                outputs = ((new_intrinsic, post_alive_mask), ctrl_weights * pre_alive_mask[..., None])
            else:
                outputs = None

            return (
                context,
                (new_intrinsic, post_alive_mask),
                ctrl_weights, dev_steps, carry_key
            ), outputs

        no_update = lambda state: (state, (state[1], state[2]) if self.output_dev_steps else None)

        # TODO: we must remove the dummy node state as well.
        return jax.lax.cond(i < dev_steps, update, no_update, state)
