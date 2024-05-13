from functools import partial
from typing import Callable, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import jax.image as jimg
import equinox as eqx
import equinox.nn as nn
from jaxtyping import Array, Float


#------------------------------ Continuous Embedding Sampler ----------------------------

class VectorSampler(eqx.Module):
    goal_size: int
    mean: Float[Array, "S A"]
    logits_logvar: Float[Array, "S A"]

    def __init__(
        self,
        goal_size: int,
        *,
        key: jax.Array,
    ):
        self.goal_size = goal_size
        self.mean = jr.normal(key, shape=(goal_size,))
        self.logits_logvar = jr.normal(key, shape=(goal_size,))

    @property
    def return_raw_probabilities(self):
        return False

    def __call__(self, n_samples, *, key):
        return jax.vmap(self.sample_vector)(jr.split(key, n_samples))

    def sample_vector(self, key):
        std = jnp.exp(0.5  * self.logits_logvar)
        return self.mean + std * jr.normal(key, (self.goal_size,))

    def partition(self):
        return eqx.partition(self, eqx.is_array)


#----------------------------------- Alive Functions ------------------------------------

class ConstantAlive(eqx.Module):
    def __call__(self, node_states: Float[Array, "C H W"]):
        H, W = node_states.shape[1:]
        return jnp.ones_like(node_states, shape=(1, H, W), dtype=jnp.bool_)


class BitAlive(eqx.Module):
    alive_bit: int
    alive_threshold: float

    def __init__(self, alive_threshold, alive_bit, *, key=None):  # key is keyword only
        super().__init__()
        self.alive_bit = alive_bit
        self.alive_threshold = alive_threshold

    def __call__(self, node_states: Float[Array, "C H W"]):
        return jnn.sigmoid(node_states[self.alive_bit:self.alive_bit + 1]) > self.alive_threshold


class MaxPoolAlive(eqx.Module):
    alive_bit: int
    alive_threshold: float
    max_pool: nn.MaxPool2d

    def __init__(self, alive_threshold, alive_bit, *, key=None):  # key is keyword only
        super().__init__()
        self.alive_bit = alive_bit
        self.alive_threshold = alive_threshold
        self.max_pool = nn.MaxPool2d(3, 1, 1)

    def __call__(self, node_states: Float[Array, "C H W"]):
        pooling = self.max_pool(node_states[self.alive_bit:self.alive_bit + 1])
        return pooling > self.alive_threshold

#----------------------------------- Message Passing ------------------------------------

_conv2d = partial(jax.scipy.signal.convolve2d, mode='same')
_sobel_kernel_x = jnp.array([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ]) / 8.0
_sobel_kernel_y = _sobel_kernel_x.T


class IdentityAndSobelFilter(eqx.Module):
    kernel_size: Tuple[int, int]

    def __init__(self, kernel_size: int = 3, *, key=None):  # key is keyword only
        super().__init__()
        if kernel_size != 3:
            raise NotImplementedError
        self.kernel_size = kernel_size, kernel_size

    def __call__(
        self,
        inputs: Float[Array, "C H W"],
        control_signal: Float[Array, "C H W"],
        alive_mask: Float[Array, "1 H W"],
    ):
        inputs = inputs + control_signal * alive_mask.astype(jnp.float32)
        x_conv = jax.vmap(_conv2d, in_axes=(0, None))(inputs, _sobel_kernel_x)
        y_conv = jax.vmap(_conv2d, in_axes=(0, None))(inputs, _sobel_kernel_y)
        return jnp.concatenate([inputs, x_conv, y_conv], axis=0)


# _north_kernel = jnp.array([
#     [0.0, 1.0, 0.0],
#     [0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0],
# ])
# _west_kernel = jnp.array([
#     [0.0, 0.0, 0.0],
#     [1.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0],
# ])
# _south_kernel = _north_kernel[::-1]
# _east_kernel = _west_kernel[:, ::-1]


# class VonNeumanNeighborhood(eqx.Module):
#     def __call__(self, messages, *, key: Optional[jax.Array] = None):
#         messages_vectors = jnp.concatenate([
#             messages,
#             jax.vmap(_conv2d, in_axes=(0, None))(messages, _north_kernel),
#             jax.vmap(_conv2d, in_axes=(0, None))(messages, _east_kernel),
#             jax.vmap(_conv2d, in_axes=(0, None))(messages, _south_kernel),
#             jax.vmap(_conv2d, in_axes=(0, None))(messages, _west_kernel),
#         ], axis=0)
#         return messages_vectors


# class DiscreteMessageGeneration(eqx.Module):
#     message_dict: nn.Embedding
#     control_projection: nn.Linear
#     cardinal_conv: nn.Conv2d

#     def __init__(self,
#         n_messages: int,
#         message_size: int,
#         state_size: int,
#         control_size: int,
#         *,
#         key: jax.Array
#     ):
#         key1, key2, key3 = jr.split(key, 3)
#         self.message_dict = nn.Embedding(n_messages, message_size, key=key1)
#         self.input_projection = nn.Linear(state_size, n_messages, use_bias=False, key=key2)
#         self.control_projection = nn.Linear(control_size, n_messages, use_bias=False, key=key3)

#     def __call__(
#         self,
#         inputs: Float[Array, "S C"],
#         control_signal: Float[Array, "S 1"],
#     ):
#         # N, H, W  - logits over possible messages
#         selection_logits = jax.vmap(self.input_projection)(inputs)
#         control_logits = jax.vmap(self.input_projection)(control_signal)
#         message_idxs = (selection_logits - jnn.relu(control_logits)).argmax(axis=1, keepdims=True)
#         # M, H, W - message vectors
#         return jax.vmap(self.message_dict)(message_idxs)


#-------------------------------------- Update Layers --------------------------------------------

class ResidualMLPUpdate(nn.Sequential):
    def __call__(self, x: Array, *, key: Optional[jax.Array] = None) -> Array:
        updates = super().__call__(x, key=key)
        return x + updates


# class (eqx.Module):
#     control_projection: nn.Linear
#     message_mlp: nn.MLP

#     def __init__(
#         self,
#         state_size: int,
#         message_size: int,
#         control_size: int,
#         n_neighbors: int,
#         *,
#         key: jax.Array
#     ):
#         self.control_projection = nn.Linear(control_size, n_neighbors, use_bias=False, key=key)
#         self.message_mlp = nn.MLP(
#             message_size,
#             state_size,
#             width_size=3 * state_size,
#             depth=1,
#         )

#     def __call__(self, inputs, messages, *, key: Optional[jax.Array] = None):
#         masks = jnn.sigmoid(self.control_projection(inputs))
#         messages = messages.reshape(self.n_neighbors, -1)
#         messages = messages * jnp.expand_dims(masks, axis=1)

#         return inputs + jax.vmap(self.message_mlp, in_axes=(0, None, None))(messages.sum(axis=0))

#     @property
#     def n_neighbors(self):
#         return self.control_projection.out_features


#--------------------------------------- Dummy layers --------------------------------------------

class ConstantContextEncoder(eqx.Module):
    state_size: int
    grid_size: Tuple[int, int]

    def __call__(self, *_):
        return jnp.zeros((self.state_size, *self.grid_size))

    @property
    def input_shape(self):
        return (self.state_size,)


class IdentityControlFn(eqx.Module):
    def __call__(self, cell_state, input_embedding, *, key=None):
        # return jnp.repeat(input_embedding, 3, 0)
        return input_embedding, jnp.zeros_like(input_embedding)


class MLPContextEncoder(nn.MLP):
    def __init__(
        self,
        context_size: int,
        embedding_size: int,
        mlp_width_factor: int = 2,
        mlp_depth: int = 1,
        *,
        key: jax.Array,
    ):
        super().__init__(
            context_size,
            embedding_size,
            context_size * mlp_width_factor,
            mlp_depth,
            activation=jnn.relu,
            use_final_bias=False,
            key=key
        )

    @property
    def input_shape(self):
        return (self.layers[0].weight.shape[1],)


class MLPControlFn(eqx.Module):
    mlp: nn.MLP

    def __init__(
        self,
        context_size: int,
        state_size: int,
        mlp_width_factor: int = 2,
        mlp_depth: int = 1,
        *,
        key: jax.Array,
    ):
        input_size = context_size + state_size
        self.mlp = nn.MLP(
            context_size + state_size,
            state_size,
            input_size * mlp_width_factor,
            mlp_depth,
            activation=jnn.relu,
            use_final_bias=False,
            key=key
        )

    def __call__(
        self,
        state: Float[Array, "S C"],
        context: Float[Array, "E"],
        *,
        key = None
    ):
        context = jnp.repeat(context[None], state.shape[0], axis=0)
        inputs = jnp.concatenate([state, context], axis=1)
        dummy_weights = jnp.zeros_like(inputs, shape=(state.shape[0], context.shape[1]))
        return jax.vmap(self.mlp)(inputs), dummy_weights


#-------------------------------------- Output transforms -----------------------------------------

class SliceOutput(eqx.Module):
    dim: int
    start_idx: int
    end_idx: int
    squashing_function: Callable

    def __init__(self, dim, end_idx, start_idx=0, squashing_function=lambda x: x, *, key=None):
        super().__init__()

        self.dim = dim
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.squashing_function = squashing_function

    def __call__(self, inputs, *, key=None):
        outputs = jax.lax.slice_in_dim(inputs, self.start_idx, self.end_idx + 1, 1, self.dim)
        return self.squashing_function(outputs)



INTERPOLATION_METHOD=Literal[
    "nearest",
    "linear" ,
    "bilinear" ,
    "trilinear" ,
    "triangle",
    "cubic",
    "bicubic",
    "tricubic",
    "lanczos3",
    "lanczos5"
]

class InterpolationUpsample(eqx.Module):
    output_shape: Tuple[int, int]
    method: INTERPOLATION_METHOD
    antialias: bool

    def __init__(self, output_shape, method="lanczos5", antialias=True):
        self.output_shape = output_shape
        self.method = method  #type: ignore
        self.antialias = antialias

    def __call__(self, inputs: Float[Array, "C H W"], key=None):
        n_channels = inputs.shape[0]
        return jimg.resize(inputs, (n_channels, *self.output_shape), self.method, self.antialias)
