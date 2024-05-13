from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from jaxtyping import Array, Float, Bool

from src.nn.embedding import Embedding, PositionEmbedding
from src.nn.attn import MultiheadAttention


class DNAContextEncoder(eqx.Module):
    """
    Class that takes a DNA string and converts it into a continuos embedding.
    """
    embedding: Embedding
    position_embedding: PositionEmbedding
    input_is_distribution: bool = eqx.static_field()

    def __init__(
        self,
        alphabet_size: int,
        sequence_length: int,
        embedding_size: int,
        input_is_distribution: bool = False,
        *,
        key: jax.Array,
    ):
        key_emb, key_pos = jr.split(key, 2)

        self.embedding = Embedding(alphabet_size, embedding_size, key_emb)
        self.position_embedding = PositionEmbedding(sequence_length, embedding_size, key_pos)
        self.input_is_distribution = input_is_distribution

    @property
    def alphabet_size(self):
        return self.embedding.alphabet_size

    @property
    def dna_seq_length(self):
        return self.position_embedding.max_sequence_size

    @property
    def dna_shape(self):
        return self.dna_seq_length, self.alphabet_size

    @property
    def input_shape(self):
        return self.dna_seq_length, self.alphabet_size

    @property
    def total_input_size(self):
        return self.dna_seq_length * self.alphabet_size

    def __call__(self, inputs: Float[Array, "S A"], key: jax.Array = None):
        if self.input_is_distribution:
            inputs = inputs.reshape(*self.dna_shape)
            idxs = inputs.argmax(1)
        else:
            idxs = inputs

        inputs = jnn.one_hot(idxs, self.alphabet_size)

        return self.position_embedding(self.embedding(inputs))


DNA_MASK = Union[None, Bool[Array,"q_seq kv_seq"], Bool[Array,"n_heads q_seq kv_seq"]]

class DNAControl(eqx.Module):
    """
    Implementation of cross attention for DNA decoding in Celluar Automatas.
    """
    attention: nn.MultiheadAttention
    dna_mask: DNA_MASK

    def __init__(
        self,
        state_size: int,
        dna_emb_size: int,
        output_size: Optional[int] = None,
        n_heads: int = 1,
        dna_mask: DNA_MASK = None,
        *,
        key: jax.Array
    ):
        if output_size is None:
            output_size = state_size

        self.dna_mask = dna_mask  # use mask to inhibit specific genes
        self.attention = MultiheadAttention(
            n_heads, state_size, dna_emb_size, dna_emb_size, output_size, state_size, key=key
        )

    def __call__(
        self,
        inputs: Float[Array, "N H"],
        dna: Float[Array, "S E"],
        key: jax.Array
    ) -> Tuple[Float[Array, "N H"], Float[Array, "S E"]]:
        attn, attn_weights = self.attention(
            inputs, dna, dna, mask=self.dna_mask, key=key
        )
        return attn, attn_weights


class DNAGenerator(eqx.Module, ABC):
    dna_shape: Tuple[int, int]

    @abstractmethod
    def sample_dna(self, key):
        raise NotImplementedError

    def __call__(self, n_samples, *, key):
        return jax.vmap(self.sample_dna)(jr.split(key, n_samples))


class DNAIndependentSampler(DNAGenerator):
    """
    Sample DNA strings by independently sampling each character in the sequence. This uses a
    normal distribution whose parameters can be fitted as part of a model.
    """
    logits_mean: Float[Array, "S A"]
    logits_logvar: Float[Array, "S A"]
    return_raw_probabilities: bool

    def __init__(
        self,
        sequence_length: int,
        alphabet_size: int,
        *,
        return_raw_probabilities: bool = True,
        key: jax.Array,
    ):
        self.dna_shape = sequence_length, alphabet_size
        self.logits_mean = jr.normal(key, shape=(sequence_length, alphabet_size))
        self.logits_logvar = jr.normal(key, shape=(sequence_length, alphabet_size))
        self.return_raw_probabilities = return_raw_probabilities

    def sample_dna(self, key):
        std = jnp.exp(0.5  * self.logits_logvar)
        logits = self.logits_mean + std * jr.normal(key, self.dna_shape)
        if self.return_raw_probabilities:
            return jax.nn.softmax(logits)  # noramlize to prevent exploding variance
        return logits.argmax(axis=-1).astype(jnp.float32)

    def partition(self):
        return eqx.partition(self, eqx.is_array)


class DNAMaxEntropySampler(DNAGenerator):
    dna_shape: Tuple[int, int]
    return_raw_probabilities: bool

    def __init__(
        self,
        sequence_length: int,
        alphabet_size: int,
        *,
        key: jax.Array,
    ):
        self.dna_shape = sequence_length, alphabet_size

    def sample_dna(self, key):
        seqlen, alphabet_size = self.dna_shape
        return jr.choice(key, jnp.arange(0, alphabet_size + 1, 1.0, dtype=jnp.float32), (seqlen,))


class DNAList(eqx.Module):
    n_dnas: int
    dna_shape: Tuple[int, int]
    dna_list: Float[Array, "N S A"]

    def __init__(
        self,
        n_dnas: int,
        sequence_length: int,
        alphabet_size: int,
        key: jax.Array,
    ):
        self.n_dnas = n_dnas
        self.dna_shape = sequence_length, alphabet_size
        self.dna_list = jr.normal(key, shape=(n_dnas, sequence_length, alphabet_size))

    def __call__(self, popsize, *, key) -> Float[Array, "N S A"]:
        assert popsize == self.n_dnas
        return self.dna_list

    def partition(self):
        return eqx.partition(self, eqx.is_array)
