# Code from https://github.com/Reytuag/transformerXL_PPO_JAX/blob/main/rel_multi_head.py


###########################################
# https://github.com/Reytuag/transformerXL_PPO_JAX/blob/main/rel_multi_head.py itself includes
# the following attributions, which are preserved here:


# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# CODE IS HEAVILY INSPIRED FROM https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/deprecated/transfo_xl/modeling_transfo_xl.py
# MOST OF THE TIME JUST A CONVERSION IN JAX


"""Relative Attention HEAVILY INSPIRED FROM https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/deprecated/transfo_xl/modeling_transfo_xl.py
, flax attention, https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py#L143, most of the time just a flax/jax conversion"""

###########################################

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import DenseGeneral, DotGeneralT, PrecisionLike, default_kernel_init
from flax.linen.module import Module, compact, merge_param
from jax import lax, random

PRNGKey = Any
Shape = tuple[int, ...]
Dtype = Any
Array = Any

roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)


def dot_product_attention_weights(
    query: Array,
    key: Array,
    r_pos_embed,
    r_r_bias,
    r_w_bias,
    bias: Array | None = None,
    mask: Array | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: PRNGKey | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
):
    """Computes dot-product attention weights given query and key."""
    query, key = promote_dtype(query, key, dtype=dtype)
    dtype = query.dtype

    assert query.ndim == key.ndim, "q, k must have same rank."
    assert query.shape[:-3] == key.shape[:-3], "q, k batch dims must match."
    assert query.shape[-2] == key.shape[-2], "q, k num_heads must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    depth = query.shape[-1]

    attn_weights = jnp.einsum("...qhd,...khd->...hqk", query + r_w_bias, key, precision=precision)

    attn_weights_r = jnp.einsum("...qhd,khd->...hqk", query + r_r_bias, r_pos_embed, precision=precision)

    attn_weights_r = roll_vmap(attn_weights_r, jnp.arange(0, query.shape[-3]) - (query.shape[-3] - 1), -1)
    attn_weights = attn_weights + attn_weights_r

    attn_weights = attn_weights / jnp.sqrt(depth).astype(dtype)

    if bias is not None:
        attn_weights = attn_weights + bias
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
            keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        else:
            keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attn_weights = attn_weights * multiplier

    return attn_weights


def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    r_pos_embed,
    r_r_bias,
    r_w_bias,
    bias: Array | None = None,
    mask: Array | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: PRNGKey | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
):
    """Computes dot-product attention given query, key, and value."""
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    # compute attention weights
    attn_weights = dot_product_attention_weights(
        query,
        key,
        r_pos_embed,
        r_r_bias,
        r_w_bias,
        bias,
        mask,
        broadcast_dropout,
        dropout_rng,
        dropout_rate,
        deterministic,
        dtype,
        precision,
    )

    # return weighted sum over values for each query position
    return jnp.einsum("...hqk,...khd->...qhd", attn_weights, value, precision=precision)


class RelMultiHeadDotProductAttention(Module):
    """Multi-head dot-product attention."""

    num_heads: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    qkv_features: int | None = None
    out_features: int | None = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.0
    deterministic: bool | None = None
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    use_bias: bool = True
    attention_fn: Callable[..., Array] = dot_product_attention
    decode: bool = False
    qkv_dot_general: DotGeneralT = lax.dot_general
    out_dot_general: DotGeneralT = lax.dot_general

    @compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        pos_embed: Array,
        mask: Array | None = None,
        deterministic: bool | None = None,
    ):
        """Applies multi-head dot product attention on the input data."""
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, f"Memory dimension ({qkv_features}) must be divisible by number of heads ({self.num_heads})."
        head_dim = qkv_features // self.num_heads

        dense = functools.partial(
            DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            dot_general=self.qkv_dot_general,
        )
        query, key, value = (
            dense(name="query")(inputs_q),
            dense(name="key")(inputs_kv),
            dense(name="value")(inputs_kv),
        )

        dense_relpos = functools.partial(
            DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            use_bias=False,
            precision=self.precision,
            dot_general=self.qkv_dot_general,
        )

        r_pos_embed = dense_relpos(name="pos_embed_mat")(pos_embed)

        r_r_bias = self.param(
            "r_r_bias",
            self.bias_init,
            (self.num_heads, head_dim),
        )
        r_w_bias = self.param(
            "r_w_bias",
            self.bias_init,
            (self.num_heads, head_dim),
        )

        if self.decode:
            is_initialized = self.has_variable("cache", "cached_key")
            cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
            cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
            cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))
            if is_initialized:
                (
                    *batch_dims,
                    max_length,
                    num_heads,
                    depth_per_head,
                ) = cached_key.value.shape
                expected_shape = (*tuple(batch_dims), 1, num_heads, depth_per_head)
                if expected_shape != query.shape:
                    raise ValueError("Autoregressive cache shape error, expected query shape %s instead got %s." % (expected_shape, query.shape))

                cur_index = cache_index.value
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(cached_value.value, value, indices)
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1

                mask = combine_masks(
                    mask,
                    jnp.broadcast_to(
                        jnp.arange(max_length) <= cur_index,
                        (*tuple(batch_dims), 1, 1, max_length),
                    ),
                )

        dropout_rng = None
        if self.dropout_rate > 0.0:
            m_deterministic = merge_param("deterministic", self.deterministic, deterministic)
            if not m_deterministic:
                dropout_rng = self.make_rng("dropout")
        else:
            m_deterministic = True

        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
            r_pos_embed,
            r_r_bias,
            r_w_bias,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=m_deterministic,
            dtype=self.dtype,
            precision=self.precision,
        )
        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            dot_general=self.out_dot_general,
            name="out",
        )(x)
        return out


def combine_masks(*masks: Array | None, dtype: Dtype = jnp.float32) -> Array:
    """Combine attention masks."""
    masks_list = [m for m in masks if m is not None]
    if not masks_list:
        return None
    assert all(x.ndim == masks_list[0].ndim for x in masks_list), f"masks must have same rank: {tuple(x.ndim for x in masks_list)}"
    mask, *other_masks = masks_list
    for other_mask in other_masks:
        mask = jnp.logical_and(mask, other_mask)
    return mask.astype(dtype)
