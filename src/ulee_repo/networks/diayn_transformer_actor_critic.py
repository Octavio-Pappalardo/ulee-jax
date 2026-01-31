from typing import Any, Protocol

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import constant, orthogonal
from flax.linen.linear import default_embed_init
from flax.typing import Dtype, Initializer

from ulee_repo.networks.tile_embedding import Embedding_xland_map
from ulee_repo.networks.transformer_xl_base import Transformer_XL

# ---------------- Observation Encoder ----------------------------


class Encoder_observations_and_skills(nn.Module):
    obs_emb_dim: int
    num_skills: int
    skill_emb_dim: int

    @nn.compact
    def __call__(self, observations, skills) -> jax.Array:
        img_encoder = nn.Sequential(
            [
                Embedding_xland_map(emb_dim=self.obs_emb_dim),
                nn.Conv(32, (2, 2), padding="VALID", kernel_init=orthogonal(np.sqrt(2))),
                nn.relu,
                nn.Conv(32, (2, 2), padding="VALID", kernel_init=orthogonal(np.sqrt(2))),
                nn.relu,
                nn.Conv(64, (2, 2), padding="VALID", kernel_init=orthogonal(np.sqrt(2))),
                nn.relu,
                nn.Conv(features=64, kernel_size=(1, 1)),
                nn.relu,
            ]
        )

        skill_encoder = nn.Embed(self.num_skills, self.skill_emb_dim)

        B, S = observations.shape[:2]  # [batch_size, seq_len]
        obs_emb = img_encoder(observations).reshape(B, S, -1)
        skill_emb = skill_encoder(skills)
        input_encoding = jnp.concatenate([obs_emb, skill_emb], axis=-1)

        return input_encoding


# ----------------- Custom Fixed (frozen) Embedding layer ----------------------------


class PromoteDtypeFn(Protocol):
    def __call__(self, *args: jax.Array | None, dtype: Any = None, inexact: bool = True) -> list[jax.Array | None]: ...


class FrozenEmbed(nn.Module):
    num_embeddings: int
    features: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    embedding_init: Initializer = default_embed_init
    promote_dtype: PromoteDtypeFn = promote_dtype

    @nn.compact
    def __call__(self, inputs):
        if self.has_variable("constants", "embedding"):
            embedding = self.get_variable("constants", "embedding")
        else:
            rng = self.make_rng("constants")
            embedding = self.variable(
                "constants",
                "embedding",
                self.embedding_init,
                rng,
                (self.num_embeddings, self.features),
                self.param_dtype,
            ).value

        # Identical behavior to nn.Embed.__call__()
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            msg = "Input type must be an integer or unsigned integer."
            raise ValueError(msg)
        (embedding,) = self.promote_dtype(embedding, dtype=self.dtype, inexact=False)
        if self.num_embeddings == 1:
            return jnp.broadcast_to(embedding, (*inputs.shape, self.features))
        return jnp.take(embedding, inputs, axis=0)


def normal_unit_rows(key, shape, dtype=jnp.float32, scale: float = 1.0, eps: float = 1e-8):
    x = jax.random.normal(key, shape, dtype)
    norms = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return (x / (norms + eps)) * scale


# ----------------- Diayn Transformer Actor Critic ---------------------------


class DiaynActorCriticTransformer(nn.Module):
    num_actions: int
    # encoder
    obs_emb_dim: int
    num_skills: int
    skill_emb_dim: int
    # transformer
    hidden_dim: int
    num_attn_heads: int
    qkv_features: int
    num_layers_in_transformer: int
    gating: bool
    gating_bias: float
    # actor and critic heads
    head_activation: str
    mlp_dim: int
    # skill bias term
    skill_bias: bool = False
    skill_bias_scale: float = 5.0

    def setup(self):
        self.encoder = Encoder_observations_and_skills(
            obs_emb_dim=self.obs_emb_dim,
            num_skills=self.num_skills,
            skill_emb_dim=self.skill_emb_dim,
        )

        self.transformer = Transformer_XL(
            hidden_dim=self.hidden_dim, num_heads=self.num_attn_heads, qkv_features=self.qkv_features, num_layers=self.num_layers_in_transformer, gating=self.gating, gating_bias=self.gating_bias
        )

        if self.head_activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

        self.actor_linear1 = nn.Dense(self.mlp_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.actor_linear2 = nn.Dense(self.mlp_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.actor_out = nn.Dense(self.num_actions, kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "uniform"), bias_init=constant(0.0))

        if self.skill_bias:
            self.actor_bias_table = FrozenEmbed(
                num_embeddings=self.num_skills, features=self.num_actions, embedding_init=orthogonal(self.skill_bias_scale)
            )  # normal_unit_rows initialization also works well

        self.critic_linear1 = nn.Dense(self.mlp_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.critic_linear2 = nn.Dense(self.mlp_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.critic_out = nn.Dense(1, kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "uniform"), bias_init=constant(0.0))

    def __call__(self, memories, obs, skill, mask):
        # obs and skill should have shapes [batch_size, 1, ...]
        encoded_input = self.encoder(obs, skill)

        # Transformer_XL.__call__() expects the input argument to be of shape [batch_size, input_dim]
        # input_dim does't have to match transformer hidden dim (the transformer applies a linear mapping to that size internally)
        x, memory_out = self.transformer(memories, encoded_input.squeeze(axis=1), mask)

        # x has shape [batch_size, hidden_dim]
        actor = self.actor_linear1(x)
        actor = self.activation_fn(actor)
        actor = self.actor_linear2(actor)
        actor = self.activation_fn(actor)
        actor = self.actor_out(actor)

        if self.skill_bias:
            skill_bias = self.actor_bias_table(skill)  # [batch_size, num_actions]
            pi = distrax.Categorical(logits=actor + skill_bias.squeeze(axis=1))
        else:
            pi = distrax.Categorical(logits=actor)

        critic = self.critic_linear1(x)
        critic = self.activation_fn(critic)
        critic = self.critic_linear2(critic)
        critic = self.activation_fn(critic)
        critic = self.critic_out(critic)

        return pi, jnp.squeeze(critic, axis=-1), memory_out

    def model_forward_eval(self, memories, obs, skill, mask):
        # obs and skill should have shapes [batch_size, 1, ...]
        encoded_input = self.encoder(obs, skill)

        x, memory_out = self.transformer.forward_eval(memories, encoded_input.squeeze(axis=1), mask)

        actor = self.actor_linear1(x)
        actor = self.activation_fn(actor)
        actor = self.actor_linear2(actor)
        actor = self.activation_fn(actor)
        actor = self.actor_out(actor)  # [batch_size, num_actions]
        if self.skill_bias:
            skill_bias = self.actor_bias_table(skill)  # [batch_size, num_actions]
            pi = distrax.Categorical(logits=actor + skill_bias.squeeze(axis=1))
        else:
            pi = distrax.Categorical(logits=actor)

        critic = self.critic_linear1(x)
        critic = self.activation_fn(critic)
        critic = self.critic_linear2(critic)
        critic = self.activation_fn(critic)
        critic = self.critic_out(critic)

        return pi, jnp.squeeze(critic, axis=-1), memory_out

    def model_forward_train(self, memories, obs, skill, mask):
        # encoder expects shapes: [batch_size, seq_len, ...]
        encoded_input = self.encoder(obs, skill)

        # Transformer_XL.forward_train() expects the input argument to be of shape [batch_size, seq_len, input_dim]
        x = self.transformer.forward_train(memories, encoded_input, mask)

        actor = self.actor_linear1(x)
        actor = self.activation_fn(actor)
        actor = self.actor_linear2(actor)
        actor = self.activation_fn(actor)
        actor = self.actor_out(actor)
        if self.skill_bias:
            skill_bias = self.actor_bias_table(skill)
            pi = distrax.Categorical(logits=actor + skill_bias)
        else:
            pi = distrax.Categorical(logits=actor)

        critic = self.critic_linear1(x)
        critic = self.activation_fn(critic)
        critic = self.critic_linear2(critic)
        critic = self.activation_fn(critic)
        critic = self.critic_out(critic)
        return pi, jnp.squeeze(critic, axis=-1)
