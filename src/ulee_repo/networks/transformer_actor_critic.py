from typing import TypedDict

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from ulee_repo.networks.tile_embedding import Embedding_xland_map
from ulee_repo.networks.transformer_xl_base import Transformer_XL

# ---------------- Observation Encoder ----------------------------


class Encoder_observations(nn.Module):
    obs_emb_dim: int

    @nn.compact
    def __call__(self, observations) -> jax.Array:
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

        B, S = observations.shape[:2]  # [batch_size, seq_len]
        obs_emb = img_encoder(observations).reshape(B, S, -1)

        return obs_emb


# ---------------- Meta learning Input Encoder ----------------------------


class ActorCriticInput(TypedDict):
    observation: jax.Array
    prev_action: jax.Array
    prev_reward: jax.Array
    prev_done: jax.Array


class Encoder_inputs(nn.Module):
    num_actions: int
    obs_emb_dim: int
    action_emb_dim: int

    @nn.compact
    def __call__(self, inputs: ActorCriticInput) -> jax.Array:
        # encoder from https://github.com/lcswillems/rl-starter-files/blob/master/model.py

        img_encoder = nn.Sequential(
            [
                Embedding_xland_map(emb_dim=self.obs_emb_dim),
                nn.Conv(16, (2, 2), padding="VALID", kernel_init=orthogonal(np.sqrt(2))),
                nn.relu,
                nn.Conv(32, (2, 2), padding="VALID", kernel_init=orthogonal(np.sqrt(2))),
                nn.relu,
                nn.Conv(64, (2, 2), padding="VALID", kernel_init=orthogonal(np.sqrt(2))),
                nn.relu,
            ]
        )
        action_encoder = nn.Embed(self.num_actions, self.action_emb_dim)

        B, S = inputs["observation"].shape[:2]
        obs_emb = img_encoder(inputs["observation"]).reshape(B, S, -1)
        act_emb = action_encoder(inputs["prev_action"])
        input_encoding = jnp.concatenate([obs_emb, act_emb, inputs["prev_reward"][..., None], inputs["prev_done"][..., None]], axis=-1)

        return input_encoding


# ----------------- Actor Critic Transformer  ---------------------------


class ActorCriticTransformer(nn.Module):
    algorithm_id: str
    num_actions: int
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
    # encoder
    obs_emb_dim: int
    action_emb_dim: int | None = None  # only for meta learning

    def setup(self):
        if self.algorithm_id == "standard_ppo":
            self.input_encoder = Encoder_observations(obs_emb_dim=self.obs_emb_dim)
        elif self.algorithm_id == "meta_learning":
            if self.action_emb_dim is None:
                msg = "action_emb_dim must be specified for setting up meta_learning network"
                raise ValueError(msg)
            self.input_encoder = Encoder_inputs(
                num_actions=self.num_actions,
                obs_emb_dim=self.obs_emb_dim,
                action_emb_dim=self.action_emb_dim,
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
        self.actor_out = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0))

        self.critic_linear1 = nn.Dense(self.mlp_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.critic_linear2 = nn.Dense(self.mlp_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.critic_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, memories, input, mask):
        # method just used in network initialization.
        encoded_input = self.input_encoder(input)
        x = self.transformer(memories, encoded_input.squeeze(axis=1), mask)

        actor = self.actor_linear1(x)
        actor = self.activation_fn(actor)
        actor = self.actor_linear2(actor)
        actor = self.activation_fn(actor)
        actor = self.actor_out(actor)
        pi = distrax.Categorical(logits=actor)

        critic = self.critic_linear1(x)
        critic = self.activation_fn(critic)
        critic = self.critic_linear2(critic)
        critic = self.activation_fn(critic)
        critic = self.critic_out(critic)
        return pi, jnp.squeeze(critic, axis=-1)

    def model_forward_eval(self, memories, input, mask):
        # input should be (batch_size, 1, ...) if it is just the obs for standard_ppo
        # for meta_learning it should be a dict with values (obs,rew,act,done) of shape [batch_size, 1, ...]
        encoded_input = self.input_encoder(input)
        # Transformer_XL.forward_eval expects the input argument to be of shape [batch_size, input_dim]
        # input_dim does't have to match transformer hidden dim (the transformer applies a linear mapping to that size internally)
        x, memory_out = self.transformer.forward_eval(memories, encoded_input.squeeze(axis=1), mask)
        # x has shape [batch_size, hidden_dim]

        actor = self.actor_linear1(x)
        actor = self.activation_fn(actor)
        actor = self.actor_linear2(actor)
        actor = self.activation_fn(actor)
        actor = self.actor_out(actor)
        pi = distrax.Categorical(logits=actor)

        critic = self.critic_linear1(x)
        critic = self.activation_fn(critic)
        critic = self.critic_linear2(critic)
        critic = self.activation_fn(critic)
        critic = self.critic_out(critic)

        return pi, jnp.squeeze(critic, axis=-1), memory_out

    def model_forward_train(self, memories, input, mask):
        # encoder expects shapes: [batch_size, seq_len, ...], or dict with values of that shape for meta-learning
        encoded_input = self.input_encoder(input)
        # Transformer_XL.forward_train() expects the input argument to be of shape [batch_size, seq_len, input_dim]
        x = self.transformer.forward_train(memories, encoded_input, mask)

        actor = self.actor_linear1(x)
        actor = self.activation_fn(actor)
        actor = self.actor_linear2(actor)
        actor = self.activation_fn(actor)
        actor = self.actor_out(actor)
        pi = distrax.Categorical(logits=actor)

        critic = self.critic_linear1(x)
        critic = self.activation_fn(critic)
        critic = self.critic_linear2(critic)
        critic = self.activation_fn(critic)
        critic = self.critic_out(critic)
        return pi, jnp.squeeze(critic, axis=-1)
