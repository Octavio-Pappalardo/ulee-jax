import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from ulee_repo.networks.transformer_actor_critic import Encoder_observations
from ulee_repo.networks.transformer_xl_base import Transformer_XL

## Only difference over standard network for standard ppo is that two value heads are used.

# ----------------- Actor Critic Transformer  ---------------------------


class ActorCriticTransformer(nn.Module):
    num_actions: int
    # transformer
    hidden_dim: int
    num_attn_heads: int
    qkv_features: int
    num_layers_in_transformer: int
    gating: bool
    gating_bias: float
    # mlp actor and critic heads
    head_activation: str
    mlp_dim: int
    # encoder
    obs_emb_dim: int

    def setup(self):
        self.input_encoder = Encoder_observations(obs_emb_dim=self.obs_emb_dim)

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

        self.critic_extrinsic_linear1 = nn.Dense(self.mlp_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.critic_extrinsic_linear2 = nn.Dense(self.mlp_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.critic_extrinsic_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        self.critic_intrinsic_linear1 = nn.Dense(self.mlp_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.critic_intrinsic_linear2 = nn.Dense(self.mlp_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.critic_intrinsic_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, memories, input, mask):
        encoded_input = self.input_encoder(input)
        x = self.transformer(memories, encoded_input.squeeze(axis=1), mask)

        actor = self.actor_linear1(x)
        actor = self.activation_fn(actor)
        actor = self.actor_linear2(actor)
        actor = self.activation_fn(actor)
        actor = self.actor_out(actor)
        pi = distrax.Categorical(logits=actor)

        critic_extrinsic = self.critic_extrinsic_linear1(x)
        critic_extrinsic = self.activation_fn(critic_extrinsic)
        critic_extrinsic = self.critic_extrinsic_linear2(critic_extrinsic)
        critic_extrinsic = self.activation_fn(critic_extrinsic)
        critic_extrinsic = self.critic_extrinsic_out(critic_extrinsic)

        critic_intrinsic = self.critic_intrinsic_linear1(x)
        critic_intrinsic = self.activation_fn(critic_intrinsic)
        critic_intrinsic = self.critic_intrinsic_linear2(critic_intrinsic)
        critic_intrinsic = self.activation_fn(critic_intrinsic)
        critic_intrinsic = self.critic_intrinsic_out(critic_intrinsic)

        return pi, jnp.squeeze(critic_extrinsic, axis=-1), jnp.squeeze(critic_intrinsic, axis=-1)

    def model_forward_eval(self, memories, input, mask):
        encoded_input = self.input_encoder(input)
        x, memory_out = self.transformer.forward_eval(memories, encoded_input.squeeze(axis=1), mask)

        actor = self.actor_linear1(x)
        actor = self.activation_fn(actor)
        actor = self.actor_linear2(actor)
        actor = self.activation_fn(actor)
        actor = self.actor_out(actor)
        pi = distrax.Categorical(logits=actor)

        critic_extrinsic = self.critic_extrinsic_linear1(x)
        critic_extrinsic = self.activation_fn(critic_extrinsic)
        critic_extrinsic = self.critic_extrinsic_linear2(critic_extrinsic)
        critic_extrinsic = self.activation_fn(critic_extrinsic)
        critic_extrinsic = self.critic_extrinsic_out(critic_extrinsic)

        critic_intrinsic = self.critic_intrinsic_linear1(x)
        critic_intrinsic = self.activation_fn(critic_intrinsic)
        critic_intrinsic = self.critic_intrinsic_linear2(critic_intrinsic)
        critic_intrinsic = self.activation_fn(critic_intrinsic)
        critic_intrinsic = self.critic_intrinsic_out(critic_intrinsic)

        return pi, jnp.squeeze(critic_extrinsic, axis=-1), jnp.squeeze(critic_intrinsic, axis=-1), memory_out

    def model_forward_train(self, memories, input, mask):
        encoded_input = self.input_encoder(input)
        x = self.transformer.forward_train(memories, encoded_input, mask)

        actor = self.actor_linear1(x)
        actor = self.activation_fn(actor)
        actor = self.actor_linear2(actor)
        actor = self.activation_fn(actor)
        actor = self.actor_out(actor)
        pi = distrax.Categorical(logits=actor)

        critic_extrinsic = self.critic_extrinsic_linear1(x)
        critic_extrinsic = self.activation_fn(critic_extrinsic)
        critic_extrinsic = self.critic_extrinsic_linear2(critic_extrinsic)
        critic_extrinsic = self.activation_fn(critic_extrinsic)
        critic_extrinsic = self.critic_extrinsic_out(critic_extrinsic)

        critic_intrinsic = self.critic_intrinsic_linear1(x)
        critic_intrinsic = self.activation_fn(critic_intrinsic)
        critic_intrinsic = self.critic_intrinsic_linear2(critic_intrinsic)
        critic_intrinsic = self.activation_fn(critic_intrinsic)
        critic_intrinsic = self.critic_intrinsic_out(critic_intrinsic)

        return pi, jnp.squeeze(critic_extrinsic, axis=-1), jnp.squeeze(critic_intrinsic, axis=-1)
