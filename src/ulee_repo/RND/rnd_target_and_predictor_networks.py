import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from xminigrid.core.constants import NUM_COLORS, NUM_TILES

from ulee_repo.networks.tile_embedding import Embedding_xland_map

## The code follows a similar structure to diayn_discriminator.py
## The main difference is that inputs are observations instead of states


# ---------------------------------------------------------
# Code for f(s) as full grid state + agent position


class RND_Encoder_Full(nn.Module):
    obs_emb_dim: int

    @nn.compact
    def __call__(self, observations) -> jax.Array:
        obs_encoder = nn.Sequential(
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
        BS = observations.shape[0]
        obs_encodings = obs_encoder(observations).reshape(BS, -1)
        return obs_encodings


# ---------------------------------------------------------
# Code for f(s) as set of objects in the grid

VOCAB_SIZE = NUM_TILES * NUM_COLORS


# Helper that maps (type, colour) to a unique id in [0, VOCAB_SIZE)
def _object_id(type_id: jnp.ndarray, color_id: jnp.ndarray) -> jnp.ndarray:
    return type_id * NUM_COLORS + color_id


class RND_Encoder_Histogram(nn.Module):
    @nn.compact
    def __call__(self, observations) -> jax.Array:
        # observations.shape: [batch_size * seq_len, ...]

        ids = _object_id(observations[..., 0], observations[..., 1]).reshape(observations.shape[0], -1)  # (B*S, H*W)

        counts = jax.vmap(lambda x: jnp.bincount(x, length=VOCAB_SIZE))(ids)  # (B*S, VOCAB_SIZE)

        return counts


# ---------------------------------------------------------


class Target_and_Predictor(nn.Module):
    encoder_mode: str
    output_embedding_dim: int
    # encoder
    obs_emb_dim: int
    # mlp head
    head_activation: str
    mlp_dim: int

    def setup(self):
        if self.encoder_mode == "full_state":
            self.input_encoder = RND_Encoder_Full(
                obs_emb_dim=self.obs_emb_dim,
            )
        elif self.encoder_mode == "objects_histogram":
            self.input_encoder = RND_Encoder_Histogram()

        if self.head_activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

        self.linear1 = nn.Dense(self.mlp_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.linear2 = nn.Dense(self.mlp_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.linear_out = nn.Dense(self.output_embedding_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

    def __call__(self, observations) -> jax.Array:
        # input_encoder() expects observations to be of shape [batch_size * seq_len, ...]
        encoded_input = self.input_encoder(observations=observations)  # (B*S, encoded_dim)

        outputs = self.linear1(encoded_input)
        outputs = self.activation_fn(outputs)
        outputs = self.linear2(outputs)
        outputs = self.activation_fn(outputs)
        output_logits = self.linear_out(outputs)  # shape [batch_size *seq_len, output_embedding_dim]

        return output_logits
