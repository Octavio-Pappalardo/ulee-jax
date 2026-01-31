import flax.linen as nn
import jax
import jax.numpy as jnp
from xminigrid.core.constants import NUM_COLORS, NUM_TILES

from ulee_repo.networks.tile_embedding import Embedding_xland_map
from ulee_repo.shared_code.trainsition_objects import State_Data

# ---------------------------------------------------------
# f(s) as full grid state + agent position


class Skill_Discriminator_Encoder_Full(nn.Module):
    grid_state_emb_dim: int

    @nn.compact
    def __call__(self, state: State_Data) -> jax.Array:
        embedding_layer = Embedding_xland_map(emb_dim=self.grid_state_emb_dim)

        grid_state_encoder = nn.Sequential(
            [
                nn.Conv(features=64, kernel_size=(1, 1)),
                nn.relu,
                nn.Conv(32, (2, 2), padding="VALID"),
                nn.relu,
                nn.Conv(32, (2, 2), padding="VALID"),
                nn.relu,
                nn.Conv(64, (2, 2), padding="VALID"),
                nn.relu,
                nn.Conv(features=16, kernel_size=(1, 1)),
                nn.relu,
            ]
        )
        BS, H, W = state.grid_state.shape[:3]

        # create masks of zeros with a one on the position of the agent
        pos_mask = jnp.zeros((BS, H, W, 1))

        def set_position(single_mask, pos):
            return single_mask.at[pos[0], pos[1], 0].set(1)

        set_positions = jax.vmap(set_position, in_axes=(0, 0))
        pos_mask = set_positions(pos_mask, state.agent_pos)
        #
        state_emb = embedding_layer(state.grid_state)
        grid_state_input = jnp.concatenate([pos_mask, state_emb], axis=-1)

        grid_state_emb = grid_state_encoder(grid_state_input).reshape(BS, -1)

        input_encoding = jnp.concatenate([grid_state_emb, state.agent_pos], axis=-1)  # [batch_size * seq_len, __ ]

        return input_encoding


# ---------------------------------------------------------
# f(s) as grid object counts

VOCAB_SIZE = NUM_TILES * NUM_COLORS


def _object_id(type_id: jnp.ndarray, color_id: jnp.ndarray) -> jnp.ndarray:
    return type_id * NUM_COLORS + color_id


class Skill_Discriminator_Encoder_Histogram(nn.Module):
    @nn.compact
    def __call__(self, state: State_Data) -> jax.Array:
        # state.grid_state shape: [batch_size * seq_len, ...]

        ids = _object_id(state.grid_state[..., 0], state.grid_state[..., 1]).reshape(state.grid_state.shape[0], -1)  # (B*S, H*W)

        counts = jax.vmap(lambda x: jnp.bincount(x, length=VOCAB_SIZE))(ids)  # (B*S, VOCAB_SIZE)

        return counts


# ---------------------------------------------------------


class Skill_Discriminator(nn.Module):
    discriminator_mode: str
    num_skills: int
    # encoder
    grid_state_emb_dim: int
    # mlp head
    head_activation: str
    mlp_dim: int

    def setup(self):
        if self.discriminator_mode == "full_state":
            self.input_encoder = Skill_Discriminator_Encoder_Full(
                grid_state_emb_dim=self.grid_state_emb_dim,
            )
        elif self.discriminator_mode == "objects_histogram":
            self.input_encoder = Skill_Discriminator_Encoder_Histogram()

        if self.head_activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

        self.linear1 = nn.Dense(self.mlp_dim)
        self.linear2 = nn.Dense(self.mlp_dim)

        self.linear_out = nn.Dense(self.num_skills, kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "uniform"), bias_init=nn.initializers.zeros)

    def __call__(self, state_input) -> jax.Array:
        # input_encoder() expects the inputs to be dicts with values of shape [batch_size * seq_len, ...]
        encoded_input = self.input_encoder(state=state_input)  # (B*S, encoded_dim)

        outputs = self.linear1(encoded_input)
        outputs = self.activation_fn(outputs)
        outputs = self.linear2(outputs)
        outputs = self.activation_fn(outputs)
        output_logits = self.linear_out(outputs)  # shape [batch_size*seq_len, num_skills]

        return output_logits
