import flax.linen as nn
import jax
import jax.numpy as jnp

from ulee_repo.networks.tile_embedding import Embedding_xland_map
from ulee_repo.shared_code.trainsition_objects import State_Data


class Judge_Encoder(nn.Module):
    grid_state_emb_dim: int

    @nn.compact
    def __call__(self, initial_state: State_Data, final_state: State_Data) -> jax.Array:
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

        B, H, W = initial_state.grid_state.shape[:3]

        # create masks of zeros with a one on the position of the agent
        init_pos_mask = jnp.zeros((B, H, W, 1))
        final_pos_mask = jnp.zeros((B, H, W, 1))

        def set_position(single_mask, pos):
            return single_mask.at[pos[0], pos[1], 0].set(1)

        set_positions = jax.vmap(set_position, in_axes=(0, 0))
        init_pos_mask = set_positions(init_pos_mask, initial_state.agent_pos)
        final_pos_mask = set_positions(final_pos_mask, final_state.agent_pos)
        #

        diff = initial_state.grid_state - final_state.grid_state
        init_state_emb = embedding_layer(initial_state.grid_state)
        final_state_emb = embedding_layer(final_state.grid_state)

        grid_state_input = jnp.concatenate([init_pos_mask, final_pos_mask, diff, init_state_emb, final_state_emb], axis=-1)

        grid_state_emb = grid_state_encoder(grid_state_input).reshape(B, -1)

        input_encoding = jnp.concatenate([grid_state_emb, initial_state.agent_pos, final_state.agent_pos], axis=-1)

        return input_encoding


class Difficulty_Judge(nn.Module):
    # encoder
    grid_state_emb_dim: int
    # mlp head
    head_activation: str
    mlp_dim: int

    def setup(self):
        self.input_encoder = Judge_Encoder(grid_state_emb_dim=self.grid_state_emb_dim)

        if self.head_activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

        self.linear1 = nn.Dense(self.mlp_dim)
        self.linear2 = nn.Dense(self.mlp_dim)
        self.linear_out = nn.Dense(1, kernel_init=nn.initializers.variance_scaling(1e-2, "fan_in", "truncated_normal"), bias_init=nn.initializers.zeros)

    def __call__(self, init_state_input: State_Data, target_state_input: State_Data):
        encoded_input = self.input_encoder(initial_state=init_state_input, final_state=target_state_input)  # (B, encoded_dim)

        outputs = self.linear1(encoded_input)
        outputs = self.activation_fn(outputs)
        outputs = self.linear2(outputs)
        outputs = self.activation_fn(outputs)
        outputs = self.linear_out(outputs)
        outputs = nn.sigmoid(outputs)  # [B, 1]

        return outputs.squeeze(axis=-1)  # [B]
