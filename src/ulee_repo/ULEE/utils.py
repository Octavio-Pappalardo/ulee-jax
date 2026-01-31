import jax
import jax.numpy as jnp
from xminigrid.core.constants import NUM_COLORS, NUM_TILES

from ulee_repo.shared_code.trainsition_objects import State_Data

# ---------------------------------------------------------
# goals as full grid state + agent position


def encode_single_goal_as_full_state(grid_state, position):
    return jnp.concatenate([grid_state.ravel(), position])


def encode_goals_as_full_states(goals: State_Data):
    # encode a batch of goals into 1d cector representations. A goal is a pytree with elements of shape (B, ...)
    flattened_grid_state = jnp.reshape(goals.grid_state, (goals.grid_state.shape[0], -1))
    return jnp.concatenate([flattened_grid_state, goals.agent_pos], axis=-1)


# ---------------------------------------------------------
# goals as grid object counts


VOCAB_SIZE = NUM_TILES * NUM_COLORS


def _object_id(type_id: jnp.ndarray, color_id: jnp.ndarray) -> jnp.ndarray:
    return type_id * NUM_COLORS + color_id


def encode_single_goal_as_object_histogram(grid_state: jnp.ndarray, position: jnp.ndarray) -> jnp.ndarray:
    # grid_state : (H, W, 2)
    ids = _object_id(grid_state[..., 0], grid_state[..., 1]).ravel()  # (H*W,)
    counts = jnp.bincount(ids, length=VOCAB_SIZE)  # (VOCAB_SIZE,) of counts of objects
    return counts


def encode_goals_as_object_histograms(
    goals,
) -> jnp.ndarray:
    # goals.grid_state : (B, H, W, 2)
    ids = _object_id(goals.grid_state[..., 0], goals.grid_state[..., 1]).reshape(goals.grid_state.shape[0], -1)  # (B, H*W)
    counts = jax.vmap(lambda x: jnp.bincount(x, length=VOCAB_SIZE))(ids)  # (B, VOCAB_SIZE)

    return counts
