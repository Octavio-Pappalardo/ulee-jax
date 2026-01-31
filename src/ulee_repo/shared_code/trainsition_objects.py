import jax
import jax.numpy as jnp
from flax import struct


class State_Data(struct.PyTreeNode):
    grid_state: jax.Array
    agent_pos: jax.Array
    # agent_dir: jax.Array
    # agent_pocket: jax.Array


class Transition_data_base(struct.PyTreeNode):
    # for ppo update
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array


class Transition_data_meta_learning(Transition_data_base):
    # for recurrent inputs
    prev_action: jax.Array
    prev_reward: jax.Array
    prev_done: jax.Array
    # for transformer
    memories_mask: jnp.ndarray
    memories_indices: jnp.ndarray


class Transition_data_standard(Transition_data_base):
    # for transformer
    memories_mask: jnp.ndarray
    memories_indices: jnp.ndarray
    #
    state_data: State_Data


class Transition_data_rnd(Transition_data_base):
    # transformer specific
    memories_mask: jnp.ndarray
    memories_indices: jnp.ndarray
    # rnd specific
    next_obs: jnp.ndarray
    intrinsic_reward: jnp.ndarray
    intrinsic_value: jnp.ndarray
    done_for_intrinsic: jnp.ndarray


class Transition_data_diayn(Transition_data_base):
    # for transformer
    memories_mask: jnp.ndarray
    memories_indices: jnp.ndarray
    # DIAYN specific
    skill: jax.Array
    state_data: State_Data
