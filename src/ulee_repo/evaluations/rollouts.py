from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState
from xminigrid.environment import Environment, EnvParams

from ulee_repo.networks.diayn_transformer_actor_critic import (
    DiaynActorCriticTransformer,
)
from ulee_repo.networks.transformer_actor_critic import ActorCriticTransformer
from ulee_repo.RND.rnd_transformer_actor_critic import (
    ActorCriticTransformer as RndActorCriticTransformer,
)


@dataclass
class RolloutEpisodeStats(struct.PyTreeNode):
    returns: jax.Array
    lengths: jax.Array


def create_benchmark_step_func(env):
    def step_fn(env_params, prev_timestep, action, goal):
        return env.step(env_params, prev_timestep, action)

    return step_fn


def create_unsupervised_step_func(env):
    def step_fn(env_params, prev_timestep, action, goal):
        return env.step(env_params, prev_timestep, action, goal)

    return step_fn


def eval_rollout(
    rng: jax.Array,
    reset_rng: jax.Array,
    env: Environment,
    env_params: EnvParams,
    step_function: Callable,
    train_state: TrainState,
    num_consecutive_episodes: int,
    algorithm_id: str,
    config,
    goal: jax.Array | None,
    skill: jax.Array | None = None,
) -> RolloutEpisodeStats:
    """Run policy in an environment for `num_consecutive_episodes` episodes,
    returning per-episode returns and lengths.
    """
    if algorithm_id == "diayn" and skill is None:
        msg = "Desired skill must be provided for eval rollout with DIAYN."
        raise ValueError(msg)

    episode_stats = RolloutEpisodeStats(returns=jnp.zeros((num_consecutive_episodes,)), lengths=jnp.zeros((num_consecutive_episodes,)))
    position_history = jnp.zeros((num_consecutive_episodes, env_params.max_steps, 2), dtype=jnp.float32)

    # Current episode counters
    ep_len = jnp.array(0, dtype=jnp.int32)
    ep_ret = jnp.array(0.0, dtype=jnp.float32)
    # episode number
    ep_num = jnp.array(0, dtype=jnp.int32)
    # Reset env and set initial inputs
    timestep = env.reset(env_params, reset_rng)
    if algorithm_id in ["standard_ppo", "rnd"]:
        additional_network_input = None
    elif algorithm_id == "diayn":
        additional_network_input = skill
    elif algorithm_id == "meta_learning":
        prev_action = jnp.array(0, dtype=jnp.int32)
        prev_reward = jnp.array(0.0, dtype=jnp.float32)
        prev_done = timestep.last()
        additional_network_input = (prev_action, prev_reward, prev_done)

    # initialize mask related variables
    memories = jnp.zeros((1, config.past_context_length, config.num_transformer_blocks, config.transformer_hidden_states_dim))
    memories_mask = jnp.zeros((1, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_)
    memories_mask_idx = jnp.zeros((1,), dtype=jnp.int32) + (config.past_context_length + 1)

    # initialize carry for the loop:
    init_carry = (rng, episode_stats, ep_num, ep_len, ep_ret, timestep, additional_network_input, memories, memories_mask, memories_mask_idx, position_history)

    # Loop condition
    def _cond_fn(carry):
        _, _, ep_num, _, _, _, _, _, _, _, _ = carry
        return ep_num < num_consecutive_episodes

    # loop body
    def _body_fn(carry):
        (rng, episode_stats, ep_num, ep_len, ep_ret, prev_timestep, additional_network_input, memories, memories_mask, memories_mask_idx, position_history) = carry

        rng, _rng = jax.random.split(rng)

        # update transformer mask
        if algorithm_id in ["standard_ppo", "rnd", "diayn"]:
            memories_mask_idx = jnp.where(prev_timestep.last(), config.past_context_length, jnp.clip(memories_mask_idx - 1, 0, config.past_context_length))
            memories_mask = jnp.where(prev_timestep.last()[None, None, None, None], jnp.zeros((1, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_), memories_mask)
            memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, config.past_context_length + 1)
        elif algorithm_id == "meta_learning":
            memories_mask_idx = jnp.clip(memories_mask_idx - 1, 0, config.past_context_length)
            memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, config.past_context_length + 1)

        memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(config.num_attn_heads, 1)
        memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

        # Sample action
        if algorithm_id == "standard_ppo":
            pi, _, memories_out = train_state.apply_fn(train_state.params, memories, prev_timestep.observation[None, None, ...], memories_mask, method=ActorCriticTransformer.model_forward_eval)
        elif algorithm_id == "rnd":
            pi, _, _, memories_out = train_state.apply_fn(train_state.params, memories, prev_timestep.observation[None, None, ...], memories_mask, method=RndActorCriticTransformer.model_forward_eval)
        elif algorithm_id == "diayn":
            skill = additional_network_input
            pi, _, memories_out = train_state.apply_fn(
                train_state.params, memories, prev_timestep.observation[None, None, ...], skill[None, None], memories_mask, method=DiaynActorCriticTransformer.model_forward_eval
            )
        elif algorithm_id == "meta_learning":
            prev_action, prev_reward, prev_done = additional_network_input
            input_in_step = {
                "observation": prev_timestep.observation[None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
                "prev_done": prev_done[None, None, ...],
            }
            pi, _, memories_out = train_state.apply_fn(train_state.params, memories, input_in_step, memories_mask, method=ActorCriticTransformer.model_forward_eval)

        action = pi.sample(seed=_rng).squeeze()

        # Update memory buffer
        memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

        position_history = position_history.at[ep_num, ep_len].set(prev_timestep.state.agent.position)

        # Step environment
        timestep = step_function(env_params, prev_timestep, action, goal)

        # Update counters
        ep_len = ep_len + 1
        ep_ret = ep_ret + timestep.reward

        #  If the episode ended, add stats to RolloutEpisodeStats
        def write_episode(condition_carry):
            episode_stats, ep_num, ep_len, ep_ret = condition_carry
            episode_stats = RolloutEpisodeStats(returns=episode_stats.returns.at[ep_num].set(ep_ret), lengths=episode_stats.lengths.at[ep_num].set(ep_len))
            return episode_stats, ep_num + 1, jnp.array(0, jnp.int32), jnp.array(0.0, jnp.float32)

        def do_nothing(condition_carry):
            return condition_carry

        (episode_stats, ep_num, ep_len, ep_ret) = jax.lax.cond(timestep.last(), write_episode, do_nothing, (episode_stats, ep_num, ep_len, ep_ret))

        # Construct the new carry
        if algorithm_id == "meta_learning":
            additional_network_input = (action, timestep.reward, timestep.last())
        new_carry = (rng, episode_stats, ep_num, ep_len, ep_ret, timestep, additional_network_input, memories, memories_mask, memories_mask_idx, position_history)

        return new_carry

    # Run loop
    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_carry)

    return final_carry[0], final_carry[1], final_carry[-1]


def eval_rollout_random(
    rng: jax.Array,
    reset_rng: jax.Array,
    env: Environment,
    env_params: EnvParams,
    step_function: Callable,
    num_consecutive_episodes: int,
    goal: jax.Array | None,
) -> RolloutEpisodeStats:
    """Run random policy in an environment for `num_consecutive_episodes` episodes,
    returning per-episode returns and lengths."""

    episode_stats = RolloutEpisodeStats(returns=jnp.zeros((num_consecutive_episodes,)), lengths=jnp.zeros((num_consecutive_episodes,)))
    position_history = jnp.zeros((num_consecutive_episodes, env_params.max_steps, 2), dtype=jnp.float32)

    ep_len = jnp.array(0, dtype=jnp.int32)
    ep_ret = jnp.array(0.0, dtype=jnp.float32)
    ep_num = jnp.array(0, dtype=jnp.int32)
    timestep = env.reset(env_params, reset_rng)

    init_carry = (rng, episode_stats, ep_num, ep_len, ep_ret, timestep, position_history)

    def _cond_fn(carry):
        _, _, ep_num, _, _, _, _ = carry
        return ep_num < num_consecutive_episodes

    def _body_fn(carry):
        (rng, episode_stats, ep_num, ep_len, ep_ret, prev_timestep, position_history) = carry

        rng, _rng = jax.random.split(rng)
        action = jax.random.randint(_rng, shape=(), minval=0, maxval=env.num_actions(env_params))

        position_history = position_history.at[ep_num, ep_len].set(prev_timestep.state.agent.position)

        timestep = step_function(env_params, prev_timestep, action, goal)

        ep_len = ep_len + 1
        ep_ret = ep_ret + timestep.reward

        def write_episode(condition_carry):
            episode_stats, ep_num, ep_len, ep_ret = condition_carry
            episode_stats = RolloutEpisodeStats(returns=episode_stats.returns.at[ep_num].set(ep_ret), lengths=episode_stats.lengths.at[ep_num].set(ep_len))
            return episode_stats, ep_num + 1, jnp.array(0, jnp.int32), jnp.array(0.0, jnp.float32)

        def do_nothing(condition_carry):
            return condition_carry

        (episode_stats, ep_num, ep_len, ep_ret) = jax.lax.cond(timestep.last(), write_episode, do_nothing, (episode_stats, ep_num, ep_len, ep_ret))
        new_carry = (rng, episode_stats, ep_num, ep_len, ep_ret, timestep, position_history)
        return new_carry

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_carry)

    return final_carry[0], final_carry[1], final_carry[-1]
