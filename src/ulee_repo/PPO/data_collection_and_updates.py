import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.tree_util import Partial

from ulee_repo.networks.transformer_actor_critic import ActorCriticTransformer
from ulee_repo.shared_code.ppo_update import calculate_gae, update_epoch
from ulee_repo.shared_code.trainsition_objects import State_Data, Transition_data_standard


def step_envs(runner_state, unused, env, env_params, config):
    rng, train_state, prev_timestep, memories, memories_mask, memories_mask_idx, current_update_step_num = runner_state

    # Update transformer mask or reset it if new episode
    memories_mask_idx = jnp.where(prev_timestep.last(), config.past_context_length, jnp.clip(memories_mask_idx - 1, 0, config.past_context_length))
    memories_mask = jnp.where(
        prev_timestep.last()[:, None, None, None], jnp.zeros((config.num_envs_per_batch, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_), memories_mask
    )
    memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, config.past_context_length + 1)
    memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(config.num_attn_heads, 1)
    memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

    # Select actions
    rng, _rng = jax.random.split(rng)
    pi, value, memories_out = train_state.apply_fn(train_state.params, memories, prev_timestep.observation[:, None], memories_mask, method=ActorCriticTransformer.model_forward_eval)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)

    # Update memory buffer
    memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

    # Step environments
    timestep = jax.vmap(env.step, in_axes=0)(env_params, prev_timestep, action)

    memory_indices = jnp.arange(0, config.past_context_length)[None, :] + current_update_step_num * jnp.ones((config.num_envs_per_batch, 1), dtype=jnp.int32)

    # Store transition data
    transition = Transition_data_standard(
        done=timestep.last(),
        action=action,
        value=value,
        reward=timestep.reward,
        log_prob=log_prob,
        obs=prev_timestep.observation,
        memories_mask=memories_mask.squeeze(),
        memories_indices=memory_indices,
        state_data=State_Data(
            grid_state=prev_timestep.state.grid,
            agent_pos=prev_timestep.state.agent.position,
        ),
    )

    # Create updated runner state
    runner_state = (rng, train_state, timestep, memories, memories_mask, memories_mask_idx, current_update_step_num + 1)

    return runner_state, (transition, memories_out)


#############-------------------------------


def collect_data(runner_state, num_steps, env, env_params, config):
    runner_state, (transitions, memories_batch) = jax.lax.scan(Partial(step_envs, env=env, env_params=env_params, config=config), runner_state, None, num_steps)

    return runner_state, transitions, memories_batch


#############-------------------------------


def update_agent(runner_state, transitions, memories_batch, config):
    rng, train_state, timestep, memories, memories_mask, memories_mask_idx, _ = runner_state

    # Compute advantages and targets (GAE)
    _, last_val, _ = train_state.apply_fn(train_state.params, memories, timestep.observation[:, None], memories_mask, method=ActorCriticTransformer.model_forward_eval)
    advantages, targets = calculate_gae(transitions, last_val, config.gamma, config.gae_lambda)

    # Compute loss and update network
    update_state = (rng, train_state, transitions, memories_batch, advantages, targets)
    update_state, metrics = jax.lax.scan(Partial(update_epoch, algorithm_id="standard_ppo", config=config), update_state, None, config.update_epochs)
    rng, train_state = update_state[:2]

    runner_state = (rng, train_state, timestep, memories, memories_mask, memories_mask_idx, 0)

    metrics = jtu.tree_map(lambda x: x.mean(-1).mean(-1), metrics)

    return runner_state, metrics


#############-------------------------------


def collect_data_and_update(runner_state, _unused, env, env_params, config):
    memories_previous = runner_state[3]  # (batch_size, past_context_length, num_tranformer_layers, hidden_dim)

    runner_state, transitions, memories_batch = collect_data(runner_state, config.num_steps_per_update, env, env_params, config)

    # Concatenate previous memory with new batch
    memories_batch = jnp.concatenate([jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0)  # (past_context_length + num_steps_per_update, num_envs, num_tranformer_layers, hidden_dim)

    runner_state, metrics = update_agent(runner_state, transitions, memories_batch, config)

    return runner_state, metrics
