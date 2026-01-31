import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.tree_util import Partial

from ulee_repo.networks.transformer_actor_critic import ActorCriticTransformer
from ulee_repo.shared_code.ppo_update import calculate_gae, update_epoch
from ulee_repo.shared_code.trainsition_objects import Transition_data_meta_learning


def step_envs(runner_state, _unused, env, env_params, config):
    rng, train_state, prev_timestep, prev_action, prev_reward, prev_done, memories, memories_mask, memories_mask_idx, current_update_step_num = runner_state

    # Update transformer mask
    memories_mask_idx = jnp.clip(memories_mask_idx - 1, 0, config.past_context_length)
    memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, config.past_context_length + 1)
    memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(config.num_attn_heads, 1)
    memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

    # Select actions
    rng, _rng = jax.random.split(rng)
    input_in_step = {
        "observation": prev_timestep.observation[:, None],
        "prev_action": prev_action[:, None],
        "prev_reward": prev_reward[:, None],
        "prev_done": prev_done[:, None],
    }
    pi, value, memories_out = train_state.apply_fn(train_state.params, memories, input_in_step, memories_mask, method=ActorCriticTransformer.model_forward_eval)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)

    # Update memory buffer
    memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

    # Step environments
    timestep = jax.vmap(env.step, in_axes=0)(env_params, prev_timestep, action)

    memory_indices = jnp.arange(0, config.past_context_length)[None, :] + current_update_step_num * jnp.ones((config.num_envs_per_batch, 1), dtype=jnp.int32)

    # Store transition data
    transition = Transition_data_meta_learning(
        done=jnp.zeros_like(timestep.last()),
        action=action,
        value=value,
        reward=timestep.reward,
        log_prob=log_prob,
        obs=prev_timestep.observation,
        prev_action=prev_action,
        prev_reward=prev_reward,
        prev_done=prev_done,
        memories_mask=memories_mask.squeeze(),
        memories_indices=memory_indices,
    )

    # Create updated runner state
    runner_state = (rng, train_state, timestep, action, timestep.reward, timestep.last(), memories, memories_mask, memories_mask_idx, current_update_step_num + 1)

    return runner_state, (transition, memories_out)


def collect_data(runner_state, num_steps, env, env_params, config):
    runner_state, (transitions, memories_batch) = jax.lax.scan(Partial(step_envs, env=env, env_params=env_params, config=config), runner_state, None, num_steps)

    return runner_state, transitions, memories_batch


def update_agent(runner_state, transitions, memories_batch, config):
    rng, train_state, timestep, prev_action, prev_reward, prev_done, memories, memories_mask, memories_mask_idx, _ = runner_state

    # Compute advantages and targets (GAE)
    last_input = {"observation": timestep.observation[:, None], "prev_action": prev_action[:, None], "prev_reward": prev_reward[:, None], "prev_done": prev_done[:, None]}
    _, last_val, _ = train_state.apply_fn(train_state.params, memories, last_input, memories_mask, method=ActorCriticTransformer.model_forward_eval)
    advantages, targets = calculate_gae(transitions, last_val, config.gamma, config.gae_lambda)

    # Compute loss and update network
    update_state = (rng, train_state, transitions, memories_batch, advantages, targets)

    update_state, metrics = jax.lax.scan(Partial(update_epoch, algorithm_id="meta_learning", config=config), update_state, None, config.update_epochs)
    rng, train_state = update_state[:2]
    runner_state = (rng, train_state, timestep, prev_action, prev_reward, prev_done, memories, memories_mask, memories_mask_idx, 0)

    metrics = jtu.tree_map(lambda x: x.mean(-1).mean(-1), metrics)

    return runner_state, metrics


def collect_data_and_update(runner_state, _unused, env, env_params, config):
    memories_previous = runner_state[6]  # (batch_size, past_context_length, num_tranformer_layers, hidden_dim)

    runner_state, transitions, memories_batch = collect_data(runner_state, config.num_steps_per_update, env, env_params, config)

    memories_batch = jnp.concatenate([jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0)  # (past_context + num_steps_per_update, num_envs, num_tranformer_layers, hidden_dim)

    runner_state, metrics = update_agent(runner_state, transitions, memories_batch, config)

    return runner_state, metrics
