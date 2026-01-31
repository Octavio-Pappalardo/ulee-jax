import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax.training.train_state import TrainState
from jax.tree_util import Partial

from ulee_repo.networks.diayn_transformer_actor_critic import DiaynActorCriticTransformer
from ulee_repo.shared_code.ppo_update import calculate_gae, update_epoch
from ulee_repo.shared_code.trainsition_objects import State_Data, Transition_data_diayn


def step_envs(runner_state, unused, env, env_params, config):
    rng, train_state, discriminator_train_state, prev_timestep, skill, memories, memories_mask, memories_mask_idx, current_update_step_num = runner_state

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
    pi, value, memories_out = train_state.apply_fn(
        train_state.params, memories, prev_timestep.observation[:, None], skill[:, None], memories_mask, method=DiaynActorCriticTransformer.model_forward_eval
    )
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)

    # Update memory buffer
    memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

    # Step environments
    timestep = jax.vmap(env.step, in_axes=0)(env_params, prev_timestep, action)

    # Update skill if episode terminated
    rng, _rng = jax.random.split(rng)
    new_skill = jax.random.randint(_rng, shape=(config.num_envs_per_batch,), minval=0, maxval=config.num_skills)
    updated_skill = jnp.where(timestep.last(), new_skill, skill)

    memory_indices = jnp.arange(0, config.past_context_length)[None, :] + current_update_step_num * jnp.ones((config.num_envs_per_batch, 1), dtype=jnp.int32)

    # Store transition data
    transition = Transition_data_diayn(
        done=timestep.last(),
        action=action,
        value=value,
        reward=timestep.reward,
        log_prob=log_prob,
        obs=prev_timestep.observation,
        skill=skill,
        state_data=State_Data(
            grid_state=prev_timestep.state.grid,
            agent_pos=prev_timestep.state.agent.position,
        ),
        memories_mask=memories_mask.squeeze(),
        memories_indices=memory_indices,
    )
    # Create updated runner state
    runner_state = (rng, train_state, discriminator_train_state, timestep, updated_skill, memories, memories_mask, memories_mask_idx, current_update_step_num + 1)

    return runner_state, (transition, memories_out)


#############-------------------------------


def collect_data(runner_state, num_steps, env, env_params, config):
    runner_state, (transitions, memories_batch) = jax.lax.scan(Partial(step_envs, env=env, env_params=env_params, config=config), runner_state, None, num_steps)

    return runner_state, transitions, memories_batch


#############-------------------------------


def update_agent(runner_state, transitions, memories_batch, config):
    rng, train_state, discriminator_train_state, timestep, skill, memories, memories_mask, memories_mask_idx, _ = runner_state

    # Compute advantages and targets (GAE)
    _, last_val, _ = train_state.apply_fn(train_state.params, memories, timestep.observation[:, None], skill[:, None], memories_mask, method=DiaynActorCriticTransformer.model_forward_eval)
    advantages, targets = calculate_gae(transitions, last_val, config.gamma, config.gae_lambda)

    # Compute loss and update network
    update_state = (rng, train_state, transitions, memories_batch, advantages, targets)
    update_state, metrics = jax.lax.scan(Partial(update_epoch, algorithm_id="diayn", config=config), update_state, None, config.update_epochs)
    rng, train_state = update_state[:2]

    runner_state = (rng, train_state, discriminator_train_state, timestep, skill, memories, memories_mask, memories_mask_idx, 0)

    metrics = jtu.tree_map(lambda x: x.mean(-1).mean(-1), metrics)

    return runner_state, metrics


#############-------------------------------


def update_discriminator(runner_state, transitions, config):
    rng, train_state, discriminator_train_state, timestep, skill, memories, memories_mask, memories_mask_idx, current_update_step_num = runner_state
    rng, discriminator_train_state, discriminator_loss_value = skill_discriminator_train(
        rng=rng,
        discriminator_train_state=discriminator_train_state,
        transitions=transitions,
        num_epochs=config.num_skill_discriminator_training_epochs,
        num_minibatches=config.num_skill_discriminator_minibatches,
    )
    runner_state = (rng, train_state, discriminator_train_state, timestep, skill, memories, memories_mask, memories_mask_idx, current_update_step_num)
    return runner_state, discriminator_loss_value


#############-------------------------------


def collect_data_and_update(runner_state, _unused, env, env_params, config):
    memories_previous = runner_state[5]  # (batch_size, past_context_length, num_tranformer_layers, hidden_dim)

    runner_state, transitions, memories_batch = collect_data(runner_state, config.num_steps_per_update, env, env_params, config)

    # Compute DIAYN intrinsic rewards
    diayn_rewards, skills_log_prob = compute_diayn_intrinsic_rewards(
        discriminator_train_state=runner_state[2], transitions=transitions, num_chunks=config.num_chunks_in_diayn_rewards_computation, num_skills=config.num_skills
    )
    diayn_rewards = diayn_rewards / env_params.max_steps
    transitions = transitions.replace(reward=diayn_rewards)

    # Concatenate previous memory with new batch
    memories_batch = jnp.concatenate([jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0)

    # Update policy and value networks
    runner_state, metrics = update_agent(runner_state, transitions, memories_batch, config)

    # Update skill discriminator
    runner_state, discriminator_loss_value = update_discriminator(runner_state, transitions, config)

    metrics.update(
        {
            "discriminator_loss": discriminator_loss_value,
            "skills_log_prob": skills_log_prob,
        }
    )
    return runner_state, metrics


#############-------------------------------


def skill_discriminator_train(
    rng,
    discriminator_train_state: TrainState,
    transitions,
    num_epochs: int,
    num_minibatches: int,
):
    # (S, B, ...) -> (S*B, ...)
    transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), transitions)
    total_num_steps = transitions.skill.shape[0]  # S*B

    minibatch_size = total_num_steps // num_minibatches

    def loss_fn(params, transitions_chunk):
        logits = discriminator_train_state.apply_fn(params, transitions_chunk.state_data)
        loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits=logits, labels=transitions_chunk.skill).mean()
        return loss

    def epoch_body_fun(carry, _unused):
        rng, train_state = carry

        # Shuffle data
        rng, shuffle_rng = jax.random.split(rng)
        indices_permuted = jax.random.permutation(shuffle_rng, total_num_steps)
        transitions_shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, indices_permuted, axis=0), transitions)

        # Reshape to (num_minibatches, minibatch_size, ...)
        transitions_shuffled = jax.tree_util.tree_map(lambda x: x.reshape((num_minibatches, minibatch_size) + x.shape[1:]), transitions_shuffled)

        def minibatch_step(train_state, transitions_chunk):
            loss, grads = jax.value_and_grad(loss_fn)(train_state.params, transitions_chunk)
            new_train_state = train_state.apply_gradients(grads=grads)
            return new_train_state, loss

        # loop over minibatches
        train_state, batch_losses = jax.lax.scan(minibatch_step, train_state, transitions_shuffled)
        epoch_loss = jnp.mean(batch_losses)

        return (rng, train_state), epoch_loss

    # Main loop over epochs
    init_carry = (rng, discriminator_train_state)
    (rng, final_discriminator_train_state), epoch_losses = jax.lax.scan(epoch_body_fun, init_carry, None, num_epochs)

    loss_value = jnp.mean(epoch_losses)

    return rng, final_discriminator_train_state, loss_value


#############-------------------------------


def compute_diayn_intrinsic_rewards(
    discriminator_train_state,
    transitions,
    num_chunks: int,
    num_skills: int,
) -> jnp.ndarray:
    # (S, B, ...) -> (S*B, ...)
    S, B = transitions.skill.shape[:2]
    transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (S * B,) + x.shape[2:]), transitions)
    total_num_steps = transitions.skill.shape[0]  # S*B

    chunk_size = total_num_steps // num_chunks

    transitions_chunked = jax.tree_util.tree_map(lambda x: x.reshape((num_chunks, chunk_size) + x.shape[1:]), transitions)  # (num_chunks, chunk_size, ...)

    def body_fun(carry, transitions_chunk):
        logits_chunk = discriminator_train_state.apply_fn(discriminator_train_state.params, transitions_chunk.state_data)
        log_probs_chunk = jax.nn.log_softmax(logits_chunk, axis=-1)  # (chunk_size, num_skills)

        # Gather log-prob for the actual skill
        skill_chunk = transitions_chunk.skill[:, None]  # (chunk_size, 1)
        log_p_chunk = jnp.take_along_axis(log_probs_chunk, skill_chunk, axis=-1).squeeze(axis=-1)  # (chunk_size,)

        return carry, log_p_chunk

    # Scan over chunks
    _, log_probs_chunked = jax.lax.scan(
        body_fun,
        None,
        transitions_chunked,
    )

    # (num_chunks, chunk_size) -> (S*B,) -> (S, B)
    skills_log_probs = log_probs_chunked.reshape((total_num_steps,))
    diayn_rewards = skills_log_probs.reshape((S, B))

    # Add the reward term -log p(z).
    diayn_rewards = diayn_rewards + jnp.log(num_skills)

    return diayn_rewards, skills_log_probs.mean()
