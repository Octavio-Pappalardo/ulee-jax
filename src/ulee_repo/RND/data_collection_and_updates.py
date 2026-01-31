import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.training.train_state import TrainState
from jax.tree_util import Partial

from ulee_repo.RND.normalization_utils import compute_intrinsic_returns, update_normalization_stats
from ulee_repo.RND.ppo_update import calculate_gae, update_epoch
from ulee_repo.RND.rnd_transformer_actor_critic import ActorCriticTransformer
from ulee_repo.shared_code.trainsition_objects import Transition_data_rnd

#############-------------------------------


def step_envs(runner_state, unused, env, env_params, config):
    rng, train_state, predictor_train_state, target_train_state, normalization_stats, prev_timestep, memories, memories_mask, memories_mask_idx, current_update_step_num = runner_state

    # Update transformer mask or reset it if new episode
    memories_mask_idx = jnp.where(prev_timestep.last(), config.past_context_length, jnp.clip(memories_mask_idx - 1, 0, config.past_context_length))
    memories_mask = jnp.where(
        prev_timestep.last()[:, None, None, None], jnp.zeros((config.num_envs_per_batch, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_), memories_mask
    )
    memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, config.past_context_length + 1)
    memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(config.num_attn_heads, 1)
    memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

    # --------------------------

    # Select actions
    rng, _rng = jax.random.split(rng)
    pi, value, value_intrinsic, memories_out = train_state.apply_fn(train_state.params, memories, prev_timestep.observation[:, None], memories_mask, method=ActorCriticTransformer.model_forward_eval)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)

    # Update memory buffer
    memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

    # Step environments
    timestep = jax.vmap(env.step, in_axes=0)(env_params, prev_timestep, action)

    memory_indices = jnp.arange(0, config.past_context_length)[None, :] + current_update_step_num * jnp.ones((config.num_envs_per_batch, 1), dtype=jnp.int32)

    # Store transition data
    transition = Transition_data_rnd(
        done=timestep.last(),
        action=action,
        value=value,
        reward=timestep.reward,
        log_prob=log_prob,
        obs=prev_timestep.observation,
        memories_mask=memories_mask.squeeze(),
        memories_indices=memory_indices,
        next_obs=timestep.observation,
        intrinsic_reward=jnp.zeros_like(timestep.reward),
        intrinsic_value=value_intrinsic,
        done_for_intrinsic=jnp.zeros_like(timestep.last()),  # set always to false because we optimize rnd intrinsic rewards across episodes
    )

    # create updated runner state
    runner_state = (rng, train_state, predictor_train_state, target_train_state, normalization_stats, timestep, memories, memories_mask, memories_mask_idx, current_update_step_num + 1)

    return runner_state, (transition, memories_out)


#############-------------------------------


def collect_data(runner_state, num_steps, env, env_params, config):
    runner_state, (transitions, memories_batch) = jax.lax.scan(Partial(step_envs, env=env, env_params=env_params, config=config), runner_state, None, num_steps)

    return runner_state, transitions, memories_batch


#############-------------------------------


def update_agent(runner_state, transitions, memories_batch, config):
    rng, train_state, predictor_train_state, target_train_state, normalization_stats, timestep, memories, memories_mask, memories_mask_idx, _ = runner_state

    # Compute advantages and targets (GAE) for both extrinsic and intrinsic rewards
    _, last_val, last_val_intrinsic, _ = train_state.apply_fn(
        train_state.params, memories, timestep.observation[:, None], memories_mask, method=ActorCriticTransformer.model_forward_eval
    )  # _, (num_envs), _
    advantages_ext, targets_ext = calculate_gae(transitions, last_val, config.gamma, config.gae_lambda)
    intrinsic_view = transitions.replace(reward=transitions.intrinsic_reward, value=transitions.intrinsic_value, done=transitions.done_for_intrinsic)
    advantages_int, targets_int = calculate_gae(intrinsic_view, last_val_intrinsic, config.gamma_intrinsic, config.gae_lambda_intrinsic)

    advantages = config.extrinsic_coef * advantages_ext + config.intrinsic_coef * advantages_int  # (num_steps, num_envs)
    targets = {"extrinsic": targets_ext, "intrinsic": targets_int}  # each of shape (num_steps, num_envs)

    # Compute loss and update network
    update_state = (rng, train_state, transitions, memories_batch, advantages, targets)
    update_state, metrics = jax.lax.scan(Partial(update_epoch, config=config), update_state, None, config.update_epochs)
    rng, train_state = update_state[:2]

    runner_state = (rng, train_state, predictor_train_state, target_train_state, normalization_stats, timestep, memories, memories_mask, memories_mask_idx, 0)

    metrics = jtu.tree_map(lambda x: x.mean(-1).mean(-1), metrics)

    return runner_state, metrics


#############-------------------------------


def update_rnd_predictor(runner_state, transitions, config):
    # only rng and predictor_train_state_are updated
    rng, train_state, predictor_train_state, target_train_state, normalization_stats, timestep, memories, memories_mask, memories_mask_idx, current_update_step_num = runner_state
    rng, predictor_train_state, predictor_loss_value = rnd_predictor_train(
        rng,
        predictor_train_state,
        target_train_state,
        transitions,
        config.rnd_predictor_update_epochs,
        config.rnd_predictor_num_minibatches,
    )
    runner_state = (rng, train_state, predictor_train_state, target_train_state, normalization_stats, timestep, memories, memories_mask, memories_mask_idx, current_update_step_num)
    return runner_state, predictor_loss_value


#############-------------------------------


def collect_data_and_update(runner_state, _unused, env, env_params, config):
    memories_previous = runner_state[6]  # (batch_size, past_context_length, num_tranformer_layers, hidden_dim)

    runner_state, transitions, memories_batch = collect_data(runner_state, config.num_steps_per_update, env, env_params, config)

    # compute raw rnd intrinsic rewards
    rnd_intrinsic_rewards = compute_rnd_intrinsic_rewards(
        predictor_train_state=runner_state[2],
        target_train_state=runner_state[3],
        transitions=transitions,
        num_chunks=config.num_chunks_in_rnd_rewards_computation,
    )  # (seq_len, batch_size)

    # normalize intrinsic rewards and update the transitions pytree
    runner_state, normalized_rnd_intrinsic_rewards = normalize_rnd_intrinsic_rewards(runner_state, rnd_intrinsic_rewards, config)
    transitions = transitions.replace(intrinsic_reward=normalized_rnd_intrinsic_rewards)

    # Concatenate previous memory with new batch
    memories_batch = jnp.concatenate([jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0)  # (past_context + num_steps_per_update, num_envs, num_tranformer_layers, hidden_dim)

    # update policy and value networks
    runner_state, metrics = update_agent(runner_state, transitions, memories_batch, config)

    # update rnd predictor network
    runner_state, predictor_loss_value = update_rnd_predictor(runner_state, transitions, config)

    metrics.update(
        {
            "rnd_predictor_loss": predictor_loss_value,
        }
    )

    return runner_state, metrics


#############-------------------------------


def compute_rnd_intrinsic_rewards(
    predictor_train_state,
    target_train_state,
    transitions,
    num_chunks: int,
) -> jnp.ndarray:
    # (S, B, ...) -> (S*B, ...)
    S, B = transitions.next_obs.shape[:2]
    transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (S * B, *x.shape[2:])), transitions)
    total_num_steps = transitions.next_obs.shape[0]  # S*B

    chunk_size = total_num_steps // num_chunks
    transitions_chunked = jax.tree_util.tree_map(lambda x: x.reshape((num_chunks, chunk_size, *x.shape[1:])), transitions)

    def body_fun(carry, transitions_chunk):
        targets_chunk = target_train_state.apply_fn(target_train_state.params, transitions_chunk.next_obs)
        preds_chunk = predictor_train_state.apply_fn(predictor_train_state.params, transitions_chunk.next_obs)
        # Compute prediction errors
        pred_errors_chunk = jnp.mean(jnp.square(targets_chunk - preds_chunk), axis=-1)  # (chunk_size,)
        return carry, pred_errors_chunk

    # Scan over chunks
    _, pred_errors_chunked = jax.lax.scan(
        body_fun,
        None,
        transitions_chunked,
    )

    # (num_chunks, chunk_size) -> (S*B,) -> (S, B)
    rnd_intrinsic_rewards = pred_errors_chunked.reshape((total_num_steps,)).reshape((S, B))

    return rnd_intrinsic_rewards


#############-------------------------------


def normalize_rnd_intrinsic_rewards(runner_state, rnd_intrinsic_rewards, config):
    # only normalization_stats is updated in runner_state
    rng, train_state, predictor_train_state, target_train_state, normalization_stats, timestep, memories, memories_mask, memories_mask_idx, current_update_step_num = runner_state

    new_running_forward_return, returns = compute_intrinsic_returns(
        running_forward_return=normalization_stats.running_forward_return,
        rewards=rnd_intrinsic_rewards,
        gamma=config.gamma_intrinsic,
    )  # (batch_size,) , (seq_len, batch_size)
    normalization_stats = normalization_stats.replace(running_forward_return=new_running_forward_return)
    normalization_stats = update_normalization_stats(normalization_stats, returns.flatten())

    rnd_intrinsic_rewards = rnd_intrinsic_rewards / jnp.sqrt(normalization_stats.var + 1e-8)

    runner_state = (rng, train_state, predictor_train_state, target_train_state, normalization_stats, timestep, memories, memories_mask, memories_mask_idx, current_update_step_num)

    return runner_state, rnd_intrinsic_rewards


#############-------------------------------


def rnd_predictor_train(
    rng,
    predictor_train_state: TrainState,
    target_train_state,
    transitions,
    num_epochs: int,
    num_minibatches: int,
):
    # (S, B, ...) -> (S*B, ...)
    transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), transitions)
    total_num_steps = transitions.next_obs.shape[0]  # S*B

    minibatch_size = total_num_steps // num_minibatches

    def loss_fn(predictor_params, transitions_chunk):
        targets_chunk = target_train_state.apply_fn(target_train_state.params, transitions_chunk.next_obs)
        preds_chunk = predictor_train_state.apply_fn(predictor_params, transitions_chunk.next_obs)
        # Compute mse loss
        loss = jnp.mean(jnp.square(targets_chunk - preds_chunk))
        return loss

    def epoch_body_fun(carry, _unused):
        rng, train_state = carry

        # Shuffle data
        rng, shuffle_rng = jax.random.split(rng)
        indices_permuted = jax.random.permutation(shuffle_rng, total_num_steps)
        transitions_shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, indices_permuted, axis=0), transitions)

        # Reshape to (num_minibatches, minibatch_size, ...)
        transitions_shuffled = jax.tree_util.tree_map(lambda x: x.reshape((num_minibatches, minibatch_size, *x.shape[1:])), transitions_shuffled)

        def minibatch_step(train_state, transitions_chunk):
            loss, grads = jax.value_and_grad(loss_fn)(train_state.params, transitions_chunk)
            new_train_state = train_state.apply_gradients(grads=grads)
            return new_train_state, loss

        # loop over minibatches
        train_state, batch_losses = jax.lax.scan(minibatch_step, train_state, transitions_shuffled)
        epoch_loss = jnp.mean(batch_losses)

        return (rng, train_state), epoch_loss

    # Main loop over epochs
    init_carry = (rng, predictor_train_state)
    (rng, final_predictor_train_state), epoch_losses = jax.lax.scan(epoch_body_fun, init_carry, None, num_epochs)

    loss_value = jnp.mean(epoch_losses)

    return rng, final_predictor_train_state, loss_value
