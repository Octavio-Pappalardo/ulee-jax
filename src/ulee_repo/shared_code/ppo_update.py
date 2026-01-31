import jax
import jax.numpy as jnp

from ulee_repo.networks.diayn_transformer_actor_critic import DiaynActorCriticTransformer
from ulee_repo.networks.transformer_actor_critic import ActorCriticTransformer
from ulee_repo.shared_code.trainsition_objects import Transition_data_base


# Utility functions
def indices_select(x, y):
    return x[y]


batch_indices_select = jax.vmap(indices_select)

roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)


def batchify(x):
    return jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:]))


##############---------------------------------------


def calculate_gae(
    transitions: Transition_data_base,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = transition.reward + gamma * next_value * (1 - transition.done) - transition.value
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + transitions.value


##############---------------------------------------


def update_epoch(update_state, unused_, algorithm_id, config):
    def _update_minbatch(train_state, batch_info):
        def _loss_fn(params, transitions, memories_batch, advantages, targets):
            # all elements in the transitions pytree, the advantages, and targets have shape (num_envs/num_minibatches, seq_len, ... ).
            # memories_batch: (num_envs/num_minibatches, seq_len + past_context_length , num_tranformer_layers, hidden_dim)

            # collect cached memories from the first step of every subsequence_length_in_loss_calculation sequence of steps
            memories_batch = batch_indices_select(memories_batch, transitions.memories_indices[:, :: config.subsequence_length_in_loss_calculation])
            memories_batch = batchify(memories_batch)
            # creat the masks for processing each subsequence_length_in_loss_calculation sequence of steps
            memories_mask = transitions.memories_mask.reshape((-1, config.subsequence_length_in_loss_calculation, *transitions.memories_mask.shape[2:]))
            memories_mask = jnp.swapaxes(memories_mask, 1, 2)
            memories_mask = jnp.concatenate((memories_mask, jnp.zeros((*memories_mask.shape[:-1], config.subsequence_length_in_loss_calculation - 1), dtype=jnp.bool_)), axis=-1)
            memories_mask = roll_vmap(memories_mask, jnp.arange(0, config.subsequence_length_in_loss_calculation), -1)

            # reshape to shapes (minibatch_size * (seq_len / subsequence_length_in_loss_calculation), subsequence_length_in_loss_calculation, ...)
            transitions, targets, advantages = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, config.subsequence_length_in_loss_calculation, *x.shape[2:])), (transitions, targets, advantages))

            # agent outputs
            if algorithm_id == "standard_ppo":
                pi, value = train_state.apply_fn(params, memories_batch, transitions.obs, memories_mask, method=ActorCriticTransformer.model_forward_train)
            elif algorithm_id == "diayn":
                pi, value = train_state.apply_fn(params, memories_batch, transitions.obs, transitions.skill, memories_mask, method=DiaynActorCriticTransformer.model_forward_train)
            elif algorithm_id == "meta_learning":
                network_inputs = {
                    "observation": transitions.obs,
                    "prev_action": transitions.prev_action,
                    "prev_reward": transitions.prev_reward,
                    "prev_done": transitions.prev_done,
                }
                pi, value = train_state.apply_fn(params, memories_batch, network_inputs, memories_mask, method=ActorCriticTransformer.model_forward_train)

            log_prob = pi.log_prob(transitions.action)

            # value loss
            value_pred_clipped = transitions.value + (value - transitions.value).clip(-config.clip_eps, config.clip_eps)
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            # actor loss
            log_ratio = log_prob - transitions.log_prob
            ratio = jnp.exp(log_ratio)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            loss_actor1 = ratio * advantages
            loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - config.clip_eps,
                    1.0 + config.clip_eps,
                )
                * advantages
            )
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            total_loss = loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy

            # KL divergence
            approx_kl = jnp.mean((ratio - 1) - log_ratio)

            return total_loss, (value_loss, loss_actor, entropy, approx_kl)

        transitions, memories_batch, advantages, targets = batch_info

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

        # compute losses and gradients
        (total_loss, (value_loss, actor_loss, entropy, approx_kl)), grads = grad_fn(train_state.params, transitions, memories_batch, advantages, targets)
        update_info = {
            "total_loss": total_loss,
            "value_loss": value_loss,
            "actor_loss": actor_loss,
            "entropy": entropy,
            "kl": approx_kl,
        }
        # perform update
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, update_info

    rng, train_state, transitions, memories_batch, advantages, targets = update_state

    ###  prepare minibatches of data
    rng, _rng = jax.random.split(rng)
    permutation = jax.random.permutation(_rng, config.num_envs_per_batch)
    batch = (transitions, memories_batch, advantages, targets)
    batch = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1),
        batch,
    )
    shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, [config.num_minibatches, -1, *list(x.shape[1:])]),
        shuffled_batch,
    )

    ### loop training over minibatches
    train_state, update_info = jax.lax.scan(_update_minbatch, train_state, minibatches)

    update_state = (rng, train_state, transitions, memories_batch, advantages, targets)

    return update_state, update_info
