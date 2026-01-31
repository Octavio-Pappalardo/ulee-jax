from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
import xminigrid
from flax.training.train_state import TrainState
from xminigrid.environment import Environment, EnvParams

from ulee_repo.networks.transformer_actor_critic import ActorCriticTransformer
from ulee_repo.RL2.config import TrainConfig
from ulee_repo.shared_code.wrappers import CustomGymAutoResetWrapper


def setup_meta_learner_train_state(rng: jax.Array, env: Environment, env_params: EnvParams, config: TrainConfig, lr_schedule: Callable | None = None):
    meta_learner_network = ActorCriticTransformer(
        algorithm_id="meta_learning",
        num_actions=env.num_actions(env_params),
        # encoder
        obs_emb_dim=config.obs_emb_dim,
        action_emb_dim=config.action_emb_dim,
        # transformer
        hidden_dim=config.transformer_hidden_states_dim,
        num_attn_heads=config.num_attn_heads,
        qkv_features=config.qkv_features,
        num_layers_in_transformer=config.num_transformer_blocks,
        gating=config.gating,
        gating_bias=config.gating_bias,
        # actor and critic heads
        head_activation=config.head_activation,
        mlp_dim=config.head_hidden_dim,
    )
    init_input = {
        "observation": jnp.zeros((2, 1, *env.observation_shape(env_params)), dtype=jnp.int32),
        "prev_action": jnp.zeros((2, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((2, 1)),
        "prev_done": jnp.zeros((2, 1), dtype=jnp.bool_),
    }
    init_memory = jnp.zeros((2, config.past_context_length, config.num_transformer_blocks, config.transformer_hidden_states_dim))
    init_mask = jnp.zeros((2, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_)
    rng, _rng = jax.random.split(rng)
    meta_learner_network_params = meta_learner_network.init(_rng, init_memory, init_input, init_mask)

    if lr_schedule is not None:
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule, eps=config.adam_eps),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=config.lr, eps=config.adam_eps),
        )

    meta_learner_train_state = TrainState.create(apply_fn=meta_learner_network.apply, params=meta_learner_network_params, tx=tx)

    return rng, meta_learner_train_state


def set_up_for_training(config: TrainConfig):
    rng = jax.random.key(config.train_seed)

    # setup environment and benchmark
    if config.episode_max_steps:
        env, env_params = xminigrid.make(config.env_id, max_steps=config.episode_max_steps)
    else:
        env, env_params = xminigrid.make(config.env_id)
    env = CustomGymAutoResetWrapper(env)
    benchmark = xminigrid.load_benchmark(config.benchmark_id)

    # setup agent training state
    if config.anneal_lr:

        def linear_schedule(count):
            total_inner_updates = config.num_minibatches * config.update_epochs * config.num_updates_per_batch
            frac = 1.0 - (count // total_inner_updates) / config.num_batches_of_envs
            return config.lr * frac

        rng, train_state = setup_meta_learner_train_state(rng, env, env_params, config, lr_schedule=linear_schedule)
    else:
        rng, train_state = setup_meta_learner_train_state(rng, env, env_params, config)

    return rng, env, env_params, benchmark, train_state
