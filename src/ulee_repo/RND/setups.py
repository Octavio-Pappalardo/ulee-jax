from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
import xminigrid
from flax.training.train_state import TrainState
from xminigrid.environment import Environment, EnvParams

from ulee_repo.RND.config import TrainConfig
from ulee_repo.RND.rnd_target_and_predictor_networks import Target_and_Predictor
from ulee_repo.RND.rnd_transformer_actor_critic import ActorCriticTransformer
from ulee_repo.shared_code.wrappers import CustomGymAutoResetWrapper


def setup_actor_critic_train_state(rng: jax.Array, env: Environment, env_params: EnvParams, config: TrainConfig, lr_schedule: Callable | None = None):
    network = ActorCriticTransformer(
        num_actions=env.num_actions(env_params),
        # encoder
        obs_emb_dim=config.obs_emb_dim,
        # transformer
        hidden_dim=config.transformer_hidden_states_dim,
        num_attn_heads=config.num_attn_heads,
        qkv_features=config.qkv_features,
        num_layers_in_transformer=config.num_transformer_blocks,
        gating=config.gating,
        gating_bias=config.gating_bias,
        # mlp actor and critic heads
        head_activation=config.head_activation,
        mlp_dim=config.head_hidden_dim,
    )
    init_obs = jnp.zeros((2, 1, *env.observation_shape(env_params)), dtype=jnp.int32)
    init_memory = jnp.zeros((2, config.past_context_length, config.num_transformer_blocks, config.transformer_hidden_states_dim))
    init_mask = jnp.zeros((2, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_)
    rng, _rng = jax.random.split(rng)
    network_params = network.init(_rng, init_memory, init_obs, init_mask)

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
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
    return rng, train_state


def setup_predictor_train_state(rng: jax.Array, env: Environment, env_params: EnvParams, config: TrainConfig):
    predictor_network = Target_and_Predictor(
        encoder_mode=config.rnd_encoder_mode,
        output_embedding_dim=config.rnd_output_embedding_dim,
        obs_emb_dim=config.obs_emb_dim,
        head_activation=config.rnd_head_activation,
        mlp_dim=config.rnd_head_hidden_dim,
    )
    init_input = jnp.zeros((2, *env.observation_shape(env_params)), dtype=jnp.int32)
    rng, _rng = jax.random.split(rng)
    predictor_network_params = predictor_network.init(_rng, init_input)
    predictor_network_optimizer = optax.chain(
        optax.inject_hyperparams(optax.adam)(learning_rate=config.predictor_network_lr, eps=config.adam_eps),
    )
    predictor_train_state = TrainState.create(apply_fn=predictor_network.apply, params=predictor_network_params, tx=predictor_network_optimizer)

    return rng, predictor_train_state


def setup_target_train_state(rng: jax.Array, env: Environment, env_params: EnvParams, config: TrainConfig):
    target_network = Target_and_Predictor(
        encoder_mode=config.rnd_encoder_mode,
        output_embedding_dim=config.rnd_output_embedding_dim,
        obs_emb_dim=config.obs_emb_dim,
        head_activation=config.rnd_head_activation,
        mlp_dim=config.rnd_head_hidden_dim,
    )
    init_input = jnp.zeros((2, *env.observation_shape(env_params)), dtype=jnp.int32)
    rng, _rng = jax.random.split(rng)
    target_network_params = target_network.init(_rng, init_input)
    target_train_state = TrainState.create(apply_fn=target_network.apply, params=target_network_params, tx=optax.identity())

    return rng, target_train_state


def set_up_for_training(config: TrainConfig):
    rng = jax.random.key(config.train_seed)

    # setup environment & benchmark
    if config.episode_max_steps:
        env, env_params = xminigrid.make(config.env_id, max_steps=config.episode_max_steps)
    else:
        env, env_params = xminigrid.make(config.env_id)
    env = CustomGymAutoResetWrapper(env)
    benchmark = xminigrid.load_benchmark(config.benchmark_id)

    # setup agent training state
    if config.anneal_lr:

        def linear_schedule(count):
            total_param_updates_per_batch = config.num_minibatches * config.update_epochs * config.num_updates_per_batch
            frac = 1.0 - (count // total_param_updates_per_batch) / config.num_batches_of_envs
            return config.lr * frac

        rng, agent_train_state = setup_actor_critic_train_state(rng, env, env_params, config, lr_schedule=linear_schedule)
    else:
        rng, agent_train_state = setup_actor_critic_train_state(rng, env, env_params, config)

    # setup rnd predictor training state
    rng, predictor_train_state = setup_predictor_train_state(rng, env, env_params, config)

    # setup rnd target network train state. Note we store the parameters in a TrainState for convenience, even if the parameters are never updated
    rng, target_train_state = setup_target_train_state(rng, env, env_params, config)

    return rng, env, env_params, benchmark, agent_train_state, predictor_train_state, target_train_state
