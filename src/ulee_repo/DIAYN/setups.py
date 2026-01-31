from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
import xminigrid
from flax import struct
from flax.training.train_state import TrainState
from xminigrid.environment import Environment, EnvParams

from ulee_repo.DIAYN.config import TrainConfig
from ulee_repo.networks.diayn_discriminator import Skill_Discriminator
from ulee_repo.networks.diayn_transformer_actor_critic import DiaynActorCriticTransformer
from ulee_repo.shared_code.trainsition_objects import State_Data
from ulee_repo.shared_code.wrappers import CustomAutoResetWrapper_NoGoals, CustomGymAutoResetWrapper


class TrainStateWithConstants(TrainState):
    constants: Any = struct.field(pytree_node=True, default=None)


def setup_diayn_agent_train_state(rng: jax.Array, env: Environment, env_params: EnvParams, config: TrainConfig, lr_schedule: Callable | None = None):
    agent_network = DiaynActorCriticTransformer(
        num_actions=env.num_actions(env_params),
        # encoder
        obs_emb_dim=config.obs_emb_dim,
        num_skills=config.num_skills,
        skill_emb_dim=config.skill_emb_dim,
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
        # skill bias
        skill_bias=config.skill_bias,
        skill_bias_scale=config.skill_bias_scale,
    )

    # setup optimizer
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

    # initialize parameters and setup train state
    rng, _rng = jax.random.split(rng)
    init_obs = jnp.zeros((2, 1, *env.observation_shape(env_params)), dtype=jnp.int32)
    init_skill = jnp.zeros((2, 1), dtype=jnp.int32)
    init_memory = jnp.zeros((2, config.past_context_length, config.num_transformer_blocks, config.transformer_hidden_states_dim))
    init_mask = jnp.zeros((2, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_)
    if not config.skill_bias:
        network_params = agent_network.init(_rng, init_memory, init_obs, init_skill, init_mask)
        agent_train_state = TrainStateWithConstants.create(apply_fn=agent_network.apply, params=network_params, tx=tx)
    else:
        _rng1, _rng2 = jax.random.split(_rng)
        network_variables = agent_network.init({"params": _rng1, "constants": _rng2}, init_memory, init_obs, init_skill, init_mask)
        network_params = network_variables["params"]
        network_constants = network_variables["constants"]

        # modifications to pass constants implicitly
        def apply_with_constants(params, *args, **kwargs):
            return agent_network.apply({"params": params, "constants": network_constants}, *args, **kwargs)

        agent_train_state = TrainStateWithConstants.create(apply_fn=apply_with_constants, params=network_params, tx=tx, constants=network_constants)

    return rng, agent_train_state


def setup_diayn_discriminator_train_state(rng: jax.Array, env_params: EnvParams, config: TrainConfig):
    skill_discriminator_network = Skill_Discriminator(
        discriminator_mode=config.discriminator_mode,
        num_skills=config.num_skills,
        grid_state_emb_dim=config.grid_state_emb_dim,
        head_activation=config.discriminator_head_activation,
        mlp_dim=config.discriminator_head_hidden_dim,
    )

    init_input = State_Data(grid_state=jnp.zeros((2, env_params.height, env_params.width, 2), dtype=jnp.int32), agent_pos=jnp.zeros((2, 2), dtype=jnp.int32))
    rng, _rng = jax.random.split(rng)
    discriminator_network_params = skill_discriminator_network.init(_rng, init_input)
    discriminator_network_optimizer = optax.chain(
        optax.inject_hyperparams(optax.adam)(learning_rate=config.discriminator_network_lr, eps=config.adam_eps),
    )
    discriminator_train_state = TrainState.create(apply_fn=skill_discriminator_network.apply, params=discriminator_network_params, tx=discriminator_network_optimizer)

    return rng, discriminator_train_state


def set_up_for_training(config: TrainConfig):
    rng = jax.random.key(config.train_seed)

    # setup environment and benchmark
    if config.episode_max_steps:
        env, env_params = xminigrid.make(config.env_id, max_steps=config.episode_max_steps)
    else:
        env, env_params = xminigrid.make(config.env_id)
    env_real_goals = CustomGymAutoResetWrapper(env)
    env_no_goals = CustomAutoResetWrapper_NoGoals(env)
    benchmark = xminigrid.load_benchmark(config.benchmark_id)

    # setup policy training state
    if config.anneal_lr:

        def linear_schedule(count):
            total_param_updates_per_batch = config.num_minibatches * config.update_epochs * config.num_updates_per_batch
            frac = 1.0 - (count // total_param_updates_per_batch) / config.num_batches_of_envs
            return config.lr * frac

        rng, agent_train_state = setup_diayn_agent_train_state(rng, env, env_params, config, lr_schedule=linear_schedule)
    else:
        rng, agent_train_state = setup_diayn_agent_train_state(rng, env, env_params, config)

    # setup discriminator training state
    rng, discriminator_train_state = setup_diayn_discriminator_train_state(rng, env_params, config)

    return rng, env_no_goals, env_real_goals, env_params, benchmark, agent_train_state, discriminator_train_state
