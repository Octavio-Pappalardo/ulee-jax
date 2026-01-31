import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax
import xminigrid
from flax.core import unfreeze
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from jax.tree_util import Partial
from xminigrid.environment import Environment, EnvParams

from ulee_repo.DIAYN.config import TrainConfig as DIAYNTrainConfig
from ulee_repo.DIAYN.setups import TrainStateWithConstants
from ulee_repo.evaluations.rollouts import create_benchmark_step_func, eval_rollout
from ulee_repo.networks.diayn_transformer_actor_critic import DiaynActorCriticTransformer
from ulee_repo.shared_code.logging import generate_run_name, wandb_log_training_metrics
from ulee_repo.shared_code.ppo_update import calculate_gae, update_epoch
from ulee_repo.shared_code.trainsition_objects import State_Data, Transition_data_diayn
from ulee_repo.shared_code.wrappers import CustomGymAutoResetWrapper

#### VISITATION HISTORY PER SKILL ####


def eval_diayn_skills_without_extrinsic_rewards(
    rng: jax.Array,
    eval_reset_rng: jax.Array,
    env: Environment,
    eval_env_params: EnvParams,
    step_func: Callable,
    train_state: TrainState,
    config,
    goals: jax.Array,
    num_episodes: int,
):
    """Collect the history of positions for each skill throughout multiple episodes"""
    rng, _rng = jax.random.split(rng, num=2)
    carry = jax.random.split(_rng, num=config.eval_num_envs)
    xs = jnp.arange(config.num_skills)

    def body_func(carry, skill, eval_reset_rng, env, eval_env_params, step_func, train_state, num_eval_episodes_per_skill, config, goals):
        actions_rng = carry
        new_actions_rng, stats, position_history = jax.vmap(eval_rollout, in_axes=(0, 0, None, 0, None, None, None, None, None, 0, None))(
            actions_rng,
            eval_reset_rng,
            env,
            eval_env_params,
            step_func,
            train_state,
            num_eval_episodes_per_skill,
            "diayn",
            config,
            goals,
            skill,
        )
        carry = new_actions_rng
        return carry, (stats, position_history)

    _, (stats, postions_history_for_each_skill) = jax.lax.scan(
        Partial(
            body_func,
            eval_reset_rng=eval_reset_rng,
            env=env,
            eval_env_params=eval_env_params,
            step_func=step_func,
            train_state=train_state,
            num_eval_episodes_per_skill=num_episodes,
            config=config,
            goals=goals,
        ),
        carry,
        xs,
    )

    #  postions_history_for_each_skill has shape (num_skills, num_episodes, env_params.max_steps, 2)
    return stats, postions_history_for_each_skill


### EVALUATION WITH EXTRINSIC REWARDS BUT WITHOUT NETWORK UPDATES / FEW SHOT ADAPTATION #####


def eval_diayn_find_best_skill(
    rng: jax.Array,
    eval_reset_rng: jax.Array,
    env: Environment,
    eval_env_params: EnvParams,
    step_func: Callable,
    train_state: TrainState,
    config,
    goals: jax.Array,
):
    """
    For each skill in [0, num_skills), run num_eval_episodes_per_skill episodes and then return an array with
    the best performing skill for each environment. #(num_envs,)
    """
    ## ------- COLLECT DATA -------

    rng, _rng = jax.random.split(rng, num=2)
    carry = jax.random.split(_rng, num=config.eval_num_envs)
    xs = jnp.arange(config.num_skills)

    def body_func(carry, skill, eval_reset_rng, env, eval_env_params, step_func, train_state, num_eval_episodes_per_skill, config, goals):
        actions_rng = carry
        new_actions_rng, stats, _ = jax.vmap(eval_rollout, in_axes=(0, 0, None, 0, None, None, None, None, None, 0, None))(
            actions_rng,
            eval_reset_rng,
            env,
            eval_env_params,
            step_func,
            train_state,
            num_eval_episodes_per_skill,
            "diayn",
            config,
            goals,
            skill,
        )
        carry = new_actions_rng
        return carry, stats

    # stats_for_each_skill is a pytree with elements of shape (num_skills, num_envs, num_eval_episodes_per_skill)
    _actions_rng, stats_for_each_skill = jax.lax.scan(
        Partial(
            body_func,
            eval_reset_rng=eval_reset_rng,
            env=env,
            eval_env_params=eval_env_params,
            step_func=step_func,
            train_state=train_state,
            num_eval_episodes_per_skill=config.num_eval_episodes_per_skill,
            config=config,
            goals=goals,
        ),
        carry,
        xs,
    )

    ##------- RETRIEVE BEST SKILL -------

    skills_performance = jnp.swapaxes(stats_for_each_skill.returns, 0, 1)  # (num_envs, num_skills, num_eval_episodes_per_skill)

    skills_performance = skills_performance.mean(axis=-1)  # (num_envs, num_skills)

    best_skills = jnp.argmax(skills_performance, axis=-1)  # (num_envs,)

    return rng, stats_for_each_skill, best_skills


def eval_diayn_selecting_best_skill(
    rng: jax.Array,
    eval_reset_rng: jax.Array,
    env: Environment,
    eval_env_params: EnvParams,
    step_func: Callable,
    train_state: TrainState,
    config,
    goals: jax.Array,
):
    """
      Returns a tuple with:
    - stats_for_each_skill: RolloutEpisodeStats with returns and lengths from the first phase (for each skill) | shape [num_skills, num_envs, num_eval_episodes_per_skill]
    - best_skill_stats: RolloutEpisodeStats with returns and lengths from the last phase (for the best skill) | shape [num_envs, num_eval_episodes_with_best_skill]
    """
    # collect data with each skill and find the best performing one for each env
    rng, stats_for_each_skill, best_skills = eval_diayn_find_best_skill(
        rng=rng,
        eval_reset_rng=eval_reset_rng,
        env=env,
        eval_env_params=eval_env_params,
        step_func=step_func,
        train_state=train_state,
        config=config,
        goals=goals,
    )

    rng, _rng = jax.random.split(rng, num=2)
    actions_rng = jax.random.split(_rng, num=config.eval_num_envs)

    # collect data on each environment using the best skill
    _, best_skill_stats, _ = jax.vmap(eval_rollout, in_axes=(0, 0, None, 0, None, None, None, None, None, 0, 0))(
        actions_rng,
        eval_reset_rng,
        env,
        eval_env_params,
        step_func,
        train_state,
        config.num_eval_episodes_with_best_skill,
        "diayn",
        config,
        goals,
        best_skills,
    )

    return (rng, stats_for_each_skill, best_skill_stats)


################  EVAL OF LONG HORIZON ADAPTATION | FINE-TUNING  #########################


def eval_diayn_finetune(
    rng: jax.Array,
    env_id: str,
    benchmark_id: str,
    weights_path: Path | str,
    results_path: Path | str,
    num_envs: int,
    total_timesteps: int,
    num_steps_per_env: int,
    num_steps_per_update: int,
    *,
    eval_on_test_benchmark: bool = True,
    **kwargs,
):
    # load params and config
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data = orbax_checkpointer.restore(weights_path)
    config = unfreeze(data["config"])
    config["env_id"] = env_id
    config["benchmark_id"] = benchmark_id
    config = DIAYNTrainConfig(**config)
    # overwrite with provided configurations for finetuning
    config = replace(
        config, eval_num_envs=num_envs, num_envs_per_batch=num_envs, total_timesteps=total_timesteps, num_steps_per_env=num_steps_per_env, num_steps_per_update=num_steps_per_update, **kwargs
    )
    config.__post_init__()

    # setup env and benchmark
    if config.episode_max_steps:
        env, env_params = xminigrid.make(env_id, max_steps=config.episode_max_steps)
    else:
        env, env_params = xminigrid.make(env_id)
    env = CustomGymAutoResetWrapper(env)
    bench = xminigrid.load_benchmark(benchmark_id)
    if eval_on_test_benchmark:
        split_rng = jax.random.key(config.benchmark_split_seed)
        _, benchmark = bench.shuffle(key=split_rng).split(prop=config.benchmark_train_percentage)
    else:
        split_rng = jax.random.key(config.benchmark_split_seed)
        benchmark, _ = bench.shuffle(key=split_rng).split(prop=config.benchmark_train_percentage)

    # Setup optimizer, network with correct configs, and create train_state with trained params
    network = DiaynActorCriticTransformer(
        num_actions=env.num_actions(env_params),
        obs_emb_dim=config.obs_emb_dim,
        num_skills=config.num_skills,
        skill_emb_dim=config.skill_emb_dim,
        hidden_dim=config.transformer_hidden_states_dim,
        num_attn_heads=config.num_attn_heads,
        qkv_features=config.qkv_features,
        num_layers_in_transformer=config.num_transformer_blocks,
        gating=config.gating,
        gating_bias=config.gating_bias,
        head_activation=config.head_activation,
        mlp_dim=config.head_hidden_dim,
        skill_bias=config.skill_bias,
        skill_bias_scale=config.skill_bias_scale,
    )

    def linear_schedule(count):
        total_param_updates_per_batch = config.num_minibatches * config.update_epochs * config.num_updates_per_batch
        frac = 1.0 - (count // total_param_updates_per_batch) / config.num_batches_of_envs
        return config.lr * frac

    if config.anneal_lr:
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=linear_schedule, eps=config.adam_eps),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=config.lr, eps=config.adam_eps),
        )
    network_variables = data["agent_params"]
    # network_params = data['best_meta_learner_params']
    if not config.skill_bias:
        train_state = TrainStateWithConstants.create(apply_fn=network.apply, params=network_variables["params"], tx=tx)
    else:

        def apply_with_constants(params, *args, **kwargs):
            return network.apply({"params": params, "constants": network_variables["constants"]}, *args, **kwargs)

        train_state = TrainStateWithConstants.create(apply_fn=apply_with_constants, params=network_variables["params"], tx=tx, constants=network_variables["constants"])

    ## Run Finetuning
    jitted_partial_finetune = jax.jit(Partial(diayn_finetune_with_fixed_skill, env=env, env_params=env_params, benchmark=benchmark, config=config))
    results = jax.block_until_ready(
        jitted_partial_finetune(
            rng=rng,
            train_state=train_state,
        )
    )

    try:
        # save results
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_dict = {
            "finetuned_params": {"params": results["agent_state"].params, "constants": results["agent_state"].constants},
            "metrics": results["metrics"],
        }
        save_args = orbax_utils.save_args_from_target(results_dict)
        orbax_checkpointer.save(results_path, results_dict, save_args=save_args)
        print(f"Saved finetuning evaluation results to {results_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

    try:
        # log training metrics to wandb
        run_name = generate_run_name(algorithm_name="DIAYN", config=config, prefix="finetuning")
        tags = ["diayn", "finetune", "final"]
        wandb_log_training_metrics(results["metrics"], config, run_name=run_name, tags=tags)
    except Exception as e:
        print(f"Error logging metrics to wandb: {e}")

    return results


def diayn_finetune_with_fixed_skill(rng: jax.Array, train_state: TrainState, env: Environment, env_params: EnvParams, benchmark, config):
    """
    1) Sample a batch of environments on which to finetune and setup for running the policy in these environments. This evaluation is performed over a fixed set of tasks and initial conditions for each.
    2) Run each skill on these environments (no network updates yet) and select the best performing skill for each environment
    3) Run the policy using the best performing skill for each environment while finetuning the network
    """
    ####   1  SETUP  ####
    rng, ruleset_rng_base, reset_rng_base = jax.random.split(rng, num=3)
    reset_rng = jax.random.split(reset_rng_base, num=config.num_envs_per_batch)

    if config.env_id.startswith("XLand-MiniGrid"):
        rulesets_rng = jax.random.split(ruleset_rng_base, num=config.num_envs_per_batch)
        rulesets = jax.vmap(benchmark.sample_ruleset)(rulesets_rng)
        env_params = env_params.replace(ruleset=rulesets)

    step_func = create_benchmark_step_func(env)
    goals = jnp.zeros((config.eval_num_envs, 10), dtype=jnp.int32)

    ####  2 FIND BEST SKILL FOR EACH ENV ###
    rng, _, best_skills = eval_diayn_find_best_skill(
        rng=rng,
        eval_reset_rng=reset_rng,
        env=env,
        eval_env_params=env_params,
        step_func=step_func,
        train_state=train_state,
        config=config,
        goals=goals,
    )

    #### 3 RUN POLICY WITH BEST SKILLS AND FINE TUNE NETWORK ####
    train_info = finetune_with_fixed_skills(
        rng,
        train_state,
        best_skills,
        env,
        reset_rng,
        env_params,
        config,
    )
    return train_info


def finetune_with_fixed_skills(
    rng,
    train_state,
    skill,
    env,
    reset_rng,
    env_params,
    config,
):
    # First we generate slight variations of the functions used for training diayn. These use fixed skills for each environment and they use extrinsic rewards instead of rewards generated by the skill discriminator.
    def step_envs(runner_state, unused, skill, env, env_params, config):
        rng, train_state, prev_timestep, memories, memories_mask, memories_mask_idx, current_update_step_num = runner_state

        memories_mask_idx = jnp.where(prev_timestep.last(), config.past_context_length, jnp.clip(memories_mask_idx - 1, 0, config.past_context_length))
        memories_mask = jnp.where(
            prev_timestep.last()[:, None, None, None], jnp.zeros((config.eval_num_envs, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_), memories_mask
        )
        memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, config.past_context_length + 1)
        memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(config.num_attn_heads, 1)
        memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

        rng, _rng = jax.random.split(rng)
        pi, value, memories_out = train_state.apply_fn(
            train_state.params, memories, prev_timestep.observation[:, None], skill[:, None], memories_mask, method=DiaynActorCriticTransformer.model_forward_eval
        )
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        timestep = jax.vmap(env.step, in_axes=0)(env_params, prev_timestep, action)

        memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)
        memory_indices = jnp.arange(0, config.past_context_length)[None, :] + current_update_step_num * jnp.ones((config.eval_num_envs, 1), dtype=jnp.int32)

        transition = Transition_data_diayn(
            done=timestep.last(),
            action=action,
            value=value,
            reward=timestep.reward,
            log_prob=log_prob,
            obs=prev_timestep.observation,
            state_data=State_Data(  # not used
                grid_state=jnp.zeros((config.eval_num_envs, 2), dtype=jnp.float32),
                agent_pos=jnp.zeros((config.eval_num_envs, 2), dtype=jnp.float32),
            ),
            skill=skill,
            memories_mask=memories_mask.squeeze(),
            memories_indices=memory_indices,
        )
        runner_state = (rng, train_state, timestep, memories, memories_mask, memories_mask_idx, current_update_step_num + 1)

        return runner_state, (transition, memories_out)

    def collect_data(runner_state, skill, env, env_params, config):
        runner_state, (transitions, memories_batch) = jax.lax.scan(Partial(step_envs, skill=skill, env=env, env_params=env_params, config=config), runner_state, None, config.num_steps_per_update)
        return runner_state, transitions, memories_batch

    def update_agent(runner_state, transitions, memories_batch, skill, config):
        rng, train_state, timestep, memories, memories_mask, memories_mask_idx, _ = runner_state

        _, last_val, _ = train_state.apply_fn(train_state.params, memories, timestep.observation[:, None], skill[:, None], memories_mask, method=DiaynActorCriticTransformer.model_forward_eval)
        advantages, targets = calculate_gae(transitions, last_val, config.gamma, config.gae_lambda)

        update_state = (rng, train_state, transitions, memories_batch, advantages, targets)
        update_state, metrics = jax.lax.scan(Partial(update_epoch, algorithm_id="diayn", config=config), update_state, None, config.update_epochs)
        rng, train_state = update_state[:2]
        runner_state = (rng, train_state, timestep, memories, memories_mask, memories_mask_idx, 0)

        metrics = jtu.tree_map(lambda x: x.mean(-1).mean(-1), metrics)

        return runner_state, metrics

    def collect_data_and_update(runner_state, _unused, skill, env, env_params, config):
        memories_previous = runner_state[3]

        runner_state, transitions, memories_batch = collect_data(runner_state, skill, env, env_params, config)

        memories_batch = jnp.concatenate([jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0)

        runner_state, metrics = update_agent(runner_state, transitions, memories_batch, skill, config)

        return runner_state, metrics

    def finetune_with_fixed_skills_one_iteration(
        meta_state,
        _unused,
        skill,
        env,
        reset_rng,
        env_params,
        config,
    ):
        rng, train_state = meta_state

        #####----------  SETUP   --------#####
        timestep = jax.vmap(env.reset, in_axes=(0, 0))(env_params, reset_rng)
        memories = jnp.zeros((config.eval_num_envs, config.past_context_length, config.num_transformer_blocks, config.transformer_hidden_states_dim))
        memories_mask = jnp.zeros((config.eval_num_envs, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_)
        memories_mask_idx = jnp.zeros((config.eval_num_envs,), dtype=jnp.int32) + (config.past_context_length + 1)

        #####------- TRAINING -------#####
        runner_state = (rng, train_state, timestep, memories, memories_mask, memories_mask_idx, 0)
        runner_state, metrics = jax.lax.scan(Partial(collect_data_and_update, skill=skill, env=env, env_params=env_params, config=config), runner_state, None, config.num_updates_per_batch)
        metrics = jtu.tree_map(lambda x: x.mean(-1), metrics)
        rng, train_state = runner_state[:2]

        ####----------  EVALUATION --------#######
        # note the evaluation is performed over the same fixed environments, resets, and skills used for finetuning.
        rng, _rng = jax.random.split(rng, num=2)
        actions_rng = jax.random.split(_rng, num=config.eval_num_envs)
        step_func = create_benchmark_step_func(env)
        goals = jnp.zeros((config.eval_num_envs, 10), dtype=jnp.int32)
        _, eval_stats, _ = jax.vmap(eval_rollout, in_axes=(0, 0, None, 0, None, None, None, None, None, 0, 0))(
            actions_rng,
            reset_rng,
            env,
            env_params,
            step_func,
            train_state,
            config.num_eval_episodes_with_best_skill,
            "diayn",
            config,
            goals,
            skill,
        )
        metrics.update(
            {
                "eval/returns": eval_stats.returns,
                "eval/lengths": eval_stats.lengths,
                "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
            }
        )
        ##############
        meta_state = (rng, train_state)
        return meta_state, metrics

    meta_state = (rng, train_state)
    meta_state, metrics = jax.lax.scan(
        Partial(finetune_with_fixed_skills_one_iteration, skill=skill, env=env, reset_rng=reset_rng, env_params=env_params, config=config), meta_state, None, config.num_batches_of_envs
    )

    return {"agent_state": meta_state[1], "metrics": metrics}
