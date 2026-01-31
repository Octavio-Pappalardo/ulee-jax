from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax
import xminigrid
from flax.core import unfreeze
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from jax.tree_util import Partial
from xminigrid.environment import Environment, EnvParams

from ulee_repo.evaluations.rollouts import create_benchmark_step_func, eval_rollout
from ulee_repo.networks.transformer_actor_critic import ActorCriticTransformer
from ulee_repo.RL2.data_collection_and_updates import collect_data_and_update as rl2_collect_data_and_update
from ulee_repo.RL2.main_loop import full_training
from ulee_repo.shared_code.logging import generate_run_name, wandb_log_training_metrics
from ulee_repo.shared_code.wrappers import CustomGymAutoResetWrapper
from ulee_repo.ULEE.config import TrainConfig as MetaLearnerTrainConfig

###### EVAL OF FINE-TUNING ON FIXED TASKS ######


def eval_meta_learner_finetune(
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
    config.pop("goal_search", None)
    config.pop("goal_judge", None)
    config["env_id"] = env_id
    config["benchmark_id"] = benchmark_id
    config = MetaLearnerTrainConfig(**config)
    # overwrite with provided configurations for finetuning
    config = replace(config, num_envs_per_batch=num_envs, total_timesteps=total_timesteps, num_steps_per_env=num_steps_per_env, num_steps_per_update=num_steps_per_update, **kwargs)
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
    config.goal_search.goal_searching_steps_per_env = config.goal_search.goal_searching_episodes_per_env * env_params.max_steps  # for compatibility (need to fill it to log config to wyb)

    # Setup optimizer, meta learning network with correct configs, and create train_state with trained params
    network = ActorCriticTransformer(
        algorithm_id="meta_learning",
        num_actions=env.num_actions(env_params),
        hidden_dim=config.transformer_hidden_states_dim,
        num_attn_heads=config.num_attn_heads,
        qkv_features=config.qkv_features,
        num_layers_in_transformer=config.num_transformer_blocks,
        gating=config.gating,
        gating_bias=config.gating_bias,
        head_activation=config.head_activation,
        mlp_dim=config.head_hidden_dim,
        obs_emb_dim=config.obs_emb_dim,
        action_emb_dim=config.action_emb_dim,
    )
    network_params = data["meta_learner_params"]
    # network_params = data['best_meta_learner_params']

    def linear_schedule(count):
        total_inner_updates = config.num_minibatches * config.update_epochs * config.num_updates_per_batch
        frac = 1.0 - (count // total_inner_updates) / config.num_batches_of_envs
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

    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    ## Run Finetuning
    jitted_partial_finetune = jax.jit(Partial(meta_learner_finetune, env=env, env_params=env_params, benchmark=benchmark, config=config))
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
            "finetuned_params": results["agent_state"].params,
            "metrics": results["metrics"],
        }
        save_args = orbax_utils.save_args_from_target(results_dict)
        orbax_checkpointer.save(results_path, results_dict, save_args=save_args)
        print(f"Saved finetuning evaluation results to {results_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

    try:
        # log training metrics to wandb
        run_name = generate_run_name(algorithm_name="ULEE", config=config, prefix="finetuning")
        tags = ["ulee", "finetune", "final"]
        wandb_log_training_metrics(results["metrics"], config, run_name=run_name, tags=tags, num_final_episodes_for_evaluating_performance=10)
    except Exception as e:
        print(f"Error logging metrics to wandb: {e}")

    return results


def meta_learner_finetune(rng: jax.Array, train_state: TrainState, env: Environment, env_params: EnvParams, benchmark, config):
    ####  SETUP  ####
    rng, ruleset_rng_base, reset_rng_base = jax.random.split(rng, num=3)
    reset_rng = jax.random.split(reset_rng_base, num=config.num_envs_per_batch)

    if config.env_id.startswith("XLand-MiniGrid"):
        rulesets_rng = jax.random.split(ruleset_rng_base, num=config.num_envs_per_batch)
        rulesets = jax.vmap(benchmark.sample_ruleset)(rulesets_rng)
        env_params = env_params.replace(ruleset=rulesets)

    #### RUN POLICY ON NEW ENVIRONMENTS WHILE FINETUNING - memories are never reset ####
    timestep = jax.vmap(env.reset, in_axes=(0, 0))(env_params, reset_rng)
    prev_action = jnp.zeros(config.num_envs_per_batch, dtype=jnp.int32)
    prev_reward = jnp.zeros(config.num_envs_per_batch)
    prev_done = timestep.last()
    memories = jnp.zeros((config.num_envs_per_batch, config.past_context_length, config.num_transformer_blocks, config.transformer_hidden_states_dim))
    memories_mask = jnp.zeros((config.num_envs_per_batch, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_)
    memories_mask_idx = jnp.zeros((config.num_envs_per_batch,), dtype=jnp.int32) + (config.past_context_length + 1)

    runner_state = (rng, train_state, timestep, prev_action, prev_reward, prev_done, memories, memories_mask, memories_mask_idx, 0)

    def finetune_one_iteration(
        runner_state,
        _unused,
        env,
        reset_rng,
        env_params,
        config,
    ):
        # Train
        runner_state, metrics = jax.lax.scan(Partial(rl2_collect_data_and_update, env=env, env_params=env_params, config=config), runner_state, None, config.num_updates_per_batch)

        metrics = jtu.tree_map(lambda x: x.mean(-1), metrics)
        rng, train_state = runner_state[:2]

        # Evaluation
        rng, actions_rng = jax.random.split(rng, num=2)
        actions_rng = jax.random.split(actions_rng, num=config.num_envs_per_batch)
        step_func = create_benchmark_step_func(env)
        goals = jnp.zeros((config.num_envs_per_batch, 10), dtype=jnp.int32)
        # collect evaluation data
        _, eval_stats, _ = jax.vmap(eval_rollout, in_axes=(0, 0, None, 0, None, None, None, None, None, 0))(
            actions_rng,
            reset_rng,
            env,
            env_params,
            step_func,
            train_state,
            config.eval_num_episodes,
            "meta_learning",
            config,
            goals,
        )
        metrics.update(
            {
                "eval/returns": eval_stats.returns,
                "eval/lengths": eval_stats.lengths,
            }
        )
        runner_state = (rng, *runner_state[1:])
        return runner_state, metrics

    final_runner_state, metrics = jax.lax.scan(Partial(finetune_one_iteration, env=env, reset_rng=reset_rng, env_params=env_params, config=config), runner_state, None, config.num_batches_of_envs)

    return {"agent_state": final_runner_state[1], "metrics": metrics}


################ EVAL OF PRE-TRAINING FOR META RL FINETUNING #########################


def eval_meta_learner_finetune_on_meta_rl(
    rng: jax.Array, env_id: str, benchmark_id: str, weights_path: Path | str, results_path: Path | str, num_envs: int, total_timesteps: int, num_steps_per_env: int, num_steps_per_update: int, **kwargs
):
    # load params and config
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data = orbax_checkpointer.restore(weights_path)
    config = unfreeze(data["config"])
    config.pop("goal_search", None)
    config.pop("goal_judge", None)
    config["env_id"] = env_id
    config["benchmark_id"] = benchmark_id
    config = MetaLearnerTrainConfig(**config)
    # overwrite with provided configurations for finetuning
    config = replace(config, num_envs_per_batch=num_envs, total_timesteps=total_timesteps, num_steps_per_env=num_steps_per_env, num_steps_per_update=num_steps_per_update, **kwargs)
    config.__post_init__()

    # setup env and benchmark
    if config.episode_max_steps:
        env, env_params = xminigrid.make(env_id, max_steps=config.episode_max_steps)
    else:
        env, env_params = xminigrid.make(env_id)
    env = CustomGymAutoResetWrapper(env)
    benchmark = xminigrid.load_benchmark(benchmark_id)
    config.goal_search.goal_searching_steps_per_env = config.goal_search.goal_searching_episodes_per_env * env_params.max_steps

    # Setup optimizer, meta learning network with correct configs, and create train_state with trained params
    network = ActorCriticTransformer(
        algorithm_id="meta_learning",
        num_actions=env.num_actions(env_params),
        hidden_dim=config.transformer_hidden_states_dim,
        num_attn_heads=config.num_attn_heads,
        qkv_features=config.qkv_features,
        num_layers_in_transformer=config.num_transformer_blocks,
        gating=config.gating,
        gating_bias=config.gating_bias,
        head_activation=config.head_activation,
        mlp_dim=config.head_hidden_dim,
        obs_emb_dim=config.obs_emb_dim,
        action_emb_dim=config.action_emb_dim,
    )
    network_params = data["meta_learner_params"]
    # network_params = data['best_meta_learner_params']

    def linear_schedule(count):
        total_inner_updates = config.num_minibatches * config.update_epochs * config.num_updates_per_batch
        frac = 1.0 - (count // total_inner_updates) / config.num_batches_of_envs
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

    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    ## Run finetuning on meta RL
    jitted_partial_training = jax.jit(Partial(full_training, env=env, env_params=env_params, benchmark=benchmark, config=config))
    results = jax.block_until_ready(
        jitted_partial_training(
            rng=rng,
            train_state=train_state,
        )
    )

    try:
        # save results
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_dict = {
            "finetuned_params": results["agent_state"].params,
            "finetuned_best_params": results["best"][1],
            "metrics": results["metrics"],
        }
        save_args = orbax_utils.save_args_from_target(results_dict)
        orbax_checkpointer.save(results_path, results_dict, save_args=save_args)
        print(f"Saved finetuning evaluation results to {results_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

    try:
        # log training metrics to wandb
        run_name = generate_run_name(algorithm_name="ULEE", config=config, prefix="finetuning_meta_rl")
        tags = ["ulee", "meta-finetune", "final"]
        wandb_log_training_metrics(results["metrics"], config, run_name=run_name, tags=tags, num_final_episodes_for_evaluating_performance=10)
    except Exception as e:
        print(f"Error logging metrics to wandb: {e}")

    return results
