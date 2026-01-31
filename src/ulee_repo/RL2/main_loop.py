import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.training.train_state import TrainState
from jax.tree_util import Partial
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import Environment, EnvParams

from ulee_repo.evaluations.rollouts import create_benchmark_step_func, eval_rollout
from ulee_repo.RL2.config import TrainConfig
from ulee_repo.RL2.data_collection_and_updates import collect_data_and_update as rl2_collect_data_and_update
from ulee_repo.shared_code.validation_metric import compute_validation_metric


def full_training(
    rng: jax.Array,
    train_state: TrainState,
    env: Environment,
    env_params: EnvParams,
    benchmark: Benchmark,
    config: TrainConfig,
):
    best = (jnp.array(-jnp.inf, dtype=jnp.float32), train_state.params)
    meta_state = (rng, train_state, best)
    split_rng = jax.random.key(config.benchmark_split_seed)
    train_bench, test_bench = benchmark.shuffle(key=split_rng).split(prop=config.benchmark_train_percentage)
    meta_state, metrics = jax.lax.scan(
        Partial(train_with_new_envs, env=env, env_params=env_params, train_bench=train_bench, test_bench=test_bench, config=config), meta_state, None, config.num_batches_of_envs
    )

    return {"agent_state": meta_state[1], "best": meta_state[-1], "metrics": metrics}


def train_with_new_envs(meta_state, _unused, env: Environment, env_params: EnvParams, train_bench: Benchmark, test_bench: Benchmark, config: TrainConfig):
    rng, train_state, best = meta_state

    ########## -------------  SET UP  -------------- ###########
    rng, ruleset_rng_base, reset_rng_base = jax.random.split(rng, num=3)
    reset_rng = jax.random.split(reset_rng_base, num=config.num_envs_per_batch)

    if config.env_id.startswith("XLand-MiniGrid"):
        rulesets_rng = jax.random.split(ruleset_rng_base, num=config.num_envs_per_batch)
        rulesets = jax.vmap(train_bench.sample_ruleset)(rulesets_rng)
        env_params = env_params.replace(ruleset=rulesets)

    timestep = jax.vmap(env.reset, in_axes=(0, 0))(env_params, reset_rng)
    prev_action = jnp.zeros(config.num_envs_per_batch, dtype=jnp.int32)
    prev_reward = jnp.zeros(config.num_envs_per_batch)
    prev_done = timestep.last()
    memories = jnp.zeros((config.num_envs_per_batch, config.past_context_length, config.num_transformer_blocks, config.transformer_hidden_states_dim))
    memories_mask = jnp.zeros((config.num_envs_per_batch, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_)
    memories_mask_idx = jnp.zeros((config.num_envs_per_batch,), dtype=jnp.int32) + (config.past_context_length + 1)

    ########## -------------  TRAINING -------------- ###########

    runner_state = (rng, train_state, timestep, prev_action, prev_reward, prev_done, memories, memories_mask, memories_mask_idx, 0)
    runner_state, metrics = jax.lax.scan(Partial(rl2_collect_data_and_update, env=env, env_params=env_params, config=config), runner_state, None, config.num_updates_per_batch)
    metrics = jtu.tree_map(lambda x: x.mean(-1), metrics)
    rng, train_state = runner_state[:2]

    ########## -------------  EVALUATION -------------- ###########
    # setup
    rng, eval_ruleset_rng, eval_reset_rng, actions_rng = jax.random.split(rng, num=4)
    eval_reset_rng = jax.random.split(eval_reset_rng, num=config.eval_num_envs)
    actions_rng = jax.random.split(actions_rng, num=config.eval_num_envs)
    if config.env_id.startswith("XLand-MiniGrid"):
        eval_ruleset_rng = jax.random.split(eval_ruleset_rng, num=config.eval_num_envs)
        eval_ruleset = jax.vmap(test_bench.sample_ruleset)(eval_ruleset_rng)
        eval_env_params = env_params.replace(ruleset=eval_ruleset)
    else:
        eval_env_params = env_params
    step_func = create_benchmark_step_func(env)

    # collect evaluation data
    _, eval_stats, _ = jax.vmap(eval_rollout, in_axes=(0, 0, None, 0, None, None, None, None, None, None))(
        actions_rng,
        eval_reset_rng,
        env,
        eval_env_params,
        step_func,
        train_state,
        config.eval_num_episodes,
        "meta_learning",
        config,
        None,
    )
    metrics.update(
        {
            "eval/returns": eval_stats.returns,
            "eval/lengths": eval_stats.lengths,
            "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
        }
    )

    #########------------  SAVE BEST PARAMETERS ----------- #########
    best_metric, _ = best
    validation_metric = compute_validation_metric(metrics["eval/returns"])
    improve = validation_metric > best_metric
    new_potential_best = (validation_metric, train_state.params)
    best = jax.lax.cond(
        improve,
        lambda _: new_potential_best,
        lambda _: best,
        operand=None,
    )

    meta_state = (rng, train_state, best)
    return meta_state, metrics
