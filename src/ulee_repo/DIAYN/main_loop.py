import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.training.train_state import TrainState
from jax.tree_util import Partial
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import Environment, EnvParams

from ulee_repo.DIAYN.config import TrainConfig
from ulee_repo.DIAYN.data_collection_and_updates import collect_data_and_update
from ulee_repo.evaluations.diayn_evals import eval_diayn_selecting_best_skill
from ulee_repo.evaluations.rollouts import create_benchmark_step_func
from ulee_repo.shared_code.validation_metric import compute_validation_metric


def full_training(
    rng: jax.Array,
    agent_train_state: TrainState,
    discriminator_train_state: TrainState,
    env_no_goals: Environment,
    env_real_goals: Environment,
    env_params: EnvParams,
    benchmark: Benchmark,
    config: TrainConfig,
):
    best = (jnp.array(-jnp.inf, dtype=jnp.float32), agent_train_state.params)
    meta_state = (rng, agent_train_state, discriminator_train_state, best)
    split_rng = jax.random.key(config.benchmark_split_seed)
    train_bench, test_bench = benchmark.shuffle(key=split_rng).split(prop=config.benchmark_train_percentage)

    meta_state, metrics = jax.lax.scan(
        Partial(train_with_new_envs, env_no_goals=env_no_goals, env_real_goals=env_real_goals, env_params=env_params, train_benchmark=train_bench, test_benchmark=test_bench, config=config),
        meta_state,
        None,
        config.num_batches_of_envs,
    )

    return {"agent_state": meta_state[1], "best": meta_state[-1], "metrics": metrics}


# ---------------------------------------------------------------------------------------------------------------------------------------


def train_with_new_envs(meta_state, _unused, env_no_goals: Environment, env_real_goals: Environment, env_params: EnvParams, train_benchmark: Benchmark, test_benchmark: Benchmark, config: TrainConfig):
    rng, train_state, discriminator_train_state, best = meta_state

    ####################### ----------------------   SETUP   ---------------------- #######################

    rng, ruleset_rng_base, reset_rng_base = jax.random.split(rng, num=3)
    reset_rng = jax.random.split(reset_rng_base, num=config.num_envs_per_batch)

    if config.env_id.startswith("XLand-MiniGrid"):
        rulesets_rng = jax.random.split(ruleset_rng_base, num=config.num_envs_per_batch)
        rulesets = jax.vmap(train_benchmark.sample_ruleset)(rulesets_rng)
        env_params = env_params.replace(ruleset=rulesets)

    timestep = jax.vmap(env_no_goals.reset, in_axes=(0, 0))(env_params, reset_rng)

    memories = jnp.zeros((config.num_envs_per_batch, config.past_context_length, config.num_transformer_blocks, config.transformer_hidden_states_dim))
    memories_mask = jnp.zeros((config.num_envs_per_batch, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_)
    memories_mask_idx = jnp.zeros((config.num_envs_per_batch,), dtype=jnp.int32) + (config.past_context_length + 1)

    rng, skill_rng = jax.random.split(rng)
    skill = jax.random.randint(skill_rng, shape=(config.num_envs_per_batch,), minval=0, maxval=config.num_skills)

    ####################### ----------------------  POLICY AND DISCRIMINATOR TRAINING  ---------------------- #######################

    runner_state = (rng, train_state, discriminator_train_state, timestep, skill, memories, memories_mask, memories_mask_idx, 0)

    runner_state, metrics = jax.lax.scan(Partial(collect_data_and_update, env=env_no_goals, env_params=env_params, config=config), runner_state, None, config.num_updates_per_batch)

    metrics = jtu.tree_map(lambda x: x.mean(-1), metrics)

    rng, train_state, discriminator_train_state = runner_state[:3]

    ####################### ----------------------  EVALUATION  ---------------------- #######################
    # setup
    rng, eval_ruleset_rng, eval_reset_rng, actions_rng = jax.random.split(rng, num=4)
    actions_rng = jax.random.split(actions_rng, num=config.eval_num_envs)
    if config.env_id.startswith("XLand-MiniGrid"):
        eval_ruleset_rng = jax.random.split(eval_ruleset_rng, num=config.eval_num_envs)
        eval_ruleset = jax.vmap(test_benchmark.sample_ruleset)(eval_ruleset_rng)
        eval_env_params = env_params.replace(ruleset=eval_ruleset)
    else:
        eval_env_params = env_params
    step_func = create_benchmark_step_func(env_real_goals)
    goals = jnp.zeros((config.eval_num_envs, 10), dtype=jnp.int32)
    # collect evaluation data
    rng, stats_for_each_skill, best_skill_stats = eval_diayn_selecting_best_skill(
        rng=rng,
        eval_reset_rng=eval_reset_rng,
        env=env_real_goals,
        eval_env_params=eval_env_params,
        step_func=step_func,
        train_state=train_state,
        config=config,
        goals=goals,
    )

    # [num_skills, num_envs, num_eval_episodes_per_skill] -> [num_envs, num_skills, num_eval_episodes_per_skill] -> [num_envs, num_skills * num_eval_episodes_per_skill]
    eval_all_skills_length_data = jnp.swapaxes(stats_for_each_skill.lengths, 0, 1)
    eval_all_skills_returns_data = jnp.swapaxes(stats_for_each_skill.returns, 0, 1)
    eval_all_skills_length_data = jnp.reshape(eval_all_skills_length_data, (eval_all_skills_length_data.shape[0], -1))
    eval_all_skills_returns_data = jnp.reshape(eval_all_skills_returns_data, (eval_all_skills_returns_data.shape[0], -1))

    eval_length_data = jnp.concatenate([eval_all_skills_length_data, best_skill_stats.lengths], axis=1)  # [num_envs, total_eval_episodes_per_env]
    eval_returns_data = jnp.concatenate([eval_all_skills_returns_data, best_skill_stats.returns], axis=1)  # [num_envs, total_eval_episodes_per_env]

    metrics.update(
        {
            "eval/returns": eval_returns_data,
            "eval/lengths": eval_length_data,
            "eval/all_skills/returns": stats_for_each_skill.returns,
            "eval/all_skills/lengths": stats_for_each_skill.lengths,
            "eval/best_skill/returns": best_skill_stats.returns,
            "eval/best_skill/lengths": best_skill_stats.lengths,
            "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
        }
    )

    #########------------  SAVE BEST PARAMETERS ----------- #########
    best_metric, _ = best
    validation_metric = compute_validation_metric(best_skill_stats.returns)
    improve = validation_metric > best_metric
    new_potential_best = (validation_metric, train_state.params)
    best = jax.lax.cond(
        improve,
        lambda _: new_potential_best,
        lambda _: best,
        operand=None,
    )

    #####  ------------------------------------  ########

    meta_state = (rng, train_state, discriminator_train_state, best)
    return meta_state, metrics
