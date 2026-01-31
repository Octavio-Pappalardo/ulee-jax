import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.training.train_state import TrainState
from jax.tree_util import Partial
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import Environment, EnvParams

from ulee_repo.evaluations.rollouts import create_benchmark_step_func, eval_rollout
from ulee_repo.RND.config import TrainConfig
from ulee_repo.RND.data_collection_and_updates import collect_data_and_update
from ulee_repo.RND.normalization_utils import NormalizationStats


def full_training_on_fixed_envs(
    rng: jax.Array,
    agent_train_state: TrainState,
    predictor_train_state: TrainState,
    target_train_state: TrainState,
    env: Environment,
    env_params: EnvParams,
    benchmark: Benchmark,
    config: TrainConfig,
):
    ####   SETUP   ####
    rng, ruleset_rng_base, reset_rng_base = jax.random.split(rng, num=3)
    reset_rng = jax.random.split(reset_rng_base, num=config.num_envs_per_batch)
    if config.env_id.startswith("XLand-MiniGrid"):
        rulesets_rng = jax.random.split(ruleset_rng_base, num=config.num_envs_per_batch)
        rulesets = jax.vmap(benchmark.sample_ruleset)(rulesets_rng)
        env_params = env_params.replace(ruleset=rulesets)

    timestep = jax.vmap(env.reset, in_axes=(0, 0))(env_params, reset_rng)
    memories = jnp.zeros((config.num_envs_per_batch, config.past_context_length, config.num_transformer_blocks, config.transformer_hidden_states_dim))
    memories_mask = jnp.zeros((config.num_envs_per_batch, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_)
    memories_mask_idx = jnp.zeros((config.num_envs_per_batch,), dtype=jnp.int32) + (config.past_context_length + 1)

    normalization_stats = NormalizationStats(running_forward_return=jnp.zeros(config.num_envs_per_batch))

    runner_state = (rng, agent_train_state, predictor_train_state, target_train_state, normalization_stats, timestep, memories, memories_mask, memories_mask_idx, 0)

    def train_on_fixed_envs_one_iteration(
        runner_state,
        _unused,
        env: Environment,
        reset_rng,
        env_params: EnvParams,
        config: TrainConfig,
    ):
        # Train
        runner_state, metrics = jax.lax.scan(Partial(collect_data_and_update, env=env, env_params=env_params, config=config), runner_state, None, config.num_updates_per_batch)

        metrics = jtu.tree_map(lambda x: x.mean(-1), metrics)
        rng, agent_train_state = runner_state[:2]

        # Evaluation
        rng, actions_rng = jax.random.split(rng, num=2)
        actions_rng = jax.random.split(actions_rng, num=config.num_envs_per_batch)
        step_func = create_benchmark_step_func(env)
        _, eval_stats, _ = jax.vmap(eval_rollout, in_axes=(0, 0, None, 0, None, None, None, None, None, None))(
            actions_rng,
            reset_rng,
            env,
            env_params,
            step_func,
            agent_train_state,
            config.eval_num_episodes,
            "rnd",
            config,
            None,
        )
        # adding evaluation metrics
        metrics.update(
            {
                "eval/returns": eval_stats.returns,
                "eval/lengths": eval_stats.lengths,
                "lr": agent_train_state.opt_state[-1].hyperparams["learning_rate"],
            }
        )
        runner_state = (rng, *runner_state[1:])
        return runner_state, metrics

    final_runner_state, metrics = jax.lax.scan(
        Partial(train_on_fixed_envs_one_iteration, env=env, reset_rng=reset_rng, env_params=env_params, config=config), runner_state, None, config.num_batches_of_envs
    )

    return {"agent_state": final_runner_state[1], "predictor_state": final_runner_state[2], "metrics": metrics}
