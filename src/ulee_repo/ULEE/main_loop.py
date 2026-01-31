import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.training.train_state import TrainState
from jax.tree_util import Partial
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import Environment, EnvParams

from ulee_repo.evaluations.rollouts import create_benchmark_step_func, create_unsupervised_step_func, eval_rollout
from ulee_repo.shared_code.trainsition_objects import State_Data
from ulee_repo.shared_code.validation_metric import compute_validation_metric
from ulee_repo.ULEE.config import TrainConfig
from ulee_repo.ULEE.data_collection_and_updates import collect_data_and_update as meta_learner_collect_data_and_update
from ulee_repo.ULEE.goal_judge import JudgeReplayBuffer, compute_goal_difficulties, train_goal_judge
from ulee_repo.ULEE.goal_sample import sample_unsupervised_goals
from ulee_repo.ULEE.goal_search import goal_search, train_goal_search
from ulee_repo.ULEE.utils import encode_goals_as_full_states, encode_goals_as_object_histograms

# ---------------------------------------------------------------------------------------------------------------------------------------


def full_training(
    rng: jax.Array,
    meta_learner_train_state: TrainState,
    judge_train_state: TrainState,
    goal_search_train_state: TrainState | tuple[TrainState, TrainState],
    judge_replay_buffer: JudgeReplayBuffer,
    env_no_goals: Environment,
    env_unsup_goals: Environment,
    env_real_goals: Environment,
    env_params: EnvParams,
    benchmark: Benchmark,
    config: TrainConfig,
):
    if config.goal_search_algorithm == "ppo":
        goal_search_params = goal_search_train_state.params
    elif config.goal_search_algorithm == "diayn":
        policy_train_state, discriminator_train_state = goal_search_train_state
        goal_search_params = (policy_train_state.params, discriminator_train_state.params)
    elif config.goal_search_algorithm == "random":
        goal_search_params = None
    best = (jnp.array(-jnp.inf, dtype=jnp.float32), meta_learner_train_state.params, goal_search_params)

    meta_state = (rng, meta_learner_train_state, judge_train_state, goal_search_train_state, judge_replay_buffer, best)
    split_rng = jax.random.key(config.benchmark_split_seed)
    train_bench, test_bench = benchmark.shuffle(key=split_rng).split(prop=config.benchmark_train_percentage)

    meta_state, metrics = jax.lax.scan(
        Partial(
            train_with_new_envs,
            env_no_goals=env_no_goals,
            env_unsup_goals=env_unsup_goals,
            env_real_goals=env_real_goals,
            env_params=env_params,
            train_benchmark=train_bench,
            test_benchmark=test_bench,
            config=config,
        ),
        meta_state,
        None,
        config.num_batches_of_envs,
    )

    return {"meta_learner_state": meta_state[1], "judge_train_state": meta_state[2], "goal_search_train_state": meta_state[3], "best": meta_state[-1], "metrics": metrics}


# ---------------------------------------------------------------------------------------------------------------------------------------


def train_with_new_envs(meta_state, _unused, env_no_goals, env_unsup_goals, env_real_goals, env_params: EnvParams, train_benchmark: Benchmark, test_benchmark: Benchmark, config: TrainConfig):
    rng, meta_learner_train_state, judge_train_state, goal_search_train_state, judge_replay_buffer, best = meta_state

    ####################### ----------------------  SET NEW ENVIRONMENTS   ---------------------------- ##############################
    rng, ruleset_rng_base, reset_rng_base = jax.random.split(rng, num=3)
    reset_rng = jax.random.split(reset_rng_base, num=config.num_envs_per_batch)

    if config.env_id.startswith("XLand-MiniGrid"):
        rulesets_rng = jax.random.split(ruleset_rng_base, num=config.num_envs_per_batch)
        rulesets = jax.vmap(train_benchmark.sample_ruleset)(rulesets_rng)
        env_params = env_params.replace(ruleset=rulesets)

    ####################### ---------------------- FIND NEW SELF-IMPOSED GOALS FOR EACH ENVIRONMENT  ---------------------------- ##############################

    # Reset all environments
    timestep = jax.vmap(env_no_goals.reset, in_axes=(0, 0))(env_params, reset_rng)
    initial_states = State_Data(grid_state=timestep.state.grid, agent_pos=timestep.state.agent.position)  # Save the data from initial states to estimate the difficulty of the goals.

    # Obtain new candidate goals (states) for each environment in the batch. potential_goals leafs have shape (S, B, ...)
    base_runner_state = (rng, goal_search_train_state, timestep)
    rng, potential_goals = goal_search(goal_search_algorithm=config.goal_search_algorithm, base_runner_state=base_runner_state, env=env_no_goals, env_params=env_params, config=config.goal_search)

    # subsample goals: keep every nth potential goal for each environment
    potential_goals = jtu.tree_map(lambda arr: arr[:: config.goal_search.subsample_step, ...], potential_goals)  # (S//subsample_step, B ...) = (S', B...)

    # Estimate the difficulty of each candidate goal. jnp.array of shape (S', B)
    difficulties = compute_goal_difficulties(judge_train_state, initial_states, potential_goals, config.goal_search.num_chunks_for_computing_difficulties_in_goal_selection)

    # Get the goal state to use in each environment. pytree with leafs of shape (B, ...)
    rng, unsupervised_goals = sample_unsupervised_goals(rng, config.goal_sampling_method, potential_goals, difficulties, config)

    # transform into encoded representation. shape (B, vector goal representation dim)
    if config.goal_mode == "full_state":
        encoded_unsupervised_goals = encode_goals_as_full_states(unsupervised_goals)
    elif config.goal_mode == "objects_histogram":
        encoded_unsupervised_goals = encode_goals_as_object_histograms(unsupervised_goals)

    ####################### ---------------------- META LEARNER TRAINING   ---------------------------- ##############################

    ## --- Initial setup -------------##

    # Reset all environments
    timestep = jax.vmap(env_unsup_goals.reset, in_axes=(0, 0))(env_params, reset_rng)
    # Initialize previous actions and rewards
    prev_action = jnp.zeros(config.num_envs_per_batch, dtype=jnp.int32)
    prev_reward = jnp.zeros(config.num_envs_per_batch)
    prev_done = timestep.last()
    # Initialize the memories and memories mask
    memories = jnp.zeros((config.num_envs_per_batch, config.past_context_length, config.num_transformer_blocks, config.transformer_hidden_states_dim))
    memories_mask = jnp.zeros((config.num_envs_per_batch, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_)
    memories_mask_idx = jnp.zeros((config.num_envs_per_batch,), dtype=jnp.int32) + (config.past_context_length + 1)
    # Initialize buffer to keep track of wether the last K episodes in each env were successes or failures.
    episode_success_buffer = jnp.zeros((config.num_envs_per_batch, config.goal_judge.num_episodes_to_compute_success_rate))
    episode_success_buffer_pointer = jnp.zeros((config.num_envs_per_batch,), dtype=jnp.int32)

    ## ----- Training ------------##

    runner_state = (rng, meta_learner_train_state, timestep, prev_action, prev_reward, prev_done, episode_success_buffer, episode_success_buffer_pointer, memories, memories_mask, memories_mask_idx, 0)

    runner_state, metrics = jax.lax.scan(
        Partial(meta_learner_collect_data_and_update, goals=encoded_unsupervised_goals, env=env_unsup_goals, env_params=env_params, config=config), runner_state, None, config.num_updates_per_batch
    )

    metrics = jtu.tree_map(lambda x: x.mean(-1), metrics)

    # obtain the success rate of the agent for each env/goal - computed over its last K episodes
    success_buffers = runner_state[6]  # (num_envs, K)
    goal_success_rates = jnp.mean(success_buffers, axis=1)  # (num_envs,)

    rng, meta_learner_train_state = runner_state[:2]

    ####################### ---------------------- DIFFICULTY PREDICTOR TRAINING   ---------------------------- ##############################

    target_difficulties = 1 - goal_success_rates
    judge_replay_buffer = judge_replay_buffer.add_batch(initial_states, unsupervised_goals, target_difficulties)
    rng, judge_train_state, judge_loss = train_goal_judge(rng, train_state=judge_train_state, replay_buffer=judge_replay_buffer, config=config.goal_judge)
    predicted_difficulties = judge_train_state.apply_fn(judge_train_state.params, initial_states, unsupervised_goals)  # just for logging purposes
    metrics.update(
        {
            "judge_loss": judge_loss,
            "training_goals_difficulties": target_difficulties,
            "predicted_difficulties": predicted_difficulties,
        }
    )

    ####################### ---------------------- GOAL SEARCH AGENT TRAINING   ---------------------------- ##############################

    rng, goal_search_train_state, metrics = train_goal_search(
        rng, config.goal_search_algorithm, metrics, goal_search_train_state, judge_train_state, env_no_goals, env_params, reset_rng, config=config.goal_search
    )

    #######################################################################################################################################
    # EVALUATIONS
    #######################################################################################################################################

    ######################  -------- META LEARNER EVALUATION ON NEW SELF-IMPOSED GOALS --------- ##############
    # setup
    rng, eval_ruleset_rng, eval_reset_rng, actions_rng = jax.random.split(rng, num=4)
    eval_reset_rng = jax.random.split(eval_reset_rng, num=config.eval_num_envs)
    actions_rng = jax.random.split(actions_rng, num=config.eval_num_envs)
    if config.env_id.startswith("XLand-MiniGrid"):
        eval_ruleset_rng = jax.random.split(eval_ruleset_rng, num=config.eval_num_envs)
        eval_ruleset = jax.vmap(test_benchmark.sample_ruleset)(eval_ruleset_rng)
        eval_env_params = env_params.replace(ruleset=eval_ruleset)
    else:
        eval_env_params = env_params
    # collect new unsupervised goals
    timestep = jax.vmap(env_unsup_goals.reset, in_axes=(0, 0))(eval_env_params, eval_reset_rng)
    base_runner_state = (rng, goal_search_train_state, timestep)
    rng, potential_goals = goal_search(goal_search_algorithm="random", base_runner_state=base_runner_state, env=env_no_goals, env_params=eval_env_params, config=config.goal_search)
    difficulties = jnp.zeros((potential_goals.agent_pos.shape[0], potential_goals.agent_pos.shape[1]))  # No need to compute difficulties because we utilize uniform sampling.
    rng, unsupervised_goals = sample_unsupervised_goals(rng, "uniform", potential_goals, difficulties, config)
    if config.goal_mode == "full_state":
        encoded_unsupervised_goals = encode_goals_as_full_states(unsupervised_goals)
    elif config.goal_mode == "objects_histogram":
        encoded_unsupervised_goals = encode_goals_as_object_histograms(unsupervised_goals)
    step_func = create_unsupervised_step_func(env_unsup_goals)
    # collect eval data
    _, eval_stats, _ = jax.vmap(eval_rollout, in_axes=(0, 0, None, 0, None, None, None, None, None, 0))(
        actions_rng,
        eval_reset_rng,
        env_unsup_goals,
        eval_env_params,
        step_func,
        meta_learner_train_state,
        config.eval_num_episodes,
        "meta_learning",
        config,
        encoded_unsupervised_goals,
    )
    metrics.update(
        {
            "eval/unsupervised/returns": eval_stats.returns,  # (eval_num_envs, eval_num_episodes)
            "eval/unsupervised/lengths": eval_stats.lengths,  # (eval_num_envs, eval_num_episodes)
            "lr": meta_learner_train_state.opt_state[-1].hyperparams["learning_rate"],
        }
    )

    ##########################  -------- META LEARNER EVALUATION ON BENCHMARK GOALS --------- ########################
    # setup
    rng, eval_ruleset_rng, eval_reset_rng, actions_rng = jax.random.split(rng, num=4)
    eval_reset_rng = jax.random.split(eval_reset_rng, num=config.eval_num_envs)
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
    _, eval_stats, _ = jax.vmap(eval_rollout, in_axes=(0, 0, None, 0, None, None, None, None, None, 0))(
        actions_rng,
        eval_reset_rng,
        env_real_goals,
        eval_env_params,
        step_func,
        meta_learner_train_state,
        config.eval_num_episodes,
        "meta_learning",
        config,
        goals,
    )
    metrics.update(
        {
            "eval/benchmark/returns": eval_stats.returns,
            "eval/benchmark/lengths": eval_stats.lengths,
        }
    )

    #########------------  SAVE BEST PARAMETERS ----------- #########
    best_metric, _, _ = best
    validation_metric = compute_validation_metric(metrics["eval/benchmark/returns"])
    improve = validation_metric > best_metric
    if config.goal_search_algorithm == "ppo":
        new_goal_search_params = goal_search_train_state.params
    elif config.goal_search_algorithm == "diayn":
        policy_train_state, discriminator_train_state = goal_search_train_state
        new_goal_search_params = (policy_train_state.params, discriminator_train_state.params)
    elif config.goal_search_algorithm == "random":
        new_goal_search_params = None
    new_potential_best = (validation_metric, meta_learner_train_state.params, new_goal_search_params)
    best = jax.lax.cond(
        improve,
        lambda _: new_potential_best,
        lambda _: best,
        operand=None,
    )

    #####  ------------------------------------  ########

    meta_state = (rng, meta_learner_train_state, judge_train_state, goal_search_train_state, judge_replay_buffer, best)
    return meta_state, metrics
