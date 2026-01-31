import jax
import jax.numpy as jnp
import optax
import xminigrid
from flax.training.train_state import TrainState
from xminigrid.environment import EnvParams

from ulee_repo.DIAYN.setups import setup_diayn_agent_train_state, setup_diayn_discriminator_train_state
from ulee_repo.networks.goal_difficulty_predictor import Difficulty_Judge
from ulee_repo.PPO.setups import setup_ppo_train_state
from ulee_repo.RL2.setups import setup_meta_learner_train_state
from ulee_repo.shared_code.trainsition_objects import State_Data
from ulee_repo.shared_code.wrappers import CustomAutoResetWrapper_NoGoals, CustomAutoResetWrapper_UnsupervisedGoals, CustomGymAutoResetWrapper
from ulee_repo.ULEE.config import GoalJudgeConfig, TrainConfig
from ulee_repo.ULEE.goal_judge import JudgeReplayBuffer

# ------------------------------------------------------------


def setup_judge_network_train_state(rng: jax.Array, env_params: EnvParams, config: GoalJudgeConfig):
    judge_network = Difficulty_Judge(
        grid_state_emb_dim=config.grid_state_emb_dim,
        head_activation=config.head_activation,
        mlp_dim=config.mlp_dim,
    )
    init_input1 = State_Data(grid_state=jnp.zeros((2, env_params.height, env_params.width, 2), dtype=jnp.int32), agent_pos=jnp.zeros((2, 2), dtype=jnp.int32))
    init_input2 = State_Data(grid_state=jnp.zeros((2, env_params.height, env_params.width, 2), dtype=jnp.int32), agent_pos=jnp.zeros((2, 2), dtype=jnp.int32))
    rng, _rng = jax.random.split(rng)
    judge_network_params = judge_network.init(_rng, init_input1, init_input2)
    judge_network_optimizer = optax.chain(
        optax.inject_hyperparams(optax.adam)(learning_rate=config.lr, eps=config.adam_eps),
    )
    judge_network_train_state = TrainState.create(apply_fn=judge_network.apply, params=judge_network_params, tx=judge_network_optimizer)
    return rng, judge_network_train_state


# ------------------------------------------------------------


def set_up_for_training(config: TrainConfig):
    """sets up things for training given a config. It returns:
    - rng: a random number generator key
    - benchmark: the chosen benchmark from which to sample tasks
    - env_params: the parameters of the chosen environment
    - env_no_goals: the chosen environment wrapped so that there are no rewards and no episode termination
    - env_unsup_goals: the chosen environment wrapped so that rewards and termination are subject to given self-generated goals
    - env_real_goals: the chosen environment with the original goals and rewards.
      Additionally, all environments are wrapped so that they automatically reset after each episode and so that all resets start from the same initial state.
    - meta_learner_train_state: the train state of the meta-learner network
    - judge_network_train_state: the train state of the goal difficulty predictor network
    - goal_search_train_state: the train state of the goal-search network
    - judge_replay_buffer: buffer for storing data to train the difficulty predictor
    """

    # setup environment & benchmark
    if config.episode_max_steps:
        env, env_params = xminigrid.make(config.env_id, max_steps=config.episode_max_steps)
    else:
        env, env_params = xminigrid.make(config.env_id)
    env_real_goals = CustomGymAutoResetWrapper(env)
    env_unsup_goals = CustomAutoResetWrapper_UnsupervisedGoals(env, goal_mode=config.goal_mode)
    env_no_goals = CustomAutoResetWrapper_NoGoals(env)

    benchmark = xminigrid.load_benchmark(config.benchmark_id)

    rng = jax.random.key(config.train_seed)

    ##### ---- SETUP META LEARNER (PRE-TRAINED POLICY) ---- #####

    if config.anneal_lr:

        def meta_learner_linear_schedule(count):
            total_inner_updates = config.num_minibatches * config.update_epochs * config.num_updates_per_batch
            frac = 1.0 - (count // total_inner_updates) / config.num_batches_of_envs
            return config.lr * frac

        rng, meta_learner_train_state = setup_meta_learner_train_state(rng, env, env_params, config, lr_schedule=meta_learner_linear_schedule)
    else:
        rng, meta_learner_train_state = setup_meta_learner_train_state(rng, env, env_params, config)

    ##### ---- SETUP DIFFICULTY PREDICTOR NETWORK  ---- #####

    rng, judge_network_train_state = setup_judge_network_train_state(rng, env_params, config.goal_judge)

    dummy_state = State_Data(grid_state=jnp.zeros((env_params.height, env_params.width, 2), dtype=jnp.uint8), agent_pos=jnp.zeros((2,), dtype=jnp.int32))

    judge_replay_buffer = JudgeReplayBuffer.create(capacity=config.goal_judge.replay_buffer_num_batches * config.num_envs_per_batch, batch_size=config.num_envs_per_batch, dummy_state=dummy_state)

    ##### ---- SETUP GOAL SEARCH NETWORK ---- #####
    if config.goal_search_algorithm == "random":
        goal_search_train_state = ()
    elif config.goal_search_algorithm == "ppo":
        rng, goal_search_train_state = setup_ppo_train_state(rng, env, env_params, config=config.goal_search)
    elif config.goal_search_algorithm == "diayn":
        rng, diayn_agent_train_state = setup_diayn_agent_train_state(rng, env, env_params, config.goal_search)
        rng, diayn_discriminator_train_state = setup_diayn_discriminator_train_state(rng, env_params, config.goal_search)
        goal_search_train_state = (diayn_agent_train_state, diayn_discriminator_train_state)
    else:
        msg = f"Unknown goal search algorithm: {config.goal_search_algorithm}"
        raise ValueError(msg)

    ####

    return rng, env_no_goals, env_unsup_goals, env_real_goals, env_params, benchmark, meta_learner_train_state, judge_network_train_state, goal_search_train_state, judge_replay_buffer
