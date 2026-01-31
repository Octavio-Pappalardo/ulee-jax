import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.tree_util import Partial

from ulee_repo.DIAYN.data_collection_and_updates import collect_data as diayn_collect_data
from ulee_repo.DIAYN.data_collection_and_updates import compute_diayn_intrinsic_rewards, update_discriminator
from ulee_repo.DIAYN.data_collection_and_updates import update_agent as diayn_update_agent
from ulee_repo.PPO.data_collection_and_updates import collect_data as ppo_collect_data
from ulee_repo.PPO.data_collection_and_updates import update_agent as ppo_update_agent
from ulee_repo.shared_code.trainsition_objects import State_Data
from ulee_repo.ULEE.config import GoalSearchConfigBase, GoalSearchConfigDIAYN, GoalSearchConfigPPO
from ulee_repo.ULEE.goal_judge import compute_goal_difficulties


def goal_search(goal_search_algorithm, base_runner_state, env, env_params, config: GoalSearchConfigBase):
    if goal_search_algorithm == "random":
        rng, _, initial_timestep = base_runner_state
        runner_state = (rng, initial_timestep)
        runner_state, potential_goals = collect_data_random(runner_state, config.goal_searching_steps_per_env, env, env_params)
        rng = runner_state[0]

    elif goal_search_algorithm == "ppo":
        config: GoalSearchConfigPPO = config
        rng, policy_train_state, initial_timestep = base_runner_state
        memories = jnp.zeros((config.num_envs_per_batch, config.past_context_length, config.num_transformer_blocks, config.transformer_hidden_states_dim))
        memories_mask = jnp.zeros((config.num_envs_per_batch, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_)
        memories_mask_idx = jnp.zeros((config.num_envs_per_batch,), dtype=jnp.int32) + (config.past_context_length + 1)
        runner_state = (rng, policy_train_state, initial_timestep, memories, memories_mask, memories_mask_idx, 0)
        # collect data
        runner_state, transitions, _ = ppo_collect_data(runner_state, config.goal_searching_steps_per_env, env, env_params, config)
        rng = runner_state[0]
        potential_goals = transitions.state_data

    elif goal_search_algorithm == "diayn":
        config: GoalSearchConfigDIAYN = config
        rng, (policy_train_state, discriminator_train_state), initial_timestep = base_runner_state
        memories = jnp.zeros((config.num_envs_per_batch, config.past_context_length, config.num_transformer_blocks, config.transformer_hidden_states_dim))
        memories_mask = jnp.zeros((config.num_envs_per_batch, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_)
        memories_mask_idx = jnp.zeros((config.num_envs_per_batch,), dtype=jnp.int32) + (config.past_context_length + 1)
        rng, skill_rng = jax.random.split(rng)
        skill = jax.random.randint(skill_rng, shape=(config.num_envs_per_batch,), minval=0, maxval=config.num_skills)
        runner_state = (rng, policy_train_state, discriminator_train_state, initial_timestep, skill, memories, memories_mask, memories_mask_idx, 0)
        # collect data
        runner_state, transitions, _ = diayn_collect_data(runner_state, config.goal_searching_steps_per_env, env, env_params, config)
        rng = runner_state[0]
        potential_goals = transitions.state_data

    return rng, potential_goals


##############---------------------------------------


def step_envs_random(runner_state, _unused, env, env_params):
    rng, prev_timestep = runner_state

    rng, _rng = jax.random.split(rng)
    action = jax.random.randint(_rng, shape=(env_params.ruleset.init_tiles.shape[0]), minval=0, maxval=env.num_actions(env_params))

    timestep = jax.vmap(env.step, in_axes=(0, 0, 0))(env_params, prev_timestep, action)

    potential_goal_data = State_Data(
        grid_state=timestep.state.grid,
        agent_pos=timestep.state.agent.position,
    )
    runner_state = (rng, timestep)

    return runner_state, potential_goal_data


def collect_data_random(runner_state, num_steps, env, env_params):
    runner_state, potential_goal_data = jax.lax.scan(Partial(step_envs_random, env=env, env_params=env_params), runner_state, None, num_steps)

    return runner_state, potential_goal_data


##############################################################


##############################################################


def train_goal_search(rng, goal_search_algorithm, metrics, goal_search_train_state, judge_train_state, env, env_params, env_reset_rng, config: GoalSearchConfigBase):
    if goal_search_algorithm == "random":
        pass

    elif goal_search_algorithm == "ppo":
        config: GoalSearchConfigPPO = config
        # setup
        timestep = jax.vmap(env.reset, in_axes=(0, 0))(env_params, env_reset_rng)
        initial_states = State_Data(grid_state=timestep.state.grid, agent_pos=timestep.state.agent.position)
        memories = jnp.zeros((config.num_envs_per_batch, config.past_context_length, config.num_transformer_blocks, config.transformer_hidden_states_dim))
        memories_mask = jnp.zeros((config.num_envs_per_batch, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_)
        memories_mask_idx = jnp.zeros((config.num_envs_per_batch,), dtype=jnp.int32) + (config.past_context_length + 1)
        # train
        runner_state = (rng, goal_search_train_state, timestep, memories, memories_mask, memories_mask_idx, 0)
        runner_state, goal_search_metrics = jax.lax.scan(
            Partial(ppo_collect_data_and_update_with_difficulty_seeking_rewards, judge_train_state=judge_train_state, initial_states=initial_states, env=env, env_params=env_params, config=config),
            runner_state,
            None,
            config.num_updates_per_batch,
        )
        rng, goal_search_train_state = runner_state[:2]

        goal_search_metrics = jtu.tree_map(lambda x: x.mean(-1), goal_search_metrics)
        metrics.update(
            {
                "goal_search/total_loss": goal_search_metrics["total_loss"],
                "goal_search/value_loss": goal_search_metrics["value_loss"],
                "goal_search/actor_loss": goal_search_metrics["actor_loss"],
                "goal_search/entropy": goal_search_metrics["entropy"],
                "goal_search/kl": goal_search_metrics["kl"],
            }
        )

    elif goal_search_algorithm == "diayn":
        config: GoalSearchConfigDIAYN = config
        # setup
        timestep = jax.vmap(env.reset, in_axes=(0, 0))(env_params, env_reset_rng)
        initial_states = State_Data(grid_state=timestep.state.grid, agent_pos=timestep.state.agent.position)
        memories = jnp.zeros((config.num_envs_per_batch, config.past_context_length, config.num_transformer_blocks, config.transformer_hidden_states_dim))
        memories_mask = jnp.zeros((config.num_envs_per_batch, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_)
        memories_mask_idx = jnp.zeros((config.num_envs_per_batch,), dtype=jnp.int32) + (config.past_context_length + 1)
        rng, skill_rng = jax.random.split(rng)
        skill = jax.random.randint(skill_rng, shape=(config.num_envs_per_batch,), minval=0, maxval=config.num_skills)

        # train
        policy_train_state, discriminator_train_state = goal_search_train_state
        runner_state = (rng, policy_train_state, discriminator_train_state, timestep, skill, memories, memories_mask, memories_mask_idx, 0)
        runner_state, goal_search_metrics = jax.lax.scan(
            Partial(diayn_collect_data_and_update_with_difficulty_seeking_rewards, judge_train_state=judge_train_state, initial_states=initial_states, env=env, env_params=env_params, config=config),
            runner_state,
            None,
            config.num_updates_per_batch,
        )
        rng, policy_train_state, discriminator_train_state = runner_state[:3]
        goal_search_train_state = (policy_train_state, discriminator_train_state)

        goal_search_metrics = jtu.tree_map(lambda x: x.mean(-1), goal_search_metrics)
        metrics.update(
            {
                "goal_search/total_loss": goal_search_metrics["total_loss"],
                "goal_search/value_loss": goal_search_metrics["value_loss"],
                "goal_search/actor_loss": goal_search_metrics["actor_loss"],
                "goal_search/entropy": goal_search_metrics["entropy"],
                "goal_search/kl": goal_search_metrics["kl"],
                "goal_search/discriminator_loss": goal_search_metrics["discriminator_loss"],
                "goal_search/skills_log_prob": goal_search_metrics["skills_log_prob"],
            }
        )

    return rng, goal_search_train_state, metrics


def ppo_collect_data_and_update_with_difficulty_seeking_rewards(runner_state, _unused, judge_train_state, initial_states, env, env_params, config: GoalSearchConfigPPO):
    memories_previous = runner_state[3]

    runner_state, transitions, memories_batch = ppo_collect_data(runner_state, config.num_steps_per_update, env, env_params, config)

    # Compute intrinsic difficulty seeking rewards
    difficulty_seeking_rewards = compute_goal_difficulties(judge_train_state, initial_states, goals=transitions.state_data, num_chunks=config.num_chunks_for_computing_intrinsic_rewards)
    difficulty_seeking_rewards = difficulty_seeking_rewards / env_params.max_steps
    transitions = transitions.replace(reward=difficulty_seeking_rewards)  # (S, B)

    # Concatenate previous memory with new batch
    memories_batch = jnp.concatenate([jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0)

    # Update agent
    runner_state, metrics = ppo_update_agent(runner_state, transitions, memories_batch, config)

    return runner_state, metrics


def diayn_collect_data_and_update_with_difficulty_seeking_rewards(runner_state, _unused, judge_train_state, initial_states, env, env_params, config: GoalSearchConfigDIAYN):
    memories_previous = runner_state[5]

    runner_state, transitions, memories_batch = diayn_collect_data(runner_state, config.num_steps_per_update, env, env_params, config)

    # Compute intrinsic dyain rewards
    diayn_rewards, skills_log_prob = compute_diayn_intrinsic_rewards(
        discriminator_train_state=runner_state[2], transitions=transitions, num_chunks=config.num_chunks_in_diayn_rewards_computation, num_skills=config.num_skills
    )
    diayn_rewards = diayn_rewards / env_params.max_steps

    # Compute intrinsic difficulty seeking rewards
    difficulty_seeking_rewards = compute_goal_difficulties(judge_train_state, initial_states, goals=transitions.state_data, num_chunks=config.num_chunks_for_computing_intrinsic_rewards)
    difficulty_seeking_rewards = difficulty_seeking_rewards / env_params.max_steps

    intrinsic_rewards = difficulty_seeking_rewards + config.diayn_coef * diayn_rewards
    transitions = transitions.replace(reward=intrinsic_rewards)

    # Concatenate previous memory with new batch
    memories_batch = jnp.concatenate([jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0)

    # Update agent
    runner_state, metrics = diayn_update_agent(runner_state, transitions, memories_batch, config)

    # Update skill discriminator
    runner_state, discriminator_loss_value = update_discriminator(runner_state, transitions, config)

    metrics.update(
        {
            "discriminator_loss": discriminator_loss_value,
            "skills_log_prob": skills_log_prob,
        }
    )

    return runner_state, metrics
