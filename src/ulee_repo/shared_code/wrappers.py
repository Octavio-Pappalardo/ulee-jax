import jax
import jax.numpy as jnp
from xminigrid.core.actions import take_action
from xminigrid.core.observation import transparent_field_of_view
from xminigrid.core.rules import check_rule
from xminigrid.environment import EnvParamsT
from xminigrid.types import IntOrArray, StepType, TimeStep
from xminigrid.wrappers import Wrapper

from ulee_repo.ULEE.utils import encode_single_goal_as_full_state, encode_single_goal_as_object_histogram


# Wrapper based on xland-minigrid original Gym-style wrapper; difference is we do not change initial agent and objects positions at every reset
class CustomGymAutoResetWrapper(Wrapper):
    def __auto_reset(self, params, timestep):
        key = timestep.state.key
        # key, _ = jax.random.split(timestep.state.key)
        reset_timestep = self.reset(params, key)

        timestep = timestep.replace(
            state=reset_timestep.state,
            observation=reset_timestep.observation,
        )
        return timestep

    def reset(self, params: EnvParamsT, key: jax.Array):
        state = self._generate_problem(params, key)
        state = state.replace(key=key)  ## New line so that state.key holds the reset_key
        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=transparent_field_of_view(state.grid, state.agent, params.view_size, params.view_size),
        )
        return timestep

    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = jax.lax.cond(
            timestep.last(),
            lambda: self.__auto_reset(params, timestep),
            lambda: timestep,
        )
        return timestep


# -------------------------------------------------------------------------------------------------


# Wrapper thar removes goal checking. Episodes are terminated only after a fixed number of steps


class CustomAutoResetWrapper_NoGoals(Wrapper):
    def __auto_reset(self, params, timestep):
        key = timestep.state.key
        reset_timestep = self.reset(params, key)

        timestep = timestep.replace(
            state=reset_timestep.state,
            observation=reset_timestep.observation,
        )
        return timestep

    def reset(self, params: EnvParamsT, key: jax.Array):
        state = self._generate_problem(params, key)
        state = state.replace(key=key)
        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=transparent_field_of_view(state.grid, state.agent, params.view_size, params.view_size),
        )
        return timestep

    def _custom_env_step(self, params: EnvParamsT, timestep, action: IntOrArray):
        new_grid, new_agent, changed_position = take_action(timestep.state.grid, timestep.state.agent, action)
        new_grid, new_agent = check_rule(timestep.state.rule_encoding, new_grid, new_agent, action, changed_position)

        new_state = timestep.state.replace(
            grid=new_grid,
            agent=new_agent,
            step_num=timestep.state.step_num + 1,
        )
        new_observation = transparent_field_of_view(new_state.grid, new_state.agent, params.view_size, params.view_size)

        # terminated = check_goal(new_state.goal_encoding, new_state.grid, new_state.agent, action, changed_position)
        terminated = jnp.array(False, dtype=jnp.bool)

        assert params.max_steps is not None
        truncated = jnp.equal(new_state.step_num, params.max_steps)

        reward = jax.lax.select(terminated, 1.0 - 0.9 * (new_state.step_num / params.max_steps), 0.0)

        step_type = jax.lax.select(terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        timestep = TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_observation,
        )
        return timestep

    def step(self, params, timestep, action):
        # timestep = self._env.step(params, timestep, action)
        timestep = self._custom_env_step(params, timestep, action)
        timestep = jax.lax.cond(
            timestep.last(),
            lambda: self.__auto_reset(params, timestep),
            lambda: timestep,
        )
        return timestep


# -------------------------------------------------------------------------------------------------

# Wrapper thar checks for self-imposed goals.


class CustomAutoResetWrapper_UnsupervisedGoals(Wrapper):
    def __init__(self, env, *, goal_mode: str = "full_state"):
        super().__init__(env)

        if goal_mode == "full_state":  # exact grid layout + agent position
            self._encode_fn = encode_single_goal_as_full_state
        elif goal_mode == "objects_histogram":  # match object counts only
            self._encode_fn = encode_single_goal_as_object_histogram
        else:
            msg = f"Unknown goal_mode '{goal_mode}'"
            raise ValueError(msg)

    def __auto_reset(self, params, timestep):
        key = timestep.state.key
        reset_timestep = self.reset(params, key)

        timestep = timestep.replace(
            state=reset_timestep.state,
            observation=reset_timestep.observation,
        )
        return timestep

    def reset(self, params: EnvParamsT, key: jax.Array):
        state = self._generate_problem(params, key)
        state = state.replace(key=key)
        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=transparent_field_of_view(state.grid, state.agent, params.view_size, params.view_size),
        )
        return timestep

    def _custom_env_step(self, params: EnvParamsT, timestep, action: IntOrArray, goal: jax.Array):
        # exactly as the original env step except it checks for completion of the passed goal instead of the extrinsic goal from the benchmark
        new_grid, new_agent, changed_position = take_action(timestep.state.grid, timestep.state.agent, action)
        new_grid, new_agent = check_rule(timestep.state.rule_encoding, new_grid, new_agent, action, changed_position)

        new_state = timestep.state.replace(
            grid=new_grid,
            agent=new_agent,
            step_num=timestep.state.step_num + 1,
        )
        new_observation = transparent_field_of_view(new_state.grid, new_state.agent, params.view_size, params.view_size)

        # terminated = check_goal(new_state.goal_encoding, new_state.grid, new_state.agent, action, changed_position)
        new_state_goal_encoding = self._encode_fn(grid_state=new_state.grid, position=new_state.agent.position)
        terminated = jnp.all(jnp.equal(goal, new_state_goal_encoding))

        assert params.max_steps is not None
        truncated = jnp.equal(new_state.step_num, params.max_steps)

        reward = jax.lax.select(terminated, 1.0 - 0.9 * (new_state.step_num / params.max_steps), 0.0)

        step_type = jax.lax.select(terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        timestep = TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_observation,
        )
        return timestep

    def step(self, params, timestep, action, custom_goal):
        # timestep = self._env.step(params, timestep, action)
        timestep = self._custom_env_step(params, timestep, action, custom_goal)
        timestep = jax.lax.cond(
            timestep.last(),
            lambda: self.__auto_reset(params, timestep),
            lambda: timestep,
        )
        return timestep
