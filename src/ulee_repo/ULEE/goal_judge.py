import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import struct
from flax.training.train_state import TrainState

from ulee_repo.shared_code.trainsition_objects import State_Data
from ulee_repo.ULEE.config import GoalJudgeConfig

##############---------------------------------------


def compute_goal_difficulties(judge_train_state, initial_states: State_Data, goals: State_Data, num_chunks: int):
    # make initial states match the shape of goals
    # (B, ...) -> (1, B, ...) -> (S, B, ...) -> (S*B, ...)
    initial_states = jtu.tree_map(lambda arr: jnp.expand_dims(arr, axis=0), initial_states)
    initial_states = jtu.tree_map(lambda x, y: jnp.broadcast_arrays(x, y)[0], initial_states, goals)
    initial_states = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), initial_states)
    S, B = goals.agent_pos.shape[:2]
    goals = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), goals)
    total_num_goals = S * B

    chunk_size = total_num_goals // num_chunks

    # Reshape everything to (num_chunks, chunk_size, ...)
    initial_states_chunked, goals_chunked = jax.tree_util.tree_map(lambda x: x.reshape((num_chunks, chunk_size, *x.shape[1:])), (initial_states, goals))

    def body_fun(carry, inputs_chunk):
        initial_states_chunk, goals_chunk = inputs_chunk
        difficulties = judge_train_state.apply_fn(judge_train_state.params, initial_states_chunk, goals_chunk)
        return carry, difficulties

    # Scan over chunks
    _, difficulties = jax.lax.scan(
        body_fun,
        None,
        (initial_states_chunked, goals_chunked),
    )
    # (num_chunks, chunk_size) -> (S*B,) -> (S, B)
    difficulties = difficulties.reshape((total_num_goals,)).reshape((S, B))

    return difficulties


##############---------------------------------------


def train_goal_judge(rng, train_state: TrainState, replay_buffer: "JudgeReplayBuffer", config: GoalJudgeConfig):
    # Sample all elements from the replay buffer
    initial_states, goal_states, target_difficulties, mask = replay_buffer.sample_all()

    total_num_datapoints = replay_buffer.capacity
    num_minibatches = total_num_datapoints // config.minibatch_size

    def loss_fn(params, initial_states, goals, target_difficulties, mask):
        preds = train_state.apply_fn(params, initial_states, goals)
        squared_errors = (preds - target_difficulties) ** 2
        masked_errors = squared_errors * mask
        loss = jnp.sum(masked_errors) / jnp.maximum(jnp.sum(mask), 1)
        return loss

    def epoch_body_fun(carry, _unused):
        rng, train_state = carry

        # Shuffle data
        rng, shuffle_rng = jax.random.split(rng)
        indices_permuted = jax.random.permutation(shuffle_rng, total_num_datapoints)
        initial_states_shuffled, goals_shuffled, target_difficulties_shuffled, mask_shuffled = jax.tree_util.tree_map(
            lambda x: jnp.take(x, indices_permuted, axis=0), (initial_states, goal_states, target_difficulties, mask)
        )

        # Reshape to (num_minibatches, minibatch_size, ...)
        initial_states_shuffled, goals_shuffled, target_difficulties_shuffled, mask_shuffled = jax.tree_util.tree_map(
            lambda x: x.reshape((num_minibatches, config.minibatch_size, *x.shape[1:])), (initial_states_shuffled, goals_shuffled, target_difficulties_shuffled, mask_shuffled)
        )

        def minibatch_step(train_state, inputs_chunk):
            initial_states_chunk, goals_chunk, target_difficulties_chunk, mask_chunk = inputs_chunk
            loss, grads = jax.value_and_grad(loss_fn)(train_state.params, initial_states_chunk, goals_chunk, target_difficulties_chunk, mask_chunk)
            new_train_state = train_state.apply_gradients(grads=grads)
            return new_train_state, loss

        # loop over minibatches
        train_state, batch_losses = jax.lax.scan(minibatch_step, train_state, (initial_states_shuffled, goals_shuffled, target_difficulties_shuffled, mask_shuffled))
        epoch_loss = jnp.mean(batch_losses)

        return (rng, train_state), epoch_loss

    # Main loop over epochs
    init_carry = (rng, train_state)
    (rng, final_train_state), epoch_losses = jax.lax.scan(epoch_body_fun, init_carry, None, config.num_epochs)

    loss_value = jnp.mean(epoch_losses)

    return rng, final_train_state, loss_value


##############---------------------------------------


class BufferData(struct.PyTreeNode):
    initial_state: State_Data
    goal_state: State_Data
    target_difficulty: jax.Array


class JudgeReplayBuffer(struct.PyTreeNode):
    data: BufferData
    size: jax.Array
    capacity: int = struct.field(pytree_node=False)
    batch_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, capacity: int, batch_size: int, dummy_state: State_Data) -> "JudgeReplayBuffer":
        # Create an empty replay buffer.
        dummy_initial = State_Data(
            grid_state=jnp.zeros((capacity, *dummy_state.grid_state.shape), dtype=dummy_state.grid_state.dtype),
            agent_pos=jnp.zeros((capacity, *dummy_state.agent_pos.shape), dtype=dummy_state.agent_pos.dtype),
        )

        dummy_goal = State_Data(
            grid_state=jnp.zeros((capacity, *dummy_state.grid_state.shape), dtype=dummy_state.grid_state.dtype),
            agent_pos=jnp.zeros((capacity, *dummy_state.agent_pos.shape), dtype=dummy_state.agent_pos.dtype),
        )

        buffer_data = BufferData(initial_state=dummy_initial, goal_state=dummy_goal, target_difficulty=jnp.zeros(capacity, dtype=jnp.float32))

        return cls(data=buffer_data, size=jnp.array(0, dtype=jnp.int32), capacity=capacity, batch_size=batch_size)

    def add_batch(self, initial_states: State_Data, goal_states: State_Data, target_difficulties: jax.Array) -> "JudgeReplayBuffer":
        # Add a batch of new datapoints, replacing oldest batch if necessary.
        batch_size = self.batch_size

        rolled_initial = State_Data(grid_state=jnp.roll(self.data.initial_state.grid_state, batch_size, axis=0), agent_pos=jnp.roll(self.data.initial_state.agent_pos, batch_size, axis=0))
        rolled_goal = State_Data(grid_state=jnp.roll(self.data.goal_state.grid_state, batch_size, axis=0), agent_pos=jnp.roll(self.data.goal_state.agent_pos, batch_size, axis=0))
        rolled_difficulties = jnp.roll(self.data.target_difficulty, batch_size, axis=0)

        new_data = BufferData(
            initial_state=State_Data(
                grid_state=rolled_initial.grid_state.at[:batch_size].set(initial_states.grid_state), agent_pos=rolled_initial.agent_pos.at[:batch_size].set(initial_states.agent_pos)
            ),
            goal_state=State_Data(grid_state=rolled_goal.grid_state.at[:batch_size].set(goal_states.grid_state), agent_pos=rolled_goal.agent_pos.at[:batch_size].set(goal_states.agent_pos)),
            target_difficulty=rolled_difficulties.at[:batch_size].set(target_difficulties),
        )
        new_size = jnp.minimum(self.size + batch_size, self.capacity)

        return self.replace(data=new_data, size=new_size)

    def sample_all(self) -> tuple[State_Data, State_Data, jax.Array, jax.Array]:
        # Sample all datapoints from the buffer. Includes invalid entries.
        mask = jnp.arange(self.capacity) < self.size
        return (self.data.initial_state, self.data.goal_state, self.data.target_difficulty, mask)
