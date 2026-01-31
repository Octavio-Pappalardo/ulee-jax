import jax
import jax.numpy as jnp
from flax import struct


class NormalizationStats(struct.PyTreeNode):
    running_forward_return: jnp.ndarray
    count: jnp.ndarray = jnp.array(0.0)
    mean: jnp.ndarray = jnp.array(0.0)
    M2: jnp.ndarray = jnp.array(0.0)
    var: jnp.ndarray = jnp.array(1.0)


def compute_intrinsic_returns(running_forward_return, rewards: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """Compute intrinsic returns for normalizing intrinsic rewards.
    Doesn't actually compute returns, but rather discounted sum of rewards forward in time. Doing it forward in time instead of backward allows to keep a running estimate across multiple calls.
    These 'forward returns' are computed across the whole training interaction without resetting at episode boundaries, following RND's original implementation.
    """

    def step(running_return, reward):
        running_return = reward + gamma * running_return
        return running_return, running_return

    new_running_forward_return, returns = jax.lax.scan(step, running_forward_return, rewards, unroll=16, reverse=False)

    return new_running_forward_return, returns


def update_normalization_stats(old_stats: NormalizationStats, new_data: jnp.ndarray) -> NormalizationStats:
    """Update normalization statistics using Chan's batch version of Welford's algorithm."""
    batch_count = jnp.array(new_data.shape[0], dtype=old_stats.count.dtype)
    batch_mean = jnp.mean(new_data)
    batch_M2 = jnp.sum((new_data - batch_mean) ** 2)

    new_count = old_stats.count + batch_count
    delta = batch_mean - old_stats.mean
    new_mean = old_stats.mean + delta * batch_count / new_count
    new_M2 = old_stats.M2 + batch_M2 + (delta**2) * (old_stats.count * batch_count) / new_count

    new_var = new_M2 / new_count

    return old_stats.replace(count=new_count, mean=new_mean, M2=new_M2, var=new_var)
