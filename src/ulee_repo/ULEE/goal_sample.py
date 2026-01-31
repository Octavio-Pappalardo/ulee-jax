import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ulee_repo.shared_code.trainsition_objects import State_Data
from ulee_repo.ULEE.config import TrainConfig


def sample_unsupervised_goals(rng, sampling_method: str, potential_goals: State_Data, difficulties: jnp.ndarray, config: TrainConfig):
    # Sample/select one unsupervised goals for each environment based on the specified sampling method.
    rng, rng_ = jax.random.split(rng)
    # (S, B, ...) -> (B, S, ...)
    (potential_goals, difficulties) = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), (potential_goals, difficulties))

    if sampling_method == "uniform":
        indices = sample_indices_uniform(rng_, difficulties)
        goals = gather_pytree_by_indices(potential_goals, indices)

    elif sampling_method == "bounded_uniform":
        indices = sample_indices_bounded_uniform(rng_, difficulties, config.bounded_uniform_sampling_lower, config.bounded_uniform_sampling_upper)
        goals = gather_pytree_by_indices(potential_goals, indices)

    elif sampling_method == "gaussian":
        indices = sample_indices_gaussian(rng_, difficulties, config.gauss_sampling_mean_difficulty, config.gauss_sampling_std)
        goals = gather_pytree_by_indices(potential_goals, indices)

    elif sampling_method == "gaussian_weights":
        indices = sample_indices_gaussian_weights(rng_, difficulties, config.gauss_sampling_mean_difficulty, config.gauss_sampling_std)
        goals = gather_pytree_by_indices(potential_goals, indices)

    else:
        msg = f"Unknown sampling method: {sampling_method}."
        raise ValueError(msg)

    return rng, goals


##############################################################################


def gather_pytree_by_indices(pytree, indices):
    def gather_fn(x):
        return x[jnp.arange(x.shape[0]), indices, ...]

    return jtu.tree_map(gather_fn, pytree)


# -----------


def sample_indices_gaussian_weights(key, difficulty_scores, mean, std):
    weights_unnorm = jnp.exp(-0.5 * ((difficulty_scores - mean) ** 2) / jnp.square(std))
    weights = weights_unnorm / jnp.sum(weights_unnorm, axis=-1, keepdims=True)
    chosen_indices = jax.random.categorical(key, jnp.log(weights + 1e-15), axis=-1)
    return chosen_indices


# -----------


def sample_indices_gaussian(key, difficulty_scores, mean, std):
    B, S = difficulty_scores.shape
    samples = mean + std * jax.random.normal(key, shape=(B,))
    dists = jnp.abs(difficulty_scores - samples[:, None])
    chosen_indices = jnp.argmin(dists, axis=1)  # (B,)
    return chosen_indices


# -----------


def uniform_sample_goals(rng, potential_goals):
    rng, goal_sampling_rng_base = jax.random.split(rng)
    num_envs = potential_goals.grid_state.shape[0]
    goal_sampling_rng = jax.random.split(goal_sampling_rng_base, num=num_envs)

    def uniform_sample_goal(rng, potential_goals):
        num_potential_goals = potential_goals.grid_state.shape[0]
        rand_idx = jax.random.randint(rng, shape=(), minval=0, maxval=num_potential_goals)
        goal = jax.tree_util.tree_map(lambda x: x[rand_idx], potential_goals)
        return goal

    goals = jax.vmap(uniform_sample_goal)(goal_sampling_rng, potential_goals)

    return rng, goals


def sample_indices_uniform(rng, difficulties):
    num_envs, num_candidates = difficulties.shape[:2]
    indices = jax.random.randint(rng, shape=(num_envs,), minval=0, maxval=num_candidates)
    return indices


# -----------


def sample_indices_bounded_uniform(key, difficulty_scores, lower, upper):
    valid = (difficulty_scores >= lower) & (difficulty_scores <= upper)
    no_valid = ~jnp.any(valid, axis=1, keepdims=True)
    logits = jnp.where(valid, 0.0, -jnp.inf)
    logits = jnp.where(no_valid, jnp.zeros_like(logits), logits)

    return jax.random.categorical(key, logits, axis=1)
