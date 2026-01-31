from pathlib import Path

artifacts_path = Path("../../../artifacts").resolve()


def build_trained_weights_path(algorithm_id, env_id, benchmark_id, seed, goal_search_algorithm="random", goal_sampling_method="uniform"):
    if algorithm_id == "ulee":
        path = artifacts_path / f"training_results/ulee/{goal_search_algorithm}_{goal_sampling_method}/{benchmark_id}_{env_id}/seed{seed}"
    else:
        path = artifacts_path / f"training_results/{algorithm_id}/{benchmark_id}_{env_id}/seed{seed}"

    return path


def build_best_weights_rollouts_path(algorithm_id, env_id, benchmark_id, seed, goal_search_algorithm="random", goal_sampling_method="uniform"):
    if algorithm_id == "ulee":
        path = artifacts_path / f"best_weights_rollouts/ulee/{goal_search_algorithm}_{goal_sampling_method}/{benchmark_id}_{env_id}/seed{seed}"
    else:
        path = artifacts_path / f"best_weights_rollouts/{algorithm_id}/{benchmark_id}_{env_id}/seed{seed}"

    return path


def build_finetuned_weights_path(algorithm_id, env_id, benchmark_id, train_seed, finetune_seed, goal_search_algorithm="random", goal_sampling_method="uniform"):
    if algorithm_id == "ulee":
        path = artifacts_path / f"training_results/ulee/{goal_search_algorithm}_{goal_sampling_method}/{benchmark_id}_{env_id}/seed{train_seed}_finetune_seed{finetune_seed}"
    else:
        path = artifacts_path / f"training_results/{algorithm_id}/{benchmark_id}_{env_id}/seed{train_seed}_finetune_seed{finetune_seed}"

    return path


def build_finetuned_on_meta_rl_path(algorithm_id, env_id, benchmark_id, train_seed, finetune_seed, goal_search_algorithm="random", goal_sampling_method="uniform"):
    if algorithm_id == "ulee":
        path = artifacts_path / f"training_results/ulee/{goal_search_algorithm}_{goal_sampling_method}/{benchmark_id}_{env_id}/seed{train_seed}_meta_finetune_seed{finetune_seed}"
    else:
        path = artifacts_path / f"training_results/{algorithm_id}/{benchmark_id}_{env_id}/seed{train_seed}_meta_finetune_seed{finetune_seed}"

    return path
