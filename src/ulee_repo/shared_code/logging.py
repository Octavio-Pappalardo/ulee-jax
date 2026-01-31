from dataclasses import asdict

import jax.numpy as jnp
import numpy as np
import wandb

# -----------------------------------------------------------------------------------------------


def generate_run_name(algorithm_name, config, prefix=""):
    if algorithm_name == "ULEE":
        run_name = prefix + f"|{config.benchmark_id}|{config.env_id}|" + algorithm_name + f"seed{config.train_seed}|" + f"{config.goal_search_algorithm}|" + f"{config.goal_sampling_method}||"
    else:
        run_name = prefix + f"|{config.benchmark_id}|{config.env_id}|" + algorithm_name + f"seed{config.train_seed}||"
    return run_name


# -----------------------------------------------------------------------------------------------


def wandb_log_training_metrics(metrics, config, run_name, project_name="ULEE", num_final_episodes_for_evaluating_performance=None, extra_batch_metrics=None, tags=None):
    run = None

    try:
        run = wandb.init(project=project_name, name=run_name, tags=tags, config=asdict(config))
        # setup
        num_batches = metrics["total_loss"].shape[0]
        eval_returns_data = metrics["eval/returns"]
        eval_lengths_data = metrics["eval/lengths"]
        if num_final_episodes_for_evaluating_performance is None:
            final_k_episodes = eval_returns_data.shape[2]  # evaluate over all episodes
        else:
            final_k_episodes = min(num_final_episodes_for_evaluating_performance, eval_returns_data.shape[2])

        #### log plots of metrics during training ####

        total_loss = metrics["total_loss"]
        actor_loss = metrics["actor_loss"]
        value_loss = metrics["value_loss"]
        entropy = metrics["entropy"]
        kl = metrics["kl"]

        eval_returns_mean = eval_returns_data.mean(1)[:, -final_k_episodes:].mean(1)
        eval_lengths_mean = eval_lengths_data.mean(1)[:, -final_k_episodes:].mean(1)
        eval_lengths_20percentile = jnp.percentile(eval_lengths_data, q=20, axis=1)[:, -final_k_episodes:].mean(1)
        eval_lengths_40percentile = jnp.percentile(eval_lengths_data, q=40, axis=1)[:, -final_k_episodes:].mean(1)
        eval_lengths_60percentile = jnp.percentile(eval_lengths_data, q=60, axis=1)[:, -final_k_episodes:].mean(1)
        eval_lengths_80percentile = jnp.percentile(eval_lengths_data, q=80, axis=1)[:, -final_k_episodes:].mean(1)

        for i in range(num_batches):
            batch_logs = {
                # Training losses
                "training/loss/total": total_loss[i],
                "training/loss/actor": actor_loss[i],
                "training/loss/value": value_loss[i],
                "training/loss/entropy": entropy[i],
                "training/kl": kl[i],
                # evaluation metrics
                "eval/returns/mean": eval_returns_mean[i],
                "eval/episode_length/mean": eval_lengths_mean[i],
                "eval/episode_length/p20": eval_lengths_20percentile[i],
                "eval/episode_length/p40": eval_lengths_40percentile[i],
                "eval/episode_length/p60": eval_lengths_60percentile[i],
                "eval/episode_length/p80": eval_lengths_80percentile[i],
            }

            if extra_batch_metrics:
                for k, arr in extra_batch_metrics.items():
                    batch_logs[k] = arr[i]

            wandb.log(batch_logs, step=i + 1, commit=False)

        ### Log data of performance across episodes within a lifetime ###

        # configuration
        batches_for_evaluation = (num_batches - min(15, num_batches // 2), num_batches)
        num_episodes = eval_returns_data.shape[2]

        # metrics to plot
        eval_episodes_lengths_mean = eval_lengths_data.mean(1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_episodes_lengths_20percentile = jnp.percentile(eval_lengths_data, q=20, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_episodes_lengths_40percentile = jnp.percentile(eval_lengths_data, q=40, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_episodes_lengths_60percentile = jnp.percentile(eval_lengths_data, q=60, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_episodes_lengths_80percentile = jnp.percentile(eval_lengths_data, q=80, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)

        for i in range(num_episodes):
            wandb.log(
                {
                    "episode/length/mean": eval_episodes_lengths_mean[i],
                    "episode/length/p20": eval_episodes_lengths_20percentile[i],
                    "episode/length/p40": eval_episodes_lengths_40percentile[i],
                    "episode/length/p60": eval_episodes_lengths_60percentile[i],
                    "episode/length/p80": eval_episodes_lengths_80percentile[i],
                    "episode_number": i + 1,
                },
                step=num_batches + 2 + i,
                commit=False,
            )

        wandb.log({}, commit=True)
    except Exception as e:
        print(f"Error while logging training metrics to wandb: {e}")
    finally:
        if run is not None:
            run.finish()


# -----------------------------------------------------------------------------------------------
# ULEE has its own logging function


def wandb_log_ulee_training_metrics(metrics, config, run_name, project_name="ULEE", tags=None, num_final_episodes_for_evaluating_performance=None):
    run = None
    try:
        run = wandb.init(project=project_name, name=run_name, tags=tags, config=asdict(config))

        num_batches = metrics["total_loss"].shape[0]

        eval_unsupervised_returns_data = metrics["eval/unsupervised/returns"]
        eval_benchmark_returns_data = metrics["eval/benchmark/returns"]
        eval_unsupervised_lengths_data = metrics["eval/unsupervised/lengths"]
        eval_benchmark_lengths_data = metrics["eval/benchmark/lengths"]

        if num_final_episodes_for_evaluating_performance is None:
            final_k_episodes = eval_benchmark_returns_data.shape[2]  # evaluate over all episodes
        else:
            final_k_episodes = min(num_final_episodes_for_evaluating_performance, eval_benchmark_returns_data.shape[2])

        #### log plots of metrics during training ####

        meta_learner_total_loss = metrics["total_loss"]
        meta_learner_actor_loss = metrics["actor_loss"]
        meta_learner_value_loss = metrics["value_loss"]
        meta_learner_entropy = metrics["entropy"]
        meta_learner_lr = metrics["lr"]
        meta_learner_kl = metrics["kl"]

        eval_unsupervised_returns_mean = eval_unsupervised_returns_data.mean(1)[:, -final_k_episodes:].mean(1)
        eval_unsupervised_lengths_mean = eval_unsupervised_lengths_data.mean(1)[:, -final_k_episodes:].mean(1)
        eval_unsupervised_lengths_20percentile = jnp.percentile(eval_unsupervised_lengths_data, q=20, axis=1)[:, -final_k_episodes:].mean(1)
        eval_unsupervised_lengths_40percentile = jnp.percentile(eval_unsupervised_lengths_data, q=40, axis=1)[:, -final_k_episodes:].mean(1)
        eval_unsupervised_lengths_60percentile = jnp.percentile(eval_unsupervised_lengths_data, q=60, axis=1)[:, -final_k_episodes:].mean(1)
        eval_unsupervised_lengths_80percentile = jnp.percentile(eval_unsupervised_lengths_data, q=80, axis=1)[:, -final_k_episodes:].mean(1)

        eval_benchmark_returns_mean = eval_benchmark_returns_data.mean(1)[:, -final_k_episodes:].mean(1)
        eval_benchmark_lengths_mean = eval_benchmark_lengths_data.mean(1)[:, -final_k_episodes:].mean(1)
        eval_benchmark_lengths_20percentile = jnp.percentile(eval_benchmark_lengths_data, q=20, axis=1)[:, -final_k_episodes:].mean(1)
        eval_benchmark_lengths_40percentile = jnp.percentile(eval_benchmark_lengths_data, q=40, axis=1)[:, -final_k_episodes:].mean(1)
        eval_benchmark_lengths_60percentile = jnp.percentile(eval_benchmark_lengths_data, q=60, axis=1)[:, -final_k_episodes:].mean(1)
        eval_benchmark_lengths_80percentile = jnp.percentile(eval_benchmark_lengths_data, q=80, axis=1)[:, -final_k_episodes:].mean(1)

        judge_loss = metrics["judge_loss"]
        task_difficulty_data = metrics["training_goals_difficulties"]  # (num_batches, num_envs_per_batch)
        task_difficulty_prediction_data = metrics["predicted_difficulties"]  # (num_batches, num_envs_per_batch)

        if config.goal_search_algorithm != "random":
            goal_search_total_loss = metrics["goal_search/total_loss"]
            goal_search_actor_loss = metrics["goal_search/actor_loss"]
            goal_search_value_loss = metrics["goal_search/value_loss"]
            goal_search_entropy = metrics["goal_search/entropy"]
            goal_search_kl = metrics["goal_search/kl"]

            if config.goal_search_algorithm == "diayn":
                discriminator_loss = metrics["goal_search/discriminator_loss"]
                skills_log_prob = metrics["goal_search/skills_log_prob"]

        for i in range(num_batches):
            logs_dict = {
                # Training losses
                "training/meta_learner/loss/total": meta_learner_total_loss[i],
                "training/meta_learner/loss/actor": meta_learner_actor_loss[i],
                "training/meta_learner/loss/value": meta_learner_value_loss[i],
                "training/meta_learner/loss/entropy": meta_learner_entropy[i],
                "training/meta_learner/learning_rate": meta_learner_lr[i],
                "training/meta_learner/kl": meta_learner_kl[i],
                # Unsupervised evaluation metrics
                "eval/unsupervised/returns/mean": eval_unsupervised_returns_mean[i],
                "eval/unsupervised/episode_length/mean": eval_unsupervised_lengths_mean[i],
                "eval/unsupervised/episode_length/p20": eval_unsupervised_lengths_20percentile[i],
                "eval/unsupervised/episode_length/p40": eval_unsupervised_lengths_40percentile[i],
                "eval/unsupervised/episode_length/p60": eval_unsupervised_lengths_60percentile[i],
                "eval/unsupervised/episode_length/p80": eval_unsupervised_lengths_80percentile[i],
                # Benchmark evaluation metrics
                "eval/benchmark/returns/mean": eval_benchmark_returns_mean[i],
                "eval/benchmark/episode_length/mean": eval_benchmark_lengths_mean[i],
                "eval/benchmark/episode_length/p20": eval_benchmark_lengths_20percentile[i],
                "eval/benchmark/episode_length/p40": eval_benchmark_lengths_40percentile[i],
                "eval/benchmark/episode_length/p60": eval_benchmark_lengths_60percentile[i],
                "eval/benchmark/episode_length/p80": eval_benchmark_lengths_80percentile[i],
                # Goal Judge metrics
                "goal_judge/loss": judge_loss[i],
                "goal_judge/task_diff_histogram": wandb.Histogram(
                    task_difficulty_data[i, :],
                    num_bins=20,
                ),
                "goal_judge/task_predicted_diff_histogram": wandb.Histogram(
                    task_difficulty_prediction_data[i, :],
                    num_bins=20,
                ),
            }

            if config.goal_search_algorithm != "random":
                logs_dict.update(
                    {
                        # goal search metrics
                        "goal_search/loss/total": goal_search_total_loss[i],
                        "goal_search/loss/actor": goal_search_actor_loss[i],
                        "goal_search/loss/value": goal_search_value_loss[i],
                        "goal_search/loss/entropy": goal_search_entropy[i],
                        "goal_search/kl": goal_search_kl[i],
                    }
                )
                if config.goal_search_algorithm == "diayn":
                    logs_dict.update(
                        {
                            "goal_search/discriminator_loss": discriminator_loss[i],
                            "goal_search/skills_log_prob": skills_log_prob[i],
                        }
                    )

            wandb.log(
                logs_dict,
                step=i + 1,
                commit=False,
            )

        wandb.log({}, commit=True)

        ### Log data of performance across episodes within a lifetime ###

        # configuration
        batches_for_evaluation = (num_batches - min(15, num_batches // 2), num_batches)
        num_episodes = eval_unsupervised_returns_data.shape[2]

        # metrics to plot
        eval_unsupervised_episodes_lengths_mean = eval_unsupervised_lengths_data.mean(1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_unsupervised_episodes_lengths_20percentile = jnp.percentile(eval_unsupervised_lengths_data, q=20, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_unsupervised_episodes_lengths_40percentile = jnp.percentile(eval_unsupervised_lengths_data, q=40, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_unsupervised_episodes_lengths_60percentile = jnp.percentile(eval_unsupervised_lengths_data, q=60, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_unsupervised_episodes_lengths_80percentile = jnp.percentile(eval_unsupervised_lengths_data, q=80, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)

        eval_benchmark_episodes_lengths_mean = eval_benchmark_lengths_data.mean(1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_benchmark_episodes_lengths_20percentile = jnp.percentile(eval_benchmark_lengths_data, q=20, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_benchmark_episodes_lengths_40percentile = jnp.percentile(eval_benchmark_lengths_data, q=40, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_benchmark_episodes_lengths_60percentile = jnp.percentile(eval_benchmark_lengths_data, q=60, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_benchmark_episodes_lengths_80percentile = jnp.percentile(eval_benchmark_lengths_data, q=80, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)

        for i in range(num_episodes):
            wandb.log(
                {
                    # Unsupervised evaluation metrics
                    "episode/unsupervised/length/mean": eval_unsupervised_episodes_lengths_mean[i],
                    "episode/unsupervised/length/p20": eval_unsupervised_episodes_lengths_20percentile[i],
                    "episode/unsupervised/length/p40": eval_unsupervised_episodes_lengths_40percentile[i],
                    "episode/unsupervised/length/p60": eval_unsupervised_episodes_lengths_60percentile[i],
                    "episode/unsupervised/length/p80": eval_unsupervised_episodes_lengths_80percentile[i],
                    # Benchmark evaluation metrics
                    "episode/benchmark/length/mean": eval_benchmark_episodes_lengths_mean[i],
                    "episode/benchmark/length/p20": eval_benchmark_episodes_lengths_20percentile[i],
                    "episode/benchmark/length/p40": eval_benchmark_episodes_lengths_40percentile[i],
                    "episode/benchmark/length/p60": eval_benchmark_episodes_lengths_60percentile[i],
                    "episode/benchmark/length/p80": eval_benchmark_episodes_lengths_80percentile[i],
                    # for x-axis
                    "episode_number": i + 1,
                },
                step=num_batches + 2 + i,
                commit=False,
            )

        wandb.log({}, commit=True)
    except Exception as e:
        print(f"Error while logging training metrics to wandb: {e}")
    finally:
        if run is not None:
            run.finish()
