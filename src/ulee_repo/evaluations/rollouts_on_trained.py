from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import orbax
import xminigrid
from flax.core import unfreeze
from flax.training import orbax_utils
from flax.training.train_state import TrainState

from ulee_repo.DIAYN.config import TrainConfig as DIAYNTrainConfig
from ulee_repo.DIAYN.setups import TrainStateWithConstants
from ulee_repo.evaluations.diayn_evals import eval_diayn_selecting_best_skill
from ulee_repo.evaluations.rollouts import RolloutEpisodeStats, create_benchmark_step_func, eval_rollout, eval_rollout_random
from ulee_repo.networks.diayn_transformer_actor_critic import DiaynActorCriticTransformer
from ulee_repo.networks.transformer_actor_critic import ActorCriticTransformer
from ulee_repo.PPO.config import TrainConfig as PPOTrainConfig
from ulee_repo.shared_code.wrappers import CustomGymAutoResetWrapper
from ulee_repo.ULEE.config import TrainConfig as ULEETrainConfig


def rollout_on_trained_weights(
    rng, num_envs: int, num_episodes: int, algorithm_id: str, env_id: str, benchmark_id: str, weights_path: Path, results_path: Path, *, eval_on_test_benchmark: bool = True
):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    ###### ------------- LOAD PARAMS AND SET UP CONFIGS --------------- ###############
    if algorithm_id == "ulee":
        data = orbax_checkpointer.restore(weights_path)
        config = unfreeze(data["config"])
        config.pop("goal_search", None)
        config.pop("goal_judge", None)
        config["env_id"] = env_id
        config["benchmark_id"] = benchmark_id
        config = ULEETrainConfig(**config)

    elif algorithm_id == "diayn":
        data = orbax_checkpointer.restore(weights_path)
        config = unfreeze(data["config"])
        config["env_id"] = env_id
        config["benchmark_id"] = benchmark_id
        config = DIAYNTrainConfig(**config)
        config.eval_num_envs = num_envs

    elif algorithm_id == "standard_ppo":
        data = orbax_checkpointer.restore(weights_path)
        config = unfreeze(data["config"])
        config["env_id"] = env_id
        config["benchmark_id"] = benchmark_id
        config = PPOTrainConfig(**config)

    elif algorithm_id == "random":
        config = None

    ###### ------------- SET UP ENVS--------------- ###############

    if algorithm_id != "random":
        if config.episode_max_steps:
            env, env_params = xminigrid.make(env_id, max_steps=config.episode_max_steps)
        else:
            env, env_params = xminigrid.make(env_id)
        env = CustomGymAutoResetWrapper(env)
        bench = xminigrid.load_benchmark(benchmark_id)
        if eval_on_test_benchmark:
            split_rng = jax.random.key(config.benchmark_split_seed)
            _, benchmark = bench.shuffle(key=split_rng).split(prop=config.benchmark_train_percentage)
        else:
            split_rng = jax.random.key(config.benchmark_split_seed)
            benchmark, _ = bench.shuffle(key=split_rng).split(prop=config.benchmark_train_percentage)
    else:
        env, env_params = xminigrid.make(env_id, max_steps=256)  # max steps hardcoded for random rollouts
        env = CustomGymAutoResetWrapper(env)
        benchmark = xminigrid.load_benchmark(benchmark_id)

    rng, ruleset_rng, reset_rng, actions_rng = jax.random.split(rng, num=4)
    reset_rng = jax.random.split(reset_rng, num=num_envs)
    actions_rng = jax.random.split(actions_rng, num=num_envs)
    if env_id.startswith("XLand-MiniGrid"):
        ruleset_rng = jax.random.split(ruleset_rng, num=num_envs)
        ruleset = jax.vmap(benchmark.sample_ruleset)(ruleset_rng)
        env_params = env_params.replace(ruleset=ruleset)
    elif config.env_id.startswith("MiniGrid"):
        env_params = env_params.replace(view_size=5)
    step_func = create_benchmark_step_func(env)

    ###### ------------- SET UP NETWORK AND COLLECT ROLLOUTS ----------- ########
    if algorithm_id == "ulee":
        meta_learner_network = ActorCriticTransformer(
            algorithm_id="meta_learning",
            num_actions=env.num_actions(env_params),
            hidden_dim=config.transformer_hidden_states_dim,
            num_attn_heads=config.num_attn_heads,
            qkv_features=config.qkv_features,
            num_layers_in_transformer=config.num_transformer_blocks,
            gating=config.gating,
            gating_bias=config.gating_bias,
            head_activation=config.head_activation,
            mlp_dim=config.head_hidden_dim,
            obs_emb_dim=config.obs_emb_dim,
            action_emb_dim=config.action_emb_dim,
        )
        meta_learner_network_params = data["meta_learner_params"]
        # meta_learner_network_params = data['best_meta_learner_params']
        meta_learner_train_state = TrainState.create(apply_fn=meta_learner_network.apply, params=meta_learner_network_params, tx=optax.sgd(learning_rate=0.0))

        _, eval_stats, _ = jax.vmap(eval_rollout, in_axes=(0, 0, None, 0, None, None, None, None, None, None))(
            actions_rng,
            reset_rng,
            env,
            env_params,
            step_func,
            meta_learner_train_state,
            num_episodes,
            "meta_learning",
            config,
            None,  # goals
        )

    elif algorithm_id == "diayn":
        diayn_network = DiaynActorCriticTransformer(
            num_actions=env.num_actions(env_params),
            obs_emb_dim=config.obs_emb_dim,
            num_skills=config.num_skills,
            skill_emb_dim=config.skill_emb_dim,
            hidden_dim=config.transformer_hidden_states_dim,
            num_attn_heads=config.num_attn_heads,
            qkv_features=config.qkv_features,
            num_layers_in_transformer=config.num_transformer_blocks,
            gating=config.gating,
            gating_bias=config.gating_bias,
            head_activation=config.head_activation,
            mlp_dim=config.head_hidden_dim,
            skill_bias=config.skill_bias,
            skill_bias_scale=config.skill_bias_scale,
        )
        diayn_network_variables = data["agent_params"]
        # diayn_network_variables = data['best_agent_params']
        if not config.skill_bias:
            diayn_train_state = TrainStateWithConstants.create(apply_fn=diayn_network.apply, params=diayn_network_variables["params"], tx=optax.sgd(learning_rate=0.0))
        else:

            def apply_with_constants(params, *args, **kwargs):
                return diayn_network.apply({"params": params, "constants": diayn_network_variables["constants"]}, *args, **kwargs)

            diayn_train_state = TrainStateWithConstants.create(
                apply_fn=apply_with_constants, params=diayn_network_variables["params"], tx=optax.sgd(learning_rate=0.0), constants=diayn_network_variables["constants"]
            )

        # compute number of episodes to run for each skill and for the best skill
        eps_per_skill = (num_episodes - 3) // config.num_skills
        if eps_per_skill == 0:
            msg = "Not enough episodes to run every skill."
            raise ValueError(msg)
        eps_with_best_skill = num_episodes - eps_per_skill * config.num_skills
        config.num_eval_episodes_per_skill = eps_per_skill
        config.num_eval_episodes_with_best_skill = eps_with_best_skill

        goals = jnp.zeros((num_envs, 10), dtype=jnp.int32)  # ignored during eval

        # collect evaluation data
        rng, stats_for_each_skill, best_skill_stats = eval_diayn_selecting_best_skill(
            rng=rng,
            eval_reset_rng=reset_rng,
            env=env,
            eval_env_params=env_params,
            step_func=step_func,
            train_state=diayn_train_state,
            config=config,
            goals=goals,
        )
        eval_all_skills_length_data = jnp.swapaxes(stats_for_each_skill.lengths, 0, 1)
        eval_all_skills_returns_data = jnp.swapaxes(stats_for_each_skill.returns, 0, 1)
        eval_all_skills_length_data = jnp.reshape(eval_all_skills_length_data, (eval_all_skills_length_data.shape[0], -1))
        eval_all_skills_returns_data = jnp.reshape(eval_all_skills_returns_data, (eval_all_skills_returns_data.shape[0], -1))
        eval_length_data = jnp.concatenate([eval_all_skills_length_data, best_skill_stats.lengths], axis=1)  # [num_envs, num_episodes]
        eval_returns_data = jnp.concatenate([eval_all_skills_returns_data, best_skill_stats.returns], axis=1)  # [num_envs, num_episodes]

        eval_stats = RolloutEpisodeStats(
            returns=eval_returns_data,
            lengths=eval_length_data,
        )

    elif algorithm_id == "standard_ppo":
        ppo_network = ActorCriticTransformer(
            algorithm_id="standard_ppo",
            num_actions=env.num_actions(env_params),
            hidden_dim=config.transformer_hidden_states_dim,
            num_attn_heads=config.num_attn_heads,
            qkv_features=config.qkv_features,
            num_layers_in_transformer=config.num_transformer_blocks,
            gating=config.gating,
            gating_bias=config.gating_bias,
            head_activation=config.head_activation,
            mlp_dim=config.head_hidden_dim,
            obs_emb_dim=config.obs_emb_dim,
        )
        ppo_network_params = data["agent_params"]
        # ppo_network_params = data['best_agent_params']
        ppo_train_state = TrainState.create(apply_fn=ppo_network.apply, params=ppo_network_params, tx=optax.sgd(learning_rate=0.0))

        _, eval_stats, _ = jax.vmap(eval_rollout, in_axes=(0, 0, None, 0, None, None, None, None, None, None))(
            actions_rng,
            reset_rng,
            env,
            env_params,
            step_func,
            ppo_train_state,
            num_episodes,
            "standard_ppo",
            config,
            None,
        )

    elif algorithm_id == "random":
        _, eval_stats, _ = jax.vmap(eval_rollout_random, in_axes=(0, 0, None, 0, None, None, None))(
            actions_rng,
            reset_rng,
            env,
            env_params,
            step_func,
            num_episodes,
            None,
        )
    else:
        msg = f"Unknown algorithm_id: {algorithm_id}. Supported values are 'ulee', 'diayn', 'ppo', and 'random'."
        raise ValueError(msg)

    # save results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "returns": eval_stats.returns,
        "lengths": eval_stats.lengths,
    }
    save_args = orbax_utils.save_args_from_target(results)
    orbax_checkpointer.save(results_path, results, save_args=save_args)
    print(f"Saved evaluation rollout results to {results_path}")

    return eval_stats.returns, eval_stats.lengths
