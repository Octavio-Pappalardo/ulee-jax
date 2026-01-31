from dataclasses import dataclass, field

######################################


@dataclass
class GoalJudgeConfig:
    num_episodes_to_compute_success_rate: int = 5
    replay_buffer_num_batches: int = 5

    adam_eps: float = 1e-5
    lr: float = 1e-4
    minibatch_size: int = 256
    num_epochs: int = 2

    # network
    grid_state_emb_dim: int = 16
    head_activation: str = "relu"
    mlp_dim: int = 256


#####################################


@dataclass
class GoalSearchConfigBase:
    num_envs_per_batch: int = field(init=False)

    num_steps_per_update: int = 256
    num_updates_per_batch: int = 3

    goal_searching_episodes_per_env: int = 2
    goal_searching_steps_per_env: int = field(init=False)

    subsample_step: int = 15

    num_chunks_for_computing_difficulties_in_goal_selection: int = 32
    num_chunks_for_computing_intrinsic_rewards: int = 32


@dataclass
class GoalSearchConfigRandom(GoalSearchConfigBase):
    pass


@dataclass
class GoalSearchConfigPPO(GoalSearchConfigBase):
    # training
    update_epochs: int = 1
    num_minibatches: int = 16
    adam_eps: float = 1e-5
    lr: float = 2e-4
    anneal_lr: bool = False
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # network
    obs_emb_dim: int = 16
    past_context_length: int = 128
    subsequence_length_in_loss_calculation: int = 64
    num_attn_heads: int = 4
    num_transformer_blocks: int = 2
    transformer_hidden_states_dim: int = 192
    qkv_features: int = 192
    gating: bool = True
    gating_bias = 2.0
    head_activation: str = "relu"
    head_hidden_dim: int = 256


@dataclass
class GoalSearchConfigDIAYN(GoalSearchConfigBase):
    discriminator_mode: str = "objects_histogram"
    diayn_coef: float = 0.1
    # encoding
    skill_emb_dim: int = 16
    obs_emb_dim: int = 16
    grid_state_emb_dim: int = 16
    # training
    update_epochs: int = 1
    num_minibatches: int = 16
    adam_eps: float = 1e-5
    lr: float = 2e-4
    anneal_lr: bool = False
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # network
    past_context_length: int = 128
    subsequence_length_in_loss_calculation: int = 64
    num_attn_heads: int = 4
    num_transformer_blocks: int = 2
    transformer_hidden_states_dim: int = 192
    qkv_features: int = 192
    gating: bool = True
    gating_bias = 2.0
    head_activation: str = "relu"
    head_hidden_dim: int = 256
    skill_bias: bool = True
    skill_bias_scale: float = 8.0
    # discriminator
    discriminator_network_lr: float = 3e-4
    num_skill_discriminator_training_epochs: int = 1
    num_skill_discriminator_minibatches: int = 64
    num_chunks_in_diayn_rewards_computation: int = 64
    discriminator_head_hidden_dim: int = 256
    discriminator_head_activation: str = "relu"


###################################################################


@dataclass
class TrainConfig:
    train_seed: int = 42
    benchmark_split_seed: int = 142
    benchmark_train_percentage: float = 0.9
    env_id: str = "XLand-MiniGrid-R1-8x8"
    benchmark_id: str = "trivial-1m"
    episode_max_steps: int | None = 256
    goal_mode: str = "objects_histogram"  # Options: "full_state", "objects_histogram"

    # training
    num_envs_per_batch: int = 2048
    num_steps_per_env: int = 5120
    num_steps_per_update: int = 256
    total_timesteps: int = 5_000_000_000

    update_epochs: int = 1
    num_minibatches: int = 16

    adam_eps: float = 1e-5
    lr: float = 2e-4
    anneal_lr: bool = False
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # encoders
    obs_emb_dim: int = 16
    action_emb_dim: int = 16

    # Transformer XL specific
    past_context_length: int = 128
    subsequence_length_in_loss_calculation: int = 64
    num_attn_heads: int = 4
    num_transformer_blocks: int = 2
    transformer_hidden_states_dim: int = 192
    qkv_features: int = 192
    gating: bool = True
    gating_bias = 2.0

    # actor and critic heads
    head_activation: str = "relu"
    head_hidden_dim: int = 256

    # Difficulty Predictor
    goal_judge: GoalJudgeConfig = field(default_factory=GoalJudgeConfig)

    # Goal-search Policy
    goal_search_algorithm: str = "ppo"  # Options: "random", "ppo", "diayn"
    goal_search: GoalSearchConfigBase | None = None

    # Goal sampling
    goal_sampling_method: str = "bounded_uniform"  # Options: "uniform", "bounded_uniform", "gaussian", "gaussian_weights"
    gauss_sampling_mean_difficulty: float = 0.6
    gauss_sampling_std: float = 0.2
    bounded_uniform_sampling_lower: float = 0.1
    bounded_uniform_sampling_upper: float = 0.9

    # eval
    eval_num_envs: int = 1024
    eval_num_episodes: int = 25

    def __post_init__(self):
        self.num_batches_of_envs = round(self.total_timesteps / (self.num_envs_per_batch * self.num_steps_per_env))
        self.num_updates_per_batch = self.num_steps_per_env // self.num_steps_per_update
        if self.num_steps_per_env % self.num_steps_per_update != 0:
            msg = f"num_steps_per_env ({self.num_steps_per_env}) must be divisible by num_steps_per_update ({self.num_steps_per_update})"
            raise ValueError(msg)

        if self.goal_sampling_method not in ["uniform", "bounded_uniform", "gaussian", "gaussian_weights"]:
            msg = f"Unknown goal sampling method: {self.goal_sampling_method}"
            raise ValueError(msg)

        if self.goal_search is None:
            if self.goal_search_algorithm == "random":
                self.goal_search = GoalSearchConfigRandom()
            elif self.goal_search_algorithm == "ppo":
                self.goal_search = GoalSearchConfigPPO()
            elif self.goal_search_algorithm == "diayn":
                self.goal_search = GoalSearchConfigDIAYN()
            else:
                msg = f"Unknown goal search algorithm: {self.goal_search_algorithm}"
                raise ValueError(msg)
        elif (
            (self.goal_search_algorithm == "random" and not isinstance(self.goal_search, GoalSearchConfigRandom))
            or (self.goal_search_algorithm == "ppo" and not isinstance(self.goal_search, GoalSearchConfigPPO))
            or (self.goal_search_algorithm == "diayn" and not isinstance(self.goal_search, GoalSearchConfigDIAYN))
        ):
            msg = "Provided goal_search config does not match goal_search_algorithm"
            raise TypeError(msg)

        self.goal_search.num_envs_per_batch = self.num_envs_per_batch

        # Checks
        if self.num_steps_per_update % self.subsequence_length_in_loss_calculation != 0:
            msg = "num_steps_per_update must be divisible by subsequence_length_in_loss_calculation "
            raise ValueError(msg)

        if self.goal_search_algorithm == "diayn":
            total_steps_per_update = self.num_envs_per_batch * self.num_steps_per_update

            if total_steps_per_update % self.goal_search.num_skill_discriminator_minibatches != 0:
                msg = "Total collected steps per update (num_envs_per_batch x num_steps_per_update) must be divisible by goal_search.num_skill_discriminator_minibatches."
                raise ValueError(msg)
            if total_steps_per_update % self.goal_search.num_chunks_in_diayn_rewards_computation != 0:
                msg = "Total collected steps must be divisible by goal_search.num_chunks_in_diayn_rewards_computation."
                raise ValueError(msg)

        if self.num_envs_per_batch % self.goal_search.num_chunks_for_computing_difficulties_in_goal_selection != 0:
            msg = f"num_envs_per_batch ({self.num_envs_per_batch}) must be divisible by num_chunks_for_computing_difficulties_in_goal_selection ({self.goal_search.num_chunks_for_computing_difficulties_in_goal_selection})"
            raise ValueError(msg)
        if self.num_envs_per_batch % self.goal_search.num_chunks_for_computing_intrinsic_rewards != 0:
            msg = f"num_envs_per_batch ({self.num_envs_per_batch}) must be divisible by num_chunks_for_computing_intrinsic_rewards ({self.goal_search.num_chunks_for_computing_intrinsic_rewards})"
            raise ValueError(msg)
        if (self.goal_judge.replay_buffer_num_batches * self.num_envs_per_batch) % self.goal_judge.minibatch_size != 0:
            msg = f"replay buffer capacity (goal_judge.replay_buffer_num_batches x num_envs_per_batch) must be divisible by goal judge minibatch_size ({self.goal_judge.minibatch_size})"
            raise ValueError(msg)

        if self.goal_mode not in ["full_state", "objects_histogram"]:
            msg = f"Unknown goal_mode: {self.goal_mode}. Options are 'full_state' and 'objects_histogram'."
            raise ValueError(msg)
