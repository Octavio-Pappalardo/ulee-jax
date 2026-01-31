from dataclasses import dataclass


@dataclass
class TrainConfig:
    train_seed: int = 42
    benchmark_split_seed: int = 142
    benchmark_train_percentage: float = 0.9
    env_id: str = "XLand-MiniGrid-R1-8x8"
    benchmark_id: str = "trivial-1m"
    episode_max_steps: int | None = 256

    # training
    num_envs_per_batch: int = 2048
    num_steps_per_env: int = 5120
    num_steps_per_update: int = 256
    total_timesteps: int = 1_000_000_000

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

    # Transformer XL specific
    past_context_length: int = 128
    subsequence_length_in_loss_calculation: int = 64
    num_attn_heads: int = 4
    num_transformer_blocks: int = 2
    transformer_hidden_states_dim: int = 192
    qkv_features: int = 192
    gating: bool = True
    gating_bias = 2.0

    # actor and critic head MLPs
    head_activation: str = "relu"
    head_hidden_dim: int = 256

    # RND specific
    rnd_encoder_mode: str = "objects_histogram"  # Options: "full_state", "objects_histogram"
    rnd_output_embedding_dim: int = 256  # 64
    rnd_head_activation: str = "relu"
    rnd_head_hidden_dim: int = 256

    predictor_network_lr: float = 1e-4  # 1e-5
    rnd_predictor_update_epochs: int = 1
    rnd_predictor_num_minibatches: int = 64
    num_chunks_in_rnd_rewards_computation: int = 64

    gamma_intrinsic: float = 0.99
    gae_lambda_intrinsic: float = 0.95

    extrinsic_coef: float = 1.0
    intrinsic_coef: float = 0.1

    # eval
    eval_num_envs: int = 1024
    eval_num_episodes: int = 20

    def __post_init__(self):
        self.num_batches_of_envs = round(self.total_timesteps / (self.num_envs_per_batch * self.num_steps_per_env))
        self.num_updates_per_batch = self.num_steps_per_env // self.num_steps_per_update
        # checks
        if self.num_steps_per_env % self.num_steps_per_update != 0:
            msg = f"num_steps_per_env ({self.num_steps_per_env}) must be divisible by num_steps_per_update ({self.num_steps_per_update})"
            raise ValueError(msg)

        if self.num_steps_per_update % self.subsequence_length_in_loss_calculation != 0:
            msg = "num_steps_per_update must be divisible by subsequence_length_in_loss_calculation "
            raise ValueError(msg)

        total_steps_per_update = self.num_envs_per_batch * self.num_steps_per_update
        if total_steps_per_update % self.rnd_predictor_num_minibatches != 0:
            msg = "Total collected steps per update (num_envs_per_batch x num_steps_per_update) must be divisible by rnd_predictor_num_minibatches."
            raise ValueError(msg)
        if total_steps_per_update % self.num_chunks_in_rnd_rewards_computation != 0:
            msg = "Total collected steps must be divisible by num_chunks_in_rnd_rewards_computation."
            raise ValueError(msg)
