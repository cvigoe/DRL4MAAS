variant = dict(
    mlflow_uri="http://128.2.210.74:8080",
    gpu=False,
    algorithm="PPO",
    version="normal",
    actor_width=64,                             # Need to tune
    critic_width=256,
    replay_buffer_size=int(3E3),
    algorithm_kwargs=dict(
        min_num_steps_before_training=0,
        num_epochs=150,
        num_eval_steps_per_epoch=1000,
        num_train_loops_per_epoch=10,
        num_expl_steps_per_train_loop=2048,
        num_trains_per_train_loop=100,
        batch_size=256,
        max_path_length=1000,
        clear_buffer_every_train_loop=True,
    ),
    trainer_kwargs=dict(
        epsilon=0.2,                            # Need to tune
        discount=.99,                           # Need to tune
        intrinsic_discount=.9999,
        policy_lr=3E-4,                         # Need to tune
        val_lr=3E-4,                                        # No need to use different
        use_rnd=False,
        rnd_coef=5,
        predictor_update_proportion=0.05,
    ),
    rnd_kwargs=dict(
        rnd_output_size=2,
        rnd_lr=3E-4,
        rnd_latent_size=2,
        use_normaliser=True,                                # Specifies whether to use observation normalisation for actor & critic
    ),
    target_kwargs=dict(
        tdlambda=0.95,
        target_lookahead=15,
        use_dones_for_rnd_critic=False,
    ),
    policy_kwargs=dict(
        std=0.1,                                           # This is a non-learnable constant if set to a scalar value
    ),
)

# env_variant = dict(
#     env_str='Swimmer-v2',
# )

env_variant = dict(
    env_str='agnosticmaas-v0',
    lam=1,
    sigma2=1,
    agent_lambda=2,
    num_hypotheses=30,
    num_timesteps=150,
    num_EA_iterations=10,
    EA_tolerance=0.0001,
    cost_iterations=10,
    upper_limit_N=10,
    log_space_resolution=100,
    MLE_regularizer=.1,
    WASSERSTEIN_ITERS=100,
    verbose=False,
    adaptive_grid=False,
    direct_wasserstein=True,
    fisher_in_state=True,
    reward_shaping=False,
)
