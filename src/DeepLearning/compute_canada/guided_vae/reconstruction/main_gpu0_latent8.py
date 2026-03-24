from main import main


if __name__ == '__main__':
    main([
        '--exp_name', 'calsnic_optuna_age_gpu0_latent8',
        '--device_idx', '0',
        '--latent_channels', '8',
        '--n_trials', '100',
        '--seed', '101',
        '--age_latent_index', '0',
        '--trial_metrics_fp', '/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/calsnic_als/models/trial_metrics_gpu0_latent8.csv',
        '--intermediate_trials_fp', '/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/calsnic_als/models/intermediate_trials_gpu0_latent8.pt',
        '--log_fp', '/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/calsnic_als/models/logs/gpu0_latent8.log',
    ])
