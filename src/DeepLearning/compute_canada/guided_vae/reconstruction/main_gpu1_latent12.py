from main import main


if __name__ == '__main__':
    main([
        '--exp_name', 'calsnic_optuna_age_gpu1_latent12',
        '--device_idx', '1',
        '--latent_channels', '12',
        '--n_trials', '100',
        '--seed', '202',
        '--age_latent_index', '0',
        '--trial_metrics_fp', '/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/calsnic_als/models/trial_metrics_gpu1_latent12.csv',
        '--intermediate_trials_fp', '/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/calsnic_als/models/intermediate_trials_gpu1_latent12.pt',
        '--log_fp', '/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/calsnic_als/models/logs/gpu1_latent12.log',
    ])
