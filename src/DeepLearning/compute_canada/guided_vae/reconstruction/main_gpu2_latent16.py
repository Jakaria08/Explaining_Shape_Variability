from main import main


if __name__ == '__main__':
    main([
        '--exp_name', 'calsnic_optuna_age_gpu2_latent16',
        '--device_idx', '2',
        '--latent_channels', '16',
        '--n_trials', '100',
        '--trial_metrics_fp', '/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/calsnic_als/models/trial_metrics_gpu2_latent16.txt',
        '--intermediate_trials_fp', '/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/calsnic_als/models/intermediate_trials_gpu2_latent16.pt',
    ])
