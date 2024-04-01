# Disentangling Hippocampal Shape Variations: A Study of Neurological Disorders Using Graph Variational Autoencoder with Contrastive Learning

This repository contains the code and resources for the paper: **Disentangling Hippocampal Shape Variations: A Study of Neurological Disorders Using Graph Variational Autoencoder with Contrastive Learning** [arXiv Link].

![overall_architecture_update.png](https://github.com/Jakaria08/Explaining_Shape_Variability/blob/master/figures/overall_architecture_update.png)

Figure 1: The overall architecture of our method. We have graph VAE with an encoder $f_\phi(x)$ and decoder $f_\theta(z)$ where $x$ is the input 3D mesh and $z$ represents the latent space. $L_{vae}$ represents the VAE loss that combines reconstruction and KL divergence loss. Another two losses are classification loss $L_{contr}^{cls}$ and regression loss $L_{contr}^{reg}$, where a specific latent variable is disentangled for a specific feature (continuous or discrete). We use the first variable for contrastive classification loss ($z_{1}$ corresponds to binary labels, and the rest variables are uncorrelated to the labels). The second variable $z_{2}$ corresponds to regression loss, and the rest variables are uncorrelated to the continuous labels. 

## Table of Contents

1. [Code](#code)
    - [Data Processing](#data-processing)
    - [Synthetic Torus Data Generation](#synthetic-torus-data-generation)
2. [Setup and Commands](#setup-and-commands)
3. [Trained Models](#trained-models)
4. [Visualizations](#visualizations)
   - [Torus Data](#torus-data)
   - [Hippocampal Volume Change with Age and Multiple Sclerosis (MS)](#hippocampal-volume-change-with-age-and-multiple-sclerosis-ms)

# Code

The code is organized in the following hierarchy (Only the directory structure is shown without the files for clarity):

```Explaining_Shape_Variability/
├─ figures/
├─ preprocessing/
├─ src/
│ ├─ DeepLearning/
│ │ ├─ compute_canada/
│ │ │ ├─ guided_vae/
│ │ │ │ ├─ conv/
│ │ │ │ ├─ datasets/
│ │ │ │ ├─ reconstruction/
│ │ │ │ ├─ utils/
├─ synthetic_data/
├─ utils/
```

## Data Processing

Hippocampus data initially has DTI scans with segmentation masks saved in .nii files. 3D meshes are created from the files and registered to a template shape. Details of preprocessing (scripts for grooming and registration of the Hippocampus data and requirement.txt files) can be found at [Preprecessing](https://github.com/Jakaria08/Explaining_Shape_Variability/tree/master/preprocessing) and corresponding ReadMe file is [Preprocessing README](https://github.com/Jakaria08/Explaining_Shape_Variability/tree/master/preprocessing#readme). Registration is done on Compute Canada GPU cluster. The hippocampus data is confidential and cannot be shared, the preprocessing scripts can be utilized for any publicly available MRI data that includes hippocampus segmentation.

## Synthetic Torus Data Generation

Synthetic torus data generation by a jupyter notebook is stored [here](https://github.com/Jakaria08/Explaining_Shape_Variability/tree/master/synthetic_data). There are other Python scripts and README files for different synthetic data generation.

# Setup and Commands

Required packages to run the code can be found [here](https://github.com/Jakaria08/Explaining_Shape_Variability/tree/master/src/DeepLearning/compute_canada) and [here](https://github.com/Jakaria08/Explaining_Shape_Variability/tree/master/src/DeepLearning). We ran the code in both Compute Canada and the local GPU machine. Setup instructions for both options can be found in the above-mentioned links.

The `data/CoMA/` directory needs to be created under `guided_vae/`. Four folders `processed/`, `raw/`, `template/`, and `transform/` needs to be created under `data/CoMA/`. Hippocampus and synthetic torus data should be stored under `raw/` folder like `raw/hippocampus/` or `raw/torus`. The template mesh shape that was generated or selected in the synthetic data generation or preprocessing steps, needs to be stored under the `template/` folder.

To train, validate, and test different models run the following command under the `reconstruction/` directory.

`python main.py`

It will show the training and testing results with default arguments (hyperparameters) reported in the paper. The arguments can be changed and set from the command line according to the line number `20` to `66` from `main.py`. The hyperparameters are trained using `Optuna` and the process is [here](https://github.com/Jakaria08/Explaining_Shape_Variability/blob/test2inhib_test_contrastive_inhibition/src/DeepLearning/compute_canada/guided_vae/reconstruction/main.py#L148)

The testing script can test multiple saved models in a specified directory and the following command is needed.

`python test.py`

Python scripts for metric calculations and visualizations are [here](https://github.com/Jakaria08/Explaining_Shape_Variability/tree/master/utils)

# Trained Model
The trained model for supervised contrastive VAE: [Link](https://drive.google.com/file/d/1M5BCEtANJcPHCGlkGTuCG8A2c3FT1QPt/view?usp=sharing)

# Visualizations
## Torus Data
![total_vis.png](https://github.com/Jakaria08/Explaining_Shape_Variability/blob/master/figures/total_vis.png)

Figure 2: On the left side of the figure, we show the combination of reconstructions and original hippocampus (left and right hippocampus) meshes from the dataset using our proposed model. The dark blue indicates a very small deviation between the reconstruction and the original mesh. On the right side, we show the original hippocampus data.

## Hippocampal Volume Change with Age and Multiple Sclerosis (MS)
![MS_range_vol_corrected.png](https://github.com/Jakaria08/Explaining_Shape_Variability/blob/master/figures/MS_range_vol_corrected.png)

Figure 3: Volume changes (between healthy and MS) are depicted in the first row by the intensity of the blue color and yellow represents the highest change in millimeters. The second row shows the healthy hippocampus. Ages are calculated by mapping the latent values and age range of the subjects of MS.
