# Disentangling Hippocampal Shape Variations: A Study of Neurological Disorders Using Graph Variational Autoencoder with Contrastive Learning
![overall_architecture_update.png](https://github.com/Jakaria08/Explaining_Shape_Variability/blob/master/figures/overall_architecture_update.png)
This repository contains the code and resources for the paper: **Disentangling Hippocampal Shape Variations: A Study of Neurological Disorders Using Graph Variational Autoencoder with Contrastive Learning** [link needed].

## Table of Contents

1. [Code](#code)
    - [Data Processing](#data-processing)
    - [Synthetic Torus Data Generation](#synthetic-torus-data-generation)
2. [Setup and Commands](#setup-and-commands)
3. [Data](#data)
4. [Trained Models](#trained-models)
5. [Supplementary Material](#supplementary-material)
    - [Visualizations](#visualizations)

# Code

The code is organized in the following hierarchy (Only the directory structure is shown without the files for clarity):

```Explaining_Shape_Variability/
├─ doc/
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

# Trained Models

If applicable, provide pre-trained models and instructions on how to use them for prediction or further training.

# Supplementary Material

Include any supplementary material related to the paper, such as additional experiments, analyses, or results.

## Visualization

Include visualizations.

