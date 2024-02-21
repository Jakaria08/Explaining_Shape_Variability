#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:4
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --mem=125G
#SBATCH --time=10:00:00
#SBATCH --account=def-uofavis-ab
module load cuda cudnn
source ~/ENV/bin/activate
python /home/jakaria/jakariaTest/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/reconstruction/main.py