import torch
from reconstruction import AE
from datasets import MeshData
from utils import utils, DataLoader, sap
import numpy as np
import pyvista as pv
from IPython.display import display
import os, sys
from math import ceil
from scipy.ndimage import zoom
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import tqdm
import geomloss
from pytorch3d.loss import chamfer_distance


device = torch.device('cuda', 1)
# Set the path to the saved model directory
#model_path = "/home/jakaria/torus_bump_500_three_scale_binary_bump_variable_noise_fixed_angle/models_classification_regression_only_correlation_loss/models/65"
#model_path = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_contrastive_inhib/146"
#model_path = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_guided/30"# Load the saved model
model_path = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_attribute/23"
model_state_dict = torch.load(f"{model_path}/model_state_dict.pt")
in_channels = torch.load(f"{model_path}/in_channels.pt")
out_channels = torch.load(f"{model_path}/out_channels.pt")
latent_channels = torch.load(f"{model_path}/latent_channels.pt")
spiral_indices_list = torch.load(f"{model_path}/spiral_indices_list.pt")
up_transform_list = torch.load(f"{model_path}/up_transform_list.pt")
down_transform_list = torch.load(f"{model_path}/down_transform_list.pt")
std = torch.load(f"{model_path}/std.pt")
mean = torch.load(f"{model_path}/mean.pt")
template_face = torch.load(f"{model_path}/faces.pt")

# Create an instance of the model
model = AE(in_channels, out_channels, latent_channels,
           spiral_indices_list, down_transform_list,
           up_transform_list)
model.load_state_dict(model_state_dict)
model.to(device)
# Set the model to evaluation mode
model.eval()