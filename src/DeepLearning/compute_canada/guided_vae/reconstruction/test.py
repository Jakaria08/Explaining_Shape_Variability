import torch
from reconstruction import AE
from datasets import MeshData
from utils import utils, DataLoader, mesh_sampling, sap
import numpy as np

device = torch.device('cuda', 0)
# Set the path to the saved model directory
model_path = "/home/jakaria/jakariaTest/results_for_hippo_age/hippocampus/models/293"

# Load the saved model
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

template_fp = "/home/jakaria/jakariaTest/results_for_hippo_age/hippocampus/template/template.ply"
data_fp = "/home/jakaria/jakariaTest/results_for_hippo_age"
test_exp = "bareteeth"
split = "interpolation"

meshdata = MeshData(data_fp,
                    template_fp,
                    split=split,
                    test_exp=test_exp)

test_loader = DataLoader(meshdata.test_dataset, batch_size=1)

ages_predict = []
mesh_predict = []
x_data = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        #print("test...")
        x = data.x.to(device)
        # pred = model(x)
        pred, mu, log_var, re = model(x)
        ages_predict.append(re)
        num_graphs = data.num_graphs

        reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
        reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean
        
        reshaped_pred = reshaped_pred.cpu().numpy()
        mesh_predict.append(reshaped_pred)

        reshaped_x = reshaped_x.cpu().numpy()
        x_data.append(reshaped_x)
        # Save the reshaped prediction as a NumPy array
        #reshaped_pred *= 300
        #reshaped_x *= 300

ages_predict = torch.concat(ages_predict)
torch.save(ages_predict, f"{model_path}ages_predict.pt")

mesh_predict_np = np.array(mesh_predict)
np.save(f"{model_path}mesh_predict.npy", mesh_predict_np)

x_data_np = np.array(x_data)
np.save(f"{model_path}x_data_np.npy", x_data_np)
