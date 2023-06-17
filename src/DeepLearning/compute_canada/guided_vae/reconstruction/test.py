import torch
from reconstruction import AE
from datasets import MeshData
from utils import utils, DataLoader, mesh_sampling, sap
import numpy as np
import os
from scipy import stats

device = torch.device('cuda', 0)
# Set the path to the saved model directory
base_path = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models"

# Get a list of all folders within the "models" folder
folders = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]

# Iterate through the folders and create the model paths
for folder_name in folders:
    model_path = os.path.join(base_path, folder_name)

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

    template_fp = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/template/template.ply"
    data_fp = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA"
    test_exp = "bareteeth"
    split = "interpolation"

    meshdata = MeshData(data_fp,
                        template_fp,
                        split=split,
                        test_exp=test_exp)

    test_loader = DataLoader(meshdata.test_dataset, batch_size=1)

    angles = []
    latent_codes = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            #print("test...")
            x = data.x.to(device)
            y = data.y.to(device)
            recon, mu, log_var, re = model(x)
            z = model.reparameterize(mu, log_var)
            latent_codes.append(z)
            angles.append(y)
        latent_codes = torch.concat(latent_codes)
        angles = torch.concat(angles).view(-1,1)

        # Pearson Correlation Coefficient
        pcc = stats.pearsonr(angles.view(-1).cpu().numpy(), latent_codes[:,0].cpu().numpy())[0]
        #print(latent_codes[:,0].cpu().numpy())

        # SAP Score
        sap_score = sap(factors=angles.cpu().numpy(), codes=latent_codes.cpu().numpy(), continuous_factors=True, regression=True)
        
        print("")
        print(f"model: {folder_name}")
        print(f"Correlation: {pcc}")
        print(f"SAP Score:   {sap_score}")
        print("")

        message = 'Model | Correlation | SAP | : {:s}+{:.3f} | {:.3f}'.format(folder_name, pcc,
                                                    sap_score)

        out_error_fp = '/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/reconstruction/test.txt'
        with open(out_error_fp, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
            
            
            # pred = model(x)
            #pred, mu, log_var, re = model(x)
            #ages_predict.append(re)
            #num_graphs = data.num_graphs

            #reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            #reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean
            
            #reshaped_pred = reshaped_pred.cpu().numpy()
            #mesh_predict.append(reshaped_pred)

            #reshaped_x = reshaped_x.cpu().numpy()
            #x_data.append(reshaped_x)
            # Save the reshaped prediction as a NumPy array
            #reshaped_pred *= 300
            #reshaped_x *= 300

    #ages_predict = torch.concat(ages_predict)
    #torch.save(ages_predict, f"{model_path}ages_predict.pt")

    #mesh_predict_np = np.array(mesh_predict)
    #np.save(f"{model_path}mesh_predict.npy", mesh_predict_np)

    #x_data_np = np.array(x_data)
    #np.save(f"{model_path}x_data_np.npy", x_data_np)
