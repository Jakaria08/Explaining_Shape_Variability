import pickle
import argparse
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
from psbody.mesh import Mesh
from scipy import stats
from reconstruction import AE, run, eval_error
from datasets import MeshData
from utils import utils, writer, DataLoader, mesh_sampling, sap, point_biserial_correlation
import optuna
from contextlib import redirect_stdout
import shutil
import random
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--exp_name', type=str, default='interpolation_exp')
parser.add_argument('--dataset', type=str, default='CoMA')
parser.add_argument('--split', type=str, default='interpolation')
parser.add_argument('--test_exp', type=str, default='bareteeth')
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=0)

# network hyperparameters
parser.add_argument('--out_channels', nargs='+', default=[32, 32, 32, 64], type=int)
parser.add_argument('--latent_channels', type=int, default=16)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--weight_decay_c', type=float, default=1e-4)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--beta', type=float, default=0)
parser.add_argument('--wcls', type=int, default=1)

# others
parser.add_argument('--correlation_loss', type=bool, default=True)
parser.add_argument('--guided_contrastive_loss', type=bool, default=False)
parser.add_argument('--guided', type=bool, default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--temperature', type=int, default=100)

args = parser.parse_args()

#args.work_dir = osp.dirname(osp.realpath(__file__))
args.work_dir = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae"
args.data_fp = osp.join(args.work_dir, 'data', args.dataset)
args.out_dir = osp.join(args.work_dir, 'data', 'out', args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
#print(args)

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

writer = writer.Writer(args)
device = torch.device('cuda', args.device_idx)
torch.set_num_threads(args.n_threads)

# deterministic
#torch.use_deterministic_algorithms(True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"

set_seed(args.seed)

# load dataset
print(args.data_fp)
template_fp = osp.join(args.data_fp, 'template', 'template.ply')
print(template_fp)
meshdata = MeshData(args.data_fp,
                    template_fp,
                    split=args.split,
                    test_exp=args.test_exp)
train_loader = DataLoader(meshdata.train_dataset, batch_size=args.batch_size)
val_loader = DataLoader(meshdata.val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(meshdata.test_dataset, batch_size=args.batch_size)

# generate/load transform matrices
transform_fp = osp.join(args.data_fp, 'transform', 'transform.pkl')
if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    mesh = Mesh(filename=template_fp)
    ds_factors = [3, 3, 2, 2]
    _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
        mesh, ds_factors)
    tmp = {
        'vertices': V,
        'face': F,
        'adj': A,
        'down_transform': D,
        'up_transform': U
    }

    with open(transform_fp, 'wb') as fp:
        pickle.dump(tmp, fp)
    print('Done!')
    print('Transform matrices are saved in \'{}\''.format(transform_fp))
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

spiral_indices_list = [
        utils.preprocess_spiral(tmp['face'][idx], args.seq_length[idx],
                                tmp['vertices'][idx],
                                args.dilation[idx]).to(device)
        for idx in range(len(tmp['face']) - 1)
]
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]

# define model and optimizer and set parameters
args.epochs = 400
args.batch_size = 16
args.wcls = 89
args.beta = 0.006076052037305708
args.lr = 0.0004627175800907143
args.lr_decay = 0.75
args.decay_step = 47
args.latent_channels = 12
args.temperature = 141

sequence_length = 6
args.seq_length = [sequence_length, sequence_length, sequence_length, sequence_length]

dilation = 2
args.dilation = [dilation, dilation, dilation, dilation]

out_channel = 24
args.out_channels = [out_channel, out_channel, out_channel, 2*out_channel]
print(args)    

model = AE(args.in_channels, args.out_channels, args.latent_channels,
        spiral_indices_list, down_transform_list,
        up_transform_list).to(device)

print('Number of parameters: {}'.format(utils.count_parameters(model)))
print(model)

optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            args.decay_step,
                                            gamma=args.lr_decay)

args.guided = False
args.guided_contrastive_loss = False
args.correlation_loss = True

for j in range(10, 0, -1):
    run(model, train_loader, val_loader, args.epochs, optimizer, scheduler,
        writer, device, args.beta, args.wcls, args.guided, args.guided_contrastive_loss, 
        args.correlation_loss, args.latent_channels, args.weight_decay_c, args.temperature, 
        j)
    
        # Test metric on train set
    euclidean_distance_train = eval_error(model, train_loader, device, meshdata, args.out_dir)

    angles_train = []
    thick_train = []
    latent_codes_train = []
    re_pre_train = []
    with torch.no_grad():
        for i_train, data_train in enumerate(train_loader):
            x_train = data_train.x.to(device)
            y_train = data_train.y.to(device)
            recon_train, mu_train, log_var_train, re_train, re_2_train = model(x_train)
            z_train = model.reparameterize(mu_train, log_var_train)
            latent_codes_train.append(z_train)
            angles_train.append(y_train[:, :, 0])
            thick_train.append(y_train[:, :, 2]) 
            re_pre_train.append(re_train)
    latent_codes_train = torch.concat(latent_codes_train)
    angles_train = torch.concat(angles_train).view(-1,1)
    thick_train = torch.concat(thick_train).view(-1,1)
    re_pre_train = torch.concat(re_pre_train).view(-1,1)
    latent_codes_train[torch.isnan(latent_codes_train) | torch.isinf(latent_codes_train)] = 0

    # Pearson Correlation Coefficient
    re_pre_train = (re_pre_train.cpu().numpy() >= 0.5).astype(int)
    pcc_train = accuracy_score(angles_train.view(-1).cpu().numpy(), re_pre_train)
    pcc_r_train = point_biserial_correlation(latent_codes_train.cpu().numpy(), angles_train.view(-1).cpu().numpy())
    pcc_thick_train = stats.pearsonr(thick_train.view(-1).cpu().numpy(), latent_codes_train[:,1].cpu().numpy())[0]
    # SAP Score
    sap_score_train = sap(factors=angles_train.cpu().numpy(), codes=latent_codes_train.cpu().numpy(), continuous_factors=False, regression=False)
    sap_score_thick_train = sap(factors=thick_train.cpu().numpy(), codes=latent_codes_train.cpu().numpy(), continuous_factors=True, regression=True)

    message_train = 'Latent Channels | Correlation | Correlation R | SAP | Correlation_2 | SAP_2 | Euclidean Distance | Model | :  | {:d} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:d} |'.format(args.latent_channels, pcc_train, pcc_r_train,
                                                    sap_score_train, pcc_thick_train, sap_score_thick_train, euclidean_distance_train, j)


    out_error_fp_train = '/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_corr/train.txt'
    with open(out_error_fp_train, 'a') as log_file_train:
        log_file_train.write('{:s}\n'.format(message_train))

    #######################################################################################
    # Test metric on test set

    euclidean_distance = eval_error(model, test_loader, device, meshdata, args.out_dir)

    angles = []
    thick = []
    latent_codes = []
    re_pre = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data.x.to(device)
            y = data.y.to(device)
            recon, mu, log_var, re, re_2 = model(x)
            z = model.reparameterize(mu, log_var)
            latent_codes.append(z)
            angles.append(y[:, :, 0])
            thick.append(y[:, :, 2]) 
            re_pre.append(re)
    latent_codes = torch.concat(latent_codes)
    angles = torch.concat(angles).view(-1,1)
    thick = torch.concat(thick).view(-1,1)
    re_pre = torch.concat(re_pre).view(-1,1)
    #print(angles.cpu().numpy().shape)
    #print(latent_codes.cpu().numpy().shape)
    #print(thick.cpu().numpy().shape)
    #print(angles.cpu().numpy())
    #print(latent_codes.cpu().numpy())
    #print(thick.cpu().numpy())
    latent_codes[torch.isnan(latent_codes) | torch.isinf(latent_codes)] = 0
    #print(angles.cpu().numpy())
    #print(latent_codes.cpu().numpy())
    #print(thick.cpu().numpy())

    # Pearson Correlation Coefficient
    re_pre = (re_pre.cpu().numpy() >= 0.5).astype(int)
    pcc = accuracy_score(angles.view(-1).cpu().numpy(), re_pre)
    pcc_r = point_biserial_correlation(latent_codes.cpu().numpy(), angles.view(-1).cpu().numpy())
    pcc_thick = stats.pearsonr(thick.view(-1).cpu().numpy(), latent_codes[:,1].cpu().numpy())[0]
    # SAP Score
    sap_score = sap(factors=angles.cpu().numpy(), codes=latent_codes.cpu().numpy(), continuous_factors=False, regression=False)
    sap_score_thick = sap(factors=thick.cpu().numpy(), codes=latent_codes.cpu().numpy(), continuous_factors=True, regression=True)

    print("")
    print(f"Correlation: {pcc}")
    print(f"Correlation R: {pcc_r}")
    print(f"Correlation thick score: {pcc_thick}")
    print(f"SAP Score:   {sap_score}")
    print(f"SAP Score Label 2:   {sap_score_thick}")
    print("")

    df = pd.DataFrame(latent_codes.cpu().numpy())
    df1 = pd.DataFrame(angles.cpu().numpy())
    df2 = pd.DataFrame(thick.cpu().numpy())
    # File path for saving the data
    excel_file_path_latent = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_corr/latent_codes.csv"
    excel_file_path_angles = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_corr/angles.csv"
    excel_file_path_thick = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_corr/thick.csv"
    # Save the DataFrame to an Excel file
    df.to_csv(excel_file_path_latent, index=False)
    df1.to_csv(excel_file_path_angles, index=False)
    df2.to_csv(excel_file_path_thick, index=False)

    message = 'Latent Channels | Correlation | Correlation R | SAP | Correlation_2 | SAP_2 | Euclidean Distance | Model | :  | {:d} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:d} |'.format(args.latent_channels, pcc, pcc_r,
                                                    sap_score, pcc_thick, sap_score_thick, euclidean_distance, j)


    out_error_fp = '/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_corr/test.txt'
    with open(out_error_fp, 'a') as log_file:
        log_file.write('{:s}\n'.format(message))

    if sap_score >= 0:
        model_path = f"/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_corr/{j}/"
        os.makedirs(model_path)
        torch.save(sap_score, f"{model_path}sap_score.pt") 
        torch.save(sap_score_thick, f"{model_path}sap_score_thick.pt") 

        torch.save(model.state_dict(), f"{model_path}model_state_dict.pt")
        torch.save(args.in_channels, f"{model_path}in_channels.pt")
        torch.save(args.out_channels, f"{model_path}out_channels.pt")
        torch.save(args.latent_channels, f"{model_path}latent_channels.pt")
        torch.save(spiral_indices_list, f"{model_path}spiral_indices_list.pt")
        torch.save(up_transform_list, f"{model_path}up_transform_list.pt")
        torch.save(down_transform_list, f"{model_path}down_transform_list.pt")
        torch.save(meshdata.std, f"{model_path}std.pt")        
        torch.save(meshdata.mean, f"{model_path}mean.pt")        
        torch.save(meshdata.template_face, f"{model_path}faces.pt")
        shutil.copy("/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/processed/train_val_test_files.pt", f"{model_path}train_val_test_files.pt")
        shutil.copy("/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/reconstruction/network.py", f"{model_path}network.py")
        shutil.copy("/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/conv/spiralconv.py", f"{model_path}spiralconv.py")

