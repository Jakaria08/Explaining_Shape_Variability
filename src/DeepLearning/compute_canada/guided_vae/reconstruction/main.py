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
from torch.utils.data import Subset
import math

parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--exp_name', type=str, default='interpolation_exp')
parser.add_argument('--dataset', type=str, default='CoMA')
parser.add_argument('--split', type=str, default='interpolation')
parser.add_argument('--test_exp', type=str, default='bareteeth')
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=1)

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
parser.add_argument('--delta', type=float, default=0.5)
parser.add_argument('--lambda1', type=float, default=0.5)
parser.add_argument('--lambda2', type=float, default=0.5)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--weight_decay_c', type=float, default=1e-4)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--beta', type=float, default=0)
parser.add_argument('--wcls', type=int, default=1)

# others
parser.add_argument('--correlation_loss', type=bool, default=False)
parser.add_argument('--guided_contrastive_loss', type=bool, default=True)
parser.add_argument('--guided', type=bool, default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--temperature', type=int, default=100)
parser.add_argument('--threshold', type=float, default=0.1)

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


# Calculate total and desired number of data
# this is for taking small subset of data from big dataset
total_data = len(train_loader)*train_loader.batch_size
# i is the percentage of train data
#print("Data Percentage: "+str(i))
desired_data = math.ceil(1.0 * total_data)
#print("desired batches: "+ str(desired_batches))
#print("total batches: " + str(total_batches))
shuffled_indices = list(range(total_data))
random.shuffle(shuffled_indices)

# Use the first 'desired_data' indices to create a subset
subset_indices = shuffled_indices[:desired_data]
# Select desired number of batches according to the percentage of train data
subset_loader = DataLoader(Subset(train_loader.dataset, subset_indices), 
                            batch_size=train_loader.batch_size)

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

def objective(trial):
    '''
    args.threshold = trial.suggest_float('threshold', 0.015, 0.1, step=0.005)
    args.lambda1 = trial.suggest_float('lambda1', 0.05, 0.95, step=0.05)
    args.lambda2 = 1.0 - args.lambda1 
    args.epochs = trial.suggest_int("epochs", 100, 400, step=100)
    args.batch_size = trial.suggest_int("batch_size", 4, 32, 4)
    args.wcls = trial.suggest_int("w_cls", 1, 100)
    args.beta = trial.suggest_float("beta", 0.001, 0.3, log=True)
    args.lr = trial.suggest_float("learning_rate", 0.0001, 0.001, log=True)
    args.lr_decay = trial.suggest_float("learning_rate_decay", 0.70, 0.99, step=0.01)
    args.delta = trial.suggest_float("delta", 0.1, 0.9, step=0.1)
    args.decay_step = trial.suggest_int("decay_step", 1, 50)
    args.latent_channels = trial.suggest_int("latent_channels", 12, 16, step=4)
    args.temperature = trial.suggest_int("temperature", 1, 200, step=20)

    sequence_length = trial.suggest_int("sequence_length", 5, 50)
    args.seq_length = [sequence_length, sequence_length, sequence_length, sequence_length]

    dilation = trial.suggest_int("dilation", 1, 2)
    args.dilation = [dilation, dilation, dilation, dilation]
    
    out_channel = trial.suggest_int("out_channel", 8, 32, 8)
    args.out_channels = [out_channel, out_channel, out_channel, 2*out_channel]
    '''
    args.threshold = 0.025
    args.lambda1 = 0.5
    args.lambda2 = 1.0 - args.lambda1 
    args.epochs = 100
    args.batch_size = 4
    args.wcls = 34
    args.beta = 0.21125819643916915
    args.lr = 0.00019176245642204008
    args.lr_decay = 0.72
    args.delta = 0.30000000000000004
    args.decay_step = 14
    args.latent_channels = 16
    args.temperature = 81

    sequence_length = 30
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
    args.guided_contrastive_loss = True
    args.correlation_loss = False

    run(model, subset_loader, val_loader, args.epochs, optimizer, scheduler,
        writer, device, args.beta, args.wcls, args.guided, args.guided_contrastive_loss, args.correlation_loss, args.latent_channels, args.weight_decay_c, args.temperature, args.delta, args.lambda1, args.lambda2, args.threshold)

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
    excel_file_path_latent = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_con_inhib_su_contrastive/latent_codes.csv"
    excel_file_path_angles = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_con_inhib_su_contrastive/angles.csv"
    excel_file_path_thick = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_con_inhib_su_contrastive/thick.csv"
    # Save the DataFrame to an Excel file
    df.to_csv(excel_file_path_latent, index=False)
    df1.to_csv(excel_file_path_angles, index=False)
    df2.to_csv(excel_file_path_thick, index=False)

    message = 'Latent Channels | Correlation | Correlation R | SAP | Correlation_2 | SAP_2 | Euclidean Distance | Model | :  | {:d} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:d} |'.format(args.latent_channels, pcc, pcc_r,
                                                    sap_score, pcc_thick, sap_score_thick, euclidean_distance, trial.number)


    out_error_fp = '/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_con_inhib_su_contrastive/test.txt'
    with open(out_error_fp, 'a') as log_file:
        log_file.write('{:s}\n'.format(message))

    if sap_score >= 0:
        model_path = f"/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_con_inhib_su_contrastive/{trial.number}/"
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


    return euclidean_distance, sap_score, sap_score_thick

class LogAfterEachTrial:
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        trials = study.trials
        torch.save(trials, "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_con_inhib_su_contrastive/intermediate_trials.pt")

log_trials = LogAfterEachTrial()
study = optuna.create_study(directions=['minimize', 'maximize', 'maximize'])
study.optimize(objective, n_trials=400, callbacks=[log_trials])
