import argparse
import copy
import math
import os
import os.path as osp
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.backends.cudnn as cudnn
from psbody.mesh import Mesh
from scipy import stats
from torch.utils.data import Subset

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets import MeshData
from reconstruction import AE, eval_error, run
from utils import DataLoader, mesh_sampling, sap, utils


DEFAULT_WORK_DIR = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae"
DEFAULT_MODELS_ROOT = (
    "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/"
    "guided_vae/data/CoMA/raw/calsnic_als/models"
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='mesh autoencoder')

    parser.add_argument('--exp_name', type=str, default='calsnic_optuna_age')
    parser.add_argument('--dataset', type=str, default='CoMA')
    parser.add_argument('--split', type=str, default='interpolation')
    parser.add_argument('--test_exp', type=str, default='bareteeth')
    parser.add_argument('--n_threads', type=int, default=4)
    parser.add_argument('--device_idx', type=int, default=0)

    # network hyperparameters
    parser.add_argument('--out_channels', nargs='+', default=[32, 32, 32, 64], type=int)
    parser.add_argument('--latent_channels', type=int, default=8)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
    parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

    # optimizer / loss hyperparameters
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
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--wcls', type=float, default=1.0)
    parser.add_argument('--covariance_weight', type=float, default=1e-4)

    # objective / labels
    parser.add_argument('--age_label_index', type=int, default=1)
    parser.add_argument('--age_latent_index', type=int, default=1)

    # loss switches (requested setup: reg-SNN + covariance only)
    parser.add_argument('--guided', type=str2bool, default=False)
    parser.add_argument('--guided_contrastive_loss', type=str2bool, default=True)
    parser.add_argument('--correlation_loss', type=str2bool, default=False)
    parser.add_argument('--use_snn_cls', type=str2bool, default=False)
    parser.add_argument('--use_snn_reg', type=str2bool, default=True)
    parser.add_argument('--use_covariance', type=str2bool, default=True)

    # optuna / run control
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--temperature', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--train_subset_fraction', type=float, default=1.0)

    # paths
    parser.add_argument('--work_dir', type=str, default=DEFAULT_WORK_DIR)
    parser.add_argument('--models_root', type=str, default=DEFAULT_MODELS_ROOT)
    parser.add_argument('--trial_metrics_fp', type=str, default='')
    parser.add_argument('--intermediate_trials_fp', type=str, default='')

    return parser.parse_args(argv)


def resolve_paths(args):
    args.data_fp = osp.join(args.work_dir, 'data', args.dataset)
    args.out_dir = osp.join(args.work_dir, 'data', 'out', args.exp_name)
    args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')

    utils.makedirs(args.out_dir)
    utils.makedirs(args.checkpoints_dir)
    utils.makedirs(args.models_root)

    if not args.trial_metrics_fp:
        args.trial_metrics_fp = osp.join(
            args.models_root,
            f'trial_metrics_gpu{args.device_idx}_latent{args.latent_channels}.txt',
        )

    if not args.intermediate_trials_fp:
        args.intermediate_trials_fp = osp.join(
            args.models_root,
            f'intermediate_trials_gpu{args.device_idx}_latent{args.latent_channels}.pt',
        )


def load_or_generate_transform(template_fp, transform_fp):
    if not osp.exists(transform_fp):
        print('Generating transform matrices...')
        mesh = Mesh(filename=template_fp)
        ds_factors = [3, 3, 2, 2]
        _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(mesh, ds_factors)
        tmp = {
            'vertices': V,
            'face': F,
            'adj': A,
            'down_transform': D,
            'up_transform': U,
        }
        with open(transform_fp, 'wb') as fp:
            pickle.dump(tmp, fp)
        print(f"Transform matrices are saved in '{transform_fp}'")
    else:
        with open(transform_fp, 'rb') as f:
            tmp = pickle.load(f, encoding='latin1')
    return tmp


def build_spiral_indices(transform_data, seq_length, dilation, device):
    return [
        utils.preprocess_spiral(
            transform_data['face'][idx],
            seq_length[idx],
            transform_data['vertices'][idx],
            dilation[idx],
        ).to(device)
        for idx in range(len(transform_data['face']) - 1)
    ]


def build_sparse_transforms(transform_data, device):
    down_transform_list = [
        utils.to_sparse(down_transform).to(device)
        for down_transform in transform_data['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform).to(device)
        for up_transform in transform_data['up_transform']
    ]
    return down_transform_list, up_transform_list


def build_loaders(meshdata, batch_size, subset_fraction, seed):
    train_dataset = meshdata.train_dataset
    if subset_fraction < 1.0:
        n_total = len(meshdata.train_dataset)
        n_use = max(1, int(math.ceil(subset_fraction * n_total)))
        rng = random.Random(seed)
        indices = list(range(n_total))
        rng.shuffle(indices)
        train_dataset = Subset(meshdata.train_dataset, indices[:n_use])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(meshdata.val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(meshdata.test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def sanitize_metric(x, fallback=0.0):
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return float(fallback)
    return x


def evaluate_age_metrics(model, test_loader, device, age_label_index, age_latent_index):
    ages = []
    latent_codes = []

    with torch.no_grad():
        for data in test_loader:
            x = data.x.to(device)
            y = data.y.to(device)
            _, mu, log_var, _, _ = model(x)
            z = model.reparameterize(mu, log_var)
            latent_codes.append(z)
            ages.append(y[:, :, age_label_index])

    if len(latent_codes) == 0:
        return 0.0, 0.0

    latent_codes = torch.cat(latent_codes, dim=0)
    ages = torch.cat(ages, dim=0).view(-1, 1)

    latent_codes[torch.isnan(latent_codes) | torch.isinf(latent_codes)] = 0

    latent_np = latent_codes.detach().cpu().numpy()
    age_np = ages.detach().cpu().numpy()

    sap_age = sap(
        factors=age_np,
        codes=latent_np,
        continuous_factors=True,
        regression=True,
    )
    sap_age = sanitize_metric(sap_age, fallback=0.0)

    age_vec = age_np.reshape(-1)
    latent_vec = latent_np[:, age_latent_index]
    if np.std(age_vec) < 1e-12 or np.std(latent_vec) < 1e-12:
        corr_age_latent = 0.0
    else:
        corr_age_latent = stats.pearsonr(age_vec, latent_vec)[0]
    corr_age_latent = sanitize_metric(corr_age_latent, fallback=0.0)

    return sap_age, corr_age_latent


class TrialLogger:
    def __init__(self, metrics_fp, trials_fp):
        self.metrics_fp = metrics_fp
        self.trials_fp = trials_fp
        self._ensure_header()

    def _ensure_header(self):
        out_dir = osp.dirname(self.metrics_fp)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if not osp.exists(self.metrics_fp):
            with open(self.metrics_fp, 'w') as f:
                f.write('trial,euclidean_distance,sap_score_age,correlation_age_latent1\n')

    def __call__(self, study, trial):
        if trial.values is None or len(trial.values) != 3:
            return

        euc, sap_age, corr_age = trial.values
        with open(self.metrics_fp, 'a') as f:
            f.write(
                f"{trial.number},{float(euc):.10f},{float(sap_age):.10f},{float(corr_age):.10f}\n"
            )

        torch.save(study.trials, self.trials_fp)


def create_objective(base_args, meshdata, transform_data, device):
    down_transform_list, up_transform_list = build_sparse_transforms(transform_data, device)

    def objective(trial):
        args = copy.deepcopy(base_args)

        # Trial ranges (latent size is fixed per launcher/file)
        args.threshold = trial.suggest_float('threshold', 0.01, 0.10, step=0.005)
        args.lambda1 = trial.suggest_float('lambda1', 0.10, 0.90, step=0.05)
        args.lambda2 = 1.0 - args.lambda1
        args.epochs = trial.suggest_int('epochs', 80, 220, step=20)
        args.batch_size = trial.suggest_int('batch_size', 4, 24, step=4)
        args.wcls = trial.suggest_float('w_reg_snn', 0.1, 100.0, log=True)
        args.covariance_weight = trial.suggest_float('covariance_weight', 1e-7, 1e-2, log=True)
        args.beta = trial.suggest_float('beta', 1e-4, 0.3, log=True)
        args.lr = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
        args.lr_decay = trial.suggest_float('learning_rate_decay', 0.70, 0.99, step=0.01)
        args.delta = trial.suggest_float('delta', 0.1, 0.9, step=0.1)
        args.decay_step = trial.suggest_int('decay_step', 5, 30)
        args.temperature = trial.suggest_int('temperature', 20, 200, step=10)
        args.weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True)
        args.weight_decay_c = trial.suggest_float('weight_decay_c', 1e-6, 1e-3, log=True)

        sequence_length = trial.suggest_int('sequence_length', 20, 40, step=2)
        args.seq_length = [sequence_length, sequence_length, sequence_length, sequence_length]

        dilation = trial.suggest_int('dilation', 1, 2)
        args.dilation = [dilation, dilation, dilation, dilation]

        out_channel = trial.suggest_int('out_channel', 16, 48, step=8)
        args.out_channels = [out_channel, out_channel, out_channel, 2 * out_channel]

        train_loader, val_loader, test_loader = build_loaders(
            meshdata=meshdata,
            batch_size=args.batch_size,
            subset_fraction=args.train_subset_fraction,
            seed=args.seed + trial.number,
        )

        spiral_indices_list = build_spiral_indices(
            transform_data=transform_data,
            seq_length=args.seq_length,
            dilation=args.dilation,
            device=device,
        )

        model = AE(
            args.in_channels,
            args.out_channels,
            args.latent_channels,
            spiral_indices_list,
            down_transform_list,
            up_transform_list,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            args.decay_step,
            gamma=args.lr_decay,
        )

        # No per-epoch checkpointing here to keep Optuna trials light.
        run(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            epochs=args.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            writer=None,
            device=device,
            beta=args.beta,
            w_cls=args.wcls,
            guided=args.guided,
            guided_contrastive_loss=args.guided_contrastive_loss,
            correlation_loss=args.correlation_loss,
            latent_channels=args.latent_channels,
            weight_decay_c=args.weight_decay_c,
            temp=args.temperature,
            delta=args.delta,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            threshold=args.threshold,
            age_label_index=args.age_label_index,
            use_snn_cls=args.use_snn_cls,
            use_snn_reg=args.use_snn_reg,
            use_covariance=args.use_covariance,
            covariance_weight=args.covariance_weight,
            save_checkpoints=False,
        )

        euclidean_distance = eval_error(model, test_loader, device, meshdata, args.out_dir)
        euclidean_distance = sanitize_metric(euclidean_distance, fallback=1e9)

        sap_age, corr_age_latent = evaluate_age_metrics(
            model=model,
            test_loader=test_loader,
            device=device,
            age_label_index=args.age_label_index,
            age_latent_index=args.age_latent_index,
        )

        print('')
        print(f"Trial {trial.number} | latent={args.latent_channels} | gpu={args.device_idx}")
        print(f"Euclidean Distance: {euclidean_distance:.6f}")
        print(f"SAP Score (Age):   {sap_age:.6f}")
        print(f"Corr(Age, z[{args.age_latent_index}]): {corr_age_latent:.6f}")
        print('')

        return euclidean_distance, sap_age, corr_age_latent

    return objective


def main(argv=None):
    args = parse_args(argv)

    if args.latent_channels not in {8, 12, 16}:
        raise ValueError(
            f"latent_channels must be one of {{8, 12, 16}} for this setup, got {args.latent_channels}."
        )

    resolve_paths(args)

    device = torch.device('cuda', args.device_idx)
    torch.set_num_threads(args.n_threads)

    set_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    print(args.data_fp)
    template_fp = osp.join(args.data_fp, 'template', 'template.ply')
    print(template_fp)

    meshdata = MeshData(
        args.data_fp,
        template_fp,
        split=args.split,
        test_exp=args.test_exp,
    )

    transform_fp = osp.join(args.data_fp, 'transform', 'transform.pkl')
    transform_data = load_or_generate_transform(template_fp, transform_fp)

    objective = create_objective(
        base_args=args,
        meshdata=meshdata,
        transform_data=transform_data,
        device=device,
    )

    trial_logger = TrialLogger(
        metrics_fp=args.trial_metrics_fp,
        trials_fp=args.intermediate_trials_fp,
    )

    study = optuna.create_study(directions=['minimize', 'maximize', 'maximize'])
    study.optimize(objective, n_trials=args.n_trials, callbacks=[trial_logger])

    torch.save(study.trials, args.intermediate_trials_fp)


if __name__ == '__main__':
    main()
