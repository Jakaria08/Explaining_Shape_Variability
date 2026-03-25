import argparse
import copy
import csv
import datetime
import json
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
from optuna.trial import TrialState
from psbody.mesh import Mesh
from scipy import stats
from torch.utils.data import Subset

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_PACKAGE_PARENT = _PROJECT_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from datasets import MeshData
from reconstruction import AE, AEModelParallel, eval_error, run
from guided_vae.utils.dataloader import DataLoader
from guided_vae.utils import mesh_sampling
from guided_vae.utils import utils as gv_utils
from guided_vae.utils.sap_holdout import sap_regression_holdout


DEFAULT_WORK_DIR = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae"
DEFAULT_MODELS_ROOT = (
    "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/"
    "guided_vae/data/CoMA/raw/calsnic_als/models"
)
SUPPORTED_CONV_TYPES = ('spiral', 'adaptive_spiral')


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
    parser.add_argument('--parallel_mode', type=str, default='model', choices=['single', 'model'])
    parser.add_argument('--model_parallel_gpus', type=str, default='0,1,2')

    # network hyperparameters
    parser.add_argument('--out_channels', nargs='+', default=[32, 32, 32, 64], type=int)
    parser.add_argument('--latent_channels', type=int, default=256)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
    parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')
    parser.add_argument(
        '--conv_type',
        type=str,
        default='spiral',
        choices=list(SUPPORTED_CONV_TYPES),
    )
    parser.add_argument('--optimize_conv_type', type=str2bool, default=False)
    parser.add_argument('--adaptive_hidden', type=int, default=32)
    parser.add_argument('--adaptive_dropout', type=float, default=0.0)
    parser.add_argument('--adaptive_global_context', type=str2bool, default=False)
    parser.add_argument(
        '--optuna_stage',
        type=str,
        default='single',
        choices=['single', 'stage1', 'stage2'],
    )
    parser.add_argument(
        '--stage1_latent_choices',
        nargs='+',
        type=int,
        default=[8, 16, 32, 64, 128],
    )
    parser.add_argument('--stage1_summary_fp', type=str, default='')
    parser.add_argument('--stage2_latent_override', type=int, default=-1)

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
    parser.add_argument('--age_latent_index', type=int, default=0)

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
    parser.add_argument('--print_epoch_objectives', type=str2bool, default=True)
    parser.add_argument('--epoch_objective_interval', type=int, default=10)
    parser.add_argument('--optimize_latent', type=str2bool, default=False)
    parser.add_argument('--latent_choices', nargs='+', type=int, default=[256])
    parser.add_argument('--study_name', type=str, default='')
    parser.add_argument('--storage', type=str, default='')

    # paths
    parser.add_argument('--work_dir', type=str, default=DEFAULT_WORK_DIR)
    parser.add_argument('--models_root', type=str, default=DEFAULT_MODELS_ROOT)
    parser.add_argument('--trial_metrics_fp', type=str, default='')
    parser.add_argument('--intermediate_trials_fp', type=str, default='')
    parser.add_argument('--trial_predictions_dir', type=str, default='')
    parser.add_argument('--trial_models_dir', type=str, default='')
    parser.add_argument('--study_summary_fp', type=str, default='')
    parser.add_argument('--study_trials_csv_fp', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='')
    parser.add_argument('--log_fp', type=str, default='')

    return parser.parse_args(argv)


def _validate_latent_choices(choices):
    if len(choices) == 0:
        raise ValueError('latent_choices cannot be empty when optimize_latent is enabled')
    invalid = [int(c) for c in choices if int(c) < 2]
    if invalid:
        raise ValueError(
            f'latent_choices must be integers >= 2; got invalid values {sorted(set(invalid))}'
        )


def resolve_latent_config(args):
    if args.optimize_latent:
        choices = sorted({int(c) for c in args.latent_choices})
        _validate_latent_choices(choices)
        args.latent_choices = choices
        args.max_latent_channels = max(choices)
        args.min_latent_channels = min(choices)
        args.latent_tag = 'search'
    else:
        if int(args.latent_channels) < 2:
            raise ValueError(
                f'latent_channels must be >= 2, got {args.latent_channels}.'
            )
        args.latent_choices = [int(args.latent_channels)]
        args.max_latent_channels = int(args.latent_channels)
        args.min_latent_channels = int(args.latent_channels)
        args.latent_tag = str(args.latent_channels)


def parse_gpu_ids(gpu_text):
    parts = [p.strip() for p in str(gpu_text).split(',') if p.strip()]
    if len(parts) == 0:
        raise ValueError('model_parallel_gpus cannot be empty.')
    gpu_ids = []
    for part in parts:
        gpu_ids.append(int(part))
    return gpu_ids


def resolve_parallel_config(args):
    args.parallel_mode = str(args.parallel_mode).strip().lower()
    if args.parallel_mode not in {'single', 'model'}:
        raise ValueError("parallel_mode must be one of {'single', 'model'}.")

    if args.parallel_mode == 'model':
        gpu_ids = parse_gpu_ids(args.model_parallel_gpus)
        if len(gpu_ids) < 2:
            raise ValueError('Model-parallel mode requires at least 2 GPUs.')
        args.model_parallel_gpu_ids = gpu_ids
        args.device_idx = int(gpu_ids[0])
        args.gpu_tag = 'g' + '-'.join(str(i) for i in gpu_ids)
    else:
        args.model_parallel_gpu_ids = [int(args.device_idx)]
        args.gpu_tag = f'g{int(args.device_idx)}'


def resolve_conv_config(args):
    args.conv_type = str(args.conv_type).strip().lower()
    if args.conv_type not in SUPPORTED_CONV_TYPES:
        raise ValueError(
            f"conv_type must be one of {SUPPORTED_CONV_TYPES}, got '{args.conv_type}'."
        )
    if int(args.adaptive_hidden) < 1:
        raise ValueError('adaptive_hidden must be >= 1.')
    if float(args.adaptive_dropout) < 0.0 or float(args.adaptive_dropout) >= 1.0:
        raise ValueError('adaptive_dropout must be in [0, 1).')
    args.conv_tag = 'convsearch' if args.optimize_conv_type else args.conv_type


def _extract_latent_from_summary(summary_dict):
    candidate_paths = [
        ('best_by_distance_metric',),
        ('best_by_distance',),
        ('best_by_objective', 0),
    ]

    for path in candidate_paths:
        node = summary_dict
        valid = True
        for key in path:
            if isinstance(key, int):
                if not isinstance(node, list) or len(node) <= key:
                    valid = False
                    break
                node = node[key]
            else:
                if not isinstance(node, dict) or key not in node:
                    valid = False
                    break
                node = node[key]
        if not valid or not isinstance(node, dict):
            continue

        params = node.get('params', {})
        user_attrs = node.get('user_attrs', {})
        latent = params.get('latent_channels', user_attrs.get('latent_channels', None))
        if latent is not None:
            latent = int(latent)
            if latent >= 2:
                return latent

    raise ValueError(
        'Could not extract latent_channels from stage-1 summary. '
        'Expected keys like best_by_distance_metric / best_by_distance / best_by_objective.'
    )


def _default_stage1_summary_fp(args):
    return osp.join(
        args.models_root,
        f'study_summary_{args.gpu_tag}_{args.conv_tag}_stage1_latentstage1search.json',
    )


def resolve_stage_config(args):
    args.optuna_stage = str(args.optuna_stage).strip().lower()
    if args.optuna_stage not in {'single', 'stage1', 'stage2'}:
        raise ValueError("optuna_stage must be one of {'single','stage1','stage2'}.")

    args.stage_tag = args.optuna_stage

    if args.optuna_stage == 'single':
        args.use_stage1_defaults = False
        args.use_stage2_defaults = False
        return

    if args.optuna_stage == 'stage1':
        stage1_choices = sorted({int(x) for x in args.stage1_latent_choices})
        _validate_latent_choices(stage1_choices)

        args.optimize_latent = True
        args.latent_choices = stage1_choices
        args.latent_tag = 'stage1search'

        # Stage-1 requirement: only latent is tuned.
        args.optimize_conv_type = False
        args.conv_tag = args.conv_type
        args.use_stage1_defaults = True
        args.use_stage2_defaults = False
        return

    # stage2
    args.optimize_latent = False
    args.optimize_conv_type = False
    args.conv_tag = args.conv_type
    args.use_stage1_defaults = False
    args.use_stage2_defaults = True

    if int(args.stage2_latent_override) >= 2:
        args.latent_channels = int(args.stage2_latent_override)
        return

    summary_fp = args.stage1_summary_fp or _default_stage1_summary_fp(args)
    if not osp.exists(summary_fp):
        raise FileNotFoundError(
            'Stage-2 requires stage-1 best latent. '
            f'Summary file not found: {summary_fp}. '
            'Pass --stage1_summary_fp or --stage2_latent_override.'
        )
    with open(summary_fp, 'r') as f:
        stage1_summary = json.load(f)
    args.latent_channels = _extract_latent_from_summary(stage1_summary)
    args.stage1_summary_fp = summary_fp


def resolve_paths(args):
    args.data_fp = osp.join(args.work_dir, 'data', args.dataset)
    args.out_dir = osp.join(args.work_dir, 'data', 'out', args.exp_name)
    args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')

    gv_utils.makedirs(args.out_dir)
    gv_utils.makedirs(args.checkpoints_dir)
    gv_utils.makedirs(args.models_root)

    if not args.trial_metrics_fp:
        args.trial_metrics_fp = osp.join(
            args.models_root,
            f'trial_metrics_{args.gpu_tag}_{args.conv_tag}_{args.stage_tag}_latent{args.latent_tag}.csv',
        )

    if not args.intermediate_trials_fp:
        args.intermediate_trials_fp = osp.join(
            args.models_root,
            f'intermediate_trials_{args.gpu_tag}_{args.conv_tag}_{args.stage_tag}_latent{args.latent_tag}.pt',
        )

    if not args.trial_predictions_dir:
        args.trial_predictions_dir = osp.join(
            args.models_root,
            f'trial_predictions_{args.gpu_tag}_{args.conv_tag}_{args.stage_tag}_latent{args.latent_tag}',
        )

    if not args.trial_models_dir:
        args.trial_models_dir = osp.join(
            args.models_root,
            f'trial_models_{args.gpu_tag}_{args.conv_tag}_{args.stage_tag}_latent{args.latent_tag}',
        )

    if not args.study_summary_fp:
        args.study_summary_fp = osp.join(
            args.models_root,
            f'study_summary_{args.gpu_tag}_{args.conv_tag}_{args.stage_tag}_latent{args.latent_tag}.json',
        )

    if not args.study_trials_csv_fp:
        args.study_trials_csv_fp = osp.join(
            args.models_root,
            f'study_trials_{args.gpu_tag}_{args.conv_tag}_{args.stage_tag}_latent{args.latent_tag}.csv',
        )

    if not args.log_dir:
        args.log_dir = osp.join(args.models_root, 'logs')

    if not args.log_fp:
        args.log_fp = osp.join(
            args.log_dir,
            f'{args.exp_name}_{args.gpu_tag}_{args.conv_tag}_{args.stage_tag}_latent{args.latent_tag}.log',
        )

    gv_utils.makedirs(args.trial_predictions_dir)
    gv_utils.makedirs(args.trial_models_dir)
    gv_utils.makedirs(args.log_dir)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        for stream in self.streams:
            if hasattr(stream, 'isatty') and stream.isatty():
                return True
        return False


def setup_logging(log_fp):
    log_dir = osp.dirname(log_fp)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    log_handle = open(log_fp, 'a', buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)

    print('=' * 80)
    print(f"Log file: {log_fp}")
    print(
        "Run start: "
        f"{datetime.datetime.now().isoformat(timespec='seconds')}"
    )
    print('=' * 80)

    return log_handle, original_stdout, original_stderr


def teardown_logging(log_handle, original_stdout, original_stderr):
    try:
        print('=' * 80)
        print(
            "Run end: "
            f"{datetime.datetime.now().isoformat(timespec='seconds')}"
        )
        print('=' * 80)
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


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
    spirals = [
        gv_utils.preprocess_spiral(
            transform_data['face'][idx],
            seq_length[idx],
            transform_data['vertices'][idx],
            dilation[idx],
        )
        for idx in range(len(transform_data['face']) - 1)
    ]
    if device is None:
        return spirals
    return [s.to(device) for s in spirals]


def build_sparse_transforms(transform_data, device):
    down_transform_list = [
        gv_utils.to_sparse(down_transform)
        for down_transform in transform_data['down_transform']
    ]
    up_transform_list = [
        gv_utils.to_sparse(up_transform)
        for up_transform in transform_data['up_transform']
    ]
    if device is not None:
        down_transform_list = [t.to(device) for t in down_transform_list]
        up_transform_list = [t.to(device) for t in up_transform_list]
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
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(meshdata.val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, train_eval_loader, val_loader


def sanitize_metric(x, fallback=0.0):
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return float(fallback)
    return x


def get_data_device(model, fallback_device):
    model_device = getattr(model, 'input_device', None)
    if model_device is not None:
        return model_device
    return fallback_device


def compute_euclidean_distance(model, data_loader, device, meshdata):
    model.eval()
    model.training = False

    errors = []
    mean = meshdata.mean
    std = meshdata.std

    with torch.no_grad():
        for data in data_loader:
            x = data.x.to(device)
            pred, _, _, _, _ = model(x)
            num_graphs = data.num_graphs

            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean

            reshaped_pred *= 300
            reshaped_x *= 300

            tmp_error = torch.sqrt(
                torch.sum((reshaped_pred - reshaped_x) ** 2, dim=2)
            )
            errors.append(tmp_error)

    if len(errors) == 0:
        return 1e9

    new_errors = torch.cat(errors, dim=0)
    mean_error = new_errors.view((-1,)).mean().item()
    return sanitize_metric(mean_error, fallback=1e9)


def collect_latents_and_age(model, loader, device, age_label_index):
    ages = []
    latent_codes = []

    model.eval()
    model.training = False

    with torch.no_grad():
        for data in loader:
            x = data.x.to(device)
            y = data.y.to(device)
            _, mu, log_var, _, _ = model(x)
            z = model.reparameterize(mu, log_var)
            latent_codes.append(z.detach().cpu())
            ages.append(y[:, :, age_label_index].view(-1, 1).detach().cpu())

    if len(latent_codes) == 0:
        return (
            np.zeros((0, model.latent_channels), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
        )

    latent_tensor = torch.cat(latent_codes, dim=0)
    age_tensor = torch.cat(ages, dim=0)

    latent_tensor[torch.isnan(latent_tensor) | torch.isinf(latent_tensor)] = 0

    return latent_tensor.numpy(), age_tensor.numpy()


def compute_corr_per_latent(latent_np, age_np):
    age_vec = age_np.reshape(-1)
    corrs = []

    for i in range(latent_np.shape[1]):
        latent_vec = latent_np[:, i]
        if np.std(age_vec) < 1e-12 or np.std(latent_vec) < 1e-12:
            corr = 0.0
        else:
            corr = stats.pearsonr(age_vec, latent_vec)[0]
        corrs.append(sanitize_metric(corr, fallback=0.0))

    return corrs


def evaluate_age_metrics_holdout(
    model,
    train_eval_loader,
    eval_loader,
    device,
    age_label_index,
    age_latent_index,
):
    train_latent_np, train_age_np = collect_latents_and_age(
        model=model,
        loader=train_eval_loader,
        device=device,
        age_label_index=age_label_index,
    )
    eval_latent_np, eval_age_np = collect_latents_and_age(
        model=model,
        loader=eval_loader,
        device=device,
        age_label_index=age_label_index,
    )

    latent_dim = model.latent_channels

    if train_latent_np.shape[0] == 0 or eval_latent_np.shape[0] == 0:
        return {
            'sap_age': 0.0,
            'corr_target': 0.0,
            'corr_per_latent': [0.0] * latent_dim,
            'r2_per_latent': [0.0] * latent_dim,
            'age_true_eval': eval_age_np.reshape(-1),
            'age_pred_by_latent': np.zeros((latent_dim, eval_latent_np.shape[0]), dtype=np.float32),
        }

    sap_age, s_matrix, pred_matrix = sap_regression_holdout(
        train_factors=train_age_np,
        train_codes=train_latent_np,
        eval_factors=eval_age_np,
        eval_codes=eval_latent_np,
    )
    sap_age = sanitize_metric(sap_age, fallback=0.0)

    corr_per_latent = compute_corr_per_latent(eval_latent_np, eval_age_np)
    corr_per_latent = [sanitize_metric(c, fallback=0.0) for c in corr_per_latent]

    r2_per_latent = []
    for i in range(latent_dim):
        r2_per_latent.append(sanitize_metric(s_matrix[0, i], fallback=0.0))

    if age_latent_index < 0 or age_latent_index >= latent_dim:
        corr_target_raw = 0.0
    else:
        corr_target_raw = corr_per_latent[age_latent_index]
    corr_target = abs(corr_target_raw)

    return {
        'sap_age': sap_age,
        'corr_target': corr_target,
        'corr_target_raw': corr_target_raw,
        'corr_per_latent': corr_per_latent,
        'r2_per_latent': r2_per_latent,
        'age_true_eval': eval_age_np.reshape(-1),
        'age_pred_by_latent': pred_matrix[0],
    }


class TrialLogger:
    def __init__(self, metrics_fp, trials_fp, latent_channels):
        self.metrics_fp = metrics_fp
        self.trials_fp = trials_fp
        self.latent_channels = int(latent_channels)
        self._ensure_header()

    def _ensure_header(self):
        out_dir = osp.dirname(self.metrics_fp)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if not osp.exists(self.metrics_fp):
            base_cols = [
                'trial',
                'stage',
                'objective_split',
                'latent_channels',
                'conv_type',
                'adaptive_hidden',
                'adaptive_dropout',
                'adaptive_global_context',
                'objective_value_0',
                'objective_value_1',
                'objective_value_2',
                'euclidean_distance',
                'sap_score_age_holdout',
                'correlation_age_target_latent',
                'correlation_age_target_latent_raw',
            ]
            corr_cols = [f'corr_age_latent_{i}' for i in range(self.latent_channels)]
            r2_cols = [f'r2_age_from_latent_{i}' for i in range(self.latent_channels)]
            extra_cols = ['prediction_fp', 'model_fp']
            with open(self.metrics_fp, 'w') as f:
                f.write(','.join(base_cols + corr_cols + r2_cols + extra_cols) + '\n')

    @staticmethod
    def _normalize_vector(values, size, fallback=0.0):
        out = []
        values = list(values) if values is not None else []
        for i in range(size):
            if i < len(values):
                out.append(sanitize_metric(values[i], fallback=fallback))
            else:
                out.append(float(fallback))
        return out

    @staticmethod
    def _f(v):
        return f"{float(v):.10f}"

    def __call__(self, study, trial):
        if trial.values is None or len(trial.values) == 0:
            return

        values = [sanitize_metric(v, fallback=float('nan')) for v in trial.values]
        while len(values) < 3:
            values.append(float('nan'))

        corr_per_latent = self._normalize_vector(
            trial.user_attrs.get('corr_per_latent', []),
            self.latent_channels,
            fallback=0.0,
        )
        r2_per_latent = self._normalize_vector(
            trial.user_attrs.get('r2_per_latent', []),
            self.latent_channels,
            fallback=0.0,
        )
        prediction_fp = str(trial.user_attrs.get('prediction_fp', ''))
        model_fp = str(trial.user_attrs.get('model_fp', ''))
        objective_split = str(trial.user_attrs.get('objective_split', 'val'))
        stage = str(trial.user_attrs.get('stage', 'single'))

        latent_channels_used = int(
            trial.params.get('latent_channels', trial.user_attrs.get('latent_channels', self.latent_channels))
        )
        conv_type = str(trial.user_attrs.get('conv_type', trial.params.get('conv_type', 'spiral')))
        adaptive_hidden = int(trial.user_attrs.get('adaptive_hidden', 0))
        adaptive_dropout = float(trial.user_attrs.get('adaptive_dropout', 0.0))
        adaptive_global_context = bool(trial.user_attrs.get('adaptive_global_context', False))
        euc = sanitize_metric(
            trial.user_attrs.get('euclidean_distance', float('nan')),
            fallback=float('nan'),
        )
        sap_age = sanitize_metric(
            trial.user_attrs.get('sap_age_holdout', float('nan')),
            fallback=float('nan'),
        )
        corr_target = sanitize_metric(
            trial.user_attrs.get('corr_target_abs', float('nan')),
            fallback=float('nan'),
        )
        corr_target_raw = sanitize_metric(
            trial.user_attrs.get('corr_target_raw', float('nan')),
            fallback=float('nan'),
        )

        row = [
            str(trial.number),
            stage,
            objective_split,
            str(latent_channels_used),
            conv_type,
            str(adaptive_hidden),
            self._f(adaptive_dropout),
            str(int(adaptive_global_context)),
            self._f(values[0]),
            self._f(values[1]),
            self._f(values[2]),
            self._f(euc),
            self._f(sap_age),
            self._f(corr_target),
            self._f(corr_target_raw),
        ]
        row.extend([self._f(v) for v in corr_per_latent])
        row.extend([self._f(v) for v in r2_per_latent])
        row.append(prediction_fp)
        row.append(model_fp)

        with open(self.metrics_fp, 'a') as f:
            f.write(','.join(row) + '\n')

        torch.save(study.trials, self.trials_fp)


def _trial_to_summary(trial):
    values = []
    if trial.values is not None:
        values = [float(v) for v in trial.values]
    return {
        'number': int(trial.number),
        'values': values,
        'params': trial.params,
        'user_attrs': trial.user_attrs,
    }


def save_study_artifacts(study, args):
    torch.save(study.trials, args.intermediate_trials_fp)
    n_objectives = len(study.directions)

    complete_trials = [
        t for t in study.trials
        if t.state == TrialState.COMPLETE and t.values is not None and len(t.values) == n_objectives
    ]

    direction_names = [
        d.name.lower() if hasattr(d, 'name') else str(d).split('.')[-1].lower()
        for d in study.directions
    ]

    summary = {
        'stage': args.optuna_stage,
        'objective_split': 'val',
        'directions': direction_names,
        'n_trials_total': len(study.trials),
        'n_trials_complete': len(complete_trials),
        'best_by_objective': [],
        'best_by_distance': None,
        'best_by_sap': None,
        'best_by_corr_target': None,
        'pareto_trials': [],
    }

    if complete_trials:
        for idx, direction in enumerate(direction_names):
            if direction == 'minimize':
                best_obj = min(complete_trials, key=lambda t: t.values[idx])
            else:
                best_obj = max(complete_trials, key=lambda t: t.values[idx])
            summary['best_by_objective'].append(
                {
                    'objective_index': int(idx),
                    'direction': direction,
                    'trial': _trial_to_summary(best_obj),
                }
            )

        best_by_distance = min(
            complete_trials,
            key=lambda t: sanitize_metric(t.user_attrs.get('euclidean_distance', float('inf')), fallback=float('inf')),
        )
        best_by_sap = max(
            complete_trials,
            key=lambda t: sanitize_metric(t.user_attrs.get('sap_age_holdout', float('-inf')), fallback=float('-inf')),
        )
        best_by_corr = max(
            complete_trials,
            key=lambda t: sanitize_metric(t.user_attrs.get('corr_target_abs', float('-inf')), fallback=float('-inf')),
        )

        summary['best_by_distance'] = _trial_to_summary(best_by_distance)
        summary['best_by_sap'] = _trial_to_summary(best_by_sap)
        summary['best_by_corr_target'] = _trial_to_summary(best_by_corr)

        if n_objectives > 1:
            pareto_trials = study.best_trials
        else:
            pareto_trials = [study.best_trial]
        for t in pareto_trials:
            summary['pareto_trials'].append(_trial_to_summary(t))

    with open(args.study_summary_fp, 'w') as f:
        json.dump(summary, f, indent=2)

    with open(args.study_trials_csv_fp, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'trial',
            'state',
            'stage',
            'value_0',
            'value_1',
            'value_2',
            'params_json',
            'user_attrs_json',
        ])
        for t in study.trials:
            values = t.values if t.values is not None else [None, None, None]
            if len(values) < 3:
                values = list(values) + [None] * (3 - len(values))
            writer.writerow([
                t.number,
                str(t.state),
                args.optuna_stage,
                values[0],
                values[1],
                values[2],
                json.dumps(t.params),
                json.dumps(t.user_attrs),
            ])


def create_objective(base_args, meshdata, transform_data, primary_device):
    transform_device = None if base_args.parallel_mode == 'model' else primary_device
    down_transform_list, up_transform_list = build_sparse_transforms(
        transform_data,
        transform_device,
    )

    def objective(trial):
        args = copy.deepcopy(base_args)
        stage = args.optuna_stage
        args.conv_type = str(args.conv_type).strip().lower()
        args.force_deterministic_latent = False

        if stage == 'stage1':
            # Stage-1: only latent is tuned; reconstruction-only optimization.
            args.latent_channels = int(
                trial.suggest_categorical('latent_channels', args.latent_choices)
            )
            args.force_deterministic_latent = True
            args.beta = 0.0
            args.guided = False
            args.guided_contrastive_loss = False
            args.correlation_loss = False
            args.use_snn_cls = False
            args.use_snn_reg = False
            args.use_covariance = False
            args.wcls = 0.0
            args.covariance_weight = 0.0
        elif stage == 'stage2':
            # Stage-2: latent is fixed; only disentanglement-related params are tuned.
            args.latent_channels = int(args.latent_channels)
            args.beta = trial.suggest_float('beta', 1e-4, 0.3, log=True)
            args.wcls = trial.suggest_float('w_reg_snn', 0.1, 100.0, log=True)
            args.covariance_weight = trial.suggest_float('covariance_weight', 1e-7, 1e-2, log=True)
            args.temperature = trial.suggest_int('temperature', 20, 200, step=10)
            args.threshold = trial.suggest_float('threshold', 0.01, 0.10, step=0.005)
            args.guided = False
            args.guided_contrastive_loss = True
            args.correlation_loss = False
            args.use_snn_cls = False
            args.use_snn_reg = True
            args.use_covariance = True
        else:
            # Original single-stage search.
            if args.optimize_latent:
                args.latent_channels = int(
                    trial.suggest_categorical('latent_channels', args.latent_choices)
                )
            else:
                args.latent_channels = int(args.latent_channels)

            if args.optimize_conv_type:
                args.conv_type = trial.suggest_categorical(
                    'conv_type',
                    list(SUPPORTED_CONV_TYPES),
                )

            args.threshold = trial.suggest_float('threshold', 0.01, 0.10, step=0.005)
            args.epochs = trial.suggest_int('epochs', 80, 220, step=20)

            # Memory-aware ranges to reduce OOM likelihood for larger latent sizes.
            if args.latent_channels >= 256:
                batch_min = 1
                batch_max = 2
                batch_step = 1
                out_max = 24
                seq_max = 28
            elif args.latent_channels >= 128:
                batch_min = 1
                batch_max = 4
                batch_step = 1
                out_max = 24
                seq_max = 30
            elif args.latent_channels >= 32:
                batch_min = 2
                batch_max = 8
                batch_step = 2
                out_max = 32
                seq_max = 34
            elif args.latent_channels >= 16:
                batch_min = 4
                batch_max = 16
                batch_step = 4
                out_max = 40
                seq_max = 38
            else:
                batch_min = 4
                batch_max = 24
                batch_step = 4
                out_max = 48
                seq_max = 40

            args.batch_size = trial.suggest_int('batch_size', batch_min, batch_max, step=batch_step)
            args.wcls = trial.suggest_float('w_reg_snn', 0.1, 100.0, log=True)
            args.covariance_weight = trial.suggest_float('covariance_weight', 1e-7, 1e-2, log=True)
            args.beta = trial.suggest_float('beta', 1e-4, 0.3, log=True)
            args.lr = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
            args.lr_decay = trial.suggest_float('learning_rate_decay', 0.70, 0.99, step=0.01)
            args.decay_step = trial.suggest_int('decay_step', 5, 30)
            args.temperature = trial.suggest_int('temperature', 20, 200, step=10)
            args.weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True)

            sequence_length = trial.suggest_int('sequence_length', 20, seq_max, step=2)
            args.seq_length = [sequence_length, sequence_length, sequence_length, sequence_length]

            dilation = trial.suggest_int('dilation', 1, 2)
            args.dilation = [dilation, dilation, dilation, dilation]

            out_channel = trial.suggest_int('out_channel', 16, out_max, step=8)
            args.out_channels = [out_channel, out_channel, out_channel, 2 * out_channel]

        if args.age_latent_index < 0 or args.age_latent_index >= args.latent_channels:
            raise ValueError(
                f"age_latent_index={args.age_latent_index} must be in [0, {args.latent_channels - 1}]"
            )

        print(
            f"Starting trial {trial.number + 1}/{args.n_trials} | stage={stage} "
            f"| latent={args.latent_channels} | conv={args.conv_type} "
            f"| mode={args.parallel_mode} | gpus={args.gpu_tag}",
            flush=True,
        )

        train_loader, train_eval_loader, val_loader = build_loaders(
            meshdata=meshdata,
            batch_size=args.batch_size,
            subset_fraction=args.train_subset_fraction,
            seed=args.seed + trial.number,
        )

        spiral_device = None if args.parallel_mode == 'model' else primary_device
        spiral_indices_list = build_spiral_indices(
            transform_data=transform_data,
            seq_length=args.seq_length,
            dilation=args.dilation,
            device=spiral_device,
        )

        if args.parallel_mode == 'model':
            model = AEModelParallel(
                args.in_channels,
                args.out_channels,
                args.latent_channels,
                spiral_indices_list,
                down_transform_list,
                up_transform_list,
                device_ids=args.model_parallel_gpu_ids,
                conv_type=args.conv_type,
                adaptive_hidden=args.adaptive_hidden,
                adaptive_dropout=args.adaptive_dropout,
                adaptive_global_context=args.adaptive_global_context,
                force_deterministic_latent=args.force_deterministic_latent,
            )
        else:
            model = AE(
                args.in_channels,
                args.out_channels,
                args.latent_channels,
                spiral_indices_list,
                down_transform_list,
                up_transform_list,
                conv_type=args.conv_type,
                adaptive_hidden=args.adaptive_hidden,
                adaptive_dropout=args.adaptive_dropout,
                adaptive_global_context=args.adaptive_global_context,
                force_deterministic_latent=args.force_deterministic_latent,
            ).to(primary_device)

        data_device = get_data_device(model, primary_device)

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

        def epoch_objective_callback(epoch, epochs, train_loss, val_loss, epoch_duration):
            del train_loss
            del val_loss
            del epoch_duration
            del epochs

            distance_epoch = compute_euclidean_distance(
                model=model,
                data_loader=val_loader,
                device=data_device,
                meshdata=meshdata,
            )
            if stage == 'stage1':
                return {
                    'distance': float(distance_epoch),
                    'sap': float('nan'),
                    'corr_abs': float('nan'),
                    'corr_raw': float('nan'),
                }
            age_metrics_epoch = evaluate_age_metrics_holdout(
                model=model,
                train_eval_loader=train_eval_loader,
                eval_loader=val_loader,
                device=data_device,
                age_label_index=args.age_label_index,
                age_latent_index=args.age_latent_index,
            )

            return {
                'distance': float(distance_epoch),
                'sap': float(age_metrics_epoch['sap_age']),
                'corr_abs': float(age_metrics_epoch['corr_target']),
                'corr_raw': float(age_metrics_epoch['corr_target_raw']),
            }

        try:
            run(
                model=model,
                train_loader=train_loader,
                test_loader=val_loader,
                epochs=args.epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                writer=None,
                device=data_device,
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
                age_latent_index=args.age_latent_index,
                use_snn_cls=args.use_snn_cls,
                use_snn_reg=args.use_snn_reg,
                use_covariance=args.use_covariance,
                covariance_weight=args.covariance_weight,
                save_checkpoints=False,
                epoch_objective_callback=(
                    epoch_objective_callback if args.print_epoch_objectives else None
                ),
                objective_log_interval=args.epoch_objective_interval,
                trial_number=trial.number,
                total_trials=args.n_trials,
            )

            # Hyperparameter objective is validation-only (no test-set optimization).
            euclidean_distance = eval_error(model, val_loader, data_device, meshdata, args.out_dir)
            euclidean_distance = sanitize_metric(euclidean_distance, fallback=1e9)

            age_metrics = evaluate_age_metrics_holdout(
                model=model,
                train_eval_loader=train_eval_loader,
                eval_loader=val_loader,
                device=data_device,
                age_label_index=args.age_label_index,
                age_latent_index=args.age_latent_index,
            )
        except RuntimeError as exc:
            if 'out of memory' in str(exc).lower():
                torch.cuda.empty_cache()
                trial.set_user_attr('objective_split', 'val')
                trial.set_user_attr('oom', True)
                trial.set_user_attr('latent_channels', int(args.latent_channels))
                trial.set_user_attr('parallel_mode', str(args.parallel_mode))
                trial.set_user_attr('gpu_tag', str(args.gpu_tag))
                trial.set_user_attr('conv_type', str(args.conv_type))
                trial.set_user_attr('stage', str(stage))
                trial.set_user_attr('oom_message', str(exc))
                print(
                    f"Trial {trial.number} pruned due to CUDA OOM on {args.gpu_tag} "
                    f"stage={stage} latent={args.latent_channels} conv={args.conv_type}",
                    flush=True,
                )
                raise optuna.TrialPruned('CUDA OOM')
            raise

        prediction_fp = osp.join(
            args.trial_predictions_dir,
            f'trial_{trial.number:04d}_latent{args.latent_channels}_predictions.pt',
        )
        torch.save(
            {
                'trial': int(trial.number),
                'objective_split': 'val',
                'age_true': torch.tensor(age_metrics['age_true_eval'], dtype=torch.float32),
                'age_pred_by_latent': torch.tensor(age_metrics['age_pred_by_latent'], dtype=torch.float32),
                'corr_per_latent': age_metrics['corr_per_latent'],
                'r2_per_latent': age_metrics['r2_per_latent'],
                'sap_age_holdout': age_metrics['sap_age'],
            },
            prediction_fp,
        )

        model_fp = osp.join(
            args.trial_models_dir,
            f'trial_{trial.number:04d}_latent{args.latent_channels}_model.pt',
        )
        torch.save(
            {
                'trial': int(trial.number),
                'objective_split': 'val',
                'model_state_dict': model.state_dict(),
                'in_channels': args.in_channels,
                'out_channels': args.out_channels,
                'latent_channels': args.latent_channels,
                'parallel_mode': args.parallel_mode,
                'model_parallel_gpu_ids': list(args.model_parallel_gpu_ids),
                'stage': stage,
                'conv_type': args.conv_type,
                'adaptive_hidden': int(args.adaptive_hidden),
                'adaptive_dropout': float(args.adaptive_dropout),
                'adaptive_global_context': bool(args.adaptive_global_context),
                'force_deterministic_latent': bool(args.force_deterministic_latent),
                'seq_length': args.seq_length,
                'dilation': args.dilation,
                'age_label_index': args.age_label_index,
                'age_latent_index': args.age_latent_index,
                'spiral_indices_list': [t.detach().cpu() for t in spiral_indices_list],
                'down_transform_list': [t.detach().cpu() for t in down_transform_list],
                'up_transform_list': [t.detach().cpu() for t in up_transform_list],
                'mean': meshdata.mean.detach().cpu(),
                'std': meshdata.std.detach().cpu(),
                'trial_params': trial.params,
                'objective_values': {
                    'euclidean_distance_val': float(euclidean_distance),
                    'sap_age_holdout_val': float(age_metrics['sap_age']),
                    'corr_target_abs_val': float(age_metrics['corr_target']),
                    'corr_target_raw_val': float(age_metrics['corr_target_raw']),
                },
                'prediction_fp': prediction_fp,
            },
            model_fp,
        )

        trial.set_user_attr('objective_split', 'val')
        trial.set_user_attr('stage', str(stage))
        trial.set_user_attr('latent_channels', int(args.latent_channels))
        trial.set_user_attr('parallel_mode', str(args.parallel_mode))
        trial.set_user_attr('gpu_tag', str(args.gpu_tag))
        trial.set_user_attr('conv_type', str(args.conv_type))
        trial.set_user_attr('adaptive_hidden', int(args.adaptive_hidden))
        trial.set_user_attr('adaptive_dropout', float(args.adaptive_dropout))
        trial.set_user_attr(
            'adaptive_global_context',
            bool(args.adaptive_global_context),
        )
        trial.set_user_attr('corr_per_latent', [float(v) for v in age_metrics['corr_per_latent']])
        trial.set_user_attr('r2_per_latent', [float(v) for v in age_metrics['r2_per_latent']])
        trial.set_user_attr('euclidean_distance', float(euclidean_distance))
        trial.set_user_attr('sap_age_holdout', float(age_metrics['sap_age']))
        trial.set_user_attr('corr_target_abs', float(age_metrics['corr_target']))
        trial.set_user_attr('corr_target_raw', float(age_metrics['corr_target_raw']))
        trial.set_user_attr('prediction_fp', prediction_fp)
        trial.set_user_attr('model_fp', model_fp)

        print('')
        print(
            f"Trial {trial.number} | stage={stage} | latent={args.latent_channels} | mode={args.parallel_mode} "
            f"| conv={args.conv_type} | gpus={args.gpu_tag} | split=val"
        )
        print(f"Euclidean Distance (val): {euclidean_distance:.6f}")
        print(f"SAP Score Age Holdout (train->val): {age_metrics['sap_age']:.6f}")
        print(
            f"|Corr(Age, z[{args.age_latent_index}])| (val): {age_metrics['corr_target']:.6f}"
        )
        print(
            f"Corr(Age, z[{args.age_latent_index}]) raw (val): {age_metrics['corr_target_raw']:.6f}"
        )
        print('')

        if stage == 'stage1':
            return (euclidean_distance,)
        if stage == 'stage2':
            return (age_metrics['sap_age'], age_metrics['corr_target'])
        return (euclidean_distance, age_metrics['sap_age'], age_metrics['corr_target'])

    return objective


def main(argv=None):
    args = parse_args(argv)

    resolve_parallel_config(args)
    resolve_conv_config(args)
    resolve_stage_config(args)
    resolve_latent_config(args)
    if args.optuna_stage == 'stage1':
        args.latent_tag = 'stage1search'

    if args.age_latent_index < 0:
        raise ValueError('age_latent_index must be >= 0')

    if args.age_latent_index >= args.min_latent_channels:
        raise ValueError(
            f"age_latent_index={args.age_latent_index} must be < smallest latent choice {args.min_latent_channels}"
        )

    if args.epoch_objective_interval < 1:
        raise ValueError('epoch_objective_interval must be >= 1')

    resolve_paths(args)

    log_handle = None
    original_stdout = None
    original_stderr = None

    try:
        log_handle, original_stdout, original_stderr = setup_logging(args.log_fp)

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for this training script.')
        available_gpus = torch.cuda.device_count()
        for gpu_id in args.model_parallel_gpu_ids:
            if gpu_id < 0 or gpu_id >= available_gpus:
                raise ValueError(
                    f'GPU id {gpu_id} is invalid for this host (available: 0..{available_gpus - 1}).'
                )
        primary_device = torch.device('cuda', args.model_parallel_gpu_ids[0])

        torch.set_num_threads(args.n_threads)

        set_seed(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

        print(f"Parallel mode: {args.parallel_mode}")
        print(f"GPU ids: {args.model_parallel_gpu_ids}")
        print(f"Primary device: {primary_device}")
        print(f"Optuna stage: {args.optuna_stage}")
        print(
            f"Conv type: {args.conv_type} "
            f"(optimize_conv_type={args.optimize_conv_type})"
        )
        if args.conv_type == 'adaptive_spiral' or args.optimize_conv_type:
            print(
                "Adaptive conv params: "
                f"hidden={args.adaptive_hidden}, "
                f"dropout={args.adaptive_dropout}, "
                f"global_context={args.adaptive_global_context}"
            )
        if args.optuna_stage == 'stage1':
            print(f"Stage-1 latent choices: {args.latent_choices}")
            print("Stage-1 loss mode: reconstruction only (KL/SNN/COV disabled)")
        if args.optuna_stage == 'stage2':
            print(f"Stage-2 fixed latent: {args.latent_channels}")
            if args.stage1_summary_fp:
                print(f"Stage-2 latent source summary: {args.stage1_summary_fp}")

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
            primary_device=primary_device,
        )

        trial_logger = TrialLogger(
            metrics_fp=args.trial_metrics_fp,
            trials_fp=args.intermediate_trials_fp,
            latent_channels=args.max_latent_channels,
        )

        if args.optuna_stage == 'stage1':
            directions = ['minimize']
            sampler = optuna.samplers.TPESampler(seed=args.seed)
        elif args.optuna_stage == 'stage2':
            directions = ['maximize', 'maximize']
            sampler = optuna.samplers.NSGAIISampler(seed=args.seed)
        else:
            directions = ['minimize', 'maximize', 'maximize']
            sampler = optuna.samplers.NSGAIISampler(seed=args.seed)

        study_kwargs = {
            'directions': directions,
            'sampler': sampler,
        }
        if args.storage:
            study_kwargs['storage'] = args.storage
            study_kwargs['study_name'] = args.study_name or f"{args.exp_name}_{args.optuna_stage}"
            study_kwargs['load_if_exists'] = True
        elif args.study_name:
            study_kwargs['study_name'] = args.study_name

        study = optuna.create_study(**study_kwargs)
        study.optimize(objective, n_trials=args.n_trials, callbacks=[trial_logger])

        save_study_artifacts(study, args)
    finally:
        if log_handle is not None:
            teardown_logging(log_handle, original_stdout, original_stderr)


if __name__ == '__main__':
    main()
