import argparse
import csv
import glob
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_PACKAGE_PARENT = _PROJECT_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from datasets import MeshData
from reconstruction import AE
from guided_vae.utils.dataloader import DataLoader
from guided_vae.utils.sap_holdout import sap_regression_holdout


DEFAULT_DATA_FP = (
    "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/"
    "guided_vae/data/CoMA"
)
DEFAULT_TEMPLATE_FP = (
    "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/"
    "guided_vae/data/CoMA/template/template.ply"
)
DEFAULT_MODELS_DIR = (
    "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/"
    "guided_vae/data/CoMA/raw/calsnic_als/models/trial_models_g0-1-2_spiral_single_latent256"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate saved trial models on test split.')
    parser.add_argument('--model_fp', type=str, default='')
    parser.add_argument('--models_dir', type=str, default=DEFAULT_MODELS_DIR)
    parser.add_argument('--pattern', type=str, default='trial_*_model.pt')
    parser.add_argument('--max_models', type=int, default=0)
    parser.add_argument('--device_idx', type=int, default=0)
    parser.add_argument('--data_fp', type=str, default=DEFAULT_DATA_FP)
    parser.add_argument('--template_fp', type=str, default=DEFAULT_TEMPLATE_FP)
    parser.add_argument('--split', type=str, default='interpolation')
    parser.add_argument('--test_exp', type=str, default='bareteeth')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--age_label_index', type=int, default=1)
    parser.add_argument('--age_latent_index', type=int, default=0)
    parser.add_argument('--out_csv', type=str, default='')
    return parser.parse_args(argv)


def _sanitize_metric(x, fallback=0.0):
    x = float(x)
    if np.isnan(x) or np.isinf(x):
        return float(fallback)
    return x


def _collect_latents_and_age(model, loader, device, age_label_index):
    latent_codes = []
    ages = []
    model.eval()
    model.training = False

    with torch.no_grad():
        for data in loader:
            x = data.x.to(device)
            y = data.y.to(device)
            _, mu, _, _, _ = model(x)
            latent_codes.append(mu.detach().cpu())
            ages.append(y[:, :, age_label_index].view(-1, 1).detach().cpu())

    if len(latent_codes) == 0:
        return np.zeros((0, model.latent_channels), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)

    z = torch.cat(latent_codes, dim=0)
    y = torch.cat(ages, dim=0)
    z[torch.isnan(z) | torch.isinf(z)] = 0
    return z.numpy(), y.numpy()


def _compute_test_distance(model, loader, device, meshdata):
    model.eval()
    model.training = False

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    with torch.no_grad():
        for data in loader:
            x = data.x.to(device)
            pred, _, _, _, _ = model(x)
            num_graphs = data.num_graphs
            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_pred *= 300
            reshaped_x *= 300
            tmp_error = torch.sqrt(torch.sum((reshaped_pred - reshaped_x) ** 2, dim=2))
            errors.append(tmp_error)

    if len(errors) == 0:
        return 1e9

    all_errors = torch.cat(errors, dim=0)
    return _sanitize_metric(all_errors.view(-1).mean().item(), fallback=1e9)


def _evaluate_model(model, train_loader, test_loader, device, meshdata, age_label_index, age_latent_index):
    distance = _compute_test_distance(model, test_loader, device, meshdata)

    train_latent_np, train_age_np = _collect_latents_and_age(
        model=model,
        loader=train_loader,
        device=device,
        age_label_index=age_label_index,
    )
    test_latent_np, test_age_np = _collect_latents_and_age(
        model=model,
        loader=test_loader,
        device=device,
        age_label_index=age_label_index,
    )

    if train_latent_np.shape[0] == 0 or test_latent_np.shape[0] == 0:
        return {
            'distance_test': distance,
            'sap_age_holdout_test': 0.0,
            'corr_target_abs_test': 0.0,
            'corr_target_raw_test': 0.0,
        }

    sap_age, s_matrix, _ = sap_regression_holdout(
        train_factors=train_age_np,
        train_codes=train_latent_np,
        eval_factors=test_age_np,
        eval_codes=test_latent_np,
    )
    sap_age = _sanitize_metric(sap_age, fallback=0.0)

    age_vec = test_age_np.reshape(-1)
    if age_latent_index < 0 or age_latent_index >= test_latent_np.shape[1]:
        corr_raw = 0.0
    else:
        latent_vec = test_latent_np[:, age_latent_index]
        if np.std(age_vec) < 1e-12 or np.std(latent_vec) < 1e-12:
            corr_raw = 0.0
        else:
            corr_raw = stats.pearsonr(age_vec, latent_vec)[0]
    corr_raw = _sanitize_metric(corr_raw, fallback=0.0)
    corr_abs = abs(corr_raw)

    del s_matrix
    return {
        'distance_test': distance,
        'sap_age_holdout_test': sap_age,
        'corr_target_abs_test': corr_abs,
        'corr_target_raw_test': corr_raw,
    }


def _find_model_files(args):
    if args.model_fp:
        if not osp.exists(args.model_fp):
            raise FileNotFoundError(f'Checkpoint not found: {args.model_fp}')
        return [args.model_fp]

    if not osp.isdir(args.models_dir):
        raise FileNotFoundError(f'Model directory not found: {args.models_dir}')

    model_files = sorted(glob.glob(osp.join(args.models_dir, args.pattern)))
    if args.max_models > 0:
        model_files = model_files[: int(args.max_models)]
    if len(model_files) == 0:
        raise FileNotFoundError(
            f'No model checkpoints found in {args.models_dir} with pattern {args.pattern}'
        )
    return model_files


def _load_model_from_checkpoint(ckpt, device):
    model = AE(
        ckpt['in_channels'],
        ckpt['out_channels'],
        ckpt['latent_channels'],
        [t.to(device) for t in ckpt['spiral_indices_list']],
        [t.to(device) for t in ckpt['down_transform_list']],
        [t.to(device) for t in ckpt['up_transform_list']],
        conv_type=ckpt.get('conv_type', 'spiral'),
        adaptive_hidden=int(ckpt.get('adaptive_hidden', 32)),
        adaptive_dropout=float(ckpt.get('adaptive_dropout', 0.0)),
        adaptive_global_context=bool(ckpt.get('adaptive_global_context', False)),
        force_deterministic_latent=bool(ckpt.get('force_deterministic_latent', False)),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    model.training = False
    return model


def _write_csv(rows, out_csv):
    out_dir = osp.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'model_fp',
                'trial',
                'latent_channels',
                'conv_type',
                'distance_test',
                'sap_age_holdout_test',
                'corr_target_abs_test',
                'corr_target_raw_test',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to run this evaluation script.')
    device = torch.device('cuda', int(args.device_idx))

    meshdata = MeshData(
        args.data_fp,
        args.template_fp,
        split=args.split,
        test_exp=args.test_exp,
    )
    train_loader = DataLoader(meshdata.train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(meshdata.test_dataset, batch_size=args.batch_size, shuffle=False)

    model_files = _find_model_files(args)
    rows = []

    print(f'Evaluating {len(model_files)} model checkpoint(s) on device={device}')
    for idx, model_fp in enumerate(model_files, start=1):
        ckpt = torch.load(model_fp, map_location='cpu')
        model = _load_model_from_checkpoint(ckpt, device)

        metrics = _evaluate_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            meshdata=meshdata,
            age_label_index=args.age_label_index,
            age_latent_index=args.age_latent_index,
        )

        row = {
            'model_fp': model_fp,
            'trial': int(ckpt.get('trial', -1)),
            'latent_channels': int(ckpt.get('latent_channels', -1)),
            'conv_type': str(ckpt.get('conv_type', 'spiral')),
            **metrics,
        }
        rows.append(row)

        print(
            f"[{idx}/{len(model_files)}] trial={row['trial']} latent={row['latent_channels']} "
            f"conv={row['conv_type']} dist={row['distance_test']:.6f} "
            f"sap={row['sap_age_holdout_test']:.6f} "
            f"corr_abs={row['corr_target_abs_test']:.6f} "
            f"corr_raw={row['corr_target_raw_test']:.6f}"
        )

    if args.out_csv:
        _write_csv(rows, args.out_csv)
        print(f'Saved evaluation CSV: {args.out_csv}')


if __name__ == '__main__':
    main()
