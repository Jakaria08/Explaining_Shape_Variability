import torch
from reconstruction import AE
from datasets import MeshData
from utils import DataLoader
import tqdm
import numpy as np
import geomloss
from pytorch3d.loss import chamfer_distance
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from numpy.linalg import norm
from sklearn.metrics import accuracy_score, mean_squared_error

device = torch.device('cuda', 1)
# Set the path to the saved model directory
#model_path = "/home/jakaria/torus_bump_500_three_scale_binary_bump_variable_noise_fixed_angle/models_classification_regression_only_correlation_loss/models/65"
model_path = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_contrastive_inhib/684"
#model_path = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_guided/30"# Load the saved model
#model_path = "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_attribute/23"
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

train_loader = DataLoader(meshdata.train_dataset, batch_size=16)
test_loader = DataLoader(meshdata.test_dataset, batch_size=16)

test_original_small = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        #print("test...")
        x = data.x.to(device)
        y = data.y.to(device)

        test_original_small.append(x)
        
        if i == 1:
            print(len(test_original_small))
            break
test_original_small = torch.cat(test_original_small)
print(test_original_small.shape)

def random_latent(n_samples, latent_dim=12):
    z = torch.randn([n_samples, latent_dim])
    z = z.to(device)
    return z

def random_generation(n_samples=16, latent_dim=12):
    z = random_latent(n_samples, latent_dim=latent_dim)
    with torch.no_grad():
        gen_verts = model.decoder(z.to(device))
    return gen_verts


def emd_approx(sample, ref):
    return geomloss.SamplesLoss()(sample, ref)


def _pairwise_emd_cd_(sample_pcs, ref_pcs, batch_size):
    print("computing Earth Mover and Chamfer distances")
    n_sample = sample_pcs.shape[0]
    n_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(n_sample)
    for sample_b_start in tqdm.tqdm(iterator):
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        for ref_b_start in range(0, n_ref, batch_size):
            ref_b_end = min(n_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(
                batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            cd_lst.append(chamfer_distance(sample_batch_exp, ref_batch,
                          batch_reduction=None)[0].unsqueeze(0))

            emd_batch = emd_approx(sample_batch_exp, ref_batch)
            emd_lst.append(emd_batch.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=-1)
        emd_lst = torch.cat(emd_lst, dim=-1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # n_sample, n_ref
    all_emd = torch.cat(all_emd, dim=0)  # n_sample, n_ref
    return all_cd, all_emd


def knn(m_xx, m_xy, m_yy, k, sqrt=False):
    n0 = m_xx.size(0)
    n1 = m_yy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(m_xx)
    mat = torch.cat((torch.cat((m_xx, m_xy), 1),
                    torch.cat((m_xy.transpose(0, 1), m_yy), 1)), 0)
    if sqrt:
        mat = mat.abs().sqrt()

    val, idx = (mat + torch.diag(
        float('inf') * torch.ones(n0 + n1).to(m_xx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(m_xx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) *
                    torch.ones(n0 + n1).to(m_xx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def lgan_mmd_cov(all_dist):
    n_sample, n_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(n_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }


def compute_all_metrics(sample_pcs, ref_pcs, batch_size):
    results = {}

    m_rs_cd, m_rs_emd = _pairwise_emd_cd_(sample_pcs, ref_pcs, batch_size)
    '''
    res_cd = lgan_mmd_cov(m_rs_cd.t())
    results.update({
        "%s-CD" % k: v for k, v in res_cd.items()
    })

    res_emd = lgan_mmd_cov(m_rs_emd.t())
    results.update({
        "%s-EMD" % k: v for k, v in res_emd.items()
    })
    '''
    m_rr_cd, m_rr_emd = _pairwise_emd_cd_(ref_pcs, ref_pcs, batch_size)
    m_ss_cd, m_ss_emd = _pairwise_emd_cd_(sample_pcs, sample_pcs, batch_size)

    # 1-NN results
    one_nn_cd_res = knn(m_rr_cd, m_rs_cd, m_ss_cd, 1, sqrt=False)
    results.update({
        "1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
    })
    one_nn_emd_res = knn(m_rr_emd, m_rs_emd, m_ss_emd, 1, sqrt=False)
    results.update({
        "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
    })

    return results



random_generation = random_generation(n_samples=32, latent_dim=12)
print(random_generation.shape)

restults = compute_all_metrics(random_generation, test_original_small, 2)
print(restults)