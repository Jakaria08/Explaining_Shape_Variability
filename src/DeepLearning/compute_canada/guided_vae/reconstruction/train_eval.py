import time
import math
import os
import torch
import torch.nn.functional as F
from reconstruction import Regressor, Classifier
from reconstruction.loss import ClsCorrelationLoss, RegCorrelationLoss, SNNLoss, SNNRegLoss
from torch.utils.data import Subset
from utils import DataLoader
import random


def matrix_log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian for all combination of bacth pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.

    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).

    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).

    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).

    batch_size: int
        number of training images in the batch
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian.

    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.

    mu: torch.Tensor or np.ndarray or float
        Mean.

    logvar: torch.Tensor or np.ndarray or float
        Log variance.
    """
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Calculates a log importance weight matrix

    Parameters
    ----------
    batch_size: int
        number of training images in the batch

    dataset_size: int
    number of training images in the dataset
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()


def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


# Batch TC specific
# TO-DO: test if mss is better!
def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, batch_size, is_mss=True):

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx

def loss_function_tc(original, reconstruction, mu, log_var, z, alpha, beta, gamma, n_data, batch_size, is_train):
    global n_train_steps
    reconstruction_loss = F.l1_loss(reconstruction, original, reduction='mean')
    #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    latent_sample = z
    latent_dist = mu, log_var
    log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                                             latent_dist,
                                                                             n_data,
                                                                             batch_size,
                                                                             is_mss=True)
    # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
    mi_loss = (log_q_zCx - log_qz).mean()
    # TC[z] = KL[q(z)||\prod_i z_i]
    tc_loss = (log_qz - log_prod_qzi).mean()
    # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
    dw_kl_loss = (log_prod_qzi - log_pz).mean()
    
    anneal_reg = (linear_annealing(0, 1, n_train_steps, 1)
                      if is_train else 1)

    loss = reconstruction_loss + (alpha * mi_loss + beta * tc_loss +
                           anneal_reg * gamma * dw_kl_loss)
    return loss


def loss_function(original, reconstruction, mu, log_var, beta):
    reconstruction_loss = F.l1_loss(reconstruction, original, reduction='mean')
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    return reconstruction_loss + beta*kld_loss

def run(model, train_loader, test_loader, epochs, optimizer, scheduler, writer,
        device, beta, w_cls, guided, guided_contrastive_loss, correlation_loss, latent_channels, weight_decay_c, temp, threshold, tc, i):
    
    model_c = Classifier(latent_channels).to(device)
    optimizer_c = torch.optim.Adam(model_c.parameters(), lr=1e-3, weight_decay=weight_decay_c)

    model_c_2 = Regressor(latent_channels).to(device)
    optimizer_c_2 = torch.optim.Adam(model_c_2.parameters(), lr=1e-3, weight_decay=weight_decay_c)

    train_losses, test_losses = [], []

    total_batches = len(train_loader)
    # i is the percentage of train data
    print("Data Percentage: "+str(i))
    desired_batches = math.ceil(i/10 * total_batches)
    print("desired batches: "+ str(desired_batches))
    print("total batches: " + str(total_batches))

    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train(model, optimizer, model_c, optimizer_c, model_c_2, optimizer_c_2, train_loader, device, beta, w_cls, guided, guided_contrastive_loss, correlation_loss, temp, threshold, tc, i)
        t_duration = time.time() - t
        test_loss = test(model, test_loader, device, beta)
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            't_duration': t_duration
        }

        writer.print_info(info)
        writer.save_checkpoint(model, optimizer, scheduler, epoch)
        torch.save(model.state_dict(), "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_contrastive_inhib_decrease_trainset/model_state_dict.pt")
        torch.save(model_c.state_dict(), "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_contrastive_inhib_decrease_trainset/model_c_state_dict.pt")

def train(model, optimizer, model_c, optimizer_c, model_c_2, optimizer_c_2, loader, device, beta, w_cls, guided, guided_contrastive_loss, correlation_loss, temp, threshold, tc, i):
    model.train()
    model_c.train()
    model_c_2.train()

    total_loss = 0
    recon_loss = 0
    reg_loss = 0
    cls1_error = 0
    cls2_error = 0

    cls1_error_2 = 0
    cls2_error_2 = 0

    snnl = 0
    snnl_reg = 0
    corrl_cls = 0
    corrl_reg = 0
    # Calculate total and desired number of batches
    total_data = len(loader.dataset)
    # i is the percentage of train data
    #print("Data Percentage: "+str(i))
    desired_data = math.ceil(i/10 * total_data)
    #print("desired batches: "+ str(desired_batches))
    #print("total batches: " + str(total_batches))
    # Shuffle the indices of the entire dataset
    indices = list(range(total_data))
    #random.shuffle(shuffled_indices)

    # Use the first 'desired_data' indices to create a subset
    subset_indices = indices[:desired_data]
    # Select desired number of batches according to the percentage of train data
    subset_loader = DataLoader(Subset(loader.dataset, subset_indices), 
                               batch_size=loader.batch_size)


    for data in subset_loader:
	    # Load Data
        x = data.x.to(device)
        label = data.y.to(device)
        batch_size = len(data)
        #print(x.shape)
        #print(x.shape[0])
        if x.shape[0] != loader.batch_size:
            continue
        #print(label)
	    # VAE + Exhibition
        optimizer.zero_grad()
        out, mu, log_var, re, re_2 = model(x) # re2 for excitation
        n_data = len(subset_loader.dataset)
        alpha = 1
        #beta = 1
        gamma = 1

        if tc:
            loss = loss_function(x, out, mu, log_var, z, alpha, beta, gamma, n_data, batch_size, is_train=True)
        else:
            loss = loss_function(x, out, mu, log_var, beta)    

        if guided:
            loss_cls = F.binary_cross_entropy(re, label[:, :, 0], reduction='mean')
            loss += loss_cls * w_cls
            #print(re[0:5])
            #print(label[:, :, 0][0:5])
            #print(loss_cls.item())
        
        if guided_contrastive_loss:
            #Classification Loss
            #SNN_Loss = SNNLCrossEntropy(temperature=temp)
            SNN_Loss = SNNLoss(temp)
    
            z = model.reparameterize(mu, log_var)
            #print(z.shape)
            #print(label[:, :, 0].shape)
            loss_snn = SNN_Loss(z, label[:, :, 0])
            loss += loss_snn * w_cls
            #print(loss_snn.item())
            snnl += loss_snn.item()

            #Regression Loss
            SNN_Loss_Reg = SNNRegLoss(temp, threshold)
            loss_snn_reg = SNN_Loss_Reg(z, label[:, :, 2])
            loss += loss_snn_reg * w_cls
            #print(loss_snn.item())
            snnl_reg += loss_snn_reg.item()

        if correlation_loss:
            corr_loss_cls = ClsCorrelationLoss()
            corr_loss_reg = RegCorrelationLoss()
            z = model.reparameterize(mu, log_var)
            #print(z.shape)
            #print(label[:, :, 0].shape)
            #cls
            loss_corr_cls = corr_loss_cls(z, label[:, :, 0])
            loss += loss_corr_cls * w_cls
            #reg
            loss_corr_reg = corr_loss_reg(z, label[:, :, 2])
            loss += loss_corr_reg * w_cls
            #print(corr_loss.item())
            corrl_cls += loss_corr_cls.item()
            corrl_reg += loss_corr_reg.item()

        loss.backward()        
        optimizer.step()
        total_loss += loss.item()

        if guided:
            # Inhibition Step 1 for label 1
            optimizer_c.zero_grad()
            z = model.reparameterize(mu, log_var).detach()
            z = z[:, 1:]
            cls1 = model_c(z)
            loss = F.binary_cross_entropy(cls1, label[:, :, 0], reduction='mean')
            cls1_error += loss.item()
            loss *= w_cls
            loss.backward()
            optimizer_c.step()

            # Inhibition Step 2 for label 1
            optimizer.zero_grad()
            mu, log_var = model.encoder(x)
            z = model.reparameterize(mu, log_var)
            z = z[:, 1:]
            cls2 = model_c(z)
            label1 = torch.empty_like(label[:, :, 0]).fill_(0.5)
            loss = F.binary_cross_entropy(cls2, label1, reduction='mean')
            cls2_error += loss.item()
            loss *= w_cls
            loss.backward()
            optimizer.step()

            #excitation for z[1]
            out, mu, log_var, re, re_2 = model(x) # re2 for excitation
            loss = loss_function(x, out, mu, log_var, beta)  
            optimizer.zero_grad()
            loss_cls_2 = F.mse_loss(re_2, label[:, :, 2], reduction='mean')
            loss += loss_cls_2 * w_cls
            #print(re_2[0:5])
            #print(label[:, :, 1][0:5])
            #print(loss_cls_2.item())
            loss.backward()        
            optimizer.step()
            total_loss += loss.item()
        
        
            # Inhibition Step 1 for label 2
            optimizer_c_2.zero_grad()
            z = model.reparameterize(mu, log_var).detach()
            z = z[:, torch.cat((torch.tensor([0]), torch.tensor(range(2, z.shape[1]))), dim=0)]
            cls1_2 = model_c_2(z)
            loss = F.mse_loss(cls1_2, label[:, :, 2], reduction='mean')
            cls1_error_2 += loss.item()
            loss *= w_cls
            loss.backward()
            optimizer_c_2.step()

            # Inhibition Step 2 for label 2
            optimizer.zero_grad()
            mu, log_var = model.encoder(x)
            z = model.reparameterize(mu, log_var)
            z = z[:, torch.cat((torch.tensor([0]), torch.tensor(range(2, z.shape[1]))), dim=0)]
            cls2_2 = model_c_2(z)
            label1 = torch.empty_like(label[:, :, 1]).fill_(0.5)
            loss = F.mse_loss(cls2_2, label1, reduction='mean')
            cls2_error_2 += loss.item()
            loss *= w_cls
            loss.backward()
            optimizer.step()
    #print(corrl_cls)
    #print(corrl_reg)
    print(snnl)
    print(snnl_reg)
    return total_loss / len(loader)


def test(model, loader, device, beta):
    model.eval()
    model.training = False

    total_loss = 0
    recon_loss = 0
    reg_loss = 0
    reg_loss_2 = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.x.to(device)
            has_nan = torch.isnan(x).any().item()
            if has_nan:
                continue
            y = data.y.to(device)
            pred, mu, log_var, re, re_2 = model(x)
            has_nan_1 = torch.isnan(re).any().item()
            if has_nan_1:
                continue
            #print(re.shape)
            total_loss += loss_function(x, pred, mu, log_var, beta)
            recon_loss += F.l1_loss(pred, x, reduction='mean')
            reg_loss += F.binary_cross_entropy(re, y[:, :, 0], reduction='mean')
            reg_loss_2 += F.mse_loss(re_2, y[:, :, 2], reduction='mean')

    return total_loss / len(loader)

def eval_error(model, test_loader, device, meshdata, out_dir):
    model.eval()
    model.training = False

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data.x.to(device)
            # pred = model(x)
            pred, mu, log_var, re, re_2 = model(x)
            num_graphs = data.num_graphs
            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean

            reshaped_pred *= 300
            reshaped_x *= 300

            tmp_error = torch.sqrt(
                torch.sum((reshaped_pred - reshaped_x)**2,
                          dim=2))  # [num_graphs, num_nodes]
            errors.append(tmp_error)
        new_errors = torch.cat(errors, dim=0)  # [n_total_graphs, num_nodes]

        mean_error = new_errors.view((-1, )).mean()
        std_error = new_errors.view((-1, )).std()
        median_error = new_errors.view((-1, )).median()

    message = 'Euclidean Error: {:.3f}+{:.3f} | {:.3f}'.format(mean_error, std_error,
                                                     median_error)

    out_error_fp = out_dir + '/euc_errors.txt'
    with open(out_error_fp, 'a') as log_file:
        log_file.write('{:s}\n'.format(message))
    print("")
    print("")
    print(message)
    print("")
    print("")

    return mean_error

