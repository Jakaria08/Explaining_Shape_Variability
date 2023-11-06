import time
import math
import os
import torch
import torch.nn.functional as F
from reconstruction import Regressor, Classifier
from reconstruction.loss import ClsCorrelationLoss, RegCorrelationLoss, SNNLoss, SNNRegLoss
from torch.utils.data import Subset
from utils import DataLoader

def loss_function(original, reconstruction, mu, log_var, beta):
    reconstruction_loss = F.l1_loss(reconstruction, original, reduction='mean')
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    return reconstruction_loss + beta*kld_loss

def run(model, train_loader, test_loader, epochs, optimizer, scheduler, writer,
        device, beta, w_cls, guided, guided_contrastive_loss, correlation_loss, latent_channels, weight_decay_c, temp, i):
    
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
        train_loss = train(model, optimizer, model_c, optimizer_c, model_c_2, optimizer_c_2, train_loader, device, beta, w_cls, guided, guided_contrastive_loss, correlation_loss, temp, i)
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
        torch.save(model.state_dict(), "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_contrastive/model_state_dict.pt")
        torch.save(model_c.state_dict(), "/home/jakaria/Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_contrastive/model_c_state_dict.pt")

def train(model, optimizer, model_c, optimizer_c, model_c_2, optimizer_c_2, loader, device, beta, w_cls, guided, guided_contrastive_loss, correlation_loss, temp, i):
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
    total_data = len(loader)*loader.batch_size
    # i is the percentage of train data
    #print("Data Percentage: "+str(i))
    desired_data = math.ceil(i/10 * total_data)
    #print("desired batches: "+ str(desired_batches))
    #print("total batches: " + str(total_batches))
    # Select desired number of batches according to the percentage of train data
    subset_loader = DataLoader(Subset(loader.dataset, range(desired_data)), 
                               batch_size=loader.batch_size)

    for data in subset_loader:
	    # Load Data
        x = data.x.to(device)
        label = data.y.to(device)
        #print(x.shape)
        #print(x.shape[0])
        if x.shape[0] != loader.batch_size:
            continue
        #print(label)
	    # VAE + Exhibition
        optimizer.zero_grad()
        out, mu, log_var, re, re_2 = model(x) # re2 for excitation
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
            loss_snn = SNN_Loss(z[:,0], label[:, :, 0])
            loss += loss_snn * w_cls
            #print(loss_snn.item())
            snnl += loss_snn.item()

            #Regression Loss
            SNN_Loss_Reg = SNNRegLoss(temp)
            loss_snn_reg = SNN_Loss_Reg(z[:,1], label[:, :, 2])
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

