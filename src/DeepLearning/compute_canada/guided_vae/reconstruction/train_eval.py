import time
import torch
import torch.nn.functional as F

from reconstruction.loss import (
    ClsCorrelationLoss,
    RegCorrelationLoss,
    SNNLoss,
    SNNRegLoss,
    WassersteinLoss,
    CovarianceLoss,
)


def loss_function(original, reconstruction, mu, log_var, beta):
    reconstruction_loss = F.l1_loss(reconstruction, original, reduction='mean')
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1),
        dim=0,
    )
    return reconstruction_loss + beta * kld_loss


def run(
    model,
    train_loader,
    test_loader,
    epochs,
    optimizer,
    scheduler,
    writer,
    device,
    beta,
    w_cls,
    guided,
    guided_contrastive_loss,
    correlation_loss,
    latent_channels,
    weight_decay_c,
    temp,
    delta,
    lambda1,
    lambda2,
    threshold,
    age_label_index=1,
    use_snn_cls=False,
    use_snn_reg=True,
    use_covariance=False,
    covariance_weight=0.0,
    save_checkpoints=True,
):
    # Keep instantiated lazily only when requested.
    snn_cls_criterion = (
        SNNLoss(temp, lambda1, lambda2)
        if guided_contrastive_loss and use_snn_cls
        else None
    )
    snn_reg_criterion = (
        SNNRegLoss(temp, threshold)
        if guided_contrastive_loss and use_snn_reg
        else None
    )
    cov_criterion = CovarianceLoss() if use_covariance else None

    corr_cls_criterion = ClsCorrelationLoss() if correlation_loss else None
    corr_reg_criterion = RegCorrelationLoss() if correlation_loss else None

    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            beta=beta,
            w_cls=w_cls,
            guided=guided,
            guided_contrastive_loss=guided_contrastive_loss,
            correlation_loss=correlation_loss,
            threshold=threshold,
            age_label_index=age_label_index,
            snn_cls_criterion=snn_cls_criterion,
            snn_reg_criterion=snn_reg_criterion,
            cov_criterion=cov_criterion,
            corr_cls_criterion=corr_cls_criterion,
            corr_reg_criterion=corr_reg_criterion,
            covariance_weight=covariance_weight,
        )
        t_duration = time.time() - t

        test_loss = test(
            model=model,
            loader=test_loader,
            device=device,
            beta=beta,
            age_label_index=age_label_index,
        )
        scheduler.step()

        if writer is not None:
            info = {
                'current_epoch': epoch,
                'epochs': epochs,
                'train_loss': train_loss,
                'test_loss': test_loss,
                't_duration': t_duration,
            }
            writer.print_info(info)
            if save_checkpoints:
                writer.save_checkpoint(model, optimizer, scheduler, epoch)


def train(
    model,
    optimizer,
    loader,
    device,
    beta,
    w_cls,
    guided,
    guided_contrastive_loss,
    correlation_loss,
    threshold,
    age_label_index,
    snn_cls_criterion,
    snn_reg_criterion,
    cov_criterion,
    corr_cls_criterion,
    corr_reg_criterion,
    covariance_weight,
):
    del threshold  # handled inside snn_reg_criterion

    model.train()
    total_loss = 0.0
    used_batches = 0

    for data in loader:
        x = data.x.to(device)
        label = data.y.to(device)

        if x.shape[0] != loader.batch_size:
            # Keep historical behavior to avoid partial-batch instability.
            continue

        optimizer.zero_grad()
        out, mu, log_var, re, re_2 = model(x)
        loss = loss_function(x, out, mu, log_var, beta)
        z = model.reparameterize(mu, log_var)

        if guided:
            # Legacy classification branch; disabled by default in current setup.
            loss_cls = F.binary_cross_entropy(re, label[:, :, 0], reduction='mean')
            loss = loss + (loss_cls * w_cls)

        if guided_contrastive_loss and snn_cls_criterion is not None:
            loss_snn_cls = snn_cls_criterion(z, label[:, :, 0])
            loss = loss + (loss_snn_cls * w_cls)

        if guided_contrastive_loss and snn_reg_criterion is not None:
            age_target = label[:, :, age_label_index]
            loss_snn_reg = snn_reg_criterion(z, age_target)
            loss = loss + (loss_snn_reg * w_cls)

        if correlation_loss and corr_cls_criterion is not None and corr_reg_criterion is not None:
            loss_corr_cls = corr_cls_criterion(z, label[:, :, 0])
            loss_corr_reg = corr_reg_criterion(z, label[:, :, age_label_index])
            loss = loss + (loss_corr_cls * w_cls) + (loss_corr_reg * w_cls)

        if cov_criterion is not None and covariance_weight > 0.0:
            loss_cov = cov_criterion(z)
            loss = loss + (covariance_weight * loss_cov)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        used_batches += 1

    if used_batches == 0:
        return 0.0
    return total_loss / used_batches


def test(model, loader, device, beta, age_label_index=1):
    model.eval()
    model.training = False

    total_loss = 0.0
    used_batches = 0

    with torch.no_grad():
        for data in loader:
            x = data.x.to(device)
            if torch.isnan(x).any().item():
                continue

            y = data.y.to(device)
            pred, mu, log_var, re, re_2 = model(x)

            if torch.isnan(re_2).any().item():
                continue

            # Keep age regression head active in validation stats.
            _ = F.mse_loss(re_2, y[:, :, age_label_index], reduction='mean')
            total_loss += float(loss_function(x, pred, mu, log_var, beta).item())
            used_batches += 1

    if used_batches == 0:
        return 0.0
    return total_loss / used_batches


def eval_error(model, test_loader, device, meshdata, out_dir):
    model.eval()
    model.training = False

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data.x.to(device)
            pred, mu, log_var, re, re_2 = model(x)
            num_graphs = data.num_graphs
            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean

            reshaped_pred *= 300
            reshaped_x *= 300

            tmp_error = torch.sqrt(
                torch.sum((reshaped_pred - reshaped_x) ** 2, dim=2)
            )
            errors.append(tmp_error)

        new_errors = torch.cat(errors, dim=0)
        mean_error = new_errors.view((-1,)).mean()
        std_error = new_errors.view((-1,)).std()
        median_error = new_errors.view((-1,)).median()

    message = 'Euclidean Error: {:.3f}+{:.3f} | {:.3f}'.format(
        mean_error,
        std_error,
        median_error,
    )

    out_error_fp = out_dir + '/euc_errors.txt'
    with open(out_error_fp, 'a') as log_file:
        log_file.write('{:s}\n'.format(message))

    print('')
    print('')
    print(message)
    print('')
    print('')

    return mean_error
