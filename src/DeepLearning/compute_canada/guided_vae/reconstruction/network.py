import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from conv import SpiralConv


def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


def _resolve_device(device_spec):
    if isinstance(device_spec, torch.device):
        return device_spec
    if isinstance(device_spec, int):
        return torch.device('cuda', int(device_spec))
    if isinstance(device_spec, str):
        text = device_spec.strip().lower()
        if text.startswith('cuda'):
            return torch.device(text)
        return torch.device('cuda', int(text))
    raise ValueError(f'Unsupported device spec: {device_spec}')


def _move_spiral_indices_in_module(module, device):
    if isinstance(module, SpiralEnblock):
        module.conv.indices = module.conv.indices.to(device)
    elif isinstance(module, SpiralDeblock):
        module.conv.indices = module.conv.indices.to(device)
    elif isinstance(module, SpiralConv):
        module.indices = module.indices.to(device)


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out


class AE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels,
                 spiral_indices, down_transform, up_transform, training=True):
        super(AE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)
        self.training = training

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.en_layers.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
        self.en_layers.append(nn.Linear(self.num_vert * out_channels[-1], 2*latent_channels))
        #self.en_layers.append(nn.Linear(8*latent_channels, 2*latent_channels))

        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx - 1],
                                  out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
            else:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx], out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
        self.de_layers.append(
            SpiralConv(out_channels[0], in_channels, self.spiral_indices[0]))

        self.reset_parameters()

	    # Excitation
        self.cls_sq = nn.Sequential(
            nn.Linear(1, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid())

        # Excitation 2
        self.reg_sq_2 = nn.Sequential(
            nn.Linear(1, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 1))

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encoder(self, x):
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        
        mu = x[:, :self.latent_channels]
        log_var = x[:, self.latent_channels:]

        return mu, log_var

    def decoder(self, x):
        num_layers = len(self.de_layers)
        num_features = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i])
            else:
                x = layer(x)
        return x

    def reparameterize(self, mu, log_var):
        if self.training:
            #log_var = log_var.clamp(max=10)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + eps * std
        else:
            sample = mu

        return sample

    def cls(self, z): # first excitation
        z = torch.split(z, 1, 1)[0]
        return self.cls_sq(z)

    def reg_2(self, z): # second excitation
        z = torch.split(z, 1, 1)[1]
        return self.reg_sq_2(z)    

    def forward(self, x, *indices):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var, self.cls(z), self.reg_2(z)


class AEModelParallel(AE):
    """Model-parallel AE across multiple CUDA devices.

    Stage layout:
    - GPU0: early encoder blocks
    - GPU1: late encoder + bottleneck + early decoder
    - GPU2: late decoder
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        latent_channels,
        spiral_indices,
        down_transform,
        up_transform,
        device_ids,
        training=True,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError('AEModelParallel requires CUDA.')
        if device_ids is None or len(device_ids) < 2:
            raise ValueError('AEModelParallel requires at least 2 CUDA devices.')

        self.mp_devices = [_resolve_device(d) for d in device_ids]
        self.input_device = self.mp_devices[0]
        self.latent_device = self.mp_devices[1] if len(self.mp_devices) > 1 else self.mp_devices[0]
        self.output_device = self.mp_devices[-1]

        super(AEModelParallel, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            spiral_indices=spiral_indices,
            down_transform=down_transform,
            up_transform=up_transform,
            training=training,
        )

        self.model_parallel = True
        self.n_en_blocks = len(self.out_channels)
        self.enc_split = max(1, min(self.n_en_blocks - 1, self.n_en_blocks // 2))
        self.dec_split = max(1, min(len(self.de_layers) - 2, self.n_en_blocks // 2))

        self._place_modules()
        self._place_runtime_tensors()

    def _encoder_device(self, idx):
        if idx < self.enc_split:
            return self.input_device
        return self.latent_device

    def _decoder_device(self, idx):
        if idx <= self.dec_split:
            return self.latent_device
        return self.output_device

    def _place_modules(self):
        # Encoder blocks
        for idx in range(self.n_en_blocks):
            dev = self._encoder_device(idx)
            layer = self.en_layers[idx].to(dev)
            _move_spiral_indices_in_module(layer, dev)
            self.en_layers[idx] = layer

        # Encoder bottleneck linear
        self.en_layers[-1] = self.en_layers[-1].to(self.latent_device)

        # Decoder blocks (first linear + deblocks + final conv)
        for idx in range(len(self.de_layers)):
            dev = self._decoder_device(idx)
            layer = self.de_layers[idx].to(dev)
            _move_spiral_indices_in_module(layer, dev)
            self.de_layers[idx] = layer

        self.cls_sq = self.cls_sq.to(self.latent_device)
        self.reg_sq_2 = self.reg_sq_2.to(self.latent_device)

    def _place_runtime_tensors(self):
        self.en_down_transform = []
        for idx in range(self.n_en_blocks):
            dev = self._encoder_device(idx)
            self.en_down_transform.append(self.down_transform[idx].to(dev))

        self.de_up_transform = {}
        num_layers = len(self.de_layers)
        num_features = num_layers - 2
        for layer_idx in range(1, num_layers - 1):
            up_idx = num_features - layer_idx
            dev = self._decoder_device(layer_idx)
            self.de_up_transform[layer_idx] = self.up_transform[up_idx].to(dev)

    def encoder(self, x):
        for idx, layer in enumerate(self.en_layers):
            if idx != len(self.en_layers) - 1:
                target_dev = self._encoder_device(idx)
                if x.device != target_dev:
                    x = x.to(target_dev, non_blocking=True)
                x = layer(x, self.en_down_transform[idx])
            else:
                if x.device != self.latent_device:
                    x = x.to(self.latent_device, non_blocking=True)
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)

        mu = x[:, :self.latent_channels]
        log_var = x[:, self.latent_channels:]
        return mu, log_var

    def decoder(self, x):
        num_layers = len(self.de_layers)
        for idx, layer in enumerate(self.de_layers):
            target_dev = self._decoder_device(idx)
            if x.device != target_dev:
                x = x.to(target_dev, non_blocking=True)

            if idx == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif idx != num_layers - 1:
                x = layer(x, self.de_up_transform[idx])
            else:
                x = layer(x)
        return x

    def cls(self, z):
        if z.device != self.latent_device:
            z = z.to(self.latent_device, non_blocking=True)
        z = torch.split(z, 1, 1)[0]
        return self.cls_sq(z)

    def reg_2(self, z):
        if z.device != self.latent_device:
            z = z.to(self.latent_device, non_blocking=True)
        if z.size(1) < 2:
            return torch.zeros((z.size(0), 1), device=z.device, dtype=z.dtype)
        z = torch.split(z, 1, 1)[1]
        return self.reg_sq_2(z)

    def forward(self, x, *indices):
        if x.device != self.input_device:
            x = x.to(self.input_device, non_blocking=True)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var, self.cls(z), self.reg_2(z)

# Inhibition classifier
class Classifier(nn.Module):
    def __init__(self, n_vae_dis):
        super(Classifier, self).__init__()

        self.cls_sq_n = nn.Sequential(
            nn.Linear(n_vae_dis - 1, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.cls_sq_n(x)

# Inhibition
class Regressor(nn.Module):
    def __init__(self, n_vae_dis):
        super(Regressor, self).__init__()

        self.reg_sq = nn.Sequential(
            nn.Linear(n_vae_dis - 1, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.reg_sq(x)
