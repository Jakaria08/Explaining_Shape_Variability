import torch
import torch.nn as nn


class AdaptiveSpiralConv(nn.Module):
    """Spiral convolution with per-neighbor adaptive gating.

    It keeps the same input/output behavior as SpiralConv:
    - Input: [N, C] or [B, N, C]
    - Output: [N, C_out] or [B, N, C_out]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        indices,
        dim=1,
        hidden_channels=32,
        dropout=0.0,
        use_global_context=False,
    ):
        super(AdaptiveSpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)
        self.use_global_context = bool(use_global_context)

        hidden_channels = int(hidden_channels)
        if hidden_channels < 1:
            raise ValueError(f'hidden_channels must be >= 1, got {hidden_channels}.')
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f'dropout must be in [0, 1), got {dropout}.')

        gate_in_channels = in_channels * (2 if self.use_global_context else 1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_channels, hidden_channels),
            nn.ELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden_channels, self.seq_length),
        )
        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.gate_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.constant_(self.layer.bias, 0)

    def _compute_gates(self, x_center):
        if self.use_global_context:
            if x_center.dim() == 2:
                global_context = x_center.mean(dim=0, keepdim=True).expand_as(x_center)
            else:
                global_context = x_center.mean(dim=1, keepdim=True).expand_as(x_center)
            gate_in = torch.cat([x_center, global_context], dim=-1)
        else:
            gate_in = x_center
        return torch.sigmoid(self.gate_mlp(gate_in))

    def forward(self, x):
        n_nodes, _ = self.indices.size()

        if x.dim() == 2:
            x_neighbors = torch.index_select(x, 0, self.indices.view(-1))
            x_neighbors = x_neighbors.view(n_nodes, self.seq_length, self.in_channels)
            gates = self._compute_gates(x).unsqueeze(-1)
            x_neighbors = x_neighbors * gates
            x_flat = x_neighbors.view(n_nodes, -1)
            return self.layer(x_flat)

        if x.dim() == 3:
            if self.dim != 1:
                raise RuntimeError(
                    f'AdaptiveSpiralConv expects dim=1 for batched input, got dim={self.dim}.'
                )
            bs = x.size(0)
            x_neighbors = torch.index_select(x, self.dim, self.indices.view(-1))
            x_neighbors = x_neighbors.view(bs, n_nodes, self.seq_length, self.in_channels)
            gates = self._compute_gates(x).unsqueeze(-1)
            x_neighbors = x_neighbors * gates
            x_flat = x_neighbors.view(bs, n_nodes, -1)
            return self.layer(x_flat)

        raise RuntimeError(f'x.dim() is expected to be 2 or 3, but received {x.dim()}')

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.in_channels}, {self.out_channels}, '
            f'seq_length={self.seq_length}, '
            f'global_context={self.use_global_context})'
        )
