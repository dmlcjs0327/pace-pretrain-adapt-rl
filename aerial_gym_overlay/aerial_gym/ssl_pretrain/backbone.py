"""Backbone used by pretraining: the encoder MLP and the GRU of PretrainAdaptPolicy, built with the same layer sizes and activations so that the pretrained weights load into the policy."""
import torch
import torch.nn as nn

def _elu_mlp(sizes, last_act=True):
    """MLP with ELU activations, as _mlp of PretrainAdaptPolicy."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if last_act or i < len(sizes) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)

class ProprioGRUBackbone(nn.Module):
    """Encoder MLP followed by the GRU, the backbone part of PretrainAdaptPolicy."""

    def __init__(self, proprio_dim=13, proprio_hidden=64, fusion_hidden=128, fusion_layers=2, gru_hidden=128):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.proprio_hidden = proprio_hidden
        self.fusion_hidden = fusion_hidden
        self.gru_hidden = gru_hidden

        self.proprio_mlp = _elu_mlp([proprio_dim, proprio_hidden, proprio_hidden])
        fusion_sizes = [proprio_hidden] + [fusion_hidden] * fusion_layers
        self.fusion_mlp = _elu_mlp(fusion_sizes)
        self.gru = nn.GRU(
            input_size=fusion_hidden,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x, h0=None):
        """
        x  : (B, T, proprio_dim)
        h0 : (1, B, gru_hidden) or None
        Returns:
            h  : (B, T, gru_hidden)
            h_n: (1, B, gru_hidden)
        """
        p = self.proprio_mlp(x)         # (B, T, 64)
        z = self.fusion_mlp(p)          # (B, T, 128)
        h, h_n = self.gru(z, h0)        # (B, T, 128)
        return h, h_n
