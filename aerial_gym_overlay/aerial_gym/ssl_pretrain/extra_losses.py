"""Additional pretraining objectives of Study 2 v2.

All take the GRU output z (B, T, 128) of a window of standardized states x (B, T, 13) and the standardized
executed commands u (B, T, 4), where row t stores the command applied at state t.
  IDM  inverse dynamics as in Fan et al. (RA-L 2026): predict the command a_t from z_t and z_{t+1}, i.e. from
       the history up to the next state
  FDM  forward dynamics: predict the standardized state x_{t+k} from z_t and the commands a_t .. a_{t+k-1},
       k in {1, 2, 4}
  BC   behavior cloning of the data collector: the actor head of the policy (Linear 128->64, ELU, Linear 64->4)
       predicts the executed command a_t in command units from z_t; the head is transferred to RL
"""
import torch
import torch.nn as nn

def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)

class InverseDynamicsLoss(nn.Module):
    def __init__(self, z_dim=128, a_dim=4):
        super().__init__()
        self.head = mlp([2 * z_dim, 256, 256, a_dim])

    def forward_with_z_full(self, z, u):
        pred = self.head(torch.cat([z[:, :-1], z[:, 1:]], dim=-1))
        return ((pred - u[:, :-1]) ** 2).mean()

class ForwardDynamicsLoss(nn.Module):
    def __init__(self, z_dim=128, a_dim=4, s_dim=13, steps=(1, 2, 4)):
        super().__init__()
        self.steps = steps
        self.heads = nn.ModuleList([mlp([z_dim + k * a_dim, 256, 256, s_dim]) for k in steps])

    def forward_with_z_full(self, z, x, u):
        T = z.shape[1]
        loss = 0.0
        for k, h in zip(self.steps, self.heads):
            n = T - k
            a_cat = torch.cat([u[:, j:j + n] for j in range(k)], dim=-1)
            pred = h(torch.cat([z[:, :n], a_cat], dim=-1))
            loss = loss + ((pred - x[:, k:k + n]) ** 2).mean()
        return loss / len(self.steps)

class BehaviorCloningLoss(nn.Module):
    """Actor head with the names and shapes of the policy, so that it can be loaded into the RL actor."""

    def __init__(self, z_dim=128, hidden=64, a_dim=4, a_mean=None, a_std=None):
        super().__init__()
        self.actor_trunk = nn.Sequential(nn.Linear(z_dim, hidden), nn.ELU())
        self.actor_mu = nn.Linear(hidden, a_dim)
        self.register_buffer("a_mean", torch.as_tensor(a_mean, dtype=torch.float32))
        self.register_buffer("a_std", torch.as_tensor(a_std, dtype=torch.float32))

    def forward_with_z_full(self, z, u):
        target = u * self.a_std + self.a_mean  # executed command in command units, as the RL actor outputs
        return ((self.actor_mu(self.actor_trunk(z)) - target) ** 2).mean()
