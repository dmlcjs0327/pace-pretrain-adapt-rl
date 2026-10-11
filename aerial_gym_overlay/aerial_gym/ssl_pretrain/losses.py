"""Pretraining objectives. Each loss takes the latent sequence of the backbone with the states and commands of a window and returns a scalar. The temporary heads are discarded after pretraining."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ForwardDynamicsLoss(nn.Module):
    """P_FD: predict z_{t+1} from (z_t, a_t). MSE on latent space."""

    def __init__(self, hidden_dim=128, action_dim=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
        )

    def forward(self, z, action_seq, state_seq):
        # z: (B, T, H); action_seq: (B, T+1, A); state_seq: (B, T+1, 13)
        raise NotImplementedError("Use forward_with_z_full instead")

    def forward_with_z_full(self, z_full, action_seq):
        """
        z_full: (B, T+1, H) — backbone output for full window
        """
        B, Tp1, H = z_full.shape
        T = Tp1 - 1
        z_t = z_full[:, :T]                          # (B, T, H)
        a_t = action_seq[:, 1:T+1]                   # (B, T, A) — action that drives z_t -> z_{t+1}
        # Target = z[t+1] (detached).
        z_target = z_full[:, 1:].detach()            # (B, T, H)
        x = torch.cat([z_t, a_t], dim=-1)            # (B, T, H+A)
        z_pred = self.head(x)                        # (B, T, H)
        loss = F.mse_loss(z_pred, z_target)
        return loss

class InverseDynamicsLoss(nn.Module):
    """P_IDM: predict a_t from (z_t, z_{t+1}). MSE on continuous action."""

    def __init__(self, hidden_dim=128, action_dim=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
        )

    def forward_with_z_full(self, z_full, action_seq):
        """
        z_full: (B, T+1, H), action_seq: (B, T+1, A)
        """
        B, Tp1, H = z_full.shape
        T = Tp1 - 1
        z_t = z_full[:, :T]                          # (B, T, H)
        z_tp1 = z_full[:, 1:]                        # (B, T, H)
        a_target = action_seq[:, 1:T+1]              # (B, T, A) — actual action from t to t+1
        x = torch.cat([z_t, z_tp1], dim=-1)          # (B, T, 2H)
        a_pred = self.head(x)                        # (B, T, A)
        loss = F.mse_loss(a_pred, a_target)
        return loss

class MaskedStateLoss(nn.Module):
    """MAE: reconstruct the hidden entries of each state from the latent."""

    def __init__(self, hidden_dim=128, state_dim=13, mask_ratio=0.3):
        super().__init__()
        self.state_dim = state_dim
        self.mask_ratio = mask_ratio
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, state_dim),
        )

    def forward_with_z_full(self, z_full, state_seq, mask):
        """
        state_seq: (B, T+1, 13) — original (non-masked) target
        mask: (B, T+1, 13) bool — True = masked
        """
        s_pred = self.head(z_full)                   # (B, T+1, 13)
        # Loss only on masked dims
        loss = ((s_pred - state_seq) ** 2 * mask.float()).sum() / (mask.float().sum() + 1e-8)
        return loss

class ContrastivePredictiveLoss(nn.Module):
    """P_CPC: InfoNCE on (z_t, z_{t+k}) for multiple horizons k.

    """

    def __init__(self, hidden_dim=128, predict_steps=(1, 2, 4, 8), temperature=0.1, n_max=8192):
        super().__init__()
        self.predict_steps = predict_steps
        self.temperature = temperature
        self.n_max = n_max
        self.W = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in predict_steps
        ])

    def forward_with_z_full(self, z_full):
        B, Tp1, H = z_full.shape
        total_loss = 0.0
        n_terms = 0
        for i, k in enumerate(self.predict_steps):
            if k >= Tp1:
                continue
            z_t = z_full[:, :Tp1 - k]                # (B, T-k, H)
            z_tk = z_full[:, k:]                     # (B, T-k, H)
            z_t_proj = self.W[i](z_t)                # (B, T-k, H)
            BL, T_prime, _ = z_t_proj.shape
            z_t_flat = z_t_proj.reshape(BL * T_prime, H)
            z_tk_flat = z_tk.reshape(BL * T_prime, H)
            N = z_t_flat.shape[0]
            if N > self.n_max:
                idx = torch.randperm(N, device=z_t_flat.device)[:self.n_max]
                z_t_flat = z_t_flat[idx]
                z_tk_flat = z_tk_flat[idx]
            logits = z_t_flat @ z_tk_flat.T          # (N', N')
            logits = logits / self.temperature
            labels = torch.arange(z_t_flat.shape[0], device=z_t_flat.device)
            loss_k = F.cross_entropy(logits, labels)
            total_loss = total_loss + loss_k
            n_terms += 1
        return total_loss / max(n_terms, 1)

class QuasiMetricLoss(nn.Module):
    """P_QM (IQE): Interval Quasimetric Embedding (Wang & Isola, ICML 2023).

    IQE form:
      d_IQE(x, y) = (1/K) Σ_k max_d ReLU(x[k,d] − y[k,d])
      universal class. Asymmetric: d(x,y) ≠ d(y,x).

    Pretrain target:
      MSE regression. Reward-free, negative-free, strict offline.

      - Wang & Isola 2023 "Optimal Goal-Reaching RL via Quasimetric Learning" (ICML)
      - Myers 2024 "Learning Temporal Distances: Contrastive Successor Features" (ICML)
      - Eysenbach 2022 "Contrastive RL as GCRL" (NeurIPS)
      - Pong 2018 "Temporal Distance Models" (ICLR) — predecessor
    """

    def __init__(self, hidden_dim=128, n_components=8, dim_per_comp=8,
                 n_pairs_per_seq=16, time_scale=1.0):
        super().__init__()
        self.K = n_components
        self.D = dim_per_comp
        self.n_pairs_per_seq = n_pairs_per_seq
        self.time_scale = time_scale
        # IQE projection: hidden → K*D, reshaped to (..., K, D)
        proj_dim = n_components * dim_per_comp
        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def _iqe_distance(self, x, y):
        """Interval Quasimetric Embedding distance.
        x, y: (..., K*D) — flattened. Reshape to (..., K, D), then
        d(x,y) = mean_k max_d ReLU(x[k,d] - y[k,d]).
        Asymmetric, satisfies triangle inequality (Wang & Isola 2023).
        """
        shape = x.shape[:-1]
        x_kd = x.reshape(*shape, self.K, self.D)
        y_kd = y.reshape(*shape, self.K, self.D)
        # asymmetric: x going to y → max over d of ReLU(x_d - y_d)
        diff = F.relu(x_kd - y_kd)
        per_comp = diff.max(dim=-1).values   # (..., K)
        return per_comp.mean(dim=-1)         # (...)

    def forward_with_z_full(self, z_full):
        """
        z_full: (B, T+1, H)
        """
        B, Tp1, H = z_full.shape
        device = z_full.device
        N = self.n_pairs_per_seq

        i_idx = torch.randint(0, Tp1, (B, N), device=device)
        max_gap = (Tp1 - 1 - i_idx).clamp(min=0)
        frac = torch.rand((B, N), device=device)
        g_idx = (frac * (max_gap.float() + 1.0)).long().clamp_max(max_gap)
        j_idx = i_idx + g_idx
        target_gap = g_idx.float() * self.time_scale

        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
        z_i = z_full[b_idx, i_idx]                                       # (B, N, H)
        z_j = z_full[b_idx, j_idx]                                       # (B, N, H)
        e_i = self.phi(z_i)                                              # (B, N, K*D)
        e_j = self.phi(z_j)                                              # (B, N, K*D)
        dist = self._iqe_distance(e_i, e_j)                              # (B, N)
        loss = F.mse_loss(dist, target_gap)
        return loss

class SelfPredictiveLoss(nn.Module):
    """P_SPR: Self-Predictive Representations (Schwarzer et al., NeurIPS 2021).

      For k = 1..K:
        target = ema_target_encoder(s_{t+k}).detach()
        loss_k = -cosine_similarity(predict(project(ẑ)), project(target))
      total_loss = Σ_k loss_k

      - Schwarzer et al. NeurIPS 2021 "Data-Efficient RL with Self-Predictive Representations"
      - Grill et al. NeurIPS 2020 "BYOL — Bootstrap Your Own Latent"
      - Guo et al. NeurIPS 2022 "BYOL-Explore"
    """

    def __init__(self, hidden_dim=128, action_dim=4, predict_steps=(1, 2, 3, 5),
                 proj_dim=64):
        super().__init__()
        self.predict_steps = predict_steps
        self.action_dim = action_dim
        # Per-step transition heads (Schwarzer uses a single shared transition model;
        self.transition_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + action_dim * k, 128),
                nn.GELU(),
                nn.Linear(128, hidden_dim),
            ) for k in predict_steps
        ])
        # BYOL-style projection + prediction
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward_with_z_full(self, z_full_online, z_full_target, action_seq):
        """
        z_full_online: (B, T+1, H) from online backbone
        z_full_target: (B, T+1, H) from EMA target backbone (no_grad)
        action_seq:    (B, T+1, A)
        """
        B, Tp1, H = z_full_online.shape
        total_loss = 0.0
        n_terms = 0
        for i, k in enumerate(self.predict_steps):
            if k >= Tp1:
                continue
            z_t = z_full_online[:, :Tp1 - k]                # (B, T-k, H)
            T_prime = z_t.shape[1]
            # Commands a_t .. a_{t+k-1} that move the vehicle from s_t to s_{t+k} (row t stores the command
            # a_{t+1} .. a_{t+k}, which dropped a_t and included a command computed from s_{t+k}.
            a_chunks = []
            for kk in range(k):
                a_chunks.append(action_seq[:, kk:kk + T_prime])
            a_cat = torch.cat(a_chunks, dim=-1)             # (B, T-k, k*A)

            x = torch.cat([z_t, a_cat], dim=-1)
            z_pred = self.transition_heads[i](x)             # (B, T-k, H)

            # BYOL-style: predictor(projector(online)) vs sg(projector(target))
            online = self.predictor(self.projector(z_pred))  # (B, T-k, P)
            with torch.no_grad():
                target = self.projector(z_full_target[:, k:k + T_prime])  # (B, T-k, P)
            online = F.normalize(online, dim=-1)
            target = F.normalize(target, dim=-1)
            loss_k = (2 - 2 * (online * target).sum(dim=-1)).mean()  # BYOL form
            total_loss = total_loss + loss_k
            n_terms += 1
        return total_loss / max(n_terms, 1)

class PIDMLoss(nn.Module):
    """P_PIDM: Proprioceptive Inverse Dynamics Model (Vandevoorde et al., 2025).

          target = K-step future actions (a_t, a_{t+1}, ..., a_{t+K-1})

    """

    def __init__(self, hidden_dim=128, action_dim=4, predict_steps=4):
        super().__init__()
        self.predict_steps = predict_steps
        self.action_dim = action_dim
        # K-step action prediction head — 4-layer MLP (Vandevoorde 2025 spec)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_dim * predict_steps),
        )

    def forward_with_z_full(self, z_full, action_seq):
        """
        z_full: (B, T+1, H) — GRU output, history-encoded
        action_seq: (B, T+1, A)
        """
        B, Tp1, H = z_full.shape
        K = self.predict_steps
        if Tp1 - 1 < K:
            return torch.tensor(0.0, device=z_full.device, requires_grad=True)
        T_eff = Tp1 - K
        z_t = z_full[:, :T_eff]                              # (B, T_eff, H)
        a_pred = self.head(z_t)                              # (B, T_eff, A*K)
        a_pred = a_pred.reshape(B, T_eff, K, self.action_dim)
        a_targets = []
        for k in range(K):
            a_targets.append(action_seq[:, 1+k:1+k+T_eff])    # (B, T_eff, A)
        a_target = torch.stack(a_targets, dim=2)             # (B, T_eff, K, A)
        return F.mse_loss(a_pred, a_target)

class VICRegLoss(nn.Module):
    """P_VICReg: Variance-Invariance-Covariance Regularization (Bardes et al., ICLR 2022).

      L_covariance = sum_offdiag(cov(Z))          — feature decorrelation

      L_total = λ_inv·L_inv + λ_var·L_var + λ_cov·L_cov
      Default: λ_inv=25, λ_var=25, λ_cov=1 (Bardes 2022)
    """

    def __init__(self, hidden_dim=128, proj_dim=64,
                 lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, proj_dim),
        )

    def _vicreg_loss(self, Z: torch.Tensor, Zp: torch.Tensor) -> torch.Tensor:
        """Shared VICReg (invariance + variance + covariance) computation
        for both the canonical augmented-view path and the legacy
        temporal-pair path. Both inputs are projected (N, D)."""
        N, D = Z.shape
        l_inv = F.mse_loss(Z, Zp)
        std_Z = torch.sqrt(Z.var(dim=0) + 1e-4)
        std_Zp = torch.sqrt(Zp.var(dim=0) + 1e-4)
        l_var = (F.relu(1.0 - std_Z).mean() + F.relu(1.0 - std_Zp).mean()) / 2.0
        Z_c = Z - Z.mean(dim=0, keepdim=True)
        Zp_c = Zp - Zp.mean(dim=0, keepdim=True)
        cov_Z = (Z_c.T @ Z_c) / max(N - 1, 1)
        cov_Zp = (Zp_c.T @ Zp_c) / max(N - 1, 1)
        eye = torch.eye(D, device=Z.device)
        l_cov = ((cov_Z * (1 - eye)).pow(2).sum() / D
                 + (cov_Zp * (1 - eye)).pow(2).sum() / D) / 2.0
        return self.lambda_inv * l_inv + self.lambda_var * l_var + self.lambda_cov * l_cov

    def forward_with_z_full(self, z_full):
        """LEGACY temporal-pair form (z_t, z_{t+1}). Kept for back-compat with
        the offline pretrain trainer that produces a single z_full sequence.
        Canonical VICReg [Bardes 2022] uses augmented views, not temporal
        adjacency; for the AuxJoint online path see forward_with_aug_views
        below — the network calls that variant when wired correctly."""
        B, Tp1, H = z_full.shape
        if Tp1 < 2:
            return torch.tensor(0.0, device=z_full.device, requires_grad=True)
        z_t = z_full[:, :Tp1 - 1].reshape(-1, H)
        z_tp1 = z_full[:, 1:].reshape(-1, H)
        Z = self.projector(z_t)
        Zp = self.projector(z_tp1)
        return self._vicreg_loss(Z, Zp)

    def forward_with_aug_views(self, z_view_a: torch.Tensor, z_view_b: torch.Tensor) -> torch.Tensor:
        """Canonical Bardes 2022 form — two augmented views of the SAME obs.
        z_view_a, z_view_b: (B, T, H) backbone latents from two augmented
        forward passes. We flatten time into the batch and apply the
        invariance / variance / covariance triad."""
        Ba, Ta, Ha = z_view_a.shape
        Bb, Tb, Hb = z_view_b.shape
        assert (Ba, Ta, Ha) == (Bb, Tb, Hb), "augmented views must match shape"
        Z = self.projector(z_view_a.reshape(-1, Ha))
        Zp = self.projector(z_view_b.reshape(-1, Hb))
        return self._vicreg_loss(Z, Zp)

class BYOLLoss(nn.Module):
    """P_BYOL: Bootstrap Your Own Latent (Grill et al., NeurIPS 2020) — single-step variant.

      - SPR: multi-step prediction + transition heads + EMA target
      - BYOL: single-step (z_t → z_{t+1}) + no transition + EMA target + predictor

    Pure BYOL form: predictor(projector(z_t)) ≈ projector(z_{t+1}^EMA).detach()
    """

    def __init__(self, hidden_dim=128, proj_dim=64):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, 256),
            nn.GELU(),
            nn.Linear(256, proj_dim),
        )

    def forward_with_z_full(self, z_full_online, z_full_target):
        """z_full_target holds the latents of the EMA target backbone, which the trainer computes."""
        B, Tp1, H = z_full_online.shape
        if Tp1 < 2:
            return torch.tensor(0.0, device=z_full_online.device, requires_grad=True)
        z_t = z_full_online[:, :Tp1 - 1].reshape(-1, H)
        z_tp1_target = z_full_target[:, 1:].reshape(-1, H)
        online = self.predictor(self.projector(z_t))
        with torch.no_grad():
            target = self.projector(z_tp1_target)
        online = F.normalize(online, dim=-1)
        target = F.normalize(target, dim=-1)
        return (2 - 2 * (online * target).sum(dim=-1)).mean()

class FusionLoss(nn.Module):
    """

      L_fusion = α·L_FD + β·L_IDM + γ·L_MAE

      - Sun et al. ICLR 2023 "SMART: Self-supervised Multi-task pretrAining with
        contRol Transformers"
      - Pathak 2017 ICM (FD + IDM only)
      - DynaMo (Cui 2024) — independent FD + IDM separate models
    """

    def __init__(self, hidden_dim=128, action_dim=4, state_dim=13,
                 alpha=1.0, beta=1.0, gamma=0.5, mask_ratio=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mask_ratio = mask_ratio
        self.fd = ForwardDynamicsLoss(hidden_dim, action_dim)
        self.idm = InverseDynamicsLoss(hidden_dim, action_dim)
        self.mae = MaskedStateLoss(hidden_dim, state_dim, mask_ratio=mask_ratio)

    def forward_with_z_full(self, z_full_clean, z_full_masked, state_seq, action_seq, mask):
        """
        z_full_clean:  (B, T+1, H) backbone(state_seq) — used for FD/IDM
        z_full_masked: (B, T+1, H) backbone(state_seq * ~mask) — used for MAE
        state_seq, action_seq, mask: as expected by individual losses
        """
        l_fd = self.fd.forward_with_z_full(z_full_clean, action_seq)
        l_idm = self.idm.forward_with_z_full(z_full_clean, action_seq)
        l_mae = self.mae.forward_with_z_full(z_full_masked, state_seq, mask)
        return self.alpha * l_fd + self.beta * l_idm + self.gamma * l_mae
