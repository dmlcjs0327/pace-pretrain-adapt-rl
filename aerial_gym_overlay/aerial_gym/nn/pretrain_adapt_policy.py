"""Framework-agnostic pretrain-adapt policy (proposal §4.2).

Architecture (configurable for simple env / navigation_task):
    obs (proprio + optional VAE latent)
      ├── ProprioMLP  (proprio_dim → 64 → 64)
      └── VAEProjMLP  (vae_dim → 64 → 64)        [active iff use_vae_branch]
    concat → fusion_input_dim (64 or 128)
      └── FusionMLP  (fusion_input_dim → 128 → 128)   ← output z_t (aux loss anchor)
        └── (optional) GRU (hidden 128, 1 layer)
          ├── Actor head  (128 → 64 → action_dim)
          └── Critic head (128 → 64 → 1)

Returned by `forward_with_extras()`:
    mu, log_std (or sigma), value, next_hidden, z_t

z_t is exposed so aux loss modules (IDM / CPC|A / RP) can attach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

@dataclass
class PolicyConfig:
    proprio_dim: int = 17           # navigation_task: 13+4=17, simple: 17 (zero-padded)
    vae_dim: int = 64               # navigation_task VAE latent
    use_vae_branch: bool = False    # False for simple env, True for navigation_task
    proprio_hidden: int = 64
    vae_proj_hidden: int = 64
    fusion_hidden: int = 128
    fusion_layers: int = 2          # number of Linear layers in fusion MLP
    use_gru: bool = True
    gru_hidden: int = 128
    head_hidden: int = 64
    action_dim: int = 4
    log_std_init: float = 0.0       # initial log std, fixed sigma
    fixed_sigma: bool = True
    use_asymmetric_backbone: bool = False
    # Section 25 of the Study 2 pre-registration (off by default, so every earlier run is unchanged):
    # V9: each backbone of the asymmetric policy keeps its own GRU state (rnn state = (actor, critic)).
    asym_separate_memory: bool = False
    # V8a: fixed whitening (h - feat_mu) feat_W^T of the backbone output before the heads (buffers set by the builder).
    feat_whiten: bool = False
    # V8b: the (normalized) observation is concatenated to the backbone output at the heads' input.
    skip_obs: bool = False
    # CNN visual encoder branch.
    # When True, obs layout = [proprio | flatten(1×H×W depth)] and the policy
    # has a trainable CNN (DepthCNN) instead of (or in addition to) the
    # frozen-VAE-latent branch. Enables gradients from CURL InfoNCE / SPR
    # multi-step latent prediction to propagate into the visual encoder —
    # the canonical AuxJoint-paradigm property that the use_vae_branch
    # setup cannot reproduce. use_vae_branch and use_cnn_vision_branch are
    # mutually exclusive in the current implementation.
    use_cnn_vision_branch: bool = False
    vision_input_h: int = 32        # downsampled depth height
    vision_input_w: int = 32        # downsampled depth width
    vision_in_channels: int = 1
    vision_latent: int = 64         # CNN output latent dim (matches vae_dim spec)
    vision_cnn_channels: Tuple[int, ...] = (16, 32, 32)

def _mlp(sizes, activation=nn.ELU, last_act=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if last_act or i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)

class PretrainAdaptPolicy(nn.Module):
    """Two-branch policy with explicit z_t exposure for auxiliary losses.

    Two backbone modes:
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.is_asymmetric = bool(cfg.use_asymmetric_backbone)

        if not self.is_asymmetric:
            self.proprio_mlp = _mlp([cfg.proprio_dim, cfg.proprio_hidden, cfg.proprio_hidden])
            # Vision branch: choose between (a) frozen-VAE-latent projection,
            # (b) trainable CNN visual encoder, (c) none. Mutually exclusive.
            if cfg.use_vae_branch and cfg.use_cnn_vision_branch:
                raise ValueError("use_vae_branch and use_cnn_vision_branch are mutually exclusive")
            if cfg.use_vae_branch:
                self.vae_proj_mlp = _mlp([cfg.vae_dim, cfg.vae_proj_hidden, cfg.vae_proj_hidden])
                self.vision_cnn = None
                fusion_input = cfg.proprio_hidden + cfg.vae_proj_hidden
            elif cfg.use_cnn_vision_branch:
                from aerial_gym.nn.depth_cnn import DepthCNN
                self.vision_cnn = DepthCNN(
                    in_channels=cfg.vision_in_channels,
                    latent_dim=cfg.vision_latent,
                    input_height=cfg.vision_input_h,
                    input_width=cfg.vision_input_w,
                    channels=cfg.vision_cnn_channels,
                )
                self.vae_proj_mlp = None
                fusion_input = cfg.proprio_hidden + cfg.vision_latent
            else:
                self.vae_proj_mlp = None
                self.vision_cnn = None
                fusion_input = cfg.proprio_hidden
            fusion_sizes = [fusion_input] + [cfg.fusion_hidden] * cfg.fusion_layers
            self.fusion_mlp = _mlp(fusion_sizes)
            self.gru = (
                nn.GRU(input_size=cfg.fusion_hidden, hidden_size=cfg.gru_hidden,
                       num_layers=1, batch_first=True)
                if cfg.use_gru else None
            )
        else:
            self.actor_proprio_mlp = _mlp([cfg.proprio_dim, cfg.proprio_hidden, cfg.proprio_hidden])
            self.critic_proprio_mlp = _mlp([cfg.proprio_dim, cfg.proprio_hidden, cfg.proprio_hidden])
            if cfg.use_vae_branch:
                self.actor_vae_proj_mlp = _mlp([cfg.vae_dim, cfg.vae_proj_hidden, cfg.vae_proj_hidden])
                self.critic_vae_proj_mlp = _mlp([cfg.vae_dim, cfg.vae_proj_hidden, cfg.vae_proj_hidden])
                fusion_input = cfg.proprio_hidden + cfg.vae_proj_hidden
            else:
                self.actor_vae_proj_mlp = None
                self.critic_vae_proj_mlp = None
                fusion_input = cfg.proprio_hidden
            fusion_sizes = [fusion_input] + [cfg.fusion_hidden] * cfg.fusion_layers
            self.actor_fusion_mlp = _mlp(fusion_sizes)
            self.critic_fusion_mlp = _mlp(fusion_sizes)
            if cfg.use_gru:
                self.actor_gru = nn.GRU(input_size=cfg.fusion_hidden,
                                        hidden_size=cfg.gru_hidden, num_layers=1, batch_first=True)
                self.critic_gru = nn.GRU(input_size=cfg.fusion_hidden,
                                         hidden_size=cfg.gru_hidden, num_layers=1, batch_first=True)
            else:
                self.actor_gru = None
                self.critic_gru = None
            self.proprio_mlp = self.actor_proprio_mlp
            self.fusion_mlp = self.actor_fusion_mlp
            self.gru = self.actor_gru
            self.vae_proj_mlp = self.actor_vae_proj_mlp

        head_in = cfg.gru_hidden if cfg.use_gru else cfg.fusion_hidden
        if cfg.feat_whiten:  # V8a: identity until the builder fits the map on the pretraining data
            self.register_buffer("feat_mu", torch.zeros(head_in))
            self.register_buffer("feat_W", torch.eye(head_in))
        if cfg.skip_obs:  # V8b
            head_in = head_in + cfg.proprio_dim

        self.actor_trunk = _mlp([head_in, cfg.head_hidden], last_act=True)
        self.actor_mu = nn.Linear(cfg.head_hidden, cfg.action_dim)

        self.critic_trunk = _mlp([head_in, cfg.head_hidden], last_act=True)
        self.critic_v = nn.Linear(cfg.head_hidden, 1)

        if cfg.fixed_sigma:
            self.log_std = nn.Parameter(torch.ones(cfg.action_dim) * cfg.log_std_init)
        else:
            self.actor_log_std = nn.Linear(cfg.head_hidden, cfg.action_dim)

    # ----- helpers ---------------------------------------------------------

    def _split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Split obs into (proprio, vision_extra) per cfg.

        - use_vae_branch=True:  obs = [proprio | vae_latent(64)],   vision_extra = vae_latent
        - use_cnn_vision_branch=True: obs = [proprio | flatten(depth 32×32)=1024], vision_extra = flat depth
        - both False: vision_extra = None
        """
        if self.cfg.use_vae_branch:
            p = obs[..., : self.cfg.proprio_dim]
            v = obs[..., self.cfg.proprio_dim : self.cfg.proprio_dim + self.cfg.vae_dim]
            return p, v
        if self.cfg.use_cnn_vision_branch:
            p = obs[..., : self.cfg.proprio_dim]
            flat_len = self.cfg.vision_in_channels * self.cfg.vision_input_h * self.cfg.vision_input_w
            v = obs[..., self.cfg.proprio_dim : self.cfg.proprio_dim + flat_len]
            return p, v
        return obs, None

    def _encode_vision(self, vision_extra: torch.Tensor) -> torch.Tensor:
        """Encode vision_extra to a fixed-size latent.
        For use_vae_branch: project via vae_proj_mlp.
        For use_cnn_vision_branch: reshape to image + CNN forward.
        """
        if self.cfg.use_cnn_vision_branch and self.vision_cnn is not None:
            # vision_extra: (..., flat_len) → reshape (..., C, H, W)
            shape = vision_extra.shape[:-1]
            img = vision_extra.reshape(
                *shape, self.cfg.vision_in_channels,
                self.cfg.vision_input_h, self.cfg.vision_input_w,
            )
            # Flatten batch+time for CNN forward
            flat = img.reshape(-1, self.cfg.vision_in_channels,
                               self.cfg.vision_input_h, self.cfg.vision_input_w)
            v_flat = self.vision_cnn(flat)
            return v_flat.reshape(*shape, self.cfg.vision_latent)
        # use_vae_branch path
        return self.vae_proj_mlp(vision_extra)

    def _fuse(self, obs: torch.Tensor) -> torch.Tensor:
        proprio, vision_extra = self._split_obs(obs)
        p = self.proprio_mlp(proprio)
        if vision_extra is not None and (self.vae_proj_mlp is not None or self.vision_cnn is not None):
            v = self._encode_vision(vision_extra)
            x = torch.cat([p, v], dim=-1)
        else:
            x = p
        return self.fusion_mlp(x)

    def _fuse_branch(self, obs: torch.Tensor, branch: str) -> torch.Tensor:
        """Asymmetric fuse: branch ∈ {'actor', 'critic'}."""
        proprio, vae = self._split_obs(obs)
        proprio_mlp = getattr(self, f"{branch}_proprio_mlp")
        fusion_mlp = getattr(self, f"{branch}_fusion_mlp")
        vae_mlp = getattr(self, f"{branch}_vae_proj_mlp", None)
        p = proprio_mlp(proprio)
        if vae_mlp is not None and vae is not None:
            v = vae_mlp(vae)
            x = torch.cat([p, v], dim=-1)
        else:
            x = p
        return fusion_mlp(x)

    def head_input(self, h: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Input of the actor and critic heads: the backbone output, whitened (V8a) and/or with the observation (V8b).
        `obs` has the same leading shape and order as `h`."""
        if self.cfg.feat_whiten:
            h = (h - self.feat_mu) @ self.feat_W.T
        if self.cfg.skip_obs:
            h = torch.cat([h, obs[..., : self.cfg.proprio_dim]], dim=-1)
        return h

    def backbone_seq(
        self,
        obs_seq: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run proprio_mlp + fusion_mlp + GRU on a (B, T, obs_dim) sequence.

        Returns post-GRU hidden sequence (B, T, gru_hidden) — matches the semantics
        of ssl_pretrain.backbone.ProprioGRUBackbone (so M_Aux can apply pretrain
        SSL losses on the SAME representation layer that M_Pretrain learned).
        """
        B, T = obs_seq.shape[:2]
        flat = obs_seq.reshape(B * T, -1)
        z = self._fuse(flat).reshape(B, T, -1)               # (B, T, fusion_hidden)
        if self.gru is not None:
            h_out, _ = self.gru(z, rnn_state)                # (B, T, gru_hidden)
        else:
            h_out = z
        return h_out

    # ----- forward ---------------------------------------------------------

    def forward_with_extras(
        self,
        obs: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
        seq_first: bool = False,
    ) -> dict:
        """Single forward pass returning all components needed by PPO + aux losses.

        Args:
            obs: [B, obs_dim] or [T, B, obs_dim] when seq_first
            rnn_state: hidden state for GRU; expected shape [num_layers=1, B, gru_hidden]
            seq_first: if True, obs is [T, B, obs_dim] for BPTT

        Returns:
            dict with keys: mu, log_std, value, z_t, next_rnn_state
        """
        if not self.is_asymmetric:
            if seq_first:
                T, B = obs.shape[:2]
                flat = obs.reshape(T * B, -1)
                z = self._fuse(flat).reshape(T, B, -1)
                if self.gru is not None:
                    z = z.transpose(0, 1)
                    h_out, next_state = self.gru(z, rnn_state)
                    h_out = h_out.transpose(0, 1)
                else:
                    h_out, next_state = z, None
                head_in_a = head_in_c = self.head_input(h_out.reshape(T * B, -1), flat)
            else:
                z = self._fuse(obs)
                if self.gru is not None:
                    h_out, next_state = self.gru(z.unsqueeze(1), rnn_state)
                    head_in_a = head_in_c = self.head_input(h_out.squeeze(1), obs)
                else:
                    head_in_a = head_in_c = self.head_input(z, obs)
                    next_state = None
            z_t = z if not seq_first else z.reshape(-1, z.shape[-1])
        else:
            if seq_first:
                T, B = obs.shape[:2]
                flat = obs.reshape(T * B, -1)
                z_a = self._fuse_branch(flat, "actor").reshape(T, B, -1)
                z_c = self._fuse_branch(flat, "critic").reshape(T, B, -1)
                if self.actor_gru is not None:
                    z_a_b = z_a.transpose(0, 1)
                    z_c_b = z_c.transpose(0, 1)
                    h_a, next_state_a = self.actor_gru(z_a_b, rnn_state)
                    h_c, _ = self.critic_gru(z_c_b, rnn_state)
                    head_in_a = h_a.transpose(0, 1).reshape(T * B, -1)
                    head_in_c = h_c.transpose(0, 1).reshape(T * B, -1)
                    next_state = next_state_a
                else:
                    head_in_a = z_a.reshape(T * B, -1)
                    head_in_c = z_c.reshape(T * B, -1)
                    next_state = None
                z_t = z_a.reshape(T * B, -1)  # actor z exposed for SSL hooks
            else:
                z_a = self._fuse_branch(obs, "actor")
                z_c = self._fuse_branch(obs, "critic")
                if self.actor_gru is not None and self.cfg.asym_separate_memory:
                    # V9 (section 25): rnn_state = (actor state, critic state), each GRU carries its own
                    s_a, s_c = rnn_state if rnn_state is not None else (None, None)
                    h_a, next_a = self.actor_gru(z_a.unsqueeze(1), s_a)
                    h_c, next_c = self.critic_gru(z_c.unsqueeze(1), s_c)
                    head_in_a = h_a.squeeze(1)
                    head_in_c = h_c.squeeze(1)
                    next_state = (next_a, next_c)
                elif self.actor_gru is not None:
                    h_a, next_state_a = self.actor_gru(z_a.unsqueeze(1), rnn_state)
                    h_c, _ = self.critic_gru(z_c.unsqueeze(1), rnn_state)
                    head_in_a = h_a.squeeze(1)
                    head_in_c = h_c.squeeze(1)
                    next_state = next_state_a
                else:
                    head_in_a = z_a
                    head_in_c = z_c
                    next_state = None
                z_t = z_a

        mu = self.actor_mu(self.actor_trunk(head_in_a))
        value = self.critic_v(self.critic_trunk(head_in_c))
        if self.cfg.fixed_sigma:
            log_std = self.log_std.expand_as(mu)
        else:
            log_std = self.actor_log_std(self.actor_trunk(head_in_a))
        return {
            "mu": mu,
            "log_std": log_std,
            "value": value,
            "z_t": z_t,
            "next_rnn_state": next_state,
        }

    def forward(self, obs: torch.Tensor, rnn_state: Optional[torch.Tensor] = None):
        """Standard nn.Module forward (mu, log_std, value, next_rnn_state)."""
        out = self.forward_with_extras(obs, rnn_state=rnn_state)
        return out["mu"], out["log_std"], out["value"], out["next_rnn_state"]

    # ----- introspection ---------------------------------------------------

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

def build_policy(cfg: PolicyConfig) -> PretrainAdaptPolicy:
    """Factory entry point (proposal W4)."""
    return PretrainAdaptPolicy(cfg)
