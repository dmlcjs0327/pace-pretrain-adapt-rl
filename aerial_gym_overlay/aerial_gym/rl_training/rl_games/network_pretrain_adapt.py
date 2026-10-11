"""rl-games network of the study: the policy of pretrain_adapt_policy.py with the adaptation modes (full, partial, and frozen backbones, LoRA, LP-FT, Small-LR, L2-SP, Aux-SSL, BC-reg) and the loading of a pretrained backbone."""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_games.algos_torch import model_builder
from rl_games.algos_torch.network_builder import NetworkBuilder

from aerial_gym.nn.pretrain_adapt_policy import PolicyConfig, PretrainAdaptPolicy

# ---------------------------------------------------------------------------
# Adaptation modes and loading of the pretrained backbone
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Frozen base Linear with a low-rank update of rank r, W x + B A x. With b_init="zeros", B starts at zero, so
    training starts from the pretrained network (the setting of the paper)."""

    def __init__(self, base_linear: nn.Linear, rank: int = 8,
                 b_init: str = "zeros", b_init_std: float = 0.02):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Linear(base_linear.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        if b_init == "zeros":
            nn.init.zeros_(self.B.weight)
        elif b_init == "gaussian":
            nn.init.normal_(self.B.weight, mean=0.0, std=b_init_std)
        else:
            raise ValueError(f"Unknown b_init: {b_init}")
        self.b_init = b_init

    def forward(self, x):
        return self.base(x) + self.B(self.A(x))

class LinearAdapterLinear(nn.Module):
    """Frozen base Linear + full-rank Linear adapter (no low-rank decomposition).

    """

    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.adapter = nn.Linear(
            base_linear.in_features, base_linear.out_features, bias=False
        )
        nn.init.zeros_(self.adapter.weight)

    def forward(self, x):
        return self.base(x) + self.adapter(x)

def _wrap_module_linears_with_lora(module: nn.Module, rank: int = 8,
                                    b_init: str = "zeros"):
    """Replace every nn.Linear of the module by a LoRALinear in place."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, b_init=b_init))
        else:
            _wrap_module_linears_with_lora(child, rank=rank, b_init=b_init)

class _LoRAWeight(nn.Module):
    """Parametrization W -> W + B A for a frozen weight matrix (used for the GRU weights)."""

    def __init__(self, out_features: int, in_features: int, rank: int = 8):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_features))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, W):
        return W + self.B @ self.A

def _add_lora_to_gru(gru: nn.GRU, rank: int = 8):
    """Low-rank updates of the input and recurrent weight matrices of a one-layer GRU.
    The original weights and the biases stay frozen; training starts exactly from them."""
    import torch.nn.utils.parametrize as parametrize
    for p in gru.parameters():
        p.requires_grad = False
    for name in ("weight_ih_l0", "weight_hh_l0"):
        W = getattr(gru, name)
        parametrize.register_parametrization(gru, name, _LoRAWeight(W.shape[0], W.shape[1], rank).to(W.device))
        getattr(gru.parametrizations, name).original.requires_grad = False

def _wrap_module_linears_with_linear_adapter(module: nn.Module):
    """Replace every nn.Linear of the module by a LinearAdapterLinear in place (full-rank adapter ablation)."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LinearAdapterLinear(child))
        else:
            _wrap_module_linears_with_linear_adapter(child)

class SVDTruncLinearAdapter(nn.Module):
    """Frozen base Linear + full-rank trainable W with forward-time SVD top-r projection.

    Forward cost: SVD per forward (small matrices ~ 128*64 → μs on RTX 3090).
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 8):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_dim = base_linear.in_features
        out_dim = base_linear.out_features
        # Tiny Gaussian init (1e-6 σ): essentially zero perturbation (matches LoRA's B=0
        # semantics for the first PPO step) while keeping torch.linalg.svd numerically
        # stable. Pure-zero init causes svd-not-converged at t=0.
        self.W_train = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.normal_(self.W_train, mean=0.0, std=1e-6)
        self.rank = rank

    def forward(self, x):
        # SVD top-r projection at each forward (graceful fallback if convergence fails).
        try:
            U, S, Vh = torch.linalg.svd(self.W_train, full_matrices=False)
            r = min(self.rank, S.shape[0])
            W_r = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
        except torch._C._LinAlgError:
            W_r = torch.zeros_like(self.W_train)
        return F.linear(x, self.base.weight + W_r, self.base.bias)

def _wrap_module_linears_with_svd_trunc_adapter(module: nn.Module, rank: int = 8):
    """Replace every nn.Linear of the module by an SVDTruncLinearAdapter in place."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, SVDTruncLinearAdapter(child, rank=rank))
        else:
            _wrap_module_linears_with_svd_trunc_adapter(child, rank=rank)

def load_ssl_backbone(policy: PretrainAdaptPolicy, ckpt_path: str) -> int:
    """
    Returns: loaded key count.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("backbone_state_dict", ckpt)
    # Filter to backbone-only keys (defensive)
    backbone_keys = {k: v for k, v in state_dict.items()
                     if k.startswith(("proprio_mlp.", "fusion_mlp.", "gru."))}
    missing, unexpected = policy.load_state_dict(backbone_keys, strict=False)
    return len(backbone_keys)

def load_asymmetric_ssl_backbones(
    policy: PretrainAdaptPolicy,
    actor_ckpt_path: str,
    critic_ckpt_path: str,
) -> int:
    """

    Returns: total loaded key count (actor + critic backbone).
    """
    if not getattr(policy, "is_asymmetric", False):
        raise RuntimeError(
            "load_asymmetric_ssl_backbones requires policy with dual backbone "
            "(set PolicyConfig.use_asymmetric_backbone=True)."
        )
    n_loaded = 0
    for ckpt_path, target_prefix in [
        (actor_ckpt_path, "actor_"),
        (critic_ckpt_path, "critic_"),
    ]:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("backbone_state_dict", ckpt)
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith(("proprio_mlp.", "fusion_mlp.", "gru.")):
                remapped[target_prefix + k] = v
        policy.load_state_dict(remapped, strict=False)
        n_loaded += len(remapped)
    return n_loaded

def apply_adapt_mode(policy: PretrainAdaptPolicy, mode: str, lora_rank: int = 8,
                     b_init: str = "zeros"):
    """

    Pattern parsing:
      - "FrozenLoRA_r{N}" → (mode="FrozenLoRA", lora_rank=N)
      - "FrozenLoRA_BinitRandom" → (mode="FrozenLoRA", b_init="gaussian")
    """
    mode = mode or "FullFT"

    # Priority 4 ablation: "FrozenLoRA_BinitRandom" → mode="FrozenLoRA", b_init="gaussian"
    if mode == "FrozenLoRA_BinitRandom":
        b_init = "gaussian"
        mode = "FrozenLoRA"

    # L1 ablation: "FrozenLoRA_r{N}" → mode="FrozenLoRA", lora_rank=N
    if mode.startswith("FrozenLoRA_r"):
        try:
            lora_rank = int(mode[len("FrozenLoRA_r"):])
            mode = "FrozenLoRA"
        except ValueError:
            raise ValueError(f"Invalid FrozenLoRA rank pattern: {mode}")

    if mode == "FullFT":
        for p in policy.parameters():
            p.requires_grad = True
        return

    # SVDTruncLinearAdapter_r{N}: parse rank
    svd_rank = 8
    if mode.startswith("SVDTruncLinearAdapter_r"):
        try:
            svd_rank = int(mode[len("SVDTruncLinearAdapter_r"):])
            mode = "SVDTruncLinearAdapter"
        except ValueError:
            raise ValueError(f"Invalid SVDTruncLinearAdapter rank pattern: {mode}")
    if mode == "SVDTruncLinearAdapter" and svd_rank == 8:
        # default svd_rank 8 already set
        pass

    if mode in ("EncFrozen", "LPFT", "LateFT", "FrozenLoRA", "LinearAdapter", "SVDTruncLinearAdapter", "BackboneLoRA"):
        # backbone freeze (proprio_mlp + fusion_mlp + gru)
        backbone_names = ["proprio_mlp", "fusion_mlp", "gru",
                          "actor_proprio_mlp", "actor_fusion_mlp", "actor_gru",
                          "critic_proprio_mlp", "critic_fusion_mlp", "critic_gru"]
        for m_name in backbone_names:
            mod = getattr(policy, m_name, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = False

    if mode in ("EncFrozen", "LPFT"):
        # training observer of runner.py unfreezes the backbone after AG_UNFREEZE_AT iterations.
        for name in ["actor_trunk", "actor_mu", "critic_trunk", "critic_v"]:
            mod = getattr(policy, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True
        if hasattr(policy, "log_std"):
            policy.log_std.requires_grad = True

    elif mode == "LateFT":
        gru_names = ["gru", "actor_gru", "critic_gru"]
        for gru_name in gru_names:
            mod = getattr(policy, gru_name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True
        for name in ["actor_trunk", "actor_mu", "critic_trunk", "critic_v"]:
            mod = getattr(policy, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True
        if hasattr(policy, "log_std"):
            policy.log_std.requires_grad = True

    elif mode == "BackboneLoRA":
        # Standard use of LoRA on the pretrained part: low-rank updates (rank r,
        # B = 0 at the start) of every weight matrix of the backbone (encoder MLP layers and the
        # GRU input and recurrent matrices), while the new heads and log_std are trained fully.
        if getattr(policy, "is_asymmetric", False):
            raise ValueError("BackboneLoRA supports the shared backbone only")
        _wrap_module_linears_with_lora(policy.proprio_mlp, rank=lora_rank, b_init=b_init)
        _wrap_module_linears_with_lora(policy.fusion_mlp, rank=lora_rank, b_init=b_init)
        if policy.gru is not None:
            _add_lora_to_gru(policy.gru, rank=lora_rank)
        for name in ["actor_trunk", "actor_mu", "critic_trunk", "critic_v"]:
            mod = getattr(policy, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True
        if hasattr(policy, "log_std"):
            policy.log_std.requires_grad = True

    elif mode == "FrozenLoRA":
        for name in ["actor_trunk", "actor_mu", "critic_trunk", "critic_v"]:
            mod = getattr(policy, name, None)
            if mod is None:
                continue
            # Freeze first
            for p in mod.parameters():
                p.requires_grad = False
        _wrap_module_linears_with_lora(policy.actor_trunk, rank=lora_rank, b_init=b_init)
        if isinstance(policy.actor_mu, nn.Linear):
            policy.actor_mu = LoRALinear(policy.actor_mu, rank=lora_rank, b_init=b_init)
        _wrap_module_linears_with_lora(policy.critic_trunk, rank=lora_rank, b_init=b_init)
        if isinstance(policy.critic_v, nn.Linear):
            policy.critic_v = LoRALinear(policy.critic_v, rank=lora_rank, b_init=b_init)
        if hasattr(policy, "log_std"):
            policy.log_std.requires_grad = True

    elif mode == "LinearAdapter":
        for name in ["actor_trunk", "actor_mu", "critic_trunk", "critic_v"]:
            mod = getattr(policy, name, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = False
        _wrap_module_linears_with_linear_adapter(policy.actor_trunk)
        if isinstance(policy.actor_mu, nn.Linear):
            policy.actor_mu = LinearAdapterLinear(policy.actor_mu)
        _wrap_module_linears_with_linear_adapter(policy.critic_trunk)
        if isinstance(policy.critic_v, nn.Linear):
            policy.critic_v = LinearAdapterLinear(policy.critic_v)
        if hasattr(policy, "log_std"):
            policy.log_std.requires_grad = True

    elif mode == "SVDTruncLinearAdapter":
        # SVD-projected trainable W at rank svd_rank (adapter ablation, not used in the paper).
        # Mathematically same rank-r solution space as LoRA r=svd_rank, but parameterization
        # differs (factored A@B vs SVD-truncated full W). Disambiguates "low-rank constraint"
        # from "low-rank factored optimization" for H2/L2.
        for name in ["actor_trunk", "actor_mu", "critic_trunk", "critic_v"]:
            mod = getattr(policy, name, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = False
        _wrap_module_linears_with_svd_trunc_adapter(policy.actor_trunk, rank=svd_rank)
        if isinstance(policy.actor_mu, nn.Linear):
            policy.actor_mu = SVDTruncLinearAdapter(policy.actor_mu, rank=svd_rank)
        _wrap_module_linears_with_svd_trunc_adapter(policy.critic_trunk, rank=svd_rank)
        if isinstance(policy.critic_v, nn.Linear):
            policy.critic_v = SVDTruncLinearAdapter(policy.critic_v, rank=svd_rank)
        if hasattr(policy, "log_std"):
            policy.log_std.requires_grad = True
    else:
        raise ValueError(f"Unknown adapt_mode: {mode}")

class PretrainAdaptBuilder(NetworkBuilder):
    """Bridge from rl_games YAML config to PretrainAdaptPolicy.

    Notes:
        - The actual policy module is `PretrainAdaptPolicy`. This wrapper only
          adapts shapes & the (mu, sigma, value, states) tuple expected by
          rl_games' continuous_a2c_logstd model.
        - z_t is stored on the network instance after each forward so an
          aux-loss hook can read it later (B2). For B1 it is unused.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = self.Network(self.params, **kwargs)
        return net

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            self.value_size = kwargs.pop("value_size", 1)
            self.num_seqs = kwargs.pop("num_seqs", 1)
            super().__init__()

            obs_dim = int(input_shape[0]) if hasattr(input_shape, "__len__") else int(input_shape)
            cfg_dict = (params or {}).get("pretrain_adapt", {}) or {}

            self.cfg = PolicyConfig(
                proprio_dim=cfg_dict.get("proprio_dim", obs_dim),
                vae_dim=cfg_dict.get("vae_dim", 64),
                use_vae_branch=cfg_dict.get("use_vae_branch", False),
                proprio_hidden=cfg_dict.get("proprio_hidden", 64),
                vae_proj_hidden=cfg_dict.get("vae_proj_hidden", 64),
                fusion_hidden=cfg_dict.get("fusion_hidden", 128),
                fusion_layers=cfg_dict.get("fusion_layers", 2),
                use_gru=cfg_dict.get("use_gru", True),
                gru_hidden=cfg_dict.get("gru_hidden", 128),
                head_hidden=cfg_dict.get("head_hidden", 64),
                action_dim=actions_num,
                log_std_init=cfg_dict.get("log_std_init", 0.0),
                fixed_sigma=cfg_dict.get("fixed_sigma", True),
                use_asymmetric_backbone=cfg_dict.get("use_asymmetric_backbone", False),
                asym_separate_memory=cfg_dict.get("asym_separate_memory", False),
                feat_whiten=cfg_dict.get("feat_whiten", False),
                skip_obs=cfg_dict.get("skip_obs", False),
            )
            self.policy = PretrainAdaptPolicy(self.cfg)

            # Loading of the pretrained backbone and the adaptation mode:
            ssl_ckpt = cfg_dict.get("ssl_backbone_ckpt", None)
            ssl_actor_ckpt = cfg_dict.get("ssl_actor_ckpt", None)
            ssl_critic_ckpt = cfg_dict.get("ssl_critic_ckpt", None)
            adapt_mode = cfg_dict.get("adapt_mode", "FullFT")
            lora_rank = int(cfg_dict.get("lora_rank", 8))
            if ssl_actor_ckpt and ssl_critic_ckpt:
                # Asymmetric pretrain (Garcin 2025 actor/critic specialization)
                n_loaded = load_asymmetric_ssl_backbones(
                    self.policy, ssl_actor_ckpt, ssl_critic_ckpt
                )
                print(f"[network_pretrain_adapt] Asymmetric SSL: actor={ssl_actor_ckpt}, "
                      f"critic={ssl_critic_ckpt}, loaded={n_loaded} keys")
            elif ssl_actor_ckpt or ssl_critic_ckpt:
                # V9 (section 25): pretrained weights in one backbone of the asymmetric policy, the other stays random
                side, ck_one = ("actor_", ssl_actor_ckpt) if ssl_actor_ckpt else ("critic_", ssl_critic_ckpt)
                if not getattr(self.policy, "is_asymmetric", False):
                    raise RuntimeError("one-sided pretrained backbone needs use_asymmetric_backbone=True")
                sd = torch.load(ck_one, map_location="cpu")
                sd = sd.get("backbone_state_dict", sd)
                remapped = {side + k: v for k, v in sd.items() if k.startswith(("proprio_mlp.", "fusion_mlp.", "gru."))}
                self.policy.load_state_dict(remapped, strict=False)
                print(f"[network_pretrain_adapt] Asymmetric SSL: {side}backbone from {ck_one} ({len(remapped)} keys), "
                      f"the other backbone random")
            elif ssl_ckpt:
                n_loaded = load_ssl_backbone(self.policy, ssl_ckpt)
                print(f"[network_pretrain_adapt] SSL backbone loaded: {n_loaded} keys from {ssl_ckpt}")
                # actor head of a behavior-cloning checkpoint
                if cfg_dict.get("ssl_actor_head", False):
                    hs = torch.load(ssl_ckpt, map_location="cpu")["head_state_dict"]
                    self.policy.actor_trunk[0].weight.data.copy_(hs["actor_trunk.0.weight"])
                    self.policy.actor_trunk[0].bias.data.copy_(hs["actor_trunk.0.bias"])
                    self.policy.actor_mu.weight.data.copy_(hs["actor_mu.weight"])
                    self.policy.actor_mu.bias.data.copy_(hs["actor_mu.bias"])
                    print("[network_pretrain_adapt] actor head loaded from the behavior-cloning checkpoint")
                    # Initial action noise of the BC policy: the
                    # maximum-likelihood std of the Gaussian BC policy, i.e. the per-dimension RMSE of the BC actor
                    # on the offline windows (experiments/scripts/bc_sigma.py). With the default std of 1 the noise
                    # hides the transferred actor (mean action: return 1,828 vs 166 with sampling, T1 x R2 seed 7).
                    if not cfg_dict.get("bc_sigma_skip", False):
                        import json as _json
                        sp = ssl_ckpt.replace("__backbone.pt", "__bc_sigma.json")
                        if not os.path.exists(sp):
                            raise FileNotFoundError(f"BC initial noise file missing: {sp} (run bc_sigma.py)")
                        sig = torch.tensor(_json.load(open(sp))["sigma"], dtype=torch.float32)
                        # S9 (pre-registration v2, section 16): sensitivity runs scale this std (stage 7: 0.25)
                        sig = sig * float(os.environ.get("AG_BC_SIGMA_SCALE", "1.0"))
                        if not hasattr(self.policy, "log_std") or self.policy.log_std.shape[-1] != sig.numel():
                            raise RuntimeError("BC initial noise: policy has no matching log_std")
                        self.policy.log_std.data.copy_(torch.log(sig).view_as(self.policy.log_std))
                        print(f"[network_pretrain_adapt] BC initial action std {[round(float(v), 4) for v in sig]}")
            apply_adapt_mode(self.policy, adapt_mode, lora_rank=lora_rank)

            # Fixed input normalization: standardize the observation with fixed statistics (the
            # pretraining data), clipped to [-5, 5] like rl-games, for every backbone. Used with
            # normalize_input: False, so the pretrained backbone receives the input scale it was trained on.
            fin = cfg_dict.get("fixed_input_norm", None)
            self._fixed_norm = bool(fin)
            if fin:
                st = torch.load(fin, map_location="cpu")
                self.register_buffer("fixed_obs_mean", torch.as_tensor(st["state_mean"], dtype=torch.float32))
                self.register_buffer("fixed_obs_std", torch.as_tensor(st["state_std"], dtype=torch.float32))
                print(f"[network_pretrain_adapt] fixed input normalization from {fin}")
            if cfg_dict.get("feat_whiten", False):
                # V8a (section 25): fit the whitening map once on the pretraining data and keep it fixed. Features: the
                # backbone output on 4,096 windows of the v3 data (every 4th of the fixed-seed windows of
                # pace_regularizers, both airframes), the first 8 steps of every window dropped. W = (S + eps I)^(-1/2)
                # with eps = 0.01 tr(S) / n.
                from aerial_gym.rl_training.rl_games.pace_regularizers import load_windows
                xs, _, _ = load_windows(torch.load(fin, map_location="cpu"))
                with torch.no_grad():
                    dev_p = next(self.policy.parameters()).device
                    h = self.policy.backbone_seq(xs[::4].float().to(dev_p))[:, 8:, :].cpu()
                    h = h.reshape(-1, h.shape[-1]).double()
                    mu = h.mean(0)
                    S = torch.cov((h - mu).T)
                    n = S.shape[0]
                    eps = 0.01 * torch.trace(S) / n
                    lam, V = torch.linalg.eigh(S)
                    W = V @ torch.diag((lam.clamp(min=0) + eps) ** -0.5) @ V.T
                self.policy.feat_mu.copy_(mu.float().to(self.policy.feat_mu.device))
                self.policy.feat_W.copy_(W.float().to(self.policy.feat_W.device))
                pr = float(lam.clamp(min=0).sum() ** 2 / (lam.clamp(min=0) ** 2).sum())
                print(f"[network_pretrain_adapt] feature whitening fitted: participation ratio {pr:.2f}, eps {float(eps):.3e}")
            # Regularizers and auxiliary losses added to the PPO loss, see pace_regularizers.py
            self.pace_reg = None
            if cfg_dict.get("reg_mode"):
                from aerial_gym.rl_training.rl_games.pace_regularizers import PaceRegularizer
                self.pace_reg = PaceRegularizer(self.policy, cfg_dict["reg_mode"], cfg_dict.get("reg_lambda", 1.0),
                                                cfg_dict.get("reg_ckpt"), cfg_dict.get("fixed_input_norm"))
                print(f"[network_pretrain_adapt] reg_mode={cfg_dict['reg_mode']} lambda={cfg_dict.get('reg_lambda', 1.0)}")
            # Critic warm-up: AG_CRITIC_WARMUP=<N> trains only the critic head for N iterations;
            # the training observer of runner.py then restores the requires_grad pattern of the adaptation mode.
            self._rg_mask = {n: p.requires_grad for n, p in self.named_parameters()}
            if os.environ.get("AG_CRITIC_WARMUP"):
                for n, p in self.named_parameters():
                    p.requires_grad = ("critic_trunk" in n) or ("critic_v" in n)
                print(f"[network_pretrain_adapt] critic warm-up for {os.environ['AG_CRITIC_WARMUP']} iterations")
            n_trainable = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.policy.parameters())
            print(f"[network_pretrain_adapt] adapt_mode={adapt_mode} trainable={n_trainable}/{n_total} ({100*n_trainable/max(n_total,1):.1f}%)")

            # Last z_t cached so external aux-loss hooks can read it (B2).
            self._last_z_t: Optional[torch.Tensor] = None

            # rl_games' calc_gradients() invokes self.model.get_aux_loss() once
            # per PPO minibatch and adds it to the total loss BEFORE backward,
            # so the aux gradient flows through the same backbone params as the
            # PPO update — matching CURL/SPR/ATC canonical paradigm. The aux
            # head and lambda are injected by A2CAgentPretrainAdapt after model
            # construction; agent.calc_gradients caches the current minibatch
            # input_dict on self._aux_input_dict so this hook can re-run the
            # backbone over the temporally-structured (num_seqs, seq_length, ...)
            # tensor and apply the SSL loss.
            self.aux_head: Optional[nn.Module] = None
            self.aux_name: str = "P0"
            self.aux_lambda: float = 0.0
            self._aux_input_dict: Optional[dict] = None
            # EMA target backbone for canonical SPR/BYOL (set by agent after
            # network construction via set_aux_target_backbone). None for
            # all other aux modes — get_aux_loss falls back to the legacy
            # degenerate variant in that case.
            self._aux_target_backbone: Optional[nn.Module] = None
            # State augmentation σ for SPR/BYOL/VICReg (canonical view-pair
            # construction on proprio). σ=0.05 ≈ 5cm/0.05 rad scale, within
            # the order of real IMU noise on a 13-D proprio vector.
            self._aux_state_aug_sigma: float = 0.05

        # ----- rl_games protocol -----------------------------------------

        def is_rnn(self) -> bool:
            return self.cfg.use_gru

        def get_default_rnn_state(self):
            if not self.cfg.use_gru:
                return None
            # rl_games expects a tuple of zero tensors with shape (num_layers, num_seqs, hidden)
            num_layers = 1
            if self.cfg.asym_separate_memory:  # V9 (section 25): actor state, critic state
                return (torch.zeros((num_layers, self.num_seqs, self.cfg.gru_hidden)),
                        torch.zeros((num_layers, self.num_seqs, self.cfg.gru_hidden)))
            return (torch.zeros((num_layers, self.num_seqs, self.cfg.gru_hidden)),)

        def get_value_layer(self):
            return self.policy.critic_v

        def set_aux_module(self, head: Optional[nn.Module], name: str, lambda_val: float) -> None:
            """Wire an SSL aux head into the network for per-minibatch joined loss."""
            self.aux_head = head
            self.aux_name = (name or "P0").upper()
            self.aux_lambda = float(lambda_val)

        def set_aux_target_backbone(self, target_bb: nn.Module) -> None:
            """Wire the EMA target backbone (canonical SPR/BYOL) into the
            network so get_aux_loss can forward augmented obs through it."""
            self._aux_target_backbone = target_bb

        def _augment_proprio_obs(self, obs_seq: torch.Tensor, proprio_dim: int) -> torch.Tensor:
            """Add Gaussian noise σ=0.05 to the proprio subvector of obs_seq.
            Vision/VAE dims (obs_seq[..., proprio_dim:]) pass through
            unchanged — depth augmentation belongs to the vision pipeline.
            Returns a fresh tensor with the noisy proprio + original suffix."""
            proprio = obs_seq[..., :proprio_dim]
            proprio_aug = proprio + self._aux_state_aug_sigma * torch.randn_like(proprio)
            if obs_seq.shape[-1] > proprio_dim:
                return torch.cat([proprio_aug, obs_seq[..., proprio_dim:]], dim=-1)
            return proprio_aug

        def _augment_full_obs(self, obs_seq: torch.Tensor, proprio_dim: int) -> torch.Tensor:
            """Augment both proprio (Gaussian noise) AND vision (CURL random
            crop + Gaussian noise on depth image) when use_cnn_vision_branch
            is active. This is the canonical CURL paradigm — augmentation
            applies to the *raw input* before the encoder. For frozen-VAE
            setups (use_vae_branch=True), VAE latent is opaque to image
            augmentation, so only proprio is augmented (degenerate)."""
            cfg = self.cfg
            proprio = obs_seq[..., :proprio_dim]
            proprio_aug = proprio + self._aux_state_aug_sigma * torch.randn_like(proprio)
            if cfg.use_cnn_vision_branch:
                # vision_extra = flat depth (1, H, W) reshape, augment, flatten
                flat_len = cfg.vision_in_channels * cfg.vision_input_h * cfg.vision_input_w
                depth_flat = obs_seq[..., proprio_dim:proprio_dim + flat_len]
                shape = depth_flat.shape[:-1]
                depth = depth_flat.reshape(*shape, cfg.vision_in_channels,
                                           cfg.vision_input_h, cfg.vision_input_w)
                # Apply per-(B,T) augmentation: flatten batch+time, augment, unflatten.
                from aerial_gym.ssl_pretrain.image_augmentation import aug_pipeline
                flat_img = depth.reshape(-1, cfg.vision_in_channels,
                                         cfg.vision_input_h, cfg.vision_input_w)
                aug_flat_img = aug_pipeline(flat_img, crop_pad=4, noise_sigma=0.05)
                depth_aug = aug_flat_img.reshape(*shape, cfg.vision_in_channels,
                                                 cfg.vision_input_h, cfg.vision_input_w)
                depth_aug_flat = depth_aug.reshape(*shape, flat_len)
                return torch.cat([proprio_aug, depth_aug_flat], dim=-1)
            elif cfg.use_vae_branch and obs_seq.shape[-1] > proprio_dim:
                # Frozen VAE latent — no image augmentation possible (latent
                # is opaque). Pass through unchanged (degenerate from CURL's
                # perspective). Documented in §6.2 L2/L11.
                return torch.cat([proprio_aug, obs_seq[..., proprio_dim:]], dim=-1)
            return proprio_aug

        def get_aux_loss(self):
            """rl-games hook, once per PPO minibatch: the PACE regularizer (if any) plus the legacy aux loss."""
            out = {}
            if getattr(self, "pace_reg", None) is not None:
                out["reg"] = self.pace_reg()
            legacy = self._legacy_get_aux_loss()
            if legacy:
                out.update(legacy)
            return out or None

        def _legacy_get_aux_loss(self):
            """Per-minibatch SSL aux loss for joint paradigm (M_Aux).

            Returns a dict {"aux": lambda * loss} so rl_games' calc_gradients
            adds it to the PPO loss in the SAME backward pass, sharing optimizer
            and gradient flow with the policy/value losses. Returns None when
            aux is disabled or the minibatch is not temporally structured.
            """
            if self.aux_head is None or self.aux_name in ("P0", ""):
                return None
            inp = self._aux_input_dict
            if inp is None:
                return None
            obs = inp.get("obs", None)
            actions = inp.get("actions", None)
            if obs is None:
                return None
            seq_len = int(inp.get("seq_length", 1) or 1)
            if seq_len <= 1:
                # Sequence-based SSL losses require T > 1; skip for non-RNN minibatches.
                return None

            B_total = obs.shape[0]
            if B_total % seq_len != 0:
                return None
            num_seqs = B_total // seq_len
            obs_seq = obs.reshape(num_seqs, seq_len, -1)               # (B, T, obs_dim)
            act_seq = (actions.reshape(num_seqs, seq_len, -1)
                       if actions is not None else None)

            proprio_dim = self.cfg.proprio_dim
            name = self.aux_name

            try:
                if name == "MAE":
                    state_seq = obs_seq[..., :proprio_dim]
                    mask = (torch.rand_like(state_seq) < 0.3)
                    masked_state = state_seq * (~mask).float()
                    masked_obs = (torch.cat([masked_state, obs_seq[..., proprio_dim:]], dim=-1)
                                  if obs_seq.shape[-1] > proprio_dim else masked_state)
                    z_full = self.policy.backbone_seq(masked_obs)
                    loss = self.aux_head.forward_with_z_full(z_full, state_seq, mask)
                else:
                    z_full = self.policy.backbone_seq(obs_seq)         # (B, T, H)
                    if name == "FD":
                        if act_seq is None:
                            return None
                        loss = self.aux_head.forward_with_z_full(z_full, act_seq)
                    elif name in ("IDM", "IDM_SSL"):
                        if act_seq is None:
                            return None
                        loss = self.aux_head.forward_with_z_full(z_full, act_seq)
                    elif name in ("CPC", "CPC_SSL"):
                        loss = self.aux_head.forward_with_z_full(z_full)
                    elif name == "QM":
                        loss = self.aux_head.forward_with_z_full(z_full)
                    elif name in ("PIDM", "PIDM_SSL"):
                        if act_seq is None:
                            return None
                        loss = self.aux_head.forward_with_z_full(z_full, act_seq)
                    elif name in ("VICREG", "VICREG_SSL"):
                        # Canonical VICReg uses two AUGMENTED views (Bardes 2022).
                        # Our previous temporal-pair form (z_t, z_{t+1}) was a
                        # online backbone on two noised views of obs_seq and pass
                        # to the aug-view loss path. Falls back to legacy temporal
                        # form if the aux head doesn't support `forward_with_aug_views`.
                        if hasattr(self.aux_head, "forward_with_aug_views"):
                            obs_a = self._augment_full_obs(obs_seq, proprio_dim)
                            obs_b = self._augment_full_obs(obs_seq, proprio_dim)
                            z_a = self.policy.backbone_seq(obs_a)
                            z_b = self.policy.backbone_seq(obs_b)
                            loss = self.aux_head.forward_with_aug_views(z_a, z_b)
                        else:
                            loss = self.aux_head.forward_with_z_full(z_full)
                    elif name in ("SPR", "SPR_SSL"):
                        # Canonical SPR (Schwarzer 2021): online encoder sees
                        # augmented view, EMA target encoder sees a separately-
                        # augmented view, BYOL-style cosine loss with multi-step
                        # latent prediction. Falls back to the legacy degenerate
                        # variant when no target backbone is wired.
                        if act_seq is None:
                            return None
                        if self._aux_target_backbone is not None:
                            obs_online = self._augment_full_obs(obs_seq, proprio_dim)
                            obs_target = self._augment_full_obs(obs_seq, proprio_dim)
                            z_online = self.policy.backbone_seq(obs_online)
                            with torch.no_grad():
                                z_target = self._aux_target_backbone.backbone_seq(obs_target)
                            loss = self.aux_head.forward_with_z_full(z_online, z_target, act_seq)
                        else:
                            z_target = z_full.detach()
                            loss = self.aux_head.forward_with_z_full(z_full, z_target, act_seq)
                    elif name in ("BYOL", "BYOL_SSL"):
                        # Canonical BYOL (Grill 2020): online + EMA target on two
                        # augmented views. Fallback to degenerate variant when no
                        # target backbone is wired.
                        if self._aux_target_backbone is not None:
                            obs_online = self._augment_full_obs(obs_seq, proprio_dim)
                            obs_target = self._augment_full_obs(obs_seq, proprio_dim)
                            z_online = self.policy.backbone_seq(obs_online)
                            with torch.no_grad():
                                z_target = self._aux_target_backbone.backbone_seq(obs_target)
                            loss = self.aux_head.forward_with_z_full(z_online, z_target)
                        else:
                            z_target = z_full.detach()
                            loss = self.aux_head.forward_with_z_full(z_full, z_target)
                    elif name in ("FUSION", "FUSION_SSL"):
                        # Fusion needs (z_clean, z_masked, state, action, mask).
                        # Build the masked branch matching offline pretrain.
                        if act_seq is None:
                            return None
                        state_seq = obs_seq[..., :proprio_dim]
                        mask = (torch.rand_like(state_seq) < 0.3)
                        masked_state = state_seq * (~mask).float()
                        masked_obs = (torch.cat([masked_state, obs_seq[..., proprio_dim:]], dim=-1)
                                      if obs_seq.shape[-1] > proprio_dim else masked_state)
                        z_full_masked = self.policy.backbone_seq(masked_obs)
                        loss = self.aux_head.forward_with_z_full(
                            z_full, z_full_masked, state_seq, act_seq, mask
                        )
                    elif name == "CURL":
                        # Canonical CURL (Laskin 2020): bilinear InfoNCE on
                        # two augmented views of the same observation. With
                        # use_cnn_vision_branch=True, the gradient propagates
                        # back into the trainable CNN visual encoder — this
                        # is the canonical AuxJoint-paradigm strength that
                        # the use_vae_branch (frozen VAE) setup cannot
                        # reproduce. With use_vae_branch, only proprio is
                        # augmented (degenerate CURL form, kept for honest
                        # cross-setup comparison).
                        if hasattr(self.aux_head, "forward_with_views"):
                            obs_q = self._augment_full_obs(obs_seq, proprio_dim)
                            obs_k = self._augment_full_obs(obs_seq, proprio_dim)
                            z_q = self.policy.backbone_seq(obs_q)
                            if self._aux_target_backbone is not None:
                                with torch.no_grad():
                                    z_k = self._aux_target_backbone.backbone_seq(obs_k)
                            else:
                                # Fallback: key encoder = online encoder (stop-grad)
                                z_k = self.policy.backbone_seq(obs_k).detach()
                            loss = self.aux_head.forward_with_views(z_q, z_k)
                        else:
                            return None
                    else:
                        return None
            except Exception as e:
                # Aux loss should never crash PPO; surface a one-shot warning.
                if not getattr(self, "_aux_err_warned", False):
                    print(f"[get_aux_loss] aux={name} failed: {e}; aux contribution skipped this minibatch")
                    self._aux_err_warned = True
                return None

            return {"aux": self.aux_lambda * loss}

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            if self._fixed_norm:
                obs = torch.clamp((obs - self.fixed_obs_mean) / self.fixed_obs_std, -5.0, 5.0)
            rnn_states = obs_dict.get("rnn_states", None)
            seq_length = obs_dict.get("seq_length", 1)

            # rl_games passes [num_seqs * seq_length, obs_dim]. For BPTT forward,
            # reshape into a batched sequence. Returns log_std (not sigma) to
            # match `continuous_a2c_logstd` model contract (models.py:294).
            if self.cfg.use_gru and seq_length > 1 and os.environ.get("AG_LEGACY_SEQ_ORDER") != "1":
                # Corrected BPTT path. rl-games orders a training minibatch
                # sequence-major (index = seq * T + t) and expects the outputs in the same order.
                # The legacy path below returned them time-major (t * B + b), which paired every
                # sample with the output of another sample. This path keeps the order and, like
                # rl-games' RnnWithDones, resets the hidden state inside a sequence where a new
                # episode starts (dones[t] = 1). AG_LEGACY_SEQ_ORDER=1 restores the old behaviour.
                pol = self.policy
                B_total = obs.shape[0]
                num_seqs = B_total // seq_length
                h = rnn_states[0] if (rnn_states is not None and len(rnn_states) > 0) else None
                dones = obs_dict.get("dones", None)
                nd = None if dones is None else (1.0 - dones.reshape(num_seqs, seq_length).float())

                def run_gru(gru, z):
                    hh, outs = h, []
                    for t in range(seq_length):
                        if nd is not None and hh is not None:
                            hh = hh * nd[:, t].view(1, -1, 1)
                        o, hh = gru(z[:, t:t + 1], hh)
                        outs.append(o)
                    return torch.cat(outs, dim=1).reshape(B_total, -1), hh

                if not pol.is_asymmetric:
                    z = pol._fuse(obs).reshape(num_seqs, seq_length, -1)       # [B, T, F]
                    head_a, h_next = run_gru(pol.gru, z)
                    head_a = pol.head_input(head_a, obs)  # V8 (section 25); identity otherwise
                    head_c = head_a
                elif pol.cfg.asym_separate_memory:
                    # V9 (section 25): each GRU starts from and carries its own state
                    z = pol._fuse_branch(obs, "actor").reshape(num_seqs, seq_length, -1)
                    zc = pol._fuse_branch(obs, "critic").reshape(num_seqs, seq_length, -1)
                    ha = rnn_states[0] if rnn_states is not None and len(rnn_states) > 0 else None
                    hc = rnn_states[1] if rnn_states is not None and len(rnn_states) > 1 else None
                    outs_a, outs_c = [], []
                    for t in range(seq_length):
                        if nd is not None:
                            if ha is not None:
                                ha = ha * nd[:, t].view(1, -1, 1)
                            if hc is not None:
                                hc = hc * nd[:, t].view(1, -1, 1)
                        oa, ha = pol.actor_gru(z[:, t:t + 1], ha)
                        oc, hc = pol.critic_gru(zc[:, t:t + 1], hc)
                        outs_a.append(oa)
                        outs_c.append(oc)
                    head_a = torch.cat(outs_a, dim=1).reshape(B_total, -1)
                    head_c = torch.cat(outs_c, dim=1).reshape(B_total, -1)
                    h_next = (ha, hc)
                else:
                    # As in the rollout path, both GRUs of a paired policy start each step from
                    # the actor's hidden state, and only the actor's state is carried forward.
                    z = pol._fuse_branch(obs, "actor").reshape(num_seqs, seq_length, -1)
                    zc = pol._fuse_branch(obs, "critic").reshape(num_seqs, seq_length, -1)
                    hh, outs_a, outs_c = h, [], []
                    for t in range(seq_length):
                        if nd is not None and hh is not None:
                            hh = hh * nd[:, t].view(1, -1, 1)
                        oc, _ = pol.critic_gru(zc[:, t:t + 1], hh)
                        oa, hh = pol.actor_gru(z[:, t:t + 1], hh)
                        outs_a.append(oa)
                        outs_c.append(oc)
                    head_a = torch.cat(outs_a, dim=1).reshape(B_total, -1)
                    head_c = torch.cat(outs_c, dim=1).reshape(B_total, -1)
                    h_next = hh
                mu = pol.actor_mu(pol.actor_trunk(head_a))
                value = pol.critic_v(pol.critic_trunk(head_c))
                if pol.cfg.fixed_sigma:
                    log_std = pol.log_std.expand_as(mu)
                else:
                    log_std = pol.actor_log_std(pol.actor_trunk(head_a))
                self._last_z_t = z.reshape(B_total, -1)
                return mu, log_std, value, (h_next if isinstance(h_next, tuple) else (h_next,))

            if self.cfg.use_gru and seq_length > 1:
                B_total = obs.shape[0]
                num_seqs = B_total // seq_length
                obs_seq = obs.reshape(num_seqs, seq_length, -1).transpose(0, 1)  # [T, B, D]
                h0 = rnn_states[0] if (rnn_states is not None and len(rnn_states) > 0) else None
                out = self.policy.forward_with_extras(obs_seq, rnn_state=h0, seq_first=True)
                self._last_z_t = out["z_t"]
                next_state = (out["next_rnn_state"],) if out["next_rnn_state"] is not None else ()
                return out["mu"], out["log_std"], out["value"], next_state

            # Single-step (rollout sampling) path.
            h0 = rnn_states[0] if (rnn_states is not None and len(rnn_states) > 0) else None
            if self.cfg.asym_separate_memory:  # V9 (section 25)
                h0 = tuple(rnn_states) if (rnn_states is not None and len(rnn_states) > 1) else None
            out = self.policy.forward_with_extras(obs, rnn_state=h0)
            self._last_z_t = out["z_t"]
            nxt = out["next_rnn_state"]
            next_state = () if nxt is None else (nxt if isinstance(nxt, tuple) else (nxt,))
            return out["mu"], out["log_std"], out["value"], next_state

def register():
    """Idempotent: register the custom network with rl_games' model_builder."""
    model_builder.register_network("aerial_gym_pretrain_adapt", PretrainAdaptBuilder)
