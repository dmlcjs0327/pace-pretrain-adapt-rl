"""Pretraining trainer: trains one backbone with one objective and one seed and saves the backbone weights only (SPR keeps an EMA target backbone during training)."""
import os as _os
from pathlib import Path as _Path
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
import isaacgym  # MUST come before torch (aerial_gym/__init__.py imports isaacgym)
import argparse
import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aerial_gym.ssl_pretrain.dataset import (
    TrajectoryWindowDataset,
    collate_window,
    make_input_seq,
)
from aerial_gym.ssl_pretrain.backbone import ProprioGRUBackbone
from aerial_gym.ssl_pretrain.losses import (
    ForwardDynamicsLoss,
    InverseDynamicsLoss,
    MaskedStateLoss,
    ContrastivePredictiveLoss,
    QuasiMetricLoss,
    SelfPredictiveLoss,
    FusionLoss,
    PIDMLoss,
    VICRegLoss,
    BYOLLoss,
)

def get_loss_module(objective, hidden_dim=128, action_dim=4, state_dim=13):
    if objective == "FD":
        return ForwardDynamicsLoss(hidden_dim, action_dim)
    elif objective == "IDM":
        return InverseDynamicsLoss(hidden_dim, action_dim)
    elif objective == "MAE":
        return MaskedStateLoss(hidden_dim, state_dim, mask_ratio=0.3)
    elif objective == "DROPOUT":
        # P_DROPOUT:
        return MaskedStateLoss(hidden_dim, state_dim, mask_ratio=0.3)
    elif objective == "CPC":
        return ContrastivePredictiveLoss(hidden_dim, predict_steps=(1, 2, 4, 8), temperature=0.1)
    elif objective == "QM":
        return QuasiMetricLoss(hidden_dim, n_components=8, dim_per_comp=8, n_pairs_per_seq=16)
    elif objective == "SPR":
        return SelfPredictiveLoss(hidden_dim, action_dim, predict_steps=(1, 2, 3, 5))
    elif objective == "FUSION":
        return FusionLoss(hidden_dim, action_dim, state_dim,
                          alpha=1.0, beta=1.0, gamma=0.5, mask_ratio=0.3)
    elif objective == "PIDM":
        return PIDMLoss(hidden_dim, action_dim, predict_steps=4)
    elif objective == "VICReg":
        return VICRegLoss(hidden_dim, proj_dim=64,
                          lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0)
    elif objective == "BYOL":
        return BYOLLoss(hidden_dim, proj_dim=64)
    elif objective == "ATOMIC":
        # Experiment C (App. D.2 atomic decomposition). ATOMIC does NOT use a
        # MaskedStateLoss head — the reconstruction decoder is built explicitly
        # in train_one() as an AtomicDecoderPair so that the encoder-decoder
        # asymmetry toggle (a) can switch between a symmetric (mirror-capacity)
        # decoder and an asymmetric (lightweight) decoder. The masking (m) and
        # partial-reconstruction (p) toggles are applied in the training loop.
        return AtomicDecoderPair(hidden_dim, state_dim)
    else:
        raise ValueError(f"Unknown objective: {objective}")

class AtomicDecoderPair(nn.Module):
    """Experiment C reconstruction module with TWO decoder heads.

    The atomic 'asymmetric encoder-decoder' component (a) of the MAE recipe is
    architectural (He et al. 2022): MAE pairs a heavy encoder with a *light*
    decoder. We make that toggleable by carrying two decoders:

      - sym_decoder  : mirror-capacity head (128->128->128->13). With a=0 the
                       encoder (proprio_mlp+fusion_mlp+gru, ~hundreds of params
                       per layer) is matched by a comparably-wide decoder, so
                       the encoder is not pressured to compress.
      - asym_decoder : lightweight bottleneck head (128->32->13). With a=1 the
                       decoder is deliberately under-capacity, forcing the
                       encoder latent to carry reconstructable structure
                       (the MAE asymmetry effect).

    forward(z, which) selects the decoder. Both heads are trained jointly only
    through whichever variant uses them; the unused head simply receives no
    gradient for a given run. The head is discarded after pretrain (ckpt keeps
    backbone only), so carrying both costs nothing downstream.
    """

    def __init__(self, hidden_dim=128, state_dim=13):
        super().__init__()
        self.sym_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, state_dim),
        )
        self.asym_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, state_dim),
        )

    def forward(self, z_full, asymmetric):
        if asymmetric:
            return self.asym_decoder(z_full)
        return self.sym_decoder(z_full)

@torch.no_grad()
def _ema_update(target_module, online_module, decay=0.99):
    for p_t, p_o in zip(target_module.parameters(), online_module.parameters()):
        p_t.data.mul_(decay).add_(p_o.data, alpha=1 - decay)
    for b_t, b_o in zip(target_module.buffers(), online_module.buffers()):
        b_t.data.copy_(b_o.data)

def make_mask(state_seq, mask_ratio):
    """For MAE: random mask per (batch, time, dim)."""
    B, T, D = state_seq.shape
    mask = torch.rand(B, T, D, device=state_seq.device) < mask_ratio
    return mask

def train_one(args):
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = TrajectoryWindowDataset(args.data, window_length=args.window,
                                 max_data_steps=args.max_data_steps)
    print(f"Dataset: {len(ds)} windows from {args.data} "
          f"(max_data_steps={args.max_data_steps if args.max_data_steps>0 else 'full'})")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_window,
        pin_memory=True,
        drop_last=True,
    )

    # Models
    backbone = ProprioGRUBackbone(proprio_dim=13).to(device)  # state-only, gru_hidden=128
    loss_module = get_loss_module(args.objective).to(device)

    target_backbone = None
    if args.objective in ("SPR", "BYOL"):
        target_backbone = copy.deepcopy(backbone).to(device)
        for p in target_backbone.parameters():
            p.requires_grad = False
        target_backbone.eval()

    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(loss_module.parameters()),
        lr=args.lr,
    )

    print(f"Training {args.objective} (seed {args.seed}) for {args.epochs} epochs ...")
    backbone.train()
    loss_module.train()

    t_start = time.time()
    history = []
    max_steps_per_epoch = args.max_steps_per_epoch  # 0 = full epoch
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_i, batch in enumerate(loader):
            if max_steps_per_epoch and batch_i >= max_steps_per_epoch:
                break
            state_seq = batch["state"].to(device, non_blocking=True)   # (B, T+1, 13)
            action_seq = batch["action"].to(device, non_blocking=True) # (B, T+1, 4)

            inp_full = state_seq  # (B, T+1, 13)

            if args.objective == "MAE":
                mask = make_mask(state_seq, mask_ratio=0.3)            # (B, T+1, 13)
                inp_masked = state_seq * (~mask).float()
                z_full, _ = backbone(inp_masked)
                loss = loss_module.forward_with_z_full(z_full, state_seq, mask)
            elif args.objective == "DROPOUT":
                # Plain denoising-AE with full-MSE loss (no asymmetric encoder-decoder).
                # Same drop rate (0.3) and same backbone arch as MAE; the only difference
                # is that DROPOUT's loss is computed over ALL output dims (mask = ones),
                # not just the masked dims. H1 disambiguation control.
                mask = make_mask(state_seq, mask_ratio=0.3)
                inp_dropped = state_seq * (~mask).float()
                z_full, _ = backbone(inp_dropped)
                full_mask = torch.ones_like(mask)  # reconstruct ALL dims, not just masked
                loss = loss_module.forward_with_z_full(z_full, state_seq, full_mask)
            elif args.objective == "ATOMIC":
                # Experiment C — 3-component atomic decomposition of the MAE recipe.
                # Each toggle is APPLIED INDEPENDENTLY so all 8 (m,a,p) cells are
                # genuinely distinct, non-degenerate objectives:
                #
                #   sel_mask : a fixed 0.3 random selection mask, generated for
                #              EVERY variant (independent of m). It defines the
                #              "subset of state dims" that masking corrupts and
                #              that partial-reconstruction restricts the loss to.
                #
                #   --atomic-mask     m  1/0 : 1 = encoder input has sel_mask dims
                #                              zeroed (corrupted); 0 = encoder sees
                #                              the clean state (no corruption).
                #   --atomic-asym     a  1/0 : 1 = lightweight bottleneck decoder
                #                              (encoder-decoder asymmetry, He 2022);
                #                              0 = mirror-capacity symmetric decoder.
                #   --atomic-partial  p  1/0 : 1 = MSE on sel_mask dims only
                #                              (partial recon); 0 = MSE on all dims.
                #
                # MAE = (m1,a1,p1) ; DROPOUT = (m1,a0,p0) ;
                # (m0,a0,p0) = plain clean autoencoder (all components off,
                # trivial / Scratch-level — but still trained, non-zero loss).
                sel_mask = make_mask(state_seq, mask_ratio=0.3)        # (B,T+1,13)
                if args.atomic_mask:
                    enc_input = state_seq * (~sel_mask).float()        # corrupt subset
                else:
                    enc_input = state_seq                              # clean input
                z_full, _ = backbone(enc_input)                        # (B,T+1,H)
                s_pred = loss_module(z_full, asymmetric=bool(args.atomic_asym))
                if args.atomic_partial:
                    loss_mask = sel_mask.float()                       # masked dims only
                else:
                    loss_mask = torch.ones_like(sel_mask).float()      # all dims
                loss = (((s_pred - state_seq) ** 2) * loss_mask).sum() / (
                    loss_mask.sum() + 1e-8)
            elif args.objective == "FUSION":
                mask = make_mask(state_seq, mask_ratio=0.3)
                inp_masked = state_seq * (~mask).float()
                z_clean, _ = backbone(inp_full)
                z_masked, _ = backbone(inp_masked)
                loss = loss_module.forward_with_z_full(
                    z_clean, z_masked, state_seq, action_seq, mask
                )
            elif args.objective == "SPR":
                z_online, _ = backbone(inp_full)
                with torch.no_grad():
                    z_target, _ = target_backbone(inp_full)
                loss = loss_module.forward_with_z_full(z_online, z_target, action_seq)
            elif args.objective == "BYOL":
                z_online, _ = backbone(inp_full)
                with torch.no_grad():
                    z_target, _ = target_backbone(inp_full)
                loss = loss_module.forward_with_z_full(z_online, z_target)
            else:
                z_full, _ = backbone(inp_full)                         # (B, T+1, H)
                if args.objective == "FD":
                    loss = loss_module.forward_with_z_full(z_full, action_seq)
                elif args.objective == "IDM":
                    loss = loss_module.forward_with_z_full(z_full, action_seq)
                elif args.objective == "CPC":
                    loss = loss_module.forward_with_z_full(z_full)
                elif args.objective == "QM":
                    loss = loss_module.forward_with_z_full(z_full)
                elif args.objective == "PIDM":
                    loss = loss_module.forward_with_z_full(z_full, action_seq)
                elif args.objective == "VICReg":
                    loss = loss_module.forward_with_z_full(z_full)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(loss_module.parameters()), 1.0
            )
            optimizer.step()

            # SPR: EMA target update (Schwarzer 2021 Eq. 3, decay=0.99)
            if target_backbone is not None:
                _ema_update(target_backbone, backbone, decay=0.99)

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)
        history.append(epoch_loss)
        elapsed = time.time() - t_start
        print(f"[{args.objective}/seed{args.seed}] epoch {epoch+1}/{args.epochs} loss={epoch_loss:.6f} elapsed={elapsed:.1f}s")

    os.makedirs(args.output, exist_ok=True)
    suffix = f"__{args.ckpt_suffix}" if args.ckpt_suffix else ""
    ckpt_name = f"P_{args.objective}__seed{args.seed}{suffix}__backbone.pt"
    ckpt_path = os.path.join(args.output, ckpt_name)
    torch.save({
        "backbone_state_dict": backbone.state_dict(),
        "objective": args.objective,
        "seed": args.seed,
        "loss_history": history,
        "proprio_dim": 13,
        "fusion_hidden": 128,
        "gru_hidden": 128,
    }, ckpt_path)
    print(f"Saved backbone to {ckpt_path}")
    return ckpt_path, history

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objective", type=str, required=True,
                   choices=["FD", "IDM", "MAE", "DROPOUT", "CPC", "QM", "SPR", "FUSION",
                            "PIDM", "VICReg", "BYOL", "ATOMIC"])
    # Experiment C — atomic-component toggles (only used when --objective ATOMIC).
    p.add_argument("--atomic-mask", type=int, default=1, choices=[0, 1],
                   help="ATOMIC: 1=mask 0.3 of input dims, 0=clean input")
    p.add_argument("--atomic-asym", type=int, default=1, choices=[0, 1],
                   help="ATOMIC: 1=encoder sees masked input (asymmetric), 0=clean")
    p.add_argument("--atomic-partial", type=int, default=1, choices=[0, 1],
                   help="ATOMIC: 1=MSE on masked dims only, 0=MSE on all dims")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output", type=str,
                   default=_AG_ROOT + "/aerial_gym/ssl_pretrain/ckpt/")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-steps-per-epoch", type=int, default=0,
                   help="N batches per epoch (0=full). Useful when dataset is huge.")
    p.add_argument("--ckpt-suffix", type=str, default="",
                   help="Extra suffix for ckpt name, e.g. 'srcbaseQuad' → "
                        "P_<obj>__seed<n>__srcbaseQuad__backbone.pt")
    p.add_argument("--max-data-steps", type=int, default=0,
                   help="Quantity ablation: use only first N transitions of dataset "
                        "(0 = full).")
    args = p.parse_args()
    train_one(args)

if __name__ == "__main__":
    main()
