"""Information probes on pretrained backbones without RL.

For every backbone file given (or every file in the pilot directory), it computes the information probe of
study2_probes.py on the fixed probe set of the v3 flight data (R2, R3, and the unseen R1): accuracy of a
logistic regression that identifies the airframe, and R^2 of ridge regressions of the velocity, tilt, and body
rate 5 and 10 steps after the window, from the final GRU output. Inputs are standardized with the fixed
statistics (data/v3/state_stats.pt), as in RL. A random backbone (seed 7) is included as the reference.
Usage: python pilot_probe_backbones.py [files...]
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
_DATA_ROOT = _os.environ.get("PACE_DATA", str(_REPO / "data"))
import isaacgym  # noqa: F401
import glob
import json
import sys

import numpy as np
import torch

import study2_drift as D
import study2_probes as P

PILOT = _AG_ROOT + "/aerial_gym/ssl_pretrain/ckpt_pilot_mae/"

def main():
    files = sys.argv[1:] or sorted(glob.glob(PILOT + "*.pt")) + [D.CK + "P_MAE__seed7__srcmixed__backbone.pt"]
    Xraw, lab, T = P.probe_set()
    st = torch.load(_DATA_ROOT + "/v3/state_stats.pt")
    x = torch.tensor((Xraw - np.array(st["state_mean"])) / np.array(st["state_std"]), dtype=torch.float32)
    x = torch.clamp(x, -5, 5)
    tr, te = P.split(len(lab))
    torch.manual_seed(7)
    from aerial_gym.ssl_pretrain.backbone import ProprioGRUBackbone
    rnd = ProprioGRUBackbone(proprio_dim=13)
    W_r = D.effective_backbone({"a2c_network.policy." + k: v for k, v in rnd.state_dict().items()})
    out = {}
    for name, W in [("random", W_r)] + [(f.split("/")[-1].replace("__srcmixed__backbone.pt", ""), D.ssl_backbone(f)) for f in files]:
        with torch.no_grad():
            Z = D.features(W, x).numpy()
        r = P.info_probe(Z, lab, T, tr, te)
        r = {k: round(max(v, 0.0) if k.startswith("r2_") else v, 3) for k, v in r.items()}
        er, dd = P.plasticity(torch.tensor(Z))
        r["erank"], r["dead"] = round(er, 2), round(dd, 3)
        out[name] = r
        print(f"{name:32s} airframe {r['acc_airframe']:.3f}  vel+10 {r['r2_vel_10']:.3f}  tilt+10 {r['r2_tilt_10']:.3f}  rate+5 {r['r2_rate_5']:.3f}  rate+10 {r['r2_rate_10']:.3f}  erank {r['erank']}", flush=True)
    json.dump(out, open(_EXP_ROOT + "/logs/pilot_mae/probe_results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
