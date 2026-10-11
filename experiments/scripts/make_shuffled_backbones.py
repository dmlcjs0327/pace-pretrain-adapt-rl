"""Structure-free, statistics-matched backbones for S7 (pre-registration v2, section 14).

For a pretrained backbone (MAE or PIDM, the action-prediction objective) and seed, every weight and bias tensor of
the encoder and the GRU is shuffled element-wise within itself, so the multiset of values (hence every per-tensor
statistic: mean, spread, norm, distribution) is kept while the learned structure is destroyed. The GRU matrices and
biases are shuffled within each gate block (reset, update, new) so that the per-gate statistics are kept too.
The permutation is drawn from a generator seeded by the run seed and the tensor name. Output:
ckpt_time_ordered_v3/P_<OBJ>shuf__seed<s>__srcmixed__backbone.pt (same format; loss_history copied so that the
launcher accepts it; key 'shuffled_from' names the source).
Usage: python make_shuffled_backbones.py --objectives MAE PIDM --seeds 7 42 99
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
import argparse
import zlib
from pathlib import Path

import torch

CK = Path(_AG_ROOT + "/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3")

def shuffle(t, gen, blocks=1):
    parts = t.reshape(blocks, -1)
    out = torch.empty_like(parts)
    for i in range(blocks):
        out[i] = parts[i][torch.randperm(parts.shape[1], generator=gen)]
    return out.reshape(t.shape)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objectives", nargs="+", default=["MAE", "PIDM"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 99])
    ap.add_argument("--src", default="srcmixed", help="source tag: srcmixed (quadrotor) or d2anymal (section 20)")
    a = ap.parse_args()
    for o in a.objectives:
        for s in a.seeds:
            src = CK / f"P_{o}__seed{s}__{a.src}__backbone.pt"
            ck = torch.load(src, map_location="cpu")
            sd = ck["backbone_state_dict"]
            new = {}
            for k, v in sd.items():
                gen = torch.Generator().manual_seed(s * 1000003 + zlib.crc32(k.encode()))
                blocks = 3 if k.startswith("gru.") else 1  # gate blocks of the GRU (r, z, n)
                new[k] = shuffle(v, gen, blocks)
                assert torch.equal(new[k].reshape(blocks, -1).sort(dim=1).values, v.reshape(blocks, -1).sort(dim=1).values)
                assert v.numel() < 2 or not torch.equal(new[k], v)
            out = {key: val for key, val in ck.items() if key not in ("backbone_state_dict", "head_state_dict")}
            out["backbone_state_dict"] = new
            out["shuffled_from"] = src.name
            dst = CK / f"P_{o}shuf__seed{s}__{a.src}__backbone.pt"
            torch.save(out, dst)
            print(dst.name, "from", src.name, {k: (round(float(v.mean()), 4), round(float(v.std()), 4)) for k, v in list(new.items())[:2]})

if __name__ == "__main__":
    main()
