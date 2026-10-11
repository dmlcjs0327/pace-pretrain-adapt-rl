"""Standalone cost measurement for S3.

Run only after all training, when nothing else uses the GPU (it refuses otherwise). Measures one job at a time,
as process wall time (start-up included, as in the RL ledger):
  collection  the four collector runs of the R2 and R3 flight data (same arguments and seeds as data/v3) into a
              temporary directory, and the merge of make_v3_data.py redirected to that directory
  pretrain    the 25-epoch pretraining of the eight objectives (seed 7) into a separate directory
  refit       the 5-epoch head refits of MAE, CPC, SPR, PIDM (frozen seed-7 backbones)
  rl          600-iteration runs on T1xR2 with seed 7: random+FullFT, random+L2SP, BC+BCreg, and AuxSSL with
              every backbone it applies to
Writes logs/study2/analysis/cost_standalone.json (seconds) and deletes the temporary flight data.
Usage: python measure_cost.py [--dry] [--only collection,pretrain,refit,rl]
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_DATA_ROOT = _os.environ.get("PACE_DATA", str(_REPO / "data"))
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "automation"))
import run_study2 as S  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = S.EXP / "logs" / "study2" / "cost"
TMP = OUT / "tmp"
RES = S.EXP / "logs" / "study2" / "analysis" / "cost_standalone.json"
COLLECT = [("x500", 5, False, 1, "x500_normal.h5"), ("x500", 4, True, 2, "x500_upset.h5"),
           ("lmf2_proprio_clean", 5, False, 3, "lmf2_normal.h5"), ("lmf2_proprio_clean", 4, True, 4, "lmf2_upset.h5")]
OBJ_ARGS = {"MAE": ["--objective", "MAE"], "CPC": ["--objective", "CPC"], "SPR": ["--objective", "SPR"],
            "PIDM": ["--objective", "PIDM"], "IDM": ["--objective", "IDM"], "FDM": ["--objective", "FDM"],
            "BC": ["--objective", "BC"], "MAEs": ["--objective", "MAE", "--mask-unit", "step", "--mask-ratio", "-1"]}
RL_CELLS = [("random", "FullFT"), ("random", "L2SP"), ("BC", "BCreg")] + [(i, "AuxSSL") for i in S.AUX_INITS]

def gpu_busy():
    """GPU processes other than the MPS server."""
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], capture_output=True,
                       text=True)
    return [ln.split(",")[0].strip() for ln in q.stdout.splitlines() if ln.strip() and "mps-server" not in ln]

def timed(cmd, log, cwd=HERE, env=None):
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    dt = time.time() - t0
    if rc:
        raise SystemExit(f"failed ({rc}): {' '.join(map(str, cmd))}, see {log}")
    return round(dt, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", default="collection,pretrain,refit,rl")
    a = ap.parse_args()
    parts = a.only.split(",")
    busy = gpu_busy()
    if busy and not a.dry:
        raise SystemExit(f"GPU in use by {busy}: the measurement must run alone")
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    res = json.loads(RES.read_text()) if RES.exists() else {}
    env = dict(os.environ, LD_LIBRARY_PATH=f"{sys.prefix}/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))
    py = sys.executable
    if "collection" in parts:
        t = {}
        for robot, ep, upset, seed, fn in COLLECT:
            cmd = [py, "collect_offline_trajectory_v3.py", "--robot", robot, "--episodes", str(ep), "--seed", str(seed),
                   "--out", str(TMP / fn)] + (["--upset"] if upset else [])
            print(" ".join(cmd)) if a.dry else t.__setitem__(fn, timed(cmd, OUT / f"collect_{fn}.log", env=env))
        merge = TMP / "make_v3_data_tmp.py"
        src = (HERE / "make_v3_data.py").read_text()
        orig = 'D = _DATA_ROOT + "/v3/"'
        assert src.count(orig) == 1
        merge.write_text(src.replace(orig, f'D = "{TMP}/"'))
        if a.dry:
            print("merge:", merge)
        else:
            t["merge"] = timed([py, str(merge)], OUT / "merge.log", env=env)
            res["t_collect"] = round(sum(t.values()), 1)
            res["t_collect_parts"] = t
            for f in TMP.glob("*.h5"):
                f.unlink()
    if "pretrain" in parts:
        res.setdefault("t_pre", {})
        for o, args in OBJ_ARGS.items():
            cmd = [py, "pretrain_time_ordered_v3.py", *args, "--seed", "7", "--epochs", "25", "--out-dir", str(TMP / "ckpt")]
            if a.dry:
                print(" ".join(cmd))
                continue
            res["t_pre"][o] = timed(cmd, OUT / f"pretrain_{o}.log", env=env)
    if "refit" in parts:
        res.setdefault("t_refit", {})
        for o in sorted(S.HEAD_REFIT):
            cmd = [py, "pretrain_time_ordered_v3.py", "--objective", o, "--seed", "7", "--epochs", "5", "--tag", "_head",
                   "--init-backbone", str(S.CKPT / f"P_{o}__seed7__srcmixed__backbone.pt"), "--freeze-backbone",
                   "--out-dir", str(TMP / "ckpt")]
            if a.dry:
                print(" ".join(cmd))
                continue
            res["t_refit"][o] = timed(cmd, OUT / f"refit_{o}.log", env=env)
    if "rl" in parts:
        S.LOG_DIR, S.YAML_DIR, S.LEDGER = OUT, OUT / "_yaml", OUT / "ledger.csv"
        res.setdefault("t_rl", {})
        for init, mode in RL_CELLS:
            c = dict(stage="cost", task="state_setpoint_x500_clean", init=init, mode=mode, seed=7, iters=600)
            if a.dry:
                print(S.run_id(c), "ready" if S.ckpt_ready(c) else "backbone missing")
                continue
            t0 = time.time()
            rc = S.dispatch(c)
            if rc:
                raise SystemExit(f"RL run failed: {S.run_id(c)}")
            res["t_rl"][f"{init}|{mode}"] = round(time.time() - t0, 1)
    if not a.dry:
        res["measured"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        RES.write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
