"""Study 2 launcher (redesigned study after the correction of the recurrent training path).

Pre-registered in experiments/configs/study2_prereg_v2.md (v2 after the review of 2026-10-01: data v3,
backbones ckpt_time_ordered_v3 without R1, fixed input normalization, T3 fixed to 2 s, calibrated PPO).
Stage C (calibration, section 2): random+FullFT, 6 settings x seeds {1001, 1002} x candidates C1-C3,
         logged under logs/study2/calib/. `--stage C`.
Stage 1: 5 initializations (random, MAE, CPC, SPR, PIDM) x 4 adaptation modes (FullFT, LateFT,
         EncFrozen, BackboneLoRA) x X500 and LMF2 x 3 tasks x seeds {7, 42, 99}.
Stage 1 extension (section 11): new backbones IDM, FDM, BC, MAEs and modes DiscLR, L2SP, LPFT, AuxSSL, BCreg.
Stage 2: read logs/study2/stage2_selection.json (written by the pre-registered selection rule, section 11):
         {"objective": <CPC|SPR|PIDM|IDM|FDM|BC|MAEs>,
          "mode": <LateFT|EncFrozen|BackboneLoRA|DiscLR|L2SP|LPFT|AuxSSL>, "iters": <int>}
         initializations {random, MAE, objective} x modes {FullFT, mode} x 9 settings x 10 seeds
         (BC with AuxSSL runs as BCreg).
Stage 3: supplement (section 10): dynamics variation and compute-matched scratch.
Stage 4: learning-rate sensitivity (section 12): stage-2 conditions x 6 screening settings x seeds {7, 42, 99}, C1.
Stage 5: S7 (section 14): shuffled MAE and PIDM backbones x {FullFT, BackboneLoRA} x 6 settings x seeds {7, 42, 99}.
Stage 6: S8 (section 15): T3 difficulty ladder, {random, MAE, PIDM} x {FullFT, BackboneLoRA} x 4 tasks x 3 seeds.
Stages 24-36: review-driven additions V1-V13 (section 25), see cells().
Runs whose backbone is not complete yet are skipped and taken up by a later chunk.

Usage (resumable; several workers share the queue through claim files):
  python run_study2.py --stage 1 --max-minutes 45
"""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_AG_ROOT = _os.environ.get("AERIAL_GYM_ROOT", "/opt/aerial_gym_simulator")
_EXP_ROOT = _os.environ.get("PACE_EXPERIMENTS", str(_REPO / "experiments"))
_DATA_ROOT = _os.environ.get("PACE_DATA", str(_REPO / "data"))
import argparse
import csv
import fcntl
import json
import os
import threading
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

RUNNER_CWD = Path(_AG_ROOT + "/aerial_gym/rl_training/rl_games")
BASE_YAML = RUNNER_CWD / "ppo_ssl_adapt.yaml"
CKPT = Path(_AG_ROOT + "/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3")
STATS = _DATA_ROOT + "/v3/state_stats.pt"
EXPAND_ID_STATS = _DATA_ROOT + "/v3/expand_identity_stats.pt"  # S20
# Second domain (section 20): IsaacGymEnvs Anymal, tasks registered in runner.py under the prefix "ige_"
D2_STATS = _EXP_ROOT + "/domain2/data/stats.pt"
TASKS_D2 = ["ige_anymal_flat", "ige_anymal_blind"]
# Section 29: the second domain moves to AnymalTerrain (same robot model, actuation, and inputs for the
# flat and rough tasks), with new reward-free data and its own normalization; the runs of stage 15 are discarded.
D3_STATS = _EXP_ROOT + "/domain2/data/terrain_stats.pt"
TASKS_D3_FLAT = ["ige_terrain_flat", "ige_terrain_flat_blind"]
TASKS_D3_ROUGH = ["ige_terrain_rough_curr", "ige_terrain_rough"]
S5_D2 = [7, 13, 21, 42, 99]
EXP = Path(_EXP_ROOT + "")
LOG_DIR = EXP / "logs" / "study2"
YAML_DIR = LOG_DIR / "_yaml"
LEDGER = LOG_DIR / "ledger.csv"
SEL = LOG_DIR / "stage2_selection.json"
PPO_CFG = LOG_DIR / "ppo_config.json"   # written by the calibration rule (study2_calib.py)
# Calibration candidates (pre-registration v2, section 2)
CANDIDATES = {"C1": dict(learning_rate=3.0e-4, lr_schedule="constant"),
              "C2": dict(learning_rate=1.0e-4, lr_schedule="constant"),
              "C3": dict(learning_rate=1.0e-4, lr_schedule="adaptive", kl_threshold=0.016, max_lr=1.0e-3)}
S_CAL = [1001, 1002]
# R1 (pre-registration v2, section 23): the calibration of section 2 repeated at width x4 with one more candidate
CANDIDATES_R1 = {**CANDIDATES, "C4": dict(learning_rate=3.0e-5, lr_schedule="constant")}
# V2 (section 25): the adaptive configuration of C3 with a ceiling three times higher
CANDIDATES_X = {**CANDIDATES_R1, "C5": dict(learning_rate=1.0e-4, lr_schedule="adaptive", kl_threshold=0.016, max_lr=3.0e-3)}
R1_JSON = LOG_DIR / "analysis" / "r1_calib.json"   # written by scripts/study2_r1_calib.py (width x4)
R2_JSON = LOG_DIR / "analysis" / "r2_calib.json"   # written by scripts/study2_r1_calib.py --stage 22 (width x8)

TASKS_R2R3 = ["state_setpoint_x500_clean", "recovery_x500", "agg_x500", "ss_lmf2_clean", "recovery_lmf2", "agg_lmf2"]
TASKS_R1 = ["state_setpoint_baseQuad_clean", "recovery_baseQuad", "agg_baseQuad"]
TASKS_DR = ["ss_dr_x500", "ss_dr_lmf2", "recovery_dr_x500", "recovery_dr_lmf2"]
INITS = ["random", "MAE", "CPC", "SPR", "PIDM"]
MODES = ["FullFT", "LateFT", "EncFrozen", "BackboneLoRA"]
# Extension (pre-registration v2 section 11): PIDM keeps its run name but is the action-prediction objective (AP)
INITS_NEW = ["IDM", "FDM", "BC", "MAEs"]
INITS_ALL = INITS + INITS_NEW
MODES_NEW_ALL = ["DiscLR", "L2SP", "LPFT"]          # for every backbone
AUX_INITS = ["random", "MAE", "CPC", "SPR", "PIDM", "IDM", "FDM", "MAEs"]
HEAD_REFIT = {"MAE", "CPC", "SPR", "PIDM"}          # heads refitted on the frozen backbone (5 epochs)
AUX_RANDOM = str(Path(__file__).resolve().parent / "aux_random_MAE.pt")
S3 = [7, 42, 99]
S10 = [7, 13, 21, 42, 99, 123, 256, 512, 777, 2024]
S7NEW = [13, 21, 123, 256, 512, 777, 2024]  # V10 (section 25): the stage-2 seeds not used in screening
S5 = [7, 13, 21, 42, 99]
TASKS_T12 = [t for t in TASKS_R2R3 + TASKS_R1 if not t.startswith("agg")]
AUX_RANDOM_AP = str(Path(__file__).resolve().parent / "aux_random_PIDM.pt")  # V11 (section 25)
AUX_RANDOM_CPC = str(Path(__file__).resolve().parent / "aux_random_CPC.pt")  # V15 (section 27)
EPOCHS_OF = {"MAEe1": 1, "PIDMe1": 1, "MAEe5": 5, "PIDMe5": 5}  # V12 (section 25): shorter pretraining
DISCLR_SCALE = {"DiscLR": 0.01, "DiscLR01": 0.1, "DiscLR03": 0.3}  # V7 (section 25)
L2SP_ALPHA = {"L2SP": 0.01, "L2SP3": 1.0e-3, "L2SP4": 1.0e-4}
STAGE1_ITERS = 600
AUX_LAMBDA = {"AuxSSL": 1.0, "AuxSSL01": 0.1, "AuxSSL001": 0.01}
PRETRAIN_EPOCHS = 25

def cells(stage):
    out = []
    if stage == "C":
        for seed in S_CAL:
            for task in TASKS_R2R3:
                for cand in CANDIDATES:
                    out.append(dict(stage="C" + cand[1], cand=cand, task=task, init="random", mode="FullFT", seed=seed,
                                    iters=STAGE1_ITERS))
    elif stage == 3:
        # Supplement (pre-registration v2, section 10): S3 compute-matched scratch, S1 LP-FT, S2 dynamics variation
        for seed in S3:
            for task in TASKS_R2R3:
                out.append(dict(stage=3, task=task, init="random", mode="FullFT", seed=seed, iters=3000))  # 2,400 -> 3,000 (16:5x note)
        for seed in S3:
            for task in TASKS_DR:
                for init in INITS:
                    for mode in MODES:
                        out.append(dict(stage=3, task=task, init=init, mode=mode, seed=seed, iters=STAGE1_ITERS))
    elif stage == 1:
        for seed in S3:  # seed-major so that partial results stay balanced across conditions
            for task in TASKS_R2R3:
                for init in INITS:
                    for mode in MODES:
                        out.append(dict(stage=1, task=task, init=init, mode=mode, seed=seed, iters=STAGE1_ITERS))
        ext = [(i, m) for i in INITS_NEW for m in MODES] + [(i, m) for i in INITS_ALL for m in MODES_NEW_ALL] \
            + [(i, "AuxSSL") for i in AUX_INITS] + [("BC", "BCreg")]
        for seed in S3:  # extension grid after the original grid
            for task in TASKS_R2R3:
                for init, mode in ext:
                    out.append(dict(stage=1, task=task, init=init, mode=mode, seed=seed, iters=STAGE1_ITERS))
    elif stage == 5:
        # S7 (section 14): statistics-matched, structure-free backbones (shuffled within each tensor / gate block)
        for seed in S3:
            for task in TASKS_R2R3:
                for init in ("MAEshuf", "PIDMshuf"):
                    for mode in ("FullFT", "BackboneLoRA"):
                        out.append(dict(stage=5, task=task, init=init, mode=mode, seed=seed, iters=STAGE1_ITERS))
    elif stage == 6:
        # S8 (section 15): difficulty ladder of T3 (target distance [1, 2] m and [2, 4] m; [4, 8] m is stage 1)
        for seed in S3:
            for task in ("agg_d1_x500", "agg_d1_lmf2", "agg_d2_x500", "agg_d2_lmf2"):
                for init in ("random", "MAE", "PIDM"):
                    for mode in ("FullFT", "BackboneLoRA"):
                        out.append(dict(stage=6, task=task, init=init, mode=mode, seed=seed, iters=STAGE1_ITERS))
    elif stage == 7:
        # S9 (section 16): BC with its initial action std scaled by 1/4 (AG_BC_SIGMA_SCALE in dispatch)
        for seed in S3:
            for task in TASKS_R2R3:
                for mode in ("FullFT", "BCreg"):
                    out.append(dict(stage=7, task=task, init="BC", mode=mode, seed=seed, iters=STAGE1_ITERS))
    elif stage == 8:  # S10 (section 17): backbones pretrained on target-swapped data, FullFT
        # T3 first: it carries the prediction of section 17; then T1 and T2 (order only, same cells)
        for tasks in (["agg_x500", "agg_lmf2"], [t for t in TASKS_R2R3 if not t.startswith("agg")]):
            for seed in S3:
                for task in tasks:
                    for init in ("MAEtswap", "PIDMtswap"):
                        out.append(dict(stage=8, task=task, init=init, mode="FullFT", seed=seed, iters=STAGE1_ITERS))
    elif stage == 9:  # S11 (section 17): MAE with AuxSSL at a smaller auxiliary weight
        for seed in S3:
            for task in TASKS_R2R3:
                for mode in ("AuxSSL01", "AuxSSL001"):
                    out.append(dict(stage=9, task=task, init="MAE", mode=mode, seed=seed, iters=STAGE1_ITERS))
    elif stage == 10:  # S13 (section 17): longer budget for T3, failed and successful pairs of the confirmation stage
        for task, seed in (("agg_x500", 21), ("agg_x500", 512), ("agg_lmf2", 13), ("agg_lmf2", 21), ("agg_lmf2", 99),
                           ("agg_lmf2", 123), ("agg_x500", 99), ("agg_lmf2", 7)):
            out.append(dict(stage=10, task=task, init="MAE", mode="FullFT", seed=seed, iters=1800))
    elif stage in (11, 12):  # S14 distance curriculum, S15 relative termination (section 18), T3 only
        for seed in S3:
            for task in ("agg_x500", "agg_lmf2"):
                for init, mode in (("random", "FullFT"), ("MAE", "FullFT"), ("PIDM", "FullFT"), ("MAE", "BackboneLoRA")):
                    out.append(dict(stage=stage, task=task, init=init, mode=mode, seed=seed, iters=STAGE1_ITERS))
    elif stage == 14:  # S17 (section 19): network width x0.5, x2, x4 (37k, 578k, 2.30M parameters; x1 = stage 1)
        for seed in S3:
            for width in (0.5, 2.0, 4.0):
                for task in TASKS_R2R3:
                    for init in ("random", "MAE"):
                        for mode in ("FullFT", "EncFrozen", "BackboneLoRA"):
                            out.append(dict(stage=14, task=task, init=init, mode=mode, seed=seed, iters=600,
                                            width=width))
    elif stage == 15:  # S18 (section 20): second domain, Anymal flat and without base linear velocity
        for seed in S5_D2:
            for task in TASKS_D2:
                for init in ("random", "MAE", "PIDM"):
                    for mode in ("FullFT", "EncFrozen", "BackboneLoRA"):
                        out.append(dict(stage=15, task=task, init=init, mode=mode, seed=seed, iters=600))
                for init in ("MAEshuf", "PIDMshuf"):
                    out.append(dict(stage=15, task=task, init=init, mode="FullFT", seed=seed, iters=600))
    elif stage == 16:  # S20 (section 21): expanded, mixed, noisy input (13 -> 128) through a fixed sensor model
        for seed in S3:
            for task in TASKS_R2R3:
                for init in ("random", "MAE"):
                    for mode in ("FullFT", "EncFrozen"):
                        out.append(dict(stage=16, task=task, init=init, mode=mode, seed=seed, iters=600, expand=True))
    elif stage == 17:  # S21 (section 21): MAE pretrained on windows with permuted time steps
        for seed in S3:
            for task in TASKS_R2R3:
                for mode in ("FullFT", "BackboneLoRA"):
                    out.append(dict(stage=17, task=task, init="MAEtshuf", mode=mode, seed=seed, iters=600))
    elif stage == 19:  # R1 (section 23): calibration rule of section 2 at width x4, scratch only, candidates C1-C4
        for seed in S_CAL:
            for task in TASKS_R2R3:
                for cand in CANDIDATES_R1:
                    out.append(dict(stage=19, cand=cand, task=task, init="random", mode="FullFT", seed=seed,
                                    iters=STAGE1_ITERS, width=4.0))
    elif stage == 20:  # R2 (section 23): width x8, the conditions of stage 14 with the study configuration (C3)
        for seed in S3:
            for task in TASKS_R2R3:
                for init in ("random", "MAE"):
                    for mode in ("FullFT", "EncFrozen", "BackboneLoRA"):
                        out.append(dict(stage=20, task=task, init=init, mode=mode, seed=seed, iters=600, width=8.0))
    elif stage == 21:  # R1 follow-up (section 23): the width-x4 conditions of stage 14 with the R1 winner (C1)
        win = json.loads(R1_JSON.read_text())["chosen"] if R1_JSON.exists() else None
        if win and win != "C3":
            for seed in S3:
                for task in TASKS_R2R3:
                    for init in ("random", "MAE"):
                        for mode in ("FullFT", "EncFrozen", "BackboneLoRA"):
                            out.append(dict(stage=21, cand=win, task=task, init=init, mode=mode, seed=seed, iters=600,
                                            width=4.0))
    elif stage == 22:  # R2 calibration (section 23, run because R1 chose a candidate other than C3): width x8
        for seed in S_CAL:
            for task in TASKS_R2R3:
                for cand in CANDIDATES_R1:
                    out.append(dict(stage=22, cand=cand, task=task, init="random", mode="FullFT", seed=seed,
                                    iters=STAGE1_ITERS, width=8.0))
    elif stage == 23:  # R2 follow-up (section 23): the width-x8 conditions with the width-x8 winner, if not C3
        win = json.loads(R2_JSON.read_text())["chosen"] if R2_JSON.exists() else None
        if win and win != "C3":
            for seed in S3:
                for task in TASKS_R2R3:
                    for init in ("random", "MAE"):
                        for mode in ("FullFT", "EncFrozen", "BackboneLoRA"):
                            out.append(dict(stage=23, cand=win, task=task, init=init, mode=mode, seed=seed, iters=600,
                                            width=8.0))
    elif stage == 18:  # S23 (section 21): (a) distance curriculum on the held-out airframe R1, (b) T2 curriculum
        for seed in S3:
            for init in ("random", "MAE", "PIDM"):
                out.append(dict(stage=18, task="agg_baseQuad", init=init, mode="FullFT", seed=seed, iters=600))
            for task in ("recovery_x500", "recovery_lmf2"):
                for init in ("random", "MAE"):
                    out.append(dict(stage=18, task=task, init=init, mode="FullFT", seed=seed, iters=600))
    elif stage == 24:  # V1 (section 25): restrained modes with C1, completing stage 4
        for seed in S3:
            for task in TASKS_R2R3:
                for init in ("random", "MAE", "PIDM"):
                    for mode in ("LateFT", "EncFrozen", "BackboneLoRA"):
                        out.append(dict(stage=24, cand="C1", task=task, init=init, mode=mode, seed=seed,
                                        iters=STAGE1_ITERS))
    elif stage == 25:  # V2 (section 25): Frozen and LoRA with the higher ceiling C5
        for seed in S3:
            for task in TASKS_R2R3:
                for init in ("random", "MAE", "PIDM"):
                    for mode in ("EncFrozen", "BackboneLoRA"):
                        out.append(dict(stage=25, cand="C5", task=task, init=init, mode=mode, seed=seed,
                                        iters=STAGE1_ITERS))
    elif stage == 26:  # V3 (section 25): backbones pretrained on task-agnostic data (v3ag)
        for seed in S3:
            for task in TASKS_R2R3:
                for init in ("MAEag", "PIDMag"):
                    for mode in ("FullFT", "EncFrozen"):
                        out.append(dict(stage=26, task=task, init=init, mode=mode, seed=seed, iters=STAGE1_ITERS))
    elif stage == 38:  # V15 (section 27): random backbone with the CPC (contrastive) auxiliary loss during RL
        for seed in S3:
            for task in TASKS_R2R3:
                out.append(dict(stage=38, task=task, init="random", mode="AuxSSLcpc", seed=seed, iters=STAGE1_ITERS))
    elif stage == 40:  # V17 (section 29): second domain on flat ground (with and without base velocity), stage-15 grid
        for seed in S5_D2:
            for task in TASKS_D3_FLAT:
                for init in ("random", "MAE", "PIDM"):
                    for mode in ("FullFT", "EncFrozen", "BackboneLoRA"):
                        out.append(dict(stage=40, task=task, init=init, mode=mode, seed=seed, iters=600))
                for init in ("MAEshuf", "PIDMshuf"):
                    out.append(dict(stage=40, task=task, init=init, mode="FullFT", seed=seed, iters=600))
    elif stage == 41:  # V18 (section 30): hard task of the second domain, rough terrain without the height scan
        for seed in S5_D2:
            for task in TASKS_D3_ROUGH:
                for init in ("random", "MAE", "PIDM"):
                    for mode in ("FullFT", "EncFrozen"):
                        out.append(dict(stage=41, task=task, init=init, mode=mode, seed=seed, iters=600))
    elif stage == 42:  # V19 (section 31): T3 with diverse initial states x success-gated curriculum x pretraining
        for seed in S3:
            for task in ("agg_upset_x500", "agg_upset_lmf2"):
                for init in ("random", "MAE"):
                    for tag in ("", "_succ"):
                        out.append(dict(stage=42, task=task, init=init, mode="FullFT", seed=seed, iters=STAGE1_ITERS,
                                        tag=tag))
    elif stage == 39:  # V16 (section 28): why the quadruped differs; masking unit x data breadth, FullFT and Frozen
        for seed in S5_D2:
            for task in TASKS_D2:
                for init in ("MAEs", "MAEbr", "MAEsbr", "PIDMbr"):
                    for mode in ("FullFT", "EncFrozen"):
                        out.append(dict(stage=39, task=task, init=init, mode=mode, seed=seed, iters=600))
    elif stage == 37:  # V14 (section 26): backbones pretrained on target-masked data, FullFT (as stage 8, T3 first)
        for tasks in (["agg_x500", "agg_lmf2"], [t for t in TASKS_R2R3 if not t.startswith("agg")]):
            for seed in S3:
                for task in tasks:
                    for init in ("MAEtmask", "PIDMtmask"):
                        out.append(dict(stage=37, task=task, init=init, mode="FullFT", seed=seed, iters=STAGE1_ITERS))
    elif stage == 27:  # V4 (section 25): shuffled backbones on the ten seeds of T1 and T2 (cells of stage 5 skipped)
        for seed in S10:
            for task in TASKS_T12:
                if seed in S3 and task in TASKS_R2R3:
                    continue
                for init in ("MAEshuf", "PIDMshuf"):
                    out.append(dict(stage=27, task=task, init=init, mode="FullFT", seed=seed, iters=STAGE1_ITERS))
    elif stage == 28:  # V5 (section 25): pipeline variants of the scratch baseline, each with its MAE pair
        for seed in S3:
            for pipe in ("bptt32", "sig05"):
                for task in TASKS_R2R3:
                    for init in ("random", "MAE"):
                        out.append(dict(stage=28, task=task, init=init, mode="FullFT", seed=seed, iters=STAGE1_ITERS,
                                        pipe=pipe, tag="_" + pipe))
            for task in TASKS_R2R3:
                out.append(dict(stage=28, task=task, init="random", mode="FullFT", seed=seed, iters=STAGE1_ITERS,
                                pipe="runnorm", tag="_runnorm"))
            for task in ("agg_x500", "agg_lmf2"):
                for init in ("random", "MAE"):
                    out.append(dict(stage=28, task=task, init=init, mode="FullFT", seed=seed, iters=STAGE1_ITERS,
                                    pipe="ent1e3", tag="_ent1e3"))
    elif stage == 29:  # V6 (section 25): T3 with a progress term k (d_{t-1} - d_t) during training only
        for seed in S5:
            for k in (10, 100):
                for task in ("agg_x500", "agg_lmf2"):
                    for init in ("random", "MAE", "PIDM"):
                        out.append(dict(stage=29, task=task, init=init, mode="FullFT", seed=seed, iters=STAGE1_ITERS,
                                        prog=k, tag=f"_prog{k}"))
    elif stage == 30:  # V7 (section 25): weaker restraint, Small-LR x0.1, x0.3 and L2-SP alpha 1e-3, 1e-4
        for seed in S3:
            for task in TASKS_R2R3:
                for init in ("random", "MAE"):
                    for mode in ("DiscLR01", "DiscLR03", "L2SP3", "L2SP4"):
                        out.append(dict(stage=30, task=task, init=init, mode=mode, seed=seed, iters=STAGE1_ITERS))
    elif stage == 31:  # V8 (section 25): Frozen with whitened backbone output, or with the observation fed to the heads
        for seed in S3:
            for feat in ("whiten", "skip"):
                for task in TASKS_R2R3:
                    for init in ("random", "MAE", "PIDM"):
                        out.append(dict(stage=31, task=task, init=init, mode="EncFrozen", seed=seed,
                                        iters=STAGE1_ITERS, feat=feat, tag="_" + feat))
    elif stage == 32:  # V9 (section 25): separate actor and critic backbones with their own memory
        for seed in S5:
            for task in [t for t in TASKS_R2R3 if not t.startswith("agg")]:
                for cond in ("none", "actor", "critic", "both"):
                    out.append(dict(stage=32, task=task, init="random" if cond == "none" else "MAE", mode="FullFT",
                                    seed=seed, iters=STAGE1_ITERS, asym=cond, tag="_dual" + cond))
    elif stage == 33:  # V10 (section 25): confirmatory test of freezing on the seven seeds not used in screening
        for seed in S7NEW:
            for task in TASKS_R2R3 + TASKS_R1:
                for init in ("MAE", "PIDM"):
                    out.append(dict(stage=33, task=task, init=init, mode="EncFrozen", seed=seed, iters=STAGE1_ITERS))
    elif stage == 34:  # V11 (section 25): random backbone with the auxiliary loss of AP (matched pair of D2)
        for seed in S10:
            for task in TASKS_R2R3 + TASKS_R1:
                out.append(dict(stage=34, task=task, init="random", mode="AuxSSLap", seed=seed, iters=STAGE1_ITERS))
    elif stage == 35:  # V12 (section 25): MAE and AP pretrained for 1 and 5 epochs
        for seed in S3:
            for task in TASKS_R2R3:
                for init in ("MAEe1", "MAEe5", "PIDMe1", "PIDMe5"):
                    out.append(dict(stage=35, task=task, init=init, mode="FullFT", seed=seed, iters=STAGE1_ITERS))
    elif stage == 36:  # V13 (section 25): success-gated distance curriculum in T3 (Aerial Gym navigation defaults)
        for seed in S10:
            for task in ("agg_x500", "agg_lmf2"):
                for init in ("random", "MAE", "PIDM"):
                    out.append(dict(stage=36, task=task, init=init, mode="FullFT", seed=seed, iters=STAGE1_ITERS,
                                    tag="_succ"))
    elif stage == 13:  # S16 (section 18): longer budget for LoRA and LP-FT in T3
        for seed in S3:
            for task in ("agg_x500", "agg_lmf2"):
                for mode in ("BackboneLoRA", "LPFT"):
                    out.append(dict(stage=13, task=task, init="MAE", mode=mode, seed=seed, iters=1800))
    elif stage == 4:
        # S4 learning-rate sensitivity (section 12): the stage-2 conditions on the screening settings with the
        # screening seeds, trained with candidate C1 (constant 3e-4)
        sel = json.loads(SEL.read_text())
        for seed in S3:
            for task in TASKS_R2R3:
                for init in ("random", "MAE", sel["objective"]):
                    for mode in ("FullFT", sel["mode"]):
                        mode = "BCreg" if (init, mode) == ("BC", "AuxSSL") else mode
                        out.append(dict(stage=4, cand="C1", task=task, init=init, mode=mode, seed=seed,
                                        iters=STAGE1_ITERS))
    else:
        sel = json.loads(SEL.read_text())
        for seed in S10:
            for task in TASKS_R2R3 + TASKS_R1:
                for init in ("random", "MAE", sel["objective"]):
                    for mode in ("FullFT", sel["mode"]):
                        # BC has no AuxSSL cell: keeping its own pretraining loss is BCreg (section 11, 16:5x note)
                        mode = "BCreg" if (init, mode) == ("BC", "AuxSSL") else mode
                        out.append(dict(stage=2, task=task, init=init, mode=mode, seed=seed, iters=int(sel["iters"])))
    return out

def run_id(c):
    init = c["init"] + (f"w{c['width']:g}" if c.get("width", 1.0) != 1.0 else "")  # S17: width in the run name
    init += "x" if c.get("expand") else ""  # S20: expanded input
    init += c["cand"] if c["stage"] in (19, 22) else ""  # R1, R2: calibration candidate in the run name
    init += c.get("tag", "")  # section 25: variant of the stage (pipeline, progress term, features, dual backbone)
    return f"s2__st{c['stage']}__{c['task']}__{init}__{c['mode']}__seed{c['seed']}__it{c['iters']}"

def ckpt_for(c):
    if c["init"] == "random":
        return None
    wtag = f"__w{c['width']:g}" if c.get("width", 1.0) != 1.0 else ""
    src = ("d2terrain" if c["task"].startswith("ige_terrain") else "d2anymal") if c["task"].startswith("ige_") \
        else "srcmixed"  # second domain (sections 20 and 29)
    xtag = "x" if c.get("expand") else ""  # S20: MAE pretrained on the expanded input (pretrain --expand --tag x)
    return str(CKPT / f"P_{c['init']}{xtag}__seed{c['seed']}__{src}{wtag}__backbone.pt")

def reg_ckpt_for(c):
    """Checkpoint whose objective and head the AuxSSL mode continues."""
    if c["mode"] == "AuxSSLap":  # V11 (section 25)
        return AUX_RANDOM_AP
    if c["mode"] == "AuxSSLcpc":  # V15 (section 27)
        return AUX_RANDOM_CPC
    if c["init"] == "random":
        return AUX_RANDOM
    if c["init"] in HEAD_REFIT:
        return str(CKPT / f"P_{c['init']}_head__seed{c['seed']}__srcmixed__backbone.pt")
    return ckpt_for(c)

def ckpt_ready(c):
    import torch
    if c["task"].startswith("ige_terrain") and not Path(D3_STATS).exists():
        return False  # second domain after section 29: normalization from the terrain data
    if c["task"].startswith("ige_") and not Path(D2_STATS).exists():
        return False  # second domain: input normalization from its reward-free data (section 20)
    if c["init"] == "BC" and not Path(ckpt_for(c).replace("__backbone.pt", "__bc_sigma.json")).exists():
        return False  # initial action noise of the BC policy
    need = [] if c["init"] == "random" else [(ckpt_for(c), EPOCHS_OF.get(c["init"], PRETRAIN_EPOCHS))]
    if c["mode"] == "AuxSSLap" and not Path(AUX_RANDOM_AP).exists():
        return False
    if c["mode"] == "AuxSSLcpc" and not Path(AUX_RANDOM_CPC).exists():
        return False
    if c["mode"].startswith("AuxSSL") and c["init"] in HEAD_REFIT:
        need.append((reg_ckpt_for(c), 5))
    for p, n in need:
        try:
            if len(torch.load(p, map_location="cpu")["loss_history"]) != n:
                return False
        except Exception:
            return False
    return True

def make_yaml(c):
    with BASE_YAML.open() as f:
        spec = yaml.safe_load(f)
    rid = run_id(c)
    spec["params"]["seed"] = c["seed"]
    cfg = spec["params"]["config"]
    cfg["name"] = cfg["full_experiment_name"] = rid
    cfg["env_name"] = c["task"]
    cfg["max_epochs"] = c["iters"]
    cfg["num_actors"] = 4096
    cfg["env_config"]["num_envs"] = 4096
    # Learning-rate setting: the calibration candidate in stage C, otherwise the calibrated one.
    lr_cfg = CANDIDATES_X[c["cand"]] if "cand" in c else json.loads(PPO_CFG.read_text())["config"]
    cfg["learning_rate"] = lr_cfg["learning_rate"]
    cfg["lr_schedule"] = lr_cfg["lr_schedule"]
    cfg["kl_threshold"] = lr_cfg.get("kl_threshold", 0.016)
    # Entropy coefficient of the PPO configuration that Aerial Gym ships for position control
    # (ppo_aerial_quad.yaml). The earlier value 0.01 made the action noise diverge.
    cfg["entropy_coef"] = 1.0e-4
    # Fixed input normalization with the pretraining statistics for every backbone (v2, section 1).
    cfg["normalize_input"] = False
    net = spec["params"].setdefault("network", {})
    net["name"] = "aerial_gym_pretrain_adapt"
    pa = net.setdefault("pretrain_adapt", {})
    base_mode = "FullFT" if c["mode"] in ("DiscLR", "L2SP", "AuxSSL", "AuxSSL01", "AuxSSL001", "BCreg", "DiscLR01",
                                         "DiscLR03", "L2SP3", "L2SP4", "AuxSSLap", "AuxSSLcpc") else c["mode"]
    w = c.get("width", 1.0)  # S17 (section 19): every hidden width x w; 1 = the study default
    pa.update(proprio_dim=48 if c["task"].startswith("ige_") else (128 if c.get("expand") else 13), proprio_hidden=int(round(64 * w)), fusion_hidden=int(round(128 * w)),
              gru_hidden=int(round(128 * w)), head_hidden=int(round(64 * w)), adapt_mode=base_mode, lora_rank=8)
    for k in ("reg_mode", "reg_lambda", "reg_ckpt", "ssl_actor_head"):
        pa.pop(k, None)
    if c["mode"] == "L2SP":
        pa.update(reg_mode="l2sp", reg_lambda=0.01)
    elif c["mode"] in AUX_LAMBDA:  # AuxSSL01 and AuxSSL001: S11 (section 17)
        pa.update(reg_mode="aux_ssl", reg_lambda=AUX_LAMBDA[c["mode"]], reg_ckpt=reg_ckpt_for(c))
    elif c["mode"] == "BCreg":
        pa.update(reg_mode="bc_reg", reg_lambda=1.0)
    elif c["mode"] in ("L2SP3", "L2SP4"):  # V7 (section 25): weaker L2-SP
        pa.update(reg_mode="l2sp", reg_lambda=L2SP_ALPHA[c["mode"]])
    elif c["mode"] in ("AuxSSLap", "AuxSSLcpc"):  # V11 (section 25), V15 (section 27): random backbone, AP or CPC loss
        pa.update(reg_mode="aux_ssl", reg_lambda=1.0, reg_ckpt=reg_ckpt_for(c))
    if c["init"] == "BC":
        pa["ssl_actor_head"] = True
    pa.setdefault("use_vae_branch", False)
    pa["fixed_input_norm"] = (D3_STATS if c["task"].startswith("ige_terrain") else D2_STATS) if c["task"].startswith("ige_") \
        else (EXPAND_ID_STATS if c.get("expand") else STATS)
    ck = ckpt_for(c)
    if ck:
        pa["ssl_backbone_ckpt"] = ck
    else:
        pa.pop("ssl_backbone_ckpt", None)
    # Section 25 variants (off for every earlier stage)
    pipe = c.get("pipe")
    if pipe == "bptt32":  # V5a: truncated backpropagation through the whole rollout of 32 steps
        cfg["seq_length"] = 32
    elif pipe == "sig05":  # V5b: initial action standard deviation 0.5
        pa["log_std_init"] = -0.6931471805599453
    elif pipe == "runnorm":  # V5c: rl-games running input normalization instead of the fixed one (scratch only)
        cfg["normalize_input"] = True
        pa.pop("fixed_input_norm", None)
    elif pipe == "ent1e3":  # V5d: entropy coefficient 1e-3
        cfg["entropy_coef"] = 1.0e-3
    if c.get("feat") == "whiten":  # V8a: fixed whitening of the frozen backbone output
        pa["feat_whiten"] = True
    elif c.get("feat") == "skip":  # V8b: the standardized observation is also fed to the heads
        pa["skip_obs"] = True
    if c.get("asym"):  # V9: separate actor and critic backbones, each with its own memory
        pa.update(use_asymmetric_backbone=True, asym_separate_memory=True)
        pa.pop("ssl_backbone_ckpt", None)
        if c["asym"] in ("actor", "both"):
            pa["ssl_actor_ckpt"] = ck
        if c["asym"] in ("critic", "both"):
            pa["ssl_critic_ckpt"] = ck
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    p = YAML_DIR / f"{rid}.yaml"
    with p.open("w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    return p

def reusable(c):
    """A stage-2 run identical to a finished stage-1 run (same setting, init, mode, seed, budget)."""
    if c["stage"] != 2:
        return None
    c1 = dict(c, stage=1)
    return c1 if done(c1) else None

def done(c):
    log = LOG_DIR / f"{run_id(c)}.log"
    n = c["iters"]
    return log.exists() and re.search(rf"epoch: {n}/{n}", log.read_text(errors="ignore")) is not None

def dispatch(c):
    rid = run_id(c)
    yp = make_yaml(c)
    log = LOG_DIR / f"{rid}.log"
    env = dict(os.environ)
    env["AG_PASS_TRAIN_INFO"] = "1"
    if c["task"].startswith("ige_terrain"):  # section 29: AnymalTerrain runs in its own GPU context, outside MPS
        env["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/no-mps-d2"
    env.pop("AG_LEGACY_SEQ_ORDER", None)
    lr_cfg = CANDIDATES_X[c["cand"]] if "cand" in c else json.loads(PPO_CFG.read_text())["config"]
    env.pop("AG_ADAPTIVE_MAX_LR", None)
    env.pop("AG_UNFREEZE_AT", None)
    env.pop("AG_BACKBONE_LR_SCALE", None)
    env.pop("AG_CRITIC_WARMUP", None)
    env.pop("AG_ACTOR_WARMUP", None)
    env.pop("AG_BC_SIGMA_SCALE", None)
    if c["stage"] == 7:  # S9 (section 16): BC initial action std x 1/4
        env["AG_BC_SIGMA_SCALE"] = "0.25"
    env.pop("AG_T3_CURRICULUM", None)
    env.pop("AG_T3_TERM_MARGIN", None)
    env.pop("AG_OBS_EXPAND", None)
    env.pop("AG_T2_CURRICULUM", None)
    env.pop("AG_T3_PROGRESS", None)
    env.pop("AG_T3_SUCCESS_CURRICULUM", None)
    if c.get("prog"):  # V6 (section 25): progress term during training only
        env["AG_T3_PROGRESS"] = str(c["prog"])
    if c["stage"] == 42 and c.get("tag") == "_succ":  # V19 (section 31): the curriculum of stage 36
        env["AG_T3_SUCCESS_CURRICULUM"] = "1:2,2:4,4:8"
    if c["stage"] == 36:  # V13 (section 25): success-gated distance curriculum during training only
        env["AG_T3_SUCCESS_CURRICULUM"] = "1:2,2:4,4:8"
    if c.get("expand"):  # S20 (section 21): the environment wrapper standardizes and expands the observation
        env["AG_OBS_EXPAND"] = "1"
        env["AG_OBS_EXPAND_STATS"] = STATS
    if c["stage"] == 18 and c["task"] == "agg_baseQuad":  # S23a: the S14 distance curriculum on R1
        env["AG_T3_CURRICULUM"] = "100:1:2,200:2:4"
    if c["stage"] == 18 and c["task"].startswith("recovery"):  # S23b: initial-perturbation curriculum for T2
        env["AG_T2_CURRICULUM"] = "100:0.25,200:0.5"
    if c["stage"] == 11:  # S14 (section 18): distance curriculum during training only
        env["AG_T3_CURRICULUM"] = "100:1:2,200:2:4"
    if c["stage"] == 12:  # S15 (section 18): termination at the initial distance + 4 m during training only
        env["AG_T3_TERM_MARGIN"] = "4"
    if c["mode"] in DISCLR_SCALE:  # backbone learning rate x0.01 (SGI); x0.1 and x0.3 in V7 (section 25)
        env["AG_BACKBONE_LR_SCALE"] = str(DISCLR_SCALE[c["mode"]])
    if c["init"] == "BC":  # critic first for 50 iterations (PIRLNav, Xing et al. 2024)
        env["AG_CRITIC_WARMUP"] = "50"
        env["AG_ACTOR_WARMUP"] = "50"  # then a linear actor learning-rate warm-up
    if c["mode"] == "LPFT":  # head-only training first, then the whole network
        env["AG_UNFREEZE_AT"] = str(c.get("unfreeze_at", 100))
    if lr_cfg.get("max_lr"):
        env["AG_ADAPTIVE_MAX_LR"] = str(lr_cfg["max_lr"])
    env["LD_LIBRARY_PATH"] = f"{sys.prefix}/lib:" + env.get("LD_LIBRARY_PATH", "")
    cmd = [sys.executable, "runner.py", f"--task={c['task']}", "--num_envs=4096", "--headless=True",
           "--use_warp=False", f"--file={yp}", "--train", f"--experiment_name={rid}"]
    t0 = time.time()
    with log.open("w") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] CMD: {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(RUNNER_CWD), stdout=f, stderr=subprocess.STDOUT, env=env)
        # Watchdog: a run whose log has not grown for 30 minutes is
        # hung (two runs hung at a full disk and held 12.6 GB and two workers for 4.5 hours). It is terminated and
        # recorded with its exit code; a run prints a line every iteration, so 30 minutes of silence never occurs otherwise.
        # wait() returns as soon as the run exits; the former sleep(60) poll rounded wall_s
        # up to the next minute, so wall_s of earlier ledgers is up to 60 s too long.
        while True:
            try:
                proc.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                pass
            if time.time() - log.stat().st_mtime > HANG_SECONDS:
                print(f"{rid} hung: no log output for {HANG_SECONDS // 60} min, terminated", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        rc = proc.returncode
    wall = time.time() - t0
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "stage", "task", "init", "mode", "seed", "iters", "exit", "wall_s", "finished"])
        w.writerow([rid, c["stage"], c["task"], c["init"] + c.get("tag", ""), c["mode"], c["seed"], c["iters"], rc, round(wall, 1),
                    datetime.now().isoformat(timespec="seconds")])
    return rc

# GPU memory gate. Measured per-run memory with 4,096 envs:
# width x4 about 5.3 GB, other widths and the second robot 3.6-4.8 GB. Five workers share 24.6 GB under MPS, so a run
# starts only when enough memory is free. A shared lock serializes the check and holds it while the new run allocates.
HANG_SECONDS = 1800  # watchdog of dispatch (operations only)
GPU_NEED_MIB = {8.0: 7000, 4.0: 5800, 2.0: 5000}  # width x8 (section 23): first runs used 5.3-6.3 GB

def gpu_free_mib():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                           text=True).split()[0])
    except Exception:
        return 10 ** 6

def wait_for_gpu_memory(c, rid):
    need = GPU_NEED_MIB.get(float(c.get("width", 1.0)), 4900)
    if c.get("mode") == "AuxSSLcpc":  # V15 (section 27): the CPC logits raise the peak to about 7.2 GB
        need = 7600
    if c["task"].startswith("ige_terrain"):  # section 29: AnymalTerrain in its own context (outside MPS)
        need = 6500
    lockf = open(LOG_DIR / "gpu_launch.lock", "w")
    fcntl.flock(lockf, fcntl.LOCK_EX)
    waited = 0
    while gpu_free_mib() < need:
        time.sleep(30)
        waited += 30
    if waited:
        print(f"{rid} waited {waited}s for {need} MiB of GPU memory", flush=True)

    def release():
        fcntl.flock(lockf, fcntl.LOCK_UN)
        lockf.close()
    threading.Timer(120, release).start()  # the new run allocates its memory within this time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["C", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"] + [str(k) for k in range(24, 43)], required=True)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    global LOG_DIR, YAML_DIR, LEDGER
    if a.stage == "C":
        LOG_DIR, YAML_DIR, LEDGER = LOG_DIR / "calib", LOG_DIR / "calib" / "_yaml", LOG_DIR / "calib" / "ledger.csv"
    else:
        a.stage = int(a.stage)
        if not PPO_CFG.exists():
            raise SystemExit("ppo_config.json missing: run the calibration stage and study2_calib.py first")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cells(a.stage) if not done(c) and not reusable(c)]
    print(f"stage {a.stage}: {len(todo)} runs to do", flush=True)
    if a.dry:
        for c in todo[:12]:
            print(run_id(c), "ready" if ckpt_ready(c) else "waiting for backbone")
        return
    t_end = time.time() + 60 * a.max_minutes
    for c in todo:
        if time.time() > t_end:
            print("time budget reached", flush=True)
            break
        rid = run_id(c)
        if done(c) or not ckpt_ready(c):
            continue
        claim = LOG_DIR / f"{rid}.claim"
        if claim.exists() and time.time() - claim.stat().st_mtime > 10800:  # wide (S17) and long runs exceed 1 h
            claim.unlink(missing_ok=True)
        try:
            os.close(os.open(str(claim), os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        wait_for_gpu_memory(c, rid)
        rc = dispatch(c)
        print(f"{rid} exit={rc}", flush=True)

if __name__ == "__main__":
    main()
