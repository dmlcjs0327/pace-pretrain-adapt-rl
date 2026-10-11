# PACE: pretrain-adapt controlled evaluation for small RL controllers

Code, pre-registration, analysis tables, and figure scripts for the manuscript

> *Does pretrain-then-adapt work for reinforcement learning in small neural networks? Full fine-tuning helps, freezing hurts.*
> U. Lee and S. Lee, submitted to *Neural Networks* (2026).

The study pretrains a small recurrent backbone (encoder + GRU, 145,801 weights in the full policy) on reward-free
flight data with eight objectives, adapts it to three quadrotor tasks on three airframes with nine adaptation modes,
and runs every mode on a randomly initialized backbone as well. Interventions (shuffled weights, target-swapped,
target-masked, time-shuffled, and task-agnostic data, whitened frozen features, separate actor and critic backbones)
and a cost account complete the comparison, and a quadruped (ANYmal, flat ground and rough terrain) repeats the main
comparison on a second robot. `experiments/logs/study2/ledger.csv` lists every RL run with its stage, and the paper reports 4,551 of them.

The large artifacts (flight data, pretrained backbones, run logs, per-iteration tables) are in the companion dataset
on Hugging Face: <https://huggingface.co/datasets/LUC0327/pace-pretrain-adapt-rl>.

## Layout

| Path | Content |
|---|---|
| `aerial_gym_overlay/` | Our additions and modifications to [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator) v2.0.0: the pretraining package (`aerial_gym/ssl_pretrain`), the policy with adaptation modes (`aerial_gym/nn`, `aerial_gym/rl_training/rl_games`), the tasks T1-T3 and their variants, the airframes R2 (X500) and R3 (LMF2), and the X500 model (`resources/robots/x500`). `setup/apply_overlay.sh` copies it onto an upstream clone. |
| `experiments/scripts/` | Flight-data collection, pretraining, evaluation, and every analysis script (`study2_*.py`). |
| `experiments/automation/` | The run dispatcher `run_study2.py` (one stage per experiment, see the table below), the GPU worker loop, and the pretraining chains (`side_jobs_*.sh`). |
| `experiments/domain2/` | The quadruped: data collection on AnymalTerrain, pretraining, the reward-free probe that chose its settings, and the environment wrapper. |
| `experiments/configs/study2_prereg_v2_ko.md` | The pre-registration, fixed in writing before each stage ran (in Korean, the working language of the authors; the supplementary material of the paper lists every prediction and its outcome in English). |
| `experiments/logs/study2/` | The run ledger and the analysis tables that the paper's numbers come from (`analysis/*.csv`, `*.json`). The two large tables (`series.csv`, per-iteration returns of every run, and `mech_logs.csv`) are on Hugging Face. |
| `paper/figsrc/`, `paper/figures/` | The scripts that draw every figure and table of the paper from the analysis tables, and their output. |
| `data/v3/` | Normalization statistics of the flight data. The data files themselves are on Hugging Face. |

## Setup

The experiments ran on Ubuntu 22.04 with one RTX 3090, Python 3.8, PyTorch 2.4.1 (CUDA 11.8), Isaac Gym Preview 4,
Aerial Gym 2.0.0, rl-games 1.6.5, and warp-lang 1.0.0 (`setup/requirements.txt`).

1. Install [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym) (NVIDIA license, not redistributable) into
   the Python environment. In `isaacgym/gymutil.py`, change `parse_args` to `parse_known_args` in `parse_arguments`,
   as Aerial Gym requires.
2. Clone Aerial Gym v2.0.0 and apply the overlay:
   ```bash
   bash setup/apply_overlay.sh /opt/aerial_gym_simulator   # clones upstream, copies aerial_gym_overlay/, pip install -e
   export AERIAL_GYM_ROOT=/opt/aerial_gym_simulator
   ```
3. For the quadruped, install [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) and set
   `ISAACGYMENVS_ROOT` to its path.
4. Download the companion dataset and place its folders as its card describes (flight data into `data/`, backbones
   into `$AERIAL_GYM_ROOT/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3/`, run logs and the large tables into
   `experiments/logs/`).

Every script reads its roots from `AERIAL_GYM_ROOT`, `PACE_EXPERIMENTS` (default `experiments/` of this repository),
`PACE_DATA` (default `data/`), and `ISAACGYMENVS_ROOT`.

## Reproducing the study

```bash
cd experiments/scripts
# 1. Reward-free flight data of R2 and R3 (4,096 vehicles, scripted collector, no reward)
python collect_offline_trajectory_v3.py --robot x500 --episodes 5 --seed 1 --out ../../data/v3/x500_normal.h5
python collect_offline_trajectory_v3.py --robot x500 --episodes 4 --seed 2 --upset --out ../../data/v3/x500_upset.h5
#    (the same for lmf2_proprio_clean), then
python make_v3_data.py
# 2. Pretrained backbones, one per objective and seed (eight objectives, 25 epochs)
bash run_pretrain_v3.sh                      # or: python pretrain_time_ordered_v3.py --objective MAE --seed 7
# 3. RL runs, stage by stage (the dispatcher writes one rl-games configuration per run and a ledger line per run)
cd ../automation
python run_study2.py --stage C               # calibration of the learning rate on training from scratch
python run_study2.py --stage 1               # screening grid
python run_study2.py --stage 2               # confirmation with ten seeds
#    worker_loop2.sh keeps one GPU worker busy with the stages listed in ../logs/stage_order.txt
# 4. Fixed-state evaluation of the saved policies
bash ../scripts/run_eval_progress.sh 1
# 5. Tables (budgets.csv, series.csv) and the analyses of the paper
cd ../scripts
python study2_parse.py && python study2_analyze.py && python study2_review_stats.py
# 6. Figures
cd ../../paper/figsrc && python make_v6_figs.py
```

Stages of `run_study2.py` and the experiment of the paper that each one is:

| Stage | Experiment | Stage | Experiment |
|---|---|---|---|
| C | Learning-rate calibration (scratch only) | 24 | Restrained modes under a constant rate |
| 1 | Screening grid, 3 seeds | 25 | Frozen and LoRA under a higher learning-rate ceiling |
| 2 | Confirmation, ten seeds | 26 | Task-agnostic pretraining data |
| 3 | Training from scratch for 3,000 iterations; dynamics varied per episode | 27 | Shuffled backbones on ten seeds |
| 4 | Confirmation conditions under a constant rate | 28 | Pipeline variants of training from scratch |
| 5 | Shuffled backbones | 29 | Progress reward in T3 |
| 6 | Difficulty ladder of T3 | 30 | Weaker restraint (Small-LR, L2-SP) |
| 7 | BC with smaller initial noise | 31 | Whitened frozen features; observation fed to the heads |
| 8 | Target-swapped data | 32 | Separate actor and critic backbones |
| 9 | Aux-SSL with smaller weights | 33 | Confirmatory test of freezing, seven new seeds |
| 10 | T3 for 1,800 iterations | 34 | Random backbone with the AP auxiliary loss |
| 11, 12 | Fixed distance curriculum; relative termination | 35 | Pretraining for 1 or 5 epochs |
| 13 | LoRA and LP-FT for 1,800 iterations in T3 | 36 | Success-gated distance curriculum |
| 14, 19, 21 | Network width (x0.5, x2, x4) and the calibration at x4 | 37 | Target-masked data |
| 16 | Expanded input (128 dimensions) | 38 | Random backbone with the CPC auxiliary loss |
| 17 | Time-shuffled data | 40, 41 | Quadruped, flat ground and rough terrain |
| 18 | Curricula on R1 and in T2 | 42 | Diverse initial states in T3 |

Stages 15, 20, 22, 23, and 39 were stopped or discarded before analysis and are not reported, as the
pre-registration records. The supervised students, the teacher data, the representation probes, and the cost
measurement are separate scripts (`study2_s19_*.py`, `study2_probes.py`, `measure_cost.py`).

## License

Our code is released under the BSD 3-Clause License (`LICENSE`). Files under `aerial_gym_overlay/` that modify
Aerial Gym Simulator keep the BSD 3-Clause license of the Autonomous Robots Lab, NTNU. The companion dataset is
released under CC BY 4.0.
