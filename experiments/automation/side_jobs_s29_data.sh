#!/bin/bash
: "${AERIAL_GYM_ROOT:?set AERIAL_GYM_ROOT}"
: "${PACE_EXPERIMENTS:?set PACE_EXPERIMENTS}"
# Section 29 (V17), steps 2-6, after the collector (side_jobs_s29_collector.sh) has finished:
#   2. reward-free data on flat ground and rough terrain (collect_terrain.py, outside MPS, one simulation per process)
#   3. fixed input normalization (terrain_stats.py)
#   4. candidate backbones, seed 7: MAE e30, e60, smix and AP h2, h4, h8 (pretrain_d2.py --data terrain)
#   5. reward-free probe and selection (probe_d2.py -> domain2/data/probe_selection.json)
#   6. final backbones of the selected settings, seeds 13 21 42 99 (seed 7 = the candidate file), and shuffled backbones
# GPU jobs take the launch lock of the RL workers, wait for free memory, and hold the lock until they have allocated.
# Resumable: finished files are skipped. Usage: bash side_jobs_s29_data.sh
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aerialgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
EXP="${PACE_EXPERIMENTS}"
D2="$EXP/domain2"
RC=${AERIAL_GYM_ROOT}/aerial_gym/rl_training/rl_games
CK=${AERIAL_GYM_ROOT}/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3
LOG="$EXP/logs/side_jobs_s29.out"
PL="$EXP/logs/pretrain_d2"
LOCK="$EXP/logs/study2/gpu_launch.lock"
mkdir -p "$PL"
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; do sleep 30; done; }
ok() { [ -f "$1" ] && python -c "import torch,sys; c=torch.load('$1',map_location='cpu'); sys.exit(0 if len(c['loss_history'])==25 else 1)" 2>/dev/null; }
gpu_job() {  # name memory_MiB ready_pattern nomps(0/1) command...
  local name=$1 mem=$2 ready=$3 nomps=$4; shift 4
  exec 9>"$LOCK"; flock 9
  waitmem "$mem"
  echo "$name start $(date +%H:%M)" >> "$LOG"
  if [ "$nomps" = "1" ]; then
    (cd "$D2" && CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps-d2 "$@" > "$PL/$name.log" 2>&1) &
  else
    (cd "$D2" && "$@" > "$PL/$name.log" 2>&1) &
  fi
  local pid=$! t=0
  sleep 90
  while [ -n "$ready" ] && kill -0 $pid 2>/dev/null && ! grep -q "$ready" "$PL/$name.log" && [ $t -lt 900 ]; do
    sleep 10; t=$((t + 10)); done
  flock -u 9; exec 9>&-
  wait $pid
  echo "$name exit $? $(date +%H:%M)" >> "$LOG"
}
pretrain() {  # objective tag seed extra-args...
  local obj=$1 tag=$2 seed=$3; shift 3
  ok "$CK/P_${obj}${tag}__seed${seed}__d2terrain__backbone.pt" && return
  gpu_job "${obj}${tag}__seed${seed}__d2terrain" 4000 "epoch 1/25" 0 python pretrain_d2.py --objective "$obj" --seed "$seed" --data terrain --tag "$tag" "$@"
}
# 1. wait for the collector
until ls $RC/runs/d2_collector_terrain/nn/last_d2_collector_terrain_ep_1500_rew_*.pth >/dev/null 2>&1; do sleep 120; done
COL=$(ls $RC/runs/d2_collector_terrain/nn/last_d2_collector_terrain_ep_1500_rew_*.pth | head -1)
echo "data start, collector $(basename "$COL") $(date +%H:%M)" >> "$LOG"
# 2. data (two processes in parallel)
for k in plane rough; do
  [ -f "$D2/data/terrain_$k.npz" ] || gpu_job "collect_terrain_$k" 4500 "" 1 python collect_terrain.py "$COL" $k &
  sleep 200
done
wait
grep -h "collector on" "$PL"/collect_terrain_*.log >> "$LOG"
# 3. normalization
[ -f "$D2/data/terrain_stats.pt" ] || (cd "$D2" && python terrain_stats.py >> "$LOG" 2>&1)
# 4. candidates (two lanes)
( pretrain MAE e30 7 --mask-unit entry --mask-ratio 0.3; pretrain MAE e60 7 --mask-unit entry --mask-ratio 0.6;
  pretrain MAE smix 7 --mask-unit step --mask-ratio -1 ) &
( pretrain PIDM h2 7 --ap-steps 2; pretrain PIDM h4 7 --ap-steps 4; pretrain PIDM h8 7 --ap-steps 8 ) &
wait
# 5. probe and selection
if [ ! -f "$D2/data/probe_selection.json" ]; then
  exec 9>"$LOCK"; flock 9; waitmem 3000
  (cd "$D2" && python probe_d2.py $CK/P_MAEe30__seed7__d2terrain__backbone.pt $CK/P_MAEe60__seed7__d2terrain__backbone.pt \
     $CK/P_MAEsmix__seed7__d2terrain__backbone.pt $CK/P_PIDMh2__seed7__d2terrain__backbone.pt \
     $CK/P_PIDMh4__seed7__d2terrain__backbone.pt $CK/P_PIDMh8__seed7__d2terrain__backbone.pt --random > "$PL/probe.log" 2>&1)
  flock -u 9; exec 9>&-
  grep "selected\|score" "$PL/probe.log" >> "$LOG"
fi
# 6. final backbones of the selected settings
args_of() {  # tag -> pretraining arguments
  case $1 in e30) echo "--mask-unit entry --mask-ratio 0.3";; e60) echo "--mask-unit entry --mask-ratio 0.6";;
    smix) echo "--mask-unit step --mask-ratio -1";; h2) echo "--ap-steps 2";; h4) echo "--ap-steps 4";; h8) echo "--ap-steps 8";; esac; }
SEL_MAE=$(python -c "import json; s=json.load(open('$D2/data/probe_selection.json'))['selected']['MAE']; print(s.split('__')[0][len('P_MAE'):])")
SEL_AP=$(python -c "import json; s=json.load(open('$D2/data/probe_selection.json'))['selected']['PIDM']; print(s.split('__')[0][len('P_PIDM'):])")
echo "selected MAE $SEL_MAE, AP $SEL_AP $(date +%H:%M)" >> "$LOG"
[ -f "$CK/P_MAE__seed7__d2terrain__backbone.pt" ] || cp "$CK/P_MAE${SEL_MAE}__seed7__d2terrain__backbone.pt" "$CK/P_MAE__seed7__d2terrain__backbone.pt"
[ -f "$CK/P_PIDM__seed7__d2terrain__backbone.pt" ] || cp "$CK/P_PIDM${SEL_AP}__seed7__d2terrain__backbone.pt" "$CK/P_PIDM__seed7__d2terrain__backbone.pt"
( for s in 13 21 42 99; do pretrain MAE "" $s $(args_of $SEL_MAE); done ) &
( for s in 13 21 42 99; do pretrain PIDM "" $s $(args_of $SEL_AP); done ) &
wait
(cd "$EXP/scripts" && python make_shuffled_backbones.py --objectives MAE PIDM --seeds 7 13 21 42 99 --src d2terrain >> "$LOG" 2>&1)
echo "side jobs s29 data done $(date +%H:%M)" >> "$LOG"
