#!/bin/bash
: "${AERIAL_GYM_ROOT:?set AERIAL_GYM_ROOT}"
: "${PACE_EXPERIMENTS:?set PACE_EXPERIMENTS}"
# Side jobs of pre-registration v2, section 28 (V16): why the quadruped differs from the quadrotor.
#   lane 1: MAE-T (run name MAEs) on the original Anymal data, then AP on the broad data (PIDMbr), seeds 7 13 21 42 99
#   lane 2: broad data (collect_data.py --broad), then MAE (MAEbr) and MAE-T (MAEsbr) on it, seeds 7 13 21 42 99
# Each GPU job takes the launch lock of the RL workers, waits for free memory, and holds the lock until its first
# epoch is done (pretraining) or for 90 s (collection), so that its memory is allocated before the workers launch.
# allocated; MAEs seed 7 ended with out-of-memory at its first step and is rerun from the start.)
# Resumable: finished backbones (25 epochs) and finished data are skipped.
# Usage: bash side_jobs_s28.sh
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aerialgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
D2="${PACE_EXPERIMENTS}/domain2"
CK=${AERIAL_GYM_ROOT}/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3
LOG="${PACE_EXPERIMENTS}/logs/side_jobs_s28.out"
PL="${PACE_EXPERIMENTS}/logs/pretrain_d2"
LOCK="${PACE_EXPERIMENTS}/logs/study2/gpu_launch.lock"
BROAD="$D2/data/anymal_flat_broad.npz"
COLLECTOR="$D2/collector_anymal_ep1000.pth"
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; do sleep 30; done; }
mkdir -p "$PL"
echo "start $(date +%H:%M)" >> "$LOG"
ok() { [ -f "$1" ] && python -c "import torch,sys; c=torch.load('$1',map_location='cpu'); sys.exit(0 if len(c['loss_history'])==25 else 1)" 2>/dev/null; }
gpu_job() {  # name memory_MiB ready_pattern command...
  local name=$1 mem=$2 ready=$3; shift 3
  exec 9>"$LOCK"; flock 9
  waitmem "$mem"
  echo "$name start $(date +%H:%M)" >> "$LOG"
  (cd "$D2" && "$@" > "$PL/$name.log" 2>&1) &
  local pid=$! t=0
  sleep 90
  while [ -n "$ready" ] && kill -0 $pid 2>/dev/null && ! grep -q "$ready" "$PL/$name.log" && [ $t -lt 900 ]; do
    sleep 10; t=$((t + 10)); done
  flock -u 9; exec 9>&-
  wait $pid
  echo "$name exit $? $(date +%H:%M)" >> "$LOG"
}
pretrain() {  # objective seed data
  local tag=""; [ "$3" = "broad" ] && tag="br"
  ok "$CK/P_${1}${tag}__seed${2}__d2anymal__backbone.pt" && return
  gpu_job "${1}${tag}__seed${2}" 4000 "epoch 1/25" python pretrain_d2.py --objective "$1" --seed "$2" --data "$3"
}
broad_ready() { [ -f "$D2/data/stats_broad.pt" ] && ! pgrep -f "collect_data.py.*--broad" >/dev/null; }
lane1() {
  for s in 7 13 21 42 99; do pretrain MAEs $s flat; done
  until broad_ready; do sleep 60; done
  for s in 7 13 21 42 99; do pretrain PIDM $s broad; done
}
lane2() {
  if ! broad_ready && ! pgrep -f "collect_data.py.*--broad" >/dev/null; then
    gpu_job collect_broad 4000 "" python collect_data.py "$COLLECTOR" --broad; fi
  until broad_ready; do sleep 60; done
  for s in 7 13 21 42 99; do pretrain MAE $s broad; pretrain MAEs $s broad; done
}
lane1 & lane2 & wait
echo "side jobs s28 done $(date +%H:%M)" >> "$LOG"
