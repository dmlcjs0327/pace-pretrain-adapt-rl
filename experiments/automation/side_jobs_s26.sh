#!/bin/bash
: "${AERIAL_GYM_ROOT:?set AERIAL_GYM_ROOT}"
: "${PACE_EXPERIMENTS:?set PACE_EXPERIMENTS}"
# Side job of pre-registration v2, section 26 (V14): MAE and AP backbones pretrained on target-masked v3 data
# (pretrain_time_ordered_v3.py --target-mask). Each GPU job takes the launch lock of the RL workers, waits for free
# memory, and holds the lock while it allocates (as side_jobs_s25.sh).
# Usage: bash side_jobs_s26.sh   (resumable: finished backbones are skipped)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aerialgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
S="${PACE_EXPERIMENTS}/scripts"
CK=${AERIAL_GYM_ROOT}/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3
LOG="${PACE_EXPERIMENTS}/logs/side_jobs_s26.out"
PL="${PACE_EXPERIMENTS}/logs/pretrain_v3"
LOCK="${PACE_EXPERIMENTS}/logs/study2/gpu_launch.lock"
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; do sleep 30; done; }
mkdir -p "$PL"
echo "start $(date +%H:%M)" >> "$LOG"
ok() { [ -f "$1" ] && python -c "import torch,sys; c=torch.load('$1',map_location='cpu'); sys.exit(0 if len(c['loss_history'])==$2 and c.get('target_mask') else 1)" 2>/dev/null; }
pretrain() {  # objective seed
  f="$CK/P_${1}tmask__seed${2}__srcmixed__backbone.pt"
  ok "$f" 25 && return
  exec 9>"$LOCK"; flock 9
  waitmem 3000
  echo "pretrain ${1}tmask seed $2 start $(date +%H:%M)" >> "$LOG"
  (cd "$S" && python pretrain_time_ordered_v3.py --objective $1 --tag tmask --epochs 25 --seed $2 --target-mask > "$PL/${1}tmask__seed${2}.log" 2>&1) &
  pid=$!; sleep 90; flock -u 9; exec 9>&-
  wait $pid
  echo "pretrain ${1}tmask seed $2 exit $? $(date +%H:%M)" >> "$LOG"
}
for s in 7 42 99; do for o in MAE PIDM; do pretrain $o $s; done; done
echo "side jobs s26 done $(date +%H:%M)" >> "$LOG"
