#!/bin/bash
: "${AERIAL_GYM_ROOT:?set AERIAL_GYM_ROOT}"
: "${PACE_EXPERIMENTS:?set PACE_EXPERIMENTS}"
: "${PACE_DATA:?set PACE_DATA}"
# Side jobs of pre-registration v2, section 25: task-agnostic data v3ag (V3) and its MAE and AP backbones, then MAE
# and AP pretrained for 1 and 5 epochs (V12). Each GPU job takes the launch lock of the RL workers, waits for free
# memory, and holds the lock while it allocates (as side_jobs_s23.sh).
# Usage: bash side_jobs_s25.sh   (resumable: finished files are skipped)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aerialgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
S="${PACE_EXPERIMENTS}/scripts"
D="${PACE_DATA}/v3ag"
CK=${AERIAL_GYM_ROOT}/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3
LOG="${PACE_EXPERIMENTS}/logs/side_jobs_s25.out"
PL="${PACE_EXPERIMENTS}/logs/pretrain_v3"
LOCK="${PACE_EXPERIMENTS}/logs/study2/gpu_launch.lock"
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; do sleep 30; done; }
mkdir -p "$D" "$PL"
echo "start $(date +%H:%M)" >> "$LOG"

# V3 data: 9 episodes of 300 steps x 4,096 vehicles per airframe (the volume of v3), seeds fixed per airframe
for r in x500 lmf2_proprio_clean; do
  short=${r%%_*}
  out="$D/ssl_v3ag_${short}.h5"
  [ -f "$out" ] && continue
  exec 9>"$LOCK"; flock 9
  waitmem 5000
  echo "collect $r start $(date +%H:%M)" >> "$LOG"
  (cd "$S" && python collect_offline_trajectory_v3ag.py --robot $r --episodes 9 --out "$out.tmp" --seed 25 > "$D/collect_${short}.log" 2>&1 && mv "$out.tmp" "$out") &
  pid=$!; sleep 120; flock -u 9; exec 9>&-
  wait $pid
  echo "collect $r exit $? $(date +%H:%M) $(grep -o "diagnostics.*" "$D/collect_${short}.log" | tail -1)" >> "$LOG"
done

ok() { [ -f "$1" ] && python -c "import torch,sys; c=torch.load('$1',map_location='cpu'); sys.exit(0 if len(c['loss_history'])==$2 else 1)" 2>/dev/null; }
pretrain() {  # objective tag epochs seed extra-args
  f="$CK/P_${1}${2}__seed${4}__srcmixed__backbone.pt"
  ok "$f" $3 && return
  exec 9>"$LOCK"; flock 9
  waitmem 3000
  echo "pretrain $1$2 seed $4 ($3 epochs) start $(date +%H:%M)" >> "$LOG"
  (cd "$S" && python pretrain_time_ordered_v3.py --objective $1 --tag "$2" --epochs $3 --seed $4 $5 > "$PL/${1}${2}__seed${4}.log" 2>&1) &
  pid=$!; sleep 90; flock -u 9; exec 9>&-
  wait $pid
  echo "pretrain $1$2 seed $4 exit $? $(date +%H:%M)" >> "$LOG"
}
# V3 backbones (25 epochs, v3 statistics)
if [ -f "$D/ssl_v3ag_x500.h5" ] && [ -f "$D/ssl_v3ag_lmf2.h5" ]; then
  for s in 7 42 99; do for o in MAE PIDM; do pretrain $o ag 25 $s "--data v3ag"; done; done
fi
# V12 backbones: 1 and 5 epochs (constant learning rate, so the same as stopping the 25-epoch run early)
for s in 7 42 99; do for o in MAE PIDM; do for e in 1 5; do pretrain $o e$e $e $s ""; done; done; done
# V8 supervised check (before reading any V8 RL result): Frozen students with whitened features, the observation at
# the heads, and linear heads (raw and whitened), T1 and T2 on R2 and R3, random, MAE, AP, screening seeds
V8CSV="${PACE_EXPERIMENTS}/logs/study2/analysis/v8_students.csv"
for cfg in "whiten 64" "skip 64" "none 0" "whiten 0"; do set -- $cfg; f=$1; h=$2
  for t in state_setpoint_x500_clean ss_lmf2_clean recovery_x500 recovery_lmf2; do for i in random MAE PIDM; do for s in 7 42 99; do
    [ -f "$V8CSV" ] && grep -q "^$t,$i,$s,Frozen,$h,iid,$f," "$V8CSV" && continue
    exec 9>"$LOCK"; flock 9
    waitmem 2500
    (cd "$S" && python study2_s19_student.py --task $t --init $i --seed $s --mode Frozen --feat $f --head-hidden $h 2>&1 | grep -E "R2 action|Error|Traceback" >> "$LOG") &
    pid=$!; sleep 30; flock -u 9; exec 9>&-
    wait $pid
  done; done; done
done
echo "side jobs s25 done $(date +%H:%M)" >> "$LOG"
