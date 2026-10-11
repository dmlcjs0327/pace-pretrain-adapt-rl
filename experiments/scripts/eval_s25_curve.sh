#!/bin/bash
: "${AERIAL_GYM_ROOT:?set AERIAL_GYM_ROOT}"
: "${PACE_EXPERIMENTS:?set PACE_EXPERIMENTS}"
# Pre-registration v2, section 25 (analysis only): fixed-state evaluation of the intermediate checkpoints (50, 150, ...,
# 500) of the stage-2 FullFT runs in T1 and T2 (random, MAE, AP; ten seeds; R1-R3), for the time to the final level of
# scratch on the noise-free evaluation (stage-1 runs already have them, section 13). Takes the launch lock of the RL
# workers and waits for free memory before each task. Finished evaluations are skipped.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aerialgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
S="${PACE_EXPERIMENTS}/scripts"
L="${PACE_EXPERIMENTS}/logs"
LOCK="$L/study2/gpu_launch.lock"
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; do sleep 30; done; }
cd ${AERIAL_GYM_ROOT}/aerial_gym/rl_training/rl_games
for t in state_setpoint_x500_clean recovery_x500 ss_lmf2_clean recovery_lmf2 state_setpoint_baseQuad_clean recovery_baseQuad; do
  runs=$(awk -F, -v t=$t '$2==2 && $3==t && $5=="FullFT" && $8==0 {print $1}' "$L/study2/ledger.csv" | sort -u | tr '\n' ' ')
  exec 9>"$LOCK"; flock 9
  waitmem 4000
  python "$S/study2_eval_batch.py" --task $t --stage 2 --iters 50 150 200 250 300 350 400 450 500 --disturb 0 --runs $runs 2>&1 | grep -E "Traceback|Error|/" | tail -1 | sed "s/^/$t curve $(date +%H:%M): /" >> "$L/eval_s25_curve.out" &
  pid=$!; sleep 90; flock -u 9; exec 9>&-
  wait $pid
done
echo "curve eval finished $(date +%H:%M)" >> "$L/eval_s25_curve.out"
