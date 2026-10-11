#!/bin/bash
: "${AERIAL_GYM_ROOT:?set AERIAL_GYM_ROOT}"
: "${PACE_EXPERIMENTS:?set PACE_EXPERIMENTS}"
# Fixed-state evaluation of every finished Study 2 run (incremental), one process per setting.
# Usage: bash run_eval_all.sh [stage]   (default 1)
STAGE=${1:-1}
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aerialgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
S="${PACE_EXPERIMENTS}/scripts"
TASKS="state_setpoint_x500_clean recovery_x500 agg_x500 ss_lmf2_clean recovery_lmf2 agg_lmf2"
[ "$STAGE" = "2" ] && TASKS="$TASKS state_setpoint_baseQuad_clean recovery_baseQuad agg_baseQuad"
cd ${AERIAL_GYM_ROOT}/aerial_gym/rl_training/rl_games
for t in $TASKS; do
  for L in 0 1 2; do python "$S/study2_eval.py" --task $t --stage $STAGE --disturb $L 2>&1 | grep -E "ret=|nothing|Error|Traceback" | tail -3; done
done
