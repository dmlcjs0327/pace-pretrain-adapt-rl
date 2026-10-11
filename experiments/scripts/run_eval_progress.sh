#!/bin/bash
: "${AERIAL_GYM_ROOT:?set AERIAL_GYM_ROOT}"
: "${PACE_EXPERIMENTS:?set PACE_EXPERIMENTS}"
# Incremental fixed-state evaluation while training runs (one process at a time: the output table is shared),
# with the batched evaluator study2_eval_batch.py (16 policies per simulation, identical initial states).
# Early checkpoints 25 and 100 without wind and the terminal checkpoints 550, 575, 600
# without wind and under the two steady winds (section 8). Finished evaluations are skipped.
# Usage: bash run_eval_progress.sh [stage]   (default 1)
STAGE=${1:-1}
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aerialgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
S="${PACE_EXPERIMENTS}/scripts"
TASKS="state_setpoint_x500_clean recovery_x500 agg_x500 ss_lmf2_clean recovery_lmf2 agg_lmf2"
[ "$STAGE" = "2" ] && TASKS="$TASKS state_setpoint_baseQuad_clean recovery_baseQuad agg_baseQuad"
# S8 difficulty ladder (section 15): only the new target-distance levels d1, d2 (d3 = T3 of stage 1)
[ "$STAGE" = "6" ] && TASKS="agg_d1_x500 agg_d1_lmf2 agg_d2_x500 agg_d2_lmf2"
# V19 (section 31): T3 with diverse initial states, evaluated on its own initial-state distribution
[ "$STAGE" = "42" ] && TASKS="agg_upset_x500 agg_upset_lmf2"
cd ${AERIAL_GYM_ROOT}/aerial_gym/rl_training/rl_games
for t in $TASKS; do
  python "$S/study2_eval_batch.py" --task $t --stage $STAGE --iters 25 100 --disturb 0 2>&1 | grep -E "Traceback|Error|/" | tail -1 | sed "s/^/$t early: /"
  # S6 (section 13): intermediate checkpoints for the noise-free learning curve (stage 1; stage 7 = S9, section 16)
  { [ "$STAGE" = "1" ] || [ "$STAGE" = "7" ]; } && python "$S/study2_eval_batch.py" --task $t --stage $STAGE --iters 50 150 200 250 300 350 400 450 500 --disturb 0 2>&1 | grep -E "Traceback|Error|/" | tail -1 | sed "s/^/$t curve: /"
  for L in 0 1 2; do
    python "$S/study2_eval_batch.py" --task $t --stage $STAGE --iters 550 575 600 --disturb $L 2>&1 | grep -E "Traceback|Error|/" | tail -1 | sed "s/^/$t terminal L$L: /"
  done
done
echo "eval pass finished $(date +%H:%M)"
