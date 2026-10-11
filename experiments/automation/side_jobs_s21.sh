#!/bin/bash
: "${AERIAL_GYM_ROOT:?set AERIAL_GYM_ROOT}"
: "${PACE_EXPERIMENTS:?set PACE_EXPERIMENTS}"
# Side jobs of pre-registration v2, section 21 (not RL runs): teacher data, code checks, pretraining for S20 and S21,
# and the supervised students of S19. Each GPU job waits until enough GPU memory is free (the RL workers share it).
# Usage: bash side_jobs_s21.sh   (resumable: finished outputs are skipped)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aerialgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
S="${PACE_EXPERIMENTS}/scripts"
RL=${AERIAL_GYM_ROOT}/aerial_gym/rl_training/rl_games
CK=${AERIAL_GYM_ROOT}/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3
S19="${PACE_EXPERIMENTS}/logs/study2/s19"
SMOKE=/tmp/claude-1000/-home-user-aerial-gym-simulator-my-research-2026-CoRL/89253667-86b1-4a9b-9acd-feb0ad9efaee/scratchpad/smoke_s21.py
LOG=../logs/side_jobs_s21.out
cd "${PACE_EXPERIMENTS}/automation"
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; do sleep 30; done; }
echo "start $(date +%H:%M)" >> $LOG

# 1. teacher data of the last T2 setting
if [ ! -f "$S19/teacher_recovery_lmf2.npz" ]; then
  waitmem 3500; (cd $RL && python "$S/study2_s19_teacher.py" --task recovery_lmf2 2>&1 | grep -E "steps x|Error" | tail -2) >> $LOG
fi
# 2. code checks (3 PPO iterations each; separate names and logs, not results)
for args in "16 state_setpoint_x500_clean random EncFrozen" "18 recovery_x500 random FullFT" "18 agg_baseQuad random FullFT"; do
  waitmem 5500; python $SMOKE $args 3 >> $LOG 2>&1
done
# 3. pretraining: MAE on the expanded input (S20) and on time-shuffled windows (S21), two at a time
pre() {  # $1 extra flags, $2 tag, $3 seed
  f=$CK/P_MAE$2__seed$3__srcmixed__backbone.pt
  [ -f "$f" ] && return
  waitmem 3500
  (cd "$S" && python pretrain_time_ordered_v3.py --objective MAE --seed $3 --tag $2 $1 2>&1 | grep -E "epoch (1|25)/|Saved|Error|Traceback") >> $LOG
}
for s in 7 42 99; do
  pre "--expand" x $s &
  sleep 120
  pre "--time-shuffle" tshuf $s
  wait
done
echo "pretraining done $(date +%H:%M)" >> $LOG
# code checks of the new backbones (3 PPO iterations each)
for args in "16 state_setpoint_x500_clean MAE FullFT" "17 state_setpoint_x500_clean MAEtshuf BackboneLoRA"; do
  waitmem 5500; python $SMOKE $args 3 >> $LOG 2>&1
done
# 4. supervised students of S19 (3 backbones x 3 seeds x 3 modes x 4 settings)
for t in state_setpoint_x500_clean ss_lmf2_clean recovery_x500 recovery_lmf2; do
  for i in random MAE PIDM; do
    for s in 7 42 99; do
      for m in Frozen PartialFT FullFT; do
        [ -f "$S19/students/s19__${t}__${i}__${m}__seed${s}.pth" ] && continue
        waitmem 2500
        (cd "$S" && python study2_s19_student.py --task $t --init $i --seed $s --mode $m 2>&1 | grep -E "R2 action|Error|Traceback") >> $LOG
      done
    done
  done
done
echo "students done $(date +%H:%M)" >> $LOG
# 5. closed-loop evaluation of the students (mean action, fixed initial states, no wind)
for t in state_setpoint_x500_clean ss_lmf2_clean recovery_x500 recovery_lmf2; do
  waitmem 4000
  (cd $RL && python "$S/study2_eval_batch.py" --task $t --files "$S19"/students/s19__${t}__*.pth --out eval_s19_students.csv 2>&1 | grep -E "/|Error" | tail -2) >> $LOG
done
echo "side jobs done $(date +%H:%M)" >> $LOG
