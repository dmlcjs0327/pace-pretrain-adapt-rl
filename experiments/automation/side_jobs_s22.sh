#!/bin/bash
: "${AERIAL_GYM_ROOT:?set AERIAL_GYM_ROOT}"
: "${PACE_EXPERIMENTS:?set PACE_EXPERIMENTS}"
# Side jobs of pre-registration v2, section 22 (no RL runs): M2 relabelling, M1 and M2 supervised students, their
# closed-loop evaluation, and the M3 early-critic analysis. Each GPU job waits until enough GPU memory is free.
# Usage: bash side_jobs_s22.sh   (resumable: finished outputs are skipped)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aerialgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
S="${PACE_EXPERIMENTS}/scripts"
RL=${AERIAL_GYM_ROOT}/aerial_gym/rl_training/rl_games
S19="${PACE_EXPERIMENTS}/logs/study2/s19"
LOG="${PACE_EXPERIMENTS}/logs/side_jobs_s22.out"
TASKS="state_setpoint_x500_clean ss_lmf2_clean recovery_x500 recovery_lmf2"
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; do sleep 30; done; }
echo "start $(date +%H:%M)" >> "$LOG"
# M2 labels of the moving-target condition
for t in $TASKS; do
  [ -f "$S19/relabel_$t.npz" ] && continue
  waitmem 2000; (cd "$S" && python study2_m2_relabel.py --task $t 2>&1 | grep -E "max |Error|Traceback") >> "$LOG"
done
echo "relabel done $(date +%H:%M)" >> "$LOG"
student() {  # task init seed mode head regime
  f="$S19/m12/s19__$1__$2__$4__seed$3__h$5__$6.pth"
  [ -f "$f" ] && return
  waitmem 2500
  (cd "$S" && python study2_s19_student.py --task $1 --init $2 --seed $3 --mode $4 --head-hidden $5 --regime $6 2>&1 | grep -E "R2 action|Error|Traceback") >> "$LOG"
}
for t in $TASKS; do for s in 7 42 99; do
  # M1: head width for frozen backbones, plus FullFT with the widest head
  for i in random MAE PIDM; do for h in 256 1024; do student $t $i $s Frozen $h iid; done; done
  student $t MAE $s FullFT 1024 iid
  # M2: non-IID minibatches and moving targets, frozen and fully fine-tuned
  for i in random MAE; do for m in Frozen FullFT; do for r in noniid nonstat; do student $t $i $s $m 64 $r; done; done; done
done; done
echo "students done $(date +%H:%M)" >> "$LOG"
for t in $TASKS; do
  waitmem 4000
  (cd $RL && python "$S/study2_eval_batch.py" --task $t --files "$S19"/m12/s19__${t}__*.pth --out eval_m12_students.csv 2>&1 | grep -E "L0.0|Error" | tail -1) >> "$LOG"
done
echo "m12 evaluation done $(date +%H:%M)" >> "$LOG"
# M3: early critic quality
for t in $TASKS; do
  waitmem 3000
  (cd $RL && python "$S/study2_m3_critic_early.py" --task $t 2>&1 | grep -E "Error|Traceback" | tail -2) >> "$LOG"
done
echo "m3 done $(date +%H:%M)" >> "$LOG"
echo "side jobs s22 done $(date +%H:%M)" >> "$LOG"
