#!/bin/bash
: "${PACE_EXPERIMENTS:?set PACE_EXPERIMENTS}"
# One GPU worker that keeps taking 45-minute chunks from the stages in priority order (pre-registration v2).
# Same as worker_loop.sh, but the priority order is read from logs/stage_order.txt before every chunk, so the order can
# change without restarting the workers.
# Stop it by creating the file STOP_<worker> in the log folder.
# Usage: bash worker_loop2.sh <worker number>
W=$1
cd "${PACE_EXPERIMENTS}/automation"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aerialgym
LOG=../logs/study2_w$W.out
while [ ! -f ../logs/STOP_w$W ]; do
  STAGES=$(cat ../logs/stage_order.txt 2>/dev/null || echo "4 3")
  ran=0
  for st in $STAGES; do
    n=$(python run_study2.py --stage $st --max-minutes 45 2>&1 | tee -a $LOG | grep -c " exit=")
    if [ "$n" -gt 0 ]; then ran=1; break; fi
  done
  [ $ran -eq 0 ] && { echo "w$W idle $(date +%H:%M)" >> $LOG; sleep 300; }
done
echo "w$W stopped $(date +%H:%M)" >> $LOG
