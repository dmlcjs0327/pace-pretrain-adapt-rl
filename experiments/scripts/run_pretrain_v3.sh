#!/bin/bash
: "${AERIAL_GYM_ROOT:?set AERIAL_GYM_ROOT}"
: "${PACE_EXPERIMENTS:?set PACE_EXPERIMENTS}"
# Time-ordered pretraining, v3 data: 4 objectives x seeds, 25 epochs each.
# Skips backbones that are complete. Usage: ./run_pretrain_v3.sh "MAE PIDM SPR CPC" "7 42 99"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aerialgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd "${PACE_EXPERIMENTS}/scripts"
LOG="${PACE_EXPERIMENTS}/logs/pretrain_v3"; mkdir -p "$LOG"
CKD=${AERIAL_GYM_ROOT}/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3
for obj in $1; do for seed in $2; do
  CK=$CKD/P_${obj}__seed${seed}__srcmixed__backbone.pt
  if [ -f "$CK" ] && python -c "import torch,sys; sys.exit(0 if len(torch.load('$CK',map_location='cpu')['loss_history'])==25 else 1)" 2>/dev/null; then
    echo "skip $obj $seed"; continue; fi
  python pretrain_time_ordered_v3.py --objective $obj --seed $seed --epochs 25 > "$LOG/${obj}__seed${seed}.log" 2>&1
  echo "done $obj $seed exit=$? $(date +%H:%M)"
done; done
