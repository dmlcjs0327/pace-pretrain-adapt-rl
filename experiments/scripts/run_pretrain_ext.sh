#!/bin/bash
: "${AERIAL_GYM_ROOT:?set AERIAL_GYM_ROOT}"
: "${PACE_EXPERIMENTS:?set PACE_EXPERIMENTS}"
# Extension pretraining (pre-registration v2 section 11): new objectives with 25 epochs, and head refits (5 epochs,
# frozen backbone) of the existing backbones for the auxiliary-loss mode. Skips finished files.
# Usage: ./run_pretrain_ext.sh "IDM FDM" "MAE CPC"     (new objectives, then objectives to refit)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aerialgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd "${PACE_EXPERIMENTS}/scripts"
LOG="${PACE_EXPERIMENTS}/logs/pretrain_v3"; mkdir -p "$LOG"
CKD=${AERIAL_GYM_ROOT}/aerial_gym/ssl_pretrain/ckpt_time_ordered_v3
ok() { [ -f "$1" ] && python -c "import torch,sys; c=torch.load('$1',map_location='cpu'); sys.exit(0 if len(c['loss_history'])==$2 else 1)" 2>/dev/null; }
for obj in $1; do for seed in 7 42 99; do
  if [ "$obj" = "MAEs" ]; then ARGS="--objective MAE --mask-unit step --mask-ratio -1 --tag s"; else ARGS="--objective $obj"; fi
  ok "$CKD/P_${obj}__seed${seed}__srcmixed__backbone.pt" 25 && { echo "skip $obj $seed"; continue; }
  python pretrain_time_ordered_v3.py $ARGS --seed $seed --epochs 25 > "$LOG/${obj}__seed${seed}.log" 2>&1
  echo "done $obj $seed exit=$? $(date +%H:%M)"
done; done
for obj in $2; do for seed in 7 42 99; do
  ok "$CKD/P_${obj}_head__seed${seed}__srcmixed__backbone.pt" 5 && { echo "skip refit $obj $seed"; continue; }
  python pretrain_time_ordered_v3.py --objective $obj --seed $seed --epochs 5 --tag _head \
    --init-backbone "$CKD/P_${obj}__seed${seed}__srcmixed__backbone.pt" --freeze-backbone > "$LOG/${obj}_head__seed${seed}.log" 2>&1
  echo "done refit $obj $seed exit=$? $(date +%H:%M)"
done; done
