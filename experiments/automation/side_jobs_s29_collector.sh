#!/bin/bash
: "${AERIAL_GYM_ROOT:?set AERIAL_GYM_ROOT}"
: "${PACE_EXPERIMENTS:?set PACE_EXPERIMENTS}"
# Section 29 (V17), step 1: train the collector of the second domain, the shipped AnymalTerrain PPO with the height scan
# (domain2/yaml/collector_terrain.yaml), on the shipped rough-terrain task with its terrain curriculum.
# The run uses its own GPU context outside MPS (CUDA_MPS_PIPE_DIRECTORY points to a missing folder): a fatal error of
# It takes the launch lock of the RL workers and waits for free memory, as the other side jobs.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aerialgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps-d2
EXP="${PACE_EXPERIMENTS}"
RC=${AERIAL_GYM_ROOT}/aerial_gym/rl_training/rl_games
LOG="$EXP/logs/side_jobs_s29.out"
LOCK="$EXP/logs/study2/gpu_launch.lock"
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; do sleep 30; done; }
if ls $RC/runs/d2_collector_terrain/nn/*ep_1500* >/dev/null 2>&1; then echo "collector exists" >> "$LOG"; exit 0; fi
exec 9>"$LOCK"; flock 9
waitmem 6500
echo "collector training start $(date +%H:%M)" >> "$LOG"
(cd $RC && python runner.py --task=ige_terrain_collector --num_envs=4096 --headless=True --use_warp=False \
   --file="$EXP/domain2/yaml/collector_terrain.yaml" --train --experiment_name=d2_collector_terrain \
   > "$EXP/domain2/collector_terrain_train.log" 2>&1) &
pid=$!
t=0; until grep -q "epoch: 2/" "$EXP/domain2/collector_terrain_train.log" 2>/dev/null || ! kill -0 $pid 2>/dev/null || [ $t -ge 900 ]; do sleep 10; t=$((t+10)); done
flock -u 9; exec 9>&-
wait $pid
echo "collector training exit $? $(date +%H:%M)" >> "$LOG"
