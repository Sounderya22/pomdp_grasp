#!/bin/bash
#SBATCH --job-name=eval_noisy_policy
#SBATCH --qos=high
#SBATCH --output=/fs/nexus-scratch/vvs22/pomdp_grasp/logs/eval_noisy_policy_%j.log
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

mkdir -p /fs/nexus-scratch/vvs22/pomdp_grasp/results
mkdir -p /fs/nexus-scratch/vvs22/.isaac-sim-cache/data
mkdir -p /fs/nexus-scratch/vvs22/.isaac-sim-cache/cache

unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_EXE
unset CONDA_PYTHON_EXE

export TERM=xterm

# The training script saves the model to logs/rsl_rl/lift_ppo_custom_noisy/<timestamp>/
# We dynamically find the latest model_*.pt in that folder.
CHECKPOINT_DIR=$(ls -td /fs/nexus-scratch/vvs22/pomdp_grasp/logs/rsl_rl/franka_lift/*_lift_ppo_custom_noisy | head -n 1)
CHECKPOINT=$(ls -t $CHECKPOINT_DIR/model_*.pt | head -n 1)

if [ -z "$CHECKPOINT" ]; then
    echo "No checkpoint found in $CHECKPOINT_DIR!"
    exit 1
fi
echo "Using checkpoint: $CHECKPOINT"

apptainer exec --nv \
  --env PYTHONNOUSERSITE=1 \
  --bind /fs/nexus-scratch/vvs22 \
  --bind /fs/nexus-scratch/vvs22/.isaac-sim-cache/data:/isaac-sim/kit/data \
  --bind /fs/nexus-scratch/vvs22/.isaac-sim-cache/cache:/isaac-sim/kit/cache \
  --bind /fs/nexus-scratch/vvs22/pomdp_grasp/configs/noisy_joint_pos_env_cfg.py:/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_env_cfg.py \
  /fs/nexus-scratch/vvs22/isaaclab.sif \
  /workspace/isaaclab/isaaclab.sh -p \
  /fs/nexus-scratch/vvs22/pomdp_grasp/experiments/eval_noise_sensitivity.py \
  --headless \
  --checkpoint $CHECKPOINT \
  --n_episodes 100 > /fs/nexus-scratch/vvs22/pomdp_grasp/results/noisy_policy_eval.log 2>&1

# Rename output to differentiate from the base policy
mv /fs/nexus-scratch/vvs22/pomdp_grasp/results/noise_sensitivity.json /fs/nexus-scratch/vvs22/pomdp_grasp/results/noisy_policy_noise_sensitivity.json

