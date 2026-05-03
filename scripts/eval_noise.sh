#!/bin/bash
#SBATCH --job-name=eval_noise
#SBATCH --qos=high
#SBATCH --output=/fs/nexus-scratch/vvs22/pomdp_grasp/logs/eval_noise_%j.log
#SBATCH --gres=gpu:rtxa6000:1
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

# UPDATE THIS to your new checkpoint
CHECKPOINT=/fs/nexus-scratch/vvs22/pomdp_grasp/logs/rsl_rl/franka_lift/<new_timestamp>/model_<last>.pt

apptainer exec --nv \
  --env PYTHONNOUSERSITE=1 \
  --bind /fs/nexus-scratch/vvs22/.isaac-sim-cache/data:/isaac-sim/kit/data \
  --bind /fs/nexus-scratch/vvs22/.isaac-sim-cache/cache:/isaac-sim/kit/cache \
  --bind /fs/nexus-scratch/vvs22/pomdp_grasp/configs/joint_pos_env_cfg.py:/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_env_cfg.py \
  /fs/nexus-scratch/vvs22/isaaclab.sif \
  /workspace/isaaclab/isaaclab.sh -p \
  /fs/nexus-scratch/vvs22/pomdp_grasp/experiments/eval_noise_sensitivity.py \
  --headless \
  --checkpoint $CHECKPOINT \
  --n_episodes 100
