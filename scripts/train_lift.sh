#!/bin/bash
#SBATCH --job-name=lift_ppo
#SBATCH --qos=high
#SBATCH --output=/fs/nexus-scratch/vvs22/pomdp_grasp/logs/lift_ppo_%j.log
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=23:59:59
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

mkdir -p /fs/nexus-scratch/vvs22/pomdp_grasp/logs
mkdir -p /fs/nexus-scratch/vvs22/.isaac-sim-cache/data
mkdir -p /fs/nexus-scratch/vvs22/.isaac-sim-cache/cache
mkdir -p /fs/nexus-scratch/vvs22/pomdp_grasp/checkpoints/lift_ppo

unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_EXE
unset CONDA_PYTHON_EXE

apptainer exec --nv \
  --env PYTHONNOUSERSITE=1 \
  --bind /fs/nexus-scratch/vvs22/.isaac-sim-cache/data:/isaac-sim/kit/data \
  --bind /fs/nexus-scratch/vvs22/.isaac-sim-cache/cache:/isaac-sim/kit/cache \
  --bind /fs/nexus-scratch/vvs22/pomdp_grasp/checkpoints/lift_ppo:/workspace/isaaclab/logs \
  /fs/nexus-scratch/vvs22/isaaclab.sif \
  /workspace/isaaclab/isaaclab.sh -p \
  /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --num_envs 1024 \
  --headless \
  --max_iterations 3000
