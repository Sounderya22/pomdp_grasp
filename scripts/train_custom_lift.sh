#!/bin/bash
#SBATCH --job-name=lift_ppo_custom
#SBATCH --qos=high
#SBATCH --output=/fs/nexus-scratch/vvs22/pomdp_grasp/logs/lift_ppo_custom_%j.log
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --time=23:59:59
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_EXE
unset CONDA_PYTHON_EXE

apptainer exec --nv --pwd /workspace/isaaclab \
  --env PYTHONNOUSERSITE=1 \
  --bind /fs/nexus-scratch/vvs22/.isaac-sim-cache/data:/isaac-sim/kit/data \
  --bind /fs/nexus-scratch/vvs22/.isaac-sim-cache/cache:/isaac-sim/kit/cache \
  --bind /fs/nexus-scratch/vvs22/pomdp_grasp:/pomdp_grasp \
  --bind /fs/nexus-scratch/vvs22/pomdp_grasp/logs/rsl_rl:/workspace/isaaclab/logs/rsl_rl \
  --bind /fs/nexus-scratch/vvs22/pomdp_grasp/outputs:/workspace/isaaclab/outputs \
  /fs/nexus-scratch/vvs22/isaaclab.sif \
  /workspace/isaaclab/isaaclab.sh -p \
  /pomdp_grasp/scripts/train_custom.py \
  --task Custom-Isaac-Lift-Cube-Franka-v0 \
  --num_envs 1024 \
  --headless \
  --max_iterations 15000
