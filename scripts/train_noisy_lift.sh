#!/bin/bash
#SBATCH --job-name=train_noisy_lift
#SBATCH --qos=high
#SBATCH --output=/fs/nexus-scratch/vvs22/pomdp_grasp/logs/train_noisy_lift_%j.log
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=23:59:59
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

mkdir -p /fs/nexus-scratch/vvs22/pomdp_grasp/logs
mkdir -p /fs/nexus-scratch/vvs22/.isaac-sim-cache/data
mkdir -p /fs/nexus-scratch/vvs22/.isaac-sim-cache/cache

unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_EXE
unset CONDA_PYTHON_EXE

export TERM=xterm

# Change to project root so train.py finds the correct logs/ folder
cd /fs/nexus-scratch/vvs22/pomdp_grasp

apptainer exec --nv \
  --env PYTHONNOUSERSITE=1 \
  --bind /fs/nexus-scratch/vvs22/.isaac-sim-cache/data:/isaac-sim/kit/data \
  --bind /fs/nexus-scratch/vvs22/.isaac-sim-cache/cache:/isaac-sim/kit/cache \
  --bind /fs/nexus-scratch/vvs22/pomdp_grasp/configs/noisy_joint_pos_env_cfg.py:/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_env_cfg.py \
  /fs/nexus-scratch/vvs22/isaaclab.sif \
  /workspace/isaaclab/isaaclab.sh -p \
  /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --num_envs 1024 \
  --headless \
  --resume \
  --load_run 2026-04-28_20-26-04 \
  --max_iterations 22500 \
  --run_name lift_ppo_custom_noisy
