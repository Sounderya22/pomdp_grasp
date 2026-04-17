#!/bin/bash
#SBATCH --job-name=lift_ppo
#SBATCH --output=/fs/nexus-scratch/vvs22/isaaclab_logs/lift_ppo_%j.log
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Trains PPO on Isaac-Lift-Cube-Franka-v0 with perfect state observation.
# Checkpoint saved to /fs/nexus-scratch/vvs22/isaaclab_logs/lift_ppo/
# Copy best_agent.pt to pomdp_grasp/checkpoints/ after training.

mkdir -p /fs/nexus-scratch/vvs22/isaaclab_logs

apptainer exec --nv /fs/nexus-scratch/vvs22/isaaclab.sif \
  /workspace/isaaclab/isaaclab.sh -p \
  /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --num_envs 1024 \
  --headless \
  --max_iterations 3000 \
  --log_dir /fs/nexus-scratch/vvs22/isaaclab_logs/lift_ppo