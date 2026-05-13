#!/bin/bash
#SBATCH --job-name=lift_ppo_custom
#SBATCH --qos=high
#SBATCH --output=/fs/nexus-scratch/vvs22/pomdp_grasp/logs/lift_ppo_custom_%j.log
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=23:59:59
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

mkdir -p /fs/nexus-scratch/vvs22/pomdp_grasp/configs
mkdir -p /fs/nexus-scratch/vvs22/.isaac-sim-cache/data
mkdir -p /fs/nexus-scratch/vvs22/.isaac-sim-cache/cache

unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_EXE
unset CONDA_PYTHON_EXE

# Step 1: Copy the env config out of the container
apptainer exec /fs/nexus-scratch/vvs22/isaaclab.sif \
  cat /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_env_cfg.py \
  > /fs/nexus-scratch/vvs22/pomdp_grasp/configs/joint_pos_env_cfg.py

# Step 2: Reduce reaching reward weight from 1.0 to 0.1
# The rewards are defined in the parent class LiftEnvCfg, which is imported.
# We override the reward weight by injecting it into __post_init__.
sed -i 's/super().__post_init__()/super().__post_init__()\n        self.rewards.reaching_object.weight = 0.1/g' \
  /fs/nexus-scratch/vvs22/pomdp_grasp/configs/joint_pos_env_cfg.py

# Verify the change
echo "=== Reward config after edit ==="
cat /fs/nexus-scratch/vvs22/pomdp_grasp/configs/joint_pos_env_cfg.py | grep -A 2 'self.rewards.reaching_object.weight'

# Step 3: Train with the modified config bind-mounted
apptainer exec --nv \
  --env PYTHONNOUSERSITE=1 \
  --bind /fs/nexus-scratch/vvs22/.isaac-sim-cache/data:/isaac-sim/kit/data \
  --bind /fs/nexus-scratch/vvs22/.isaac-sim-cache/cache:/isaac-sim/kit/cache \
  --bind /fs/nexus-scratch/vvs22/pomdp_grasp/configs/joint_pos_env_cfg.py:/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_env_cfg.py \
  /fs/nexus-scratch/vvs22/isaaclab.sif \
  /workspace/isaaclab/isaaclab.sh -p \
  /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --num_envs 1024 \
  --headless \
  --max_iterations 6000