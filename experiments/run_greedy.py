# experiments/run_greedy.py
"""
Validates greedy baseline position error across noise levels.
Run with:
    ./isaaclab.sh -p ~/dev/pomdp_grasp/experiments/run_greedy.py --headless
"""
import argparse
import sys
import numpy as np
sys.path.insert(0, "/fs/nexus-scratch/vvs22/pomdp_grasp")

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
from envs.lift_env import NoisyLiftEnv
from baselines.greedy import GreedyAgent

N_EPISODES = 50
NOISE_LEVELS = {"low": 0.01, "med": 0.05, "high": 0.10}

env_cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-v0", device="cuda:0", num_envs=1)
raw_env = ManagerBasedRLEnv(cfg=env_cfg)

with open("/fs/nexus-scratch/vvs22/pomdp_grasp/results/greedy_results.txt", "w") as f:
    f.write("=== Greedy Baseline: Position Error Validation ===\n\n")
    for label, std in NOISE_LEVELS.items():
        env = NoisyLiftEnv(raw_env, noise_std=std)
        agent = GreedyAgent(env)
    
        errors = []
        for _ in range(N_EPISODES):
            result = agent.run_episode()
            errors.append(result["error_m"])
    
        errors = np.array(errors)
        f.write(f"  Noise={label} (σ={std}m) over {N_EPISODES} episodes:\n")
        f.write(f"    Mean position error : {errors.mean():.4f} m\n")
        f.write(f"    Std  position error : {errors.std():.4f} m\n")
        f.write(f"    Max  position error : {errors.max():.4f} m\n")
        f.write(f"    % episodes > 0.05m  : {(errors > 0.05).mean()*100:.1f}%\n\n")

raw_env.close()
simulation_app.close()