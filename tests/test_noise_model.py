# tests/test_noise_model.py
"""
Smoke test: verify NoisyLiftEnv returns noisy observations
with roughly the right mean and variance.
Run from inside IsaacLab:
    ./isaaclab.sh -p ~/dev/pomdp_grasp/tests/test_noise_model.py --headless
"""
import argparse
import torch
import numpy as np
import sys
sys.path.insert(0, "/home/vv/dev/pomdp_grasp")

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
from envs.lift_env import NoisyLiftEnv

NOISE_STD = 0.05
N_SAMPLES = 50

env_cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-v0", device="cuda:0", num_envs=1)
raw_env = ManagerBasedRLEnv(cfg=env_cfg)
env = NoisyLiftEnv(raw_env, noise_std=NOISE_STD)

errors = []
for _ in range(N_SAMPLES):
    noisy_xy, true_xy, _ = env.reset()
    errors.append((noisy_xy - true_xy).numpy())

errors = np.array(errors)
print(f"\n=== Noise Model Validation (N={N_SAMPLES}) ===")
print(f"  Mean error (x, y): {errors.mean(axis=0)}  (expected ~[0, 0])")
print(f"  Std  error (x, y): {errors.std(axis=0)}   (expected ~[{NOISE_STD}, {NOISE_STD}])")

assert np.abs(errors.mean()) < 0.02,  "ERROR: noise is biased"
assert np.abs(errors.std() - NOISE_STD) < 0.01, "ERROR: noise std is wrong"
print("\nAll checks passed.")

env.close()
simulation_app.close()
