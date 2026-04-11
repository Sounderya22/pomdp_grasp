# tests/test_pf_live.py
"""
Live test: particle filter running inside an Isaac Lab episode.
At each step the agent gets a noisy observation and updates its belief.
No action execution yet — just verify belief converges to true position
over several reobservations within a single episode.

Run with:
    ./isaaclab.sh -p ~/dev/pomdp_grasp/tests/test_pf_live.py --headless
"""
import argparse
import sys
import numpy as np
import torch
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
from belief.particle_filter import ParticleFilter

NOISE_STD = 0.05
N_OBS = 8  # reobservations per episode
N_EPISODES = 5

env_cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-v0", device="cuda:0", num_envs=1)
raw_env = ManagerBasedRLEnv(cfg=env_cfg)
env = NoisyLiftEnv(raw_env, noise_std=NOISE_STD)
pf = ParticleFilter(n_particles=300, noise_std=NOISE_STD)

print(f"\n=== Live PF Test (σ={NOISE_STD}m, {N_OBS} obs/episode) ===\n")

for ep in range(N_EPISODES):
    noisy_xy, true_xy, _ = env.reset()
    pf.reset()
    pf.update(noisy_xy.numpy())

    for i in range(N_OBS - 1):
        # Take another observation without moving (same position, new noise sample)
        # noisy_xy, true_xy, _, _, _ = env.step(
        #     raw_env.action_space.sample() * 0  # zero action = stay still
        # )
        zero_action = torch.zeros(1, 8, device="cuda:0")
        noisy_xy, true_xy, _, _, _ = env.step(zero_action)
        pf.update(noisy_xy.numpy())

    final_error = np.linalg.norm(pf.mean() - true_xy.numpy())
    print(f"  ep {ep+1}: true={true_xy.numpy().round(4)}  "
          f"belief mean={pf.mean().round(4)}  "
          f"error={final_error:.4f}m  entropy={pf.entropy():.3f}")

print("\nDone.")
raw_env.close()
simulation_app.close()