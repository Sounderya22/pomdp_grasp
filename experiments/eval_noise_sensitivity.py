# experiments/eval_noise_sensitivity.py
import argparse
import sys
import os
import json
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--n_episodes", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.modules import ActorCritic

NOISE_LEVELS = [0.00, 0.01, 0.02, 0.05, 0.08, 0.10]
OBJ_XYZ      = slice(18, 21)
LIFT_Z       = 0.15
MAX_STEPS    = 500

env_cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-v0", device="cuda:0", num_envs=1)
env = ManagerBasedRLEnv(cfg=env_cfg)

obs_dim = env.observation_space["policy"].shape[-1]
act_dim = env.action_space.shape[-1]

print(f"\n[INFO] Loading checkpoint: {args_cli.checkpoint}")
ckpt = torch.load(args_cli.checkpoint, map_location="cuda:0")

actor_critic = ActorCritic(
    num_actor_obs=obs_dim,
    num_critic_obs=obs_dim,
    num_actions=act_dim,
    actor_hidden_dims=[256, 128, 64],
    critic_hidden_dims=[256, 128, 64],
    activation="elu",
    init_noise_std=1.0,
).to("cuda:0")
actor_critic.load_state_dict(ckpt["model_state_dict"])
actor_critic.eval()
print("[INFO] Policy loaded.\n")

def run_episode(noise_std):
    obs, _ = env.reset()
    for _ in range(MAX_STEPS):
        noisy_obs = obs["policy"].clone()
        if noise_std > 0:
            noisy_obs[0, OBJ_XYZ] += torch.randn(3, device="cuda:0") * noise_std
        with torch.no_grad():
            action = actor_critic.act_inference(noisy_obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if obs["policy"][0, 20].item() > LIFT_Z:
            return 1
        if terminated.any() or truncated.any():
            break
    return 0

print(f"=== Noise Sensitivity ({args_cli.n_episodes} episodes/level) ===\n")
results = {}

for std in NOISE_LEVELS:
    successes = [run_episode(std) for _ in range(args_cli.n_episodes)]
    rate = np.mean(successes) * 100
    results[str(std)] = rate
    print(f"  σ={std:.2f}m  →  success rate = {rate:.1f}%")

out_dir = "/fs/nexus-scratch/vvs22/pomdp_grasp/results"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "noise_sensitivity.json")
with open(out_path, "w") as f:
    json.dump({"noise_std": NOISE_LEVELS, "success_rate": results}, f, indent=2)
print(f"\nResults saved to {out_path}")

env.close()
simulation_app.close()
