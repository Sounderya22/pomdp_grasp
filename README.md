# pomdp_grasp

Belief-space planning for robot grasping under pose uncertainty.

<!-- A course project for **ENAE 788Z: Decision Making under Uncertainty** at the University of Maryland. -->

## Problem

Depth cameras introduce noise in estimating object position on a table. A robot acting greedily on a noisy estimate fails when the error is large, but taking extra observations costs time. This project formalizes the decision — **grasp now or reobserve first?** — as a POMDP and solves it with POMCP.

Simulation is done in [Isaac Lab](https://github.com/isaac-sim/IsaacLab) using the `Isaac-Lift-Cube-Franka-v0` environment.

## Approach

| Method | Description |
|---|---|
| **Greedy baseline** | Single observation at episode start, immediate grasp at noisy estimate |
| **POMCP + particle filter** | Maintains belief over cube (x, y), plans over future observations before committing to grasp |

Evaluated across three noise levels (σ = 0.01, 0.05, 0.10 m) over 100 episodes each.

## Repo Structure

```
pomdp_grasp/
├── envs/
│   ├── lift_env.py          # NoisyLiftEnv: wraps Isaac Lab, injects Gaussian noise
│   └── noise_models.py      # Noise model utilities
├── belief/
│   └── particle_filter.py   # Particle filter: init, reweight, resample, entropy
├── planner/
│   ├── pomdp_domain.py      # pomdp_py domain definitions (State, Action, Models)
│   └── pomcp_agent.py       # POMCP solver wrapper
├── baselines/
│   └── greedy.py            # Greedy agent: no belief, immediate grasp
├── experiments/
│   ├── run_greedy.py        # Evaluate greedy baseline
│   ├── run_pomcp.py         # Evaluate POMCP agent
│   └── configs/             # Noise level configs (low/med/high)
├── eval/
│   ├── metrics.py           # Success rate, reobservation count, plotting
│   └── logger.py            # Episode logger
├── tests/
│   ├── test_noise_model.py  # Validates noise wrapper mean/variance
│   ├── test_particle_filter.py
│   └── test_greedy.py
└── results/                 # Auto-generated outputs (gitignored)
```

## Prerequisites

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) installed locally with the `env_isaaclab` conda environment
- Isaac Sim 5.1
- NVIDIA GPU (tested on RTX 4070 Laptop)

```bash
pip install pomdp-py numpy scipy matplotlib pyyaml
```

## Getting Started

All scripts must be launched through Isaac Lab's Python wrapper so Isaac Sim modules load correctly.

```bash
cd ~/dev/IsaacLab
```

**1. Verify the environment runs:**
```bash
./isaaclab.sh -p scripts/environments/random_agent.py \
  --task Isaac-Lift-Cube-Franka-v0 --num_envs 1
```

**2. Run the noise model test:**
```bash
./isaaclab.sh -p ~/dev/pomdp_grasp/tests/test_noise_model.py --headless
```

**3. Run the greedy baseline:**
```bash
./isaaclab.sh -p ~/dev/pomdp_grasp/experiments/run_greedy.py --headless
```

**4. Run the POMCP agent:**
```bash
./isaaclab.sh -p ~/dev/pomdp_grasp/experiments/run_pomcp.py --headless
```

## Key Design Decisions

**Observation indices confirmed from env inspection:**
```
obs["policy"][0, 18:20]  →  cube (x, y) position  ← noise injected here
obs["policy"][0, 20]     →  cube z position
obs["policy"][0, 0:9]    →  joint positions
obs["policy"][0, 9:17]   →  joint velocities
```

**NoisyLiftEnv returns both `noisy_xy` and `true_xy`** on every step so the particle filter can be evaluated against ground truth without a separate oracle call.

**The particle filter is decoupled from Isaac Lab** — it operates purely on (x, y) floats and has no Isaac Lab dependency, making it independently unit-testable.

## Results

*To be populated after experiments.*

| Method | σ=0.01 | σ=0.05 | σ=0.10 |
|---|---|---|---|
| Greedy baseline | - | - | - |
| POMCP + particle filter | - | - | - |

## References

- [POMCP — Silver & Veness, 2010](https://papers.nips.cc/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html)
- [pomdp_py library](https://h2r.github.io/pomdp-py/)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
<!-- - Algorithms for Decision Making — Kochenderfer, Wheeler, Wray (course textbook) -->