# Sensitivity of Reinforcement Learning Policies to Observation Noise

A final project for the University of Maryland evaluating the robustness of a Franka robot lifting policy under observation noise, characterizing the performance gap between MDP training and POMDP deployment.

## Overview

Reinforcement learning policies for robotic manipulation are typically trained under the assumption of perfect state observation. In deployment, depth cameras introduce noise into object pose estimates, transforming the problem into a Partially Observable Markov Decision Process (POMDP). 

This project investigates how severely this transition degrades policy performance in a concrete manipulation task: lifting a cube using a Franka Panda arm simulated in Isaac Lab. We characterize the sensitivity of an MDP-trained policy to Gaussian noise and evaluate domain randomization (noise augmentation) strategies.

### Key Contributions
- **Nominal Policy Training:** Modified reward shaping to successfully train a PPO baseline under perfect observability.
- **Sensitivity Analysis:** Automated evaluation pipeline to sweep observation noise ($\sigma \in [0, 10]$ cm) and quantify the exact degradation curve.
- **Physical Limits of Domain Randomization:** Demonstrated that unbounded noise training (10cm) causes catastrophic forgetting, while fine-tuning with physically bounded noise (3cm max, restricted by the gripper aperture) successfully transfers performance while adding robustness.
- **Particle Filter Denoising:** Implemented and validated a belief-space particle filter capable of collapsing 7cm of positional error down to 0.5cm in just 10 sequential observations.

## Repository Structure

```
pomdp_grasp/
├── configs/                 # Isaac Lab environment configs (e.g. reward weights)
├── envs/                    # NoisyLiftEnv wrapper for injecting Gaussian noise
├── experiments/
│   ├── eval_noise.py        # Sweeps noise levels for the nominal policy
│   ├── eval_noisy_policy.py # Sweeps noise levels for the fine-tuned noisy policy
│   ├── run_greedy.py        # Evaluates the deterministic kinematic baseline
│   ├── run_pf.py            # Validates particle filter convergence
│   └── generate_plots.py    # Generates PNGs for the paper
├── scripts/                 # SLURM submission scripts for the UMD Nexus cluster
│   ├── train_lift.sh        # Trains nominal PPO policy
│   ├── train_noisy.sh       # Fine-tunes the 3cm bounded noisy policy
│   ├── eval_*.sh            # Scripts for running evaluations in headless mode
├── belief/
│   └── particle_filter.py   # Explicit belief maintenance 
├── baselines/
│   └── greedy.py            # Greedy open-loop kinematic agent
├── results/                 # JSON outputs from evaluation runs
└── plots/                   # Generated PNG plots (Sensitivity, PF convergence)
```

## Setup & Prerequisites

This project was developed on the UMD UMIACS Nexus cluster using an Apptainer container for Isaac Lab.

1. **Isaac Lab Container:** Ensure you have the `isaaclab.sif` image mounted.
2. **Dependencies:** `numpy` and `matplotlib` (installed locally).

**Important:** Before running Apptainer scripts, ensure conflicting host environments are disabled:
```bash
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset PYTHONHOME
unset PYTHONPATH
```

## Running the Code

All compute-intensive scripts are designed to be run via SLURM (`sbatch`).

### 1. Training Policies
To train the nominal baseline policy from scratch (takes ~1-2 hours on an RTX A6000):
```bash
cd scripts
sbatch train_lift.sh
```

To fine-tune the noise-augmented policy (3cm bound) from a nominal checkpoint:
```bash
sbatch train_noisy.sh
```

### 2. Running Evaluations
To generate the sensitivity curves (evaluating both policies from $\sigma=0$ to $0.10$ m):
```bash
sbatch eval_noise.sh         # Evaluates nominal policy
sbatch eval_noisy_policy.sh  # Evaluates fine-tuned policy
```
*Results are saved as JSON files in the `results/` directory.*

### 3. Running the Greedy Baseline
To execute the deterministic open-loop baseline and measure position error:
```bash
sbatch eval_greedy.sh
```

### 4. Generating Plots
Once the evaluation JSONs are generated, create the `.png` plots used in the paper:
```bash
apptainer exec --bind /fs/nexus-scratch/vvs22 /fs/nexus-scratch/vvs22/isaaclab.sif /workspace/isaaclab/isaaclab.sh -p experiments/generate_plots.py
```
*Plots are saved to the `plots/` directory.*

## Future Work & Unused Components
The original project scope involved fully integrating a POMCP online planner with the Isaac Lab environment. While the project pivoted to a rigorous sensitivity analysis of PPO policies, the foundational code for the POMCP planner remains in the repository for future work:
- `planner/pomdp_domain.py`: Contains the `pomdp_py` definitions for the POMDP State, Action, Observation, Transition Model, and Reward Model.
- `planner/pomcp_agent.py`: A wrapper for the POMCP solver that utilizes the particle filter (`belief/particle_filter.py`) for belief tracking.

Future work will focus on reconciling the sequential CPU-based tree search of `pomdp_py` with the highly vectorized, GPU-accelerated environment loop of Isaac Lab to deploy the closed-loop POMCP agent.

## Key Results
- **Nominal Policy:** Degrades gracefully from 100% success at 0cm noise to 84% at 10cm noise.
- **Greedy Baseline:** Fails catastrophically under noise, missing the grasp entirely in 62% of episodes at just 5cm noise.
- **3cm Fine-Tuned Policy:** Retains 97% success within its 3cm target domain, but domain specialization causes it to degrade faster than the baseline under extreme out-of-distribution noise (dropping to 64% at 10cm). 
- **Particle Filter:** Reduces belief entropy and collapses positional uncertainty by a factor of 15 (7.2cm $\rightarrow$ 0.47cm) within 10 observations (200ms penalty).