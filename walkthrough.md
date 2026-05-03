# Isaac Lab Lift Task: Progress & Modifications Walkthrough

This document outlines the end-to-end modifications made to the Franka Lifting Task pipeline to resolve training issues, improve motion smoothness, and ensure successful grasping behavior.

## 1. Custom Environment & Reward Design
Initially, the policy suffered from a reward plateau (e.g. `Episode_Reward/lifting_object` stabilizing near ~0.1200) where the agent learned to reach the object but wouldn't lift it. 

To create an environment we could control, we migrated from standard Isaac Lab templates to a custom setup:
- Created a standalone `custom_lift_env_cfg.py` allowing deep modification of reward and configuration structures.
- Created standalone training scripts (`train_custom_lift.sh` / `train_custom.py`) to launch training inside the Apptainer container.

## 2. Resolving Jerky Motion
Once the policy was properly learning to reach and pull at the block, the execution motion was overly jerky—a consequence of RL policies learning high-frequency "bang-bang" commands.

> [!NOTE]
> Jerky motion in continuous robotic RL tasks is almost entirely mitigated by artificially penalizing rapid shifts in joint states or chosen actions.

**Changes Made:**
We drastically increased standard penalty scales in `envs/custom_lift_env_cfg.py`.
* Increased basal penalty weights for `action_rate` and `joint_vel` from `-1e-4` to `-0.01` in the `RewardsCfg` class.
* Updated `CurriculumCfg` so the penalty progressively drops to an asymptopic weight of `-0.05` instead of a meager `-0.001` over 20,000 steps.

## 3. Fixing Gripper Inaction
While the above changes smoothed the arm motion, they introduced a new failure mode: **the gripper refused to open or close**. Because the gripper utilizes binary action dimensions (open vs close), actuating it inherently incurs high action-rate and velocity shifts. Our new aggressive penalties heavily penalized the robot for trying to grab the object.

To solve this we selectively un-penalized the gripper.

### Step 3a: Isolating Action Rate Penalty
We built a custom reward function in `envs/lift_mdp/rewards.py` to calculate action variance *only* across the first 7 action dimensions (the arm links), deliberately ignoring the 8th dimension (the gripper). 
```python
def arm_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the arm actions (first 7 dimensions) using L2-norm."""
    # Assuming the first 7 dimensions are the arm actions
    return torch.sum(torch.square(env.action_manager.action[:, :7] - env.action_manager.prev_action[:, :7]), dim=1)
```

### Step 3b: Isolating Joint Velocity Penalty 
We modified the environment configuration to use our custom `arm_action_rate_l2`, and restricted `joint_vel_l2` via regex so it only scans nodes matching `panda_joint.*`, formally sparing `panda_finger.*`.

```python
    # custom arm action rate penalty (ignores gripper dimension)
    action_rate = RewTerm(func=mdp.arm_action_rate_l2, weight=-0.01)

    # restricted joint velocity penalty (ignores panda_finger.*)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )
```

## 4. Tuning the "Goldilocks" Penalty Zone
While fixing the gripper allowed it to close, we discovered the arm would reach the object, close the gripper, and then refuse to lift. The action penalties were so high (`-0.05`) that the agent preferred taking a `0` penalty for standing perfectly still, rather than incurring significant joint velocity/action rate penalties navigating upward against gravity to receive the `15.0` lifting reward.

To mitigate this, we stepped the penalties down slightly. We decreased the base weights and the curriculum weights dynamically, achieving a balance that penalizes erratic behaviors but permits the agent to lift comfortably:
* `action_rate` base reduced to `-0.005`, and curriculum target set to `-0.01`.
* `joint_vel` base reduced to `-0.001`, and curriculum target set to `-0.005`.

## Summary
The current system now features:
1. Custom environment overrides for tuning heuristics locally.
2. A hardened curriculum guaranteeing silky-smooth translation of the arm links.
3. Unhindered gripper mobility for instantaneous latching and release logic, enabling smooth, unpenalized lift executions.
