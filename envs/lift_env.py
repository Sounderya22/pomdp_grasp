# envs/lift_env.py
import torch
import numpy as np

# Observation vector indices
OBJ_XY_IDX = slice(18, 20)  # cube (x, y) in world frame
OBJ_Z_IDX  = 20             # cube z (not corrupted, used for grasp height)


class NoisyLiftEnv:
    """
    Wraps Isaac-Lift-Cube-Franka-v0.
    Exposes the cube's (x, y) position with additive Gaussian noise,
    simulating RealSense-style depth uncertainty.
    """

    def __init__(self, env, noise_std: float = 0.05):
        """
        Args:
            env:        A live ManagerBasedRLEnv instance
            noise_std:  Std dev of Gaussian noise added to cube (x, y) in metres.
        """
        self.env = env
        self.noise_std = noise_std
        self.device = env.device

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self):
        obs, info = self.env.reset()
        true_xy = self._extract_xy(obs)
        noisy_xy = self._corrupt(true_xy)
        return noisy_xy, true_xy, info

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, info = self.env.step(action)
        true_xy = self._extract_xy(obs)
        noisy_xy = self._corrupt(true_xy)
        done = terminated | truncated
        return noisy_xy, true_xy, reward, done, info

    def close(self):
        self.env.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_xy(self, obs: dict) -> torch.Tensor:
        """Returns true cube (x, y) as shape (2,) tensor on CPU."""
        return obs["policy"][0, OBJ_XY_IDX].cpu()

    def _corrupt(self, true_xy: torch.Tensor) -> torch.Tensor:
        """Adds zero-mean Gaussian noise to cube (x, y)."""
        noise = torch.tensor(
            np.random.normal(0, self.noise_std, size=2),
            dtype=torch.float32
        )
        return true_xy + noise
