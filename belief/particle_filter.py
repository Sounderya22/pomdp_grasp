# belief/particle_filter.py
import numpy as np


class ParticleFilter:
    """
    Maintains a belief over cube (x, y) position as a set of weighted particles.
    
    Cube spawn range observed from env:
        x in [0.40, 0.55]
        y in [-0.25, 0.25]
    """

    X_RANGE = (0.35, 0.60)
    Y_RANGE = (-0.30, 0.30)

    def __init__(self, n_particles: int = 300, noise_std: float = 0.05):
        """
        Args:
            n_particles: number of particles
            noise_std:   observation noise std — must match NoisyLiftEnv
        """
        self.n = n_particles
        self.noise_std = noise_std
        self.particles = None   # shape (n, 2)
        self.weights = None     # shape (n,)

    def reset(self):
        """Initialize particles uniformly over the cube spawn region."""
        self.particles = np.column_stack([
            np.random.uniform(*self.X_RANGE, self.n),
            np.random.uniform(*self.Y_RANGE, self.n),
        ])
        self.weights = np.ones(self.n) / self.n

    def update(self, noisy_xy: np.ndarray):
        """
        Reweight particles using Gaussian likelihood of the observation.
        Then resample using low-variance resampling.

        Args:
            noisy_xy: observed (x, y) with noise, shape (2,)
        """
        # Reweight
        diff = self.particles - noisy_xy          # (n, 2)
        sq_dist = (diff ** 2).sum(axis=1)         # (n,)
        log_w = -0.5 * sq_dist / (self.noise_std ** 2)
        log_w -= log_w.max()                       # numerical stability
        self.weights = np.exp(log_w)
        self.weights /= self.weights.sum()

        # Low-variance resample
        self.particles = self._resample()
        self.weights = np.ones(self.n) / self.n

    def mean(self) -> np.ndarray:
        """Returns weighted mean of particles as best (x, y) estimate."""
        return (self.weights[:, None] * self.particles).sum(axis=0)

    def entropy(self) -> float:
        """
        Proxy for belief uncertainty: log of area of 2-std ellipse.
        Lower = more certain.
        """
        std = self.particles.std(axis=0)          # (2,)
        return float(np.log(std[0] * std[1] + 1e-8))

    def _resample(self) -> np.ndarray:
        """Low-variance (systematic) resampling."""
        positions = (np.arange(self.n) + np.random.uniform()) / self.n
        cumsum = np.cumsum(self.weights)
        indices = np.searchsorted(cumsum, positions)
        return self.particles[indices]