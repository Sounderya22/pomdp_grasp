# tests/test_particle_filter.py
"""
Unit test: particle filter should converge to true position
after several observations at known noise level.

Run standalone (no Isaac Lab needed):
    python ~/dev/pomdp_grasp/tests/test_particle_filter.py
"""
import sys
import numpy as np
sys.path.insert(0, "/home/vv/dev/pomdp_grasp")

from belief.particle_filter import ParticleFilter

NOISE_STD = 0.05
N_OBS = 10  # observations before checking convergence
TRUE_XY = np.array([0.50, 0.10])

pf = ParticleFilter(n_particles=300, noise_std=NOISE_STD)
pf.reset()

print(f"True cube position: {TRUE_XY}")
print(f"Initial belief mean: {pf.mean().round(4)}")
print(f"Initial entropy:     {pf.entropy():.4f}\n")

for i in range(N_OBS):
    obs = TRUE_XY + np.random.normal(0, NOISE_STD, size=2)
    pf.update(obs)
    print(f"  obs {i+1:2d}: noisy={obs.round(4)}  "
          f"belief mean={pf.mean().round(4)}  "
          f"entropy={pf.entropy():.4f}")

final_error = np.linalg.norm(pf.mean() - TRUE_XY)
print(f"\nFinal position error: {final_error:.4f} m")
assert final_error < 0.05, f"PF did not converge: error={final_error:.4f}"
print("Test passed.")