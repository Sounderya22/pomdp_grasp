# planner/pomcp_agent.py
"""
POMCP agent: wraps pomdp_py's POUCT solver with the grasp domain.
Uses the particle filter belief as the internal simulation model.
"""
import numpy as np
import pomdp_py

from belief.particle_filter import ParticleFilter
from planner.pomdp_domain import (
    CubeState, CubeObservation,
    GraspAction, ReobserveAction,
    GraspTransitionModel, GraspObservationModel,
    GraspRewardModel, GraspPolicyModel,
    ALL_ACTIONS,
)


class POMCPAgent:
    def __init__(
        self,
        noise_std: float = 0.05,
        n_particles: int = 300,
        planning_time: float = 0.5,   # seconds per planning step
        discount: float = 0.95,
        grasp_threshold: float = 0.05,
    ):
        self.noise_std = noise_std
        self.grasp_threshold = grasp_threshold
        self.pf = ParticleFilter(n_particles=n_particles, noise_std=noise_std)

        # pomdp_py agent components
        self.transition_model  = GraspTransitionModel()
        self.observation_model = GraspObservationModel(noise_std)
        self.reward_model      = GraspRewardModel(grasp_threshold)
        self.policy_model      = GraspPolicyModel()

        self.planning_time = planning_time
        self.discount = discount
        self._solver = None

    def reset(self, initial_obs: np.ndarray):
        """Call at the start of each episode with first noisy observation."""
        self.pf.reset()
        self.pf.update(initial_obs)
        self._build_solver()

    def act(self) -> str:
        """Run POMCP and return the best action name."""
        action = self._solver.plan(self._make_agent())
        return action

    def update(self, obs: np.ndarray):
        """Update belief with a new noisy observation."""
        self.pf.update(obs)

    def belief_mean(self) -> np.ndarray:
        return self.pf.mean()

    def entropy(self) -> float:
        return self.pf.entropy()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _make_belief(self):
        """Convert particle filter into pomdp_py Particles belief."""
        particles = [CubeState(xy) for xy in self.pf.particles]
        return pomdp_py.Particles(particles)

    def _make_agent(self):
        return pomdp_py.Agent(
            self._make_belief(),
            self.policy_model,
            self.transition_model,
            self.observation_model,
            self.reward_model,
        )

    def _build_solver(self):
        self._solver = pomdp_py.POUCT(
            max_depth=6,           # was 5
            discount_factor=self.discount,
            planning_time=1.0,     # was 0.5 — more time to explore
            exploration_const=20.0, # was 5.0 — force more exploration
            rollout_policy=self.policy_model,
            num_visits_init=1,
        )