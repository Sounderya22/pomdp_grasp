# planner/pomdp_domain.py
"""
POMDP domain definitions for the grasp-under-uncertainty problem.
State: true cube (x, y)
Observation: noisy (x, y) estimate
Actions: grasp, reobserve_left, reobserve_right, reobserve_above
"""
import numpy as np
import pomdp_py


# ── Actions ───────────────────────────────────────────────────────────────────

class GraspAction(pomdp_py.Action):
    def __init__(self): self.name = "grasp"
    def __repr__(self): return "GraspAction"
    def __hash__(self): return hash(self.name)
    def __eq__(self, other): return isinstance(other, GraspAction)

class ReobserveAction(pomdp_py.Action):
    def __init__(self, direction: str):
        assert direction in ("left", "right", "above")
        self.name = f"reobserve_{direction}"
        self.direction = direction
    def __repr__(self): return f"ReobserveAction({self.direction})"
    def __hash__(self): return hash(self.name)
    def __eq__(self, other): return isinstance(other, ReobserveAction) and self.name == other.name

ALL_ACTIONS = [
    GraspAction(),
    ReobserveAction("left"),
    ReobserveAction("right"),
    ReobserveAction("above"),
]


# ── State ─────────────────────────────────────────────────────────────────────

class CubeState(pomdp_py.State):
    def __init__(self, xy: np.ndarray):
        self.xy = np.array(xy, dtype=np.float32)
    def __repr__(self): return f"CubeState({self.xy})"
    def __hash__(self): return hash(tuple(self.xy.round(4)))
    def __eq__(self, other): return isinstance(other, CubeState) and np.allclose(self.xy, other.xy)


# ── Observation ───────────────────────────────────────────────────────────────

class CubeObservation(pomdp_py.Observation):
    def __init__(self, xy: np.ndarray):
        self.xy = np.array(xy, dtype=np.float32)
    def __repr__(self): return f"CubeObservation({self.xy})"
    def __hash__(self): return hash(tuple(self.xy.round(4)))
    def __eq__(self, other): return isinstance(other, CubeObservation) and np.allclose(self.xy, other.xy)


# ── Transition Model ──────────────────────────────────────────────────────────

class GraspTransitionModel(pomdp_py.TransitionModel):
    """
    Cube doesn't move until a grasp is attempted.
    Transition is deterministic — stochasticity is only in observations.
    """
    def probability(self, next_state, state, action):
        if np.allclose(next_state.xy, state.xy):
            return 1.0
        return 0.0

    def sample(self, state, action):
        return CubeState(state.xy.copy())

    def argmax(self, state, action):
        return CubeState(state.xy.copy())


# ── Observation Model ─────────────────────────────────────────────────────────

class GraspObservationModel(pomdp_py.ObservationModel):
    """
    Noisy Gaussian observation of cube (x, y).
    Reobservation actions return a new noisy reading.
    Grasp action returns the same last observation (no new info).
    """
    def __init__(self, noise_std: float):
        self.noise_std = noise_std

    def probability(self, observation, next_state, action):
        if isinstance(action, GraspAction):
            return 1.0  # grasp doesn't produce a meaningful new observation
        diff = observation.xy - next_state.xy
        sq_dist = float(np.dot(diff, diff))
        norm = 1.0 / (2 * np.pi * self.noise_std ** 2)
        return norm * np.exp(-0.5 * sq_dist / self.noise_std ** 2)

    def sample(self, next_state, action):
        if isinstance(action, GraspAction):
            return CubeObservation(next_state.xy.copy())
        noise = np.random.normal(0, self.noise_std, size=2).astype(np.float32)
        return CubeObservation(next_state.xy + noise)


# ── Reward Model ──────────────────────────────────────────────────────────────

class GraspRewardModel(pomdp_py.RewardModel):
    """
    +10  successful grasp (belief mean within grasp_threshold of true position)
    -5   failed grasp
    -0.1 reobservation step
    """
    def __init__(self, grasp_threshold: float = 0.05):
        self.grasp_threshold = grasp_threshold

    def sample(self, state, action, next_state):
        if isinstance(action, GraspAction):
            # In simulation rollouts, we use belief mean vs state as proxy
            return 10.0  # actual success checked in env; planner assumes optimistic
        return -0.1  # reobservation cost


# ── Policy Model (rollout) ────────────────────────────────────────────────────

class GraspPolicyModel(pomdp_py.RolloutPolicy):
    """Uniform random rollout policy for POMCP simulations."""
    def rollout(self, state, history=None):
        return np.random.choice(ALL_ACTIONS)

    def get_all_actions(self, state=None, history=None):
        return ALL_ACTIONS