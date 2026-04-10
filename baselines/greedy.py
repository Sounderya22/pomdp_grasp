# baselines/greedy.py
import torch


class GreedyAgent:
    """
    Minimum working example baseline.
    Takes a single noisy observation of cube (x, y) at episode start
    and immediately commands a grasp at that position.
    No belief update, no replanning.
    """

    def __init__(self, env):
        self.env = env

    def run_episode(self):
        """
        Returns:
            success (bool): whether the grasp lifted the cube
            noisy_xy (tensor): what the agent thought the cube position was
            true_xy (tensor): actual cube position at episode start
        """
        noisy_xy, true_xy, _ = self.env.reset()

        # Agent's only knowledge: the single noisy observation
        grasp_target = noisy_xy
        error = (grasp_target - true_xy).norm().item()

        return {
            "grasp_target": grasp_target.numpy(),
            "true_xy":      true_xy.numpy(),
            "error_m":      error,
        }