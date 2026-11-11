"""Grid-world with a LEARNED, FROZEN policy (table exported from the PyTorch trainer)."""
import numpy as np
from safety_gym_env import GridSafetyEnv

ACTIONS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # stay, up, down, left, right


class LearnedPolicyEnv(GridSafetyEnv):
    def __init__(self, policy_table, **kw):
        super().__init__(**kw)
        self.policy_table = np.asarray(policy_table)  # [N*N, 5], rows sum to 1

    def base_kernel(self, cell, goal_active):
        nbrs = self.neighbors(cell)
        if not goal_active:  # Decay: neutral random walk, same as the original domain
            return nbrs, np.ones(len(nbrs)) / len(nbrs)
        probs = self.policy_table[self._cell_to_idx(cell)]
        p = []
        for (rr, cc) in nbrs:
            a = ACTIONS.index((rr - cell[0], cc - cell[1]))
            p.append(probs[a])
        p = np.array(p) + 1e-12
        return nbrs, p / p.sum()