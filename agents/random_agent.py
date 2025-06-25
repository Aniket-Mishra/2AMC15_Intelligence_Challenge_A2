"""Random Agent.

This is an agent that takes a random action from the available action space.
"""

import numpy as np

from agents import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that performs a random action every time. """
    def update(self, state: tuple[float, float, float], reward: float, action):
        pass

    def take_action(self, state: tuple[float, float]) -> int:
        # Move forward with 70% chance, rotate left with 15% chance, or rotate right with 15% chance
        return np.random.choice([0,1,2], p=[0.7, 0.15, 0.15])