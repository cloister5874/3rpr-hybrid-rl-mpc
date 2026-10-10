import random
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, desired_state, action, reward, next_state):
        experience = (
            state,
            desired_state,
            action,
            reward,
            next_state
        )

        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)