import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, state_dim=6, action_dim=3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim * 2 + action_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, state, desired_state, action):

        x = torch.cat(
            [state, desired_state, action],
            dim=1
        )

        q_value = self.network(x)

        return q_value