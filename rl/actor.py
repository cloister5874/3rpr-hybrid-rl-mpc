import torch
import torch.nn as nn


class Actor(nn.Module):

    def __init__(self, state_dim=6, action_dim=3, max_action=0.2):
        super().__init__()

        self.max_action = max_action

        self.network = nn.Sequential(
            nn.Linear(state_dim * 2, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, state, desired_state):

        x = torch.cat([state, desired_state], dim=1)

        action = self.network(x)

        # Scale Actor output from [-1, 1] to [-0.2, 0.2]
        action = self.max_action * action

        return action