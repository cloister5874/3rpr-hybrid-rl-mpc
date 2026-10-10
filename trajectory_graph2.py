import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\madha\OneDrive\Desktop\project_1"
)

MODEL_PATH = (
    PROJECT_DIR
    / "training"
    / "final_model"
    / "sac_expert_sac_final.pt"
)

sys.path.insert(0, str(PROJECT_DIR))

from dynamics.dynamics import step


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

STATE_LOW = np.array(
    [0.0, 0.0, -0.5, -0.2, -0.2, -0.2],
    dtype=np.float32
)

STATE_HIGH = np.array(
    [1.0, 1.0, 0.5, 0.2, 0.2, 0.2],
    dtype=np.float32
)

MAX_ACTION = 2.0
DT = 0.01
STEPS = 300


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(x):

    low = torch.tensor(
        STATE_LOW,
        dtype=torch.float32,
        device=x.device
    )

    high = torch.tensor(
        STATE_HIGH,
        dtype=torch.float32,
        device=x.device
    )

    return 2.0 * (x - low) / (high - low) - 1.0


# ============================================================
# ACTOR
# ============================================================

class Actor(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(12, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.mean = nn.Linear(256, 3)
        self.log_std = nn.Linear(256, 3)

    def deterministic(self, state, desired):

        state = normalize(state)
        desired = normalize(desired)

        x = torch.cat(
            [state, desired],
            dim=1
        )

        h = self.net(x)

        return (
            MAX_ACTION
            * torch.tanh(self.mean(h))
        )


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

actor = Actor().to(DEVICE)

actor.load_state_dict(
    checkpoint["actor"]
)

actor.eval()

print("Model loaded successfully.")


# ============================================================
# SAME REPRESENTATIVE UNSEEN TASK
# SEED 2027
# ============================================================

rng = np.random.default_rng(2027)

state = np.array(
    [
        rng.uniform(0.4, 0.6),
        rng.uniform(0.4, 0.6),
        rng.uniform(-0.1, 0.1),
        rng.uniform(-0.1, 0.1),
        rng.uniform(-0.1, 0.1),
        rng.uniform(-0.1, 0.1)
    ],
    dtype=np.float64
)

desired = np.array(
    [
        rng.uniform(0.4, 0.6),
        rng.uniform(0.4, 0.6),
        rng.uniform(-0.1, 0.1),
        0.0,
        0.0,
        0.0
    ],
    dtype=np.float64
)


# ============================================================
# STORE TRAJECTORY
# ============================================================

trajectory = [state.copy()]


# ============================================================
# SIMULATION
# ============================================================

for t in range(STEPS):

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    desired_tensor = torch.tensor(
        desired,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    with torch.no_grad():

        action = actor.deterministic(
            state_tensor,
            desired_tensor
        ).cpu().numpy()[0]

    action = np.clip(
        action,
        -MAX_ACTION,
        MAX_ACTION
    )

    next_state = np.asarray(
        step(
            state,
            action,
            dt=DT
        ),
        dtype=np.float64
    )

    if (
        not np.all(np.isfinite(next_state))
        or np.any(np.abs(next_state) > 5.0)
    ):
        print("Simulation became unstable.")
        break

    state = next_state

    trajectory.append(
        state.copy()
    )


# ============================================================
# CONVERT TO ARRAY
# ============================================================

trajectory = np.asarray(
    trajectory
)


# ============================================================
# GRAPH
# ============================================================

plt.figure(figsize=(9, 7))

plt.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    linewidth=2,
    label="SAC trajectory"
)

plt.scatter(
    trajectory[0, 0],
    trajectory[0, 1],
    s=100,
    label="Initial position"
)

plt.scatter(
    desired[0],
    desired[1],
    s=100,
    marker="x",
    label="Desired position"
)

plt.scatter(
    trajectory[-1, 0],
    trajectory[-1, 1],
    s=100,
    label="Final position"
)

plt.xlabel("X position")

plt.ylabel("Y position")

plt.title(
    "3-RPR End-Effector Trajectory on an Unseen Task"
)

plt.grid(
    True,
    alpha=0.3
)

plt.axis("equal")

plt.legend()

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

OUTPUT = (
    PROJECT_DIR
    / "training"
    / "rpr_xy_trajectory_unseen.png"
)

plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print()
print("=" * 60)
print("TRAJECTORY GRAPH GENERATED")
print("=" * 60)

print("Initial position:")
print(
    trajectory[0, :2]
)

print("\nDesired position:")
print(
    desired[:2]
)

print("\nFinal position:")
print(
    trajectory[-1, :2]
)

print("\nSaved to:")
print(OUTPUT)