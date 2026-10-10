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
# 12 -> 256 -> 256 -> 256 -> 3
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
# REPRESENTATIVE UNSEEN TASK
# Seed 2027
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
# SIMULATION
# ============================================================

states = [state.copy()]
errors = []

initial_error = np.linalg.norm(
    state[:3] - desired[:3]
)

errors.append(initial_error)


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

    states.append(state.copy())

    error = np.linalg.norm(
        state[:3] - desired[:3]
    )

    errors.append(error)


# ============================================================
# RESULTS
# ============================================================

final_error = errors[-1]

print()
print("=" * 60)
print("REPRESENTATIVE UNSEEN TASK")
print("=" * 60)

print("Initial state:")
print(state if False else states[0])

print()
print("Desired state:")
print(desired)

print()
print(f"Initial pose error : {initial_error:.6f}")
print(f"Final pose error   : {final_error:.6f}")

print(
    f"Improvement        : "
    f"{initial_error - final_error:.6f}"
)

print(
    "Success            :",
    "YES" if final_error < initial_error else "NO"
)


# ============================================================
# GRAPH
# ============================================================

time = np.arange(len(errors)) * DT

plt.figure(figsize=(10, 6))

plt.plot(
    time,
    errors,
    linewidth=2
)

plt.xlabel("Time (s)")
plt.ylabel("Pose Error")

plt.title(
    "SAC Pose Error Convergence on an Unseen Task"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

OUTPUT = (
    PROJECT_DIR
    / "training"
    / "pose_error_convergence_unseen.png"
)

plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print()
print("Graph saved to:")
print(OUTPUT)