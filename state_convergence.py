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
# REPRESENTATIVE UNSEEN TASK
# SAME TASK USED FOR PREVIOUS GRAPHS
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
# STORE STATES
# ============================================================

states = [state.copy()]


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

    states.append(
        state.copy()
    )


# ============================================================
# CONVERT
# ============================================================

states = np.asarray(states)

time = np.arange(
    len(states)
) * DT


# ============================================================
# X CONVERGENCE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    time,
    states[:, 0],
    linewidth=2,
    label="Actual x"
)

plt.axhline(
    desired[0],
    linestyle="--",
    linewidth=2,
    label="Desired x"
)

plt.xlabel("Time (s)")
plt.ylabel("X Position")

plt.title(
    "X Position Convergence on an Unseen Task"
)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

OUTPUT_X = (
    PROJECT_DIR
    / "training"
    / "x_convergence_unseen.png"
)

plt.savefig(
    OUTPUT_X,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Y CONVERGENCE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    time,
    states[:, 1],
    linewidth=2,
    label="Actual y"
)

plt.axhline(
    desired[1],
    linestyle="--",
    linewidth=2,
    label="Desired y"
)

plt.xlabel("Time (s)")
plt.ylabel("Y Position")

plt.title(
    "Y Position Convergence on an Unseen Task"
)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

OUTPUT_Y = (
    PROJECT_DIR
    / "training"
    / "y_convergence_unseen.png"
)

plt.savefig(
    OUTPUT_Y,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# THETA CONVERGENCE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    time,
    states[:, 2],
    linewidth=2,
    label="Actual θ"
)

plt.axhline(
    desired[2],
    linestyle="--",
    linewidth=2,
    label="Desired θ"
)

plt.xlabel("Time (s)")
plt.ylabel("Orientation θ")

plt.title(
    "Orientation Convergence on an Unseen Task"
)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

OUTPUT_THETA = (
    PROJECT_DIR
    / "training"
    / "theta_convergence_unseen.png"
)

plt.savefig(
    OUTPUT_THETA,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FINAL VALUES
# ============================================================

print()
print("=" * 60)
print("STATE CONVERGENCE RESULTS")
print("=" * 60)

print(f"Initial x : {states[0,0]:.6f}")
print(f"Final x   : {states[-1,0]:.6f}")
print(f"Desired x : {desired[0]:.6f}")

print()

print(f"Initial y : {states[0,1]:.6f}")
print(f"Final y   : {states[-1,1]:.6f}")
print(f"Desired y : {desired[1]:.6f}")

print()

print(f"Initial θ : {states[0,2]:.6f}")
print(f"Final θ   : {states[-1,2]:.6f}")
print(f"Desired θ : {desired[2]:.6f}")

print()
print("Graphs saved:")
print(OUTPUT_X)
print(OUTPUT_Y)
print(OUTPUT_THETA)