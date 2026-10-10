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

SEEDS = [2026, 2027, 2028, 2029, 2030]
TASKS_PER_RUN = 100


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
# COLLECT FINAL ERRORS
# ============================================================

all_final_errors = []
all_initial_errors = []

successes = 0
unstable = 0

print("\nEvaluating 500 unseen tasks...")


for seed in SEEDS:

    rng = np.random.default_rng(seed)

    for task in range(TASKS_PER_RUN):

        # ----------------------------------------------------
        # INITIAL STATE
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # DESIRED STATE
        # ----------------------------------------------------

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


        initial_error = np.linalg.norm(
            state[:3] - desired[:3]
        )

        failed = False


        # ----------------------------------------------------
        # SIMULATION
        # ----------------------------------------------------

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

            try:

                next_state = np.asarray(
                    step(
                        state,
                        action,
                        dt=DT
                    ),
                    dtype=np.float64
                )

            except Exception:

                failed = True
                unstable += 1
                break

            if (
                not np.all(np.isfinite(next_state))
                or np.any(np.abs(next_state) > 5.0)
            ):

                failed = True
                unstable += 1
                break

            state = next_state


        # ----------------------------------------------------
        # FINAL ERROR
        # ----------------------------------------------------

        if failed:

            continue

        final_error = np.linalg.norm(
            state[:3] - desired[:3]
        )

        all_initial_errors.append(
            initial_error
        )

        all_final_errors.append(
            final_error
        )

        if final_error < initial_error:

            successes += 1


# ============================================================
# NUMPY ARRAYS
# ============================================================

initial_errors = np.asarray(
    all_initial_errors
)

final_errors = np.asarray(
    all_final_errors
)


# ============================================================
# STATISTICS
# ============================================================

print()
print("=" * 65)
print("CLEAN 500-TASK ERROR STATISTICS")
print("=" * 65)

print(
    f"Stable tasks          : {len(final_errors)}/500"
)

print(
    f"Unstable tasks        : {unstable}/500"
)

print(
    f"Successes             : {successes}/500"
)

print(
    f"Success rate          : "
    f"{successes / 500 * 100:.2f}%"
)

print(
    f"Average initial error : "
    f"{np.mean(initial_errors):.6f}"
)

print(
    f"Average final error   : "
    f"{np.mean(final_errors):.6f}"
)

print(
    f"Median final error    : "
    f"{np.median(final_errors):.6f}"
)

print(
    f"Minimum final error   : "
    f"{np.min(final_errors):.6f}"
)

print(
    f"Maximum final error   : "
    f"{np.max(final_errors):.6f}"
)


# ============================================================
# HISTOGRAM
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    final_errors,
    bins=30
)

plt.xlabel("Final Pose Error")

plt.ylabel("Number of Tasks")

plt.title(
    "Distribution of Final Pose Errors Across 500 Unseen Tasks"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# SAVE GRAPH
# ============================================================

OUTPUT = (
    PROJECT_DIR
    / "training"
    / "final_pose_error_distribution_500_unseen.png"
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