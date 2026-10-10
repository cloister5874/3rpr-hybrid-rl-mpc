import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

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

        return MAX_ACTION * torch.tanh(
            self.mean(h)
        )


# ============================================================
# LOAD FINAL MODEL
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

print("Final SAC model loaded successfully.")


# ============================================================
# COLLECT CLEAN STATISTICS
# ============================================================

initial_errors = []
final_errors = []
improvements = []
rewards = []

successes = 0
unstable = 0

run_results = []


for seed in SEEDS:

    rng = np.random.default_rng(seed)

    run_successes = 0
    run_initial = []
    run_final = []
    run_improvement = []

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

        total_reward = 0.0
        failed = False

        # ----------------------------------------------------
        # SIMULATION
        # ----------------------------------------------------

        for _ in range(STEPS):

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

            reward = (
                -np.sum(
                    (next_state - desired) ** 2
                )
                -0.01 * np.sum(action ** 2)
            )

            total_reward += reward

            state = next_state

        # ----------------------------------------------------
        # IGNORE UNSTABLE TASKS FOR ERROR STATISTICS
        # ----------------------------------------------------

        if failed:
            continue

        final_error = np.linalg.norm(
            state[:3] - desired[:3]
        )

        improvement = (
            initial_error - final_error
        )

        initial_errors.append(
            initial_error
        )

        final_errors.append(
            final_error
        )

        improvements.append(
            improvement
        )

        rewards.append(
            total_reward
        )

        run_initial.append(
            initial_error
        )

        run_final.append(
            final_error
        )

        run_improvement.append(
            improvement
        )

        if final_error < initial_error:
            successes += 1
            run_successes += 1


# ============================================================
# NUMPY ARRAYS
# ============================================================

initial_errors = np.asarray(
    initial_errors
)

final_errors = np.asarray(
    final_errors
)

improvements = np.asarray(
    improvements
)

rewards = np.asarray(
    rewards
)


# ============================================================
# FINAL CLEAN RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL CLEAN STATISTICS — 500 UNSEEN TASKS")
print("=" * 70)

print()
print("Total tasks              :", 500)
print("Stable tasks             :", len(final_errors))
print("Unstable tasks           :", unstable)
print("Successful tasks         :", successes)
print(
    "Overall success rate     : "
    f"{successes / 500 * 100:.2f}%"
)

print()
print("ERROR STATISTICS — STABLE TASKS ONLY")
print("-" * 70)

print(
    "Mean initial pose error  : "
    f"{np.mean(initial_errors):.6f}"
)

print(
    "Mean final pose error    : "
    f"{np.mean(final_errors):.6f}"
)

print(
    "Median final pose error  : "
    f"{np.median(final_errors):.6f}"
)

print(
    "Std final pose error     : "
    f"{np.std(final_errors):.6f}"
)

print(
    "Minimum final error      : "
    f"{np.min(final_errors):.6f}"
)

print(
    "Maximum final error      : "
    f"{np.max(final_errors):.6f}"
)

print(
    "Mean improvement         : "
    f"{np.mean(improvements):.6f}"
)

print(
    "Median improvement       : "
    f"{np.median(improvements):.6f}"
)

print()
print("REWARD STATISTICS — STABLE TASKS")
print("-" * 70)

print(
    "Mean episode reward      : "
    f"{np.mean(rewards):.6f}"
)

print(
    "Median episode reward    : "
    f"{np.median(rewards):.6f}"
)

print(
    "Std episode reward       : "
    f"{np.std(rewards):.6f}"
)

print()
print("=" * 70)
print("PER-RUN SUCCESS")
print("=" * 70)

# Recalculate per-run success rates cleanly
for seed in SEEDS:

    rng = np.random.default_rng(seed)

    run_success = 0
    run_stable = 0

    for task in range(TASKS_PER_RUN):

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

        initial_error = np.linalg.norm(
            state[:3] - desired[:3]
        )

        failed = False

        for _ in range(STEPS):

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
                break

            if (
                not np.all(np.isfinite(next_state))
                or np.any(np.abs(next_state) > 5.0)
            ):

                failed = True
                break

            state = next_state

        if not failed:

            run_stable += 1

            final_error = np.linalg.norm(
                state[:3] - desired[:3]
            )

            if final_error < initial_error:
                run_success += 1

    print(
        f"Seed {seed}: "
        f"{run_success}/100 "
        f"({run_success:.0f}%)"
        f" | Stable: {run_stable}/100"
    )

print()
print("Statistics complete.")