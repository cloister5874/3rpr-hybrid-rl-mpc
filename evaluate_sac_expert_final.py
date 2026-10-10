import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\madha\OneDrive\Desktop\project_1"
)

CKPT = (
    PROJECT_DIR
    / "training"
    / "final_model"
    / "sac_expert_sac_final.pt"
)

sys.path.insert(0, str(PROJECT_DIR))

from dynamics.dynamics import step


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMPORTANT:
# THESE ARE THE NORMALIZATION LIMITS USED BY THE TRAINED MODEL
# DO NOT CHANGE THESE TO THE EVALUATION TASK RANGE
# ============================================================

STATE_LOW = np.array(
    [
        0.0,
        0.0,
        -0.5,
        -0.2,
        -0.2,
        -0.2
    ],
    dtype=np.float32
)

STATE_HIGH = np.array(
    [
        1.0,
        1.0,
        0.5,
        0.2,
        0.2,
        0.2
    ],
    dtype=np.float32
)

MAX_ACTION = 2.0
DT = 0.01


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

    return (
        2.0 * (x - low) / (high - low)
        - 1.0
    )


# ============================================================
# ACTOR
#
# EXACT ARCHITECTURE OF FINAL CHECKPOINT
#
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
# START
# ============================================================

print("=" * 70)
print("SAC EXPERT — FIXED EVALUATION")
print("=" * 70)

print()
print("Checkpoint:")
print(CKPT)

print()
print(
    "Device:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)


# ============================================================
# LOAD EXISTING MODEL
# ============================================================

ckpt = torch.load(
    CKPT,
    map_location=DEVICE,
    weights_only=False
)

actor = Actor().to(DEVICE)

actor.load_state_dict(
    ckpt["actor"]
)

actor.eval()


print()
print("Checkpoint loaded successfully.")
print("Algorithm:", ckpt.get("algorithm", "SAC"))
print("State dimension:", ckpt.get("state_dim", 6))
print("Desired dimension:", ckpt.get("desired_dim", 6))
print("Action dimension:", ckpt.get("action_dim", 3))
print("Maximum action:", ckpt.get("max_action", 2.0))


# ============================================================
# FIXED EVALUATION SETTINGS
# ============================================================

EVAL_SEED = 12345
EVAL_TASKS = 100
EVAL_STEPS = 300

rng = np.random.default_rng(EVAL_SEED)


# ============================================================
# RESULT STORAGE
# ============================================================

initial_errors = []
final_errors = []
improvements = []
best_errors = []
rewards_all = []

successes = 0
unstable = 0


# ============================================================
# 100 FIXED UNSEEN TASKS
#
# IMPORTANT:
# Generate each state and desired state sequentially.
# ============================================================

for task in range(EVAL_TASKS):

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # DESIRED STATE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # INITIAL POSE ERROR
    # --------------------------------------------------------

    initial_error = np.linalg.norm(
        state[:3] - desired[:3]
    )

    best_error = initial_error

    total_reward = 0.0

    failed = False


    # ========================================================
    # SIMULATION
    # ========================================================

    for t in range(EVAL_STEPS):

        s_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)


        d_tensor = torch.tensor(
            desired,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)


        # ----------------------------------------------------
        # DETERMINISTIC SAC POLICY
        # ----------------------------------------------------

        with torch.no_grad():

            action = actor.deterministic(
                s_tensor,
                d_tensor
            ).cpu().numpy()[0]


        # ----------------------------------------------------
        # ACTION LIMIT
        # ----------------------------------------------------

        action = np.clip(
            action,
            -MAX_ACTION,
            MAX_ACTION
        )


        # ----------------------------------------------------
        # ACTUAL 3-RPR DYNAMICS
        # ----------------------------------------------------

        try:

            next_state = np.asarray(
                step(
                    state,
                    action,
                    dt=DT
                ),
                dtype=np.float64
            )

        except Exception as e:

            failed = True
            unstable += 1
            break


        # ----------------------------------------------------
        # STABILITY CHECK
        # ----------------------------------------------------

        if (
            not np.all(np.isfinite(next_state))
            or np.any(np.abs(next_state) > 5.0)
        ):

            failed = True
            unstable += 1
            break


        # ----------------------------------------------------
        # EXACT PROJECT REWARD
        # ----------------------------------------------------

        reward = (
            -np.sum(
                (next_state - desired) ** 2
            )
            -0.01 * np.sum(action ** 2)
        )


        total_reward += reward

        state = next_state


        # ----------------------------------------------------
        # POSE ERROR
        # ----------------------------------------------------

        error = np.linalg.norm(
            state[:3] - desired[:3]
        )

        best_error = min(
            best_error,
            error
        )


    # ========================================================
    # EPISODE RESULT
    # ========================================================

    if failed:

        final_error = 999.0
        improvement = -999.0

    else:

        final_error = np.linalg.norm(
            state[:3] - desired[:3]
        )

        improvement = (
            initial_error - final_error
        )

        # SUCCESS DEFINITION
        if final_error < initial_error:
            successes += 1


    initial_errors.append(initial_error)
    final_errors.append(final_error)
    improvements.append(improvement)
    best_errors.append(best_error)
    rewards_all.append(total_reward)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"Average initial pose error : "
    f"{np.mean(initial_errors):.6f}"
)

print(
    f"Average final pose error   : "
    f"{np.mean(final_errors):.6f}"
)

print(
    f"Average improvement        : "
    f"{np.mean(improvements):.6f}"
)

print(
    f"Average best pose error    : "
    f"{np.mean(best_errors):.6f}"
)

print(
    f"Best final error           : "
    f"{np.min(final_errors):.6f}"
)

print(
    f"Worst final error          : "
    f"{np.max(final_errors):.6f}"
)

print(
    f"Average reward             : "
    f"{np.mean(rewards_all):.6f}"
)

print(
    f"Successes                  : "
    f"{successes}/100"
)

print(
    f"Success rate               : "
    f"{successes:.2f}%"
)

print(
    f"Unstable                   : "
    f"{unstable}/100"
)

print("=" * 70)