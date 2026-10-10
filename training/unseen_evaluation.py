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
# MODEL NORMALIZATION
# EXACTLY THE SAME AS TRAINING
# ============================================================

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
# EXACT FINAL MODEL ARCHITECTURE
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
# LOAD MODEL
# ============================================================

print("=" * 75)
print("SAC — UNSEEN EVALUATION")
print("=" * 75)

print("\nCheckpoint:")
print(CKPT)

print(
    "\nDevice:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)


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

print("\nCheckpoint loaded successfully.")
print("Algorithm:", ckpt.get("algorithm", "SAC"))
print("State dimension:", ckpt.get("state_dim", 6))
print("Desired dimension:", ckpt.get("desired_dim", 6))
print("Action dimension:", ckpt.get("action_dim", 3))


# ============================================================
# UNSEEN EVALUATION SETTINGS
# ============================================================

SEEDS = [
    2026,
    2027,
    2028,
    2029,
    2030
]

TASKS_PER_RUN = 100
STEPS = 300


# ============================================================
# OVERALL RESULTS
# ============================================================

overall_successes = 0
overall_tasks = 0
overall_unstable = 0

overall_initial_errors = []
overall_final_errors = []
overall_improvements = []
overall_rewards = []


# ============================================================
# RUN EACH NEW SEED
# ============================================================

for run_number, seed in enumerate(SEEDS, start=1):

    print("\n")
    print("=" * 75)
    print(
        f"UNSEEN RUN {run_number}/"
        f"{len(SEEDS)}  —  SEED {seed}"
    )
    print("=" * 75)

    rng = np.random.default_rng(seed)

    successes = 0
    unstable = 0

    initial_errors = []
    final_errors = []
    improvements = []
    rewards_all = []


    # ========================================================
    # 100 NEW TASKS
    # ========================================================

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


        # ----------------------------------------------------
        # INITIAL ERROR
        # ----------------------------------------------------

        initial_error = np.linalg.norm(
            state[:3] - desired[:3]
        )

        total_reward = 0.0
        failed = False


        # ====================================================
        # 300 SIMULATION STEPS
        # ====================================================

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


            # ------------------------------------------------
            # SAC DETERMINISTIC ACTION
            # ------------------------------------------------

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


            # ------------------------------------------------
            # 3-RPR DYNAMICS
            # ------------------------------------------------

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


            # ------------------------------------------------
            # STABILITY CHECK
            # ------------------------------------------------

            if (
                not np.all(np.isfinite(next_state))
                or np.any(np.abs(next_state) > 5.0)
            ):

                failed = True
                unstable += 1
                break


            # ------------------------------------------------
            # EXACT PROJECT REWARD
            # ------------------------------------------------

            reward = (
                -np.sum(
                    (next_state - desired) ** 2
                )
                - 0.01 * np.sum(action ** 2)
            )

            total_reward += reward

            state = next_state


        # ====================================================
        # TASK RESULT
        # ====================================================

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

            # ----------------------------------------------
            # SUCCESS CRITERION
            # ----------------------------------------------

            if final_error < initial_error:
                successes += 1


        initial_errors.append(initial_error)
        final_errors.append(final_error)
        improvements.append(improvement)
        rewards_all.append(total_reward)


    # ========================================================
    # RUN RESULTS
    # ========================================================

    success_rate = (
        successes / TASKS_PER_RUN
    ) * 100.0

    print(
        f"\nSuccesses        : "
        f"{successes}/{TASKS_PER_RUN}"
    )

    print(
        f"Success rate     : "
        f"{success_rate:.2f}%"
    )

    print(
        f"Average initial  : "
        f"{np.mean(initial_errors):.6f}"
    )

    print(
        f"Average final    : "
        f"{np.mean(final_errors):.6f}"
    )

    print(
        f"Average improve  : "
        f"{np.mean(improvements):.6f}"
    )

    print(
        f"Average reward   : "
        f"{np.mean(rewards_all):.6f}"
    )

    print(
        f"Unstable         : "
        f"{unstable}/{TASKS_PER_RUN}"
    )


    # ========================================================
    # ADD TO OVERALL RESULTS
    # ========================================================

    overall_successes += successes
    overall_tasks += TASKS_PER_RUN
    overall_unstable += unstable

    overall_initial_errors.extend(
        initial_errors
    )

    overall_final_errors.extend(
        final_errors
    )

    overall_improvements.extend(
        improvements
    )

    overall_rewards.extend(
        rewards_all
    )


# ============================================================
# FINAL 500-TASK RESULT
# ============================================================

overall_success_rate = (
    overall_successes / overall_tasks
) * 100.0


print("\n")
print("=" * 75)
print("OVERALL UNSEEN EVALUATION")
print("=" * 75)

print(
    f"\nTotal unseen tasks      : "
    f"{overall_tasks}"
)

print(
    f"Successful tasks       : "
    f"{overall_successes}/{overall_tasks}"
)

print(
    f"Overall success rate   : "
    f"{overall_success_rate:.2f}%"
)

print(
    f"Average initial error  : "
    f"{np.mean(overall_initial_errors):.6f}"
)

print(
    f"Average final error    : "
    f"{np.mean(overall_final_errors):.6f}"
)

print(
    f"Average improvement    : "
    f"{np.mean(overall_improvements):.6f}"
)

print(
    f"Average reward         : "
    f"{np.mean(overall_rewards):.6f}"
)

print(
    f"Unstable episodes      : "
    f"{overall_unstable}/{overall_tasks}"
)

print("=" * 75)

print("\nSeeds used:")
for seed in SEEDS:
    print(f"  {seed}")

print("\nEvaluation complete.")