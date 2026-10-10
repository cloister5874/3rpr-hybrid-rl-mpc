from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import sys

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"C:\Users\madha\OneDrive\Desktop\project_1")

CHECKPOINT = (
    PROJECT_DIR
    / "training"
    / "final_model"
    / "sac_expert_sac_final.pt"
)

sys.path.insert(0, str(PROJECT_DIR))

from dynamics.dynamics import step


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAX_ACTION = 2.0

DT = 0.01

MAX_STEPS = 300

# Position/orientation error at which we consider
# the desired pose reached.
POSITION_TOLERANCE = 0.01


STATE_LOW = np.array(
    [0.0, 0.0, -0.5, -0.2, -0.2, -0.2],
    dtype=np.float32
)

STATE_HIGH = np.array(
    [1.0, 1.0, 0.5, 0.2, 0.2, 0.2],
    dtype=np.float32
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(x):

    lo = torch.tensor(
        STATE_LOW,
        dtype=torch.float32,
        device=x.device
    )

    hi = torch.tensor(
        STATE_HIGH,
        dtype=torch.float32,
        device=x.device
    )

    return 2.0 * (x - lo) / (hi - lo) - 1.0


# ============================================================
# SAC ACTOR
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


    def deterministic(self, state, desired_state):

        x = torch.cat(
            [
                normalize(state),
                normalize(desired_state)
            ],
            dim=1
        )

        h = self.net(x)

        action = MAX_ACTION * torch.tanh(
            self.mean(h)
        )

        return action


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 75)
print("3-RPR SAC CLOSED-LOOP CONTROLLER")
print("=" * 75)

print("Checkpoint:")
print(CHECKPOINT)

print("\nDevice:")

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("CPU")


if not CHECKPOINT.exists():

    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)


actor = Actor().to(DEVICE)

actor.load_state_dict(
    checkpoint["actor"]
)

actor.eval()

print("\nSAC actor loaded successfully.")


# ============================================================
# USER INPUT
# ============================================================

print("\n" + "=" * 75)
print("ENTER INITIAL STATE")
print("=" * 75)

print("""
[x, y, theta, x_dot, y_dot, theta_dot]

Example:
[0.45, 0.55, 0.05, 0.02, -0.01, 0.03]
""")

current_input = input(
    "Initial state: "
).strip()


print("\n" + "=" * 75)
print("ENTER DESIRED STATE")
print("=" * 75)

print("""
[xd, yd, theta_d, 0, 0, 0]

Example:
[0.50, 0.50, 0.00, 0, 0, 0]
""")

desired_input = input(
    "Desired state: "
).strip()


# ============================================================
# PARSE INPUT
# ============================================================

try:

    current_state = np.array(
        [
            float(x.strip())
            for x in current_input.strip("[]").split(",")
        ],
        dtype=np.float64
    )

    desired_state = np.array(
        [
            float(x.strip())
            for x in desired_input.strip("[]").split(",")
        ],
        dtype=np.float64
    )

except ValueError:

    raise ValueError(
        "Invalid input. Enter six comma-separated numbers."
    )


if len(current_state) != 6:

    raise ValueError(
        "Initial state must contain exactly 6 values."
    )


if len(desired_state) != 6:

    raise ValueError(
        "Desired state must contain exactly 6 values."
    )


# ============================================================
# INITIAL ERROR
# ============================================================

initial_pose_error = np.linalg.norm(
    current_state[:3] - desired_state[:3]
)


print("\nInitial pose error:")
print(f"{initial_pose_error:.6f}")


# ============================================================
# TRAJECTORY
# ============================================================

trajectory = []

state = current_state.copy()

reached = False


# ============================================================
# CLOSED-LOOP SAC
# ============================================================

for t in range(MAX_STEPS):

    # --------------------------------------------------------
    # Check whether desired pose has been reached
    # --------------------------------------------------------

    pose_error = np.linalg.norm(
        state[:3] - desired_state[:3]
    )

    if pose_error <= POSITION_TOLERANCE:

        reached = True

        break


    # --------------------------------------------------------
    # SAC INPUT
    # --------------------------------------------------------

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)


    desired_tensor = torch.tensor(
        desired_state,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)


    # --------------------------------------------------------
    # SAC ACTION
    # --------------------------------------------------------

    with torch.no_grad():

        action = actor.deterministic(
            state_tensor,
            desired_tensor
        )


    forces = action.squeeze(0).cpu().numpy()

    forces = np.clip(
        forces,
        -MAX_ACTION,
        MAX_ACTION
    )


    # --------------------------------------------------------
    # APPLY FORCES TO 3-RPR DYNAMICS
    # --------------------------------------------------------

    next_state = np.asarray(
        step(
            state,
            forces,
            dt=DT
        ),
        dtype=np.float64
    )


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if (
        not np.all(np.isfinite(next_state))
        or np.any(np.abs(next_state) > 5.0)
    ):

        print("\nWARNING: Dynamics became unstable.")

        break


    # --------------------------------------------------------
    # NEW ERROR
    # --------------------------------------------------------

    next_pose_error = np.linalg.norm(
        next_state[:3] - desired_state[:3]
    )


    # --------------------------------------------------------
    # STORE COMPLETE STEP
    # --------------------------------------------------------

    trajectory.append({

        "step": t + 1,

        "forces": forces.copy(),

        "next_state": next_state.copy(),

        "pose_error": next_pose_error
    })


    # --------------------------------------------------------
    # MOVE TO NEXT STATE
    # --------------------------------------------------------

    state = next_state


# ============================================================
# FINAL RESULTS
# ============================================================

final_pose_error = np.linalg.norm(
    state[:3] - desired_state[:3]
)


print("\n" + "=" * 75)
print("SAC TRAJECTORY")
print("=" * 75)


for item in trajectory:

    print(
        f"Step {item['step']:3d} | "
        f"Force = "
        f"[{item['forces'][0]: .6f}, "
        f"{item['forces'][1]: .6f}, "
        f"{item['forces'][2]: .6f}] | "
        f"Next state = "
        f"[{item['next_state'][0]: .6f}, "
        f"{item['next_state'][1]: .6f}, "
        f"{item['next_state'][2]: .6f}, "
        f"{item['next_state'][3]: .6f}, "
        f"{item['next_state'][4]: .6f}, "
        f"{item['next_state'][5]: .6f}] | "
        f"Error = {item['pose_error']:.6f}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL RESULT")
print("=" * 75)

print(f"Steps taken          : {len(trajectory)}")
print(f"Initial pose error   : {initial_pose_error:.6f}")
print(f"Final pose error     : {final_pose_error:.6f}")
print(f"Improvement          : "
      f"{initial_pose_error - final_pose_error:.6f}")

print(f"Desired pose reached : {reached}")

print("\nFinal state:")
print(state)

print("\nDesired state:")
print(desired_state)


# ============================================================
# ARRAYS
# ============================================================

force_array = np.array(
    [x["forces"] for x in trajectory]
)

state_array = np.array(
    [x["next_state"] for x in trajectory]
)


print("\n" + "=" * 75)
print("FORCE ARRAY")
print("=" * 75)

print(force_array)


print("\n" + "=" * 75)
print("NEXT-STATE ARRAY")
print("=" * 75)

print(state_array)


# ============================================================
# COMBINED TRAJECTORY ARRAY
# ============================================================

# Each row:
#
# [F1,F2,F3, x,y,theta,x_dot,y_dot,theta_dot]
#
# So this gives:
#
# [
#   [F1,F2,F3, next_state...],
#   [F1,F2,F3, next_state...],
#   ...
# ]

combined_trajectory = np.array(
    [
        np.concatenate(
            [
                x["forces"],
                x["next_state"]
            ]
        )
        for x in trajectory
    ]
)


print("\n" + "=" * 75)
print("COMBINED TRAJECTORY ARRAY")
print("=" * 75)

print(combined_trajectory)


print("\nShape:")
print(combined_trajectory.shape)