import os
import csv
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 3-RPR SAC CONTROLLER
# Standalone terminal program
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "training",
    "final_model",
    "sac_90_percent.pt"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "training",
    "final_model",
    "terminal_runs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SYSTEM PARAMETERS
# ============================================================

STATE_DIM = 6
DESIRED_STATE_DIM = 6
ACTOR_INPUT_DIM = 12
ACTION_DIM = 3

MAX_ACTION = 2.0
DT = 0.01

POSITION_TOLERANCE = 0.005
MAX_CONTROL_STEPS = 500

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMPORT ACTUAL 3-RPR DYNAMICS
# ============================================================

from dynamics.dynamics import step


# ============================================================
# SAC ACTOR
# Same architecture used during training
# ============================================================

class SACActor(nn.Module):

    def __init__(
        self,
        input_dim=ACTOR_INPUT_DIM,
        action_dim=ACTION_DIM,
        max_action=MAX_ACTION
    ):

        super().__init__()

        self.max_action = max_action

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                256
            ),

            nn.ReLU()
        )

        self.mean_layer = nn.Linear(
            256,
            action_dim
        )

        self.log_std_layer = nn.Linear(
            256,
            action_dim
        )


    def forward(
        self,
        state,
        desired_state
    ):

        x = torch.cat(
            [
                state,
                desired_state
            ],
            dim=-1
        )

        x = self.network(x)

        mean = self.mean_layer(x)

        log_std = self.log_std_layer(x)

        log_std = torch.clamp(
            log_std,
            -20,
            2
        )

        return mean, log_std


# ============================================================
# STATE NORMALIZATION
# Same normalization used by trained SAC
# ============================================================

def normalize_state(
    state
):

    center = torch.tensor(
        [
            0.5,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        dtype=torch.float32,
        device=DEVICE
    )

    scale = torch.tensor(
        [
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1
        ],
        dtype=torch.float32,
        device=DEVICE
    )

    return (
        state - center
    ) / scale


# ============================================================
# LOAD TRAINED SAC MODEL
# ============================================================

print()
print("=" * 75)
print("3-RPR SAC CONTROLLER")
print("=" * 75)

print(
    "\nDevice:",
    DEVICE
)

print(
    "Model:",
    MODEL_PATH
)


if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        "\nSAC model not found.\n"
        "Expected location:\n"
        + MODEL_PATH
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


# ============================================================
# CREATE ACTOR
# ============================================================

actor = SACActor(
    input_dim=ACTOR_INPUT_DIM,
    action_dim=ACTION_DIM,
    max_action=MAX_ACTION
).to(DEVICE)


# ============================================================
# LOAD TRAINED ACTOR WEIGHTS
# ============================================================

actor.load_state_dict(
    checkpoint["actor"]
)

actor.eval()


# ============================================================
# LOAD ACTION SCALE
# The final 90% evaluation used 4.0
# ============================================================

ACTION_SCALE = float(
    checkpoint.get(
        "action_scale",
        4.0
    )
)


print(
    "\nSAC model loaded successfully."
)

print(
    "Algorithm: SAC"
)

print(
    "State dimension:",
    STATE_DIM
)

print(
    "Desired state dimension:",
    DESIRED_STATE_DIM
)

print(
    "Actor input:",
    ACTOR_INPUT_DIM
)

print(
    "Action dimension:",
    ACTION_DIM
)

print(
    "Maximum force:",
    MAX_ACTION
)

print(
    "Action scale:",
    ACTION_SCALE
)


# ============================================================
# USER INPUT
# Comma-separated values
# ============================================================

def get_state(
    message
):

    while True:

        user_input = input(
            message
        ).strip()

        try:

            values = [
                float(x.strip())
                for x in user_input.split(",")
            ]

            if len(values) != 6:

                print(
                    "\nERROR: Enter exactly 6 values."
                )

                print(
                    "Format:"
                )

                print(
                    "x,y,theta,x_dot,y_dot,theta_dot"
                )

                continue

            return np.array(
                values,
                dtype=np.float32
            )

        except ValueError:

            print(
                "\nERROR: Invalid input."
            )

            print(
                "Example:"
            )

            print(
                "0.45,0.55,0.05,0.02,-0.03,0.01"
            )


# ============================================================
# SAC FORCE GENERATION
# ============================================================

def get_sac_force(
    current_state,
    desired_state
):

    current_tensor = torch.tensor(
        current_state,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    desired_tensor = torch.tensor(
        desired_state,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)


    # Normalize neural-network inputs
    current_normalized = normalize_state(
        current_tensor
    )

    desired_normalized = normalize_state(
        desired_tensor
    )


    # SAC deterministic policy
    with torch.no_grad():

        mean, _ = actor(
            current_normalized,
            desired_normalized
        )

        action = (
            MAX_ACTION *
            torch.tanh(mean)
        )


    action = (
        action
        .cpu()
        .numpy()[0]
    )


    # Action scaling used by final 90% model
    action = (
        action *
        ACTION_SCALE
    )


    # Keep physical forces within [-2, 2]
    action = np.clip(
        action,
        -MAX_ACTION,
        MAX_ACTION
    )


    return action.astype(
        np.float32
    )


# ============================================================
# BASELINE REWARD
# ============================================================

def calculate_reward(
    next_state,
    desired_state,
    action
):

    state_error = np.sum(
        (
            next_state -
            desired_state
        ) ** 2
    )

    force_cost = (
        0.01 *
        np.sum(action ** 2)
    )

    reward = (
        -state_error -
        force_cost
    )

    return float(
        reward
    )


# ============================================================
# GET INITIAL STATE
# ============================================================

print()
print("=" * 75)
print("ENTER INITIAL STATE")
print("=" * 75)

print(
    "\nFormat:"
)

print(
    "x,y,theta,x_dot,y_dot,theta_dot"
)

initial_state = get_state(
    "\nInitial state: "
)


# ============================================================
# GET DESIRED STATE
# ============================================================

print()
print("=" * 75)
print("ENTER DESIRED STATE")
print("=" * 75)

print(
    "\nFormat:"
)

print(
    "x,y,theta,x_dot,y_dot,theta_dot"
)

desired_state = get_state(
    "\nDesired state: "
)


# ============================================================
# INITIAL ERROR
# ============================================================

initial_error = np.linalg.norm(
    initial_state[:3] -
    desired_state[:3]
)


# ============================================================
# DISPLAY TASK
# ============================================================

print()
print("=" * 75)
print("CONTROL TASK")
print("=" * 75)

print(
    "\nInitial state:"
)

print(
    initial_state
)

print(
    "\nDesired state:"
)

print(
    desired_state
)

print(
    "\nInitial pose error:",
    f"{initial_error:.6f}"
)


# ============================================================
# STORAGE
# ============================================================

state_history = []

force_history = []

reward_history = []

error_history = []

time_history = []


# ============================================================
# INITIAL STATE STORAGE
# ============================================================

state = initial_state.copy()

state_history.append(
    state.copy()
)

error_history.append(
    initial_error
)

time_history.append(
    0.0
)


# ============================================================
# CLOSED-LOOP SAC CONTROL
# ============================================================

print()
print("=" * 75)
print("SAC CLOSED-LOOP CONTROL")
print("=" * 75)

for step_number in range(
    MAX_CONTROL_STEPS
):

    # --------------------------------------------------------
    # SAC receives CURRENT state and desired state
    # --------------------------------------------------------

    force_array = get_sac_force(
        state,
        desired_state
    )


    # --------------------------------------------------------
    # Store F1, F2, F3
    # --------------------------------------------------------

    force_history.append(
        force_array.copy()
    )


    # --------------------------------------------------------
    # Apply forces to actual 3-RPR dynamics
    # --------------------------------------------------------

    next_state = step(
        state,
        force_array,
        DT
    )


    # --------------------------------------------------------
    # Calculate reward
    # --------------------------------------------------------

    reward = calculate_reward(
        next_state,
        desired_state,
        force_array
    )

    reward_history.append(
        reward
    )


    # --------------------------------------------------------
    # Calculate pose error
    # --------------------------------------------------------

    error = np.linalg.norm(
        next_state[:3] -
        desired_state[:3]
    )


    # --------------------------------------------------------
    # Store state and error
    # --------------------------------------------------------

    state_history.append(
        next_state.copy()
    )

    error_history.append(
        error
    )

    time_history.append(
        (step_number + 1) * DT
    )


    # --------------------------------------------------------
    # Display current control action
    # --------------------------------------------------------

    print(
        f"Step {step_number + 1:3d} | "
        f"F1={force_array[0]: .4f} | "
        f"F2={force_array[1]: .4f} | "
        f"F3={force_array[2]: .4f} | "
        f"Error={error:.6f}"
    )


    # --------------------------------------------------------
    # Move to next state
    # --------------------------------------------------------

    state = next_state.copy()


    # --------------------------------------------------------
    # Target check
    # --------------------------------------------------------

    if error <= POSITION_TOLERANCE:

        print()
        print(
            "TARGET REACHED"
        )

        break


# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

state_history = np.array(
    state_history,
    dtype=np.float32
)

force_history = np.array(
    force_history,
    dtype=np.float32
)

reward_history = np.array(
    reward_history,
    dtype=np.float32
)

error_history = np.array(
    error_history,
    dtype=np.float32
)

time_history = np.array(
    time_history,
    dtype=np.float32
)


# ============================================================
# 3D FORCE ARRAY
#
# Shape:
#
# (trajectory, timestep, force)
#
# Example:
#
# (1, 408, 3)
# ============================================================

force_array_3d = (
    force_history[
        np.newaxis,
        :,
        :
    ]
)


# ============================================================
# SAVE COMPLETE TRAJECTORY
# ============================================================

npz_path = os.path.join(
    OUTPUT_DIR,
    "latest_trajectory.npz"
)

np.savez(
    npz_path,
    initial_state=initial_state,
    desired_state=desired_state,
    state_history=state_history,
    force_history=force_history,
    force_array_3d=force_array_3d,
    reward_history=reward_history,
    error_history=error_history,
    time_history=time_history
)


# ============================================================
# SAVE FORCE VALUES AS CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "latest_forces.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow(
        [
            "timestep",
            "time_seconds",
            "F1",
            "F2",
            "F3"
        ]
    )


    for i in range(
        len(force_history)
    ):

        writer.writerow(
            [
                i + 1,
                time_history[i + 1],
                force_history[i][0],
                force_history[i][1],
                force_history[i][2]
            ]
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

final_error = error_history[-1]

total_reward = np.sum(
    reward_history
)


print()
print("=" * 75)
print("FINAL SAC RESULT")
print("=" * 75)

print(
    "\nInitial state:"
)

print(
    initial_state
)

print(
    "\nDesired state:"
)

print(
    desired_state
)

print(
    "\nNumber of control steps:"
)

print(
    len(force_history)
)

print(
    "\nSimulation time:"
)

print(
    f"{len(force_history) * DT:.2f} seconds"
)

print(
    "\nInitial pose error:"
)

print(
    f"{initial_error:.6f}"
)

print(
    "\nFinal pose error:"
)

print(
    f"{final_error:.6f}"
)

print(
    "\nFinal state:"
)

print(
    state_history[-1]
)

print(
    "\nTotal reward:"
)

print(
    f"{total_reward:.6f}"
)

print(
    "\nForce history shape:"
)

print(
    force_history.shape
)

print(
    "\n3D force array shape:"
)

print(
    force_array_3d.shape
)

print(
    "\nFirst force [F1,F2,F3]:"
)

print(
    force_history[0]
)

print(
    "\nLast force [F1,F2,F3]:"
)

print(
    force_history[-1]
)


# ============================================================
# FINAL STATUS
# ============================================================

if final_error <= POSITION_TOLERANCE:

    print(
        "\nSTATUS: TARGET REACHED"
    )

else:

    print(
        "\nSTATUS: TARGET NOT REACHED"
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print()
print(
    "Trajectory data saved to:"
)

print(
    npz_path
)

print()
print(
    "Force values saved to:"
)

print(
    csv_path
)

print()
print("=" * 75)
print("CONTROL COMPLETE")
print("=" * 75)