
import os
import sys
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = r"C:\Users\madha\OneDrive\Desktop\project_1"

MODEL_PATH = os.path.join(
    ROOT,
    "training",
    "final_model",
    "sac_expert_sac_final.pt"
)

DYNAMICS_PATH = os.path.join(
    ROOT,
    "dynamics",
    "dynamics.py"
)

RESULTS_DIR = os.path.join(
    ROOT,
    "training",
    "final_results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# ============================================================
# FIXED PROJECT CONSTRAINTS
# ============================================================

STATE_DIM = 6
ACTION_DIM = 3

MAX_ACTION = 2.0
DT = 0.01
GAMMA = 0.99

NUM_TASKS = 100
STEPS = 300
SEED = 12345

STATE_LOW = np.array(
    [0.4, 0.4, -0.1, -0.1, -0.1, -0.1],
    dtype=np.float32
)

STATE_HIGH = np.array(
    [0.6, 0.6, 0.1, 0.1, 0.1, 0.1],
    dtype=np.float32
)

# ============================================================
# LOAD DYNAMICS
# ============================================================

spec = importlib.util.spec_from_file_location(
    "rpr_final_dynamics",
    DYNAMICS_PATH
)

dynamics = importlib.util.module_from_spec(spec)

sys.modules["rpr_final_dynamics"] = dynamics

spec.loader.exec_module(dynamics)

step = dynamics.step

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
            nn.ReLU()
        )

        self.mean = nn.Linear(
            256,
            3
        )

        self.log_std = nn.Linear(
            256,
            3
        )

    def forward(
        self,
        state,
        desired
    ):

        low = torch.tensor(
            STATE_LOW,
            dtype=torch.float32,
            device=state.device
        )

        high = torch.tensor(
            STATE_HIGH,
            dtype=torch.float32,
            device=state.device
        )

        state = (
            2.0 *
            (state - low) /
            (high - low)
            - 1.0
        )

        desired = (
            2.0 *
            (desired - low) /
            (high - low)
            - 1.0
        )

        x = torch.cat(
            [state, desired],
            dim=1
        )

        h = self.net(x)

        mean = self.mean(h)

        log_std = torch.clamp(
            self.log_std(h),
            -5.0,
            -1.0
        )

        return mean, log_std


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

actor = Actor().to(device)

actor.load_state_dict(
    checkpoint["actor"]
)

actor.eval()

# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 75)
print("3-RPR SAC FINAL RESULTS GENERATOR")
print("=" * 75)

print("Model:")
print(MODEL_PATH)

print("\nAlgorithm:")
print("Soft Actor-Critic (SAC)")

print("\nDevice:")
print(device)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("\nState dimension:", STATE_DIM)
print("Action dimension:", ACTION_DIM)

print("\nState:")
print("[x, y, theta, x_dot, y_dot, theta_dot]")

print("\nDesired state:")
print("[x_d, y_d, theta_d, 0, 0, 0]")

print("\nState range:")
print("LOW :", STATE_LOW)
print("HIGH:", STATE_HIGH)

print("\nAction range:")
print("[-2, 2] for each force")

print("\nSimulation:")
print("dt =", DT)
print("Steps =", STEPS)
print("Tasks =", NUM_TASKS)
print("Seed =", SEED)

print("=" * 75)

# ============================================================
# GENERATE EXACT FIXED 100 TASKS
# ============================================================

rng = np.random.default_rng(SEED)

initial_states = np.zeros(
    (NUM_TASKS, 6),
    dtype=np.float32
)

desired_states = np.zeros(
    (NUM_TASKS, 6),
    dtype=np.float32
)

# Current x,y
initial_states[:, 0:2] = rng.uniform(
    0.4,
    0.6,
    (NUM_TASKS, 2)
)

# Current theta
initial_states[:, 2] = rng.uniform(
    -0.1,
    0.1,
    NUM_TASKS
)

# Current velocities
initial_states[:, 3:] = rng.uniform(
    -0.1,
    0.1,
    (NUM_TASKS, 3)
)

# Desired x,y
desired_states[:, 0:2] = rng.uniform(
    0.4,
    0.6,
    (NUM_TASKS, 2)
)

# Desired theta
desired_states[:, 2] = rng.uniform(
    -0.1,
    0.1,
    NUM_TASKS
)

# Desired velocities = zero
desired_states[:, 3:] = 0.0

# ============================================================
# STORAGE
# ============================================================

task_rows = []

all_error = np.full(
    (NUM_TASKS, STEPS + 1),
    np.nan
)

all_reward = np.full(
    (NUM_TASKS, STEPS),
    np.nan
)

all_actions = np.full(
    (NUM_TASKS, STEPS, ACTION_DIM),
    np.nan
)

all_states = np.full(
    (NUM_TASKS, STEPS + 1, STATE_DIM),
    np.nan
)

all_desired = np.zeros(
    (NUM_TASKS, STATE_DIM),
    dtype=np.float32
)

successes = 0
unstable = 0

# ============================================================
# EVALUATE
# ============================================================

for task_id in range(NUM_TASKS):

    state = initial_states[
        task_id
    ].astype(np.float64)

    desired = desired_states[
        task_id
    ].astype(np.float64)

    all_desired[
        task_id
    ] = desired

    initial_error = np.linalg.norm(
        state[:3] - desired[:3]
    )

    best_error = initial_error

    all_states[
        task_id,
        0
    ] = state

    all_error[
        task_id,
        0
    ] = initial_error

    valid = True

    total_reward = 0.0

    for t in range(STEPS):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        desired_tensor = torch.tensor(
            desired,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        with torch.no_grad():

            mean, log_std = actor(
                state_tensor,
                desired_tensor
            )

            # Deterministic SAC evaluation
            action_tensor = (
                MAX_ACTION *
                torch.tanh(mean)
            )

        action = (
            action_tensor
            .cpu()
            .numpy()[0]
            .astype(np.float64)
        )

        action = np.clip(
            action,
            -MAX_ACTION,
            MAX_ACTION
        )

        all_actions[
            task_id,
            t
        ] = action

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

            print(
                f"Task {task_id+1}: dynamics error:",
                repr(e)
            )

            valid = False
            break

        if not np.all(
            np.isfinite(next_state)
        ):

            valid = False
            break

        if np.linalg.norm(next_state) > 100:

            valid = False
            break

        # ----------------------------------------------------
        # EXACT PROJECT REWARD
        # ----------------------------------------------------

        reward = (
            -np.sum(
                (
                    next_state
                    -
                    desired
                ) ** 2
            )
            -
            0.01 *
            np.sum(
                action ** 2
            )
        )

        total_reward += reward

        state = next_state

        pose_error = np.linalg.norm(
            state[:3] - desired[:3]
        )

        best_error = min(
            best_error,
            pose_error
        )

        all_states[
            task_id,
            t + 1
        ] = state

        all_error[
            task_id,
            t + 1
        ] = pose_error

        all_reward[
            task_id,
            t
        ] = reward

    if not valid:

        unstable += 1

        final_error = 999.0

    else:

        final_error = np.linalg.norm(
            state[:3] - desired[:3]
        )

    improvement = (
        initial_error -
        final_error
    )

    if (
        valid
        and final_error < initial_error
    ):

        success = 1
        successes += 1

    else:

        success = 0

    task_rows.append(
        {
            "task": task_id + 1,

            "initial_x":
                initial_states[task_id, 0],

            "initial_y":
                initial_states[task_id, 1],

            "initial_theta":
                initial_states[task_id, 2],

            "initial_x_dot":
                initial_states[task_id, 3],

            "initial_y_dot":
                initial_states[task_id, 4],

            "initial_theta_dot":
                initial_states[task_id, 5],

            "desired_x":
                desired[0],

            "desired_y":
                desired[1],

            "desired_theta":
                desired[2],

            "initial_pose_error":
                initial_error,

            "final_pose_error":
                final_error,

            "best_pose_error":
                best_error,

            "improvement":
                improvement,

            "final_reward":
                (
                    np.nansum(
                        all_reward[task_id]
                    )
                ),

            "success":
                success,

            "unstable":
                int(not valid)
        }
    )

    if (task_id + 1) % 10 == 0:

        print(
            f"Task {task_id+1:03d}/100 | "
            f"Successes: {successes} | "
            f"Unstable: {unstable}"
        )

# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    task_rows
)

csv_path = os.path.join(
    RESULTS_DIR,
    "fixed_100_task_results.csv"
)

df.to_csv(
    csv_path,
    index=False
)

# ============================================================
# METRICS
# ============================================================

avg_initial = df[
    "initial_pose_error"
].mean()

avg_final = df[
    "final_pose_error"
].mean()

avg_improvement = (
    avg_initial -
    avg_final
)

avg_best = df[
    "best_pose_error"
].mean()

best_final = df[
    "final_pose_error"
].min()

worst_final = df[
    "final_pose_error"
].max()

avg_reward = df[
    "final_reward"
].mean()

success_rate = (
    successes /
    NUM_TASKS
) * 100.0

# ============================================================
# TEXT REPORT
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "final_results_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "3-RPR SAC FINAL RESULTS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Algorithm: Soft Actor-Critic (SAC)\n"
    )

    f.write(
        "Model: sac_expert_sac_final.pt\n"
    )

    f.write(
        f"Evaluation tasks: {NUM_TASKS}\n"
    )

    f.write(
        f"Random seed: {SEED}\n"
    )

    f.write(
        f"Simulation steps/task: {STEPS}\n"
    )

    f.write(
        f"Time step: {DT}\n\n"
    )

    f.write(
        "STATE RANGE\n"
    )

    f.write(
        f"LOW  = {STATE_LOW.tolist()}\n"
    )

    f.write(
        f"HIGH = {STATE_HIGH.tolist()}\n\n"
    )

    f.write(
        "ACTION RANGE\n"
    )

    f.write(
        "[-2, 2] for each of the three forces\n\n"
    )

    f.write(
        "RESULTS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Average initial pose error : "
        f"{avg_initial:.6f}\n"
    )

    f.write(
        f"Average final pose error   : "
        f"{avg_final:.6f}\n"
    )

    f.write(
        f"Average improvement        : "
        f"{avg_improvement:.6f}\n"
    )

    f.write(
        f"Average best pose error    : "
        f"{avg_best:.6f}\n"
    )

    f.write(
        f"Best final error           : "
        f"{best_final:.6f}\n"
    )

    f.write(
        f"Worst final error          : "
        f"{worst_final:.6f}\n"
    )

    f.write(
        f"Average episode reward     : "
        f"{avg_reward:.6f}\n"
    )

    f.write(
        f"Successes                  : "
        f"{successes}/{NUM_TASKS}\n"
    )

    f.write(
        f"Success rate               : "
        f"{success_rate:.2f}%\n"
    )

    f.write(
        f"Unstable tasks             : "
        f"{unstable}/{NUM_TASKS}\n"
    )

# ============================================================
# SAVE RAW ARRAYS
# ============================================================

np.savez(
    os.path.join(
        RESULTS_DIR,
        "complete_trajectories.npz"
    ),
    states=all_states,
    actions=all_actions,
    rewards=all_reward,
    errors=all_error,
    initial_states=initial_states,
    desired_states=desired_states
)

# ============================================================
# GRAPH 1 — AVERAGE ERROR VS TIME
# ============================================================

mean_error = np.nanmean(
    all_error,
    axis=0
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    np.arange(STEPS + 1) * DT,
    mean_error
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Pose Error"
)

plt.title(
    "Average Pose Error During SAC Control"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "01_average_error_vs_time.png"
    ),
    dpi=200
)

plt.close()

# ============================================================
# GRAPH 2 — INITIAL VS FINAL ERROR
# ============================================================

plt.figure(
    figsize=(10, 6)
)

x = np.arange(
    1,
    NUM_TASKS + 1
)

plt.plot(
    x,
    df["initial_pose_error"],
    label="Initial Error"
)

plt.plot(
    x,
    df["final_pose_error"],
    label="Final Error"
)

plt.xlabel(
    "Task"
)

plt.ylabel(
    "Pose Error"
)

plt.title(
    "Initial vs Final Pose Error — 100 Evaluation Tasks"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "02_initial_vs_final_error.png"
    ),
    dpi=200
)

plt.close()

# ============================================================
# GRAPH 3 — IMPROVEMENT PER TASK
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    x,
    df["improvement"]
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Task"
)

plt.ylabel(
    "Initial Error - Final Error"
)

plt.title(
    "Pose Error Improvement Per Evaluation Task"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "03_improvement_per_task.png"
    ),
    dpi=200
)

plt.close()

# ============================================================
# GRAPH 4 — FINAL ERROR DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    df["final_pose_error"],
    bins=20
)

plt.xlabel(
    "Final Pose Error"
)

plt.ylabel(
    "Number of Tasks"
)

plt.title(
    "Distribution of Final Pose Error"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "04_final_error_distribution.png"
    ),
    dpi=200
)

plt.close()

# ============================================================
# GRAPH 5 — SAMPLE ERROR TRAJECTORIES
# ============================================================

plt.figure(
    figsize=(10, 6)
)

sample_tasks = [
    0,
    1,
    2,
    9,
    49
]

for task_id in sample_tasks:

    plt.plot(
        np.arange(STEPS + 1) * DT,
        all_error[task_id],
        label=f"Task {task_id + 1}"
    )

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Pose Error"
)

plt.title(
    "Sample SAC Pose-Error Trajectories"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "05_sample_error_trajectories.png"
    ),
    dpi=200
)

plt.close()

# ============================================================
# GRAPH 6 — SAMPLE ACTIONS
# ============================================================

task_id = 0

time = np.arange(
    STEPS
) * DT

plt.figure(
    figsize=(10, 6)
)

for j in range(3):

    plt.plot(
        time,
        all_actions[
            task_id,
            :,
            j
        ],
        label=f"F{j+1}"
    )

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Force"
)

plt.title(
    "SAC Control Forces — Sample Task 1"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "06_sample_control_forces.png"
    ),
    dpi=200
)

plt.close()

# ============================================================
# GRAPH 7 — SAMPLE REWARD
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    time,
    all_reward[task_id]
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Reward"
)

plt.title(
    "Step Reward — Sample Task 1"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "07_sample_reward.png"
    ),
    dpi=200
)

plt.close()

# ============================================================
# GRAPH 8 — SAMPLE X/Y TRAJECTORY
# ============================================================

plt.figure(
    figsize=(8, 8)
)

plt.plot(
    all_states[
        task_id,
        :,
        0
    ],
    all_states[
        task_id,
        :,
        1
    ],
    label="Robot trajectory"
)

plt.scatter(
    initial_states[
        task_id,
        0
    ],
    initial_states[
        task_id,
        1
    ],
    label="Initial"
)

plt.scatter(
    desired_states[
        task_id,
        0
    ],
    desired_states[
        task_id,
        1
    ],
    label="Desired"
)

plt.xlabel(
    "X Position"
)

plt.ylabel(
    "Y Position"
)

plt.title(
    "3-RPR End-Effector XY Trajectory — Sample Task 1"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.axis(
    "equal"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "08_xy_trajectory.png"
    ),
    dpi=200
)

plt.close()

# ============================================================
# GRAPH 9 — STATE VARIABLES
# ============================================================

labels = [
    "x",
    "y",
    "theta",
    "x_dot",
    "y_dot",
    "theta_dot"
]

for j in range(6):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        time=np.arange(STEPS + 1) * DT,
        y=all_states[
            task_id,
            :,
            j
        ]
    )

    if j < 3:

        plt.axhline(
            desired_states[
                task_id,
                j
            ],
            linestyle="--",
            label="Desired"
        )

        plt.legend()

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        labels[j]
    )

    plt.title(
        f"Sample Task 1 — {labels[j]} vs Time"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"09_state_{j+1}_{labels[j]}.png"
        ),
        dpi=200
    )

    plt.close()

# ============================================================
# GRAPH 10 — SUCCESS SUMMARY
# ============================================================

success_values = [
    successes,
    NUM_TASKS - successes
]

plt.figure(
    figsize=(7, 6)
)

plt.bar(
    ["Successful", "Failed"],
    success_values
)

plt.ylabel(
    "Number of Tasks"
)

plt.title(
    "SAC Evaluation Success Summary"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "16_success_summary.png"
    ),
    dpi=200
)

plt.close()

# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("FINAL SAC RESULTS")
print("=" * 75)

print(
    f"Average initial pose error : {avg_initial:.6f}"
)

print(
    f"Average final pose error   : {avg_final:.6f}"
)

print(
    f"Average improvement        : {avg_improvement:.6f}"
)

print(
    f"Average best pose error    : {avg_best:.6f}"
)

print(
    f"Best final error           : {best_final:.6f}"
)

print(
    f"Worst final error          : {worst_final:.6f}"
)

print(
    f"Average episode reward     : {avg_reward:.6f}"
)

print(
    f"Successes                  : {successes}/100"
)

print(
    f"Success rate               : {success_rate:.2f}%"
)

print(
    f"Unstable tasks             : {unstable}/100"
)

print("\nResults saved to:")

print(
    RESULTS_DIR
)

print("\nCSV:")
print(csv_path)

print("\nSummary:")
print(summary_path)

print("=" * 75)
