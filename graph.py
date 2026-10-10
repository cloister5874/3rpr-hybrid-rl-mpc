import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\madha\OneDrive\Desktop\project_1"
)

NOTEBOOK = (
    PROJECT_DIR
    / "training"
    / "sac_training_FINAL.ipynb"
)


# ============================================================
# LOAD NOTEBOOK
# ============================================================

with open(NOTEBOOK, "r", encoding="utf-8") as f:
    notebook = json.load(f)


# ============================================================
# FIND EPISODE REWARD DATA FROM NOTEBOOK OUTPUTS
# ============================================================

episode_rewards = None

for cell in notebook["cells"]:

    outputs = cell.get("outputs", [])

    for output in outputs:

        text_parts = []

        if "text" in output:
            text_parts.extend(output["text"])

        if "data" in output:
            data = output["data"]

            if "text/plain" in data:
                text_parts.extend(data["text/plain"])

        text = "".join(text_parts)

        if "episode_rewards" in text:

            print("Found episode reward output.")

            # Try to extract a Python-style list
            start = text.find("[")

            end = text.rfind("]")

            if start != -1 and end != -1:

                try:

                    episode_rewards = eval(
                        text[start:end + 1],
                        {"__builtins__": {}},
                        {}
                    )

                    break

                except Exception:
                    pass

    if episode_rewards is not None:
        break


# ============================================================
# IF NOT FOUND
# ============================================================

if episode_rewards is None:

    print()
    print("Could not find saved episode_rewards data")
    print("inside sac_training_FINAL.ipynb.")
    print()
    print("No graph was generated.")
    print("The model was NOT changed.")

    raise SystemExit


# ============================================================
# CONVERT TO NUMPY
# ============================================================

episode_rewards = np.asarray(
    episode_rewards,
    dtype=np.float64
)


episodes = np.arange(
    1,
    len(episode_rewards) + 1
)


# ============================================================
# SMOOTHING
# ============================================================

window = min(50, len(episode_rewards))

if window > 1:

    moving_average = np.convolve(
        episode_rewards,
        np.ones(window) / window,
        mode="valid"
    )

    moving_x = np.arange(
        window,
        len(episode_rewards) + 1
    )

else:

    moving_average = episode_rewards
    moving_x = episodes


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    episodes,
    episode_rewards,
    alpha=0.35,
    label="Episode reward"
)

plt.plot(
    moving_x,
    moving_average,
    linewidth=2,
    label=f"{window}-episode moving average"
)

plt.xlabel("Training episode")

plt.ylabel("Episode reward")

plt.title(
    "SAC Training Reward Convergence"
)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

OUTPUT = (
    PROJECT_DIR
    / "training"
    / "sac_training_reward_curve.png"
)

plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print()
print("=" * 60)
print("GRAPH GENERATED")
print("=" * 60)
print("Episodes:", len(episode_rewards))
print("Saved to:")
print(OUTPUT)