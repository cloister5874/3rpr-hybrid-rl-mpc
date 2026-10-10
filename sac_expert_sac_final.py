
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\madha\OneDrive\Desktop\project_1")
DATA = PROJECT_DIR / "training" / "final_model" / "sac_expert_demonstrations.npz"
OUT = PROJECT_DIR / "training" / "final_model" / "sac_expert_sac_final.pt"

sys.path.insert(0, str(PROJECT_DIR))

from dynamics.dynamics import step

torch.set_float32_matmul_precision("high")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("SAC EXPERT-PRETRAINED FINAL")
print("=" * 70)
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ------------------------------------------------------------
# SAC constants
# ------------------------------------------------------------
STATE_LOW  = np.array([0.0,0.0,-0.5,-0.2,-0.2,-0.2], dtype=np.float32)
STATE_HIGH = np.array([1.0,1.0, 0.5, 0.2, 0.2, 0.2], dtype=np.float32)

MAX_ACTION = 2.0
GAMMA = 0.99
TAU = 0.005

def normalize(x):
    lo = torch.tensor(STATE_LOW, device=x.device)
    hi = torch.tensor(STATE_HIGH, device=x.device)
    return 2.0 * (x - lo) / (hi - lo) - 1.0

# ------------------------------------------------------------
# Actor
# ------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256,3)
        self.log_std = nn.Linear(256,3)

    def forward(self, state, desired):
        x = torch.cat([normalize(state), normalize(desired)], dim=1)
        h = self.net(x)

        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -5.0, -1.0)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        y = torch.tanh(z)

        action = MAX_ACTION * y

        log_prob = dist.log_prob(z)
        log_prob -= torch.log(1 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def deterministic(self, state, desired):
        x = torch.cat([normalize(state), normalize(desired)], dim=1)
        h = self.net(x)
        return MAX_ACTION * torch.tanh(self.mean(h))

# ------------------------------------------------------------
# Critic
# ------------------------------------------------------------
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(15,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )

    def forward(self, state, desired, action):
        x = torch.cat(
            [normalize(state), normalize(desired), action / MAX_ACTION],
            dim=1
        )
        return self.q(x)

# ------------------------------------------------------------
# Load demonstrations
# ------------------------------------------------------------
data = np.load(DATA)

states = torch.tensor(data["states"], dtype=torch.float32, device=DEVICE)
desired = torch.tensor(data["desired_states"], dtype=torch.float32, device=DEVICE)
actions = torch.tensor(data["actions"], dtype=torch.float32, device=DEVICE)
next_states = torch.tensor(data["next_states"], dtype=torch.float32, device=DEVICE)
rewards = torch.tensor(data["rewards"], dtype=torch.float32, device=DEVICE).unsqueeze(1)

N = len(states)

print("Expert transitions:", N)

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
actor = Actor().to(DEVICE)
critic1 = Critic().to(DEVICE)
critic2 = Critic().to(DEVICE)

target1 = Critic().to(DEVICE)
target2 = Critic().to(DEVICE)

target1.load_state_dict(critic1.state_dict())
target2.load_state_dict(critic2.state_dict())

# ------------------------------------------------------------
# STAGE 1 — Strong actor pretraining
# ------------------------------------------------------------
print("\nSTAGE 1 — SAC ACTOR EXPERT PRETRAINING")

actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)

batch_size = 512
steps = 30000

best_loss = float("inf")

for i in range(steps):
    idx = torch.randint(0, N, (batch_size,), device=DEVICE)

    pred = actor.deterministic(states[idx], desired[idx])

    loss = F.mse_loss(pred, actions[idx])

    actor_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_opt.step()

    if (i + 1) % 3000 == 0:
        print(f"BC {i+1:5d}/{steps}  loss={loss.item():.8f}")

    if loss.item() < best_loss:
        best_loss = loss.item()

print("Best BC loss:", best_loss)

# ------------------------------------------------------------
# STAGE 2 — Critic supervised initialization
# ------------------------------------------------------------
print("\nSTAGE 2 — CRITIC INITIALIZATION")

critic_opt1 = torch.optim.Adam(critic1.parameters(), lr=2e-4)
critic_opt2 = torch.optim.Adam(critic2.parameters(), lr=2e-4)

# Monte Carlo approximation from stored trajectories.
# Each episode contains 300 transitions.
returns = torch.zeros_like(rewards)

with torch.no_grad():
    running = torch.zeros(1,1,device=DEVICE)

    for i in range(N-1, -1, -1):
        if i % 300 == 299:
            running.zero_()

        running = rewards[i:i+1] + GAMMA * running
        returns[i:i+1] = running

print(
    "Return range:",
    returns.min().item(),
    "to",
    returns.max().item()
)

critic_steps = 8000

for i in range(critic_steps):
    idx = torch.randint(0, N, (batch_size,), device=DEVICE)

    q1 = critic1(states[idx], desired[idx], actions[idx])
    q2 = critic2(states[idx], desired[idx], actions[idx])

    loss1 = F.smooth_l1_loss(q1, returns[idx])
    loss2 = F.smooth_l1_loss(q2, returns[idx])

    critic_opt1.zero_grad(set_to_none=True)
    loss1.backward()
    torch.nn.utils.clip_grad_norm_(critic1.parameters(), 1.0)
    critic_opt1.step()

    critic_opt2.zero_grad(set_to_none=True)
    loss2.backward()
    torch.nn.utils.clip_grad_norm_(critic2.parameters(), 1.0)
    critic_opt2.step()

    if (i + 1) % 1000 == 0:
        print(
            f"Critic {i+1:5d}/{critic_steps} "
            f"Q1={loss1.item():.5f} Q2={loss2.item():.5f}"
        )

# ------------------------------------------------------------
# STAGE 3 — ACTUAL SAC
# ------------------------------------------------------------
print("\nSTAGE 3 — ACTUAL SAC")

# Very small actor learning rate so SAC improves rather than destroys
# the expert policy.
actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-7)
critic_opt1 = torch.optim.Adam(critic1.parameters(), lr=5e-5)
critic_opt2 = torch.optim.Adam(critic2.parameters(), lr=5e-5)

log_alpha = torch.tensor(
    np.log(0.0001),
    dtype=torch.float32,
    device=DEVICE,
    requires_grad=True
)

alpha_opt = torch.optim.Adam([log_alpha], lr=1e-7)

TARGET_ENTROPY = -3.0
BC_LAMBDA = 100.0

sac_steps = 5000

for i in range(sac_steps):

    idx = torch.randint(0, N, (batch_size,), device=DEVICE)

    s = states[idx]
    d = desired[idx]
    a = actions[idx]
    ns = next_states[idx]
    r = rewards[idx]

    alpha = log_alpha.exp().detach()

    # --------------------------------------------------------
    # SAC target
    # --------------------------------------------------------
    with torch.no_grad():

        na, nlogp = actor(ns, d)

        tq1 = target1(ns, d, na)
        tq2 = target2(ns, d, na)

        tq = torch.min(tq1, tq2) - alpha * nlogp

        target = r + GAMMA * tq

        target = torch.clamp(target, -100.0, 10.0)

    q1 = critic1(s,d,a)
    q2 = critic2(s,d,a)

    critic_loss = (
        F.smooth_l1_loss(q1,target)
        + F.smooth_l1_loss(q2,target)
    )

    critic_opt1.zero_grad(set_to_none=True)
    critic_opt2.zero_grad(set_to_none=True)

    critic_loss.backward()

    torch.nn.utils.clip_grad_norm_(critic1.parameters(),1.0)
    torch.nn.utils.clip_grad_norm_(critic2.parameters(),1.0)

    critic_opt1.step()
    critic_opt2.step()

    # --------------------------------------------------------
    # SAC actor
    # --------------------------------------------------------
    new_a, logp = actor(s,d)

    q1_new = critic1(s,d,new_a)
    q2_new = critic2(s,d,new_a)

    q_new = torch.min(q1_new,q2_new)

    sac_actor_loss = (alpha * logp - q_new).mean()

    # Keep actor close to expert behavior.
    expert_action = actor.deterministic(s,d)
    bc_loss = F.mse_loss(expert_action,a)

    actor_loss = sac_actor_loss + BC_LAMBDA * bc_loss

    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()

    torch.nn.utils.clip_grad_norm_(actor.parameters(),0.25)
    actor_opt.step()

    # --------------------------------------------------------
    # Alpha
    # --------------------------------------------------------
    alpha_loss = -(log_alpha * (logp + TARGET_ENTROPY).detach()).mean()

    alpha_opt.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_opt.step()

    log_alpha.data.clamp_(np.log(1e-5), np.log(0.01))

    # --------------------------------------------------------
    # Target networks
    # --------------------------------------------------------
    with torch.no_grad():
        for p,t in zip(critic1.parameters(),target1.parameters()):
            t.mul_(1-TAU)
            t.add_(TAU*p)

        for p,t in zip(critic2.parameters(),target2.parameters()):
            t.mul_(1-TAU)
            t.add_(TAU*p)

    if (i+1) % 500 == 0:
        print(
            f"SAC {i+1:5d}/{sac_steps} "
            f"critic={critic_loss.item():.6f} "
            f"actor={actor_loss.item():.6f} "
            f"BC={bc_loss.item():.8f} "
            f"alpha={log_alpha.exp().item():.7f}"
        )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
torch.save({
    "actor": actor.state_dict(),
    "critic1": critic1.state_dict(),
    "critic2": critic2.state_dict(),
    "target1": target1.state_dict(),
    "target2": target2.state_dict(),
    "log_alpha": log_alpha.detach(),
    "algorithm": "SAC",
    "state_dim": 6,
    "desired_dim": 6,
    "action_dim": 3,
    "max_action": 2.0,
    "gamma": GAMMA,
    "tau": TAU
}, OUT)

print("\n" + "="*70)
print("SAVED:", OUT)
print("="*70)
