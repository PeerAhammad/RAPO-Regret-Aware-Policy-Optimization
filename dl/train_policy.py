import sys; sys.path.insert(0, '.')
import numpy as np, torch, torch.nn as nn
from safety_gym_env import GridSafetyEnv
from dl.policy_env import ACTIONS, LearnedPolicyEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(SEED); np.random.seed(SEED)
env = GridSafetyEnv(grid_size=8); N = env.N
net = nn.Sequential(nn.Linear(N * N, 128), nn.ReLU(), nn.Linear(128, 5))
opt = torch.optim.Adam(net.parameters(), lr=3e-3)

def mask_of(cell):
    r, c = cell
    return torch.tensor([0 <= r + dr < N and 0 <= c + dc < N for dr, dc in ACTIONS])

def onehot(cell):
    x = torch.zeros(N * N); x[cell[0] * N + cell[1]] = 1.0
    return x

def dist(cell):
    logits = net(onehot(cell)).masked_fill(~mask_of(cell), -1e9)
    return torch.distributions.Categorical(logits=logits)

baseline = 0.0
for ep in range(3000):
    cell = env.start; logps, ents, rewards = [], [], []
    for t in range(40):
        d = dist(cell); a = d.sample()
        dr, dc = ACTIONS[a.item()]; nxt = (cell[0] + dr, cell[1] + dc)
        done = nxt == env.goal
        logps.append(d.log_prob(a)); ents.append(d.entropy())
        rewards.append(1.0 if done else -0.01)   # NO hazard term: policy is harm-blind
        cell = nxt
        if done: break
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + 0.98 * G; returns.append(G)
    returns = torch.tensor(returns[::-1])
    baseline = 0.99 * baseline + 0.01 * returns[0].item()
    loss = -(torch.stack(logps) * (returns - baseline)).sum() - 0.02 * torch.stack(ents).sum()
    opt.zero_grad(); loss.backward(); opt.step()
    if ep % 500 == 0: print(ep, float(returns[0]))

# export frozen policy as a table
table = np.zeros((N * N, 5))
with torch.no_grad():
    for r in range(N):
        for c in range(N):
            table[r * N + c] = dist((r, c)).probs.numpy()
np.save(f'dl/policy_table_s{SEED}.npy', table)

# sanity: success and hazard-crossing rate of the frozen policy
e = LearnedPolicyEnv(table); rng = np.random.default_rng(1)
succ = cross = 0; E = 500
for _ in range(E):
    cell = e.start; crossed = ok = False
    for t in range(40):
        cell, h = e.step(cell, True, np.ones(N * N), rng, False)
        crossed |= h > 0
        if cell == e.goal: ok = True; break
    succ += ok; cross += crossed
print(f"success={succ/E:.2f}  crossed_hazard={cross/E:.2f}")