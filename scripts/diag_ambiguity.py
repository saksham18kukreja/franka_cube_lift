"""Is the expert a function of the observation, or does it carry hidden state?

For sampled states, find nearest neighbours in *other* episodes and measure how
much the expert's action disagrees. Large disagreement at near-zero observation
distance means the demos are ambiguous and MSE will average incompatible modes.
"""
import numpy as np, torch

d = np.load("../demos/state_demos.npz", allow_pickle=True)
obs, act = d["obs"], d["act"]
starts, lengths = d["episode_starts"], d["episode_lengths"]

ep_id = np.zeros(len(obs), dtype=np.int64)
for i, (s, L) in enumerate(zip(starts, lengths)):
    ep_id[s:s+L] = i

om, osd = obs.mean(0), obs.std(0) + 1e-6
X = torch.as_tensor((obs - om)/osd, device="cuda")
A = torch.as_tensor(act, device="cuda")
E = torch.as_tensor(ep_id, device="cuda")

rng = np.random.default_rng(0)
q = torch.as_tensor(rng.choice(len(obs), 1500, replace=False), device="cuda")

dists, disagree, self_scale = [], [], []
for i in range(0, len(q), 256):
    qi = q[i:i+256]
    dmat = torch.cdist(X[qi], X)
    dmat[E[qi][:, None] == E[None, :]] = float("inf")   # other episodes only
    dd, nn = dmat.min(dim=1)
    dists.append(dd.cpu().numpy())
    disagree.append((A[qi] - A[nn]).abs().cpu().numpy())
disc = np.concatenate(disagree); dist = np.concatenate(dists)

print(f"nearest-neighbour obs distance (normalised): "
      f"median {np.median(dist):.4f}  p10 {np.percentile(dist,10):.4f}")
print(f"action std across whole dataset : dq {act[:,:7].std():.5f}  grip {act[:,7].std():.4f}")
print(f"action disagreement at NN       : dq {disc[:,:7].mean():.5f}  grip {disc[:,7].mean():.4f}")

close = dist < np.percentile(dist, 10)
print(f"\nfor the 10% CLOSEST matches (obs distance < {np.percentile(dist,10):.4f}):")
print(f"  dq disagreement   {disc[close][:,:7].mean():.5f} rad")
print(f"  grip disagreement {disc[close][:,7].mean():.4f}   (0=identical, 1=opposite)")
frac = (disc[close][:,7] > 0.5).mean()
print(f"  fraction with OPPOSITE gripper command: {100*frac:.1f}%")
