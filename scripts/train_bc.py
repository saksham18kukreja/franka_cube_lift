"""Behaviour cloning on the scripted expert's demonstrations.

Trains an MLP to map state -> joint-delta action, then rolls the policy out in
the environment. The rollout number is the one that matters: validation MSE
measures how well we match the expert frame by frame, but a policy can score
well there and still fail the task, because errors compound once the policy is
driving the state distribution itself.

  python train_bc.py --demos ../demos/state_demos.npz --epochs 100
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import FrankaCubeLift  # noqa: E402
from collect_demos import observe, GRIPPER_SCALE  # noqa: E402

DEC = 10
GRIP_THRESHOLD = {"mse": 0.5, "bce": 0.0}  # open if above (label 1 = open)

# The expert holds the closed gripper still for exactly 120 steps and then
# lifts. That timer is invisible in a single frame, so --time-feature appends
# how many consecutive steps the gripper has been commanded closed. Capped past
# the expert's longest close phase (150) and scaled to roughly [0, 2].
CLOSED_CAP = 200


def closed_feature(n_closed):
    return np.minimum(n_closed, CLOSED_CAP) / 100.0


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, depth=2):
        super().__init__()
        layers, d = [], obs_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, act_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def frame_features(obs, tcp_mat, frame):
    """Re-express the 17-d world-frame observation.

      world    unchanged
      gripper  arm_q, grip_width, cube position in the TCP frame (11-d)
      both     world features + cube position in the TCP frame (20-d)
    """
    if frame == "world":
        return obs
    obs, tcp_mat = np.atleast_2d(obs), np.asarray(tcp_mat).reshape(-1, 3, 3)
    rel_g = np.einsum("nji,nj->ni", tcp_mat, obs[:, 14:17])  # R^T (cube - tcp)
    base = obs[:, :8] if frame == "gripper" else obs
    return np.concatenate([base, rel_g], axis=1).astype(np.float32)


def demo_tcp_mats(obs):
    """TCP orientation for each stored frame, from the recorded joint angles."""
    env = FrankaCubeLift()
    mats = np.empty((len(obs), 3, 3), dtype=np.float32)
    for i, q in enumerate(obs[:, :7]):
        env.data.qpos[env.arm_qpos_adr] = q
        mujoco.mj_kinematics(env.model, env.data)
        mats[i] = env.tcp_mat
    env.close()
    return mats


def demo_closed_counts(act, starts, lengths):
    """Consecutive closed commands issued *before* each step, per episode."""
    counts = np.zeros(len(act), dtype=np.float32)
    for s, L in zip(starts, lengths):
        closed = act[s : s + L, 7] < 0.5          # label 0 = closed
        c = 0
        for t in range(L):
            counts[s + t] = c
            c = c + 1 if closed[t] else 0
    return counts


def policy_obs(env, frame, time_feature, n_closed, cube_pos=None):
    """The observation the policy sees at rollout, matching load_demos.

    cube_pos overrides the simulator's cube position (e.g. a camera estimate);
    every cube-derived feature is recomputed from it.
    """
    o = observe(env)
    if cube_pos is not None:
        o[11:14] = cube_pos
        o[14:17] = cube_pos - o[8:11]  # cube - tcp
    o = frame_features(o, env.tcp_mat, frame).reshape(-1)
    if time_feature:
        o = np.append(o, closed_feature(n_closed)).astype(np.float32)
    return o


def load_demos(path, val_frac=0.1, seed=0, frame="world", time_feature=False):
    """Split by episode, not by transition: frames inside an episode are highly
    correlated, so a random transition split leaks the validation set."""
    d = np.load(path, allow_pickle=True)
    obs, act = d["obs"], d["act"]
    if frame != "world":
        obs = frame_features(obs, demo_tcp_mats(obs), frame)
    starts, lengths = d["episode_starts"], d["episode_lengths"]
    if time_feature:
        feat = closed_feature(demo_closed_counts(act, starts, lengths))
        obs = np.concatenate([obs, feat[:, None]], axis=1).astype(np.float32)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lengths))
    n_val = max(1, int(len(lengths) * val_frac))
    val_eps, train_eps = order[:n_val], order[n_val:]

    def gather(eps):
        idx = np.concatenate([np.arange(starts[e], starts[e] + lengths[e]) for e in eps])
        return obs[idx], act[idx]

    return gather(train_eps), gather(val_eps), len(train_eps), len(val_eps)


def rollout_policy(env, policy, norm, cube_xy, device, max_steps=700, frame="world",
                   grip_loss="mse", time_feature=False):
    """Run the learned policy closed-loop. No expert, no privileged targets."""
    env.reset(cube_xy=cube_xy)
    obs_mean, obs_std, act_mean, act_std = norm
    n_closed = 0  # from the policy's own past gripper commands

    for _ in range(max_steps):
        o = policy_obs(env, frame, time_feature, n_closed)
        with torch.no_grad():
            x = torch.as_tensor((o - obs_mean) / obs_std, dtype=torch.float32,
                                device=device).unsqueeze(0)
            a = policy(x).squeeze(0).cpu().numpy() * act_std + act_mean

        arm_target = np.clip(
            env.arm_qpos + a[:7], env.arm_limits[:, 0], env.arm_limits[:, 1]
        )
        # The expert's gripper is binary. MSE smooths the open->close step and
        # a half-closed command mid-descent drives the policy off-distribution
        # (see diag_collapse.py), so snap it. With bce, a[7] is a logit.
        grip = float(a[7] > GRIP_THRESHOLD[grip_loss]) * GRIPPER_SCALE
        env.step(arm_target, grip, n_substeps=DEC)
        n_closed = n_closed + 1 if grip == 0 else 0

        if env.lifted():
            return True
    return env.lifted()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=str, default="../demos/state_demos.npz")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="../models/bc_policy.pt")
    ap.add_argument("--frame", choices=["world", "gripper", "both"], default="world",
                    help="coordinate frame for the observation (see frame_features)")
    ap.add_argument("--grip-loss", choices=["mse", "bce"], default="mse",
                    help="mse: regress the gripper like the joints; bce: classify "
                         "open/closed from a logit")
    ap.add_argument("--grip-weight", type=float, default=1.0,
                    help="weight of the bce gripper term relative to the joint mse")
    ap.add_argument("--time-feature", action="store_true",
                    help="append steps-since-gripper-closed to the observation")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    (tr_o, tr_a), (va_o, va_a), n_tr, n_va = load_demos(args.demos, seed=args.seed,
                                                     frame=args.frame,
                                                     time_feature=args.time_feature)
    print(f"train {tr_o.shape[0]} transitions / {n_tr} episodes | "
          f"val {va_o.shape[0]} / {n_va} episodes")

    # Normalise from the training split only.
    obs_mean, obs_std = tr_o.mean(0), tr_o.std(0) + 1e-6
    act_mean, act_std = tr_a.mean(0), tr_a.std(0) + 1e-6
    if args.grip_loss == "bce":
        # Leave the 0/1 gripper label unnormalised so output 7 is a plain logit.
        act_mean[7], act_std[7] = 0.0, 1.0
    norm = (obs_mean, obs_std, act_mean, act_std)

    to = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    Xtr, Ytr = to((tr_o - obs_mean) / obs_std), to((tr_a - act_mean) / act_std)
    Xva, Yva = to((va_o - obs_mean) / obs_std), to((va_a - act_mean) / act_std)

    policy = MLPPolicy(tr_o.shape[1], tr_a.shape[1], hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()

    def lossf(pred, y):
        if args.grip_loss == "mse":
            return mse(pred, y)
        return mse(pred[:, :7], y[:, :7]) + args.grip_weight * bce(pred[:, 7], y[:, 7])

    def grip_acc(pred, y):
        g = pred[:, 7] * act_std[7] + act_mean[7]
        label = y[:, 7] * act_std[7] + act_mean[7]
        return ((g > GRIP_THRESHOLD[args.grip_loss]) == (label > 0.5)).float().mean().item()

    n = Xtr.shape[0]
    t0 = time.time()
    for ep in range(args.epochs):
        policy.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            opt.zero_grad(set_to_none=True)
            loss = lossf(policy(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()

        if (ep + 1) % 10 == 0 or ep == 0:
            policy.eval()
            with torch.no_grad():
                pv = policy(Xva)
                val = lossf(pv, Yva).item()
                val_dq = mse(pv[:, :7], Yva[:, :7]).item()
                acc = grip_acc(pv, Yva)
            print(f"epoch {ep+1:4d}  train {total/n:.5f}  val {val:.5f}  "
                  f"dq {val_dq:.5f}  grip_acc {100*acc:.2f}%  "
                  f"({time.time()-t0:.0f}s)")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "norm": norm,
                "obs_dim": tr_o.shape[1], "act_dim": tr_a.shape[1],
                "hidden": args.hidden, "frame": args.frame,
                "grip_loss": args.grip_loss, "time_feature": args.time_feature}, out)
    print(f"saved {out}")

    # The number that actually matters.
    print(f"\nrolling out {args.eval_episodes} episodes...")
    policy.eval()
    env = FrankaCubeLift()
    rng = np.random.default_rng(1234)
    wins = []
    for _ in range(args.eval_episodes):
        xy = np.array([0.55, 0.0]) + rng.uniform([-0.08, -0.15], [0.08, 0.15])
        wins.append(rollout_policy(env, policy, norm, xy, device, frame=args.frame,
                                   grip_loss=args.grip_loss,
                                   time_feature=args.time_feature))
    env.close()
    print(f"BC policy success: {100*np.mean(wins):.1f}%  "
          f"({sum(wins)}/{len(wins)})    [expert: 100%]")


if __name__ == "__main__":
    main()
