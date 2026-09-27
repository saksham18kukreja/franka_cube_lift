"""Where do collapsed BC runs fail?

Rolls each checkpoint out on the eval cube positions (rng seed 1234, same as
train_bc.py) and records the furthest stage every episode reaches:

  reach   TCP at the grasp pose: within 15 mm of the cube centre in xy, 20 mm in z
  close   gripper commanded closed (grip < 0.5) while at the grasp pose
  grasp   fingers stopped on the cube (width 30-50 mm) while at the grasp pose
  rise    cube centre 10 mm above its resting height
  lifted  env.lifted() (100 mm)

The first stage an episode misses is its failure mode.

  python diag_collapse.py ../models/bc_clean3000.pt other.pt ...
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import FrankaCubeLift, TABLE_HEIGHT, CUBE_HALF  # noqa: E402
from collect_demos import GRIPPER_SCALE  # noqa: E402
from train_bc import MLPPolicy, policy_obs  # noqa: E402

DEC = 10
STAGES = ["reach", "close", "grasp", "rise", "lifted"]
REST_Z = TABLE_HEIGHT + CUBE_HALF


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    pol = MLPPolicy(ck["obs_dim"], ck["act_dim"], hidden=ck["hidden"]).to(device)
    pol.load_state_dict(ck["state_dict"])
    pol.eval()
    # Checkpoints trained with --grip-loss bce emit a logit; map it to [0, 1].
    squash = ck.get("grip_loss", "mse") == "bce"
    return pol, ck["norm"], ck.get("frame", "world"), squash, ck.get("time_feature", False)


def episode(env, pol, norm, frame, cube_xy, device, max_steps=700, squash=False,
            time_feature=False):
    env.reset(cube_xy=cube_xy)
    om, osd, am, asd = norm
    first = {s: None for s in STAGES}
    log, n_closed = [], 0
    for t in range(max_steps):
        o = policy_obs(env, frame, time_feature, n_closed)
        with torch.no_grad():
            x = torch.as_tensor((o - om) / osd, dtype=torch.float32,
                                device=device).unsqueeze(0)
            a = pol(x).squeeze(0).cpu().numpy() * asd + am
        tgt = np.clip(env.arm_qpos + a[:7], env.arm_limits[:, 0], env.arm_limits[:, 1])
        g = float(0.5 * (1 + np.tanh(a[7] / 2))) if squash else float(np.clip(a[7], 0.0, 1.0))
        env.step(tgt, g * GRIPPER_SCALE, n_substeps=DEC)
        n_closed = n_closed + 1 if g < 0.5 else 0

        rel = env.cube_pos - env.tcp_pos
        at_pose = np.linalg.norm(rel[:2]) < 0.015 and abs(rel[2]) < 0.020
        w = env.gripper_width
        hits = {
            "reach": at_pose,
            "close": at_pose and g < 0.5,
            "grasp": at_pose and 0.030 < w < 0.050,
            "rise": env.cube_pos[2] > REST_Z + 0.010,
            "lifted": env.lifted(),
        }
        for s in STAGES:
            if hits[s] and first[s] is None:
                first[s] = t
        log.append((t, np.linalg.norm(rel[:2]), rel[2], g, w, env.cube_pos[2],
                    np.linalg.norm(env.cube_pos[:2] - cube_xy), np.abs(a[:7]).sum()))
        if hits["lifted"]:
            break

    # Stages must be reached in order; the first gap is the failure mode.
    reached = 0
    for s in STAGES:
        if first[s] is None:
            break
        reached += 1
    return reached, first, np.array(log)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(1234)
    positions = [np.array([0.55, 0.0]) + rng.uniform([-0.08, -0.15], [0.08, 0.15])
                 for _ in range(50)]
    env = FrankaCubeLift()

    for path in sys.argv[1:]:
        pol, norm, frame, squash, tf = load(path, device)
        out = [episode(env, pol, norm, frame, xy, device, squash=squash, time_feature=tf)
               for xy in positions]
        reached = np.array([r for r, _, _ in out])
        fails = {s: int((reached == i).sum()) for i, s in enumerate(STAGES)}

        print(f"\n== {os.path.basename(path)}  (frame={frame})  "
              f"success {100*np.mean(reached == len(STAGES)):.0f}%")
        print("   failed before: " + "  ".join(f"{s} {n}" for s, n in fails.items() if n))

        # Timing of each stage among episodes that reached it.
        for s in STAGES:
            ts = [f[s] for _, f, _ in out if f[s] is not None]
            if ts:
                print(f"   {s:6s} reached {len(ts):2d}/50  step median {np.median(ts):5.0f}")

        # Snapshot of the failed episodes at the end of the rollout.
        bad = [(xy, r, lg) for xy, (r, _, lg) in zip(positions, out) if r < len(STAGES)]
        if bad:
            ends = np.array([lg[-1] for _, _, lg in bad])
            print(f"   failed eps at step 700: xy_err {np.median(ends[:,1])*1000:.1f} mm  "
                  f"dz {np.median(ends[:,2])*1000:+.1f} mm  grip {np.median(ends[:,3]):.2f}  "
                  f"width {np.median(ends[:,4])*1000:.1f} mm  "
                  f"cube pushed {np.median(ends[:,6])*1000:.1f} mm  "
                  f"|dq| {np.median(ends[:,7]):.4f}")
    env.close()


if __name__ == "__main__":
    main()
