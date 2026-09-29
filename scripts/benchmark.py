"""Held-out benchmark: 200 unseen cube positions, stage scores, retries.

Report versions with this; debug with eval_bc.py / diag_collapse.py on the
old 50-position set. These positions (rng seed 2026) are never used for
debugging.

Per position the policy gets up to --attempts tries of --max-steps each. The
sim and policy are deterministic, so a retry from the same start would replay
the same failure. A retry therefore starts from where the last attempt left
things: the arm is returned home with the gripper open and the cube stays
wherever it ended up (pushed, dropped, tilted).

Stages, each worth one point (furthest stage reached counts):

  reach  TCP within 15 mm of the cube centre in xy and 20 mm in z
  close  gripper commanded closed while at the grasp pose
  grasp  finger opening 30-50 mm while at the grasp pose
  rise   cube centre 10 mm above its resting height
  lift   cube centre 100 mm above its resting height (env.lifted())
  hold   lift sustained for 50 steps (1 s) with both fingers on the cube

Metrics:

  stage score  mean over positions of stages reached on attempt 1, out of 6
  success@1    hold reached on the first attempt
  success@K    hold reached within K attempts
  attempts     mean attempts used, over positions that succeeded
  lifted@1     old lenient metric (lift on attempt 1), for comparison

  python benchmark.py ../models/bc_clean3000_both_bce_time_s0.pt --label v3.0
"""

import argparse
import datetime
import os
import subprocess
import sys

import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import FrankaCubeLift, TABLE_HEIGHT, CUBE_HALF  # noqa: E402
from collect_demos import GRIPPER_SCALE  # noqa: E402
from train_bc import MLPPolicy, policy_obs, GRIP_THRESHOLD  # noqa: E402

DEC = 10
HELDOUT_SEED = 2026
STAGES = ["reach", "close", "grasp", "rise", "lift", "hold"]
REST_Z = TABLE_HEIGHT + CUBE_HALF
HOLD_STEPS = 50
SETTLE_STEPS = 25  # control steps with the arm at home before a retry
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                       "benchmark.tsv")


def heldout_positions(n=200):
    rng = np.random.default_rng(HELDOUT_SEED)
    return [np.array([0.55, 0.0]) + rng.uniform([-0.08, -0.15], [0.08, 0.15])
            for _ in range(n)]


class Agent:
    """A saved BC checkpoint plus the per-attempt state its inputs need."""

    def __init__(self, path, device, raw_grip=False):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.policy = MLPPolicy(ck["obs_dim"], ck["act_dim"], hidden=ck["hidden"]).to(device)
        self.policy.load_state_dict(ck["state_dict"])
        self.policy.eval()
        self.norm = ck["norm"]
        self.frame = ck.get("frame", "world")
        self.grip_loss = ck.get("grip_loss", "mse")
        self.time_feature = ck.get("time_feature", False)
        # raw_grip reproduces the v1.0 rollout: the MSE gripper output sent as-is.
        self.raw_grip = raw_grip
        self.device = device
        self.n_closed = 0

    def reset(self):
        self.n_closed = 0

    def act(self, env):
        om, osd, am, asd = self.norm
        o = policy_obs(env, self.frame, self.time_feature, self.n_closed)
        with torch.no_grad():
            x = torch.as_tensor((o - om) / osd, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            a = self.policy(x).squeeze(0).cpu().numpy() * asd + am
        arm = np.clip(env.arm_qpos + a[:7], env.arm_limits[:, 0], env.arm_limits[:, 1])
        if self.raw_grip:
            g = float(np.clip(a[7], 0.0, 1.0))
        else:
            g = float(a[7] > GRIP_THRESHOLD[self.grip_loss])
        closed = g < 0.5
        self.n_closed = self.n_closed + 1 if closed else 0
        return arm, g * GRIPPER_SCALE, closed


def fingers_on_cube(env):
    m, d = env.model, env.data
    touching = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if env.cube_id in (b1, b2):
            other = b2 if b1 == env.cube_id else b1
            touching.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other))
    return {"left_finger", "right_finger"} <= touching


def return_home(env):
    """Arm and fingers back to the start pose; the cube keeps its current state."""
    adr = env.cube_qpos_adr
    dadr = env.cube_dof_adr
    cube_q = env.data.qpos[adr : adr + 7].copy()
    cube_v = env.data.qvel[dadr : dadr + 6].copy()
    env.reset()
    env.data.qpos[adr : adr + 7] = cube_q
    env.data.qvel[dadr : dadr + 6] = cube_v
    mujoco.mj_forward(env.model, env.data)
    home = env.arm_qpos
    for _ in range(SETTLE_STEPS):  # let a released cube fall and settle
        env.step(home, GRIPPER_SCALE, n_substeps=DEC)


def attempt(env, agent, max_steps):
    """One try from the current state. Returns the index of the furthest stage."""
    agent.reset()
    best, held = -1, 0
    for _ in range(max_steps):
        arm, grip, closed = agent.act(env)
        env.step(arm, grip, n_substeps=DEC)

        rel = env.cube_pos - env.tcp_pos
        at_pose = np.linalg.norm(rel[:2]) < 0.015 and abs(rel[2]) < 0.020
        lifted = env.lifted()
        held = held + 1 if lifted and fingers_on_cube(env) else 0
        hits = [
            at_pose,
            at_pose and closed,
            at_pose and 0.030 < env.gripper_width < 0.050,
            env.cube_pos[2] > REST_Z + 0.010,
            lifted,
            held >= HOLD_STEPS,
        ]
        best = max([best] + [i for i, h in enumerate(hits) if h])
        if best == len(STAGES) - 1:
            break
    return best


def evaluate(env, agent, positions, attempts, max_steps):
    rows = []
    for xy in positions:
        env.reset(cube_xy=xy)
        first, used, success = None, attempts, False
        for k in range(attempts):
            if k > 0:
                if env.cube_pos[2] < TABLE_HEIGHT - 0.05:  # fell off the table
                    break
                return_home(env)
            best = attempt(env, agent, max_steps)
            if first is None:
                first = best
            if best == len(STAGES) - 1:
                used, success = k + 1, True
                break
        rows.append((first, success, used))
    return rows


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(RESULTS), text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--label", type=str, default="",
                    help="version label for the results row, e.g. v3.0")
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--raw-grip", action="store_true",
                    help="send the MSE gripper output unsnapped (v1.0 rollout)")
    ap.add_argument("--no-log", action="store_true", help="don't append to results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    positions = heldout_positions(args.positions)
    env = FrankaCubeLift()

    for path in args.checkpoints:
        agent = Agent(path, device, raw_grip=args.raw_grip)
        rows = evaluate(env, agent, positions, args.attempts, args.max_steps)
        first = np.array([r[0] for r in rows])
        success = np.array([r[1] for r in rows])
        used = np.array([r[2] for r in rows])

        stage_score = 100 * np.mean((first + 1) / len(STAGES))
        s1 = 100 * np.mean(success & (used == 1))
        sk = 100 * np.mean(success)
        mean_att = float(np.mean(used[success])) if success.any() else float("nan")
        lifted1 = 100 * np.mean(first >= STAGES.index("lift"))
        stage_rates = [100 * np.mean(first >= i) for i in range(len(STAGES))]
        att_hist = [int(np.sum(success & (used == k))) for k in range(1, args.attempts + 1)]

        name = os.path.basename(path)
        print(f"\n== {name}  {args.label}  (frame={agent.frame}, grip={agent.grip_loss}"
              f"{', raw' if args.raw_grip else ''}, time={agent.time_feature})")
        print(f"   stage score {stage_score:5.1f}   success@1 {s1:5.1f}%   "
              f"success@{args.attempts} {sk:5.1f}%   attempts {mean_att:.2f}   "
              f"lifted@1 {lifted1:5.1f}%")
        print("   attempt 1 stage rates: " + "  ".join(
            f"{s} {r:.0f}%" for s, r in zip(STAGES, stage_rates)))
        print(f"   succeeded on attempt 1/2/…: {att_hist}   failed: {int((~success).sum())}")

        if not args.no_log:
            new = not os.path.exists(RESULTS)
            with open(RESULTS, "a") as f:
                if new:
                    f.write("date\tcommit\tlabel\tcheckpoint\tframe\tgrip_loss\traw_grip\t"
                            "time_feature\tpositions\tattempts\tstage_score\tsuccess_1\t"
                            "success_k\tmean_attempts\tlifted_1\t"
                            + "\t".join(f"rate_{s}" for s in STAGES) + "\n")
                f.write("\t".join(map(str, [
                    datetime.date.today().isoformat(), git_commit(), args.label, name,
                    agent.frame, agent.grip_loss, args.raw_grip, agent.time_feature,
                    args.positions, args.attempts, f"{stage_score:.1f}", f"{s1:.1f}",
                    f"{sk:.1f}", f"{mean_att:.2f}", f"{lifted1:.1f}",
                    *[f"{r:.1f}" for r in stage_rates]])) + "\n")
    env.close()


if __name__ == "__main__":
    main()
