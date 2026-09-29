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

Perception (--perception) sets where the policy's cube position comes from:

  gt     the simulator's true position every step (the original setup)
  noisy  true position + Gaussian noise (--pos-noise, mm per axis) + a fixed
         offset (--pos-bias, mm, random direction per position), refreshed at
         --perception-hz and held in between. Used to find the error budget a
         detector must meet.
  color  table_cam RGB-D + perception.detectors.ColorDetector at
         --perception-hz. A rejected detection (cube too hidden) keeps the last
         good estimate; before the first good one, the workspace centre is used.

  python benchmark.py ../models/bc_clean3000_both_bce_time_s0.pt --label v3.0
  python benchmark.py <ckpt> --label v3.0 --perception noisy --pos-noise 5
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
PERCEPTION_COLS = ["perception", "pos_noise_mm", "pos_bias_mm", "perception_hz",
                   "pos_err_med_mm", "pos_err_p95_mm", "detect_reject_pct"]
SETTLE_STEPS = 25  # control steps with the arm at home before a retry
CONTROL_HZ = 50
WORKSPACE_CENTRE = np.array([0.55, 0.0, TABLE_HEIGHT + CUBE_HALF])
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                       "benchmark.tsv")


def heldout_positions(n=200):
    rng = np.random.default_rng(HELDOUT_SEED)
    return [np.array([0.55, 0.0]) + rng.uniform([-0.08, -0.15], [0.08, 0.15])
            for _ in range(n)]


class Perception:
    """Where the policy's cube position comes from. None means ground truth."""

    def __init__(self, mode="gt", noise_mm=0.0, bias_mm=0.0, hz=10, min_visible=0.6):
        self.mode = mode
        self.min_visible = min_visible
        self.noise = noise_mm / 1000.0
        self.bias_mag = bias_mm / 1000.0
        self.every = max(1, round(CONTROL_HZ / hz))
        self.errors = []
        self.calls = self.rejects = 0
        self.detector = None
        self.last = {}

    def reset_position(self, idx):
        # Seeded by position, so every checkpoint sees the same noise.
        self.rng = np.random.default_rng(HELDOUT_SEED * 1000 + idx)
        d = self.rng.normal(size=3)
        self.bias = self.bias_mag * d / np.linalg.norm(d)
        self.estimate = None

    def reset_attempt(self):
        self.t = 0

    def _detect(self, env):
        if self.detector is None:
            from perception.camera import Camera
            from perception.detectors import ColorDetector
            self.detector = ColorDetector(Camera(env), min_visible=self.min_visible)
        self.calls += 1
        p, info = self.detector.detect()
        if p is None:
            self.rejects += 1
            fallback = self.estimate is None
            self.last = dict(info, status="fallback" if fallback else "rejected")
            return WORKSPACE_CENTRE if fallback else self.estimate
        self.last = dict(info, status="accepted")
        return p

    def cube_pos(self, env):
        if self.mode == "gt":
            return None
        if self.t % self.every == 0 or self.estimate is None:
            if self.mode == "color":
                self.estimate = self._detect(env)
            else:
                self.estimate = env.cube_pos + self.bias + self.rng.normal(0, self.noise, 3)
        self.t += 1
        self.errors.append(np.linalg.norm(self.estimate - env.cube_pos))
        return self.estimate


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

    def act(self, env, cube_pos=None):
        om, osd, am, asd = self.norm
        o = policy_obs(env, self.frame, self.time_feature, self.n_closed, cube_pos)
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


def attempt(env, agent, perception, max_steps, on_step=None):
    """One try from the current state. Returns the index of the furthest stage.

    on_step(best, closed, n_closed) is called after every control step (used
    by make_bc_video.py).
    """
    agent.reset()
    perception.reset_attempt()
    best, held = -1, 0
    for _ in range(max_steps):
        arm, grip, closed = agent.act(env, perception.cube_pos(env))
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
        if on_step is not None:
            on_step(best, closed, agent.n_closed)
        if best == len(STAGES) - 1:
            break
    return best


def evaluate(env, agent, perception, positions, attempts, max_steps):
    rows = []
    for idx, xy in enumerate(positions):
        env.reset(cube_xy=xy)
        perception.reset_position(idx)
        first, used, success = None, attempts, False
        for k in range(attempts):
            if k > 0:
                if env.cube_pos[2] < TABLE_HEIGHT - 0.05:  # fell off the table
                    break
                return_home(env)
            best = attempt(env, agent, perception, max_steps)
            if first is None:
                first = best
            if best == len(STAGES) - 1:
                used, success = k + 1, True
                break
        rows.append((first, success, used))
    return rows


def perception_tag(args):
    if args.perception == "color" and args.min_visible != 0.6:
        return f"color_minvis{args.min_visible:g}"
    return args.perception


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
    ap.add_argument("--perception", choices=["gt", "noisy", "color"], default="gt",
                    help="source of the policy's cube position (see module doc)")
    ap.add_argument("--pos-noise", type=float, default=0.0,
                    help="noisy: Gaussian noise per axis, mm")
    ap.add_argument("--pos-bias", type=float, default=0.0,
                    help="noisy: fixed offset magnitude, mm")
    ap.add_argument("--perception-hz", type=float, default=10.0,
                    help="noisy: estimate refresh rate (control runs at 50 Hz)")
    ap.add_argument("--min-visible", type=float, default=0.6,
                    help="color: reject detections below this visible fraction "
                         "(0 accepts every detection)")
    ap.add_argument("--no-log", action="store_true", help="don't append to results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    positions = heldout_positions(args.positions)
    env = FrankaCubeLift()

    for path in args.checkpoints:
        agent = Agent(path, device, raw_grip=args.raw_grip)
        perception = Perception(args.perception, args.pos_noise, args.pos_bias,
                                args.perception_hz, args.min_visible)
        rows = evaluate(env, agent, perception, positions, args.attempts, args.max_steps)
        errs = np.array(perception.errors) * 1000 if perception.errors else np.zeros(1)
        err_med, err_p95 = np.median(errs), np.percentile(errs, 95)
        reject = 100 * perception.rejects / max(1, perception.calls)
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
              f"{', raw' if args.raw_grip else ''}, time={agent.time_feature}, "
              f"perception={args.perception} noise={args.pos_noise:g}mm "
              f"bias={args.pos_bias:g}mm @{args.perception_hz:g}Hz)")
        print(f"   stage score {stage_score:5.1f}   success@1 {s1:5.1f}%   "
              f"success@{args.attempts} {sk:5.1f}%   attempts {mean_att:.2f}   "
              f"lifted@1 {lifted1:5.1f}%")
        print("   attempt 1 stage rates: " + "  ".join(
            f"{s} {r:.0f}%" for s, r in zip(STAGES, stage_rates)))
        print(f"   succeeded on attempt 1/2/…: {att_hist}   failed: {int((~success).sum())}")
        if args.perception != "gt":
            print(f"   cube position error: median {err_med:.1f} mm  p95 {err_p95:.1f} mm"
                  + (f"   detections rejected {reject:.1f}%" if args.perception == "color" else ""))

        if not args.no_log:
            new = not os.path.exists(RESULTS)
            with open(RESULTS, "a") as f:
                if new:
                    f.write("date\tcommit\tlabel\tcheckpoint\tframe\tgrip_loss\traw_grip\t"
                            "time_feature\tpositions\tattempts\tstage_score\tsuccess_1\t"
                            "success_k\tmean_attempts\tlifted_1\t"
                            + "\t".join(f"rate_{s}" for s in STAGES)
                            + "\t" + "\t".join(PERCEPTION_COLS) + "\n")
                f.write("\t".join(map(str, [
                    datetime.date.today().isoformat(), git_commit(), args.label, name,
                    agent.frame, agent.grip_loss, args.raw_grip, agent.time_feature,
                    args.positions, args.attempts, f"{stage_score:.1f}", f"{s1:.1f}",
                    f"{sk:.1f}", f"{mean_att:.2f}", f"{lifted1:.1f}",
                    *[f"{r:.1f}" for r in stage_rates],
                    perception_tag(args), f"{args.pos_noise:g}", f"{args.pos_bias:g}",
                    f"{args.perception_hz:g}", f"{err_med:.2f}", f"{err_p95:.2f}",
                    f"{reject:.1f}"])) + "\n")
    env.close()


if __name__ == "__main__":
    main()
