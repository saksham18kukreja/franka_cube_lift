"""Generate imitation-learning demonstrations with the scripted expert.

The expert is privileged: it reads the exact cube pose out of the simulator.
The observation we log is what a student policy is allowed to see, which for
now is still state -- but the point of logging them separately is that the
expert's advantage is explicit, not baked into the dataset.

  python collect_demos.py --episodes 300 --out ../demos/state_demos.npz
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import ScriptedGrasp  # noqa: E402
from env import FrankaCubeLift  # noqa: E402

DEC = 10
GRIPPER_SCALE = 255.0

OBS_LABELS = (
    [f"arm_q{i+1}" for i in range(7)]
    + ["grip_width"]
    + ["tcp_x", "tcp_y", "tcp_z"]
    + ["cube_x", "cube_y", "cube_z"]
    + ["rel_x", "rel_y", "rel_z"]
)
ACT_LABELS = [f"dq{i+1}" for i in range(7)] + ["grip"]


def observe(env):
    """Student-visible state. 17-d."""
    tcp, cube = env.tcp_pos, env.cube_pos
    return np.concatenate(
        [env.arm_qpos, [env.gripper_width], tcp, cube, cube - tcp]
    ).astype(np.float32)


def collect_episode(env, cube_xy, max_steps=1000, noise=0.0, rng=None):
    """Roll the expert. With noise>0, execute a perturbed action but label
    the state with the expert's *correct* action, so the dataset contains
    recovery behaviour off the nominal path (DART-style injection)."""
    env.reset(cube_xy=cube_xy)
    pol = ScriptedGrasp(env)

    obs_list, act_list, phase_list = [], [], []
    for _ in range(max_steps):
        obs = observe(env)
        q_before = env.arm_qpos

        arm_target, grip = pol.act()
        phase = pol.phase

        # Action is a joint-space delta: zero-centred and small, which trains
        # far better than regressing absolute joint targets.
        action = np.concatenate(
            [arm_target - q_before, [grip / GRIPPER_SCALE]]
        ).astype(np.float32)

        obs_list.append(obs)
        act_list.append(action)
        phase_list.append(phase)

        exec_target = arm_target
        if noise > 0.0:
            exec_target = np.clip(
                arm_target + rng.normal(0.0, noise, size=7),
                env.arm_limits[:, 0], env.arm_limits[:, 1])
        env.step(exec_target, grip, n_substeps=DEC)
        if pol.done:
            break

    return {
        "obs": np.array(obs_list),
        "act": np.array(act_list),
        "phases": np.array(phase_list),
        "success": env.lifted(),
        "timed_out": pol.timed_out,
        "cube_xy": np.asarray(cube_xy, dtype=np.float32),
        "cube_z": env.cube_height(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", type=str, default="../demos/state_demos.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="std [rad] of joint-target noise during execution")
    ap.add_argument("--keep-failures", action="store_true",
                    help="store unsuccessful episodes too (off by default)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = FrankaCubeLift()

    obs_all, act_all, starts, lengths, meta = [], [], [], [], []
    n_fail = n_timeout = 0
    t0 = time.time()

    for i in range(args.episodes):
        xy = np.array([0.55, 0.0]) + rng.uniform([-0.08, -0.15], [0.08, 0.15])
        ep = collect_episode(env, xy, noise=args.noise, rng=rng)

        n_timeout += int(ep["timed_out"])
        if not ep["success"]:
            n_fail += 1
            if not args.keep_failures:
                continue

        starts.append(sum(lengths))
        lengths.append(len(ep["obs"]))
        obs_all.append(ep["obs"])
        act_all.append(ep["act"])
        meta.append((ep["cube_xy"][0], ep["cube_xy"][1],
                     float(ep["success"]), ep["cube_z"]))

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.episodes} episodes, "
                  f"{len(lengths)} kept, {n_fail} failed, {time.time()-t0:.0f}s")

    env.close()

    obs = np.concatenate(obs_all)
    act = np.concatenate(act_all)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        obs=obs, act=act,
        episode_starts=np.array(starts), episode_lengths=np.array(lengths),
        meta=np.array(meta, dtype=np.float32),
        obs_labels=np.array(OBS_LABELS), act_labels=np.array(ACT_LABELS),
    )

    print(f"\nepisodes requested {args.episodes}, kept {len(lengths)}, "
          f"failed {n_fail}, phase-timeouts {n_timeout}")
    print(f"transitions {len(obs)}  obs {obs.shape}  act {act.shape}")
    print(f"episode length: mean {np.mean(lengths):.0f} "
          f"min {np.min(lengths)} max {np.max(lengths)}")
    print(f"action |dq| max {np.abs(act[:, :7]).max():.4f} rad")
    print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
