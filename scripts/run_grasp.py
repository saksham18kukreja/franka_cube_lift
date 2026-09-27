"""Run the scripted grasp and report whether the cube was lifted.

  python run_grasp.py                 # single episode, cube at the default spot
  python run_grasp.py --video out.mp4 # also write a video
  python run_grasp.py --trials 20 --randomize
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import ScriptedGrasp  # noqa: E402
from env import FrankaCubeLift  # noqa: E402

CONTROL_DECIMATION = 10  # 500 Hz physics -> 50 Hz control


def rollout(env, cube_xy=None, randomize=False, max_steps=1200, render_every=None):
    """One episode. Returns a dict of outcome + logged trajectory."""
    env.reset(cube_xy=cube_xy, randomize=randomize)
    policy = ScriptedGrasp(env)

    frames, log = [], []
    for step in range(max_steps):
        arm_target, grip = policy.act()
        env.step(arm_target, grip, n_substeps=CONTROL_DECIMATION)

        log.append(
            {
                "phase": policy.phase,
                "tcp": env.tcp_pos,
                "cube": env.cube_pos,
                "width": env.gripper_width,
            }
        )
        if render_every and step % render_every == 0:
            frames.append(env.render())
        if policy.done:
            break

    return {
        "success": env.lifted(),
        "cube_height": env.cube_height(),
        "steps": step + 1,
        "final_phase": policy.phase,
        "grasp_error": float(np.linalg.norm(env.tcp_pos[:2] - env.cube_pos[:2])),
        "frames": frames,
        "log": log,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = FrankaCubeLift(seed=args.seed)
    print(f"model: nq={env.model.nq} nv={env.model.nv} nu={env.model.nu} dt={env.dt}")

    successes = []
    for trial in range(args.trials):
        want_video = args.video is not None and trial == 0
        res = rollout(
            env,
            randomize=args.randomize,
            render_every=1 if want_video else None,
        )
        successes.append(res["success"])
        print(
            f"trial {trial:3d}  success={str(res['success']):5s}  "
            f"cube_z={res['cube_height']:.3f}  steps={res['steps']:4d}  "
            f"phase={res['final_phase']:8s}  xy_err={res['grasp_error']*1000:5.1f} mm"
        )

        if want_video and res["frames"]:
            import imageio

            os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
            imageio.mimsave(args.video, res["frames"], fps=30)
            print(f"wrote {args.video} ({len(res['frames'])} frames)")

    rate = float(np.mean(successes))
    print(f"\nsuccess rate: {rate*100:.0f}%  ({sum(successes)}/{len(successes)})")
    env.close()
    return 0 if rate > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
