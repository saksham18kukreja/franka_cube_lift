"""Render the scripted controller to an mp4.

Renders several randomized episodes back to back so the video shows the
controller handling different cube positions, not one lucky run.

  python make_video.py --episodes 4 --out ../videos/grasp.mp4
"""

import argparse
import os
import sys

import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import ScriptedGrasp  # noqa: E402
from env import FrankaCubeLift  # noqa: E402

DEC = 10          # 500 Hz physics -> 50 Hz control
FPS = 50          # render every control step => real time


def episode_frames(env, cube_xy, camera, width, height, max_steps=1200):
    env.reset(cube_xy=cube_xy)
    pol = ScriptedGrasp(env)
    frames = []
    for _ in range(max_steps):
        arm, grip = pol.act()
        env.step(arm, grip, n_substeps=DEC)
        frames.append(env.render(camera=camera, width=width, height=height))
        if pol.done:
            break
    # Hold the final frame briefly so the lift is readable.
    frames.extend([frames[-1]] * 15)
    return frames, env.lifted()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--out", type=str, default="../videos/grasp.mp4")
    ap.add_argument("--camera", type=str, default="grasp_cam")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = FrankaCubeLift()

    all_frames, results = [], []
    for i in range(args.episodes):
        xy = np.array([0.55, 0.0]) + rng.uniform([-0.08, -0.15], [0.08, 0.15])
        frames, ok = episode_frames(env, xy, args.camera, args.width, args.height)
        all_frames.extend(frames)
        results.append(ok)
        print(f"episode {i}: cube_xy=({xy[0]:.3f}, {xy[1]:.3f})  "
              f"lifted={ok}  frames={len(frames)}")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimsave(out, all_frames, fps=FPS, macro_block_size=1)
    env.close()

    secs = len(all_frames) / FPS
    print(f"\nwrote {out}  ({len(all_frames)} frames, {secs:.1f}s @ {FPS}fps)")
    print(f"lifted {sum(results)}/{len(results)}")


if __name__ == "__main__":
    main()
