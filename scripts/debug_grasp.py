"""Instrumented rollout: where does the scripted grasp actually lose the cube?"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import ScriptedGrasp, TOP_DOWN  # noqa: E402
from env import FrankaCubeLift  # noqa: E402

DEC = 10


def probe(env, cube_xy, max_steps=1200):
    env.reset(cube_xy=cube_xy)
    pol = ScriptedGrasp(env)

    marks = {}
    prev = pol.phase
    cube0 = env.cube_pos.copy()

    for step in range(max_steps):
        tgt = pol.target()
        arm, grip = pol.act()
        env.step(arm, grip, n_substeps=DEC)

        def snapshot():
            full = pol.ik.error(tgt, TOP_DOWN)
            return {
                "end_step": step,
                "tcp": env.tcp_pos.copy(),
                "cube": env.cube_pos.copy(),
                "width": env.gripper_width,
                "err": float(np.linalg.norm(tgt - env.tcp_pos)),
                # rot_gain is folded into ik.error, divide it back out
                "rot": float(np.linalg.norm(full[3:]) / pol.ik.rot_gain),
            }

        if pol.phase != prev:
            marks[prev] = snapshot()
            prev = pol.phase
        if pol.done:
            break

    tgt = pol.target()
    marks[prev] = snapshot()

    print(f"\ncube start xy = ({cube0[0]:.3f}, {cube0[1]:.3f})   lifted={env.lifted()}")
    print(f"{'phase':10s} {'end@':>6s} {'tcp_err':>9s} {'rot_err':>9s} {'grip_w':>8s} "
          f"{'cube_dxy':>9s} {'cube_z':>8s}")
    for name in ("hover", "descend", "close", "lift"):
        if name not in marks:
            print(f"{name:10s} {'never':>6s}")
            continue
        m = marks[name]
        dxy = np.linalg.norm(m["cube"][:2] - cube0[:2])
        print(f"{name:10s} {m['end_step']:6d} {m['err']*1000:8.1f}mm "
              f"{np.degrees(m['rot']):8.1f}d {m['width']*1000:7.1f}mm "
              f"{dxy*1000:8.1f}mm {m['cube'][2]:8.3f}")


def main():
    env = FrankaCubeLift()
    # A success and several failures from the randomized sweep.
    for xy in [(0.55, 0.0), (0.58, 0.12), (0.50, -0.14), (0.62, 0.10), (0.47, 0.14)]:
        probe(env, np.array(xy))
    env.close()


if __name__ == "__main__":
    main()
