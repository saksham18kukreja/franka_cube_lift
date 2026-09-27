"""Which change actually matters? Same cube positions across all configurations.

  gravcomp   body_gravcomp on the whole model (mjctrl default)
  nullspace  scale on the per-joint posture gain Kn (0 disables)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import ScriptedGrasp  # noqa: E402
from env import FrankaCubeLift  # noqa: E402

DEC = 10


def episode(env, cube_xy, nullspace_scale, max_steps=1200):
    env.reset(cube_xy=cube_xy)
    pol = ScriptedGrasp(env, nullspace_scale=nullspace_scale)
    for _ in range(max_steps):
        arm, grip = pol.act()
        env.step(arm, grip, n_substeps=DEC)
        if pol.done:
            break
    return env.lifted(), float(np.linalg.norm(env.tcp_pos[:2] - env.cube_pos[:2]))


def main():
    rng = np.random.default_rng(0)
    positions = [
        np.array([0.55, 0.0]) + rng.uniform([-0.08, -0.15], [0.08, 0.15])
        for _ in range(25)
    ]

    configs = []
    for gravcomp in (False, True):
        for ns in (0.0, 0.1, 1.0):
            configs.append((gravcomp, ns))

    print(f"{len(positions)} cube positions, identical across configs\n")
    print(f"{'gravcomp':>9s} {'nullspace':>10s} {'success':>9s} {'med xy_err':>11s}")
    for gravcomp, ns in configs:
        env = FrankaCubeLift(gravcomp=gravcomp)
        out = [episode(env, xy, ns) for xy in positions]
        env.close()
        wins = [o[0] for o in out]
        errs = [o[1] for o in out]
        print(f"{str(gravcomp):>9s} {ns:10.1f} {100*np.mean(wins):8.1f}% "
              f"{np.median(errs)*1000:9.1f} mm")


if __name__ == "__main__":
    main()
