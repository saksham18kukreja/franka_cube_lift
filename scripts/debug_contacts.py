"""Inspect the finger/cube contacts at the moment of grasp and during the lift."""

import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import ScriptedGrasp  # noqa: E402
from env import FrankaCubeLift  # noqa: E402

DEC = 10


def cube_contacts(env):
    """Contacts touching the cube: (other geom, contact point, normal force)."""
    m, d = env.model, env.data
    cube_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        if cube_gid not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == cube_gid else c.geom1
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom{other}"
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        out.append((name, c.pos.copy(), float(f[0])))
    return out


def report(env, label):
    cube = env.cube_pos
    print(f"\n--- {label} ---")
    print(f"  cube center z = {cube[2]:.4f}   tcp z = {env.tcp_pos[2]:.4f}   "
          f"tcp-cube dz = {(env.tcp_pos[2]-cube[2])*1000:+.1f} mm")
    print(f"  grip width = {env.gripper_width*1000:.1f} mm")
    cons = cube_contacts(env)
    if not cons:
        print("  NO CONTACTS on cube")
    for name, pos, fn in cons:
        print(f"  {name:28s} at z={pos[2]:.4f} "
              f"({(pos[2]-cube[2])*1000:+6.1f} mm vs cube center)  Fn={fn:7.3f} N")


def timeseries(env, cube_xy, label):
    """Track the lift step by step to find where the cube is lost."""
    env.reset(cube_xy=cube_xy)
    pol = ScriptedGrasp(env)
    print(f"\n===== {label}  cube_xy={tuple(cube_xy)} =====")
    print(f"{'lift@':>6s} {'cube_z':>7s} {'tcp_z':>7s} {'setp_z':>7s} "
          f"{'grip':>6s} {'nFng':>5s} {'Fn_tot':>7s} {'slipXY':>7s}")

    cube0 = None
    for step in range(1500):
        arm, grip = pol.act()
        env.step(arm, grip, n_substeps=DEC)

        if pol.phase == "lift":
            if cube0 is None:
                cube0 = env.cube_pos.copy()
            if pol.phase_steps % 15 == 0:
                cons = [c for c in cube_contacts(env) if "table" not in c[0]]
                fn = sum(c[2] for c in cons)
                slip = np.linalg.norm(env.cube_pos[:2] - cube0[:2])
                print(f"{pol.phase_steps:6d} {env.cube_pos[2]:7.4f} {env.tcp_pos[2]:7.4f} "
                      f"{pol.setpoint[2]:7.4f} {env.gripper_width*1000:5.1f}m "
                      f"{len(cons):5d} {fn:7.3f} {slip*1000:6.1f}m")
        if pol.done:
            break
    print(f"  -> lifted={env.lifted()}  final cube z={env.cube_height():.3f}")


def main():
    env = FrankaCubeLift()
    timeseries(env, np.array([0.58, 0.12]), "FAILURE case")
    timeseries(env, np.array([0.55, 0.00]), "SUCCESS case")
    env.close()


if __name__ == "__main__":
    main()
