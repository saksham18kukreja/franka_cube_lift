"""Scripted reach-and-grasp controller for the Panda cube task.

Two pieces:

  DiffIK          resolved-rate inverse kinematics on the 7 arm joints, solved
                  with damped least squares so it degrades gracefully near
                  singularities instead of blowing up.

  ScriptedGrasp   a state machine that takes the cube position and drives the
                  gripper through hover -> descend -> close -> lift.

The state machine is deliberately open-loop in its targets (it reads the cube
position once per phase, not per step). That keeps it usable later as a
demonstrator: it produces clean, repeatable trajectories to imitate.
"""

import mujoco
import numpy as np

from env import CUBE_HALF, GRIPPER_CLOSE, GRIPPER_OPEN, TABLE_HEIGHT

# Top-down grasp: hand +z points at the table, fingers close along world y.
# Columns are the desired x, y, z axes of the hand frame in world coordinates.
TOP_DOWN = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)


class DiffIK:
    """Differential IK on the arm joints via damped least squares."""

    # Nullspace P gain per joint (mjctrl diffik_nullspace.py:26).
    KN = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

    def __init__(self, env, damping=1e-4, max_angvel=0.785, pos_gain=0.95,
                 rot_gain=0.95, integration_dt=0.1, nullspace_scale=0.0):
        self.env = env
        self.damping = damping
        self.max_angvel = max_angvel
        self.pos_gain = pos_gain
        self.rot_gain = rot_gain
        self.integration_dt = integration_dt
        self.nullspace_scale = nullspace_scale

        nv = env.model.nv
        self._jacp = np.zeros((3, nv))
        self._jacr = np.zeros((3, nv))

        # Home posture the nullspace term biases toward, keeping the arm away
        # from joint limits and badly-conditioned configurations.
        self.q_home = env.model.key_qpos[env.home_key][: len(env.arm_qpos_adr)].copy()

    def error(self, target_pos, target_mat):
        """6-vector of position and orientation error at the TCP."""
        pos_err = target_pos - self.env.tcp_pos

        cur_quat = np.zeros(4)
        des_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cur_quat, self.env.tcp_mat.flatten())
        mujoco.mju_mat2Quat(des_quat, np.asarray(target_mat, dtype=float).flatten())

        cur_inv = np.zeros(4)
        err_quat = np.zeros(4)
        rot_err = np.zeros(3)
        mujoco.mju_negQuat(cur_inv, cur_quat)
        mujoco.mju_mulQuat(err_quat, des_quat, cur_inv)
        mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)

        return np.concatenate([pos_err * self.pos_gain, rot_err * self.rot_gain])

    def solve(self, target_pos, target_mat):
        """Joint position targets that move the TCP toward the goal pose."""
        env = self.env
        mujoco.mj_jac(env.model, env.data, self._jacp, self._jacr, env.tcp_pos, env.hand_id)

        jac = np.vstack([self._jacp[:, env.arm_dof_adr], self._jacr[:, env.arm_dof_adr]])

        # Twist is a velocity: the gain-scaled error covered over integration_dt.
        twist = self.error(target_pos, target_mat) / self.integration_dt

        # Damped least squares: dq = J^T (J J^T + lambda I)^-1 * twist
        dq = jac.T @ np.linalg.solve(jac @ jac.T + self.damping * np.eye(6), twist)

        # Nullspace term pulling toward the home posture without disturbing the
        # task-space motion (mjctrl diffik_nullspace.py:114).
        if self.nullspace_scale:
            n = len(env.arm_qpos_adr)
            nullspace = np.eye(n) - np.linalg.pinv(jac) @ jac
            gain = self.KN * self.nullspace_scale
            dq += nullspace @ (gain * (self.q_home - env.arm_qpos))

        # Clamp maximum joint velocity (rad/s).
        dq_max = np.abs(dq).max()
        if dq_max > self.max_angvel:
            dq *= self.max_angvel / dq_max

        q = env.arm_qpos + dq * self.integration_dt
        return np.clip(q, env.arm_limits[:, 0], env.arm_limits[:, 1])


class ScriptedGrasp:
    """Hover above the cube, descend, close, lift.

    Phase targets are step changes, so they are not commanded directly: a
    setpoint is ramped toward them at a per-phase speed limit. Without that the
    arm lunges at the IK rate limit when a phase flips and tears the cube out
    of the fingers.
    """

    # (name, settle steps, position tolerance [m], setpoint speed [m/s], timeout)
    # The timeout bounds every phase: without it a phase that never quite meets
    # tolerance runs to the episode cap and yields a long ragged trajectory,
    # which is poor imitation data even when the lift itself succeeds.
    PHASES = [
        ("hover", 0, 0.010, 0.35, 250),
        ("descend", 0, 0.006, 0.10, 250),
        ("close", 120, None, 0.05, 150),
        ("lift", 0, 0.020, 0.12, 300),
    ]

    def __init__(self, env, hover_height=0.12, lift_height=0.25, grasp_offset=0.0,
                 control_dt=0.02, leash=0.04, **ik_kwargs):
        self.env = env
        self.ik = DiffIK(env, **ik_kwargs)
        self.hover_height = hover_height
        self.lift_height = lift_height
        self.grasp_offset = grasp_offset
        self.control_dt = control_dt
        self.leash = leash
        self.phases = self.PHASES
        self.reset()

    def reset(self):
        self.phase_idx = 0
        self.phase_steps = 0
        self.grasp_pos = None
        self.done = False
        self.timed_out = False
        self.setpoint = self.env.tcp_pos.copy()

    @property
    def phase(self):
        return self.phases[self.phase_idx][0]

    def target(self):
        """Desired TCP position for the current phase."""
        cube = self.env.cube_pos

        if self.phase == "hover":
            return np.array([cube[0], cube[1], cube[2] + self.hover_height])
        if self.phase == "descend":
            return np.array([cube[0], cube[1], cube[2] + self.grasp_offset])
        # Once the fingers are on the cube, hold the latched pose rather than
        # chasing the cube: it now moves with the gripper.
        if self.phase == "close":
            return self.grasp_pos.copy()
        return self.grasp_pos + np.array([0.0, 0.0, self.lift_height])

    def gripper(self):
        return GRIPPER_OPEN if self.phase in ("hover", "descend") else GRIPPER_CLOSE

    def act(self):
        """One control step: returns (arm targets, gripper ctrl)."""
        target_pos = self.target()

        # Ramp the setpoint toward the phase target under a speed limit.
        speed = self.phases[self.phase_idx][3]
        delta = target_pos - self.setpoint
        max_move = speed * self.control_dt
        dist = np.linalg.norm(delta)
        if dist > max_move:
            delta *= max_move / dist
        self.setpoint = self.setpoint + delta

        # Leash: keep the setpoint within reach of where the arm actually is.
        # The servos droop under gravity and the arm can stall against a joint
        # limit; without this the setpoint integrates away from the TCP, the IK
        # saturates its rate limit every tick, and the judder shakes the cube
        # out of the fingers.
        lag = self.setpoint - self.env.tcp_pos
        lag_dist = np.linalg.norm(lag)
        if lag_dist > self.leash:
            self.setpoint = self.env.tcp_pos + lag * (self.leash / lag_dist)

        arm_target = self.ik.solve(self.setpoint, TOP_DOWN)
        grip = self.gripper()

        self._advance(target_pos)
        return arm_target, grip

    def _advance(self, target_pos):
        name, settle, tol, _, timeout = self.phases[self.phase_idx]
        self.phase_steps += 1

        if name == "close":
            ready = self.phase_steps >= settle
        else:
            ready = np.linalg.norm(target_pos - self.env.tcp_pos) < tol

        if self.phase_steps >= timeout:
            self.timed_out = True
            ready = True

        if not ready:
            return

        if self.phase_idx == len(self.phases) - 1:
            self.done = True
        else:
            # Latch the grasp pose at the moment the gripper starts closing.
            if name == "descend":
                self.grasp_pos = self.env.tcp_pos.copy()
            self.phase_idx += 1
            self.phase_steps = 0
