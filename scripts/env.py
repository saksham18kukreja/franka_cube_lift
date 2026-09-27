"""Franka Panda cube-lifting task in MuJoCo.

Thin wrapper around the menagerie Panda model plus a table and a free-floating
cube. Exposes just what a controller or a policy needs: cube pose, TCP pose,
arm joint state, and a step() that takes position targets.
"""

import os

import mujoco
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "models", "franka_emika_panda", "cube_lift_scene.xml")
)

# Offset from the `hand` body frame to the grasp center between the fingertips,
# along the hand's +z axis. This is the standard Franka flange-to-TCP distance.
TCP_OFFSET = np.array([0.0, 0.0, 0.1034])

# actuator8 maps ctrl 0..255 onto 0..0.04 m of opening per finger.
GRIPPER_OPEN = 255.0
GRIPPER_CLOSE = 0.0

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
TABLE_HEIGHT = 0.40
CUBE_HALF = 0.02


class FrankaCubeLift:
    """Panda arm tasked with lifting a cube off a table."""

    def __init__(self, model_path=MODEL_PATH, seed=0, grip_stiffness=10.0, gravcomp=True):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        # The real Panda compensates gravity in its own controller, so drooping
        # position servos are the unrealistic case, not this. Note that setting
        # model.body_gravcomp at runtime does NOT work: MuJoCo decides at
        # compile time whether to compute gravcomp at all, so on a model whose
        # XML never mentions it, qfrc_gravcomp stays zero. Apply it by hand
        # instead, the way mjctrl's opspace.py does (tau += qfrc_bias).
        self.gravcomp = bool(gravcomp)

        if grip_stiffness != 1.0:
            self._scale_grip_stiffness(grip_stiffness)

        self.hand_id = self._body("hand")
        self.cube_id = self._body("cube")

        cube_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self.cube_qpos_adr = self.model.jnt_qposadr[cube_joint]
        self.cube_dof_adr = self.model.jnt_dofadr[cube_joint]

        # Arm joints are addressed by name so we never depend on cube/finger ordering.
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.arm_qpos_adr = np.array([self.model.jnt_qposadr[j] for j in jids])
        self.arm_dof_adr = np.array([self.model.jnt_dofadr[j] for j in jids])
        self.arm_limits = self.model.jnt_range[jids].copy()

        fjids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in ("finger_joint1", "finger_joint2")
        ]
        self.finger_qpos_adr = np.array([self.model.jnt_qposadr[j] for j in fjids])

        # Robot DOFs only: compensating the cube's free joint would levitate it.
        self.robot_dof_adr = np.concatenate(
            [self.arm_dof_adr, [self.model.jnt_dofadr[j] for j in fjids]]
        )

        # "start" is defined in our scene and spells out the cube pose too;
        # panda.xml's "home" zero-pads it to the world origin.
        self.home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "start")
        if self.home_key < 0:
            self.home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._renderer = None

    def _body(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _scale_grip_stiffness(self, scale):
        """Stiffen the finger servo so a full close actually squeezes.

        The stock actuator is `force = gainprm*ctrl - kp*length - kv*vel` with
        kp=100. Closing onto a 40 mm cube only deflects the fingers ~0.05 mm,
        so ctrl=0 yields just kp*0.02 = 2 N -- enough to hold the cube at rest
        but not through an accelerating lift. Scaling gainprm and kp together
        raises the force without changing the ctrl -> opening mapping, so
        ctrl=255 still means 0.04 m per finger.
        """
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.model.actuator_gainprm[aid, 0] *= scale
        self.model.actuator_biasprm[aid, 1] *= scale  # kp
        self.model.actuator_biasprm[aid, 2] *= scale  # kv, keeps damping ratio
        lo, hi = self.model.actuator_forcerange[aid]
        self.model.actuator_forcerange[aid] = [lo * scale, hi * scale]

    # ---------------------------------------------------------------- state

    @property
    def dt(self):
        return self.model.opt.timestep

    @property
    def arm_qpos(self):
        return self.data.qpos[self.arm_qpos_adr].copy()

    @property
    def cube_pos(self):
        return self.data.xpos[self.cube_id].copy()

    @property
    def cube_quat(self):
        return self.data.xquat[self.cube_id].copy()

    @property
    def tcp_pos(self):
        """Grasp center in world coordinates."""
        rot = self.data.xmat[self.hand_id].reshape(3, 3)
        return self.data.xpos[self.hand_id] + rot @ TCP_OFFSET

    @property
    def tcp_mat(self):
        return self.data.xmat[self.hand_id].reshape(3, 3).copy()

    @property
    def gripper_width(self):
        """Total opening between the fingers, in metres."""
        return float(self.data.qpos[self.finger_qpos_adr].sum())

    # ----------------------------------------------------------- simulation

    def reset(self, cube_xy=None, randomize=False):
        """Reset to the home pose; optionally place or randomize the cube."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)

        if cube_xy is None:
            cube_xy = np.array([0.55, 0.0])
            if randomize:
                cube_xy = cube_xy + self.rng.uniform([-0.08, -0.15], [0.08, 0.15])
        cube_xy = np.asarray(cube_xy, dtype=float)

        adr = self.cube_qpos_adr
        self.data.qpos[adr : adr + 3] = [cube_xy[0], cube_xy[1], TABLE_HEIGHT + CUBE_HALF]
        self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.cube_dof_adr : self.cube_dof_adr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def step(self, arm_target, gripper_ctrl, n_substeps=1):
        """Command arm joint position targets and a gripper opening."""
        self.data.ctrl[:7] = arm_target
        self.data.ctrl[7] = gripper_ctrl
        for _ in range(n_substeps):
            if self.gravcomp:
                # qfrc_bias is gravity + Coriolis at the current state, so this
                # must be refreshed every substep, not once per control step.
                idx = self.robot_dof_adr
                self.data.qfrc_applied[idx] = self.data.qfrc_bias[idx]
            mujoco.mj_step(self.model, self.data)
        return self.observation()

    def observation(self):
        return {
            "arm_qpos": self.arm_qpos,
            "tcp_pos": self.tcp_pos,
            "cube_pos": self.cube_pos,
            "gripper_width": self.gripper_width,
        }

    # ------------------------------------------------------------- outcomes

    def cube_height(self):
        return float(self.data.xpos[self.cube_id][2])

    def lifted(self, threshold=0.10):
        """True once the cube has risen `threshold` above its resting height."""
        return self.cube_height() > TABLE_HEIGHT + CUBE_HALF + threshold

    # -------------------------------------------------------------- render

    def render(self, camera="grasp_cam", width=640, height=480):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
