"""RGB-D camera over the table: rendering, calibration and back-projection.

The camera is `table_cam` in cube_lift_scene.xml, a fixed pinhole camera, so
its intrinsics come from fovy and image size, and its extrinsics from the
simulator's camera pose. MuJoCo cameras look along their local -z with +y up;
image rows grow downward.
"""

import mujoco
import numpy as np


class Camera:
    def __init__(self, env, name="table_cam", width=640, height=480):
        self.env = env
        self.name = name
        self.width, self.height = width, height
        m = env.model
        self.cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if self.cam_id < 0:
            raise ValueError(f"no camera named {name!r} in the scene")
        fovy = np.radians(m.cam_fovy[self.cam_id])
        self.fy = 0.5 * height / np.tan(0.5 * fovy)
        self.fx = self.fy  # square pixels
        self.cx, self.cy = 0.5 * (width - 1), 0.5 * (height - 1)
        self._rgb = mujoco.Renderer(m, height=height, width=width)
        self._depth = mujoco.Renderer(m, height=height, width=width)
        self._depth.enable_depth_rendering()
        self._seg = mujoco.Renderer(m, height=height, width=width)
        self._seg.enable_segmentation_rendering()

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def pose(self):
        """Camera position and rotation (columns = camera axes) in world frame."""
        d = self.env.data
        return d.cam_xpos[self.cam_id].copy(), d.cam_xmat[self.cam_id].reshape(3, 3).copy()

    def _render(self, r):
        r.update_scene(self.env.data, camera=self.name)
        return r.render()

    def rgb(self):
        return self._render(self._rgb)

    def depth(self):
        """Per-pixel distance along the optical axis, metres."""
        return self._render(self._depth)

    def segmentation(self):
        """Per-pixel geom id (-1 for background)."""
        seg = self._render(self._seg)
        obj_id, obj_type = seg[..., 0], seg[..., 1]
        return np.where(obj_type == mujoco.mjtObj.mjOBJ_GEOM, obj_id, -1)

    def project(self, p_world):
        """World point -> (u, v) pixel and depth along the optical axis."""
        pos, R = self.pose()
        pc = R.T @ (np.asarray(p_world) - pos)
        z = -pc[2]
        u = self.cx + self.fx * pc[0] / z
        v = self.cy - self.fy * pc[1] / z
        return np.array([u, v]), z

    def backproject(self, u, v, depth):
        """Pixel (u, v) at the given optical-axis depth -> world point."""
        pos, R = self.pose()
        pc = np.array([(u - self.cx) / self.fx * depth,
                       -(v - self.cy) / self.fy * depth,
                       -depth])
        return pos + R @ pc

    def close(self):
        for r in (self._rgb, self._depth, self._seg):
            r.close()
