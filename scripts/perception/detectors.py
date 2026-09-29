"""Cube detectors: image(s) from `table_cam` -> 3D cube centre in world frame.

Every detector returns (position or None, info). None means "not found"; the
caller decides what to do then (e.g. keep the last estimate).
"""

import numpy as np

CUBE_HALF = 0.02


def centre_from_points(pts):
    """Cube centre from its visible surface points (world frame).

    table_cam sits on the +x side of the cube, looking down, so it sees the top
    face and the +x face. Robust extents of those faces give the centre:
    top z - half, front x - half, and the middle of the y span.
    """
    x_hi = np.percentile(pts[:, 0], 98)
    z_hi = np.percentile(pts[:, 2], 98)
    y_lo, y_hi = np.percentile(pts[:, 1], [2, 98])
    return np.array([x_hi - CUBE_HALF, 0.5 * (y_lo + y_hi), z_hi - CUBE_HALF])


def mask_to_points(cam, mask, depth):
    """Back-project every masked pixel with its own depth."""
    v, u = np.nonzero(mask)
    d = depth[v, u]
    pos, R = cam.pose()
    pc = np.stack([(u - cam.cx) / cam.fx * d, -(v - cam.cy) / cam.fy * d, -d], axis=1)
    return pos + pc @ R.T


# pixels * depth^2 for a fully visible cube in table_cam (median over the 200
# held-out positions; p95 762). Image area falls with distance squared, so this
# normalises the red-pixel count into a visible fraction.
FULL_VIEW_K = 747.0


class ColorDetector:
    """Baseline: threshold red pixels, back-project, fit the cube centre.

    Assumes the cube is the only strongly red object in view. Detections with
    less than `min_visible` of the expected cube area are rejected: with the
    top face partly hidden the centre fit goes wrong (3-34 mm error below 60%
    visible, 0.2 mm median above it, measured during v3.0 episodes).
    """

    name = "color"

    def __init__(self, cam, min_pixels=30, min_visible=0.6):
        self.cam = cam
        self.min_pixels = min_pixels
        self.min_visible = min_visible

    def mask(self, rgb):
        r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
        return (r > 90) & (r > 2 * g) & (r > 2 * b)

    def detect(self, rgb=None, depth=None):
        rgb = self.cam.rgb() if rgb is None else rgb
        depth = self.cam.depth() if depth is None else depth
        m = self.mask(rgb)
        n = int(m.sum())
        if n < self.min_pixels:
            return None, {"pixels": n, "visible": 0.0}
        visible = n * float(np.median(depth[m])) ** 2 / FULL_VIEW_K
        info = {"pixels": n, "visible": visible}
        if visible < self.min_visible:
            return None, info
        return centre_from_points(mask_to_points(self.cam, m, depth)), info
