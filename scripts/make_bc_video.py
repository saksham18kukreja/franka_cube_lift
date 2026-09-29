"""Render a trained BC policy to an mp4.

Default mode: the robot scene only, on the eval positions from train_bc.py
(rng seed 1234), so episode i in the video is episode i of that eval.

  python make_bc_video.py --model ../models/bc_clean3000.pt --out ../videos/bc_clean3000.mp4

--overlay mode: the whole system, side by side. Left, the robot scene; right,
what table_cam sees, with the detector's red mask tinted, a wireframe cube at
the estimate the policy is using (green = accepted detection, orange = rejected
so the last estimate is held, red = no estimate yet, workspace-centre
fallback) and the true cube centre as a white dot. Episodes run through
benchmark.py's Agent, Perception and attempt() on the held-out positions,
retries included, so the video shows exactly what the benchmark scores.

  python make_bc_video.py --overlay --perception color \\
      --clip ../models/bc_clean3000_both_bce_time_s0.pt:0 \\
      --clip ../models/bc_clean3000_both_bce_time_s2.pt:151 \\
      --out ../videos/system_color.mp4
"""

import argparse
import os
import sys

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import FrankaCubeLift  # noqa: E402
from collect_demos import GRIPPER_SCALE  # noqa: E402
from train_bc import MLPPolicy, policy_obs, GRIP_THRESHOLD  # noqa: E402

DEC = 10          # 500 Hz physics -> 50 Hz control
FPS = 50          # render every control step => real time


def episode_frames(env, policy, norm, cube_xy, device, camera, width, height,
                   max_steps=700, post_lift=40, frame="world", grip_loss="mse",
                   time_feature=False):
    env.reset(cube_xy=cube_xy)
    obs_mean, obs_std, act_mean, act_std = norm
    frames, lifted_at = [], None
    n_closed = 0
    for t in range(max_steps):
        o = policy_obs(env, frame, time_feature, n_closed)
        with torch.no_grad():
            x = torch.as_tensor((o - obs_mean) / obs_std, dtype=torch.float32,
                                device=device).unsqueeze(0)
            a = policy(x).squeeze(0).cpu().numpy() * act_std + act_mean
        arm_target = np.clip(
            env.arm_qpos + a[:7], env.arm_limits[:, 0], env.arm_limits[:, 1]
        )
        grip = float(a[7] > GRIP_THRESHOLD[grip_loss]) * GRIPPER_SCALE  # as in train_bc.py
        env.step(arm_target, grip, n_substeps=DEC)
        n_closed = n_closed + 1 if grip == 0 else 0
        frames.append(env.render(camera=camera, width=width, height=height))
        # Keep running briefly past the success check so the lift is visible.
        if lifted_at is None and env.lifted():
            lifted_at = t
        if lifted_at is not None and t - lifted_at >= post_lift:
            break
    frames.extend([frames[-1]] * 15)
    return frames, lifted_at is not None


# ------------------------------------------------------------ overlay mode

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BAR_H = 96
STATUS_COLOR = {"accepted": (12, 163, 12), "rejected": (250, 178, 25),
                "fallback": (208, 59, 59), "gt": (200, 200, 200)}
STATUS_TEXT = {"accepted": "detected", "rejected": "rejected: holding last estimate",
               "fallback": "no estimate yet: workspace-centre fallback",
               "gt": "ground truth (no detector)"}
CUBE_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default(size=size)


class OverlayRenderer:
    def __init__(self, env, perception, scene_camera, width, height):
        from perception.camera import Camera
        from perception.detectors import ColorDetector
        self.env, self.perception = env, perception
        self.scene_camera, self.w, self.h = scene_camera, width, height
        self.cam = Camera(env, width=width, height=height)
        self.mask = ColorDetector(self.cam).mask
        self.f_big, self.f, self.f_small = font(22), font(17), font(14)

    def _camera_panel(self):
        rgb = self.cam.rgb()
        img = rgb.copy()
        m = self.mask(rgb)
        img[m] = (0.45 * img[m] + 0.55 * np.array([60, 230, 90])).astype(np.uint8)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)

        est = self.perception.estimate
        status = self.perception.last.get("status", "gt") if est is not None else "gt"
        color = STATUS_COLOR[status]
        if est is not None:
            h = 0.02
            corners = [est + np.array([sx, sy, sz]) * h
                       for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            px = [tuple(self.cam.project(c)[0]) for c in corners]
            for a, b in CUBE_EDGES:
                d.line([px[a], px[b]], fill=color, width=2)
        (u, v), _ = self.cam.project(self.env.cube_pos)
        d.ellipse([u - 4, v - 4, u + 4, v + 4], fill=(255, 255, 255), outline=(0, 0, 0))

        d.rectangle([0, 0, self.w, 30], fill=(0, 0, 0))
        d.text((10, 5), "table_cam  |  detector view", font=self.f, fill=(255, 255, 255))
        d.rectangle([0, self.h - 56, self.w, self.h], fill=(0, 0, 0))
        d.rectangle([10, self.h - 44, 26, self.h - 28], fill=color)
        d.text((34, self.h - 48), STATUS_TEXT[status], font=self.f, fill=(255, 255, 255))
        if est is not None:
            err = np.linalg.norm(est - self.env.cube_pos) * 1000
            vis = self.perception.last.get("visible")
            extra = f"   visible {100 * vis:.0f}%" if vis is not None and status != "gt" else ""
            d.text((34, self.h - 26), f"estimate error {err:5.1f} mm{extra}",
                   font=self.f_small, fill=(210, 210, 210))
        d.text((self.w - 190, self.h - 26), "white dot = true centre",
               font=self.f_small, fill=(170, 170, 170))
        return np.asarray(pil)

    def frame(self, header, best, closed, n_closed, t, banner=None):
        from benchmark import STAGES
        scene = Image.fromarray(self.env.render(camera=self.scene_camera,
                                                width=self.w, height=self.h))
        sd = ImageDraw.Draw(scene)
        sd.rectangle([0, 0, self.w, 30], fill=(0, 0, 0))
        sd.text((10, 5), "robot scene", font=self.f, fill=(255, 255, 255))
        if banner:
            sd.rectangle([0, self.h // 2 - 30, self.w, self.h // 2 + 30], fill=(0, 0, 0))
            sd.text((20, self.h // 2 - 14), banner, font=self.f, fill=(255, 255, 255))

        top = np.concatenate([np.asarray(scene), self._camera_panel()], axis=1)
        bar = Image.new("RGB", (2 * self.w, BAR_H), (18, 18, 18))
        bd = ImageDraw.Draw(bar)
        bd.text((14, 10), header, font=self.f_big, fill=(255, 255, 255))
        grip = f"gripper {'closed' if closed else 'open'}   steps closed {n_closed}"
        bd.text((14, 44), f"t = {t * 0.02:5.2f} s   {grip}", font=self.f, fill=(210, 210, 210))
        x = 14
        bd.text((x, 70), "stages:", font=self.f_small, fill=(170, 170, 170))
        x += 62
        for i, s in enumerate(STAGES):
            done = i <= best
            fill = (12, 163, 12) if done else (70, 70, 70)
            wtxt = bd.textlength(s, font=self.f_small) + 16
            bd.rounded_rectangle([x, 68, x + wtxt, 88], radius=4, fill=fill)
            bd.text((x + 8, 70), s, font=self.f_small, fill=(255, 255, 255))
            x += wtxt + 6
        return np.concatenate([top, np.asarray(bar)], axis=0)


def overlay_video(args):
    import benchmark as bm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clips = args.clip or [f"{args.model}:{i}" for i in range(args.episodes)]
    positions = bm.heldout_positions()
    env = FrankaCubeLift()
    perception = bm.Perception(args.perception, hz=args.perception_hz,
                               min_visible=args.min_visible)
    renderer = None
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    writer = imageio.get_writer(out, fps=FPS, macro_block_size=1)
    n_frames, summary = 0, []

    for clip in clips:
        path, idx = clip.rsplit(":", 1)
        idx = int(idx)
        agent = bm.Agent(path, device)
        env.reset(cube_xy=positions[idx])
        perception.reset_position(idx)
        if renderer is None:
            renderer = OverlayRenderer(env, perception, args.camera, args.width, args.height)
        name = os.path.basename(path).replace(".pt", "")
        result = "failed"
        for k in range(args.attempts):
            if k > 0:
                if env.cube_pos[2] < bm.TABLE_HEIGHT - 0.05:
                    break
                bm.return_home(env)
            head = (f"{name}   held-out position {idx}   attempt {k + 1}/{args.attempts}"
                    f"   perception: {args.perception}")
            state = {"t": 0, "closed": False}

            def on_step(best, closed, n_closed, head=head, state=state):
                nonlocal n_frames
                writer.append_data(renderer.frame(head, best, closed, n_closed, state["t"]))
                state["t"] += 1
                state["closed"] = closed
                n_frames += 1

            best = bm.attempt(env, agent, perception, args.max_steps, on_step=on_step)
            if best == len(bm.STAGES) - 1:
                result = f"success on attempt {k + 1}"
                break
            if k + 1 < args.attempts:
                msg = (f"attempt {k + 1} failed (reached: "
                       f"{bm.STAGES[best] if best >= 0 else 'none'}) -> arm home, retry")
                hold = renderer.frame(head, best, False, 0, state["t"], banner=msg)
                for _ in range(FPS):
                    writer.append_data(hold)
                n_frames += FPS
        last = renderer.frame(head, best, state["closed"], agent.n_closed, state["t"],
                              banner=f"result: {result}")
        for _ in range(FPS):
            writer.append_data(last)
        n_frames += FPS
        summary.append((name, idx, result))
        print(f"{name} position {idx}: {result}")

    writer.close()
    env.close()
    print(f"\nwrote {out}  ({n_frames} frames, {n_frames / FPS:.1f}s @ {FPS}fps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="../models/bc_clean3000.pt")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--out", type=str, default="../videos/bc_clean3000.mp4")
    ap.add_argument("--camera", type=str, default="grasp_cam")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--overlay", action="store_true",
                    help="side-by-side system view with the detector's camera")
    ap.add_argument("--clip", action="append",
                    help="overlay: checkpoint:heldout_index, repeatable")
    ap.add_argument("--perception", choices=["gt", "color"], default="color")
    ap.add_argument("--perception-hz", type=float, default=10.0)
    ap.add_argument("--min-visible", type=float, default=0.6)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=700)
    args = ap.parse_args()

    if args.overlay:
        overlay_video(args)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.model, map_location=device, weights_only=False)
    policy = MLPPolicy(ck["obs_dim"], ck["act_dim"], hidden=ck["hidden"]).to(device)
    policy.load_state_dict(ck["state_dict"])
    policy.eval()

    rng = np.random.default_rng(1234)
    env = FrankaCubeLift()

    all_frames, results = [], []
    for i in range(args.episodes):
        xy = np.array([0.55, 0.0]) + rng.uniform([-0.08, -0.15], [0.08, 0.15])
        frames, ok = episode_frames(env, policy, ck["norm"], xy, device,
                                    args.camera, args.width, args.height,
                                    frame=ck.get("frame", "world"),
                                    grip_loss=ck.get("grip_loss", "mse"),
                                    time_feature=ck.get("time_feature", False))
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
