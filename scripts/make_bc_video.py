"""Render a trained BC policy to an mp4.

Uses the same cube positions as the rollout eval in train_bc.py (rng seed 1234),
so episode i in the video is episode i of the reported success rate.

  python make_bc_video.py --model ../models/bc_clean3000.pt --out ../videos/bc_clean3000.mp4
"""

import argparse
import os
import sys

import imageio
import numpy as np
import torch

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="../models/bc_clean3000.pt")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--out", type=str, default="../videos/bc_clean3000.mp4")
    ap.add_argument("--camera", type=str, default="grasp_cam")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

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
