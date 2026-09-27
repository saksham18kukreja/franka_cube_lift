"""Re-score saved BC checkpoints with the rollout eval from train_bc.py.

Same 50 cube positions (rng seed 1234) and the same rollout_policy, so numbers
are directly comparable to the ones train_bc.py prints.

  python eval_bc.py ../models/bc_clean3000.pt ../models/bc_clean3000_gripper.pt
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import FrankaCubeLift  # noqa: E402
from train_bc import MLPPolicy, rollout_policy  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = FrankaCubeLift()
    for path in sys.argv[1:]:
        ck = torch.load(path, map_location=device, weights_only=False)
        policy = MLPPolicy(ck["obs_dim"], ck["act_dim"], hidden=ck["hidden"]).to(device)
        policy.load_state_dict(ck["state_dict"])
        policy.eval()
        frame = ck.get("frame", "world")

        rng = np.random.default_rng(1234)
        wins = []
        for _ in range(50):
            xy = np.array([0.55, 0.0]) + rng.uniform([-0.08, -0.15], [0.08, 0.15])
            wins.append(rollout_policy(env, policy, ck["norm"], xy, device, frame=frame,
                                       grip_loss=ck.get("grip_loss", "mse"),
                                       time_feature=ck.get("time_feature", False)))
        print(f"{path}\t{frame}\t{100*np.mean(wins):.1f}%\t({sum(wins)}/{len(wins)})")
    env.close()


if __name__ == "__main__":
    main()
