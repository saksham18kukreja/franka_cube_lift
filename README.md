# franka_cube_lift

Behaviour cloning on a Franka Panda cube-lift task in MuJoCo. A scripted
controller (`scripts/controller.py`) generates demonstrations; an MLP policy
(`scripts/train_bc.py`) learns to imitate it from state.

## Setup

```bash
conda env create -f environment.yml
conda activate franka_cube_lift
cd scripts
```

Demos are not committed (`clean3000.npz` exceeds GitHub's file limit).
Regenerate them with:

```bash
python collect_demos.py --episodes 3000 --out ../demos/clean3000.npz
```

## Task

- Physics 500 Hz, policy 50 Hz (10 substeps per action).
- Cube placed uniformly in x 0.47–0.63 m, y ±0.15 m, unrotated.
- Success: cube centre rises 10 cm above its resting height (`env.lifted()`).
- Expert success: 100% (failed demos are discarded at collection time).

## Policy

MLP 17 → 256 → 256 → 8 (ReLU, 72k params), MSE on z-scored actions,
AdamW lr 1e-3, cosine schedule, batch 256, episode-level 90/10 split.

| Inputs (17) | Outputs (8) |
|---|---|
| `arm_q1..7`, `grip_width`, `tcp_xyz`, `cube_xyz`, `cube − tcp` | `dq1..7` (joint-target delta), `grip` ∈ [0,1] |

`--frame gripper` swaps the world-frame positions for the cube position in the
TCP frame (11-d); `--frame both` appends it to the world features (20-d).

## Results

Rollout success over 50 fixed cube positions (eval rng seed 1234).

**Dataset size** (`results/sweep.tsv`, world frame):

| Demos | Epochs | Seeds | Success |
|---|---|---|---|
| ~300 (`state_demos`) | 340 | 0, 1, 2 | 48 / 52 / 60 → 53.3% |
| 1000 | 100 | 0, 1, 2 | 68 / 38 / 40 → 48.7% |
| 1000 | 340 | 0 | 22% |

**Observation frame** (3000 demos, 100 epochs):

| Frame | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| world | 86% | 12% | 78% | 58.7% |
| gripper | 84% | 90% | 4% | 59.3% |
| both | 16% | 98% | 78% | 64.0% |

Takeaways:

- Runs are bimodal: each either works (78–98%) or collapses (4–16%),
  depending on seed, not on the observation frame.
- Validation MSE does not predict rollout success; the failures are
  closed-loop compounding errors.

## Layout

```
scripts/   env, scripted expert, demo collection, BC training, diagnostics, video
models/    Panda + cube scene (MuJoCo Menagerie based) and BC checkpoints
results/   sweep outputs
```

The controller follows patterns from
[kevinzakka/mjctrl](https://github.com/kevinzakka/mjctrl) (commit `f1c82c2`).
