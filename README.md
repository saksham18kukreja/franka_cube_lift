# franka_cube_lift

Behaviour cloning on a Franka Panda cube-lift task in MuJoCo. A scripted
controller (`scripts/controller.py`) generates demonstrations; an MLP policy
(`scripts/train_bc.py`) learns to imitate it from state.

This cube-lift policy is the first skill of a larger goal: a harness for an
agentic robotics system. See [PLAN.md](PLAN.md) for the roadmap.

[![BC policy v3.0 lifting the cube](videos/bc_both_bce_time_preview.gif)](videos/bc_both_bce_time.mp4)

*Policy v3.0 (seed 0, 100% success), two eval episodes in real time. Click for
the full 6-episode video.*

## Versions

Rollout success over 50 fixed cube positions, 3 training seeds each
(3000 demos, 100 epochs).

| Version | Change | Inputs | Gripper | Seed 0 / 1 / 2 | Mean |
|---|---|---|---|---|---|
| v1.0 | Baseline | 17 (world) | MSE, raw command | 86 / 12 / 78% | 58.7% |
| v1.1 | Snap gripper to open/closed at rollout | 17 (world) | MSE, snapped | 98 / 96 / 76% | 90.0% |
| v2.0 | Gripper as classifier; add gripper-frame cube position | 20 (both) | BCE logit | 100 / 92 / 80% | 90.7% |
| **v3.0** | **Add steps-since-gripper-closed input** | **21 (both + counter)** | **BCE logit** | **100 / 100 / 100%** | **100%** |

What each version fixed:

- **v1.1**: MSE regression blurs the binary open/close label. Some seeds
  half-closed the gripper ~8 cm above the cube mid-descent, left the training
  distribution, and drifted away without touching it
  (`scripts/diag_collapse.py`). Snapping the command removes the collapse.
- **v2.0**: trains the gripper as what it is, a binary decision
  (`BCEWithLogitsLoss`). Every seed now reaches and grasps the cube in 50/50
  episodes. Remaining failures: the policy grasps the cube and then freezes.
- **v3.0**: after closing, the expert holds still for exactly 120 steps and
  then lifts, on an internal timer a single frame cannot see. The added input is
  the number of consecutive steps the gripper has been commanded closed
  (capped at 200, scaled by 1/100). It is built from the expert's labels in
  training and from the policy's own past gripper commands at rollout.

Train v3.0:

```bash
python train_bc.py --demos ../demos/clean3000.npz --frame both \
    --grip-loss bce --time-feature --seed 0 \
    --out ../models/bc_clean3000_both_bce_time_s0.pt
```

Checkpoints: `models/bc_clean3000_both_bce_time_s{0,1,2}.pt` (v3.0),
`models/bc_clean3000_both_bce_s{0,1,2}.pt` (v2.0). Re-score any checkpoint on
the standard eval with `python eval_bc.py <checkpoint>...`.

v1.0 numbers use the original rollout, which sent the raw gripper output. The
rollout now always snaps it, so re-evaluating a v1.0 checkpoint gives the v1.1
number.

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
- Expert phases: hover, descend, close (hold 120 steps), lift.

## Policy (v3.0)

MLP 21 → 256 → 256 → 8 (ReLU, 73k params), AdamW lr 1e-3, cosine schedule,
batch 256, episode-level 90/10 split. Loss: MSE on the z-scored joint deltas
plus BCE on the gripper logit.

| Inputs (21) | Outputs (8) |
|---|---|
| `arm_q1..7`, `grip_width`, `tcp_xyz`, `cube_xyz`, `cube − tcp` (world), `cube − tcp` (TCP frame), steps closed / 100 | `dq1..7` (joint-target delta), gripper logit (open if > 0) |

Training options (defaults reproduce v1):

| Flag | Values | Effect |
|---|---|---|
| `--frame` | `world` (default), `gripper`, `both` | observation frame (`frame_features`) |
| `--grip-loss` | `mse` (default), `bce` | regress or classify the gripper |
| `--grip-weight` | float, default 1.0 | weight of the BCE term |
| `--time-feature` | flag | append steps-since-gripper-closed |

## Earlier experiments

Rollout success over the same 50 cube positions, with the original raw gripper
command (v1.0 rollout).

**Dataset size** (`results/sweep.tsv`, world frame):

| Demos | Epochs | Seeds | Success |
|---|---|---|---|
| ~300 (`state_demos`) | 340 | 0, 1, 2 | 48 / 52 / 60 → 53.3% |
| 1000 | 100 | 0, 1, 2 | 68 / 38 / 40 → 48.7% |
| 1000 | 340 | 0 | 22% |

**Observation frame** (3000 demos, 100 epochs):

| Frame | Seed 0 | Seed 1 | Seed 2 | Mean (raw) | Mean (snapped) |
|---|---|---|---|---|---|
| world | 86% | 12% | 78% | 58.7% | 90.0% |
| gripper | 84% | 90% | 4% | 59.3% | 80.7% |
| both | 16% | 98% | 78% | 64.0% | 94.7% |

With the raw command, runs were bimodal (78–98% or 4–16%) depending on seed,
not frame; that was the gripper failure fixed in v1.1. With it fixed, the frame
differences are within seed noise. Validation MSE never predicted rollout
success.

## Layout

```
scripts/   env, scripted expert, demo collection, BC training (train_bc.py),
           eval (eval_bc.py), failure diagnosis (diag_collapse.py), video
models/    Panda + cube scene (MuJoCo Menagerie based) and BC checkpoints
results/   sweep outputs
videos/    v1.0 and v3.0 rollout videos and README previews
```

The controller follows patterns from
[kevinzakka/mjctrl](https://github.com/kevinzakka/mjctrl) (commit `f1c82c2`).
