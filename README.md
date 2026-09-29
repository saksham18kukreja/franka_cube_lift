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

3 training seeds per version (3000 demos, 100 epochs). The last column is
lenient success on the 50-position set used while debugging; the held-out
[benchmark](#benchmark-results) below is the number to report.

| Version | Change | Inputs | Gripper | Debug set, seed 0 / 1 / 2 | Mean |
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
number (`benchmark.py --raw-grip` reproduces v1.0).

## Benchmark results

200 held-out cube positions (never used for debugging), up to 3 attempts each,
3 seeds per version. Mean over seeds; definitions in
[Evaluation](#evaluation).

| Version | Stage score | Success@1 | Success@3 | Mean attempts | Lifted@1 (old) |
|---|---|---|---|---|---|
| v1.0 | 64.7 | 58.7% | 67.5% | 1.14 | 58.7% |
| v1.1 | 94.5 | 89.0% | 97.7% | 1.11 | 89.2% |
| v2.0 | 95.4 | 86.8% | 93.0% | 1.08 | 92.3% |
| **v3.0** | **100** | **100%** | **100%** | **1.00** | **100%** |

![Held-out benchmark by version](results/benchmark_versions.png)

Per seed (seed 0 / 1 / 2):

| Version | Stage score | Success@1 | Success@3 | Mean attempts |
|---|---|---|---|---|
| v1.0 | 90.5 / 12.0 / 91.6 | 79.0 / 12.0 / 85.0% | 93.5 / 13.0 / 96.0% | 1.20 / 1.08 / 1.15 |
| v1.1 | 97.5 / 98.2 / 87.8 | 95.0 / 96.5 / 75.5% | 98.0 / 98.5 / 96.5% | 1.04 / 1.03 / 1.26 |
| v2.0 | 98.8 / 95.3 / 92.0 | 92.5 / 89.5 / 78.5% | 92.5 / 97.5 / 89.0% | 1.00 / 1.10 / 1.13 |
| v3.0 | 100 / 100 / 100 | 100 / 100 / 100% | 100 / 100 / 100% | 1.00 / 1.00 / 1.00 |

Stage funnel, % of positions reaching each stage on attempt 1 (mean over seeds):

| Version | reach | close | grasp | rise | lift | hold |
|---|---|---|---|---|---|---|
| v1.0 | 69.2 | 69.2 | 69.2 | 63.3 | 58.7 | 58.7 |
| v1.1 | 100 | 99.7 | 99.7 | 89.3 | 89.2 | 89.0 |
| v2.0 | 100 | 100 | 100 | 93.0 | 92.3 | 86.8 |
| v3.0 | 100 | 100 | 100 | 100 | 100 | 100 |

What the results show:

- **v3.0 holds on unseen positions**: 600/600 first-attempt strict successes.
  The debug-set numbers track the held-out ones closely, so the debug set was
  not badly overfit.
- **v1.0 fails at the approach** (69% reach) because of the gripper collapse;
  seed 1 fails the same way every time (12% → 13% with retries).
- **v2.0 drops the cube**: 92.3% lifted but 86.8% held. Its failures are drops
  during the lift or lifts too late to finish the hold.
- **Retries help when failures leave a familiar scene.** v1.1 gains 8.7 points
  from retries (failed attempts leave the cube in place). v2.0 gains 6.2: a
  dropped cube tumbles to x ≈ 0.30 m, outside the 0.47–0.63 m training range,
  and the retry heads the wrong way. This is an early case of the handoff
  bottleneck in [PLAN.md](PLAN.md).

Run it:

```bash
python benchmark.py ../models/bc_clean3000_both_bce_time_s{0,1,2}.pt --label v3.0
python plot_benchmark.py        # results/benchmark_versions.png
```

`benchmark.py` appends one row per checkpoint to `results/benchmark.tsv`
(with the git commit). Options: `--positions`, `--attempts`, `--max-steps`,
`--raw-grip`, `--no-log`.

## Evaluation

**Protocol** (`scripts/benchmark.py`):

- **Positions:** 200 cube positions drawn with rng seed 2026 from the training
  range (x 0.47–0.63 m, y ±0.15 m). Reporting only, never debugging.
- **Attempts:** up to 3 per position, 700 steps (14 s) each.
- **Retries:** the sim and policy are deterministic, so re-running from the
  same start would replay the same failure. A retry starts from where the last
  attempt left things: the arm returns home with the gripper open, and the
  cube stays wherever it ended up (pushed, dropped, tilted).

**Stages**, one point each; the furthest stage reached counts:

| Stage | Condition |
|---|---|
| reach | TCP within 15 mm of the cube centre in xy and 20 mm in z |
| close | gripper commanded closed while at the grasp pose |
| grasp | finger opening 30–50 mm while at the grasp pose |
| rise | cube centre 10 mm above its resting height |
| lift | cube centre 100 mm above its resting height |
| hold | lift sustained for 50 steps (1 s) with both fingers on the cube |

**Metrics:**

- **Stage score**: points from stages reached on attempt 1, out of 6,
  averaged over positions (%). *How far does the policy get?* It gives partial
  credit (always grasping but never lifting scores 50, never reaching scores
  0) but hides which stage fails.
- **Stage funnel**: % of positions reaching each stage on attempt 1. *Where
  does it fail?* The biggest drop between two stages locates the failure. This
  is how both of the v1 → v3 bugs were found.
- **Success@1**: hold reached on the first attempt. *How good is the skill
  on its own?* The headline measure of the skill with no help.
- **Success@3**: hold reached within 3 attempts. *Does it get there with a
  retry loop?* This measures the policy inside a minimal harness.
  Success@3 − Success@1 shows how recoverable its failures are.
- **Mean attempts**: attempts used, over positions that eventually succeed.
  *What does success cost?* Retries cost time and, on hardware, wear. It
  ignores failed positions, so read it next to Success@3.
- **Lifted@1**: cube 10 cm up on any single frame of attempt 1 (the original
  metric). It links to all earlier results. Lifted@1 − Success@1 counts drops
  and lifts too late to finish the hold.

How they fit together:

| Question | Metric |
|---|---|
| How good is the skill alone? | Success@1 |
| How close do the failures get? | Stage score |
| Where does it break? | Stage funnel |
| Can a retry loop rescue it? | Success@3 − Success@1 |
| What does rescue cost? | Mean attempts |
| Does it hold on, or just touch the threshold? | Lifted@1 − Success@1 |

The retry gap is an early form of the harness's *recovery rate*, and the funnel
an early form of *failure attribution* (see [PLAN.md](PLAN.md)).

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
- Success: cube lifted 10 cm and held for 1 s with both fingers on it (see
  [Evaluation](#evaluation)). Earlier results use the lenient single-frame
  check `env.lifted()`.
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
           held-out benchmark (benchmark.py, plot_benchmark.py), quick eval
           on the debug set (eval_bc.py), failure diagnosis (diag_collapse.py),
           video
models/    Panda + cube scene (MuJoCo Menagerie based) and BC checkpoints
results/   benchmark results and plot, dataset sweep
videos/    v1.0 and v3.0 rollout videos and README previews
```

The controller follows patterns from
[kevinzakka/mjctrl](https://github.com/kevinzakka/mjctrl) (commit `f1c82c2`).
