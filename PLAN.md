# Project plan

Living roadmap. Read it before starting an experiment and update it after one:
tick boxes, add a Log row, and move items between phases as priorities change.
README.md shows the results; this file holds the direction, the rules, and
what's next.

## Goal

Build a **harness for an agentic robotics system**. A high-level model (a
VLA, VLM or LLM planner) acts as the orchestrator. Around it sit the same
pieces that made agentic AI work: skills (specialised policies), memory,
verification and recovery.

Build it one component at a time, as with the cube-lift policy, and measure
each addition. The point isn't only to build a working system but to find
**where this approach hits its limits** and which component is the bottleneck.

| Agentic AI | Robotics harness |
|---|---|
| LLM orchestrator | VLA / VLM / LLM planner choosing the next step |
| Tools / skills | Specialised policies (e.g. cube lift), motion planners, grasp detectors |
| Memory | World state (objects, poses, progress), episode history, skill track records |
| Tests / error messages | Success detectors, failure monitors |
| Retry / replan | Recovery behaviours, replanning after a failed skill |

```
            task (language)
                  │
          ┌───────▼────────┐        ┌──────────────┐
          │  orchestrator   │◄──────►│    memory     │
          │ (planner / VLA) │        │ world state,  │
          └───────┬────────┘        │ history       │
         skill call│  ▲ result      └──────▲───────┘
          ┌───────▼──┴─────┐               │
          │ skill library   │── state ─────┘
          │ lift, place, …  │
          └───────┬────────┘
       50 Hz actions│  ▲ observations
          ┌───────▼──┴─────┐
          │ robot / sim     │◄── verifier: did it work?
          └────────────────┘
```

## Expected bottlenecks

These are the hypotheses the phases are designed to test.

1. **Skill handoffs.** Skill B must start from wherever skill A stopped. BC
   policies are brittle outside their training states (v1.0 collapsed from a
   small drift), so chain success may fall well below the product of
   individual skill success rates.
2. **Verification.** A robot has to *perceive* failure, and physical mistakes
   can't be undone. The harness can only recover from failures it detects.
3. **Timescales.** Planners and VLAs run at roughly 1–10 Hz and skills at
   50 Hz, so passing control between the layers is a design problem.
4. **Grounding.** A planner can produce steps that make sense on paper but
   can't be carried out from the current physical state.
5. **Compute.** A laptop RTX 4060 (8 GB) runs small skills and small VLAs
   (~0.5B, e.g. SmolVLA) but can't fine-tune 7B models. The orchestrator can
   be an API model.

## Rules for every experiment

1. **One change at a time**, with a written hypothesis ("X fails because Y;
   changing Z should fix it").
2. **At least 3 seeds.** Single-seed results swung from 4% to 98% here. Report
   all seeds and the mean, never just the best.
3. **Judge by task success in rollouts**, not by training or validation loss.
4. **Diagnose before fixing.** Break failures into stages, and confirm the
   cause with an intervention before changing anything.
5. **Ablate every harness component.** Measure task success with and without
   it (memory, verifier, retries, planner). The difference is its measured
   contribution.
6. **Attribute every failure** to one of: plan, skill execution, handoff,
   perception, verification. Track the counts; they show where the
   bottleneck is.
7. **Keep defaults backward compatible.** New behaviour goes behind a flag.
8. **Log it.** Benchmark every new version with `scripts/benchmark.py` (one
   row per checkpoint in `results/benchmark.tsv`), put checkpoints in
   `models/`, and add a line to the Log below.
9. **Versioning.** Minor bump for eval or rollout changes, major bump for a
   new model, input, loss, dataset or harness component. Tag the commit.

## Metrics

| Metric | Meaning |
|---|---|
| Skill success | One skill, from its standard start states |
| Handoff success | The same skill, started from another skill's end states |
| Chain success vs product | Multi-step task success compared with the product of the individual skill rates. The gap is the harness's loss. |
| Failure attribution | Share of failures from plan / execution / handoff / perception / verification |
| Recovery rate | Share of detected failures the harness recovers from |
| Verifier precision/recall | How often "success" is right, and how many failures it catches |
| Ablation delta | Task success with a component minus success without it |

## Where we are

**Skill #1: cube lift, v3.0.** State-based MLP, 21 inputs, BCE gripper,
steps-since-closed counter. **100% Success@1 on 3/3 seeds on the held-out
benchmark** (200 unseen positions, strict hold-for-1-s success; 600/600).

**v4.0: the same policy from camera input**: 99.3% Success@1, 99.8%
Success@3 on the held-out benchmark (fixed RGB-D camera + color detector).

Limits:

- A start-of-episode blind spot: the hand at its start pose hides cubes close
  to the robot (all v4.0 failures).
- The color detector relies on the cube being the only red object; a learned
  detector is needed once the scene has more objects.
- Narrow task: unrotated cube, no noise or disturbances, and positions only
  within the training range. A dropped cube can land outside it, and the
  policy can't recover from there (seen in v2.0 retries).
- The counter depends on the expert's fixed 120-step hold.

## Roadmap

### Phase 0: Evaluation you can trust (done)

The harness is only as good as its success signal, so this comes first.

- [x] **Held-out test set:** 200 new cube positions (rng seed 2026) used
      only to report versions. v1.0–v3.0 re-scored (`scripts/benchmark.py`).
- [x] **Strict success:** cube above the threshold for 50 consecutive steps
      (1 s) with both fingers in contact ("hold" stage).
- [x] **Stage score, Success@1/@3 with retries, mean attempts** (defined in
      README → Evaluation).
- [x] **Results log:** `results/benchmark.tsv`, one row per checkpoint, with
      the git commit. Plot: `scripts/plot_benchmark.py`.
- [ ] **Git tags** v1.0, v1.1, v2.0, v3.0.

Done when: v3.0 has test-set and strict-success numbers, both logged. **Met;
only the tags remain.**

### Phase 1: Perception (v4.0 done; one fix open)

Question: *how much does replacing the simulator's true cube position with a
camera and detector cost?*

Modular design: `table_cam` (RGB-D) → detector → 3D cube position → the
v3.0 policy, unchanged. Detection at 10 Hz; the last good estimate is kept when
the cube is hidden (the first bit of harness memory). Every result is compared
with true positions on the held-out benchmark.

- [x] **Error budget**: v3.0 tolerates ~2 mm random error and ~5 mm fixed
      offset; at σ 10 mm it drops to 64% (seed 2: 23.5%).
- [x] **Camera module**: fixed `table_cam`, RGB/depth/segmentation, pixel →
      3D; depth agrees with geometry to 0.56 mm.
- [x] **Color baseline detector**: 0.2 mm median error; detections with
      < 60% of the cube visible are rejected and the last estimate held.
- [x] **Integrate and benchmark** (v4.0): 99.3% Success@1, 99.8% Success@3
      from camera input; rejection/memory ablated (no measurable effect).
- [x] **Occlusion analysis**: occlusion only matters at the episode start
      (hand over cubes near the robot); rare along the policy's path.
- [x] **System video** with detector overlay (`make_bc_video.py --overlay`).
- [ ] **Look before acting**: don't start the skill without a valid
      detection; move the arm out of the camera's view first if needed. Fixes
      the remaining start-of-episode blind spot.
- Deferred: **learned detector (YOLO)**, until the scene has several or
  unfamiliar objects; the color detector is already 0.2 mm here.
- Not needed: **retraining with position noise**; the detector is well inside
  the error budget.

Done when: v3.0 runs from camera input on the benchmark, with success,
position error and failure rate logged for each perception mode. **Met
(v4.0).**

### Phase 2: Skill interface and library

Question: *what does a skill need to expose to be usable by a harness?*

- [ ] **Skill API**: `start conditions → run → result`, with a success
      signal, failure signals, a timeout and a result object
      (status, final state, reason for failure).
- [ ] **Wrap cube-lift v3.0** as the first skill.
- [ ] **Remove the expert's hidden timer** (lift once the fingers stop
      closing), so the skill doesn't depend on a fixed hold that won't exist
      in other skills.
- [ ] **Add 2–3 skills**: place-at-target, push, retreat/home. Each gets a
      scripted expert, then a BC policy with the v3 recipe.
- [ ] **Handoff tests**: start each skill from other skills' end states,
      plus perturbed starts. Record skill success vs handoff success.

Done when: each skill has skill success and handoff success, logged.

### Phase 3: Harness v0, fixed sequences

Question: *how much does chaining lose, and why?*

- [ ] **Multi-object scene** and 2–3 step tasks (e.g. lift, then place at
      a target).
- [ ] **Scripted orchestrator**: a fixed skill sequence, a verifier after
      each step, and retry on failure.
- [ ] **Measure** chain success vs product, and attribute every failure.
- [ ] **Ablate** the verifier and the retries.

### Phase 4: Planner as orchestrator

Question: *are failures in the plan or in carrying it out?*

- [ ] **LLM/VLM planner** that calls skills as tools from a text state
      description, on language-specified tasks.
- [ ] **Task suite** with measurable success, including tasks that need
      reasoning (ordering, conditions).
- [ ] **Measure** plan failures vs execution failures; compare against the
      scripted orchestrator.

### Phase 5: Memory and image-based skills

Question: *what does memory buy you?*

- [ ] **Image-based skills** (end-to-end, images in, actions out), compared
      with the Phase 1 modular pipeline.
- [ ] **World-state memory**: object poses, task progress, skill history.
- [ ] **Ablate memory** on tasks with objects that go out of view or with
      long horizons.

### Phase 6: Recovery

Question: *which failures can the harness recover from, and which can't it?*

- [ ] **Replanning** on verifier failure.
- [ ] **Disturbances**: bump the cube mid-skill, drop it during lift.
- [ ] **Recovery data**: DART-style noisy demos (`collect_demos.py --noise`).
- [ ] **Measure** recovery rate per failure type.

### Phase 7: Generalist vs specialists

Question: *when does a generalist beat specialists plus a harness?*

- [ ] **Small pretrained VLA** (e.g. SmolVLA), fine-tuned on the task suite.
- [ ] **Compare** on the same tasks: end-to-end VLA vs specialist skills plus
      harness vs hybrid (VLA as a skill or fallback).
- [ ] **Sim-to-real**: decide whether real hardware is in scope.

## Backlog

Useful, but not on the harness's critical path. Pull an item in when a phase
needs it.

- Data efficiency: how few demos reach ≥95% with the v3 recipe.
- Frame question (world / gripper / both), 5 seeds each.
- Action chunking or diffusion policy for single skills.
- Physics and visual randomization (friction, mass, lighting, textures).
- Cube yaw randomization with a yaw-aligning expert.

## Decisions to make

- [ ] **Orchestrator** for Phase 4: API model (e.g. Claude) or a local
      small model.
- [ ] **Simulator** for multi-object scenes: extend the current MuJoCo
      setup or adopt a framework (e.g. robosuite, ManiSkill, LIBERO).
- [ ] **Compute** for Phase 7: whether cloud GPUs are available for VLA
      fine-tuning.

## Lessons so far

- **Validation loss didn't predict rollout success.** Only rollouts count.
- **Single seeds mislead.** The same config gave 4%–98% across seeds.
- **Binary actions need a classification loss.** MSE blurs the switch, and
  in-between commands push the policy into states it never trained on.
- **Hidden state in the expert makes imitation ambiguous.** When the expert
  acts on something the observation doesn't contain (here, a timer), the
  policy can't copy it. Either expose it or remove it. This is a memory
  problem in miniature, and it will come back at the harness level.
- **Diagnose by stage.** Breaking rollouts into reach / close / grasp / rise
  / lift found both failures quickly. The harness needs the same kind of
  failure attribution.
- **Measure the error budget before building perception.** Degrading the
  true position first showed the policy needs ~2 mm, so a simple detector was
  enough and a learned one could wait.
- **Memory can't help at step 0.** Holding the last estimate covers occlusion
  mid-task, but the only real occlusion was at the start. A skill needs a
  perception precondition before it starts.
- **A retry only helps if the failure leaves a state the skill can handle.**
  v2.0's dropped cubes landed outside the training range and retries couldn't
  recover them. This is the handoff bottleneck, seen already with one skill.

## Log

| Date | Version | What | Result |
|---|---|---|---|
| 2026-09-18 | — | Dataset sweep (300 / 1000 demos), world frame | ~50%, high seed variance |
| 2026-09-26 | v1.0 | 3000 demos, world frame | 58.7% (86 / 12 / 78) |
| 2026-09-26 | — | Observation frame comparison (world / gripper / both) | Within seed noise |
| 2026-09-26 | v1.1 | Snap gripper at rollout | 90.0% |
| 2026-09-26 | v2.0 | BCE gripper, `both` frame | 90.7%; the "grasp then freeze" failure appears |
| 2026-09-27 | v3.0 | Steps-since-closed input | 100% (3/3 seeds, standard eval) |
| 2026-09-28 | — | Project goal set: agentic robotics harness; plan rewritten | — |
| 2026-09-28 | — | Held-out benchmark (200 positions, stages, strict hold, 3 attempts) | Success@1: v1.0 58.7, v1.1 89.0, v2.0 86.8, v3.0 100% |
| 2026-09-28 | — | Perception moved to Phase 1 (camera + detector before the skill interface) | — |
| 2026-09-28 | — | Perception error budget (noisy cube position, 7 settings) | ≤2 mm random / ≤5 mm offset keeps ~100%; σ 10 mm → 64% |
| 2026-09-29 | v4.0 | Camera + color detector replace the simulator's cube position | Success@1 99.3%, Success@3 99.8%; failures = start-of-episode blind spot |
