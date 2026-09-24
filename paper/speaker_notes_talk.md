# Speaker notes — commensurate complexity

Framing for the talk. The paper argues something narrower, so see the last section before using this in writing.

## The frame, in one line

- The experiment is commensurate complexity: hold the control architecture fixed, vary the model, and see which parts of the architecture each model element actually earns.
- A spectrum of models plus one deployed controller lets you map the correspondence in both directions — what the architecture needs, what it carries that nothing justifies, and what it lacks.
- Say early that the architecture is ArduRover's, not ours. The audience flies it. That is what makes this an audit rather than a modelling exercise.

## Why the setup supports that claim

- Three models of the same vehicle from the same data, deliberately spanning interpretable to opaque.
- One fixed compensator: two decoupled loops, PI plus feedforward, derivative gain zero on this vehicle.
- Because the compensator is fixed, "is this model good enough" becomes answerable: does a change of model level move the gains, the achievable bandwidth, or the tuning procedure?

## The map — the central slide

| ArduRover element | Verdict | Evidence |
|---|---|---|
| Two decoupled loops | warranted | no cross-coupling, $r=-0.08$; time constants differ 4× |
| Steering: fixed PI | sufficient | Level 2 inert on yaw, $\lVert \ell T \rVert_\infty = 0.034$ |
| Steering: speed scaling | present, unjustified | Diagnostic 1 cannot resolve it |
| Speed: scalar feedforward | structurally insufficient | Diagnostic 2, quadratic drag |
| Speed: fixed PI gains | valid over $[0.35, 1.28]$ m/s only | Diagnostic 4 |
| Derivative gains $=0$ | untested | nothing in this work touches it |
| Anything beyond PI+FF | unsupported by the data | Level 3 envelope limit |

- Read the table as three columns of verdict: warranted, superfluous, missing. Two rows are neither, and say so rather than letting someone find it.

## The feedforward result — build a slide on this one

- `ATC_SPEED_FF` is one number, so it implements $u = \mathrm{FF}\cdot v$, the inverse of a *linear* plant.
- Level 2's steady state is $u = d_2 v\lvert v\rvert / k_T$, so the feedforward gain the vessel actually requires is $d_2 v / k_T$, which grows with speed.

| $v$ (m/s) | required gain | deployed $\mathrm{FF}=0.2$ |
|---|---|---|
| 0.5 | 0.055 | over by 0.072 |
| 1.22 | 0.135 | over by 0.080 |
| 1.81 | 0.200 | exact |
| 2.9 | 0.321 | under by 0.350 |

- The required gain varies 5.8× across the tested envelope. One scalar can be right at exactly one speed, and here that speed is 1.81 m/s.
- This is the sharpest thing in the talk: a quantitative statement about deployed autopilot software, derived from parameters already in the paper.
- The fix is not more model. It is a feedforward term of the right *form*, quadratic rather than linear, which the model hands you directly.

## What the spectrum says about itself

- Level 1 settles the loop structure and nothing more. Two time constants a factor of four apart is the whole argument for two loops.
- Level 2 earns its place on the speed channel and is inert on steering. Say both; the negative is as useful as the positive.
- Level 3 is the bound, not a competitor. Inside the speeds its data covers it is a serviceable simulator; outside them it saturates near 1.3 m/s because its corpus is capped at 3.14.
- On equal terms — refit on the same partition, scored on the same held-out segments — the ordering is $+0.53$, $+0.64$, $+0.84$. The network is the better predictor. Do not oversell structure.

## Anticipated questions

- *Why not just tune on the water?* You still will. The models buy loop structure, a simulation to verify it in, a basis for comparing variants, and a rehearsed tuning procedure. They do not buy transferable gains, and we say so.
- *Is this specific to ArduPilot?* The method is not. The map is, because the map is an audit of one architecture. Another autopilot with the same PI+FF structure would give the same rows.
- *Why is the NARX result so modest at $R^2 = 0.47$?* Because 6.3% of the trial carries 49% of the error, all of it above 1.5 m/s where the training corpus does not reach. Below 1.5 m/s the same simulation returns 0.60. See [[speaker_notes_narx]] for the architecture defence.
- *Did you tune the network?* No, and the paper says so. It is a reference point, not a proposed method.

## Honesty check before using this in writing

- The paper makes a narrower argument. Diagnostic 4 asks whether a model change moves the design; this framing generalizes that to a structural audit of the whole architecture.
- Two rows of the map are not results. The derivative row is a genuine blank. The speed-scaling row is a negative — the firmware has a feature the data cannot evaluate.
- The feedforward row is an inference from Level 2's steady state, not something the paper measures directly on the vehicle. It is sound, but it is arithmetic on identified parameters rather than an experiment.
- If the framing survives the talk, it is the natural spine for the journal version.
