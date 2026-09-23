# Speaker notes — defending the Level-3 architecture

Use if asked "why 19 lags / 64 units / increments?" The frame is that this is a *generic instance of a standard family*, deliberately so, and the paper's claims do not rest on the specifics.

## Lead with the frame

- The network is a reference point in the comparison, not a proposed method. We make no claim it is optimal, and the paper says so explicitly.
- It is a textbook delayed-input NARX — Narendra and Parthasarathy 1990, delayed-tap variants from Lin 1997. Nothing here is novel and nothing is meant to be.
- The contribution is the comparison and the sufficiency test. The network is the thing being compared against, so it needs to be *representative*, not *tuned*.

## Why the family is the right choice

- NARX is the standard black-box form for input-output dynamics when you have actuator commands and measured states and nothing else.
- Multi-input multi-output, so it is free to discover the surge-yaw coupling that Levels 1 and 2 assume away. That freedom is the point of including it.
- It bounds what added complexity buys at this data scale, which is the question the model spectrum exists to answer.

## The specifics are ordinary, and defensible on ordinary grounds

- Tap spacing 0, 1, 2, 5, 10, 20 is geometric: dense near zero, sparse far out. Covers two decades of timescale in 19 inputs where a dense bank would need 60.
- History reach matches the vehicle. Throttle reaches 2.0 s against a surge time constant of 1.95 s, so about one τ. Rudder reaches 1.0 s against a yaw time constant of 0.471 s, about two τ. Each input sees roughly one time constant of the channel it drives.
- Outputs are state increments, not absolute states, so "no change" is the default and the network learns a correction. At 10 Hz consecutive states are nearly identical, which makes the absolute form badly conditioned.
- tanh rather than ReLU because it is bounded and smooth. A ReLU network fed its own output in a long rollout can grow without limit.
- 1,410 parameters against 28,023 training rows is about 20 samples per parameter. Conservative for a network this size.

## Why the specifics do not change the conclusions

- **Strongest point.** The finding is that a purely data-driven model cannot locate an equilibrium the data never visited. That is a property of learning from samples, not of a tap structure.
- We checked this. Refitting the same data with a completely different formulation — acceleration as a function of current speed and throttle, no lag bank at all — shows the same envelope limit: accurate inside the covered region, saturating outside it.
- So the architecture is not what produces the result. Change the architecture and the coverage limit survives.

## If pressed harder, concede cleanly

- No architecture search was run. The checkpoint was frozen by hand after broader evaluation, not selected by a declared rule. Say so; it is in the record.
- The lagged-state inputs are the weak part. They let the network satisfy training through persistence without learning the throttle-to-speed map, which is why data curation mattered so much.
- A model of acceleration as a function of state and input did better on the same data. That is a direction for the extension, not a defence of this one.
- What we will not concede: the comparison is unfair. On equal terms, refit on the same partition and scored on the same held-out segments, the network is the *better* predictor at +0.84 against +0.53 and +0.64.
