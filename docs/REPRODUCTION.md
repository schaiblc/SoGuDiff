# Reproduction notes

What the released code does and does not cover, and the caveats behind the
reported numbers.

## Evaluation protocol

All methods are scored on the same 500 scenes. Those scenes are generated
procedurally from `configs/env.config` (`test_sim = evaluation`,
`test_size = 500`, `testoffset = 10`), seeded per episode index, so a scene
index refers to the same situation for every method and no scene data has to be
distributed. The saved `.npz` sets are used by the style, composition and
ablation experiments instead. Each episode places robot and humans uniformly in a
15 m square with the robot's goal at least 5 m from its start, 1–10 humans,
radii 0.25 m, `v_pref` 1 m/s, a 0.25 s time step and a 25 s limit.

Every method acts through the same unicycle action space: forward speed
≤ `max_vel`, |Δv| ≤ `max_accel·dt`, |ω| ≤ `max_wrot`, |Δω| ≤ `max_w_accel·dt`.
This is the single most important detail for comparing against published
numbers, which are frequently reported under holonomic dynamics and are not
directly comparable.

## Kinematics across baselines

Every method acts through the same unicycle action space. How each gets there
differs, and the paper states the protocol:

- **Adapted** — SFM, ORCA and SICNav are analytic or optimization-based, and
  are configured to emit unicycle actions directly. SFM and ORCA use a 0.15 m
  safety inflation for consistency with their reference implementations.
- **Retrained** — CADRL, LSTM-RL, SARL and RGL are natively holonomic and were
  retrained under unicycle dynamics on the same randomized scenario
  distribution.
- **Trained holonomic, limits enforced at evaluation** — DSRNN, NaviSTAR and
  HEIGHT did not converge when retrained unicycle-native, so each keeps its
  authors' holonomic training and has the unicycle limits enforced at
  evaluation time, by the same projection ORCA and SFM use.

For DSRNN the unicycle-native attempt was made four ways (an omega
accumulator, two clipping curricula, and post-hoc clamping) and every variant
collapsed to 5-44% success, with each failure traceable to the re-engineered
action dynamics rather than to the network.

The `*_holonomic` policy names (`dsrnn_holonomic`, `navistar_holonomic`,
`height_holonomic`) run those three checkpoints under their native dynamics,
without the unicycle projection. **These are diagnostic variants and are not
reported in the paper**, which gives only the unicycle-enforced numbers. They
exist so the cost of the conversion can be measured; treat them as such, and do
not compare them against the table rows, which operate under a stricter action
space.

## Known non-determinism

- The diffusion policy samples `num_samples` trajectories per step; run-to-run
  variation of a few tenths of a percent in success rate is expected. Fix the seed for exact repeats.
- MPPI seeds per scene and style offline, and from a base seed plus a per-call
  counter online. `mppi_seed = -1` makes the sampler genuinely
  nondeterministic.
- acados solve times vary with machine and compiler, so latency figures are
  hardware-dependent even when trajectories are identical.


## Running evaluations

Run these from the repository root.

1. `scripts/download_assets.sh weights` — model weights. The comparison
   table needs nothing else; add `eval-scenes` for the style experiments.
2. `scripts/run_baseline_table.sh` — every method on the 500 procedural scenes.
3. Observe `crowdnav_env/crowd_nav/results_<METHOD>/`.

Ensure that
`[action_space]` in `configs/policy.config` and the `sim`/`env` blocks of
`configs/env.config` are unmodified, and that the run did **not** pass
`--npz_hard` — evaluating the comparison table against a saved scene set scores
a different benchmark (ensure consistency across runs).
