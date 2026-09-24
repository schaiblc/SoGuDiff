# Vendored baseline forks

Each directory here is a fork of a published navigation baseline, modified so
that it trains on the same task this paper evaluates on and so that its
checkpoints load through `crowdnav_env`'s policy interface. The forks are
vendored rather than referenced as submodules because the paper's numbers
depend on those modifications; an unmodified upstream clone reproduces
something different.

Upstream licenses are retained inside each directory and continue to govern
that code. This repository's own MIT license covers only the modifications and
the surrounding harness.

| Directory | Upstream | License | Policies |
|---|---|---|---|
| `dsrnn/` | [Shuijing725/CrowdNav_DSRNN](https://github.com/Shuijing725/CrowdNav_DSRNN) | MIT (© 2021 Shuijing725) | `dsrnn`, `dsrnn_holonomic` |
| `navistar/` | [SAN-NaviSTAR](https://github.com/SMARTlab-Purdue/SAN-NaviSTAR) | MIT (© 2024 Weizheng Wang) | `navistar`, `navistar_holonomic` |
| `height/` | [Shuijing725/CrowdNav_HEIGHT](https://github.com/Shuijing725/CrowdNav_HEIGHT) | MIT (© 2021 Shuijing725) | `height`, `height_holonomic` |
| `sicnav/` | [sepsamavi/safe-interactive-crowdnav](https://github.com/sepsamavi/safe-interactive-crowdnav) | MIT (© 2024 sepsamavi) | `sicnav` |

## What was changed

The same three changes apply to `dsrnn/`, `navistar/` and `height/`:

1. **Task.** Each fork's environment config was moved onto this paper's
   evaluation scenario -- robot and human start/goal sampled uniformly in a
   15 m square with the goal at least 5 m from the start, 1-10 humans per
   episode, radii 0.25 m, `v_pref` 1 m/s, 0.25 s time step, 25 s time limit.
   The upstream `circle_crossing` benchmark is not used. Padding is incorporated to account for varying numbers of humans.
2. **Dynamics and hyperparameters kept as published.** Retraining each policy
   unicycle-native was attempted and did not converge, so each keeps its
   authors' holonomic action space and training recipe. The conversion to a
   shared action space happens at evaluation time instead, by projecting
   (vx, vy) onto the unicycle envelope.
3. **Cluster compatibility.** Optional heavy imports -- pybullet-backed
   TurtleBot environments in particular -- are made lazy so the training code
   runs on machines without those dependencies. No numerical behavior changes.

`sicnav/` is an optimization-based controller with no training step. Its
changes are confined to `sicnav/configs/policy.config`, aligning action limits,
human radius and the assumed maximum human speed with this repository's
environment. `crowd_nav/policy/sicnav.py` documents each value and why.

Every modification carries an inline comment at its site. To see the full
diff against upstream, clone the upstream repository at the commit named in
its README and diff it against the directory here.

## Omitted from the release

Development-time material that does not affect published results was left out:
per-fork SLURM job scripts (replaced by `scripts/slurm/`), smoke tests,
profiling harnesses, and training logs and figures.

`height/` additionally drops the upstream pybullet sim2real path -- the four
TurtleBot environments and their ~5 MB of meshes and textures. That fork
trains and evaluates on the kinematic `CrowdSimVarNum-v0` environment only,
and upstream already guards those imports, so nothing here referenced them.
Restore them from upstream if you want the sim2real setup. Earlier fork variants that
did not produce a paper row -- including four abandoned unicycle-native
retraining attempts and an upstream-faithful reproduction run used only as a
sanity check -- are also omitted; `docs/REPRODUCTION.md` records what they were
and what they showed.
