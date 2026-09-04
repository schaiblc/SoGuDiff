# Running experiments

Every experiment in this repository is a plain `python` command. The SLURM
scripts under `scripts/slurm/` only activate an environment and run one of
these; each spells its command out in its header comment, so nothing here
depends on having a scheduler.

If you do use SLURM, copy `scripts/slurm/cluster_env.sh` to
`cluster_env.local.sh` (gitignored) and set your account, partition and
environment activation there. Then:

| Script | Runs |
|---|---|
| `evaluate.slurm <policy> [model_dir]` | one policy over the benchmark |
| `evaluate_sicnav.slurm` | SICNav, which needs its own interpreter |
| `train_sogudiff.slurm` | a full training run, resuming from the newest checkpoint |
| `generate_expert_data.slurm` | expert demonstrations for one scene directory |

`_common.sh` is the shared preamble those source; it is not run directly.

Unless stated otherwise, run from `crowdnav_env/crowd_nav/`.

## Conventions

Shared across the evaluation entry points:

| Flag | Meaning |
|---|---|
| `--policy` | Policy name from `crowd_nav/policy/policy_factory.py` |
| `--model_dir` | For baselines whose architecture is pinned by a directory (CADRL, LSTM-RL, SARL, RGL) |
| `--npz_hard --npz_dir DIR --npz_num_eval N` | Evaluate on `N` scenes from a saved scene set |
| `--results_suffix TAG` | Write to `results_TAG/` |
| `--no_video` | Skip rendering. On large runs this is most of the wall time |
| `--gpu` | Use CUDA where the policy supports it |
| `--infer_mode {joint,per_axis}` | Diffusion only: apply guidance jointly or per style axis |
| `--w_normalize` | Diffusion only: normalize guidance weights across axes |

The paper's configuration for the proposed method is `--infer_mode per_axis
--w_normalize`, with the style vector from `configs/policy.config`.

Scene indices are the evaluation loop's 0-based counter, while the renderer
prints episode numbers 1-based. The scene shown as "216" is index 215.

## Two scene sources

This matters more than any single flag, because it decides which table a run
reproduces.

**Procedural (the default).** With no `--npz_hard`, scenes are generated from
`configs/env.config` — `test_sim = evaluation`, `test_size = 500`,
`testoffset = 10` — seeding each episode by its index. This is deterministic
and identical across methods, and it is what the paper's **comparison table**
uses. No download required.

**Saved scene sets.** With `--npz_hard --npz_dir DIR`, scenes are loaded from
`.npz` files. These are the hand-built sets behind the **style, composition and
online-expert experiments**, where a fixed set of adversarial encounters is
replayed under every style variant. A scene is used if it carries a
`threat_type` or is labelled `difficulty = hard`; `--npz_num_eval` caps how
many are taken.

Either download the set:

```bash
scripts/download_assets.sh eval-scenes        # -> data/scenes/eval_500
```

or rebuild it — it is deterministic given the seed, and takes under a minute:

```bash
python scenegen/build_eval_set_500.py --out_dir data/scenes/eval_500 --seed 0
```

`--npz_num_eval` has no effect without `--npz_hard`: the procedural path takes
its episode count from `env.config`.

## Scoring one policy

```bash
python evaluate.py \
    --policy sogudiff \
    --infer_mode per_axis --w_normalize --gpu --no_video \
    --results_suffix sogudiff
```

Writes `results_sogudiff/` containing a summary text file and per-scene metrics
as CSV: success, collision and timeout rates; time and path length to goal;
clearance; and the social metrics shared with the style sweep — minimum
distance, PSI rate, passing-side bias, time-to-collision infractions,
path-cutoff rate and group-split rate.

SICNav runs identically but from its own interpreter, and needs
`configs/policy_sicnav.config`:

```bash
PYTHONPATH=../..:$PYTHONPATH "$SOGUDIFF_SICNAV_PYTHON" evaluate.py \
    --policy sicnav --policy_config configs/policy_sicnav.config \
    --phase test --no_video --results_suffix sicnav
```

## The whole comparison table

```bash
scripts/run_baseline_table.sh                    # every method, 500 scenes
scripts/run_baseline_table.sh --only ORCA,SFM    # a subset first
```

Evaluate one baseline per process. Each vendored fork ships its own top-level
`crowd_sim`/`crowd_nav` packages, so only one can be loaded per interpreter —
which is why the runner shells out per method rather than looping in-process.

## Style experiments

**Fixed grid.** Ten variants — each of the four axes at ±1, plus neutral and a
neutral-without-projection ablation — over every scene, rendered as one
time-synchronized multi-panel video per scene:

```bash
python evaluate_styles.py \
    --policy sogudiff \
    --infer_mode per_axis --w_normalize --gpu \
    --npz_hard --npz_dir ../../data/scenes/eval_500 --npz_num_eval 500 \
    --results_suffix STYLES
```

**Axis sweeps** `evaluate_styles.py` only visits ±1, so it
shows an axis has an effect but not that the effect scales with the style
scalar. Sweep one axis with the others held at zero:

```bash
for ax in prox pass yield group; do
    python evaluate_axis_sweep.py --axis "$ax" \
        --values -1,-0.5,0.5,1 \
        --policy sogudiff \
        --infer_mode per_axis --w_normalize --gpu --no_video \
        --npz_hard --npz_dir ../../data/scenes/eval_500 --npz_num_eval 500 \
        --results_suffix "SWEEP_$ax"
done
```

**Composed styles.** Arbitrary full 4-axis vectors — "cautious *and* yielding"
is `1,0,1,0` — scored over the same scenes with the same metrics. Each
composition is given as `LABEL:prox,pass,yield,group`, repeated per style;
omitting the flag uses a six-composition default set:

```bash
python evaluate_composition.py \
    --composition "Neutral:0,0,0,0" \
    --composition "Cautious:1,0,0,0" \
    --composition "Cautious & Yielding:1,0,1,0" \
    --composition "Social Passing:0,1,0,1" \
    --policy sogudiff \
    --infer_mode per_axis --w_normalize --gpu \
    --npz_hard --npz_dir ../../data/scenes/eval_500 --npz_num_eval 500 \
    --results_suffix COMPOSITION
```

**Baselines on the style scenes.** `evaluate_baselines.py` scores the unstyled
baselines on the same saved scenes the style sweep uses, computing the full
social-metric set rather than the two `evaluate.py` reports, so styled and
unstyled rows are directly comparable:

```bash
python evaluate_baselines.py \
    --baselines CADRL,LSTM-RL,SARL,RGL,ORCA,SFM \
    --gpu --no_video \
    --npz_hard --npz_dir ../../data/scenes/eval_500 --npz_num_eval 500 \
    --results_suffix baselines_on_style_scenes
```

**Figures.** The scripts in `crowd_nav/figures/` take one hand-picked scene and
draw many stochastic rollouts per style, rather than aggregating over a scene
set. They share their rollout and metric code with `evaluate_styles.py`, so a
figure shows exactly what the evaluation scores:

| Script | Figure |
|---|---|
| `style_sweep_figure.py` | One axis swept over {-1, 0, +1} on a single scene, all rollouts overlaid |
| `style_composition_figure.py` | Several composed style vectors on a single scene |
| `style_composition_step_figure.py` | The raw diffusion candidate pool at a single instant, before scoring and projection |

## The online-expert ablation

`mppi_expert` runs the demonstration generator's own sampling-based planner
online as a policy, against the same projection back-end as SoGuDiff. Because
the two share the feasibility layer, a difference in metrics is attributable to
the planner rather than to projection.

It is configured at the generator's full search budget, which is the setting
the paper's ablation reports:

```bash
python evaluate_styles.py --policy mppi_expert \
    --gpu --no_video \
    --npz_hard --npz_dir ../../data/scenes/eval_500 --npz_num_eval 500 \
    --results_suffix online_expert
```

This is slow — the planner does thousands of rollouts per control step. Reduce `--npz_num_eval` to sanity-check it
first.

`mppi_budget` in `configs/policy.config` scales that search down if you want to
probe cheaper settings; `evaluate_styles.py` also exposes it as
`--mppi_budget`. Restarts and iteration count are deliberately not scaled by
it, since they define the planner rather than its compute — see the comment in
the config.

## Videos

Every evaluation entry point renders when `--visualize` is passed and
`--no_video` is omitted, concatenating the run's episodes into `--video_file`:

```bash
python evaluate.py --policy sogudiff \
    --infer_mode per_axis --w_normalize --gpu --visualize \
    --npz_hard --npz_dir ../../data/scenes/eval_500 --npz_num_eval 20 \
    --video_file sogudiff.mp4 --results_suffix sogudiff_video
```

Rendering dominates wall time, so keep the scene count small. The style
scripts instead produce one time-synchronized multi-panel video per scene, which
is the more useful form for comparing variants.

To inspect a specific failure mode, collect the scenes of interest into their
own directory and point `evaluate_scene_subset.py` at it; it renders one panel
per scene for a single variant.

## Training the RL baselines

CADRL, LSTM-RL, SARL and RGL train through `train_rl_baseline.py`, which writes
`output.log` plus `il_model.pth`/`rl_model.pth` into its output directory:

```bash
python train_rl_baseline.py --policy sarl --output_dir models/sarl_new
python utils/plot.py models/sarl_new/output.log --plot_sr --plot_reward
```

`utils/plot.py` draws success-rate, collision-rate and reward curves from one
or more of those logs. DSRNN, NaviSTAR, HEIGHT and SICNav do not train here —
see `baselines/` and [NOTICE.md](../baselines/NOTICE.md).

## Training SoGuDiff

From the repository root. Two ways to point at data:

**One directory** — everything in a single folder:

```bash
python sogudiff/train.py \
    --dataset_dir data/expert_trajectories \
    --exp_name my_model --cfg_mode union --use_map \
    --batch_size 128 --lr 2e-4 --num_steps 500000 \
    --num_diffusion_timesteps 100 --horizon 32 --token_dim 128
```

**Several directories, selecting by style arm** — `--dataset_dirs` takes any
number of demonstration folders and resolves full paths, so sources need no
prefixing or merging. `--arm_counts` then chooses which style arms to train on
and how many of each (`all`, or a cap). This is how the released checkpoints
were produced:

```bash
# checkpoints/sogudiff_single_axis.pt -- the reported model
python sogudiff/train.py \
    --dataset_dirs data/expert/custom data/expert/interiorgs \
                   data/expert/matterport data/expert/tartanground \
    --arm_counts "neutral=all,axis=all" \
    --file_seed 42 --exp_name sogudiff_single_axis \
    --cfg_mode union --use_map --batch_size 128 --lr 2e-4 \
    --num_steps 1000000 --num_diffusion_timesteps 100 \
    --horizon 32 --token_dim 128 --save_every 10000

# checkpoints/sogudiff_joint.pt -- only for the composition-scheme ablation
python sogudiff/train.py \
    --dataset_dirs <same four directories> \
    --arm_counts "neutral=all,joint+corner=990634" \
    --file_seed 42 --exp_name sogudiff_joint \
    ...otherwise identical...
```

The arm is read from the filename after `__` (`neutral`, `axis_*`, `joint*`,
`corner*`), so it works whatever prefix the generator used. `--file_seed` fixes
which files a capped arm selects, so truncations nest reproducibly.

| Flag | Meaning |
|---|---|
| `--cfg_mode` | How style conditioning is dropped for classifier-free guidance. `union` is the reported setting |
| `--use_map` | Condition on the static occupancy map as well as the crowd |
| `--horizon` | Trajectory length in waypoints |
| `--token_dim` | Conditioning token width |
| `--arm_counts` | Which style arms to train on, and how many of each |
| `--num_workers 0` | Run the data pipeline in-process; use when debugging |
| `--no_wandb` | Disable experiment tracking |

Checkpoints go to `checkpoints/<exp_name>/` as `ckpt_step<N>_<exp_name>.pt`,
alongside `norm_stats_<exp_name>.npy` — the per-channel normalization
statistics computed from that dataset. Training resumes from the newest
checkpoint there, so re-running continues rather than restarting. Roughly one
day on a single 40 GB GPU for 500k steps, extend to 1M if validation loss has not plateaued.

To evaluate your own run, point `ckpt_path` and `norm_file` under `[sogudiff]`
in `configs/policy.config` at your `ckpt_step<N>_<exp_name>.pt` and
`norm_stats_<exp_name>.npy`. Both accept repository-relative paths.

### Training on your own demonstrations

Nothing downstream of the demonstrations is tied to the sampling-based planner
that produced ours. `train.py` reads `.npz` files and never asks where they came
from, so any source of style-labelled trajectories works: a different planner,
an optimization-based expert, human teleoperation, or logged demonstrations.

Each `.npz` needs:

| Field | Shape | Required |
|---|---|---|
| `traj_xy` (or `trajectory`) | `(horizon, 2)` | **yes** — the demonstrated path |
| `style_values` | `(4,)` | **yes** for style conditioning: the style vector this trajectory demonstrates, roughly in [-1, 1] |
| `start_state` | `(2,)` | no — defaults to the first waypoint |
| `goal` | `(2,)` | no — defaults to the last waypoint |
| `obstacles` | `(k, 4)` | no — defaults to none. Each row is a pedestrian as (x, y, vx, vy) |
| `occupancy_map` | `(map_size, map_size)` | no — needed with `--use_map`. Ego-centerd, robot at the center cell, +x forward |
| `has_map` | scalar | no — inferred from whether `occupancy_map` is present |

Anything else in the file is ignored, so the extra diagnostic fields our
generator writes are optional.

The filename carries one contract: `<anything>__<arm><...>.npz`, where `<arm>`
is `neutral`, `axis_<name><sign>`, `joint<n>` or `corner<n>`. That suffix is
what `--arm_counts` selects on. If you do not need arm selection, use
`--dataset_dir` and the naming does not matter.

Everything else is unaffected: the guidance scheme, the projection layer, the
evaluation harness and the baselines all stay as they are, so a model trained
on your own demonstrations is scored by exactly the same commands.

## Regenerating the dataset

Only needed to train from scratch on new scenes; the released dataset is this
pipeline's output.

### What the released dataset is made of

Four map sources, combined so the model sees both photorealistic indoor layouts
and procedurally hard geometry:

| Source | Occupancy maps | 50x50 crops | Scenes | Demonstrations |
|---|---:|---:|---:|---:|
| Custom (procedural, no scan) | — | — | 50,000 | 329,373 |
| InteriorGS | 1,000 | 50,000 | 100,000 | 658,114 |
| Matterport3D | 799 | 50,000 | 100,000 | 657,278 |
| TartanGround | 63 | 100,000 | 200,000 | 1,316,420 |
| **Total** | **1,862** | **200,000** | **450,000** | **2,961,185** |

TartanGround has few environments but they are large, so it yields the most
crops per map and was generated in two halves (`--offset 0` and
`--offset 100000`) to fit inside job walltimes.

Those 2,961,185 demonstrations split by style arm as:

| Arm | Demonstrations | Share |
|---|---:|---:|
| neutral | 495,649 | 16.7% |
| axis (single-axis) | 990,634 | 33.5% |
| joint | 1,233,309 | 41.6% |
| corner | 241,593 | 8.2% |

**Training does not use all of them.** Both released checkpoints select
**1,486,283 demonstrations** from this pool with `--arm_counts`, and the two
selections are deliberately size-matched so the composition-scheme ablation
compares like with like:

| Checkpoint | `--arm_counts` | Selected |
|---|---|---:|
| `sogudiff_single_axis.pt` | `neutral=all,axis=all` | 495,649 + 990,634 = 1,486,283 |
| `sogudiff_joint.pt` | `neutral=all,joint+corner=990634` | 495,649 + 990,634 = 1,486,283 |

Nothing is merged or subsampled on disk — `--dataset_dirs` reads the four
source directories in place and `--arm_counts` picks from them, with
`--file_seed` fixing which files a capped arm takes.

**These counts are what the reported models used, not a requirement.** Every
stage takes a count argument and the pipeline is linear in it, so scaling all
four sources down by the same factor preserves the mix and trains a working
model on a single machine. Expert generation is the expensive stage by a wide
margin — CPU-bound and resumable, so it can also be grown incrementally.

### Stage 1 — occupancy maps from raw scans

One script per source. Each writes a flat directory of
`<name>_occupancy.png` + `<name>_metadata.json` pairs (255 free, 0 obstacle),
which is what stage 2 globs for. TartanGround names its pairs
`<env>_sem_occupancy.png`; the extra `_sem` simply becomes part of the scene
name and needs no special handling.

```bash
# Matterport3D: raycast navmesh from .glb meshes  (scenegen environment)
python scenegen/occupancy_from_matterport.py \
    --root data/raw/matterport --output data/occupancy_maps/matterport --resolution 0.05

# TartanGround: project semantic point clouds  (main environment)
# --env picks one environment; omit it to process all of them.
python scenegen/occupancy_from_tartanground.py \
    --root_dir data/raw/tartanground --output_dir data/occupancy_maps/tartanground

# InteriorGS: ships occupancy grids already, so this only flattens and
# binarizes them  (main environment)
python scenegen/occupancy_from_interiorgs.py \
    --root data/raw/interiorgs --out data/occupancy_maps/interiorgs
```

Only the Matterport step needs the `scenegen` environment (`habitat-sim`). Each
source has its own license — see [assets/MANIFEST.md](../assets/MANIFEST.md).
The custom source has no stage 1: its geometry is generated procedurally in
stage 3.

Each script expects the layout its dataset is distributed in:

| Source | Expected under the root |
|---|---|
| Matterport3D (HM3D) | one folder per scene, each containing `<id>.basis.glb`; found recursively |
| TartanGround | one folder per environment, containing the semantic `.pcd` and `seg_labels.zip` |
| InteriorGS | one folder per scene, each containing `occupancy.png` and `occupancy.json` |

A scene without the expected file is skipped with a message rather than
failing the run — in the HM3D release used here, 1 of 800 scenes ships a
navmesh but no mesh, which is why the map count is 799.

### Stage 2 — crop navigable regions

```bash
python scenegen/generate_random_maps.py \
    --input_dir data/occupancy_maps/matterport \
    --output_dir data/maps/matterport \
    --n_samples 50000 --max_tries_per_scene 20
```

Writes one `maps.npy` of 50x50 crops plus metadata. Repeat per source;
TartanGround used `--n_samples 100000`.

### Stage 3 — populate crops with crowds, goals and obstacles

```bash
python scenegen/generate_scenes.py \
    --maps_dir data/maps/matterport --save_dir data/scenes/matterport --n_samples 100000

# the custom source, which needs no map input
python scenegen/generate_scenes_custom.py \
    --save_dir data/scenes/custom --n_samples 50000 --seed 42
```

### Stage 4 — expert demonstrations

Run once per source. Keeping one output directory per source is what the
released checkpoints used, because training reads several directories directly:

```bash
for src in custom interiorgs matterport tartanground; do
    python sogudiff/generate_expert_trajectories.py \
        --in_dir  data/scenes/$src \
        --out_dir data/expert/$src \
        --workers 64 --style_threads 1 --n_modes 4
done
```

Then train with `--dataset_dirs data/expert/*` (see "Training SoGuDiff"). No
merge step, and no subsampling on disk: `--arm_counts` selects at load time.

If you would rather have everything in **one** directory, that works too, but
give each source its own `--out_prefix`:

```bash
python sogudiff/generate_expert_trajectories.py \
    --in_dir data/scenes/matterport --out_dir data/expert_trajectories \
    --out_prefix matterport_ --workers 64 --n_modes 4
```

The prefix is not optional in that case. Every scene directory numbers its
scenes `sample_0.npz`, `sample_1.npz`, ... independently, so the same filename
exists in all four sources. Without a distinct prefix the second source would
overwrite the first's demonstrations and, worse, be silently skipped by the
first's resume markers. Arm selection is unaffected either way, since the arm
is read from the part of the filename after `__`.

CPU-bound and resumable: a scene counts as complete only once its atomic
`<prefix><scene>.done` marker is written, after every style and mode has been
saved. Both the generator's skip logic and the SLURM wrapper's progress count
read those markers rather than testing for output files, so a job killed
mid-scene regenerates that scene in full instead of leaving it half-written.
Re-run the same command to resume; `--force_redo` starts over, and only clears
markers matching that source's prefix.

To split one source across jobs, give each shard its own `--offset`/`--limit`.
Shards of the same source may share an output directory, since scene numbering
within a source is already unique. TartanGround was generated this way, in two
halves of 100,000 scenes.

Keep the thread budget bounded: `workers x style_threads x NUMBA_NUM_THREADS`
should not exceed the cores available, or the sampling kernels oversubscribe
badly. `export NUMBA_NUM_THREADS=1` with a high `--workers` is the usual choice.
