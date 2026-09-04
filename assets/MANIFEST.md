# Released assets

Model weights and datasets are distributed outside the git repository: the
SoGuDiff checkpoints are ~360 MB each and the expert-trajectory dataset is
larger still. `scripts/download_assets.sh` fetches them into the paths below,
which are exactly the paths the shipped configs already point at.

Every path is relative to the repository root.

## Model weights

| Destination | Size | Used by |
|---|---|---|
| `checkpoints/sogudiff_single_axis.pt` | 343 MB | **The model.** Behind every reported SoGuDiff result, and the default in `configs/policy.config` |
| `checkpoints/sogudiff_joint.pt` | 343 MB | Trained on joint and corner demonstrations instead. Needed only for the composition-scheme ablation |
| `baselines/dsrnn/data/trained/checkpoints/best.pt` | 3.7 MB | `dsrnn`, `dsrnn_holonomic` |
| `baselines/navistar/data/navigation/star_sac/checkpoints/best_sac_actor.pt` | 21 MB | `navistar`, `navistar_holonomic` |
| `baselines/height/data/trained/checkpoints/best.pt` | 1.7 MB | `height`, `height_holonomic` |
| `crowdnav_env/crowd_nav/models/cadrl/{il,rl}_model.pth` | 0.2 MB | `cadrl` |
| `crowdnav_env/crowd_nav/models/lstm_rl/{il,rl}_model.pth` | 0.4 MB | `lstm_rl` |
| `crowdnav_env/crowd_nav/models/sarl/{il,rl}_model.pth` | 0.8 MB | `sarl` |
| `crowdnav_env/crowd_nav/models/rgl/{il,rl}_model.pth` | 0.3 MB | `rgl` |

All weights together are about **710 MB**.

The `.config` files inside each `models/<name>/` directory are tracked in git,
because they pin the architecture each checkpoint was trained under and the
evaluation harness reads them from `--model_dir`. Only the `.pth` weights are
downloaded.

`crowdnav_env/crowd_nav/sogudiff_norm_stats.npy` is tracked in git rather than
downloaded — it is 821 bytes of per-channel normalization statistics, and the
diffusion policy is unusable without it.

NaviSTAR's checkpoint keeps its upstream directory layout, which that fork's
training code derives from its task and policy names; the others are written to
a uniform `data/trained/` by their forks' configs.

## Scenes and datasets

These arrive as zstd tarballs. Both sizes are measured: **download** is the
compressed archive, **on disk** is what it expands to.

| Destination | Download | On disk | Used by |
|---|---|---|---|
| `data/scenes/eval_500/` | 20 KB | 5.7 MB | The 500 hand-built adversarial scenes for the style, composition and online-expert experiments. **Not** needed for the comparison table |
| `data/maps/` | 39 MB | 520 MB | 200,000 50x50 navigable crops. Needed only to regenerate scenes |
| `data/expert/custom/` | 316 MB | 5.2 GB | 329,373 demonstrations |
| `data/expert/interiorgs/` | 644 MB | 10.4 GB | 658,114 demonstrations |
| `data/expert/matterport/` | 657 MB | 10.3 GB | 657,278 demonstrations |
| `data/expert/tartanground/` | 1.25 GB | 20.7 GB | 1,316,420 demonstrations |

Demonstrations ship as one archive per source (~16.5 KB per file, 2.96M in
total). They are needed only to retrain from scratch, and training can use any
subset of them via `--dataset_dirs`, so a single source is a valid smaller
starting point.

Each archive also carries the generator's `.done` resume markers, so pointing
`generate_expert_trajectories.py` at an extracted directory continues from
where it left off instead of regenerating. Training ignores them: the loader
globs `*.npz` and `*.npy` only.

The generated scenes themselves (5.2 GB on disk) are not distributed: they are
an intermediate, and `scenegen/` regenerates them deterministically from the
maps.

The paper's comparison table needs **no** scene download: those 500 scenes are
generated procedurally from `configs/env.config` with fixed per-scene seeds
(see docs/RUNNING.md, "Two scene sources"). `scripts/download_assets.sh weights`
is therefore enough to reproduce it.

The 500-scene style set is small either way, and rebuilding it locally takes
under a minute:

    python scenegen/build_eval_set_500.py --out_dir data/scenes/eval_500 --seed 0

Training selects 1,486,283 of the 2.96M demonstrations by style arm
(`--arm_counts`), reading the source directories in place, so nothing is merged
or subsampled on disk. Every stage that produces them ships in `scenegen/` and
`sogudiff/` and takes a count argument, so a smaller dataset can be regenerated
end to end; docs/RUNNING.md, "Regenerating the dataset", gives the per-source
breakdown and the exact commands.

## Where these come from

All 19 files are attached to this repository's GitHub Release, and
`scripts/download_assets.sh` fetches them by exact filename. Filenames are flat
and significant; the local destinations in the tables above are created for
you.

Every file carries a sha256 checksum in the script, so a truncated or corrupted
download is caught rather than silently used.

To host the assets elsewhere — a mirror, an institutional server, a Zenodo
record — point `SOGUDIFF_ASSET_URL` at any flat-namespace location, keeping the
filenames unchanged.

## Third-party data licenses

Occupancy maps are derived from datasets with their own terms, which this
repository does not relicense. Obtain each from its source and accept its
license before regenerating maps:

- **Matterport3D (HM3D)** — <https://github.com/matterport/habitat-matterport-3dresearch/>
- **InteriorGS** — <https://huggingface.co/datasets/spatialverse/InteriorGS>
- **TartanGround** — <https://tartanair.org/tartanground.html>

Redistributed map derivatives cover only what those licenses permit; when in
doubt, regenerate from your own copy of the source scans using `scenegen/`.
