# Installation

Three environments exist, and most users need only the first.

| Environment | Python | Needed for |
|---|---|---|
| **main** (`requirements/main.txt`) | 3.10 | SoGuDiff training and inference, the evaluation harness, and every baseline except SICNav |
| **sicnav** (`requirements/sicnav.txt`) | 3.8 | The SICNav baseline only |
| **scenegen** (`requirements/scenegen.txt`) | 3.9 | Rebuilding occupancy maps from raw 3D scans only |

They are separate because their constraints genuinely conflict, not out of
caution. SICNav reaches IPOPT through CasADi's plugin system, which resolves
reliably only against the conda-forge build on Python 3.8; scene generation
needs `habitat-sim`, which ships compiled renderer binaries for its own Python
version. The evaluation harness is environment-agnostic, so a policy can be
scored from whichever interpreter it needs.

## Main environment

Run every command from the repository root. The order matters — `gym` must go
in first, under its own toolchain, before anything else is installed:

```bash
python3.10 -m venv .venv && source .venv/bin/activate

# 1. gym 0.21 only builds under this exact toolchain (see below).
pip install "pip==23.3.2"
pip install "setuptools==65.5.0" "wheel==0.38.4"
pip install --no-build-isolation "gym==0.21.0"

# 2. everything else.
pip install -r requirements/main.txt
pip install -e crowdnav_env
```

Then build Python-RVO2, which has no PyPI distribution — this is **not**
optional, and not only for SICNav: `crowd_sim/envs/crowd_sim.py` imports `rvo2`
at module level, so every policy needs it. It compiles a small C++ library, so
`cmake` and a C++ compiler must be available:

```bash
pip install "Cython<3"
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2
python setup.py build && pip install --no-build-isolation .
cd ..
```

Do **not** use Python-RVO2's own `requirements.txt`: it pins `Cython==0.21.1`,
which does not build on Python 3.10 (`cannot import name 'build_py_2to3'`).

Three pins carry real constraints:

- **`numpy==1.26.4`.** Both `numba` 0.61 and `gym` 0.21 break against the
  NumPy 2 ABI, and the expert generator's numba kernels are on the hot path.
- **`gym==0.21.0`.** The environments and every vendored baseline fork use the
  pre-0.22 `step`/`reset` API. Its `setup.py` declares `opencv-python>=3.`,
  which is not a valid PEP 440 specifier, so it is rejected by setuptools ≥ 66
  (at build), by wheel ≥ 0.40 (during metadata generation) and by pip ≥ 24.1
  (when the finished wheel is read back). Pinning setuptools alone is not
  enough — pip builds in an isolated environment with its own newest
  setuptools, which is why `--no-build-isolation` is required.
- **`pip==23.3.2`.** Any pip < 24.1 works. You can upgrade pip again after
  `gym` is installed; it is only the install of that one package that needs it.

If `gym` is installed as part of `requirements/main.txt` rather than before it,
the failure aborts the whole file and **nothing else gets installed** — while a
subsequent `pip install -e crowdnav_env` may still appear to succeed.

For a non-default CUDA version, install torch first from the
[official index](https://pytorch.org/get-started/locally/), then the rest.

### Optional: experiment tracking

Weights & Biases is **not** required and is commented out in
`requirements/main.txt`. `sogudiff/train.py` imports it behind a try/except and
substitutes a no-op if it is missing, so a fresh clone trains without a W&B
account and still prints every metric to stdout. If you do install it, pass
`--no_wandb` to disable tracking for a run, or `--wandb_project NAME` to choose
the project.

Verify — this needs no downloaded weights:

```bash
cd crowdnav_env/crowd_nav
python evaluate.py --policy orca_unicycle --phase test --no_video
```

Scenes are generated procedurally here, so nothing needs downloading. Expect
500 episodes in under a minute.

## acados

The diffusion policy's projection layer and the MPPI ablation compile their
optimal-control problem through [acados](https://docs.acados.org). It is a C
library with a Python interface and cannot be installed from PyPI alone.
**Every other policy runs without it**, so skip this section if you only want
to reproduce the baseline table.

```bash
git clone https://github.com/acados/acados.git third_party/acados
cd third_party/acados && git submodule update --recursive --init
mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
cd ../../..

pip install -e third_party/acados/interfaces/acados_template

export ACADOS_SOURCE_DIR="$PWD/third_party/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH"
```

Those two exports are needed in every shell that runs the diffusion policy; put
them in your shell profile or in `scripts/slurm/cluster_env.local.sh`. On first
use acados compiles a C project into `c_generated_proj*/`, which takes a minute
and is gitignored.

Check it:

```bash
python -c "import acados_template; print('acados OK')"
```

Without acados, `policy_factory` still imports and every other policy works;
only requesting `sogudiff` or `mppi_expert` fails, with
a message naming the missing package.

## SICNav environment

```bash
conda create -n sicnav python=3.8 && conda activate sicnav
conda install -c conda-forge ipopt casadi=3.6.5

# gym 0.21 needs the same toolchain as the main environment
pip install "pip==23.3.2"
pip install "setuptools==65.5.0" "wheel==0.38.4"
pip install --no-build-isolation "gym==0.21.0"

pip install -r requirements/sicnav.txt
pip install -e crowdnav_env
pip install -e baselines/sicnav
```

This environment needs Python-RVO2 too — build it exactly as in the main
environment above, from this interpreter:

```bash
pip install "Cython<3"
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2
python setup.py build && pip install --no-build-isolation .
```

Point the runner scripts at this interpreter:

```bash
export SOGUDIFF_SICNAV_PYTHON="$(conda run -n sicnav which python)"
```

## Scene-generation environment

Needed only to rebuild occupancy maps from raw 3D scans. The released scene
sets are this stage's output, and everything downstream of it runs in the main
environment.

```bash
conda create -n sogudiff-scenegen python=3.9 && conda activate sogudiff-scenegen
conda install -c conda-forge -c aihabitat habitat-sim=0.3.3 headless
pip install -r requirements/scenegen.txt
```

You also need your own licensed copy of Matterport3D, InteriorGS or
TartanGround; see [assets/MANIFEST.md](../assets/MANIFEST.md).

## Assets

Weights and datasets are attached to this repository's GitHub Release.
`scripts/download_assets.sh` places every file where the shipped configs expect
it and verifies each against a sha256 checksum, so a truncated download is
caught rather than silently used.

```bash
scripts/download_assets.sh               # weights + 500-scene eval set (~715 MB)
scripts/download_assets.sh weights       # weights only (~710 MB)
scripts/download_assets.sh all           # adds maps and demos (~3.6 GB down)
```

Files land exactly where the shipped configs expect them. Re-running skips what
is already present.

## Troubleshooting

**`error in gym setup command: 'extras_require' must be a dictionary ...`**, or
**`InvalidRequirement: ... opencv-python>=3.`** — the `gym` toolchain above was
not used. Pin pip, setuptools and wheel and install `gym` with
`--no-build-isolation` *before* `requirements/main.txt`.

**`ModuleNotFoundError: No module named 'rvo2'`** — Python-RVO2 was not built.
It is required by every policy, not only SICNav; see the main environment
section.

**`ImportError: cannot import name 'build_py_2to3'`** while building
Python-RVO2 — its `requirements.txt` pinned `Cython==0.21.1`. Install
`"Cython<3"` instead and build with `python setup.py build`.

**`ld: cannot find /lib64/libm.so.6`** while building Python-RVO2, usually in
the Python 3.8 SICNav environment. This appears on systems that do not lay
libraries out under `/lib64` — a Gentoo Prefix, Nix, or an HPC software stack
mounted somewhere else — where conda's own toolchain cannot find the host C
library. It does not occur on a distribution with a conventional `/lib64`.
Give the environment a self-contained toolchain and stop Python from
overriding its sysroot:

```bash
conda install -c conda-forge gcc_linux-64 gxx_linux-64 sysroot_linux-64 libxcrypt
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CPATH="$CONDA_PREFIX/include"
# Python 3.8's build config injects `-Wl,--sysroot=/`, which sends the linker
# to the host's absent /lib64 instead of the toolchain's own sysroot.
export LDSHARED="$CXX -pthread -shared"
python setup.py build && pip install --no-build-isolation .
```

`libxcrypt` is needed because Python 3.8's `Python.h` includes `crypt.h`,
which newer sysroots no longer ship (`fatal error: crypt.h: No such file or
directory`).

**Asset download returns 404 or "Not Found".** The release may not be
published yet, or you may be working from a fork without access to it. Point
the script at your own copy with `SOGUDIFF_ASSET_URL`, or set `GITHUB_TOKEN`
(read access) if you have been given access to a pre-release copy.

**`ImportError: ... requires acados_template`** — expected when acados is not
installed; only the diffusion policy and MPPI ablation need it.

**`ModuleNotFoundError: crowd_sim`** — `pip install -e crowdnav_env` was not
run, or a different environment is active.

**A baseline fails with a `crowd_sim` attribute error.** Each vendored fork
ships its own top-level `crowd_sim`/`crowd_nav` packages, whose names collide
with this repository's. The adapters isolate those imports, but only one fork
can be loaded per process — so evaluate one baseline per process rather than
looping over several in a single script.

**Paths not found after moving the repository.** Configs use repository-relative
paths resolved by `crowd_nav/paths.py`. If you run from an unusual working
directory or an installed (non-editable) package, set `SOGUDIFF_ROOT` to the
repository root.

**CUDA out of memory during training.** Lower `--batch_size`; 128 assumes a
40 GB card. Evaluation is far lighter and runs on 8 GB.
