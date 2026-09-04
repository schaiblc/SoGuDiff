# Third-party code

This repository is MIT licensed (see [LICENSE](LICENSE)), but it incorporates
code from several other projects, each of which keeps its own license. This
file records what came from where.

## Simulator and evaluation harness — `crowdnav_env/`

Built on **CrowdNav** (MIT, © 2018 VITA lab at EPFL) —
<https://github.com/vita-epfl/CrowdNav>. Its license is retained at
[`crowdnav_env/LICENSE`](crowdnav_env/LICENSE). The environment, the
`crowd_nav`/`crowd_sim` package layout, and the CADRL, LSTM-RL and SARL
policies derive from it.

Additional policies in that tree:

| Component | Source | License |
|---|---|---|
| `crowd_nav/policy/rgl.py` | [ChanganVR/RelationalGraphLearning](https://github.com/ChanganVR/RelationalGraphLearning) (Chen et al., ICRA 2020) | MIT |
| `crowd_sim/envs/policy/social_force.py` | [Shuijing725/CrowdNav_Prediction_AttnGraph](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph) (ICRA 2023) | MIT |
| `crowd_sim/envs/policy/orca.py` | ORCA via [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) | Apache 2.0 |

The `*_unicycle` variants of ORCA and SFM, the diffusion policy, the projection
layer and the evaluation harness are this work.

## Denoising backbone — `diffusers_unet_1d_condition/`

Vendored **unmodified** from
[lucainiaoge/diffusers-unet-1d-condition](https://github.com/lucainiaoge/diffusers-unet-1d-condition),
itself an adaptation of [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
v0.30.3 `UNet2DConditionModel`. Apache 2.0 — see
[`diffusers_unet_1d_condition/LICENSE`](diffusers_unet_1d_condition/LICENSE)
and that directory's README.

## Baselines — `baselines/`

Four forks of published navigation methods, modified to train on this paper's
task. Upstream URLs, licenses and the exact changes are in
[`baselines/NOTICE.md`](baselines/NOTICE.md).

## Solvers and datasets

- **acados** — the projection layer's optimal-control solver.
  <https://github.com/acados/acados>, BSD 2-clause. Not vendored; installed
  separately (see [docs/INSTALL.md](docs/INSTALL.md)).
- **Matterport3D**, **InteriorGS**, **TartanGround** — scan datasets behind the
  occupancy maps, each under its own terms. Not redistributed here; see
  [assets/MANIFEST.md](assets/MANIFEST.md).
