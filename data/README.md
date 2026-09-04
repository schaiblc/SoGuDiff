# Datasets

Empty by design; `.gitignore` excludes everything here except this file.
`scripts/download_assets.sh` fills it in, and
[../assets/MANIFEST.md](../assets/MANIFEST.md) describes the layout:

```
data/
├── scenes/eval_500/         500 evaluation scenes
├── scenes/<source>/         per-source scene sets, for training
├── maps/                    occupancy-map crops
├── occupancy_maps/          full-scene occupancy maps
└── expert/<source>/         expert demonstrations, the training set
```
