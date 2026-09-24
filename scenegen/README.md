# Scene generation

Builds the evaluation and training scene sets, in pipeline order.

| File | Stage |
|---|---|
| `occupancy_from_matterport.py` | Matterport3D `.glb` meshes to 2D occupancy maps (needs `habitat-sim`) |
| `occupancy_from_tartanground.py` | TartanGround semantic point clouds to occupancy maps |
| `occupancy_from_interiorgs.py` | Flattens and binarizes the occupancy grids InteriorGS already ships |
| `generate_random_maps.py` | Samples 50x50 navigable crops out of those occupancy maps |
| `generate_scenes.py` | Populates crops with crowds, goals and static obstacles |
| `generate_scenes_custom.py` | Procedural layouts needing no map input |
| `interiorgs_scene_builder.py` | Interactive builder for one-off InteriorGS scenes |
| `style_probe_scenes.py` | Library of hand-designed style-probe scene categories; runnable on its own |
| `build_eval_set_500.py` | Assembles the 500-scene `.npz` set used by the style experiments |

Only `occupancy_from_matterport.py` needs the separate `scenegen` environment
(`requirements/scenegen.txt`); everything downstream runs in the main one.

The scene sets are this pipeline's output.
Regenerating maps requires your own licensed copy of the source datasets.

See [../docs/RUNNING.md](../docs/RUNNING.md) for the full command sequence.
