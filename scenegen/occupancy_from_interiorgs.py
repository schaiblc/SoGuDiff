"""Convert InteriorGS scene folders into the occupancy-map layout used here.

InteriorGS ships a rendered occupancy grid with every scene, so unlike
Matterport3D and TartanGround there is nothing to raycast -- this stage only
flattens the per-scene folders into one directory, binarizes the grid, and
fills in the metadata fields the cropping stage reads.

Input (one directory per scene, as distributed):

    <root>/<scene_id>/occupancy.png     uint8, 255 free / 127 unknown / 0 occupied
    <root>/<scene_id>/occupancy.json    {scale, center, lower, upper, min, max}

Output (flat, matching occupancy_from_matterport.py and
occupancy_from_tartanground.py):

    <out>/<scene_id>_occupancy.png      uint8, 255 free / 0 obstacle
    <out>/<scene_id>_metadata.json      the source fields plus metres_per_pixel,
                                        resolution and the pixel convention

Unknown cells (127) are folded into obstacle rather than free: they are space
the scan never observed, and treating them as navigable would let the planner
route through unmapped regions.

Usage:
    python scenegen/occupancy_from_interiorgs.py \
        --root data/raw/interiorgs --out data/occupancy_maps/interiorgs
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

FREE, OBSTACLE = 255, 0


def convert_scene(scene_dir, out_dir, scene_id):
    """Convert one scene folder. Returns True if it produced output."""
    png_in = os.path.join(scene_dir, "occupancy.png")
    json_in = os.path.join(scene_dir, "occupancy.json")
    if not (os.path.isfile(png_in) and os.path.isfile(json_in)):
        return False

    grid = np.array(Image.open(png_in).convert("L"))
    # Free only where the source says free; unknown (127) becomes obstacle.
    binary = np.where(grid == FREE, FREE, OBSTACLE).astype(np.uint8)
    Image.fromarray(binary).save(os.path.join(out_dir, f"{scene_id}_occupancy.png"))

    meta = json.load(open(json_in))
    h, w = binary.shape

    # Derive per-axis pixel size from the world bounds where they are present,
    # so a non-square pixel is not silently squared off. 'scale' is the
    # nominal value and is kept as the fallback.
    scale = float(meta.get("scale", 0.05))
    lo, hi = meta.get("min"), meta.get("max")
    if lo is not None and hi is not None and w > 0 and h > 0:
        mpp = {"x": (hi[0] - lo[0]) / w, "y": (hi[1] - lo[1]) / h}
    else:
        mpp = {"x": scale, "y": scale}

    meta.update({
        "metres_per_pixel": mpp,
        "original_resolution": [w, h],
        "output_resolution": [w, h],
        "convention": {
            "free": FREE,
            "obstacle": OBSTACLE,
            "note": (
                "Binary map. Obstacle = original occupied (0) OR unknown (127). "
                "row=0 is world y_max (top of map). "
                "world_x = min[0] + col * metres_per_pixel['x']. "
                "world_y = min[1] + (H - 1 - row) * metres_per_pixel['y']."
            ),
        },
    })
    with open(os.path.join(out_dir, f"{scene_id}_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return True


def main():
    p = argparse.ArgumentParser(
        description="Flatten InteriorGS scenes into binary occupancy maps.")
    p.add_argument("--root", required=True,
                   help="InteriorGS root containing one directory per scene.")
    p.add_argument("--out", required=True, help="Destination directory.")
    p.add_argument("--limit", type=int, default=None,
                   help="Convert at most this many scenes (for a quick trial).")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    scenes = sorted(d for d in os.listdir(a.root)
                    if os.path.isdir(os.path.join(a.root, d)))
    if a.limit:
        scenes = scenes[:a.limit]

    done = skipped = 0
    for sid in scenes:
        if convert_scene(os.path.join(a.root, sid), a.out, sid):
            done += 1
        else:
            skipped += 1
            print(f"  skip {sid}: no occupancy.png/.json")
    print(f"Converted {done} scenes to {a.out} ({skipped} skipped).")


if __name__ == "__main__":
    main()
