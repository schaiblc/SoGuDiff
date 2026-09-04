import os
import glob
import numpy as np
from PIL import Image
import habitat_sim
from scipy.ndimage import binary_opening, binary_closing
import json

# python occupancy_from_matterport.py --root data/raw/matterport --output data/occupancy_maps/matterport --resolution 0.05

def generate_occupancy_map(glb_path, resolution=0.05):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = glb_path
    sim_cfg.load_semantic_mesh = False

    cfg = habitat_sim.Configuration(sim_cfg, [habitat_sim.AgentConfiguration()])
    try:
        sim = habitat_sim.Simulator(cfg)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    pf = sim.pathfinder
    if not pf.is_loaded:
        sim.close()
        return None

    bounds = pf.get_bounds()
    lo, hi = bounds[0], bounds[1]

    w = int((hi[0] - lo[0]) / resolution) + 1
    h = int((hi[2] - lo[2]) / resolution) + 1

    # Only look between 0.1m and 1.5m above the scene's lowest point
    # This covers robot/human navigable floors and excludes ceilings
    y_min = lo[1] + 0.1   # just above lowest floor
    y_max = lo[1] + 1.5   # max robot standing height
    y_levels = np.arange(y_min, y_max, 0.25)

    print(f"  Sampling Y: {y_min:.2f} to {y_max:.2f} ({len(y_levels)} levels)")

    # Count navigable cells per Y to find actual floors
    stride = 15
    y_counts = {}
    for test_y in y_levels:
        count = sum(
            1 for row in range(0, h, stride)
            for col in range(0, w, stride)
            if pf.is_navigable([lo[0] + col * resolution, test_y, lo[2] + row * resolution])
        )
        y_counts[test_y] = count

    # Keep Y levels with meaningful coverage
    max_count = max(y_counts.values()) if y_counts else 1
    threshold = max(3, max_count * 0.05)
    candidate_ys = sorted([y for y, c in y_counts.items() if c >= threshold])
    print(f"  Candidate floors: {[round(y,2) for y in candidate_ys]}")

    # Cluster into distinct floors
    floors = []
    for y in candidate_ys:
        if not floors or y - floors[-1] > 0.5:
            floors.append(y)
        else:
            if y_counts[y] > y_counts[floors[-1]]:
                floors[-1] = y

    print(f"  Distinct floors: {[round(f,2) for f in floors]}")

    if not floors:
        print("  WARNING: no floors detected, falling back to y_min")
        floors = [y_min]

    # Rasterize all detected floors
    occ = np.zeros((h, w), dtype=np.uint8)
    for floor_y in floors:
        for row in range(h):
            for col in range(w):
                x = lo[0] + col * resolution
                z = lo[2] + row * resolution
                if pf.is_navigable([x, floor_y, z]):
                    occ[row, col] = 255

    sim.close()

    from scipy.ndimage import binary_opening, binary_closing
    cleaned = binary_opening(occ > 0, iterations=1)
    cleaned = binary_closing(cleaned, iterations=2)
    result = (np.flipud(cleaned) * 255).astype(np.uint8)

    return result, floors, lo, hi


def batch_generate(root_dir, output_dir, resolution=0.05):
    os.makedirs(output_dir, exist_ok=True)
    glb_files = glob.glob(os.path.join(root_dir, "**", "*.basis.glb"), recursive=True)

    if not glb_files:
        print(f"No .basis.glb files found under {root_dir}")
        return

    print(f"Found {len(glb_files)} scenes\n")
    success, failed = [], []

    for i, glb_path in enumerate(sorted(glb_files)):
        scene_id = os.path.basename(os.path.dirname(glb_path))
        
        # Match naming convention expected by generate_InteriorGS.py
        out_png  = os.path.join(output_dir, f"{scene_id}_occupancy.png")
        out_meta = os.path.join(output_dir, f"{scene_id}_metadata.json")

        if os.path.exists(out_png) and os.path.exists(out_meta):
            print(f"[{i+1}/{len(glb_files)}] Skipping {scene_id} (already exists)")
            success.append(scene_id)
            continue

        print(f"[{i+1}/{len(glb_files)}] Processing {scene_id} ...")

        output = generate_occupancy_map(glb_path, resolution)
        if output is None:
            print(f"  FAILED: {scene_id}")
            failed.append(scene_id)
            continue

        occ_map, floors, lo, hi = output  # unpack lo/hi too (see below)

        # Save PNG
        Image.fromarray(occ_map).save(out_png)

        # Save metadata
        h, w = occ_map.shape
        metadata = {
            "scene_id": scene_id,
            "metres_per_pixel": {"x": resolution, "y": resolution},
            "min": [float(lo[0]), float(lo[2])],   # world XZ origin
            "max": [float(hi[0]), float(hi[2])],
            "original_resolution": [h, w],
            "floors_y": [round(f, 3) for f in floors],
            "source_glb": glb_path,
        }
        with open(out_meta, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved -> {out_png}")
        success.append(scene_id)

    print(f"\n{'='*40}")
    print(f"Done: {len(success)} succeeded, {len(failed)} failed")
    if failed:
        print("Failed scenes:")
        for s in failed:
            print(f"  - {s}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate occupancy maps from Matterport GLB scenes")
    parser.add_argument("--root",       required=True,          help="Root directory containing scene folders")
    parser.add_argument("--output",     default="data/occupancy_maps/matterport",
                        help="Output directory for PNG maps")
    parser.add_argument("--resolution", type=float, default=0.05, help="Meters per pixel (default: 0.05)")
    args = parser.parse_args()

    batch_generate(args.root, args.output, args.resolution)