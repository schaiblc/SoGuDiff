"""
Stage 1: Crop Scene Generator from InteriorGS Occupancy Maps
=============================================================
Takes the binary occupancy maps produced by process_interiorgs_occupancy.py
and generates 50,000 ego-centric 50x50 grid crops matched to the scene
generator's coordinate system:

    MAP_SIZE     = 50 cells
    MAP_EXTENT_M = 10 m
    MAP_RES      = 0.20 m/cell       (10 / 50)
    FREE         = 255, OBSTACLE = 0  (same as InteriorGS binary convention)

InteriorGS source maps:
    - 1024x1024 pixels at 0.05 m/pixel  (or as stored in metadata)
    - Crop window in source pixels = MAP_EXTENT_M / src_res  (e.g. 150px @ 0.05)
    - Downsampled to 50x50 with NEAREST to preserve binary values

Filtering:
    - Free-space ratio in [FREE_RATIO_MIN, FREE_RATIO_MAX]   (default 25–80%)
    - Ego cell and surrounding SAFETY_RADIUS must be fully free

Ego sampling:
    - Random position inside a valid free-space disk
    - Random rotation theta in [-pi, pi]
    - Stored so downstream code can place robot at (0,0) in ego frame

Output (new directory  cropped_scenes/):
    maps.npy          uint8 [N, 50, 50]  — memory-mapped friendly
    metadata.npz      structured array with per-scene fields:
                          scene_name, source_path, ego_x, ego_y, ego_theta,
                          free_ratio, src_res, scene_idx
    scene_index.json  human-readable list of all scenes + metadata

Usage:
    python generate_random_maps.py --input_dir  data/occupancy_maps/interiorgs --output_dir data/maps/interiorgs --n_samples  50000 --max_tries_per_scene 20
    python generate_random_maps.py --input_dir  data/occupancy_maps/matterport --output_dir data/maps/matterport --n_samples  50000 --max_tries_per_scene 20
    python generate_random_maps.py --input_dir  data/occupancy_maps/tartanground --output_dir data/maps/tartanground --n_samples  100000 --max_tries_per_scene 20

    Move Matterport to outside, occupancy_maps_Matterport and then run above

    # Faster debug run:
    python generate_occupancy_maps.py \\
        --input_dir  ./occupancy_maps \\
        --output_dir ./cropped_scenes_debug \\
        --n_samples  500
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm

# ── Target grid spec (must match scene generator) ────────────────────────────
MAP_SIZE      = 50        # cells per side
MAP_EXTENT_M  = 10       # meters per side
MAP_RES       = MAP_EXTENT_M / MAP_SIZE   # 0.20 m/cell
ROBOT_SAFETY_RADIUS = 1.0                 # meters — ego spawn clearance from walls (≥ MAP_SAFETY_RADIUS in planner)

# ── Free-space quality filter ─────────────────────────────────────────────────
FREE_RATIO_MIN = 0.25     # reject crops that are almost entirely blocked
FREE_RATIO_MAX = 0.95     # reject crops that are nearly open (< 5 % obstacle content)
FREE_VAL       = 255
OBS_VAL        = 0

# ── Ego placement: keep ego away from crop border so the disk fits ────────────
# In cells, how far from the border the ego can be placed
BORDER_MARGIN_CELLS = int(np.ceil(ROBOT_SAFETY_RADIUS / MAP_RES)) + 2   # ~5 (ceil(0.5/0.2)+2)


# ── Internals ─────────────────────────────────────────────────────────────────

def load_scene_list(input_dir: Path):
    """
    Load all processed scenes.  Expects files named
        <scene_name>_occupancy.png   (binary, 0/255)
        <scene_name>_metadata.json
    OR uses scene_index.json if present.
    """
    index_path = input_dir / "scene_index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        scenes = []
        for entry in index:
            png  = input_dir / entry["occupancy_png"]
            meta = input_dir / entry["metadata_json"]
            if png.exists() and meta.exists():
                scenes.append((entry["scene"], png, meta))
        return scenes

    # Fallback: glob
    scenes = []
    for png in sorted(input_dir.glob("*_occupancy.png")):
        name = png.stem.replace("_occupancy", "")
        meta = input_dir / f"{name}_metadata.json"
        if meta.exists():
            scenes.append((name, png, meta))
    return scenes


def load_binary_map(png_path: Path) -> np.ndarray:
    """Load a binary uint8 occupancy PNG as numpy array (H, W)."""
    return np.array(Image.open(png_path).convert("L"), dtype=np.uint8)


def compute_src_crop_half(meta: dict) -> int:
    """
    How many source pixels correspond to MAP_EXTENT_M / 2.
    Uses metres_per_pixel if available, else falls back to 'scale' field.
    """
    if "metres_per_pixel" in meta:
        res = meta["metres_per_pixel"]["x"]   # assume square pixels
    elif "scale" in meta:
        res = meta["scale"]
    else:
        res = 0.05   # InteriorGS default
    half_px = (MAP_EXTENT_M / 2.0) / res
    return int(np.ceil(half_px)), res


def sample_ego_pixels_batch(src_map: np.ndarray, half_px: int,
                             safety_src: int, rng: np.random.Generator,
                             free_yx: np.ndarray, margin: int,
                             batch: int = 2000):
    """
    Sample `batch` candidate positions at once, return all that pass the
    safety-radius check as (rows, cols) arrays. Vectorized — no Python loop.
    """
    if len(free_yx) == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int)

    idx  = rng.choice(len(free_yx), size=min(batch, len(free_yx)), replace=True)
    cands = free_yx[idx]                         # (batch, 2)  interior coords
    rows = cands[:, 0] + margin
    cols = cands[:, 1] + margin

    H, W = src_map.shape

    # Vectorized safety check: for each candidate, check a box around it
    # (conservative: use box instead of disk — close enough, much faster)
    r0 = np.clip(rows - safety_src, 0, H - 1)
    r1 = np.clip(rows + safety_src + 1, 0, H)
    c0 = np.clip(cols - safety_src, 0, W - 1)
    c1 = np.clip(cols + safety_src + 1, 0, W)

    # Check each candidate's safety window — still a loop but over batch,
    # and we break as soon as we have enough passing candidates
    valid_rows, valid_cols = [], []
    for i in range(len(rows)):
        window = src_map[r0[i]:r1[i], c0[i]:c1[i]]
        if (window == FREE_VAL).all():
            valid_rows.append(rows[i])
            valid_cols.append(cols[i])

    return np.array(valid_rows, dtype=int), np.array(valid_cols, dtype=int)


def crop_and_resize(src_map: np.ndarray, row: int, col: int,
                    half_px: int) -> np.ndarray:
    """
    Crop [row-half_px : row+half_px, col-half_px : col+half_px] from src_map
    and resize to (MAP_SIZE, MAP_SIZE) with NEAREST interpolation (binary safe).
    """
    patch = src_map[row - half_px: row + half_px,
                    col - half_px: col + half_px]
    img = Image.fromarray(patch, mode="L")
    img = img.resize((MAP_SIZE, MAP_SIZE), Image.NEAREST)
    return np.array(img, dtype=np.uint8)


def apply_ego_rotation(crop: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate the occupancy crop by -theta so the robot's heading points
    toward the +x axis in the stored map (standard ego frame convention).
    Uses NEAREST to keep binary values exact.
    Theta is in radians; positive = counter-clockwise.
    """
    degrees = -np.degrees(theta)
    img = Image.fromarray(crop, mode="L")
    # expand=False: keeps 50x50; fill with OBS_VAL for unknown border cells
    img = img.rotate(degrees, resample=Image.NEAREST, expand=False,
                     fillcolor=int(OBS_VAL))
    return np.array(img, dtype=np.uint8)


def free_ratio(crop: np.ndarray) -> float:
    return float(np.sum(crop == FREE_VAL)) / crop.size


def ego_cell_free(crop: np.ndarray) -> bool:
    """
    The ego is at the center of the 50x50 crop.
    Check that a disk of ROBOT_SAFETY_RADIUS is entirely free.
    """
    cy = cx = MAP_SIZE // 2
    r_cells = int(np.ceil(ROBOT_SAFETY_RADIUS / MAP_RES))
    ys, xs = np.ogrid[-r_cells: r_cells + 1, -r_cells: r_cells + 1]
    mask = (ys ** 2 + xs ** 2) <= r_cells ** 2
    # Extract the disk region
    r0 = max(0, cy - r_cells);  r1 = min(MAP_SIZE, cy + r_cells + 1)
    c0 = max(0, cx - r_cells);  c1 = min(MAP_SIZE, cx + r_cells + 1)
    disk_crop = crop[r0:r1, c0:c1]
    # Trim mask to same shape (border cases)
    disk_mask = mask[:disk_crop.shape[0], :disk_crop.shape[1]]
    return bool((disk_crop[disk_mask] == FREE_VAL).all())


def fast_free_ratio_check(src_map: np.ndarray, row: int, col: int, 
                           half_px: int) -> float:
    """Check free ratio directly on source patch — no resize needed."""
    patch = src_map[row - half_px: row + half_px,
                    col - half_px: col + half_px]
    return float(np.mean(patch == FREE_VAL))  # FREE_VAL=255, so mean of bool


def pixel_to_world_coords(row: int, col: int, meta: dict):
    """
    Convert source-map pixel (row, col) to world (x, y) using metadata.
    row=0 is world y_max (top of map).
    """
    if "metres_per_pixel" in meta:
        spx = meta["metres_per_pixel"]["x"]
        spy = meta["metres_per_pixel"]["y"]
    elif "scale" in meta:
        spx = spy = meta["scale"]
    else:
        spx = spy = 0.05

    H = meta.get("original_resolution", [1024, 1024])[1]
    x = meta["min"][0] + col * spx
    y = meta["min"][1] + (H - 1 - row) * spy
    return float(x), float(y)


# ── Main generation loop ──────────────────────────────────────────────────────

def generate(input_dir: str, output_dir: str, n_samples: int,
             max_tries_per_scene: int = 200, seed: int = 42):

    rng = np.random.default_rng(seed)
    random.seed(seed)

    in_root  = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    scenes = load_scene_list(in_root)
    if not scenes:
        raise RuntimeError(f"No processed scenes found in {in_root}. "
                           "Run process_interiorgs_occupancy.py first.")
    print(f"Loaded {len(scenes)} source scenes from {in_root}")

    # Pre-load all maps into memory (50 scenes × 1024² × 1 byte ≈ 50 MB — fine)
    loaded = []
    for scene_name, png_path, meta_path in scenes:
        with open(meta_path) as f:
            meta = json.load(f)
        src_map = load_binary_map(png_path)          # keep in memory
        half_px, src_res = compute_src_crop_half(meta)
        safety_src = max(1, int(np.ceil(ROBOT_SAFETY_RADIUS / src_res)))

        H, W = src_map.shape
        margin = half_px
        if margin < H - margin and margin < W - margin:
            interior = src_map[margin:H - margin, margin:W - margin]
            free_yx_full = np.argwhere(interior == FREE_VAL)
            MAX_CANDIDATES = 100_000
            if len(free_yx_full) > MAX_CANDIDATES:
                rng_pre = np.random.default_rng(seed + hash(str(png_path)) % (2**31))
                idx = rng_pre.choice(len(free_yx_full), size=MAX_CANDIDATES, replace=False)
                free_yx = free_yx_full[idx]
            else:
                free_yx = free_yx_full
            del free_yx_full
        else:
            free_yx = np.empty((0, 2), dtype=np.int64)

        loaded.append({
            "name":       scene_name,
            "src_map":    src_map,               # ← keep loaded
            "meta":       meta,
            "half_px":    half_px,
            "safety_src": safety_src,
            "src_res":    src_res,
            "free_yx":    free_yx,
            "margin":     margin,
        })
        # del src_map  # free immediately after computing free_yx
    
    # ── Per-scene quotas for balanced coverage ────────────────────────────────
    n_scenes         = len(loaded)
    target_per_scene = n_samples // n_scenes
    remainder        = n_samples % n_scenes
    quotas           = {i: target_per_scene + (1 if i < remainder else 0)
                        for i in range(n_scenes)}
    scene_counts     = {i: 0 for i in range(n_scenes)}

    print(f"Indexed {len(loaded)} maps (on-demand loading). Starting crop generation...\n")

    # Output buffers
    maps_arr  = np.zeros((n_samples, MAP_SIZE, MAP_SIZE), dtype=np.uint8)

    # Metadata lists (will become structured array)
    meta_scene_name = []
    meta_ego_x      = np.zeros(n_samples, dtype=np.float32)
    meta_ego_y      = np.zeros(n_samples, dtype=np.float32)
    meta_ego_theta  = np.zeros(n_samples, dtype=np.float32)
    meta_free_ratio = np.zeros(n_samples, dtype=np.float32)
    meta_src_res    = np.zeros(n_samples, dtype=np.float32)
    meta_scene_idx  = np.zeros(n_samples, dtype=np.int32)

    generated = 0
    attempts  = 0
    rejected_free  = 0
    rejected_place = 0

    pbar = tqdm(total=n_samples, desc="Generating crops")

    BATCH = 4000   # candidates per scene per round

    while generated < n_samples:
        eligible = set(i for i in range(n_scenes)
                   if quotas[i] > 0 and len(loaded[i]["free_yx"]) > 0)
        scene_idx = int(rng.choice(list(eligible)))
        sc        = loaded[scene_idx]
        src_map   = sc["src_map"]
        half_px   = sc["half_px"]

        # ── Batch sample valid ego positions ──────────────────────────────────
        valid_rows, valid_cols = sample_ego_pixels_batch(
            src_map, half_px, sc["safety_src"], rng,
            sc["free_yx"], sc["margin"], batch=BATCH
        )
        if len(valid_rows) == 0:
            continue

        # ── Vectorized free-ratio check on all candidates at once ─────────────
        # Stack all patches into one array, compute mean along axes 1&2
        patches = np.stack([
            src_map[r - half_px: r + half_px, c - half_px: c + half_px]
            for r, c in zip(valid_rows, valid_cols)
        ])                                            # (N, 2*half_px, 2*half_px)
        free_ratios = (patches == FREE_VAL).mean(axis=(1, 2))  # (N,)
        mask = (free_ratios >= FREE_RATIO_MIN) & (free_ratios <= FREE_RATIO_MAX)
        valid_rows = valid_rows[mask]
        valid_cols = valid_cols[mask]

        if len(valid_rows) == 0:
            continue

        # ── Process passing candidates ─────────────────────────────────────────
        for r, c in zip(valid_rows, valid_cols):
            if generated >= n_samples:
                break

            crop_raw = crop_and_resize(src_map, r, c, half_px)

            if not ego_cell_free(crop_raw):
                rejected_place += 1
                continue

            theta    = float(rng.uniform(-np.pi, np.pi))
            crop_rot = apply_ego_rotation(crop_raw, theta)

            if not ego_cell_free(crop_rot):
                rejected_place += 1
                continue
            fr_rot = free_ratio(crop_rot)
            if not (FREE_RATIO_MIN <= fr_rot <= FREE_RATIO_MAX):
                rejected_free += 1
                continue

            # Accept
            maps_arr[generated] = crop_rot
            ego_wx, ego_wy = pixel_to_world_coords(r, c, sc["meta"])
            meta_scene_name.append(sc["name"])
            meta_ego_x[generated]      = ego_wx
            meta_ego_y[generated]      = ego_wy
            meta_ego_theta[generated]  = theta
            meta_free_ratio[generated] = fr_rot
            meta_src_res[generated]    = sc["src_res"]
            meta_scene_idx[generated]  = scene_idx

            generated += 1
            pbar.update(1)
            
            scene_counts[scene_idx] += 1
            if scene_counts[scene_idx] >= quotas[scene_idx]:
                eligible.discard(scene_idx)
            if not eligible:
                break
            # break

        attempts += len(valid_rows)   # count actual candidates evaluated

        if attempts % 5000 == 0:
            rate = generated / max(1, attempts)
            pbar.set_postfix({
                "accept_rate": f"{rate:.2%}",
                "rej_free":  rejected_free,
                "rej_place": rejected_place,
            })

    pbar.close()

    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"\nSaving {n_samples} crops to {out_root} ...")

    # 1. Memory-mapped array: maps.npy  — the primary training artifact
    maps_path = out_root / "maps.npy"
    np.save(maps_path, maps_arr)
    print(f"  Saved maps.npy  shape={maps_arr.shape}  "
          f"dtype={maps_arr.dtype}  "
          f"size={maps_arr.nbytes / 1e6:.1f} MB")

    # 2. Numerical metadata as .npz (fast random access in training)
    meta_npz_path = out_root / "metadata.npz"
    np.savez(
        meta_npz_path,
        ego_x      = meta_ego_x,
        ego_y      = meta_ego_y,
        ego_theta  = meta_ego_theta,
        free_ratio = meta_free_ratio,
        src_res    = meta_src_res,
        scene_idx  = meta_scene_idx,
    )
    print(f"  Saved metadata.npz")

    # 3. Human-readable index as JSON (scene names need a list, not numpy)
    index = []
    for i in range(n_samples):
        index.append({
            "idx":        i,
            "scene":      meta_scene_name[i],
            "ego_x":      float(meta_ego_x[i]),
            "ego_y":      float(meta_ego_y[i]),
            "ego_theta":  float(meta_ego_theta[i]),
            "free_ratio": float(meta_free_ratio[i]),
            "src_res":    float(meta_src_res[i]),
        })
    json_path = out_root / "scene_index.json"
    with open(json_path, "w") as f:
        json.dump(index, f)
    print(f"  Saved scene_index.json")

    # 4. Config snapshot so downstream code knows the grid spec
    config = {
        "MAP_SIZE":            MAP_SIZE,
        "MAP_EXTENT_M":        MAP_EXTENT_M,
        "MAP_RES":             MAP_RES,
        "ROBOT_SAFETY_RADIUS": ROBOT_SAFETY_RADIUS,
        "FREE_VAL":            FREE_VAL,
        "OBS_VAL":             OBS_VAL,
        "FREE_RATIO_MIN":      FREE_RATIO_MIN,
        "FREE_RATIO_MAX":      FREE_RATIO_MAX,
        "n_samples":           n_samples,
        "n_source_scenes":     len(loaded),
        "seed":                seed,
    }
    config_path = out_root / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Generation complete.")
    print(f"  Total attempts : {attempts}")
    print(f"  Accepted       : {generated}")
    print(f"  Accept rate    : {generated/attempts:.2%}")
    print(f"  Rejected (free): {rejected_free}")
    print(f"  Rejected (place): {rejected_place}")
    print(f"  Free ratio     : mean={meta_free_ratio.mean():.3f}  "
          f"std={meta_free_ratio.std():.3f}  "
          f"min={meta_free_ratio.min():.3f}  "
          f"max={meta_free_ratio.max():.3f}")
    # Scene coverage
    scene_usage = np.bincount(meta_scene_idx, minlength=len(loaded))
    print(f"  Crops per source scene: "
          f"mean={scene_usage.mean():.0f}  "
          f"min={scene_usage.min()}  "
          f"max={scene_usage.max()}")
    print(f"\nOutput: {out_root.resolve()}")


# ── Loading utility for Stage 2 ───────────────────────────────────────────────

def load_cropped_dataset(output_dir: str, mmap: bool = True):
    """
    Convenience loader for the scene generator (Stage 2).

    Returns:
        maps      : np.ndarray  uint8  [N, 50, 50]  (memory-mapped if mmap=True)
        meta      : dict of np.ndarrays  (ego_x, ego_y, ego_theta, free_ratio, ...)
        config    : dict  (MAP_SIZE, MAP_RES, etc.)

    Usage:
        maps, meta, cfg = load_cropped_dataset("./cropped_scenes")
        occ_map = maps[i]          # uint8 50x50, FREE=255, OBS=0
        free_mask = occ_map == 255 # bool mask for agent placement

    The map is already in ego frame: robot is at cell (25, 25),
    heading toward +col (right = forward).
    """
    root = Path(output_dir)
    mode = "r" if mmap else None
    maps = np.load(root / "maps.npy", mmap_mode=mode)

    raw  = np.load(root / "metadata.npz")
    meta = {k: raw[k] for k in raw.files}

    with open(root / "config.json") as f:
        config = json.load(f)

    with open(root / "scene_index.json") as f:
        scene_names = [e["scene"] for e in json.load(f)]
    meta["scene_name"] = scene_names

    return maps, meta, config


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 50k ego-centric occupancy crops from InteriorGS maps."
    )
    parser.add_argument("--input_dir",  required=True,
                        help="Folder with *_occupancy.png + *_metadata.json files")
    parser.add_argument("--output_dir", required=True,
                        help="Destination for maps.npy, metadata.npz, etc.")
    parser.add_argument("--n_samples",  type=int, default=50000)
    parser.add_argument("--max_tries_per_scene", type=int, default=200,
                        help="Max ego placement attempts per scene pick")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    generate(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        max_tries_per_scene=args.max_tries_per_scene,
        seed=args.seed,
    )