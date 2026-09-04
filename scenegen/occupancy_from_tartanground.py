"""
Build binary occupancy PNGs + metadata JSONs from TartanGround semantic PCD files.
Uses np.memmap — never loads the full PCD into RAM.

Output per environment, in the layout generate_random_maps.py expects:
    data/occupancy_maps/tartanground/
        <env>_sem_occupancy.png      uint8, FREE=255, OBS=0
        <env>_sem_metadata.json      metres_per_pixel, min, bounds, etc.

Label map format (seg_label_map.json inside seg_labels.zip):
    {"name_map": {"classname": int_id, ...}}
    The 4th float32 column of the PCD binary data IS the class ID (packed as float32).

Usage:
    python scenegen/occupancy_from_tartanground.py --root_dir data/raw/tartanground
    python scenegen/occupancy_from_tartanground.py --root_dir data/raw/tartanground --env AmusementPark
    python scenegen/occupancy_from_tartanground.py --root_dir data/raw/tartanground \
        --resolution 0.10 --z_min 0.20 --z_max 1.00 --chunk 2000000
"""

import argparse
import json
import os
import struct
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_dilation

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_RESOLUTION       = 0.10   # meters/cell
DEFAULT_Z_MIN            = 0.20   # above ground
DEFAULT_Z_MAX            = 1.00   # below ceiling
DEFAULT_GROUND_PCT       = 8      # percentile for ground estimation
DEFAULT_CHUNK            = 2_000_000
DEFAULT_KEEP_N           = 5      # subsample: use every Nth point for ground est
DEFAULT_OCC_THRESHOLD    = 3      # min point-count to mark a cell occupied
DEFAULT_INFLATE          = 1      # binary dilation iterations



 
# ── label map ─────────────────────────────────────────────────────────────────
 
def load_label_map(env_dir: Path) -> dict:
    """
    Find and parse seg_label_map.json.
    Returns {"classname": int_id, ...} from the name_map key.
    Also returns the full raw dict for reference.
    """
    for pattern in ["**/seg_label_map.json", "**/seg_labels.json", "**/*.json"]:
        matches = list(env_dir.glob(pattern))
        if matches:
            label_file = matches[0]
            break
    else:
        raise FileNotFoundError(f"No label JSON found under {env_dir}")
 
    print(f"  Label file: {label_file.relative_to(env_dir.parent)}")
    with open(label_file) as f:
        raw = json.load(f)
 
    # Format: {"name_map": {"classname": int_id, ...}}
    if "name_map" in raw:
        name_map = raw["name_map"]
    elif isinstance(raw, dict):
        # Fallback: maybe it IS the name_map directly
        name_map = raw
    else:
        raise ValueError(f"Unrecognized label map format in {label_file}")
 
    print(f"  Classes ({len(name_map)}): {list(name_map.keys())}")
    return name_map
 
 
# ── PCD header ────────────────────────────────────────────────────────────────
 
def read_pcd_header(pcd_path: Path) -> tuple[dict, int]:
    """Return header dict and byte offset to binary data start."""
    header = {}
    header_bytes = 0
    with open(pcd_path, "rb") as f:
        while True:
            line = f.readline()
            header_bytes += len(line)
            decoded = line.decode("utf-8", errors="ignore").strip()
            if decoded.startswith("DATA"):
                header["data"] = decoded.split()[1]
                break
            if decoded.startswith("#") or not decoded:
                continue
            parts = decoded.split()
            header[parts[0].lower()] = parts[1:]
 
    header["num_points"] = int(header.get("points", [0])[0])
    header["fields"]     = header.get("fields", [])
    header["count"]      = [int(c) for c in header.get("count", [])]
    header["size"]       = [int(s) for s in header.get("size", [])]
    header["type"]       = header.get("type", [])
    return header, header_bytes
 
 
def get_num_fields(header: dict) -> int:
    """Total number of float32 columns per point."""
    return sum(int(c) for c in header.get("count", [1] * len(header["fields"])))
 
 
# ── core processing ───────────────────────────────────────────────────────────
 
def process_env(
    env: str,
    root_dir: Path,
    output_dir: Path,
    resolution: float,
    z_min_rel: float,
    z_max_rel: float,
    ground_pct: int,
    chunk_size: int,
    keep_n: int,
    occ_threshold: int,
    inflate: int,
    env_dir: Path = None,
):
    env_dir  = env_dir or (root_dir / env)
    pcd_path = env_dir / f"{env}_sem.pcd"
 
    if not pcd_path.exists():
        print(f"  [skip] {pcd_path.name} not found.")
        return
 
    # Load label map (for metadata only — not needed for binary occupancy)
    try:
        name_map = load_label_map(env_dir)
    except FileNotFoundError as e:
        print(f"  [warn] {e} — continuing without class labels")
        name_map = {}
 
    # Read header
    header, header_bytes = read_pcd_header(pcd_path)
    num_fields  = get_num_fields(header)
    num_points  = header["num_points"]
    data_format = header["data"]
 
    print(f"  PCD: {num_points:,} points, {num_fields} fields, format={data_format}")
 
    if data_format != "binary":
        print(f"  [warn] format={data_format} — only binary tested; proceeding anyway")
 
    # Memory-map the raw float32 data (no full load into RAM)
    raw = np.memmap(pcd_path, dtype=np.float32, mode="r", offset=header_bytes)
    # Reshape: each row = one point = num_fields floats
    # Guard against trailing bytes
    n_complete = len(raw) // num_fields
    points = raw[:n_complete * num_fields].reshape(n_complete, num_fields)
 
    xyz = points[:, :3]   # X Y Z
    # Column 3 is class ID packed as float32 — reinterpret as uint32 -> int
    cls_raw = points[:, 3].view(np.float32)   # still float32 view
    # Reinterpret bytes as uint32 to get the integer class ID
    cls_uint = cls_raw.view(np.uint32) if cls_raw.dtype == np.float32 else cls_raw.astype(np.uint32)
 
    print(f"  Memmap OK. Estimating ground Z ...")
 
    # ── Ground estimation (subsampled) ────────────────────────────────────────
    sample_z = xyz[::keep_n, 2]
    sample_z = sample_z[np.isfinite(sample_z)]
    ground_z = float(np.percentile(sample_z, ground_pct))
    print(f"  Ground Z = {ground_z:.2f} m")
 
    abs_z_min = ground_z + z_min_rel
    abs_z_max = ground_z + z_max_rel
 
    # ── Pass 1: world bounds ──────────────────────────────────────────────────
    print(f"  Pass 1: bounds  Z=[{abs_z_min:.2f}, {abs_z_max:.2f}] ...")
    xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
 
    for start in range(0, len(xyz), chunk_size):
        chunk_xyz = xyz[start:start + chunk_size]
        z = chunk_xyz[:, 2]
        mask = np.isfinite(chunk_xyz).all(axis=1) & (z > abs_z_min) & (z < abs_z_max)
        c = chunk_xyz[mask]
        if len(c) == 0:
            continue
        xmin = min(xmin, float(c[:, 0].min()))
        xmax = max(xmax, float(c[:, 0].max()))
        ymin = min(ymin, float(c[:, 1].min()))
        ymax = max(ymax, float(c[:, 1].max()))
 
    if not np.isfinite(xmin):
        print(f"  [skip] No points in Z range — try wider --z_min/--z_max.")
        return
 
    width  = int((xmax - xmin) / resolution) + 1
    height = int((ymax - ymin) / resolution) + 1
    print(f"  Grid: {width} x {height} px  ({width*resolution:.0f} x {height*resolution:.0f} m)")
 
    grid = np.zeros((height, width), dtype=np.uint16)
 
    # ── Pass 2: fill grid ─────────────────────────────────────────────────────
    print(f"  Pass 2: filling grid ...")
    for start in range(0, len(xyz), chunk_size):
        chunk_xyz = xyz[start:start + chunk_size]
        z = chunk_xyz[:, 2]
        mask = np.isfinite(chunk_xyz).all(axis=1) & (z > abs_z_min) & (z < abs_z_max)
        c = chunk_xyz[mask]
        if len(c) == 0:
            continue
        ix = ((c[:, 0] - xmin) / resolution).astype(np.int32)
        iy = ((c[:, 1] - ymin) / resolution).astype(np.int32)
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        np.add.at(grid, (iy[valid], ix[valid]), 1)
 
    # ── Threshold + clean ─────────────────────────────────────────────────────
    occupied = grid > occ_threshold
    if inflate > 0:
        occupied = binary_dilation(occupied, iterations=inflate)
 
    # FREE=255, OBS=0, flip Y so north is up
    img_arr = np.flipud((~occupied).astype(np.uint8) * 255)
 
    # ── Save PNG ──────────────────────────────────────────────────────────────
    stem     = f"{env}_sem"
    out_png  = output_dir / f"{stem}_occupancy.png"
    out_meta = output_dir / f"{stem}_metadata.json"
 
    Image.fromarray(img_arr).save(out_png)
    print(f"  Saved PNG  -> {out_png}")
 
    # ── Save metadata JSON (format expected by generate_randomMAPS.py) ────────
    metadata = {
        "scene":              env,
        "source_pcd":         str(pcd_path),
        "metres_per_pixel":   {"x": resolution, "y": resolution},
        "resolution":         resolution,
        "min":                [xmin, ymin],
        "max":                [xmax, ymax],
        "original_resolution": [width, height],  # [1] is read as H by generate_randomMAPS.py
        "ground_z":           ground_z,
        "z_filter":           [abs_z_min, abs_z_max],
        "occ_threshold":      occ_threshold,
        "inflate_iterations": inflate,
        "name_map":           name_map,
    }
    with open(out_meta, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved meta -> {out_meta}")
 
 
# ── CLI ───────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="TartanGround sem PCD -> binary occupancy PNG + metadata JSON"
    )
    parser.add_argument("--root_dir",   required=True,
                        help="Root dir containing environment folders.")
    parser.add_argument("--env",        default=None,
                        help="Single environment name. Omit to process all found.")
    parser.add_argument("--output_dir", default="data/occupancy_maps/tartanground",
                        help="Output folder (default: occupancy_maps_TartanGround).")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    parser.add_argument("--z_min",      type=float, default=DEFAULT_Z_MIN,
                        help="Min height above ground (m).")
    parser.add_argument("--z_max",      type=float, default=DEFAULT_Z_MAX,
                        help="Max height above ground (m).")
    parser.add_argument("--ground_pct", type=int,   default=DEFAULT_GROUND_PCT,
                        help="Percentile of Z used as ground estimate.")
    parser.add_argument("--chunk",      type=int,   default=DEFAULT_CHUNK,
                        help="Points per processing chunk.")
    parser.add_argument("--keep_n",     type=int,   default=DEFAULT_KEEP_N,
                        help="Subsample every Nth point for ground estimation.")
    parser.add_argument("--occ_thresh", type=int,   default=DEFAULT_OCC_THRESHOLD,
                        help="Min point count to mark a cell occupied.")
    parser.add_argument("--inflate",    type=int,   default=DEFAULT_INFLATE,
                        help="Binary dilation iterations.")
    args = parser.parse_args()
 
    root_dir   = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    env_dirs = {}
    if args.env:
        envs = [args.env]
        pcd_files = list(root_dir.rglob(f"{args.env}_sem.pcd"))
        if pcd_files:
            env_dirs[args.env] = pcd_files[0].parent
        else:
            env_dirs[args.env] = root_dir / args.env
    else:
        # Walk entire tree to find all *_sem.pcd files
        pcd_files = list(root_dir.rglob("*_sem.pcd"))
        # Derive env name from filename: AmusementPark_sem.pcd -> AmusementPark
        envs = sorted(set(
            p.stem.replace("_sem", "") for p in pcd_files
        ))
        # Map env name -> its directory (may be nested)
        env_dirs = {
            p.stem.replace("_sem", ""): p.parent for p in pcd_files
        }
        print(f"Found {len(envs)} environments:")
        for e in envs:
            print(f"  {e}  ->  {env_dirs[e]}")
 
    for env in envs:
        print(f"\n[{env}]")
        process_env(
            env, root_dir, output_dir,
            args.resolution, args.z_min, args.z_max,
            args.ground_pct, args.chunk, args.keep_n,
            args.occ_thresh, args.inflate,
            env_dir=env_dirs.get(env),
        )
 
    print("\nAll done.")
 
 
if __name__ == "__main__":
    main()