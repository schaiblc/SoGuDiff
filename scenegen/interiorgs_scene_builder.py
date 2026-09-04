#!/usr/bin/env python3
"""
interiorgs_manual_scene_builder.py
====================================
Dedicated, hands-on scene builder for ONE specific InteriorGS building map
(default: occupancy_maps_InteriorGS/0007_840137_occupancy.png). Unlike
make_interiorgs_scene.py (which auto-searches for a good robot
origin/heading), this script lets YOU pick the exact robot position +
heading in the real building, plus the exact pedestrian count/positions/
velocities, then builds the .npz + renders preview images so you can
iterate quickly.

Workflow
--------
1. First, render the whole raw building map with a world-coordinate meter
   grid so you can read off candidate positions:

     python interiorgs_manual_scene_builder.py --preview_map

   -> writes  <out_dir>/<png_stem>_world_preview.png

2. Pick a robot position + heading (in WORLD meters/degrees, read off that
   image) and pedestrians + goal (in LOCAL/ego frame — same convention as
   every other hand-crafted scene: +x = robot heading, +y = robot's left,
   relative to the robot's start), then build the scene:

     python interiorgs_manual_scene_builder.py \
        --robot_x 0.0 --robot_y -4.0 --robot_heading_deg 255 \
        --goal 4.3 4.0 \
        --ped 4.2 3.3 -0.1 -0.15 \
        --ped 3.4 4.1 -0.1 -0.15 \
        --ped 2.0 -0.2 -0.45 0.95
         --out_name my_scene

   -> writes  <out_dir>/sample_<out_name>.npz
              <out_dir>/sample_<out_name>_preview.png        (final local scene)
              <out_dir>/sample_<out_name>_world_context.png   (world map with the
                                                                chosen crop + robot
                                                                pose overlaid, for
                                                                sanity-checking the
                                                                placement in situ)

Only needs numpy + matplotlib (+ PIL if available; falls back to a tiny
built-in PNG reader for these 8-bit grayscale non-interlaced files if not).
"""
import argparse
import json
import os
import struct
import zlib
from collections import deque

import numpy as np

# =============================================================================
# Defaults — override any of these via CLI flags
# =============================================================================
DEFAULT_PNG_PATH = "data/occupancy_maps/interiorgs/0007_840137_occupancy.png"
DEFAULT_OUT_DIR = "data/scenes/eval_extra"

MAP_SIZE = 50
MAP_EXTENT = 10.0
RES = MAP_EXTENT / MAP_SIZE  # 0.2 m/cell
ROBOT_RADIUS = 0.25
HUMAN_RADIUS = 0.25
NAV_RADIUS_M = 0.30
START_CLEAR_M = 0.6


# =============================================================================
# PNG loading (PIL if available, else a tiny stdlib-only PNG decoder)
# =============================================================================
def _read_png_gray8_builtin(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos = 8
    width = height = bitdepth = colortype = interlace = None
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8].decode("ascii")
        chunk = data[pos + 8:pos + 8 + length]
        pos += 8 + length + 4
        if ctype == "IHDR":
            width, height, bitdepth, colortype, comp, filt, interlace = \
                struct.unpack(">IIBBBBB", chunk)
        elif ctype == "IDAT":
            idat += chunk
        elif ctype == "IEND":
            break
    assert bitdepth == 8 and colortype == 0 and interlace == 0, \
        "only 8-bit grayscale non-interlaced PNGs are supported by the built-in reader"
    raw = zlib.decompress(bytes(idat))
    stride = width
    img = np.zeros((height, width), dtype=np.int16)
    prev = np.zeros(stride, dtype=np.int16)
    off = 0
    for r in range(height):
        filt_type = raw[off]; off += 1
        line = np.frombuffer(raw[off:off + stride], dtype=np.uint8).astype(np.int16)
        off += stride
        if filt_type == 1:      # Sub
            for i in range(1, stride):
                line[i] = (line[i] + line[i - 1]) % 256
        elif filt_type == 2:    # Up
            line = (line + prev) % 256
        elif filt_type == 3:    # Average
            for i in range(stride):
                a = line[i - 1] if i >= 1 else 0
                b = prev[i]
                line[i] = (line[i] + (a + b) // 2) % 256
        elif filt_type == 4:    # Paeth
            for i in range(stride):
                a = int(line[i - 1]) if i >= 1 else 0
                b = int(prev[i])
                c = int(prev[i - 1]) if i >= 1 else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[i] = (line[i] + pr) % 256
        img[r] = line
        prev = line
    return img.astype(np.uint8)


def read_occupancy_png(path):
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("L"), dtype=np.uint8)
    except ImportError:
        return _read_png_gray8_builtin(path)


# =============================================================================
# Local-frame grid helpers
# =============================================================================
def world_to_grid_local(x, y):
    col = int(round(x / RES + MAP_SIZE / 2))
    row = int(round(y / RES + MAP_SIZE / 2))
    return (int(np.clip(row, 0, MAP_SIZE - 1)), int(np.clip(col, 0, MAP_SIZE - 1)))


def is_free_box_local(occ, x, y, half_m):
    r, c = world_to_grid_local(x, y)
    rad = int(np.ceil(half_m / RES))
    r0, r1 = max(0, r - rad), min(MAP_SIZE, r + rad + 1)
    c0, c1 = max(0, c - rad), min(MAP_SIZE, c + rad + 1)
    return not occ[r0:r1, c0:c1].any()


def inflate(occ, radius_m):
    rad = int(np.ceil(radius_m / RES))
    out = occ.copy()
    rows, cols = np.where(occ > 0.5)
    for r, c in zip(rows, cols):
        out[max(0, r - rad):min(MAP_SIZE, r + rad + 1),
            max(0, c - rad):min(MAP_SIZE, c + rad + 1)] = 1.0
    return out


def bfs_reachable_local(occ_inflated, start_xy, goal_xy):
    sr, sc = world_to_grid_local(*start_xy)
    gr, gc = world_to_grid_local(*goal_xy)
    if occ_inflated[sr, sc] or occ_inflated[gr, gc]:
        return False
    visited = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    visited[sr, sc] = True
    q = deque([(sr, sc)])
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while q:
        r, c = q.popleft()
        if r == gr and c == gc:
            return True
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < MAP_SIZE and 0 <= nc < MAP_SIZE and not visited[nr, nc] and not occ_inflated[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))
    return False


# =============================================================================
# Rotated crop: robot sits at (robot_x, robot_y) in WORLD meters, facing
# `heading_deg` (measured counter-clockwise from world +x, degrees). The
# local scene frame's +x axis is defined to point along that heading, +y
# is 90 degrees counter-clockwise from it (the robot's left) — exactly the
# ego-frame convention every other hand-crafted scene uses.
# =============================================================================
def build_rotated_crop(raw_free, mpp_x, mpp_y, wmin_x, wmin_y, H_raw, W_raw,
                       robot_x, robot_y, heading_deg, subsamples=3):
    theta = np.deg2rad(heading_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    occ = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    half_cell = RES / 2.0
    offsets = np.linspace(-half_cell, half_cell, subsamples)

    for r in range(MAP_SIZE):
        ly = (r - MAP_SIZE / 2 + 0.5) * RES
        for c in range(MAP_SIZE):
            lx = (c - MAP_SIZE / 2 + 0.5) * RES
            obstacle = False
            for dly in offsets:
                for dlx in offsets:
                    slx, sly = lx + dlx, ly + dly
                    # local (ego) -> world:  world = robot + R(theta) @ local
                    wx = robot_x + cos_t * slx - sin_t * sly
                    wy = robot_y + sin_t * slx + cos_t * sly
                    col = (wx - wmin_x) / mpp_x
                    row = (H_raw - 1) - (wy - wmin_y) / mpp_y
                    ci, ri = int(round(col)), int(round(row))
                    if ci < 0 or ri < 0 or ci >= W_raw or ri >= H_raw:
                        obstacle = True
                        break
                    if not raw_free[ri, ci]:
                        obstacle = True
                        break
                if obstacle:
                    break
            occ[r, c] = 1.0 if obstacle else 0.0
    return occ


def local_to_world(lx, ly, robot_x, robot_y, heading_deg):
    theta = np.deg2rad(heading_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    wx = robot_x + cos_t * lx - sin_t * ly
    wy = robot_y + sin_t * lx + cos_t * ly
    return wx, wy


# =============================================================================
# Mode 1: render the whole raw map with a world-coordinate meter grid, for
# picking robot_x / robot_y / heading_deg by eye.
# =============================================================================
def render_world_preview(raw_gray, meta, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mpp_x, mpp_y = meta["metres_per_pixel"]["x"], meta["metres_per_pixel"]["y"]
    wmin_x, wmin_y = meta["min"][0], meta["min"][1]
    wmax_x, wmax_y = meta["max"][0], meta["max"][1]

    fig, ax = plt.subplots(figsize=(10, 10 * raw_gray.shape[0] / raw_gray.shape[1]))
    ax.imshow(raw_gray, extent=[wmin_x, wmax_x, wmin_y, wmax_y],
             origin="upper", cmap="gray_r", interpolation="nearest")
    ax.set_xlabel("world x (m)", fontsize=13)
    ax.set_ylabel("world y (m)", fontsize=13)
    ax.set_title("Raw building map — read off robot_x/robot_y here (world frame)",
                fontsize=12)
    ax.set_xticks(np.arange(np.floor(wmin_x), np.ceil(wmax_x) + 1, 1.0), minor=True)
    ax.set_yticks(np.arange(np.floor(wmin_y), np.ceil(wmax_y) + 1, 1.0), minor=True)
    ax.set_xticks(np.arange(np.floor(wmin_x / 2) * 2, np.ceil(wmax_x) + 1, 2.0))
    ax.set_yticks(np.arange(np.floor(wmin_y / 2) * 2, np.ceil(wmax_y) + 1, 2.0))
    ax.grid(which="both", color="red", alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=10)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved world preview: {out_path}")
    print(f"World bounds: x[{wmin_x:.2f}, {wmax_x:.2f}]  y[{wmin_y:.2f}, {wmax_y:.2f}]")


# =============================================================================
# Mode 2: build the scene from explicit robot pose + pedestrians + goal
# =============================================================================
def build_scene(args):
    raw_gray = read_occupancy_png(args.png_path)
    meta_path = args.png_path.replace("_occupancy.png", "_metadata.json")
    meta = json.load(open(meta_path))
    mpp_x, mpp_y = meta["metres_per_pixel"]["x"], meta["metres_per_pixel"]["y"]
    wmin_x, wmin_y = meta["min"][0], meta["min"][1]
    free_val = meta["convention"]["free"]
    H_raw, W_raw = raw_gray.shape
    raw_free = (raw_gray == free_val)

    occ = build_rotated_crop(raw_free, mpp_x, mpp_y, wmin_x, wmin_y, H_raw, W_raw,
                             args.robot_x, args.robot_y, args.robot_heading_deg)
    free_ratio = 1.0 - occ.mean()
    print(f"Built local crop at robot=({args.robot_x},{args.robot_y}) "
         f"heading={args.robot_heading_deg} deg -> free_ratio={free_ratio:.3f}")

    goal = np.array(args.goal, dtype=np.float32)
    obstacles = np.array(args.ped, dtype=np.float32).reshape(-1, 4) if args.ped else \
        np.zeros((0, 4), dtype=np.float32)

    # ---- sanity checks (WARN, don't crash — you may deliberately want an
    #      edge case) ----
    if not is_free_box_local(occ, 0.0, 0.0, START_CLEAR_M):
        print("  [WARN] robot start position is not clear of walls")
    if not is_free_box_local(occ, goal[0], goal[1], 0.4):
        print("  [WARN] goal cell is not clear of walls")
    inflated = inflate(occ, NAV_RADIUS_M)
    if not bfs_reachable_local(inflated, (0.0, 0.0), goal):
        print("  [WARN] goal is NOT reachable from the robot start on this map")
    for i, (px, py, vx, vy) in enumerate(obstacles):
        if not is_free_box_local(occ, px, py, HUMAN_RADIUS):
            print(f"  [WARN] pedestrian {i} at ({px},{py}) is blocked by a wall")

    kwargs = dict(
        start_state=np.array([0.6, 0.0], dtype=np.float32),
        goal=goal,
        obstacles=obstacles,
        occupancy_map=occ.astype(np.float32),
        has_map=np.float32(1.0),
        threat_type=np.array(args.threat_type),
        source_png=np.array(os.path.basename(args.png_path)),
        robot_world_xy=np.array([args.robot_x, args.robot_y], dtype=np.float32),
        robot_heading_deg=np.float32(args.robot_heading_deg),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_stem = f"sample_{args.out_name}"
    npz_path = os.path.join(args.out_dir, out_stem + ".npz")
    np.savez(npz_path, **kwargs)
    print(f"Saved: {npz_path}")

    render_local_scene(occ, goal, obstacles,
                       os.path.join(args.out_dir, out_stem + "_preview.png"))
    render_world_context(raw_gray, meta, args.robot_x, args.robot_y, args.robot_heading_deg,
                         goal, obstacles,
                         os.path.join(args.out_dir, out_stem + "_world_context.png"))


def render_local_scene(occ, goal, obstacles, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    half = MAP_EXTENT / 2.0
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.imshow(occ, extent=[-half, half, -half, half], origin="lower",
              cmap="gray_r", alpha=0.55, zorder=0, interpolation="nearest")

    all_x = [0.0, float(goal[0]), -half, half]
    all_y = [0.0, float(goal[1]), -half, half]
    if len(obstacles):
        all_x += obstacles[:, 0].tolist()
        all_y += obstacles[:, 1].tolist()
    margin = 0.8
    xmin, xmax = min(all_x) - margin, max(all_x) + margin
    ymin, ymax = min(all_y) - margin, max(all_y) + margin
    r = max(xmax - xmin, ymax - ymin)
    xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
    ax.set_xlim(xc - r / 2, xc + r / 2)
    ax.set_ylim(yc - r / 2, yc + r / 2)

    cmap_human = plt.cm.get_cmap("hsv", 10)
    for j, (px, py, vx, vy) in enumerate(obstacles):
        h_color = cmap_human(j % 10)
        ax.add_artist(plt.Circle((px, py), HUMAN_RADIUS, fill=False,
                                 edgecolor=h_color, linewidth=2.2, zorder=5))
        ax.text(px, py, str(j), color=h_color, fontsize=14, fontweight="bold",
                ha="center", va="center", zorder=6)
        speed = float(np.hypot(vx, vy))
        if speed > 0.05:
            ux, uy = vx / speed, vy / speed
            start = (px + HUMAN_RADIUS * ux, py + HUMAN_RADIUS * uy)
            end = (start[0] + 0.55 * ux, start[1] + 0.55 * uy)
            ax.add_patch(patches.FancyArrowPatch(
                start, end, color=h_color, linewidth=2.6,
                arrowstyle=patches.ArrowStyle("->", head_length=5, head_width=3),
                shrinkA=0, shrinkB=0, zorder=6))

    ax.add_artist(plt.Circle((0.0, 0.0), ROBOT_RADIUS, fill=True,
                             facecolor="gold", edgecolor="black",
                             linewidth=2.2, zorder=7))
    ax.add_patch(patches.FancyArrowPatch(
        (0.0, 0.0), (ROBOT_RADIUS, 0.0), color="red",
        arrowstyle=patches.ArrowStyle("->", head_length=4, head_width=2),
        linewidth=1.6, zorder=8))
    ax.plot(goal[0], goal[1], marker="*", color="red", markersize=14, zorder=7)

    ax.set_xlabel("x (m)", fontsize=14)
    ax.set_ylabel("y (m)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title("Local scene preview (ego frame, +x = robot heading)", fontsize=12)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved local scene preview: {out_path}")


def render_world_context(raw_gray, meta, robot_x, robot_y, heading_deg, goal, obstacles, out_path):
    """Overlay the chosen robot pose + 10x10m crop footprint + pedestrians +
    goal on top of the FULL raw building map, in world coordinates — lets
    you confirm the placement makes sense in situ before committing to it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    wmin_x, wmin_y = meta["min"][0], meta["min"][1]
    wmax_x, wmax_y = meta["max"][0], meta["max"][1]

    fig, ax = plt.subplots(figsize=(10, 10 * raw_gray.shape[0] / raw_gray.shape[1]))
    ax.imshow(raw_gray, extent=[wmin_x, wmax_x, wmin_y, wmax_y],
             origin="upper", cmap="gray_r", interpolation="nearest")

    half = MAP_EXTENT / 2.0
    corners_local = [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]
    corners_world = [local_to_world(lx, ly, robot_x, robot_y, heading_deg) for lx, ly in corners_local]
    xs, ys = zip(*corners_world)
    ax.plot(xs, ys, '-', color='lime', linewidth=2.0, label='10m x 10m crop footprint')

    ax.plot(robot_x, robot_y, marker='o', markerfacecolor='gold',
           markeredgecolor='black', markersize=12, zorder=6, label='Robot')
    theta = np.deg2rad(heading_deg)
    ax.add_patch(patches.FancyArrowPatch(
        (robot_x, robot_y),
        (robot_x + 1.0 * np.cos(theta), robot_y + 1.0 * np.sin(theta)),
        color='red', linewidth=2.2,
        arrowstyle=patches.ArrowStyle('->', head_length=6, head_width=4), zorder=7))

    gx_w, gy_w = local_to_world(goal[0], goal[1], robot_x, robot_y, heading_deg)
    ax.plot(gx_w, gy_w, marker='*', color='red', markersize=16, zorder=6, label='Goal')

    cmap_human = plt.cm.get_cmap('hsv', 10)
    for j, (px, py, vx, vy) in enumerate(obstacles):
        pwx, pwy = local_to_world(px, py, robot_x, robot_y, heading_deg)
        ax.plot(pwx, pwy, marker='o', markerfacecolor='none',
               markeredgecolor=cmap_human(j % 10), markeredgewidth=2.2,
               markersize=10, zorder=6)
        ax.text(pwx, pwy, str(j), color=cmap_human(j % 10), fontsize=11,
               fontweight='bold', ha='center', va='center', zorder=7)

    ax.set_xlabel('world x (m)', fontsize=13)
    ax.set_ylabel('world y (m)', fontsize=13)
    ax.set_title('World context — chosen robot pose + crop footprint', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.tick_params(labelsize=10)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"Saved world-context preview: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--png_path', type=str, default=DEFAULT_PNG_PATH)
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR)

    parser.add_argument('--preview_map', action='store_true', default=False,
                        help='Just render the whole raw map with a world meter grid, then exit.')

    parser.add_argument('--robot_x', type=float, default=None, help='World x (m)')
    parser.add_argument('--robot_y', type=float, default=None, help='World y (m)')
    parser.add_argument('--robot_heading_deg', type=float, default=0.0,
                        help='World heading, degrees CCW from world +x (default 0)')
    parser.add_argument('--goal', type=float, nargs=2, metavar=('X', 'Y'),
                        default=[6.0, 0.0],
                        help='Goal in LOCAL/ego frame, e.g. --goal 6.0 0.0 (default: 6.0 0.0). '
                             'Space-separated (NOT comma-separated) so negative values like '
                             '"-6.0 -7.0" parse correctly.')
    parser.add_argument('--ped', type=float, nargs=4, action='append', default=None,
                        metavar=('X', 'Y', 'VX', 'VY'),
                        help='One pedestrian in LOCAL/ego frame, e.g. --ped 3.0 -0.5 -0.6 0.0. '
                             'Space-separated. Repeat for multiple pedestrians '
                             '(count = number of --ped flags).')
    parser.add_argument('--threat_type', type=str, default='manual_interiorgs')
    parser.add_argument('--out_name', type=str, default='InteriorGS_manual4',
                        help='Output basename -> sample_<out_name>.npz')

    args = parser.parse_args()

    if args.preview_map:
        raw_gray = read_occupancy_png(args.png_path)
        meta_path = args.png_path.replace("_occupancy.png", "_metadata.json")
        meta = json.load(open(meta_path))
        os.makedirs(args.out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.png_path))[0]
        render_world_preview(raw_gray, meta, os.path.join(args.out_dir, stem + "_world_preview.png"))
        return

    if args.robot_x is None or args.robot_y is None:
        parser.error('--robot_x and --robot_y are required unless --preview_map is given '
                    '(run with --preview_map first to pick coordinates)')

    if args.ped is None:
        args.ped = []

    build_scene(args)


if __name__ == '__main__':
    main()
