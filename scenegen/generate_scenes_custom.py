"""
Scene generator: hand-designed procedural map layouts
==========================================================================

Generates scenes with procedurally defined static map structures (corridors,
doorways, rooms, L-corners, dead-ends, pillars, narrow passages) combined with
the same social-obstacle placement logic used in generate_scenes.py.

Key design choices:
  - 60% MAP_PRESENCE_PROB — all scenes use one of the hand-crafted map types
  - Obstacle placement respects the generated static map (checks during placement
    AND post-hoc filter), consistent with generate_scenes.py
  - Spawn safety is a dynamic imminent-collision guard (R_c), not a static keep-out disc
    (0.9m) in the expert-trajectory generator, giving the planner room to react
  - V_MAX=1.0 matches the planner and other generators
  - Fully seeded via --seed for reproducibility

Usage:
    python generate_scenes_custom.py --save_dir data/scenes/custom --n_samples 50000 --seed 42
"""

import argparse
import os
import numpy as np
import random

# =========================
# Config (must match generate_scenes.py and the expert-traj generator)
# =========================
pos_min, pos_max       = -7.5, 7.5

# --- Agent radii: SINGLE SOURCE OF TRUTH, must match the expert cost -----------
# r_robot = r_human = 0.25 m, contact R_c = 0.50 m (≈ Hall intimate 0.46 m).
ROBOT_R    = 0.25
HUMAN_R    = 0.25
R_C        = ROBOT_R + HUMAN_R    # 0.50 m
MIN_GAP    = 2 * R_C              # 1.00 m minimum passable gap
NAV_RADIUS = ROBOT_R             # C-space erosion radius (was 0.30)

MAX_AGENTS    = 10               # model keeps k-closest MAX_AGENTS; never place more
max_obstacles = MAX_AGENTS
ROBOT_ORIGIN           = np.array([0.0, 0.0])
OBSTACLE_MIN_SEPARATION= 0.5
MAX_PLACEMENT_ATTEMPTS = 50

HORIZON = 32
DT      = 0.05
LATERAL_SPREAD_HARD   = 0.4
LATERAL_SPREAD_MEDIUM = 1.2
V_MAX   = 1.0   # matches planner
W0_MAX  = np.pi #max Jackal limit

# Grid spec — must match all other generators
MAP_SIZE     = 50
MAP_EXTENT_M = 10
MAP_RES      = MAP_EXTENT_M / MAP_SIZE   # 0.20 m/cell

MAP_PRESENCE_PROB = 0.85

MAP_TYPE_PROBS = {
    "open":            0.05,
    "single_wall":     0.15,
    "corridor":        0.15,
    "L_corner":        0.12,
    "dead_end":        0.10,
    "doorway":         0.13,
    "room":            0.10,
    "pillars":         0.12,
    "narrow_passage":  0.08,
}

WALL_THICKNESS_M      = 0.3
GOAL_PATH_CLEARANCE_M = 0.5
START_CLEARANCE_M     = 0.6   # ≥ map inflation in the expert-traj generator
MIN_NON_TRIVIAL_FRAC = 0.40   # post-hoc filter for non-trivial (detour) paths

# --- Pedestrian kinematics: grounded in the same literature as the cost --------
# v_bar = 1.34 m/s = Weidmann/Helbing free-flow speed the yield ladder is
# calibrated to (tau_a = D(s)/v_bar).
PED_SPEED_MEAN  = 1.34   # m/s — free-flow preferred walking speed (matches v_bar)
PED_SPEED_STD   = 0.26
PED_SPEED_MIN   = 0.5
PED_SPEED_MAX   = 1.8    # fast walk; running is out of distribution on purpose

# --- Spawn safety: dynamic imminent-collision guard, no static keep-out disc ---
# Allows close-but-safe (closing) starts inside the personal/social zone — the
# regime the prox/yield/cut axes shape — while forbidding immediate and imminent-
# unavoidable collisions, so the robot is never taught it may start IN a violation.
SPAWN_MIN_DIST  = R_C + 0.30            # 0.80 m — never start inside intimate space
T_SPAWN_SAFE    = 0.6    # s  — imminent-collision guard horizon
T_PLAN          = HORIZON * DT          # 1.6 s
ENCOUNTER_T_LO  = 0.35   # s  — stage interactions inside the plan window
ENCOUNTER_T_HI  = 1.6
ONCOMING_AIM_STD   = np.deg2rad(8)      # tight aim → real mutual-aim encounters
GROUP_MEMBER_VEL_STD = 0.04             # m/s per axis — keeps pairs coherent
GROUP_ABREAST_SPACING = (0.6, 1.1)      # m — empirical dyad/triad spacing

# =========================
# Grid helpers
# =========================
def world_to_grid(x, y):
    col = int(round(x / MAP_RES + MAP_SIZE / 2))
    row = int(round(y / MAP_RES + MAP_SIZE / 2))
    return np.clip(row, 0, MAP_SIZE - 1), np.clip(col, 0, MAP_SIZE - 1)


def grid_to_world(row, col):
    return (col - MAP_SIZE / 2 + 0.5) * MAP_RES, (row - MAP_SIZE / 2 + 0.5) * MAP_RES


def empty_map():
    return np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)


# =========================
# Map structure builders
# =========================
def add_wall_segment(occ, p0, p1, thickness_m=WALL_THICKNESS_M):
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    length = np.linalg.norm(p1 - p0)
    if length < 1e-6:
        return
    half_t = thickness_m / 2.0
    seg = p1 - p0
    seg_len_sq = float(np.dot(seg, seg))
    xmin, xmax = min(p0[0], p1[0]) - half_t - MAP_RES, max(p0[0], p1[0]) + half_t + MAP_RES
    ymin, ymax = min(p0[1], p1[1]) - half_t - MAP_RES, max(p0[1], p1[1]) + half_t + MAP_RES
    r0, c0 = world_to_grid(xmin, ymin)
    r1, c1 = world_to_grid(xmax, ymax)
    for r in range(min(r0, r1), max(r0, r1) + 1):
        for c in range(min(c0, c1), max(c0, c1) + 1):
            x, y = grid_to_world(r, c)
            pt = np.array([x, y], dtype=np.float32)
            t = float(np.dot(pt - p0, seg) / seg_len_sq)
            t = max(0.0, min(1.0, t))
            d = np.linalg.norm(pt - (p0 + t * seg))
            if d <= half_t:
                occ[r, c] = 1.0


def add_rect(occ, center, size_m, angle_rad=0.0):
    cx, cy = center
    w, h = size_m
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    bbox = max(w, h)
    r0, c0 = world_to_grid(cx - bbox, cy - bbox)
    r1, c1 = world_to_grid(cx + bbox, cy + bbox)
    for r in range(min(r0, r1), max(r0, r1) + 1):
        for c in range(min(c0, c1), max(c0, c1) + 1):
            x, y = grid_to_world(r, c)
            dx, dy = x - cx, y - cy
            lx = cos_a * dx + sin_a * dy
            ly = -sin_a * dx + cos_a * dy
            if abs(lx) <= w / 2 and abs(ly) <= h / 2:
                occ[r, c] = 1.0


def start_clearance_ok(occ, clearance_m=START_CLEARANCE_M):
    r, c = world_to_grid(0.0, 0.0)
    rad = int(np.ceil(clearance_m / MAP_RES))
    r_lo, r_hi = max(0, r - rad), min(MAP_SIZE, r + rad + 1)
    c_lo, c_hi = max(0, c - rad), min(MAP_SIZE, c + rad + 1)
    return not occ[r_lo:r_hi, c_lo:c_hi].any()


def robot_can_reach_goal(occ_float, goal, obstacles):
    """[FEASIBILITY] After pedestrians are placed, does a robot-radius-wide
    corridor from start to goal still exist? Rasterizes each ped's t=0 disc
    (radius R_C), erodes by the robot radius, and tests that the robot's
    8-connected free component reaches the goal cell (or crop border for
    out-of-crop goals). This is where MIN_GAP bites: two obstacles < 2*R_C apart
    sever the component. Keeps infeasible ground truth out of training."""
    from scipy.ndimage import binary_erosion, label as _label
    occ = (occ_float > 0.5).copy() if occ_float is not None else \
        np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    ped_rad = int(np.ceil(R_C / MAP_RES))
    for o in obstacles:
        r, c = world_to_grid(o[0], o[1])
        r0, r1 = max(0, r - ped_rad), min(MAP_SIZE, r + ped_rad + 1)
        c0, c1 = max(0, c - ped_rad), min(MAP_SIZE, c + ped_rad + 1)
        occ[r0:r1, c0:c1] = True
    nav = max(1, int(np.ceil(NAV_RADIUS / MAP_RES)))
    struct = np.ones((2 * nav + 1, 2 * nav + 1), dtype=bool)
    free = binary_erosion(~occ, structure=struct, border_value=1)  # outside crop = free (global planner routes there)
    rr = rc = MAP_SIZE // 2
    if not free[rr, rc]:
        return False
    labeled, _ = _label(free, structure=np.ones((3, 3), dtype=int))
    comp = labeled == labeled[rr, rc]
    if abs(goal[0]) < MAP_EXTENT_M / 2 and abs(goal[1]) < MAP_EXTENT_M / 2:
        gr, gc = world_to_grid(goal[0], goal[1])
        return bool(comp[np.clip(gr, 0, MAP_SIZE - 1), np.clip(gc, 0, MAP_SIZE - 1)])
    border = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    return bool((comp & border).any())


def map_blocks_pos(occ, pos, clearance_m=0.3):
    r, c = world_to_grid(pos[0], pos[1])
    rad = int(np.ceil(clearance_m / MAP_RES))
    r_lo, r_hi = max(0, r - rad), min(MAP_SIZE, r + rad + 1)
    c_lo, c_hi = max(0, c - rad), min(MAP_SIZE, c + rad + 1)
    return bool(occ[r_lo:r_hi, c_lo:c_hi].any())


def goal_is_reachable(occ: np.ndarray, goal: np.ndarray) -> bool:
    """
    BFS 8-connected reachability check in C-space from robot (0, 0) to goal.
    Free space is eroded by NAV_RADIUS (robot radius) before the check so that gaps
    narrower than the Jackal body are treated as impassable — a path through
    a 0.1m crack is not valid even if cells are technically connected.
    A blocked straight-line path is fine (and desired for hard scenarios).
    """
    from scipy.ndimage import binary_erosion
    free_raw = occ < 0.5
    _nav_rad = max(1, int(np.ceil((NAV_RADIUS + 0.2) / MAP_RES)))  # C-space erosion at robot radius
    _struct   = np.ones((2 * _nav_rad + 1, 2 * _nav_rad + 1), dtype=bool)
    free = binary_erosion(free_raw, structure=_struct, border_value=1)

    rr, rc = world_to_grid(0.0, 0.0)
    gr, gc = world_to_grid(float(goal[0]), float(goal[1]))
    if not free[rr, rc] or not free[gr, gc]:
        return False
    if rr == gr and rc == gc:
        return True
    visited = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    stack = [(rr, rc)]
    visited[rr, rc] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < MAP_SIZE and 0 <= nc < MAP_SIZE and not visited[nr, nc] and free[nr, nc]:
                if nr == gr and nc == gc:
                    return True
                visited[nr, nc] = True
                stack.append((nr, nc))
    return False


def path_is_clear_proc(occ: np.ndarray, goal: np.ndarray, n_steps: int = 40) -> bool:
    """True if the straight-line path from (0, 0) to goal is obstacle-free (for stats)."""
    xs   = np.linspace(0.0, float(goal[0]), n_steps)
    ys   = np.linspace(0.0, float(goal[1]), n_steps)
    cols = np.clip(np.round(xs / MAP_RES + MAP_SIZE / 2).astype(int), 0, MAP_SIZE - 1)
    rows = np.clip(np.round(ys / MAP_RES + MAP_SIZE / 2).astype(int), 0, MAP_SIZE - 1)
    return not (occ[rows, cols] > 0.5).any()


# ---- Map generators ----
def gen_open(goal, rng):
    return empty_map()


def gen_single_wall(goal, rng):
    occ = empty_map()
    if np.linalg.norm(goal) < 0.1:
        wall_angle = rng.uniform(0, np.pi)
    else:
        wall_angle = np.arctan2(goal[1], goal[0]) + np.pi / 2 + rng.normal(0, 0.3)
    length = rng.uniform(2.0, 4.5)
    offset_dir = np.array([np.cos(wall_angle - np.pi / 2), np.sin(wall_angle - np.pi / 2)])
    offset_dist = rng.uniform(1.5, 3.0) * rng.choice([-1, 1])
    center = offset_dir * offset_dist
    dx, dy = np.cos(wall_angle), np.sin(wall_angle)
    p0 = center + np.array([dx, dy]) * (length / 2)
    p1 = center - np.array([dx, dy]) * (length / 2)
    add_wall_segment(occ, p0, p1)
    return occ


def gen_corridor(goal, rng):
    occ = empty_map()
    axis_angle = np.arctan2(goal[1], goal[0]) if np.linalg.norm(goal) > 0.1 else rng.uniform(0, np.pi)
    half_w = rng.uniform(1.0, 1.5)
    axis  = np.array([np.cos(axis_angle), np.sin(axis_angle)])
    perp  = np.array([-axis[1], axis[0]])
    length = rng.uniform(4.0, 6.5)
    for side in [-1, 1]:
        c = side * half_w * perp
        add_wall_segment(occ, c + axis * (length / 2), c - axis * (length / 2))
    return occ


def gen_L_corner(goal, rng):
    occ = empty_map()
    corner    = rng.uniform(-2.5, 2.5, size=2).astype(np.float32)
    base_angle = rng.uniform(0, 2 * np.pi)
    arm1_dir   = np.array([np.cos(base_angle), np.sin(base_angle)])
    arm2_dir   = np.array([-arm1_dir[1], arm1_dir[0]])
    if rng.random() < 0.5:
        arm2_dir = -arm2_dir
    len1, len2 = rng.uniform(2.0, 3.5), rng.uniform(2.0, 3.5)
    add_wall_segment(occ, corner, corner + arm1_dir * len1)
    add_wall_segment(occ, corner, corner + arm2_dir * len2)
    return occ


def gen_dead_end(goal, rng):
    occ = empty_map()
    if np.linalg.norm(goal) > 0.1:
        away = -goal / np.linalg.norm(goal)
        pocket_center = away * rng.uniform(2.0, 3.0) + rng.normal(0, 0.8, size=2)
    else:
        pocket_center = rng.uniform(-2.5, 2.5, size=2)
    opening_angle = rng.uniform(0, 2 * np.pi)
    open_dir = np.array([np.cos(opening_angle), np.sin(opening_angle)])
    perp     = np.array([-open_dir[1], open_dir[0]])
    width    = rng.uniform(1.2, 2.0)
    depth    = rng.uniform(1.5, 2.5)
    back_center = pocket_center - open_dir * depth
    p0 = back_center + perp * width
    p1 = back_center - perp * width
    add_wall_segment(occ, p0, p1)
    add_wall_segment(occ, p0, pocket_center + perp * width)
    add_wall_segment(occ, p1, pocket_center - perp * width)
    return occ


def gen_doorway(goal, rng):
    occ = empty_map()
    if np.linalg.norm(goal) < 0.1:
        wall_angle = rng.uniform(0, np.pi)
    else:
        wall_angle = np.arctan2(goal[1], goal[0]) + np.pi / 2 + rng.normal(0, 0.2)
    axis = np.array([np.cos(wall_angle), np.sin(wall_angle)])
    midpoint = goal * rng.uniform(0.35, 0.65) if np.linalg.norm(goal) > 0.1 else np.zeros(2)
    door_width  = rng.uniform(1.5, 2.2)
    door_offset = rng.uniform(-0.5, 0.5)
    door_center = midpoint + axis * door_offset
    half_len    = 3.5
    add_wall_segment(occ, door_center + axis * (door_width / 2), door_center + axis * half_len)
    add_wall_segment(occ, door_center - axis * (door_width / 2), door_center - axis * half_len)
    return occ


def gen_room(goal, rng):
    occ = empty_map()
    half_w = rng.uniform(2.0, 3.0)
    half_h = rng.uniform(2.0, 3.0)
    cx, cy = rng.uniform(-0.5, 0.5, size=2)
    corners = [
        np.array([cx - half_w, cy - half_h]),
        np.array([cx + half_w, cy - half_h]),
        np.array([cx + half_w, cy + half_h]),
        np.array([cx - half_w, cy + half_h]),
    ]
    if np.linalg.norm(goal) > 0.1:
        gx, gy = goal
        sides = [abs(gy - (cy - half_h)), abs(gx - (cx + half_w)),
                 abs(gy - (cy + half_h)), abs(gx - (cx - half_w))]
        opening_side = int(np.argmin(sides))
    else:
        opening_side = int(rng.integers(0, 4))
    door_width = rng.uniform(1.5, 2.2)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        if i == opening_side:
            mid = (a + b) / 2
            direction = (b - a) / (np.linalg.norm(b - a) + 1e-6)
            add_wall_segment(occ, a, mid - direction * (door_width / 2))
            add_wall_segment(occ, mid + direction * (door_width / 2), b)
        else:
            add_wall_segment(occ, a, b)
    return occ


def gen_pillars(goal, rng):
    occ = empty_map()
    n_pillars = int(rng.integers(3, 8))
    placed = []
    for _ in range(n_pillars):
        for _ in range(20):
            x = rng.uniform(-3.0, 3.0)
            y = rng.uniform(-3.0, 3.0)
            if np.linalg.norm([x, y]) < 0.8:
                continue
            if np.linalg.norm([x - goal[0], y - goal[1]]) < 0.6:
                continue
            if any(np.linalg.norm([x - px, y - py]) < 0.8 for px, py in placed):
                continue
            size = rng.uniform(0.3, 0.6)
            add_rect(occ, (x, y), (size, size), angle_rad=rng.uniform(0, np.pi / 2))
            placed.append((x, y))
            break
    return occ


def gen_narrow_passage(goal, rng):
    occ = empty_map()
    if np.linalg.norm(goal) < 0.1:
        passage_angle = rng.uniform(0, np.pi)
        midpoint = rng.uniform(-2, 2, size=2)
    else:
        passage_angle = np.arctan2(goal[1], goal[0]) + np.pi / 2
        midpoint = goal * rng.uniform(0.3, 0.6)
    perp = np.array([np.cos(passage_angle), np.sin(passage_angle)])
    gap  = rng.uniform(1.5, 1.8)
    for side in [-1, 1]:
        center = midpoint + side * perp * (gap / 2 + 0.6)
        size   = (rng.uniform(0.8, 1.4), rng.uniform(0.8, 1.4))
        add_rect(occ, center, size, angle_rad=rng.uniform(0, np.pi / 2))
    return occ


MAP_GENERATORS = {
    "open":            gen_open,
    "single_wall":     gen_single_wall,
    "corridor":        gen_corridor,
    "L_corner":        gen_L_corner,
    "dead_end":        gen_dead_end,
    "doorway":         gen_doorway,
    "room":            gen_room,
    "pillars":         gen_pillars,
    "narrow_passage":  gen_narrow_passage,
}


def sample_static_map(goal, rng, max_retries=8):
    types = list(MAP_TYPE_PROBS.keys())
    probs = np.array([MAP_TYPE_PROBS[t] for t in types], dtype=np.float64)
    probs /= probs.sum()
    for _ in range(max_retries):
        map_type = rng.choice(types, p=probs)
        occ = MAP_GENERATORS[map_type](goal, rng)
        if not start_clearance_ok(occ):
            continue
        gr, gc = world_to_grid(float(goal[0]), float(goal[1]))
        if occ[gr, gc] > 0.5:
            continue
        # Only reject if goal is truly unreachable (sealed off); a blocked
        # straight-line path is desirable — that is the navigation challenge.
        if map_type != "open" and not goal_is_reachable(occ, goal):
            continue
        return occ, 1.0, map_type
    # Exhausted retries: this is a genuine no-map scene (all-zero occupancy), so
    # label it has_map=0.0 / "none" to match the explicit no-map branch. Labeling
    # an identical all-zero array has_map=1.0 gives the model contradictory
    # supervision for the same visual input.
    return empty_map(), 0.0, "none"


# =========================
# Sampling helpers — consistent with generate_scenes.py
# =========================
def sample_goal_distance(rng):
    r = rng.random()
    if r < 0.15:
        return rng.uniform(0.3, 1.0)
    elif r < 0.30:
        return rng.uniform(1.0, 2.0)
    elif r < 0.80:
        return rng.uniform(2.0, 5.0)
    else:
        return rng.uniform(5.0, 7.0)


def sample_initial_velocity(rng):
    r = rng.random()
    if r < 0.15:
        return 0.0
    elif r < 0.25:
        return V_MAX
    else:
        return float(rng.uniform(0.0, V_MAX))


def sample_obstacle_count(difficulty, rng):
    if difficulty == "easy":
        return int(rng.integers(0, max_obstacles + 1))
    r = rng.random()
    if r < 0.30:
        return int(rng.integers(0, 4))
    elif r < 0.80:
        return int(rng.integers(3, 8))
    else:
        return int(rng.integers(7, max_obstacles + 1))


def sample_ped_speed(rng, lo=PED_SPEED_MIN, hi=PED_SPEED_MAX):
    return float(np.clip(rng.normal(PED_SPEED_MEAN, PED_SPEED_STD), lo, hi))


def _rot(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _path_frame(goal):
    path_vec = goal - ROBOT_ORIGIN
    L = float(np.linalg.norm(path_vec))
    d = path_vec / (L + 1e-6)
    return L, d, np.array([-d[1], d[0]])


def spawn_is_safe(pos, vel, t_safe=T_SPAWN_SAFE, r_contact=R_C,
                  d_floor=SPAWN_MIN_DIST):
    """[T5] Spawn admissibility: (1) starts outside intimate space (never IN a
    violation at t=0), (2) constant-velocity line does not reach contact (R_c)
    with the robot start within t_safe. Allows close-but-closing starts inside
    the personal/social zone — the regime the prox/yield/cut axes shape."""
    p = np.asarray(pos, dtype=np.float64)
    v = np.asarray(vel, dtype=np.float64)
    if float(np.linalg.norm(p)) < d_floor:
        return False
    vv = float(v @ v)
    t_star = 0.0 if vv < 1e-9 else float(np.clip(-(p @ v) / vv, 0.0, t_safe))
    return float(np.linalg.norm(p + v * t_star)) >= r_contact


def velocity_clears_map(occ_float, pos, vel, horizon_s=T_PLAN, step_s=0.2,
                        clearance_m=0.15):
    """[T6] True if the ped's constant-velocity path stays out of walls for
    the whole plan horizon (cells outside the map count as free)."""
    if occ_float is None:
        return True
    p = np.asarray(pos, dtype=np.float64)
    v = np.asarray(vel, dtype=np.float64)
    half = MAP_EXTENT_M / 2
    for t in np.arange(step_s, horizon_s + 1e-9, step_s):
        q = p + v * t
        if abs(q[0]) > half or abs(q[1]) > half:
            continue
        r = int(np.clip(round(q[1] / MAP_RES + MAP_SIZE / 2), 0, MAP_SIZE - 1))
        c = int(np.clip(round(q[0] / MAP_RES + MAP_SIZE / 2), 0, MAP_SIZE - 1))
        rad = int(np.ceil(clearance_m / MAP_RES))
        r0, r1 = max(0, r - rad), min(MAP_SIZE, r + rad + 1)
        c0, c1 = max(0, c - rad), min(MAP_SIZE, c + rad + 1)
        if occ_float[r0:r1, c0:c1].any():
            return False
    return True


def sanitize_obstacles(obstacles, occ_float, rng):
    """[T5]+[T6] Final pass: fix spawn-unsafe or wall-crossing peds by
    re-aiming (few tries), then slowing, then freezing — never deleting, so
    obstacle-count statistics are preserved."""
    out = []
    for o in obstacles:
        pos = np.asarray(o[:2], dtype=np.float64)
        vel = np.asarray(o[2:4], dtype=np.float64)
        ok = (spawn_is_safe(pos, vel)
              and velocity_clears_map(occ_float, pos, vel))
        if not ok:
            speed = float(np.linalg.norm(vel))
            for _ in range(8):                       # re-aim
                ang = rng.uniform(0, 2 * np.pi)
                v_try = speed * np.array([np.cos(ang), np.sin(ang)])
                if (spawn_is_safe(pos, v_try)
                        and velocity_clears_map(occ_float, pos, v_try)):
                    vel, ok = v_try, True
                    break
        if not ok and spawn_is_safe(pos, vel * 0.3) \
                and velocity_clears_map(occ_float, pos, vel * 0.3):
            vel, ok = vel * 0.3, True                # slow down
        if not ok:
            vel = np.zeros(2)                        # freeze
        out.append([float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1])])
    return out


def sample_oncoming_encounter(goal, placed_obstacles, rng, occ_float=None,
                              map_blocks_fn=None):
    """Corridor pass: ped ahead on the path walking back at the robot with a
    small lateral offset and tight aim. Stages CPA inside the plan window:
    closing ≈ V_MAX + ped speed. Exercises c1 (which side to pass) and c2
    (mutual head-on)."""
    L, d, perp = _path_frame(goal)
    if L < 1.5:
        return None
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        speed   = sample_ped_speed(rng)
        closing = V_MAX + speed
        t_meet  = rng.uniform(ENCOUNTER_T_LO + 0.2, ENCOUNTER_T_HI + 0.4)
        dist    = max(1.1, min(closing * t_meet, L + 1.0, 6.5))
        lateral = float(rng.normal(0, 0.45))
        pos = ROBOT_ORIGIN + dist * d + lateral * perp
        vel = speed * _rot(-d, rng.normal(0, ONCOMING_AIM_STD))
        if (_spot_free(pos, placed_obstacles, occ_float, map_blocks_fn)
                and spawn_is_safe(pos, vel)
                and velocity_clears_map(occ_float, pos, vel)):
            return [pos[0], pos[1], float(vel[0]), float(vel[1])]
    return None


def sample_overtake_encounter(goal, placed_obstacles, rng, occ_float=None,
                              map_blocks_fn=None):
    """Slower ped ahead walking the SAME direction: the robot catches up from
    behind. Exercises c3 (don't tailgate dead-astern → pull to a flank) and
    c1 (which flank). Catch-up time = gap / (V_MAX − ped speed) staged
    in-window."""
    L, d, perp = _path_frame(goal)
    if L < 1.2:
        return None
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        speed   = float(rng.uniform(0.15, 0.55))           # clearly slower
        t_catch = rng.uniform(ENCOUNTER_T_LO + 0.3, ENCOUNTER_T_HI)
        gap     = max(1.05, (V_MAX - speed) * t_catch)
        lateral = float(rng.normal(0, 0.3))
        pos = ROBOT_ORIGIN + gap * d + lateral * perp
        vel = speed * _rot(d, rng.normal(0, ONCOMING_AIM_STD))
        if (_spot_free(pos, placed_obstacles, occ_float, map_blocks_fn)
                and spawn_is_safe(pos, vel)
                and velocity_clears_map(occ_float, pos, vel)):
            return [pos[0], pos[1], float(vel[0]), float(vel[1])]
    return None


def sample_side_obstacle_timed(goal, placed_obstacles, rng, occ_float=None,
                               map_blocks_fn=None):
    """[T1] Crossing ped whose ARRIVAL at the robot's path is synchronized
    with the robot's own arrival (both inside the plan window), instead of a
    spatial fraction of a possibly-7 m path with an unrelated speed draw.
    Exercises c5 (cut in front vs sweep behind / yield)."""
    L, d, perp = _path_frame(goal)
    if L < 1.0:
        return None
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        v_nom   = rng.uniform(0.6, 1.0) * V_MAX            # robot nominal speed
        t_meet  = rng.uniform(ENCOUNTER_T_LO, ENCOUNTER_T_HI)
        d_i     = min(v_nom * t_meet, 0.9 * L)
        intercept = ROBOT_ORIGIN + d_i * d
        side    = rng.choice([-1.0, 1.0])
        speed   = sample_ped_speed(rng)
        standoff = speed * t_meet                          # arrives on time
        pos = intercept + side * standoff * perp \
              + rng.normal(0, 0.2) * d
        to_i = intercept - pos
        n = float(np.linalg.norm(to_i))
        if n < 0.5:
            continue
        vel = speed * _rot(to_i / n, rng.normal(0, np.deg2rad(10)))
        if (_spot_free(pos, placed_obstacles, occ_float, map_blocks_fn)
                and spawn_is_safe(pos, vel)
                and velocity_clears_map(occ_float, pos, vel)):
            return [pos[0], pos[1], float(vel[0]), float(vel[1])]
    return None


def sample_standing_group(goal, placed_obstacles, rng, occ_float=None,
                          map_blocks_fn=None, size_range=(2, 3)):
    """[T2] Conversational group: 2–3 STATIC peds spaced 0.6–1.5 m near/on
    the path, inside the plan window. The robot must choose between
    threading the formation (c7) and going around."""
    L, d, perp = _path_frame(goal)
    if L < 1.2:
        return []
    n = int(rng.integers(size_range[0], size_range[1] + 1))
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        along   = rng.uniform(0.9, min(2.8, 0.9 * L))
        lateral = float(rng.normal(0, 0.25))
        center  = ROBOT_ORIGIN + along * d + lateral * perp
        if rng.random() < 0.7:                        # block the corridor
            axis = _rot(perp, rng.normal(0, np.deg2rad(20)))
        else:
            axis = _rot(np.array([1.0, 0.0]), rng.uniform(0, 2 * np.pi))
        spacing = rng.uniform(0.7, 1.5)
        offsets = (np.arange(n) - (n - 1) / 2.0) * spacing
        members = []
        for off in offsets:
            pos = center + off * axis + rng.normal(0, 0.08, size=2)
            if not _spot_free(pos, placed_obstacles + members, occ_float,
                              map_blocks_fn):
                members = []
                break
            members.append([float(pos[0]), float(pos[1]), 0.0, 0.0])
        if len(members) >= 2:
            return members
    return []


def sample_pedestrian_group_v2(goal, placed_obstacles, rng, occ_float=None,
                               map_blocks_fn=None, group_size_range=(2, 4)):
    """[T3] Walking group: ABREAST formation perpendicular to a SHARED group
    velocity (per-member noise N(0, 0.04) per axis — pairs always pass a
    ±30°/0.3 m/s co-movement test). Direction mix: crossing 45 %, oncoming
    35 %, along-path 20 %. Anchor staged so the group is inside the plan
    window when the robot meets it."""
    L, d, perp = _path_frame(goal)
    if L < 1.2:
        return []
    n = int(rng.integers(group_size_range[0], group_size_range[1] + 1))
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        speed  = sample_ped_speed(rng, lo=0.5, hi=1.4)
        t_meet = rng.uniform(0.5, 1.4)               # robot meets group then
        r = rng.random()
        if r < 0.45:                                  # crossing
            g_dir = _rot(perp * rng.choice([-1.0, 1.0]),
                         rng.normal(0, np.deg2rad(12)))
            d_i   = max(1.05, rng.uniform(0.7, 1.0) * V_MAX * t_meet)
            meet  = ROBOT_ORIGIN + d_i * d + rng.normal(0, 0.25) * perp
            anchor = meet - g_dir * speed * t_meet    # arrives on time
        elif r < 0.80:                                # oncoming
            g_dir = _rot(-d, rng.normal(0, np.deg2rad(10)))
            along = max(1.3, (V_MAX + speed) * t_meet)
            anchor = ROBOT_ORIGIN + along * d + rng.normal(0, 0.25) * perp
        else:                                         # along path (overtake)
            g_dir = _rot(d, rng.normal(0, np.deg2rad(10)))
            along = max(1.05, (V_MAX - speed) * t_meet + 0.5)
            anchor = ROBOT_ORIGIN + along * d + rng.normal(0, 0.25) * perp
        g_left  = np.array([-g_dir[1], g_dir[0]])
        spacing = rng.uniform(*GROUP_ABREAST_SPACING)
        offsets = (np.arange(n) - (n - 1) / 2.0) * spacing
        v_base  = g_dir * speed
        members = []
        for off in offsets:
            pos = anchor + off * g_left + rng.normal(0, 0.07, size=2)
            vel = v_base + rng.normal(0, GROUP_MEMBER_VEL_STD, size=2)
            if not (_spot_free(pos, placed_obstacles + members, occ_float,
                               map_blocks_fn)
                    and spawn_is_safe(pos, vel)
                    and velocity_clears_map(occ_float, pos, vel)):
                members = []
                break
            members.append([float(pos[0]), float(pos[1]),
                            float(vel[0]), float(vel[1])])
        if len(members) >= 2:
            return members
    return []


def sample_path_obstacle_zone(goal, lateral_spread, placed_obstacles, rng,
                              occ_float=None, map_blocks_fn=None):
    """[T1] In-zone replacement for fraction-based path placement: absolute
    along-path distance U(1.2, 3.2) m (clipped to the path), where a 1.6 s
    plan at V_MAX can actually meet it."""
    L, d, perp = _path_frame(goal)
    if L < 1e-3:
        return None
    hi = min(3.2, max(1.25, 0.9 * L))
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        along   = rng.uniform(min(1.2, hi - 0.05), hi)
        lateral = float(rng.normal(0, lateral_spread))
        pos = ROBOT_ORIGIN + along * d + lateral * perp
        if _spot_free(pos, placed_obstacles, occ_float, map_blocks_fn):
            return pos
    return None


MAP_THREAT_TABLE = {
    # corridor: crossing threats can't penetrate walls; oncoming/overtake are
    # the native encounters (canonical pass-axis geometry)
    "corridor":       dict(oncoming=0.34, overtake=0.20, frontal=0.08,
                           rear=0.06, group=0.16, standing_group=0.06,
                           near_goal=0.10),
    # doorway: who-goes-first at the door (oncoming staged across the gap),
    # plus people standing near the doorway
    "doorway":        dict(oncoming=0.30, overtake=0.08, frontal=0.10,
                           side=0.08, group=0.12, standing_group=0.14,
                           near_goal=0.10, mixed=0.08),
    # room: conversational groups are the natural occupants
    "room":           dict(standing_group=0.22, group=0.16, oncoming=0.14,
                           frontal=0.10, side=0.12, near_goal=0.16,
                           mixed=0.10),
    "narrow_passage": dict(oncoming=0.24, overtake=0.10, frontal=0.12,
                           side=0.12, standing_group=0.12, group=0.12,
                           near_goal=0.10, mixed=0.08),
}
DEFAULT_THREAT_TABLE = dict(oncoming=0.14, overtake=0.10, frontal=0.10,
                            side=0.13, rear=0.04, mixed=0.10, pincer=0.06,
                            group=0.13, standing_group=0.08, near_goal=0.12)


def pick_threat_for_map(map_type, rng):
    """[H3] Map-aware threat selection: pair the structure with the social
    encounter it naturally hosts. Falls back to the default (patched)
    distribution for open/none/single_wall/L_corner/dead_end/pillars."""
    tbl = MAP_THREAT_TABLE.get(str(map_type), DEFAULT_THREAT_TABLE)
    names = list(tbl)
    p = np.array([tbl[n] for n in names], dtype=np.float64)
    p /= p.sum()
    return str(rng.choice(names, p=p))


# =========================
# Obstacle placement helpers (accept occ for in-placement wall checks)
# =========================
def is_valid_position(pos, placed_obstacles, occ=None):
    if np.linalg.norm(pos - ROBOT_ORIGIN) < SPAWN_MIN_DIST:
        return False
    for prev in placed_obstacles:
        if np.linalg.norm(pos - np.array(prev[:2])) < OBSTACLE_MIN_SEPARATION:
            return False
    if not (pos_min <= pos[0] <= pos_max and pos_min <= pos[1] <= pos_max):
        return False
    if occ is not None and map_blocks_pos(occ, pos, clearance_m=0.3):
        return False
    return True


def _spot_free(pos, placed_obstacles, occ_float, map_blocks_fn=None):
    """Position-only validity (host's is_valid_position minus the call cycle)."""
    if np.linalg.norm(pos - ROBOT_ORIGIN) < SPAWN_MIN_DIST:
        return False
    for prev in placed_obstacles:
        if np.linalg.norm(pos - np.asarray(prev[:2])) < OBSTACLE_MIN_SEPARATION:
            return False
    if not (pos_min <= pos[0] <= pos_max and pos_min <= pos[1] <= pos_max):
        return False
    if occ_float is not None:
        if map_blocks_fn is not None:
            return not map_blocks_fn(occ_float, pos, 0.3)
        return velocity_clears_map(occ_float, pos, np.zeros(2), horizon_s=0.21)
    return True


def make_obstacle(pos, rng, toward_robot_prob=0.3, force_static=False):
    if force_static:
        return [pos[0], pos[1], 0.0, 0.0]
    if rng.random() < toward_robot_prob:
        to_robot = -pos / (np.linalg.norm(pos) + 1e-6)
        noise_angle = rng.normal(0, np.pi / 9)
        c, s = np.cos(noise_angle), np.sin(noise_angle)
        direction = np.array([c * to_robot[0] - s * to_robot[1],
                              s * to_robot[0] + c * to_robot[1]])
        speed = sample_ped_speed(rng)
    else:
        angle = rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        speed = sample_ped_speed(rng)
    vx, vy = direction * speed
    return [pos[0], pos[1], float(vx), float(vy)]


def sample_rear_obstacle(goal, placed_obstacles, rng, occ=None):
    path_vec = goal - ROBOT_ORIGIN
    path_dir = path_vec / (np.linalg.norm(path_vec) + 1e-6)
    perp_dir = np.array([-path_dir[1], path_dir[0]])
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        t = rng.uniform(0.5, 2.5)
        lateral = rng.normal(0, 0.6)
        pos = ROBOT_ORIGIN - t * path_dir + lateral * perp_dir
        if is_valid_position(pos, placed_obstacles, occ):
            return pos
    return None


def sample_side_obstacle(goal, placed_obstacles, rng, occ=None):
    path_vec = goal - ROBOT_ORIGIN
    path_len = np.linalg.norm(path_vec)
    path_dir = path_vec / (path_len + 1e-6)
    perp_dir = np.array([-path_dir[1], path_dir[0]])
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        t = rng.uniform(0.2, 0.7) * path_len
        intercept = ROBOT_ORIGIN + t * path_dir
        side = rng.choice([-1, 1])
        lateral_dist = rng.uniform(1.5, 4.0)
        pos = intercept + side * lateral_dist * perp_dir
        if is_valid_position(pos, placed_obstacles, occ):
            return pos, intercept
    return None, None


def make_rear_obstacle(pos, goal, rng):
    path_vec = goal - ROBOT_ORIGIN
    path_dir = path_vec / (np.linalg.norm(path_vec) + 1e-6)
    aim_point = ROBOT_ORIGIN + path_dir * rng.uniform(0.5, 2.0)
    to_aim    = aim_point - pos
    to_aim   /= (np.linalg.norm(to_aim) + 1e-6)
    noise_angle = rng.normal(0, np.pi / 6)
    c, s = np.cos(noise_angle), np.sin(noise_angle)
    direction = np.array([c * to_aim[0] - s * to_aim[1],
                          s * to_aim[0] + c * to_aim[1]])
    speed = sample_ped_speed(rng)
    vx, vy = direction * speed
    return [pos[0], pos[1], float(vx), float(vy)]


def make_side_obstacle(pos, intercept, rng):
    to_intercept = intercept - pos
    dist = np.linalg.norm(to_intercept)
    if dist < 1e-3:
        return make_obstacle(pos, rng)
    direction = to_intercept / dist
    noise_angle = rng.normal(0, np.pi / 5)
    c, s = np.cos(noise_angle), np.sin(noise_angle)
    direction = np.array([c * direction[0] - s * direction[1],
                          s * direction[0] + c * direction[1]])
    speed = sample_ped_speed(rng)
    vx, vy = direction * speed
    return [pos[0], pos[1], float(vx), float(vy)]


def sample_path_obstacle(goal, lateral_spread, placed_obstacles, rng,
                         occ=None, t_fraction_range=(0.1, 0.9)):
    path_vec = goal - ROBOT_ORIGIN
    path_len = np.linalg.norm(path_vec)
    if path_len < 1e-3:
        return None
    path_dir = path_vec / path_len
    perp_dir = np.array([-path_dir[1], path_dir[0]])
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        t      = rng.uniform(*t_fraction_range)
        lateral = rng.normal(0, lateral_spread)
        pos    = ROBOT_ORIGIN + t * path_len * path_dir + lateral * perp_dir
        if is_valid_position(pos, placed_obstacles, occ):
            return pos
    return None


def sample_random_obstacle(placed_obstacles, rng, occ=None):
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        r     = rng.uniform(0.5, 6.0)
        theta = rng.uniform(0, 2 * np.pi)
        pos   = np.array([r * np.cos(theta), r * np.sin(theta)])
        if is_valid_position(pos, placed_obstacles, occ):
            return pos
    return None


def sample_near_goal_obstacle(goal, placed_obstacles, rng, occ=None):
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        r     = rng.uniform(0.7, 1.2)
        theta = rng.uniform(0, 2 * np.pi)
        pos   = goal + np.array([r * np.cos(theta), r * np.sin(theta)])
        if is_valid_position(pos, placed_obstacles, occ):
            return pos
    return None


def sample_pedestrian_group(goal, placed_obstacles, rng, occ=None,
                             group_size_range=(2, 4)):
    path_vec = goal - ROBOT_ORIGIN
    path_len = np.linalg.norm(path_vec)
    path_dir = path_vec / (path_len + 1e-6)
    perp_dir = np.array([-path_dir[1], path_dir[0]])
    group_size = int(rng.integers(group_size_range[0], group_size_range[1] + 1))
    GROUP_SPREAD = 0.5
    GROUP_MIN_SEP = 0.4
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        t      = rng.uniform(0.2, 0.7) * path_len
        lateral = rng.normal(0, LATERAL_SPREAD_HARD)
        anchor = ROBOT_ORIGIN + t * path_dir + lateral * perp_dir
        if not (pos_min <= anchor[0] <= pos_max and pos_min <= anchor[1] <= pos_max):
            continue
        if np.linalg.norm(anchor - ROBOT_ORIGIN) < SPAWN_MIN_DIST + 0.5:
            continue
        if occ is not None and map_blocks_pos(occ, anchor, 0.3):
            continue
        group_speed = rng.uniform(0.3, 1.2)
        perp_bias   = rng.choice([-1, 1])
        along_bias  = rng.uniform(-0.3, 0.3)
        group_dir   = perp_bias * perp_dir + along_bias * path_dir
        group_dir  /= (np.linalg.norm(group_dir) + 1e-6)
        members, failed = [], False
        for _ in range(group_size):
            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                offset = rng.normal(0, GROUP_SPREAD, size=2)
                pos    = anchor + offset
                too_close = any(np.linalg.norm(pos - np.array(m[:2])) < GROUP_MIN_SEP
                                for m in members)
                if (is_valid_position(pos, placed_obstacles, occ) and not too_close
                        and pos_min <= pos[0] <= pos_max
                        and pos_min <= pos[1] <= pos_max):
                    noise = rng.normal(0, 0.15, size=2)
                    vel   = group_dir * group_speed + noise
                    members.append([pos[0], pos[1], float(vel[0]), float(vel[1])])
                    break
            else:
                failed = True
                break
        if not failed and len(members) >= 2:
            return members
    return []


# =========================
# Main generation
# =========================
def generate(save_dir: str, n_samples: int, seed: int = 42):
    os.makedirs(save_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    # (All sampling in this file goes through `rng`; the map generators take it
    # explicitly. No bare np.random.* / random.* calls, so no global seeding.)

    difficulty_counts    = {"hard": 0, "medium": 0, "easy": 0, "at_goal": 0}
    threat_counts        = {}
    map_type_counts      = {}
    nontrivial_path_count = 0   # straight line from robot to goal is wall-blocked
    map_with_path_count   = 0   # denominator: non-at_goal scenes with a map

    saved = 0
    k = 0
    while saved < n_samples:
        k += 1
        if saved % 10000 == 0:
            print(f"  {saved}/{n_samples} ...")

        r = rng.random()
        if r < 0.05:
            scenario = "at_goal"
        elif r < 0.65:
            scenario = "hard"
        elif r < 0.88:
            scenario = "medium"
        else:
            scenario = "easy"

        obstacles = []

        # ── Static map ────────────────────────────────────────────────────────
        map_prob = 0.4 if scenario == "at_goal" else MAP_PRESENCE_PROB
        use_map  = (rng.random() < map_prob)
        if use_map:
            # Need goal to generate map — use a temp goal estimate if at_goal
            temp_goal = np.array([0.0, 0.0]) if scenario == "at_goal" else None

            # For non-at_goal, we generate goal first so map can align with it
            if scenario != "at_goal":
                goal_dist  = float(sample_goal_distance(rng))
                force_behind = (scenario != "easy") and (rng.random() < 0.10)
                if force_behind:
                    goal_angle = float(rng.uniform(np.pi / 2, 3 * np.pi / 2))
                    if goal_angle > np.pi:
                        goal_angle -= 2 * np.pi
                else:
                    goal_angle = float(rng.uniform(-np.pi, np.pi))
                goal = np.array([goal_dist * np.cos(goal_angle),
                                 goal_dist * np.sin(goal_angle)], dtype=np.float32)
                goal = np.clip(goal, pos_min, pos_max)
                occ_map, has_map_flag, map_type = sample_static_map(goal, rng)
            else:
                # at_goal: tiny goal, generate map against it
                goal_dist  = float(rng.uniform(0.0, 0.3))
                goal_angle = float(rng.uniform(-np.pi, np.pi))
                goal = np.array([goal_dist * np.cos(goal_angle),
                                 goal_dist * np.sin(goal_angle)], dtype=np.float32)
                occ_map, has_map_flag, map_type = sample_static_map(goal, rng)
        else:
            occ_map      = empty_map()
            has_map_flag = 0.0
            map_type     = "none"
            if scenario == "at_goal":
                goal_dist  = float(rng.uniform(0.0, 0.3))
                goal_angle = float(rng.uniform(-np.pi, np.pi))
                goal = np.array([goal_dist * np.cos(goal_angle),
                                 goal_dist * np.sin(goal_angle)], dtype=np.float32)
            else:
                goal_dist  = float(sample_goal_distance(rng))
                force_behind = (scenario != "easy") and (rng.random() < 0.10)
                if force_behind:
                    goal_angle = float(rng.uniform(np.pi / 2, 3 * np.pi / 2))
                    if goal_angle > np.pi:
                        goal_angle -= 2 * np.pi
                else:
                    goal_angle = float(rng.uniform(-np.pi, np.pi))
                goal = np.array([goal_dist * np.cos(goal_angle),
                                 goal_dist * np.sin(goal_angle)], dtype=np.float32)
                goal = np.clip(goal, pos_min, pos_max)

        occ_arg = occ_map if use_map else None

        # ── Scenario ──────────────────────────────────────────────────────────
        if scenario == "at_goal":
            v0 = float(rng.uniform(0.0, 0.3))
            w0 = float(rng.uniform(-0.5, 0.5))
            n_obs = int(rng.integers(0, 5))
            for _ in range(n_obs):
                pos = sample_random_obstacle(obstacles, rng, occ_arg)
                if pos is not None and np.linalg.norm(pos - ROBOT_ORIGIN) >= 1.5:
                    obs = make_obstacle(pos, rng, toward_robot_prob=0.0,
                                        force_static=(rng.random() < 0.7))
                    obstacles.append(obs)
            threat_type = "at_goal"

        else:
            v0 = sample_initial_velocity(rng)
            w0 = float(rng.uniform(-W0_MAX, W0_MAX))
            force_static_field = (rng.random() < 0.10)

            if scenario == "hard":
                threat_type = pick_threat_for_map(map_type if use_map else "none", rng)

                if threat_type in ("frontal", "mixed"):
                    for _ in range(int(rng.integers(1, 3))):
                        pos = sample_path_obstacle_zone(goal, LATERAL_SPREAD_HARD,
                                                   obstacles, rng, occ_arg,
                                                   map_blocks_pos)
                        if pos is not None:
                            obstacles.append(make_obstacle(pos, rng,
                                toward_robot_prob=0.5,
                                force_static=force_static_field))

                if threat_type in ("side", "mixed"):
                    for _ in range(int(rng.integers(1, 3))):
                        obs = sample_side_obstacle_timed(goal, obstacles, rng,
                                                          occ_arg, map_blocks_pos)
                        if obs is not None:
                            obstacles.append([obs[0], obs[1], 0.0, 0.0]
                                              if force_static_field else obs)

                if threat_type in ("rear", "mixed"):
                    pos = sample_rear_obstacle(goal, obstacles, rng, occ_arg)
                    if pos is not None:
                        obs = ([pos[0], pos[1], 0.0, 0.0] if force_static_field
                               else make_rear_obstacle(pos, goal, rng))
                        obstacles.append(obs)

                if threat_type == "pincer":
                    for _ in range(2):
                        obs = sample_side_obstacle_timed(goal, obstacles, rng,
                                                          occ_arg, map_blocks_pos)
                        if obs is not None:
                            obstacles.append([obs[0], obs[1], 0.0, 0.0]
                                              if force_static_field else obs)

                if threat_type == "group":
                    for _ in range(int(rng.integers(1, 3))):
                        members = sample_pedestrian_group_v2(goal, obstacles, rng,
                                                              occ_arg, map_blocks_pos,
                                                              group_size_range=(2, 4))
                        if force_static_field:
                            members = [[m[0], m[1], 0.0, 0.0] for m in members]
                        obstacles.extend(members)

                if threat_type == "oncoming":
                    for _ in range(int(rng.integers(1, 3))):
                        obs = sample_oncoming_encounter(goal, obstacles, rng, occ_arg,
                                                        map_blocks_pos)
                        if obs is not None:
                            obstacles.append([obs[0], obs[1], 0.0, 0.0]
                                            if force_static_field else obs)

                if threat_type == "overtake":
                    obs = sample_overtake_encounter(goal, obstacles, rng, occ_arg,
                                                    map_blocks_pos)
                    if obs is not None and not force_static_field:
                        obstacles.append(obs)

                if threat_type == "standing_group":
                    obstacles.extend(sample_standing_group(goal, obstacles, rng, occ_arg,
                                                        map_blocks_pos))

                if threat_type == "near_goal":
                    for _ in range(int(rng.integers(1, 3))):
                        pos = sample_near_goal_obstacle(goal, obstacles, rng, occ_arg)
                        if pos is not None:
                            is_static = rng.random() < 0.6
                            obstacles.append(make_obstacle(pos, rng,
                                toward_robot_prob=0.0,
                                force_static=is_static or force_static_field))
                    for _ in range(int(rng.integers(0, 3))):
                        pos = sample_path_obstacle_zone(goal, LATERAL_SPREAD_HARD,
                                                   obstacles, rng, occ_arg,
                                                   map_blocks_pos)
                        if pos is not None:
                            obstacles.append(make_obstacle(pos, rng,
                                toward_robot_prob=0.2,
                                force_static=force_static_field))

                target = sample_obstacle_count("hard", rng)
                for _ in range(max(0, target - len(obstacles))):
                    pos = sample_random_obstacle(obstacles, rng, occ_arg)
                    if pos is not None:
                        obstacles.append(make_obstacle(pos, rng,
                            toward_robot_prob=0.0,
                            force_static=force_static_field))

                if force_static_field:
                    # Preserve the structural label (pincer/group/...) — the
                    # geometry is still in the obstacle array, only frozen. A bare
                    # "static_field" would discard per-scenario breakdowns for eval.
                    threat_type = f"static_field:{threat_type}"

            elif scenario == "medium":
                if rng.random() < 0.3:
                    members = sample_pedestrian_group(goal, obstacles, rng, occ_arg,
                                                      group_size_range=(2, 3))
                    if force_static_field:
                        members = [[m[0], m[1], 0.0, 0.0] for m in members]
                    obstacles.extend(members)

                for _ in range(int(rng.integers(1, 3))):
                    pos = sample_path_obstacle_zone(goal, LATERAL_SPREAD_MEDIUM,
                                               obstacles, rng, occ_arg,
                                               map_blocks_pos)
                    if pos is not None:
                        obstacles.append(make_obstacle(pos, rng,
                            toward_robot_prob=0.2,
                            force_static=force_static_field))

                target = sample_obstacle_count("medium", rng)
                for _ in range(max(0, target - len(obstacles))):
                    pos = sample_random_obstacle(obstacles, rng, occ_arg)
                    if pos is not None:
                        obstacles.append(make_obstacle(pos, rng,
                            toward_robot_prob=0.0,
                            force_static=force_static_field))

                threat_type = "static_field" if force_static_field else "medium"

            else:  # easy
                n_obs = sample_obstacle_count("easy", rng)
                for _ in range(n_obs):
                    pos = sample_random_obstacle(obstacles, rng, occ_arg)
                    if pos is not None:
                        obs = make_obstacle(pos, rng, toward_robot_prob=0.0,
                                            force_static=force_static_field)
                        if not force_static_field:
                            obs[2] *= 0.3
                            obs[3] *= 0.3
                        obstacles.append(obs)
                threat_type = "easy"

        # ── Final obstacle filter against static map ───────────────────────────
        if has_map_flag > 0.5 and len(obstacles) > 0:
            obstacles = [obs for obs in obstacles
                         if not map_blocks_pos(occ_map, obs[:2], clearance_m=0.3)]

        obstacles = sanitize_obstacles(obstacles, occ_map if has_map_flag > 0.5 else None, rng)

        # Cap at MAX_AGENTS (closest to robot): the dataloader keeps only the
        # k=MAX_AGENTS closest peds, so anything beyond is generated then dropped.
        if len(obstacles) > MAX_AGENTS:
            d = [float(np.linalg.norm(np.asarray(o[:2]) - ROBOT_ORIGIN)) for o in obstacles]
            obstacles = [obstacles[i] for i in np.argsort(d)[:MAX_AGENTS]]

        obstacles_arr = (np.array(obstacles, dtype=np.float32)
                         if obstacles else np.zeros((0, 4), dtype=np.float32))

        # [FEASIBILITY] Reject scenes pedestrians made unsolvable (pincer arms
        # < MIN_GAP apart, standing group blocking a passage, frozen field).
        # Only place that sees combined static+dynamic geometry; keeps ground
        # truth that would breach the R_c contact constraint out of training.
        if goal is not None and len(obstacles) > 0:
            if not robot_can_reach_goal(occ_map if has_map_flag > 0.5 else None,
                                        goal, obstacles):
                continue

        is_nontrivial = use_map and scenario != "at_goal" and not path_is_clear_proc(occ_map, goal)

        if (saved > 500 and use_map and scenario != "at_goal"
                and not is_nontrivial and map_with_path_count > 0
                and nontrivial_path_count / map_with_path_count
                    < MIN_NON_TRIVIAL_FRAC):
            continue

        np.savez(
            f"{save_dir}/sample_{k}.npz",
            start_state   = np.array([v0, w0], dtype=np.float32),
            goal          = goal.astype(np.float32),
            obstacles     = obstacles_arr,
            threat_type   = np.array(threat_type),
            occupancy_map = occ_map,
            has_map       = np.array(has_map_flag, dtype=np.float32),
            map_type      = np.array(map_type),
        )

        saved += 1

        difficulty_counts[scenario] += 1
        threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        map_type_counts[map_type] = map_type_counts.get(map_type, 0) + 1
        if use_map and scenario != "at_goal":
            map_with_path_count += 1
            if is_nontrivial:
                nontrivial_path_count += 1

    nontrivial_frac = (nontrivial_path_count / map_with_path_count
                       if map_with_path_count > 0 else 0.0)
    print(f"\nDataset generation complete.")
    print(f"Difficulty       : {difficulty_counts}")
    print(f"Threats          : {threat_counts}")
    print(f"Map types        : {map_type_counts}")
    print(f"Non-trivial paths: {nontrivial_path_count}/{map_with_path_count} "
          f"({nontrivial_frac:.1%}) of map+goal scenarios have wall-blocked straight line")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate hard custom-map scenes for diffusion planning."
    )
    parser.add_argument("--save_dir",   default="customMAPS",
                        help="Destination for sample_*.npz files")
    parser.add_argument("--n_samples",  type=int, default=50000)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    generate(save_dir=args.save_dir, n_samples=args.n_samples, seed=args.seed)