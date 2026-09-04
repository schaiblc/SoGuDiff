"""
Scene Generator for Diffusion Planning — Real Map Version
==========================================================
Replaces procedural map generation with real InteriorGS occupancy crops
produced by generate_cropped_scenes.py.

Key change: map is sampled FIRST, then goal is placed on free space within
that map. Obstacle placement is then filtered against the real map geometry.

Everything else (difficulty tiers, obstacle types, velocities, threat types)
is unchanged from the Coverage-Improved version.

Usage:
    python generate_scenes.py  --maps_dir  data/maps/interiorgs  --save_dir  data/scenes/interiorgs  --n_samples 100000
    python generate_scenes.py  --maps_dir  data/maps/matterport  --save_dir  data/scenes/matterport  --n_samples 100000
    python generate_scenes.py  --maps_dir  data/maps/tartanground  --save_dir  data/scenes/tartanground  --n_samples 200000

"""

import numpy as np
import os
import argparse
from pathlib import Path

# =========================
# Config
# =========================
# World bounds for obstacle/goal placement — intentionally wider than local map
# (MAP_EXTENT_M=10 → ±5 m covered by costmap; goals/obstacles at ±5..7.5 m are
#  in unobserved free space. The expert projects the goal to the crop border
#  along the true geodesic; there is no Euclidean fallback.)
pos_min, pos_max = -7.5, 7.5

# --- Agent radii: SINGLE SOURCE OF TRUTH, must match the expert cost ----------
# The expert plans, and the social cost is defined, with r_robot = r_human = 0.25
# m and contact at R_c = 0.50 m (≈ Hall intimate, 0.46 m).  The scene generator
# MUST size gaps and keep-outs to the same robot, or it rejects goals the expert
# can reach and carves gaps the expert cannot use.
ROBOT_R   = 0.25                 # robot disc radius (== HUMAN_R in the cost)
HUMAN_R   = 0.25
R_C       = ROBOT_R + HUMAN_R    # 0.50 m contact (hard constraint in the cost)
MIN_GAP   = 2 * R_C              # 1.00 m minimum passable gap for the robot disc
NAV_RADIUS = ROBOT_R             # C-space erosion radius (was 0.30 — inconsistent)

# Max AGENTS considered by the model dataloader (k-closest).  We never place more
# than this: everything beyond the closest MAX_AGENTS is invisible to the policy,
# so generating denser scenes only wastes compute and biases the map channel.
MAX_AGENTS   = 10
max_obstacles = MAX_AGENTS       # back-compat alias used by tier helpers

ROBOT_ORIGIN = np.array([0.0, 0.0])
OBSTACLE_MIN_SEPARATION = 0.50
MAX_PLACEMENT_ATTEMPTS = 50

HORIZON = 32
DT = 0.05
LATERAL_SPREAD_HARD = 0.4
LATERAL_SPREAD_MEDIUM = 1.2
V_MAX = 1.0
W0_MAX  = np.pi  #max Jackal limit

# Grid spec — must match generate_cropped_scenes.py
MAP_SIZE     = 50
MAP_EXTENT_M = 10
MAP_RES      = MAP_EXTENT_M / MAP_SIZE   # 0.20 m/cell
FREE_VAL     = 255
OBS_VAL      = 0

# Probability that a scene uses a real map (vs. pure open space)
MAP_PRESENCE_PROB    = 0.85  # high map presence — real maps are the primary training signal
NONTRIVIAL_GOAL_BIAS = 0.65  # fraction of real-map goals where the geodesic detours around walls
MIN_NON_TRIVIAL_FRAC = 0.40  # post-hoc filter to ensure enough scenes have non-trivial (detour) paths

# --- Pedestrian kinematics: grounded in the same literature as the cost --------
# v_bar = 1.34 m/s is the Weidmann '93 / Helbing & Molnar '95 free-flow walking
# speed the yield ladder (tau_a = D(s)/v_bar) is calibrated to.  Using 1.1 here
# would make the g_cut capsule ~18% shorter in the data than the cost assumes.
PED_SPEED_MEAN  = 1.34   # m/s — free-flow preferred walking speed (matches v_bar)
PED_SPEED_STD   = 0.26   # Weidmann inter-individual s.d.
PED_SPEED_MIN   = 0.5
PED_SPEED_MAX   = 1.8    # fast walk; running is out of distribution on purpose

# --- Spawn safety: the DYNAMIC check owns it; no static keep-out disc ----------
# We forbid immediate collisions and imminent-unavoidable collisions (a ped whose
# constant-velocity path enters R_c within T_SPAWN_SAFE), so the robot is never
# taught that it may be IN a violation at t=0.  But we DO allow spawns inside the
# personal / social zone that are closing — those are the close-encounter starts
# the prox/yield/cut axes exist to shape, and forbidding them (as the old static
# 1.0 m disc did) starved exactly that regime.  Floor keeps a ped from spawning
# in intimate space (which the robot cannot recover from within the plan).
SPAWN_MIN_DIST  = R_C + 0.30            # 0.80 m — never start inside intimate space
T_SPAWN_SAFE    = 0.6                   # s  — imminent-collision guard horizon
T_PLAN          = HORIZON * DT          # 1.6 s
ENCOUNTER_T_LO  = 0.35   # s  — stage interactions inside the plan window
ENCOUNTER_T_HI  = 1.6
ONCOMING_AIM_STD   = np.deg2rad(8)      # tight aim → real mutual-aim encounters
GROUP_MEMBER_VEL_STD = 0.04             # m/s per axis — keeps pairs coherent (Δv well under DV_MAX)
GROUP_ABREAST_SPACING = (0.6, 1.1)      # m — empirical dyad/triad spacing (inside D_per)

# =========================
# Real map loading
# =========================

def load_real_maps(maps_dir: str):
    """
    Load the pre-generated crop dataset.
    Returns (maps, rng_indices) where maps is uint8 [N, 50, 50].
    Memory-mapped so only accessed pages are loaded.
    """
    root = Path(maps_dir)
    maps = np.load(root / "maps.npy", mmap_mode="r")   # uint8 [N, 50, 50]
    print(f"Loaded {len(maps)} real map crops from {root}")
    return maps


def empty_map_float():
    """All-free map in the float32 convention used downstream (0=free, 1=obs)."""
    return np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)


def real_map_to_float(crop_uint8: np.ndarray) -> np.ndarray:
    """
    Convert uint8 crop (FREE=255, OBS=0) to float32 (FREE=0.0, OBS=1.0)
    to match the convention in the existing scene generator.
    """
    return (crop_uint8 != FREE_VAL).astype(np.float32)


# =========================
# Grid helpers (unchanged)
# =========================

def world_to_grid(x, y):
    col = int(round(x / MAP_RES + MAP_SIZE / 2))
    row = int(round(y / MAP_RES + MAP_SIZE / 2))
    col = np.clip(col, 0, MAP_SIZE - 1)
    row = np.clip(row, 0, MAP_SIZE - 1)
    return row, col


def grid_to_world(row, col):
    x = (col - MAP_SIZE / 2 + 0.5) * MAP_RES
    y = (row - MAP_SIZE / 2 + 0.5) * MAP_RES
    return x, y


def map_blocks_obstacle(occ_float, pos, clearance_m=0.3):
    """Check if a dynamic-obstacle spawn point conflicts with the static map."""
    r, c = world_to_grid(pos[0], pos[1])
    rad = int(np.ceil(clearance_m / MAP_RES))
    r_lo, r_hi = max(0, r - rad), min(MAP_SIZE, r + rad + 1)
    c_lo, c_hi = max(0, c - rad), min(MAP_SIZE, c + rad + 1)
    return occ_float[r_lo:r_hi, c_lo:c_hi].any()


def robot_can_reach_goal(occ_float, goal, obstacles):
    """[FEASIBILITY] After pedestrians are placed, does a robot-radius-wide
    corridor from the start to the goal still exist?

    The static reachability check in sample_goal_on_map runs BEFORE pedestrians
    exist, so it cannot see a doorway a standing group blocks, a pincer whose two
    arms sit < MIN_GAP apart, or a frozen field. We rasterize each pedestrian's
    t=0 disc (radius R_C, so the robot center needs a full body-radius of
    clearance to pass without breaching contact) into a copy of the occupancy,
    erode by the robot radius, and test that the robot's 8-connected free
    component reaches the goal cell (or the crop border, for out-of-crop goals a
    global planner would route through). Returns True if solvable.

    This is where MIN_GAP finally bites: two obstacles closer than 2*R_C leave no
    eroded free cell between them, so the component is severed and the scene is
    rejected."""
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
    # Out-of-crop goal: solvable iff the robot can still reach the crop border,
    # where a global planner supplies the continuation.
    border = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    return bool((comp & border).any())


def paths_are_clear_batch(occ_float: np.ndarray, goals: np.ndarray,
                          n_steps: int = 40) -> np.ndarray:
    """
    Vectorized straight-line visibility check from (0, 0) to each goal.
    Returns bool array (N,) — True = line-of-sight is clear, False = wall blocks it.
    A blocked path means the robot *must* navigate around static obstacles to reach
    the goal, which is exactly the behavior we want the map channel to teach.
    """
    N = len(goals)
    if N == 0:
        return np.empty(0, dtype=bool)
    t      = np.linspace(0.0, 1.0, n_steps)          # (S,)
    points = goals[:, None, :] * t[None, :, None]     # (N, S, 2)  — origin at (0,0)
    cols = np.clip(
        np.round(points[..., 0] / MAP_RES + MAP_SIZE / 2).astype(int),
        0, MAP_SIZE - 1)
    rows = np.clip(
        np.round(points[..., 1] / MAP_RES + MAP_SIZE / 2).astype(int),
        0, MAP_SIZE - 1)
    blocked = occ_float[rows, cols] > 0.5             # (N, S)
    return ~blocked.any(axis=1)                        # True = clear


def path_is_clear(occ_float: np.ndarray, goal: np.ndarray, n_steps: int = 40) -> bool:
    """Scalar wrapper: True if straight line from (0, 0) to goal is obstacle-free."""
    return bool(paths_are_clear_batch(occ_float, goal[None])[0])


# =========================
# Real map sampling  (replaces sample_static_map)
# =========================

def sample_real_map(maps: np.ndarray, rng: np.random.Generator):
    """
    Pick a random crop from the pre-generated dataset.
    Returns (occ_float32, map_idx) where occ_float32 follows the
    existing convention: 0.0 = free, 1.0 = obstacle.
    """
    idx = int(rng.integers(0, len(maps)))
    crop = maps[idx]                        # uint8 [50, 50]
    occ  = real_map_to_float(crop)          # float32 [50, 50]
    return occ, idx


def sample_goal_on_map(occ_float, rng, min_dist=0.3, max_dist=7.0,
                       goal_clearance_m=0.4, max_attempts=100):
    """
    Sample a goal that is either:
      - Inside the map extent AND in a free cell reachable from the robot, OR
      - Outside the map extent but within [pos_min, pos_max] world bounds
        (treated as free space since map doesn't cover it)

    Connectivity check: only cells in the same 8-connected free component as
    the robot (MAP_SIZE//2, MAP_SIZE//2) are eligible. This prevents wasting
    scenes on goals in walled-off rooms that would be rejected in Stage 3.
    """
    goal_rad_cells = max(1, int(np.ceil(goal_clearance_m / MAP_RES)))

    from scipy.ndimage import binary_erosion, label as _label
    free_mask = (occ_float == 0.0)

    # --- C-space: erode free space by robot radius so gaps narrower than the
    #     Jackal body are treated as impassable before connectivity is checked ---
    robot_r = robot_c = MAP_SIZE // 2   # robot is at world (0,0) = grid center
    _nav_rad = max(1, int(np.ceil((NAV_RADIUS + 0.2) / MAP_RES)))  # C-space erosion at robot radius
    _nav_struct = np.ones((2 * _nav_rad + 1, 2 * _nav_rad + 1), dtype=bool)
    free_cspace = binary_erosion(free_mask, structure=_nav_struct, border_value=0)

    # --- Reachability: restrict to robot's connected component (8-connected) ---
    struct8 = np.ones((3, 3), dtype=int)
    if free_cspace[robot_r, robot_c]:
        labeled, _ = _label(free_cspace, structure=struct8)
        robot_label = int(labeled[robot_r, robot_c])
        reachable = labeled == robot_label
    else:
        reachable = free_mask   # robot in/near wall — keep all as best-effort

    # --- Erode the reachable free space for goal-clearance ---
    struct = np.ones((2 * goal_rad_cells + 1, 2 * goal_rad_cells + 1), dtype=bool)
    free_eroded = binary_erosion(reachable, structure=struct, border_value=0)

    free_cells = np.argwhere(free_eroded)
    if len(free_cells) == 0:
        cell_worlds = np.zeros((0, 2), dtype=np.float32)
    else:
        cell_worlds = np.array([grid_to_world(r, c) for r, c in free_cells],
                               dtype=np.float32)

    # Distance filter on in-map free cells
    if len(cell_worlds) == 0:
        valid_worlds = np.zeros((0, 2), dtype=np.float32)
    else:
        dists = np.linalg.norm(cell_worlds, axis=1)

        valid = (dists >= min_dist) & (dists <= max_dist)
        valid_worlds = cell_worlds[valid]

    # Out-of-map candidates — only sensible if the robot can actually exit the
    # map. Check that the robot's reachable component touches the grid boundary.
    # (Stage 3 catches residual cases via start_unreachable, but this avoids
    #  generating and saving scenes that will be thrown away.)
    boundary = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = True
    robot_can_exit = bool((reachable & boundary).any())

    out_of_map_candidates = []
    if robot_can_exit:
        half_extent = MAP_EXTENT_M / 2
        for _ in range(max_attempts):
            gx = rng.uniform(pos_min, pos_max)
            gy = rng.uniform(pos_min, pos_max)
            if abs(gx) > half_extent or abs(gy) > half_extent:
                d = np.sqrt(gx**2 + gy**2)
                if min_dist <= d <= max_dist:
                    out_of_map_candidates.append([gx, gy])

    out_of_map_candidates = np.array(out_of_map_candidates, dtype=np.float32)

    # Bias in-map candidates toward goals that require navigating around walls.
    # Without this, most sampled goals are reachable via a straight line and the
    # model learns to ignore the occupancy map channel entirely.
    if len(valid_worlds) > 0:
        clear_mask  = paths_are_clear_batch(occ_float, valid_worlds)
        nontrivial  = valid_worlds[~clear_mask]
        trivial_wds = valid_worlds[ clear_mask]
        if len(nontrivial) > 0 and len(trivial_wds) > 0:
            valid_worlds = nontrivial if rng.random() < NONTRIVIAL_GOAL_BIAS else trivial_wds
        elif len(nontrivial) > 0:
            valid_worlds = nontrivial
        # else: all trivial (very open map) — keep valid_worlds unchanged

    # Pool all valid candidates
    if len(valid_worlds) > 0 and len(out_of_map_candidates) > 0:
        all_candidates = np.vstack([valid_worlds, out_of_map_candidates])
    elif len(valid_worlds) > 0:
        all_candidates = valid_worlds
    elif len(out_of_map_candidates) > 0:
        all_candidates = out_of_map_candidates
    else:
        return None

    idx = int(rng.integers(0, len(all_candidates)))
    gx, gy = all_candidates[idx]
    return np.array([gx, gy], dtype=np.float32)


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
    """[T5] Spawn admissibility. Two conditions, both grounded in the cost:
      (1) the ped starts outside intimate space: ||pos|| >= SPAWN_MIN_DIST, so
          the robot is never IN a violation at t=0 and never taught recovery
          from an already-collided state;
      (2) the ped's constant-velocity line does not reach contact (R_c) with the
          robot's START within the guard horizon t_safe — no imminent, t=0-
          unavoidable collision.
    Crucially this ALLOWS spawns inside the personal/social zone that are
    closing: those are the close-encounter starts the prox/yield/cut axes exist
    to shape.  The old static 1.0 m keep-out forbade them and starved that
    regime."""
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

# =========================
# Tiered sampling helpers (unchanged)
# =========================


def sample_goal_distance(rng: np.random.Generator):
    r = rng.random()
    if r < 0.15:
        return rng.uniform(0.3, 1.0)
    elif r < 0.30:
        return rng.uniform(1.0, 2.0)
    elif r < 0.80:
        return rng.uniform(2.0, 5.0)
    else:
        return rng.uniform(5.0, 7.0)


def sample_initial_velocity(rng: np.random.Generator):
    r = rng.random()
    if r < 0.15:
        return 0.0
    elif r < 0.25:
        return V_MAX
    else:
        return float(rng.uniform(0.0, V_MAX))


def sample_obstacle_count(difficulty: str, rng: np.random.Generator):
    if difficulty == "easy":
        return int(rng.integers(0, max_obstacles + 1))
    r = rng.random()
    if r < 0.30:
        return int(rng.integers(0, 4))
    elif r < 0.80:
        return int(rng.integers(3, 8))
    else:
        return int(rng.integers(7, max_obstacles + 1))


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

# =========================
# Obstacle helpers (unchanged from original)
# =========================

def is_valid_position(pos, placed_obstacles, occ_float=None):
    if np.linalg.norm(pos - ROBOT_ORIGIN) < SPAWN_MIN_DIST:
        return False
    for prev in placed_obstacles:
        if np.linalg.norm(pos - np.array(prev[:2])) < OBSTACLE_MIN_SEPARATION:
            return False
    if not (pos_min <= pos[0] <= pos_max and pos_min <= pos[1] <= pos_max):
        return False
    # Also reject if position is inside a wall in the real map
    if occ_float is not None and map_blocks_obstacle(occ_float, pos, clearance_m=0.3):
        return False
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


def sample_rear_obstacle(goal, placed_obstacles, rng, occ_float=None):
    path_vec = goal - ROBOT_ORIGIN
    path_dir = path_vec / (np.linalg.norm(path_vec) + 1e-6)
    perp_dir = np.array([-path_dir[1], path_dir[0]])
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        t = rng.uniform(0.5, 2.5)
        lateral = rng.normal(0, 0.6)
        pos = ROBOT_ORIGIN - t * path_dir + lateral * perp_dir
        if is_valid_position(pos, placed_obstacles, occ_float):
            return pos
    return None


def sample_side_obstacle(goal, placed_obstacles, rng, occ_float=None):
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
        if is_valid_position(pos, placed_obstacles, occ_float):
            return pos, intercept
    return None, None


def make_rear_obstacle(pos, goal, rng):
    path_vec = goal - ROBOT_ORIGIN
    path_dir = path_vec / (np.linalg.norm(path_vec) + 1e-6)
    aim_point = ROBOT_ORIGIN + path_dir * rng.uniform(0.5, 2.0)
    to_aim = aim_point - pos
    to_aim /= (np.linalg.norm(to_aim) + 1e-6)
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
                         occ_float=None, t_fraction_range=(0.1, 0.9)):
    path_vec = goal - ROBOT_ORIGIN
    path_len = np.linalg.norm(path_vec)
    if path_len < 1e-3:
        return None
    path_dir = path_vec / path_len
    perp_dir = np.array([-path_dir[1], path_dir[0]])
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        t = rng.uniform(*t_fraction_range)
        along = t * path_len
        lateral = rng.normal(0, lateral_spread)
        pos = ROBOT_ORIGIN + along * path_dir + lateral * perp_dir
        if is_valid_position(pos, placed_obstacles, occ_float):
            return pos
    return None


def sample_random_obstacle(placed_obstacles, rng, occ_float=None):
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        r = rng.uniform(0.5, 6.0)
        theta = rng.uniform(0, 2 * np.pi)
        pos = np.array([r * np.cos(theta), r * np.sin(theta)])
        if is_valid_position(pos, placed_obstacles, occ_float):
            return pos
    return None


def sample_near_goal_obstacle(goal, placed_obstacles, rng, occ_float=None):
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        r = rng.uniform(0.7, 1.2)
        theta = rng.uniform(0, 2 * np.pi)
        pos = goal + np.array([r * np.cos(theta), r * np.sin(theta)])
        if is_valid_position(pos, placed_obstacles, occ_float):
            return pos
    return None


def sample_pedestrian_group(goal, placed_obstacles, rng, occ_float=None,
                            group_size_range=(2, 4)):
    path_vec = goal - ROBOT_ORIGIN
    path_len = np.linalg.norm(path_vec)
    path_dir = path_vec / (path_len + 1e-6)
    perp_dir = np.array([-path_dir[1], path_dir[0]])
    group_size = int(rng.integers(group_size_range[0], group_size_range[1] + 1))
    GROUP_SPREAD = 0.5
    GROUP_MIN_SEPARATION = 0.4
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        t = rng.uniform(0.2, 0.7) * path_len
        lateral = rng.normal(0, LATERAL_SPREAD_HARD)
        anchor = ROBOT_ORIGIN + t * path_dir + lateral * perp_dir
        if not (pos_min <= anchor[0] <= pos_max and pos_min <= anchor[1] <= pos_max):
            continue
        if np.linalg.norm(anchor - ROBOT_ORIGIN) < SPAWN_MIN_DIST + 0.5:
            continue
        if occ_float is not None and map_blocks_obstacle(occ_float, anchor, 0.25):
            continue
        group_speed = rng.uniform(0.3, 1.2)
        perp_bias = rng.choice([-1, 1])
        along_bias = rng.uniform(-0.3, 0.3)
        group_dir = perp_bias * perp_dir + along_bias * path_dir
        group_dir /= (np.linalg.norm(group_dir) + 1e-6)
        members = []
        failed = False
        for _ in range(group_size):
            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                offset = rng.normal(0, GROUP_SPREAD, size=2)
                pos = anchor + offset
                too_close = any(
                    np.linalg.norm(pos - np.array(m[:2])) < GROUP_MIN_SEPARATION
                    for m in members
                )
                if (is_valid_position(pos, placed_obstacles, occ_float)
                        and not too_close
                        and pos_min <= pos[0] <= pos_max
                        and pos_min <= pos[1] <= pos_max):
                    noise = rng.normal(0, 0.15, size=2)
                    vel = group_dir * group_speed + noise
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

def generate(maps_dir: str, save_dir: str, n_samples: int, seed: int = 42):
    os.makedirs(save_dir, exist_ok=True)

    rng = np.random.default_rng(seed)

    # ── Load real maps once ───────────────────────────────────────────────────
    maps = load_real_maps(maps_dir)

    # ── Try to import scipy for goal erosion; fall back gracefully ────────────
    try:
        from scipy.ndimage import binary_erosion as _be
        _has_scipy = True
    except ImportError:
        _has_scipy = False
        print("[warn] scipy not found — goal clearance erosion disabled. "
              "pip install scipy for better goal placement.")

    difficulty_counts    = {"hard": 0, "medium": 0, "easy": 0, "at_goal": 0}
    threat_counts        = {}
    map_use_counts       = {"real": 0, "open": 0}
    nontrivial_goal_count = 0   # real-map goals where straight-line path is wall-blocked
    real_map_goal_count   = 0   # denominator for nontrivial fraction

    saved = 0
    k = 0
    while saved < n_samples:
        k += 1
        if saved % 10000 == 0:
            print(f"  {saved}/{n_samples} ...", flush = True)

        # ── Scenario tier ─────────────────────────────────────────────────────
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
        threat_type = scenario

        # ── Decide map source ─────────────────────────────────────────────────
        map_prob = 0.4 if scenario == "at_goal" else MAP_PRESENCE_PROB
        use_real_map = (rng.random() < map_prob)

        if use_real_map:
            occ_map, map_idx = sample_real_map(maps, rng)
            has_map_flag = 1.0
            map_use_counts["real"] += 1
        else:
            occ_map = empty_map_float()
            has_map_flag = 0.0
            map_idx = -1
            map_use_counts["open"] += 1

        # ── AT_GOAL scenario ──────────────────────────────────────────────────
        if scenario == "at_goal":
            v0 = float(rng.uniform(0.0, 0.3))
            w0 = float(rng.uniform(-0.5, 0.5))
            goal_dist = float(rng.uniform(0.0, 0.3))
            goal_angle = float(rng.uniform(-np.pi, np.pi))
            goal = np.array([goal_dist * np.cos(goal_angle),
                             goal_dist * np.sin(goal_angle)], dtype=np.float32)

            n_obs = int(rng.integers(0, 5))
            for _ in range(n_obs):
                pos = sample_random_obstacle(obstacles, rng,
                                             occ_map if use_real_map else None)
                if pos is not None and np.linalg.norm(pos - ROBOT_ORIGIN) >= 1.5:
                    obs = make_obstacle(pos, rng, toward_robot_prob=0.0,
                                       force_static=(rng.random() < 0.7))
                    obstacles.append(obs)
            threat_type = "at_goal"

        # ── All other scenarios ───────────────────────────────────────────────
        else:
            v0 = sample_initial_velocity(rng)
            w0 = float(rng.uniform(-W0_MAX, W0_MAX))
            force_static_field = (rng.random() < 0.10)

            # ── Goal: if real map, sample from free cells; else use tiered dist ──
            if use_real_map and _has_scipy:
                goal_dist_tier = sample_goal_distance(rng)
                # Try to find a goal at approximately the right distance
                goal = sample_goal_on_map(
                    occ_map, rng,
                    min_dist=max(0.3, goal_dist_tier * 0.5),
                    max_dist=min(7.5, goal_dist_tier * 1.5),
                )
                if goal is None:
                    # Fallback: any free cell reachable
                    goal = sample_goal_on_map(occ_map, rng, min_dist=0.3, max_dist=7.5)
                if goal is None:
                    # Map too cluttered — treat as open
                    occ_map = empty_map_float()
                    has_map_flag = 0.0
                    use_real_map = False
                    map_idx = -1   # discard the cluttered map's index (saved occ is empty)
                    goal_dist  = float(sample_goal_distance(rng))
                    goal_angle = float(rng.uniform(-np.pi, np.pi))
                    goal = np.array([goal_dist * np.cos(goal_angle),
                                     goal_dist * np.sin(goal_angle)], dtype=np.float32)
            else:
                # No real map or no scipy — original tiered distance approach
                goal_dist = float(sample_goal_distance(rng))
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

            occ_arg = occ_map if use_real_map else None

            # ── Obstacle placement ──────────────────────────────────────────────
            if scenario == "hard":
                threat_type = rng.choice(
                    ["oncoming", "overtake", "frontal", "side", "rear", "mixed",
                     "pincer", "group", "standing_group", "near_goal"],
                    p=[0.14, 0.10, 0.10, 0.13, 0.04, 0.10, 0.06, 0.13, 0.08, 0.12]
                )

                if threat_type in ("frontal", "mixed"):
                    for _ in range(int(rng.integers(1, 3))):
                        pos = sample_path_obstacle_zone(goal, LATERAL_SPREAD_HARD,
                                                   obstacles, rng, occ_arg,
                                                   map_blocks_obstacle)
                        if pos is not None:
                            obstacles.append(make_obstacle(
                                pos, rng, toward_robot_prob=0.5,
                                force_static=force_static_field))

                if threat_type in ("side", "mixed"):
                    for _ in range(int(rng.integers(1, 3))):
                        obs = sample_side_obstacle_timed(goal, obstacles, rng,
                                                          occ_arg, map_blocks_obstacle)
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
                                                          occ_arg, map_blocks_obstacle)
                        if obs is not None:
                            obstacles.append([obs[0], obs[1], 0.0, 0.0]
                                              if force_static_field else obs)

                if threat_type == "group":
                    for _ in range(int(rng.integers(1, 3))):
                        members = sample_pedestrian_group_v2(
                            goal, obstacles, rng, occ_arg, map_blocks_obstacle,
                            group_size_range=(2, 4))
                        if force_static_field:
                            members = [[m[0], m[1], 0.0, 0.0] for m in members]
                        obstacles.extend(members)

                if threat_type == "oncoming":
                    for _ in range(int(rng.integers(1, 3))):
                        obs = sample_oncoming_encounter(goal, obstacles, rng, occ_arg,
                                                        map_blocks_obstacle)
                        if obs is not None:
                            obstacles.append([obs[0], obs[1], 0.0, 0.0]
                                            if force_static_field else obs)

                if threat_type == "overtake":
                    obs = sample_overtake_encounter(goal, obstacles, rng, occ_arg,
                                                    map_blocks_obstacle)
                    if obs is not None and not force_static_field:
                        obstacles.append(obs)

                if threat_type == "standing_group":
                    obstacles.extend(sample_standing_group(goal, obstacles, rng, occ_arg,
                                                        map_blocks_obstacle))

                if threat_type == "near_goal":
                    for _ in range(int(rng.integers(1, 3))):
                        pos = sample_near_goal_obstacle(goal, obstacles, rng, occ_arg)
                        if pos is not None:
                            is_static = rng.random() < 0.6
                            obstacles.append(make_obstacle(
                                pos, rng, toward_robot_prob=0.0,
                                force_static=is_static or force_static_field))
                    for _ in range(int(rng.integers(0, 3))):
                        pos = sample_path_obstacle_zone(goal, LATERAL_SPREAD_HARD,
                                                   obstacles, rng, occ_arg,
                                                   map_blocks_obstacle)
                        if pos is not None:
                            obstacles.append(make_obstacle(
                                pos, rng, toward_robot_prob=0.2,
                                force_static=force_static_field))

                # Filler
                target = sample_obstacle_count("hard", rng)
                for _ in range(max(0, target - len(obstacles))):
                    pos = sample_random_obstacle(obstacles, rng, occ_arg)
                    if pos is not None:
                        obstacles.append(make_obstacle(
                            pos, rng, toward_robot_prob=0.0,
                            force_static=force_static_field))

                if force_static_field:
                    # Preserve the structural label (pincer/group/...) — the
                    # geometry is still in the obstacle array, only frozen. A bare
                    # "static_field" would discard per-scenario breakdowns for eval.
                    threat_type = f"static_field:{threat_type}"

            elif scenario == "medium":
                if rng.random() < 0.3:
                    members = sample_pedestrian_group(
                        goal, obstacles, rng, occ_arg, group_size_range=(2, 3))
                    if force_static_field:
                        members = [[m[0], m[1], 0.0, 0.0] for m in members]
                    obstacles.extend(members)

                for _ in range(int(rng.integers(1, 3))):
                    pos = sample_path_obstacle_zone(goal, LATERAL_SPREAD_MEDIUM,
                                               obstacles, rng, occ_arg,
                                               map_blocks_obstacle)
                    if pos is not None:
                        obstacles.append(make_obstacle(
                            pos, rng, toward_robot_prob=0.2,
                            force_static=force_static_field))

                target = sample_obstacle_count("medium", rng)
                for _ in range(max(0, target - len(obstacles))):
                    pos = sample_random_obstacle(obstacles, rng, occ_arg)
                    if pos is not None:
                        obstacles.append(make_obstacle(
                            pos, rng, toward_robot_prob=0.0,
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

        # ── Final obstacle filter against map ─────────────────────────────────
        # (redundant for obstacles sampled with occ_arg, but catches edge cases)
        if has_map_flag > 0.5 and len(obstacles) > 0:
            obstacles = [obs for obs in obstacles
                         if not map_blocks_obstacle(occ_map, obs[:2], clearance_m=0.3)]

        obstacles = sanitize_obstacles(obstacles, occ_map if has_map_flag > 0.5 else None, rng)

        # Hard cap at MAX_AGENTS. The model dataloader keeps only the k=MAX_AGENTS
        # closest pedestrians, so any beyond that are generated, saved, and then
        # silently dropped — wasted compute, and the expert would plan against
        # agents the policy never sees. Keep the closest MAX_AGENTS to the robot
        # start so scene and policy see the same set. (Group/threat scenarios add
        # peds on top of the tier count, so scenes can exceed the cap.)
        if len(obstacles) > MAX_AGENTS:
            d = [float(np.linalg.norm(np.asarray(o[:2]) - ROBOT_ORIGIN)) for o in obstacles]
            obstacles = [obstacles[i] for i in np.argsort(d)[:MAX_AGENTS]]

        obstacles_arr = (np.array(obstacles, dtype=np.float32)
                         if obstacles else np.zeros((0, 4), dtype=np.float32))

        # [FEASIBILITY] Reject scenes the pedestrians made unsolvable (pincer arms
        # < MIN_GAP apart, standing group blocking a doorway, frozen field). The
        # static reachability check ran before peds existed; this is the only
        # place that sees the combined static+dynamic geometry, and it is what
        # keeps infeasible ground truth — trajectories that can only reach goal by
        # breaching the R_c contact constraint — out of the training set.
        if goal is not None and len(obstacles) > 0:
            if not robot_can_reach_goal(occ_map if has_map_flag > 0.5 else None,
                                        goal, obstacles):
                continue

        # Track what fraction of real-map goals actually require navigating around walls
        is_nontrivial = (use_real_map and scenario != "at_goal"
                         and goal is not None
                         and not path_is_clear(occ_map, goal))

        if (saved > 500 and use_real_map and scenario != "at_goal"
                and not is_nontrivial and real_map_goal_count > 0
                and nontrivial_goal_count / real_map_goal_count
                    < MIN_NON_TRIVIAL_FRAC):
            continue

        np.savez(
            f"{save_dir}/sample_{k}.npz",
            start_state  = np.array([v0, w0], dtype=np.float32),
            goal         = goal.astype(np.float32),
            obstacles    = obstacles_arr,
            threat_type  = np.array(threat_type),
            occupancy_map= occ_map,                              # float32 [50,50]
            has_map      = np.array(has_map_flag, dtype=np.float32),
            map_idx      = np.array(map_idx, dtype=np.int32),   # index into maps.npy
        )

        saved += 1

        difficulty_counts[scenario] += 1
        threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        if use_real_map and scenario != "at_goal" and goal is not None:
            real_map_goal_count += 1
            if is_nontrivial:
                nontrivial_goal_count += 1

    nontrivial_frac = (nontrivial_goal_count / real_map_goal_count
                       if real_map_goal_count > 0 else 0.0)
    print(f"\nDataset generation complete.")
    print(f"Difficulty        : {difficulty_counts}")
    print(f"Threats           : {threat_counts}")
    print(f"Map source        : {map_use_counts}")
    print(f"Non-trivial goals : {nontrivial_goal_count}/{real_map_goal_count} "
          f"({nontrivial_frac:.1%}) of real-map goals require navigating around walls")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate social navigation scenes using real InteriorGS map crops."
    )
    parser.add_argument("--maps_dir",  required=True,
                        help="Output dir from generate_cropped_scenes.py "
                             "(contains maps.npy)")
    parser.add_argument("--save_dir",  required=True,
                        help="Destination for sample_*.npz files")
    parser.add_argument("--n_samples", type=int, default=100000)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    generate(
        maps_dir  = args.maps_dir,
        save_dir  = args.save_dir,
        n_samples = args.n_samples,
        seed      = args.seed,
    )