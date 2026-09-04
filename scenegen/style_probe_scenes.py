"""Library of hand-designed style-probe scenes.

Defines the canonical scene categories used by the style, composition and
online-expert experiments, and can also be run directly to emit them:

    python scenegen/style_probe_scenes.py --out_dir data/scenes/style_probe

build_eval_set_500.py imports this module to assemble the larger evaluation
set; see that file for how the categories are sampled and filtered.



Generate 100 canonical social-navigation test scenes as .npz files:
the original 50 open-space scenes (Categories A-H, has_map = 0), PLUS 50
scenes designed specifically to make style-axis differences (prox, pass,
yield, group) as clear and unambiguous as possible (Categories Y, GS, R,
FW, COR).

Design intent for the new 50 scenes (Y, GS, R, FW, COR)
---------------------------------------------------------
An earlier version of this file used simple static-map features (walls,
corners, corridors, pillars, doorways) for the second half of the test
set. That set is replaced here because (a) several of those scenes probed
map-vs-style conflicts rather than cleanly isolating a single style axis,
and (b) a few of the narrow-corridor scenes could make the goal
unreachable once the corridor walls were inflated by the agent radius for
collision checking, which meant the "test" wasn't actually solvable. The
new scenes below are built with two rules:

  1. EVERY scene must always be solvable / the goal must always be
     reachable. Scenes that use a static map (only Category COR) use a
     corridor half-width chosen so that, even after inflating the walls
     by AGENT_RADIUS on a 0.2 m grid, a passable channel of at least
     ~1.0 m remains down the centerline (verified below with the same
     BFS reachability check used previously). No scene in this file
     relies on a fully-enclosed room, a single non-detourable gap, or a
     corridor pinch that could close off entirely.
  2. Each category is built to isolate ONE style axis as cleanly as
     possible, with minimal confounds from other axes:
       Y   (yield)  - conflict timing / who-goes-first scenarios where
                       yield+ vs yield- should visibly diverge (wait vs.
                       assert) while pass/prox/group stay uninformative.
       GS  (group)  - stationary or co-moving clusters of 2-4 pedestrians
                       where group+ should route fully around the
                       cluster while group- may cut between members;
                       includes borderline gap sizes.
       R   (prox/pass, rear approach) - the robot catches up to and must
                       overtake pedestrians ahead of it (not head-on),
                       including drift, sudden stops, and turns, so
                       pass-side and following distance are isolated.
       FW  (pass/yield, intersection) - multi-directional crossings at a
                       shared conflict point, testing pass-side choice
                       and yield timing under 2-4 simultaneous agents.
       COR (prox, corridor/counterflow) - a static corridor (guaranteed
                       passable, see rule 1) with oncoming singles,
                       pairs, and triples, isolating how tightly the
                       robot hugs one side under lateral confinement.

Coordinate & map conventions (must match crowd_sim.py's load_npz_scenario)
----------------------------------------------------------------------------
  Robot always starts at (0, 0), heading = 0 (facing +x), v0 = 0.6, w0 = 0.
  Obstacles: [x, y, vx, vy]  — position + constant velocity, ego frame.
  Goal:      [gx, gy]  (kept along +x, by convention,
             so the goal itself never introduces a left/right bias — any
             asymmetry in the new scenes comes only from the pedestrian
             configuration or, for Category COR, the corridor map).
  +y is the robot's LEFT, -y is the robot's RIGHT.
  occupancy_map: (50, 50) float32, map_extent = 10 m (0.2 m / cell), ROW
             indexes y and COLUMN indexes x, ego frame == world frame at
             t=0 (robot starts at the map center, heading 0). This exactly
             mirrors SceneMapGenerator in
             crowd_sim/envs/utils/map_scene_generator.py, whose drawing
             primitives (_add_segment / _add_rect) are ported below in
             deterministic form (fixed parameters instead of an RNG) so
             the corridor scenes are exactly reproducible and
             hand-explainable.
  has_map = 1.0 for Category COR (and scene Y10, which uses a doorway
             gap); has_map = 0.0 for everything else.
-----------------------------------------------------------------------

Category overview (13 categories, 100 scenes total)
  Original 50, open space, has_map = 0:
    A. Head-on single pedestrian              (5)
    B. Group head-on, varying size/spacing    (8)
    C. Crossing / intersection, single ped    (8)
    D. Overtaking a slow/stationary ped ahead (6)
    E. Overtaking a slow group ahead          (6)
    F. Oblique / diagonal group encounters    (6)
    G. Stationary blockades, varying gaps     (5)
    H. Multi-agent compositions               (6)
  ---------------------------------------------------------------
  New 50, built to make style-axis differences clear and unambiguous,
  always solvable:
    Y.   Yielding-dominant scenarios              (10)  [yield axis]
    GS.  Group cohesion / splitting               (10)  [group axis]
    R.   Rear approach / overtaking                (10)  [pass/prox axis]
    FW.  Four-way intersection negotiation         (10)  [pass/yield axis]
    COR. Corridor / counterflow (has_map = 1)      (10)  [prox axis]
"""

import argparse
import os
import numpy as np
from collections import deque


# =============================================================================
# Map constants (must match crowd_sim.py's load_npz_scenario: map_extent=10,
# row -> y, col -> x) and map_scene_generator.py's SceneMapGenerator defaults.
# =============================================================================
MAP_SIZE     = 50
MAP_EXTENT   = 10.0
RES          = MAP_EXTENT / MAP_SIZE     # 0.2 m / cell
HALF         = MAP_EXTENT / 2.0          # 5.0 m
WALL_THICK   = 0.30                      # matches WALL_THICKNESS_M
AGENT_RADIUS = 0.50
START_CLEAR  = 0.55

EMPTY_MAP = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)


# =============================================================================
# Deterministic ports of SceneMapGenerator's drawing primitives / scene
# formulas from crowd_sim/envs/utils/map_scene_generator.py. Identical math,
# but every random draw is replaced with an explicit, hand-chosen constant
# so each scene is a fixed, reproducible, explainable layout.
# =============================================================================

def _world_to_grid(x, y):
    col = int(round(x / RES + MAP_SIZE / 2))
    row = int(round(y / RES + MAP_SIZE / 2))
    return (int(np.clip(row, 0, MAP_SIZE - 1)), int(np.clip(col, 0, MAP_SIZE - 1)))


def _grid_to_world(row, col):
    return ((col - MAP_SIZE / 2 + 0.5) * RES, (row - MAP_SIZE / 2 + 0.5) * RES)


def _new_map():
    return np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)


def _add_segment(occ, p0, p1, thickness=WALL_THICK):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    seg = p1 - p0
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq < 1e-10:
        return
    half_t = thickness / 2.0
    pad = half_t + RES
    xlo = min(p0[0], p1[0]) - pad;  xhi = max(p0[0], p1[0]) + pad
    ylo = min(p0[1], p1[1]) - pad;  yhi = max(p0[1], p1[1]) + pad
    r0, c0 = _world_to_grid(xlo, ylo)
    r1, c1 = _world_to_grid(xhi, yhi)
    for r in range(min(r0, r1), max(r0, r1) + 1):
        for c in range(min(c0, c1), max(c0, c1) + 1):
            x, y = _grid_to_world(r, c)
            pt = np.array([x, y], dtype=np.float64)
            t = float(np.dot(pt - p0, seg) / seg_len_sq)
            closest = p0 + max(0.0, min(1.0, t)) * seg
            if np.linalg.norm(pt - closest) <= half_t:
                occ[r, c] = 1.0


def _add_rect(occ, center, size_m, angle_rad=0.0):
    cx, cy = float(center[0]), float(center[1])
    w, h = float(size_m[0]), float(size_m[1])
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    bbox = max(w, h) + RES
    r0, c0 = _world_to_grid(cx - bbox, cy - bbox)
    r1, c1 = _world_to_grid(cx + bbox, cy + bbox)
    for r in range(min(r0, r1), max(r0, r1) + 1):
        for c in range(min(c0, c1), max(c0, c1) + 1):
            x, y = _grid_to_world(r, c)
            dx, dy = x - cx, y - cy
            if abs(cos_a * dx + sin_a * dy) <= w / 2 and abs(-sin_a * dx + cos_a * dy) <= h / 2:
                occ[r, c] = 1.0


def wall_map(goal, t, cross_off, side, length, thickness=WALL_THICK):
    """Deterministic port of SceneMapGenerator._gen_single_wall.

    A partial wall perpendicular to the goal direction, anchored `t` of the
    way to the goal and offset by `cross_off` so it crosses the direct
    path, then extended `length` meters toward +y (side=+1, blocking the
    LEFT) or -y (side=-1, blocking the RIGHT). The opposite side is left
    open — that open side is the only viable route to the goal.
    """
    goal = np.asarray(goal, dtype=np.float64)
    goal_dir = goal / np.linalg.norm(goal)
    perp_dir = np.array([-goal_dir[1], goal_dir[0]])
    path_point = goal * t
    wall_start = path_point - perp_dir * side * cross_off
    wall_end = wall_start + perp_dir * side * length
    occ = _new_map()
    _add_segment(occ, wall_start, wall_end, thickness)
    return occ


def corridor_map(goal, half_w, length, thickness=WALL_THICK):
    """Deterministic port of _gen_corridor: two parallel walls straddling
    the goal direction, `half_w` meters either side, forming a channel of
    usable width ~ 2*half_w minus wall thickness."""
    goal = np.asarray(goal, dtype=np.float64)
    axis_angle = np.arctan2(goal[1], goal[0])
    axis = np.array([np.cos(axis_angle), np.sin(axis_angle)])
    perp = np.array([-axis[1], axis[0]])
    occ = _new_map()
    for side in [-1, 1]:
        center = side * half_w * perp
        _add_segment(occ, center + axis * (length / 2), center - axis * (length / 2), thickness)
    return occ


def l_corner_map(goal, t, side, offset, len1, len2, arm2_sign, thickness=WALL_THICK):
    """Deterministic port of _gen_L_corner (goal_norm > 0.5 branch): an
    L-shaped wall corner sitting `offset` meters to one side of the direct
    path. arm1 runs back across the path (forcing a real detour), arm2
    runs along the goal direction. Not a closed room — only two of four
    sides are ever walled, so a route around the open sides always exists.
    """
    goal = np.asarray(goal, dtype=np.float64)
    goal_dir = goal / np.linalg.norm(goal)
    perp_dir = np.array([-goal_dir[1], goal_dir[0]])
    corner = goal * t + perp_dir * side * offset
    arm1_dir = perp_dir * (-side)
    arm2_dir = goal_dir * arm2_sign
    occ = _new_map()
    _add_segment(occ, corner, corner + arm1_dir * len1, thickness)
    _add_segment(occ, corner, corner + arm2_dir * len2, thickness)
    return occ


def doorway_map(goal, t, door_width, door_offset, half_len, thickness=WALL_THICK):
    """Deterministic port of _gen_doorway: a wall spanning the scene,
    perpendicular to the goal direction, with a single gap of width
    `door_width` centerd `door_offset` meters off the direct path. The
    ONLY route to the goal is through that gap."""
    goal = np.asarray(goal, dtype=np.float64)
    goal_angle = np.arctan2(goal[1], goal[0])
    wall_angle = goal_angle + np.pi / 2
    midpoint = goal * t
    axis = np.array([np.cos(wall_angle), np.sin(wall_angle)])
    door_center = midpoint + axis * door_offset
    occ = _new_map()
    _add_segment(occ, door_center + axis * (door_width / 2), door_center + axis * half_len, thickness)
    _add_segment(occ, door_center - axis * (door_width / 2), door_center - axis * half_len, thickness)
    return occ


def narrow_passage_map(goal, t, gap, extra0, extra1, size0, size1, angle0=0.0, angle1=0.0):
    """Deterministic port of _gen_narrow_passage: two rectangular blocks
    centerd on the direct path, separated by a squeeze of width `gap`."""
    goal = np.asarray(goal, dtype=np.float64)
    passage_angle = np.arctan2(goal[1], goal[0]) + np.pi / 2
    perp = np.array([np.cos(passage_angle), np.sin(passage_angle)])
    midpoint = goal * t
    occ = _new_map()
    _add_rect(occ, midpoint + perp * (gap / 2 + extra0), size0, angle0)
    _add_rect(occ, midpoint - perp * (gap / 2 + extra1), size1, angle1)
    return occ


def pillars_map(items):
    """items: list of (x, y, size_m, angle_rad) -> square pillars."""
    occ = _new_map()
    for x, y, size, angle in items:
        _add_rect(occ, (x, y), (size, size), angle)
    return occ


def boxes_map(items):
    """items: list of (x, y, w, h, angle_rad) -> rectangular obstacles."""
    occ = _new_map()
    for x, y, w, h, angle in items:
        _add_rect(occ, (x, y), (w, h), angle)
    return occ


# =============================================================================
# Validity check (mirrors map_scene_generator.py's challenge-mode filter):
# start must be clear, goal cell must be free, and the goal must be BFS
# reachable on the agent-inflated grid. Run at generation time so a bad
# hand-picked geometry is caught instead of silently shipped.
# =============================================================================

def _is_free(occ, pos_xy, clearance_m):
    r, c = _world_to_grid(float(pos_xy[0]), float(pos_xy[1]))
    rad = int(np.ceil(clearance_m / RES))
    return not occ[max(0, r - rad):min(MAP_SIZE, r + rad + 1),
                   max(0, c - rad):min(MAP_SIZE, c + rad + 1)].any()


def _inflated(occ, agent_radius=AGENT_RADIUS):
    rad = int(np.ceil(agent_radius / RES))
    result = occ.copy()
    rows, cols = np.where(occ > 0.5)
    for r, c in zip(rows, cols):
        result[max(0, r - rad):min(MAP_SIZE, r + rad + 1),
               max(0, c - rad):min(MAP_SIZE, c + rad + 1)] = 1.0
    return result


def _bfs_reachable(occ, goal_xy, agent_radius=AGENT_RADIUS):
    inflated = _inflated(occ, agent_radius)
    sr, sc = _world_to_grid(0.0, 0.0)
    # Clip the goal into the map extent for the reachability check only —
    # scene goals (e.g. x=6) may sit outside the +/-5m map, which is fine
    # since the map only needs to keep the NEAR-FIELD route open.
    gx = float(np.clip(goal_xy[0], -HALF + 0.1, HALF - 0.1))
    gy = float(np.clip(goal_xy[1], -HALF + 0.1, HALF - 0.1))
    gr, gc = _world_to_grid(gx, gy)
    if inflated[sr, sc] or inflated[gr, gc]:
        return False
    visited = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    visited[sr, sc] = True
    queue = deque([(sr, sc)])
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while queue:
        r, c = queue.popleft()
        if r == gr and c == gc:
            return True
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < MAP_SIZE and 0 <= nc < MAP_SIZE and not visited[nr, nc] and not inflated[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))
    return False


def _check_map_scene(name, occ, goal):
    if not _is_free(occ, (0.0, 0.0), START_CLEAR):
        print(f"  [WARN] {name}: start position is not clear of walls")
    if not _bfs_reachable(occ, goal):
        print(f"  [WARN] {name}: goal is NOT reachable given this map")


# =============================================================================
# Packing helpers
# =============================================================================

def _npz_kwargs(goal, obstacles, occupancy_map=None, start_v=0.6, start_w=0.0, threat_type=None):
    obs_arr = (np.asarray(obstacles, dtype=np.float32).reshape(-1, 4)
               if len(obstacles) else np.zeros((0, 4), dtype=np.float32))
    has_map = occupancy_map is not None
    kwargs = dict(
        start_state=np.array([start_v, start_w], dtype=np.float32),
        goal=np.asarray(goal, dtype=np.float32),
        obstacles=obs_arr,
        occupancy_map=(occupancy_map if has_map else EMPTY_MAP).astype(np.float32),
        has_map=np.float32(1.0 if has_map else 0.0),
    )
    if threat_type is not None:
        kwargs["threat_type"] = np.array(threat_type)
    return kwargs


# =============================================================================
# Category A — Head-on single pedestrian (5)      [ported unchanged]
# =============================================================================
def scene_head_on_far_slow():
    return "A1_head_on_far_slow", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[4.5, 0.0, -0.5, 0.0]], threat_type="head_on")


def scene_head_on_medium():
    return "A2_head_on_medium", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[3.0, 0.0, -0.7, 0.0]], threat_type="head_on")


def scene_head_on_close():
    return "A3_head_on_close", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[2.0, 0.0, -0.8, 0.0]], threat_type="head_on")


def scene_head_on_imminent():
    return "A4_head_on_imminent", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[1.5, 0.0, -0.2, 0.0]], threat_type="head_on")


def scene_head_on_fast_far():
    return "A5_head_on_fast_far", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[5.0, 0.0, -1.2, 0.0]], threat_type="head_on")


# =============================================================================
# Category B — Group head-on, varying size & spacing (8)  [ported unchanged]
# =============================================================================
def scene_group2_tight_far():
    return "B1_group2_tight_far", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[4.0, -0.4, -0.6, 0.0], [4.0, 0.4, -0.6, 0.0]],
        threat_type="group_head_on")


def scene_group2_wide_far():
    return "B2_group2_wide_far", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[4.0, -1.5, -0.6, 0.0], [4.0, 1.5, -0.6, 0.0]],
        threat_type="group_head_on")


def scene_group3_tight_far():
    return "B3_group3_tight_far", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[4.0, -0.6, -0.6, 0.0], [4.0, 0.0, -0.6, 0.0], [4.0, 0.6, -0.6, 0.0]],
        threat_type="group_head_on")


def scene_group3_wide():
    return "B4_group3_wide", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[4.0, -1.5, -0.6, 0.0], [4.0, 0.0, -0.6, 0.0], [4.0, 1.5, -0.6, 0.0]],
        threat_type="group_head_on")


def scene_group4_tight():
    return "B5_group4_tight", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[4.0, -0.9, -0.6, 0.0], [4.0, -0.3, -0.6, 0.0],
                   [4.0, 0.3, -0.6, 0.0], [4.0, 0.9, -0.6, 0.0]],
        threat_type="group_head_on")


def scene_group2_close_imminent():
    return "B6_group2_close_imminent", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[1.8, -0.4, -0.15, 0.0], [1.8, 0.4, -0.15, 0.0]],
        threat_type="group_head_on")


def scene_group_asymmetric():
    return "B7_group_asymmetric", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[4.0, 0.3, -0.6, 0.0], [4.0, 0.9, -0.6, 0.0], [4.0, -1.8, -0.6, 0.0]],
        threat_type="group_head_on")


def scene_group5_large_tight():
    return "B8_group5_large_tight", _npz_kwargs(
        goal=[6.5, 0.0],
        obstacles=[[4.0, -1.2, -0.6, 0.0], [4.0, -0.6, -0.6, 0.0], [4.0, 0.0, -0.6, 0.0],
                   [4.0, 0.6, -0.6, 0.0], [4.0, 1.2, -0.6, 0.0]],
        threat_type="group_head_on")


# =============================================================================
# Category C — Crossing / intersection, single ped (8)   [ported unchanged]
# =============================================================================
def scene_cross_L2R_far():
    return "C1_cross_L2R_far", _npz_kwargs(
        goal=[5.5, 0.0], obstacles=[[4.5, 3.0, 0.0, -0.8]], threat_type="cutoff")


def scene_cross_L2R_moderate():
    return "C2_cross_L2R_moderate", _npz_kwargs(
        goal=[5.5, 0.0], obstacles=[[3.0, 2.0, 0.0, -0.8]], threat_type="cutoff")


def scene_cross_L2R_close():
    return "C3_cross_L2R_close", _npz_kwargs(
        goal=[5.5, 0.0], obstacles=[[1.8, 1.2, 0.0, -0.8]], threat_type="cutoff")


def scene_cross_L2R_imminent():
    return "C4_cross_L2R_imminent", _npz_kwargs(
        goal=[5.5, 0.0], obstacles=[[1.0, 0.6, 0.0, -0.8]], threat_type="cutoff")


def scene_cross_R2L_far():
    return "C5_cross_R2L_far", _npz_kwargs(
        goal=[5.5, 0.0], obstacles=[[4.5, -3.0, 0.0, 0.8]], threat_type="cutoff")


def scene_cross_R2L_moderate():
    return "C6_cross_R2L_moderate", _npz_kwargs(
        goal=[5.5, 0.0], obstacles=[[3.0, -2.0, 0.0, 0.8]], threat_type="cutoff")


def scene_cross_R2L_close():
    return "C7_cross_R2L_close", _npz_kwargs(
        goal=[5.5, 0.0], obstacles=[[1.8, -1.2, 0.0, 0.8]], threat_type="cutoff")


def scene_cross_R2L_imminent():
    return "C8_cross_R2L_imminent", _npz_kwargs(
        goal=[5.5, 0.0], obstacles=[[1.0, -0.6, 0.0, 0.8]], threat_type="cutoff")


# =============================================================================
# Category D — Overtaking a slow/stationary pedestrian directly ahead (6)
#              [ported unchanged]
# =============================================================================
def scene_overtake_slow_far():
    return "D1_overtake_slow_far", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[3.0, 0.0, 0.3, 0.0]], threat_type="sneak_up")


def scene_overtake_slow_close():
    return "D2_overtake_slow_close", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[1.5, 0.0, 0.3, 0.0]], threat_type="sneak_up")


def scene_overtake_veryslow():
    return "D3_overtake_veryslow", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[2.0, 0.0, 0.1, 0.0]], threat_type="sneak_up")


def scene_overtake_stationary():
    return "D4_overtake_stationary", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[2.0, 0.0, 0.0, 0.0]], threat_type="sneak_up")


def scene_overtake_stationary_close():
    return "D5_overtake_stationary_close", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[1.2, 0.0, 0.0, 0.0]], threat_type="sneak_up")


def scene_overtake_small_diff():
    return "D6_overtake_small_diff", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[2.5, 0.0, 0.4, 0.0]], threat_type="sneak_up")


# =============================================================================
# Category E — Overtaking a slow-moving GROUP ahead (6)   [ported unchanged]
# =============================================================================
def scene_overtake_group2_tight():
    return "E1_overtake_group2_tight", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[2.5, -0.4, 0.3, 0.0], [2.5, 0.4, 0.3, 0.0]],
        threat_type="group_sneak_up")


def scene_overtake_group2_close():
    return "E2_overtake_group2_close", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[1.5, -0.4, 0.2, 0.0], [1.5, 0.4, 0.2, 0.0]],
        threat_type="group_sneak_up")


def scene_overtake_group3_tight():
    return "E3_overtake_group3_tight", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[2.5, -0.6, 0.3, 0.0], [2.5, 0.0, 0.3, 0.0], [2.5, 0.6, 0.3, 0.0]],
        threat_type="group_sneak_up")


def scene_overtake_group2_wide_gap():
    return "E4_overtake_group2_wide_gap", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[2.5, -1.0, 0.3, 0.0], [2.5, 1.0, 0.3, 0.0]],
        threat_type="group_sneak_up")


def scene_overtake_group3_staggered():
    return "E5_overtake_group3_staggered", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[2.0, -0.5, 0.3, 0.0], [2.6, 0.0, 0.25, 0.0], [3.2, 0.5, 0.3, 0.0]],
        threat_type="group_sneak_up")


def scene_overtake_group2_offset_left():
    return "E6_overtake_group2_offset_left", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[2.2, 0.2, 0.4, 0.0], [2.2, 1.0, 0.4, 0.0]],
        threat_type="group_sneak_up")


# =============================================================================
# Category F — Oblique / diagonal group encounters (6)   [ported unchanged]
# =============================================================================
def scene_oblique_group2_topright_to_bottomleft():
    return "F1_oblique_group2_TR_to_BL", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.5, 1.5, -0.5, -0.5], [4.0, 2.0, -0.5, -0.5]],
        threat_type="group_oblique")


def scene_oblique_group2_bottomright_to_topleft():
    return "F2_oblique_group2_BR_to_TL", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.5, -1.5, -0.5, 0.5], [4.0, -2.0, -0.5, 0.5]],
        threat_type="group_oblique")


def scene_oblique_group3_crossing():
    return "F3_oblique_group3_crossing", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.0, 2.5, -0.4, -0.6], [3.5, 3.0, -0.4, -0.6], [4.0, 3.5, -0.4, -0.6]],
        threat_type="group_oblique")


def scene_diag_oncoming_group2_left():
    return "F4_diag_oncoming_group2_left", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.5, 1.0, -0.6, -0.3], [4.0, 1.5, -0.6, -0.3]],
        threat_type="group_oblique")


def scene_diag_oncoming_group2_right():
    return "F5_diag_oncoming_group2_right", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.5, -1.0, -0.6, 0.3], [4.0, -1.5, -0.6, 0.3]],
        threat_type="group_oblique")


def scene_sequential_cross_two_peds():
    return "F6_sequential_cross_two_peds", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[2.0, 2.0, 0.0, -0.7], [4.0, -2.5, 0.0, 0.7]],
        threat_type="multi_cutoff")


# =============================================================================
# Category G — Stationary blockades with varying gap sizes (5)
#              [ported unchanged]
# =============================================================================
def scene_blockade_gap_threadable():
    return "G1_blockade_gap_threadable", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.5, -0.6, 0.0, 0.0], [3.5, 0.6, 0.0, 0.0]],
        threat_type="stationary_blockade")


def scene_blockade_gap_tight():
    return "G2_blockade_gap_tight", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.5, -0.3, 0.0, 0.0], [3.5, 0.3, 0.0, 0.0]],
        threat_type="stationary_blockade")


def scene_blockade_gap_wide():
    return "G3_blockade_gap_wide", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.5, -1.0, 0.0, 0.0], [3.5, 1.0, 0.0, 0.0]],
        threat_type="stationary_blockade")


def scene_single_static_blocker():
    return "G4_single_static_blocker", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[2.5, 0.0, 0.0, 0.0]], threat_type="stationary_blockade")


def scene_blockade_asymmetric_gaps():
    return "G5_blockade_asymmetric_gaps", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.5, -2.0, 0.0, 0.0], [3.5, 0.0, 0.0, 0.0], [3.5, 0.8, 0.0, 0.0]],
        threat_type="stationary_blockade")


# =============================================================================
# Category H — Multi-agent compositions (6)   [ported unchanged]
# =============================================================================
def scene_cross_plus_slow_ahead():
    return "H1_cross_plus_slow_ahead", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.0, 2.0, 0.0, -0.8], [2.0, 0.0, 0.3, 0.0]],
        threat_type="composite")


def scene_headon_plus_cross():
    return "H2_headon_plus_cross", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.0, 0.0, -0.7, 0.0], [2.5, -2.0, 0.0, 0.7]],
        threat_type="composite")


def scene_converging_cross_both_sides():
    return "H3_converging_cross_both_sides", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[2.5, 2.0, 0.0, -0.8], [2.5, -2.0, 0.0, 0.8]],
        threat_type="composite")


def scene_sandwich_slow_and_oncoming():
    return "H4_sandwich_slow_and_oncoming", _npz_kwargs(
        goal=[6.5, 0.0],
        obstacles=[[1.5, 0.0, 0.3, 0.0], [4.0, 0.0, -0.7, 0.0]],
        threat_type="composite")


def scene_group_ahead_plus_cross():
    return "H5_group_ahead_plus_cross", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[2.5, -0.4, 0.3, 0.0], [2.5, 0.4, 0.3, 0.0], [4.0, 2.0, 0.0, -0.8]],
        threat_type="composite")


def scene_two_independent_crossings_timed():
    return "H6_two_independent_crossings_timed", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[1.5, 1.0, 0.0, -0.6], [4.0, 2.5, 0.0, -0.6]],
        threat_type="composite")


# =============================================================================
# Category Y — Yielding-dominant scenarios (10)
#
# Every scene here is built so pass-side and prox stay roughly
# uninformative (the geometry doesn't force a side, and there's normally
# plenty of room) while WHO GOES FIRST / HOW EARLY THE ROBOT CONCEDES
# space is the only thing that should visibly differ between yield+ and
# yield- policies. All open space (has_map = 0) except Y10, which uses a
# doorway gap on purpose (the point of Y10 is literally "don't squeeze
# through a blocked doorway").
# =============================================================================
def scene_y1_parent_child_crossing():
    """Two pedestrians crossing together, side-by-side, gap between them
    (~0.7 m) is geometrically threadable but they should not be treated
    as two independent single peds to slip between — yield+ should
    concede the whole lane, yield- may try to slot through the gap."""
    return "Y1_parent_child_crossing", _npz_kwargs(
        goal=[5.5, 0.0],
        obstacles=[[3.5, 2.2, 0.0, -0.7], [3.5, 2.9, 0.0, -0.7]],
        threat_type="group_cutoff")


def scene_y2_elderly_slow_crossing():
    """Single slow crosser with a long dwell time inside the conflict
    region — yield+ should hang back for the whole crossing; yield-
    should try to nose through before/after the narrow window."""
    return "Y2_elderly_slow_crossing", _npz_kwargs(
        goal=[5.5, 0.0], obstacles=[[3.0, 2.0, 0.0, -0.25]], threat_type="cutoff")


def scene_y3_human_already_in_intersection():
    """The pedestrian has already entered the conflict region (close to
    the robot's path) by the time the robot arrives — a clean
    right-of-way test with no ambiguity about who arrived first."""
    return "Y3_human_already_in_intersection", _npz_kwargs(
        goal=[5.5, 0.0], obstacles=[[3.0, 0.3, 0.0, -0.6]], threat_type="cutoff")


def scene_y4_simultaneous_arrival():
    """Speeds/positions tuned so the pedestrian and the robot reach the
    conflict point at roughly the same time — the ambiguous 50/50 case
    where yield+ concedes and yield- pushes through."""
    return "Y4_simultaneous_arrival", _npz_kwargs(
        goal=[5.5, 0.0], obstacles=[[3.5, 3.4, 0.0, -0.6]], threat_type="cutoff")


def scene_y5_human_standing_then_crossing():
    """Pedestrian is essentially stationary at the path edge with only a
    tiny residual velocity into the lane — tests anticipation (does the
    robot treat this as "about to cross" ahead of time) vs. pure reaction
    to already-observed motion."""
    return "Y5_human_standing_then_crossing", _npz_kwargs(
        goal=[5.5, 0.0], obstacles=[[3.0, 1.0, 0.0, -0.12]], threat_type="cutoff")


def scene_y6_diagonal_crosser_growing_conflict():
    """Pedestrian moves diagonally, angling into the robot's lane so the
    conflict grows over time rather than being present from the start —
    tests whether yield- keeps pushing even as the conflict worsens."""
    return "Y6_diagonal_crosser_growing_conflict", _npz_kwargs(
        goal=[5.5, 0.0], obstacles=[[3.0, 2.0, -0.3, -0.6]], threat_type="cutoff")


def scene_y7_small_group_crossing_together():
    """Three pedestrians crossing together as one unit — a larger-scale
    version of Y1, isolating whether yield deference scales with group
    size the way it should regardless of the group axis."""
    return "Y7_small_group_crossing_together", _npz_kwargs(
        goal=[5.5, 0.0],
        obstacles=[[3.5, 2.0, 0.0, -0.7], [3.5, 2.6, 0.0, -0.7], [3.5, 3.2, 0.0, -0.7]],
        threat_type="group_cutoff")


def scene_y8_human_crossing_distracted():
    """A closer, tighter-margin crossing meant to stand in for a
    distracted / inattentive pedestrian who won't react to the robot —
    more conservative (yield+) styles should concede noticeably earlier
    here than on the equivalent Y3-style baseline."""
    return "Y8_human_crossing_distracted", _npz_kwargs(
        goal=[5.5, 0.0], obstacles=[[2.5, 1.5, 0.0, -0.7]], threat_type="cutoff")


def scene_y9_two_humans_pass_each_other():
    """Two pedestrians are walking toward each other (not toward the
    robot) and are about to pass one another right where the robot's
    path crosses theirs — the robot must judge when that shared space
    frees up rather than just tracking either one individually."""
    return "Y9_two_humans_pass_each_other", _npz_kwargs(
        goal=[5.5, 0.0],
        obstacles=[[3.0, -2.0, 0.0, 0.6], [3.0, 2.0, 0.0, -0.6]],
        threat_type="multi_cutoff")


def scene_y10_human_stopped_at_doorway():
    """A wide-enough (3.0 m) doorway gap with a pedestrian standing
    stationary inside it — the robot should wait for the person to clear
    rather than trying to squeeze past inside the gap. Uses a static map
    on purpose; the doorway is comfortably wide when clear, so this is a
    yield test, not a prox/pinch test."""
    occ = doorway_map(goal=[6.0, 0.0], t=0.5, door_width=3.0, door_offset=0.0, half_len=3.5)
    return "Y10_human_stopped_at_doorway", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[3.0, 0.0, 0.0, 0.0]],
        occupancy_map=occ, threat_type="doorway_yield")


# =============================================================================
# Category GS — Group cohesion / splitting (10)
#
# Stationary or co-moving clusters of 2-4 pedestrians, several with
# gaps that are geometrically threadable. group+ should route fully
# around the cluster as a unit; group- may cut between members. All open
# space, has_map = 0.
# =============================================================================
def scene_gs1_two_person_conversation():
    """Two pedestrians stopped, facing each other, directly on the
    robot's path — the simplest possible "go around, don't go through"
    test."""
    return "GS1_two_person_conversation", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.0, -0.35, 0.0, 0.0], [3.0, 0.35, 0.0, 0.0]],
        threat_type="stationary_group")


def scene_gs2_three_person_conversation_circle():
    """Three pedestrians stopped in a small conversational circle
    straddling the direct path; the goal is beyond them, so the robot
    must commit to one side of the whole circle."""
    return "GS2_three_person_conversation_circle", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[2.8, -0.5, 0.0, 0.0], [2.8, 0.5, 0.0, 0.0], [3.5, 0.0, 0.0, 0.0]],
        threat_type="stationary_group")


def scene_gs3_four_person_conversation_square():
    """Four pedestrians stopped in a small square formation centerd on
    the path — a larger stationary cluster than GS2, still with an
    internal gap a naive planner could be tempted to cut through."""
    return "GS3_four_person_conversation_square", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[2.6, -0.45, 0.0, 0.0], [2.6, 0.45, 0.0, 0.0],
                   [3.4, -0.45, 0.0, 0.0], [3.4, 0.45, 0.0, 0.0]],
        threat_type="stationary_group")


def scene_gs4_walking_pair_shoulder_to_shoulder():
    """Two pedestrians walking together, shoulder-to-shoulder, diagonally
    across the robot's path (not stationary) — tests whether the moving
    version of a tight pair is still treated as one unit."""
    return "GS4_walking_pair_shoulder_to_shoulder", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.5, 2.0, -0.3, -0.6], [3.5, 2.7, -0.3, -0.6]],
        threat_type="group_oblique")


def scene_gs5_walking_triple_abreast():
    """Three pedestrians walking abreast, crossing diagonally — the
    moving, larger-group counterpart to GS4."""
    return "GS5_walking_triple_abreast", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.5, 2.0, -0.2, -0.6], [3.5, 2.6, -0.2, -0.6], [3.5, 3.2, -0.2, -0.6]],
        threat_type="group_oblique")


def scene_gs6_leader_with_follower_behind():
    """Two pedestrians walking in-line, one a step behind the other
    (same lateral offset, staggered longitudinally) — technically
    splittable (they're never side-by-side), but socially still a
    single group walking together; tests whether group+ recognizes an
    in-line pair as a unit rather than two independent singles."""
    return "GS6_leader_with_follower_behind", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[4.0, 0.3, -0.5, 0.0], [3.3, 0.3, -0.5, 0.0]],
        threat_type="group_head_on")


def scene_gs7_diamond_formation_group_of_four():
    """Four pedestrians walking head-on in a diamond formation (one
    front, one back, one each side) — the largest moving cluster in this
    set, testing whether group+ arcs around the full diamond rather than
    just the nearest member."""
    return "GS7_diamond_formation_group_of_four", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[4.6, 0.0, -0.5, 0.0], [3.4, 0.0, -0.5, 0.0],
                   [4.0, 0.5, -0.5, 0.0], [4.0, -0.5, -0.5, 0.0]],
        threat_type="group_head_on")


def scene_gs8_two_person_borderline_gap():
    """Two pedestrians walking head-on with a borderline 1.2 m gap
    between them — wide enough that a purely local planner might thread
    it, narrow enough that it's a clear group- vs group+ fork."""
    return "GS8_two_person_borderline_gap", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[4.0, -0.6, -0.5, 0.0], [4.0, 0.6, -0.5, 0.0]],
        threat_type="group_head_on")


def scene_gs9_pair_crossing_diagonally():
    """A tight pair crossing the path diagonally (not stopped, not
    head-on) — contrast case to GS4/GS8 with a different geometry so
    group cohesion is tested across encounter types, not just head-on."""
    return "GS9_pair_crossing_diagonally", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.2, 2.4, -0.5, -0.65], [3.2, 3.0, -0.5, -0.65]],
        threat_type="group_oblique")


def scene_gs10_pair_robot_arrives_through_center_gap():
    """The strongest group-axis test in this set: a tight pair (1.0 m
    gap) is positioned exactly straddling the robot's direct line to the
    goal, so the naive, shortest path drives the robot straight between
    them. group+ must actively detour around the whole pair even though
    the direct route is geometrically clear."""
    return "GS10_pair_robot_arrives_through_center_gap", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.0, -0.5, -0.35, 0.0], [3.0, 0.5, -0.35, 0.0]],
        threat_type="group_head_on")


# =============================================================================
# Category R — Rear approach / overtaking (10)
#
# The robot catches up to pedestrian(s) walking ahead of it in roughly
# the same direction (not head-on), isolating pass-side choice and
# following distance from any head-on yield behavior. All open space,
# has_map = 0.
# =============================================================================
def scene_r1_overtake_left_rear_quarter():
    """Slow pedestrian ahead, offset slightly to the robot's right, so
    the natural overtake line passes the pedestrian's left-rear
    quarter."""
    return "R1_overtake_left_rear_quarter", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[2.5, -0.3, 0.25, 0.0]], threat_type="sneak_up")


def scene_r2_overtake_right_rear_quarter():
    """Mirror of R1: pedestrian offset slightly to the robot's left, so
    the natural overtake line passes the right-rear quarter."""
    return "R2_overtake_right_rear_quarter", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[2.5, 0.3, 0.25, 0.0]], threat_type="sneak_up")


def scene_r3_fast_robot_catching_slow_ped():
    """Robot's default speed is much greater than the pedestrian's — a
    clean, unhurried catch-up-and-pass with plenty of lead time to
    choose a side and clearance."""
    return "R3_fast_robot_catching_slow_ped", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[2.0, 0.0, 0.12, 0.0]], threat_type="sneak_up")


def scene_r4_fast_robot_catching_walking_pair():
    """Same large speed differential as R3, but overtaking a slow pair
    walking together instead of a single pedestrian — combines pass and
    group axes in an overtake context."""
    return "R4_fast_robot_catching_walking_pair", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[2.2, -0.3, 0.2, 0.0], [2.2, 0.3, 0.2, 0.0]],
        threat_type="group_sneak_up")


def scene_r5_human_drifting_left_while_overtaking():
    """Pedestrian ahead drifts toward +y (the robot's left) while being
    overtaken — tests whether pass- keeps close on a side that's
    actively closing, vs. pass+ giving more margin."""
    return "R5_human_drifting_left_while_overtaking", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[2.3, 0.0, 0.3, 0.15]], threat_type="sneak_up")


def scene_r6_human_drifting_right_while_overtaking():
    """Mirror of R5: pedestrian drifts toward -y (the robot's right)
    while being overtaken."""
    return "R6_human_drifting_right_while_overtaking", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[2.3, 0.0, 0.3, -0.15]], threat_type="sneak_up")


def scene_r7_human_stops_suddenly():
    """Pedestrian directly ahead, close, with almost zero remaining
    forward velocity — approximates a pedestrian that has just stopped,
    forcing an abrupt overtake-or-brake decision with little lead
    time."""
    return "R7_human_stops_suddenly", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[1.8, 0.0, 0.02, 0.0]], threat_type="sneak_up")


def scene_r8_human_turns_across_path():
    """Pedestrian ahead is moving mostly laterally (turning across the
    robot's path) rather than continuing straight — an overtake that
    turns into a crossing conflict partway through."""
    return "R8_human_turns_across_path", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[2.5, 0.0, 0.1, -0.5]], threat_type="sneak_up")


def scene_r9_two_staggered_peds_ahead():
    """Two independent (not grouped) pedestrians ahead at different
    positions and speeds, staggered left/right — tests whether the
    robot picks one consistent side to pass both rather than weaving."""
    return "R9_two_staggered_peds_ahead", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[2.0, 0.3, 0.2, 0.0], [3.0, -0.3, 0.15, 0.0]],
        threat_type="multi_sneak_up")


def scene_r10_one_stationary_one_moving_ahead():
    """A stationary pedestrian close ahead plus a second, slower-moving
    pedestrian farther out on the opposite side — the robot must clear
    both in sequence, choosing sides independently for each."""
    return "R10_one_stationary_one_moving_ahead", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[2.0, 0.4, 0.0, 0.0], [3.2, -0.2, 0.25, 0.0]],
        threat_type="multi_sneak_up")


# =============================================================================
# Category FW — Four-way intersection negotiation (10)
#
# Multiple pedestrians converge from different directions toward a
# shared conflict point on the robot's path, testing pass-side choice
# and yield timing together under 1-4 simultaneous agents. All open
# space, has_map = 0.
# =============================================================================
def scene_fw1_human_from_left():
    """Single pedestrian crossing in from the robot's left toward the
    conflict point — baseline single-direction case."""
    return "FW1_human_from_left", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[3.0, 2.5, 0.0, -0.7]], threat_type="cutoff")


def scene_fw2_human_from_right():
    """Mirror of FW1: pedestrian crossing in from the robot's right."""
    return "FW2_human_from_right", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[3.0, -2.5, 0.0, 0.7]], threat_type="cutoff")


def scene_fw3_humans_from_both_sides_simultaneous():
    """Two pedestrians converge on the conflict point from left and
    right at the same time — the robot must resolve two simultaneous
    yield decisions, not just one."""
    return "FW3_humans_from_both_sides_simultaneous", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.0, 2.5, 0.0, -0.7], [3.0, -2.5, 0.0, 0.7]],
        threat_type="multi_cutoff")


def scene_fw4_four_agents_NSEW():
    """Four pedestrians approach the same conflict point from all four
    directions (N/S along y, E/W along x) — the busiest, most demanding
    intersection scene in this set."""
    return "FW4_four_agents_NSEW", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.2, 3.0, 0.0, -0.6], [2.8, -3.0, 0.0, 0.6],
                   [6.0, 0.2, -0.6, 0.0], [0.6, -0.2, 0.6, 0.0]],
        threat_type="multi_cutoff")


def scene_fw5_three_agents_converging():
    """Three of the four FW4 directions (N, E, W) converge on the
    conflict point — a slightly less crowded intersection contrast to
    FW4."""
    return "FW5_three_agents_converging", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.2, 3.0, 0.0, -0.6], [6.0, 0.2, -0.6, 0.0], [0.6, -0.2, 0.6, 0.0]],
        threat_type="multi_cutoff")


def scene_fw6_crossing_plus_headon():
    """One pedestrian crosses laterally while a second approaches
    head-on along the robot's own lane — combines a crossing yield
    decision with a head-on one."""
    return "FW6_crossing_plus_headon", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.0, 2.5, 0.0, -0.7], [4.5, 0.0, -0.6, 0.0]],
        threat_type="multi_cutoff")


def scene_fw7_two_crossing_plus_stationary_observer():
    """Two pedestrians cross from opposite sides while a third stands
    stationary just off to one side — tests whether the stationary
    bystander is still given appropriate clearance amid the busier
    crossing conflict."""
    return "FW7_two_crossing_plus_stationary_observer", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.0, 2.5, 0.0, -0.7], [3.0, -2.5, 0.0, 0.7], [3.5, 1.0, 0.0, 0.0]],
        threat_type="multi_cutoff")


def scene_fw8_human_already_traversing_center():
    """A pedestrian is already close to the conflict point (mid-crossing)
    by the time the robot arrives — an unambiguous right-of-way case
    inside the intersection framing."""
    return "FW8_human_already_traversing_center", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[3.0, 0.3, 0.0, -0.6]], threat_type="cutoff")


def scene_fw9_human_reaches_center_before_robot():
    """Timed so the pedestrian reaches the conflict point clearly before
    the robot would — yield+ should visibly let them fully clear first;
    yield- has a real, unambiguous gap it could try to beat instead."""
    return "FW9_human_reaches_center_before_robot", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[3.0, 1.2, 0.0, -0.8]], threat_type="cutoff")


def scene_fw10_human_reaches_center_after_robot():
    """Timed so the pedestrian would reach the conflict point clearly
    after the robot — the "safe to proceed" contrast case to FW9, useful
    for checking that yield+ doesn't over-conservatively stop when
    there's genuinely no conflict."""
    return "FW10_human_reaches_center_after_robot", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[3.0, 3.2, 0.0, -0.5]], threat_type="cutoff")


# =============================================================================
# Category COR — Corridor / counterflow (10, has_map = 1)
#
# A static corridor (two parallel walls) confines the robot laterally,
# isolating how tightly a given prox setting hugs one side vs. the other
# under oncoming singles, pairs, and triples. Every corridor here uses a
# half-width chosen so the BFS reachability check (agent-inflated, same
# as the original map-scene generator) always finds a route of at least
# ~1.0 m clear width down the centerline — no scene in this category can
# leave the goal unreachable.
# =============================================================================
def scene_cor1_narrow_corridor_single_oncoming():
    """Narrow corridor (usable width ~2.5 m, ~1.3 m clear after agent
    inflation) with a single oncoming pedestrian — the tightest
    single-ped prox test in this set that's still always solvable."""
    occ = corridor_map(goal=[6.0, 0.0], half_w=1.4, length=6.0)
    return "COR1_narrow_corridor_single_oncoming", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[3.0, 0.0, -0.6, 0.0]],
        occupancy_map=occ, threat_type="corridor_head_on")


def scene_cor2_wide_corridor_single_oncoming():
    """Same oncoming pedestrian as COR1, but a much wider corridor
    (usable width ~4.1 m) — contrast case to see whether lateral
    deviation scales with the room the map actually provides."""
    occ = corridor_map(goal=[6.0, 0.0], half_w=2.2, length=6.0)
    return "COR2_wide_corridor_single_oncoming", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[3.0, 0.0, -0.6, 0.0]],
        occupancy_map=occ, threat_type="corridor_head_on")


def scene_cor3_narrow_corridor_oncoming_pair():
    """Narrower-ish corridor with a tight oncoming
    pair — combines the prox axis with a group encounter under lateral
    confinement."""
    occ = corridor_map(goal=[6.0, 0.0], half_w=2.0, length=6.0)
    return "COR3_narrow_corridor_oncoming_pair", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.5, -0.4, -0.5, 0.0], [3.5, 0.4, -0.5, 0.0]],
        occupancy_map=occ, threat_type="corridor_group_head_on")


def scene_cor4_corridor_stationary_ped_on_side():
    """A pedestrian stands stationary close to one wall of the corridor
    — tests whether the robot still gives it reasonable clearance rather
    than hugging that same wall."""
    occ = corridor_map(goal=[6.0, 0.0], half_w=1.6, length=6.0)
    return "COR4_corridor_stationary_ped_on_side", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[2.5, 0.8, 0.0, 0.0]],
        occupancy_map=occ, threat_type="corridor_sneak_up")


def scene_cor5_corridor_slow_ped_ahead():
    """A slow pedestrian walks ahead of the robot inside the corridor,
    same direction — an overtake confined by walls rather than open
    space, contrasting Category R's open-space overtakes."""
    occ = corridor_map(goal=[6.0, 0.0], half_w=1.5, length=6.0)
    return "COR5_corridor_slow_ped_ahead", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[2.5, 0.0, 0.25, 0.0]],
        occupancy_map=occ, threat_type="corridor_sneak_up")


def scene_cor6_counterflow_lane_of_two():
    """A moderately wide corridor with two independent (not grouped)
    oncoming pedestrians spaced well apart across the lane — a
    counterflow case where the robot must pick a consistent side rather
    than weaving between them."""
    occ = corridor_map(goal=[6.0, 0.0], half_w=2.0, length=6.0)
    return "COR6_counterflow_lane_of_two", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.5, -0.6, -0.4, 0.0], [3.5, 0.6, -0.4, 0.0]],
        occupancy_map=occ, threat_type="corridor_group_head_on")


def scene_cor7_counterflow_lane_of_three():
    """Wide corridor with three independent oncoming pedestrians spread
    across the lane — the busiest counterflow case, testing whether the
    robot still commits to one side rather than threading between
    them."""
    occ = corridor_map(goal=[6.0, 0.0], half_w=2.2, length=6.0)
    return "COR7_counterflow_lane_of_three", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.5, -0.6, -0.4, 0.0], [3.5, 0.0, -0.4, 0.0], [3.5, 0.6, -0.4, 0.0]],
        occupancy_map=occ, threat_type="corridor_group_head_on")


def scene_cor8_oncoming_plus_overtake_simultaneous():
    """A slow pedestrian ahead (to be overtaken) and a faster oncoming
    pedestrian farther out, both sharing the same confined corridor —
    the robot must overtake the near one before the oncoming one
    arrives, with limited room to spare for either manoeuvre."""
    occ = corridor_map(goal=[6.0, 0.0], half_w=1.8, length=6.0)
    return "COR8_oncoming_plus_overtake_simultaneous", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[4.0, 0.0, -0.6, 0.0], [2.0, 0.0, 0.2, 0.0]],
        occupancy_map=occ, threat_type="corridor_group_head_on")


def scene_cor9_corridor_bottleneck_stationary_pair():
    """A stationary pair stands together mid-corridor — the corridor
    itself is always passable, but the pair (not the walls) forms the
    actual bottleneck the robot must route around."""
    occ = corridor_map(goal=[6.0, 0.0], half_w=1.8, length=6.0)
    return "COR9_corridor_bottleneck_stationary_pair", _npz_kwargs(
        goal=[6.0, 0.0],
        obstacles=[[3.0, -0.3, 0.0, 0.0], [3.0, 0.3, 0.0, 0.0]],
        occupancy_map=occ, threat_type="corridor_group_head_on")


def scene_cor10_human_emerges_from_side():
    """A pedestrian starts close to one corridor wall and moves inward
    across the lane, approximating someone stepping out from a side
    opening into the flow of travel — tests reaction to a lateral
    entrance rather than a straight-on approach."""
    occ = corridor_map(goal=[6.0, 0.0], half_w=1.8, length=6.0)
    return "COR10_human_emerges_from_side", _npz_kwargs(
        goal=[6.0, 0.0], obstacles=[[3.0, 1.5, 0.0, -0.6]],
        occupancy_map=occ, threat_type="corridor_head_on")


# =============================================================================
# Writer
# =============================================================================

SCENES = [
    # A — head-on single ped
    scene_head_on_far_slow,
    scene_head_on_medium,
    scene_head_on_close,
    scene_head_on_imminent,
    scene_head_on_fast_far,
    # B — group head-on
    scene_group2_tight_far,
    scene_group2_wide_far,
    scene_group3_tight_far,
    scene_group3_wide,
    scene_group4_tight,
    scene_group2_close_imminent,
    scene_group_asymmetric,
    scene_group5_large_tight,
    # C — crossing / intersection
    scene_cross_L2R_far,
    scene_cross_L2R_moderate,
    scene_cross_L2R_close,
    scene_cross_L2R_imminent,
    scene_cross_R2L_far,
    scene_cross_R2L_moderate,
    scene_cross_R2L_close,
    scene_cross_R2L_imminent,
    # D — overtaking a single slow/stationary ped
    scene_overtake_slow_far,
    scene_overtake_slow_close,
    scene_overtake_veryslow,
    scene_overtake_stationary,
    scene_overtake_stationary_close,
    scene_overtake_small_diff,
    # E — overtaking a slow group
    scene_overtake_group2_tight,
    scene_overtake_group2_close,
    scene_overtake_group3_tight,
    scene_overtake_group2_wide_gap,
    scene_overtake_group3_staggered,
    scene_overtake_group2_offset_left,
    # F — oblique / diagonal group encounters
    scene_oblique_group2_topright_to_bottomleft,
    scene_oblique_group2_bottomright_to_topleft,
    scene_oblique_group3_crossing,
    scene_diag_oncoming_group2_left,
    scene_diag_oncoming_group2_right,
    scene_sequential_cross_two_peds,
    # G — stationary blockades
    scene_blockade_gap_threadable,
    scene_blockade_gap_tight,
    scene_blockade_gap_wide,
    scene_single_static_blocker,
    scene_blockade_asymmetric_gaps,
    # H — multi-agent compositions
    scene_cross_plus_slow_ahead,
    scene_headon_plus_cross,
    scene_converging_cross_both_sides,
    scene_sandwich_slow_and_oncoming,
    scene_group_ahead_plus_cross,
    scene_two_independent_crossings_timed,
    # Y — yielding-dominant scenarios
    scene_y1_parent_child_crossing,
    scene_y2_elderly_slow_crossing,
    scene_y3_human_already_in_intersection,
    scene_y4_simultaneous_arrival,
    scene_y5_human_standing_then_crossing,
    scene_y6_diagonal_crosser_growing_conflict,
    scene_y7_small_group_crossing_together,
    scene_y8_human_crossing_distracted,
    scene_y9_two_humans_pass_each_other,
    scene_y10_human_stopped_at_doorway,
    # GS — group cohesion / splitting
    scene_gs1_two_person_conversation,
    scene_gs2_three_person_conversation_circle,
    scene_gs3_four_person_conversation_square,
    scene_gs4_walking_pair_shoulder_to_shoulder,
    scene_gs5_walking_triple_abreast,
    scene_gs6_leader_with_follower_behind,
    scene_gs7_diamond_formation_group_of_four,
    scene_gs8_two_person_borderline_gap,
    scene_gs9_pair_crossing_diagonally,
    scene_gs10_pair_robot_arrives_through_center_gap,
    # R — rear approach / overtaking
    scene_r1_overtake_left_rear_quarter,
    scene_r2_overtake_right_rear_quarter,
    scene_r3_fast_robot_catching_slow_ped,
    scene_r4_fast_robot_catching_walking_pair,
    scene_r5_human_drifting_left_while_overtaking,
    scene_r6_human_drifting_right_while_overtaking,
    scene_r7_human_stops_suddenly,
    scene_r8_human_turns_across_path,
    scene_r9_two_staggered_peds_ahead,
    scene_r10_one_stationary_one_moving_ahead,
    # FW — four-way intersection negotiation
    scene_fw1_human_from_left,
    scene_fw2_human_from_right,
    scene_fw3_humans_from_both_sides_simultaneous,
    scene_fw4_four_agents_NSEW,
    scene_fw5_three_agents_converging,
    scene_fw6_crossing_plus_headon,
    scene_fw7_two_crossing_plus_stationary_observer,
    scene_fw8_human_already_traversing_center,
    scene_fw9_human_reaches_center_before_robot,
    scene_fw10_human_reaches_center_after_robot,
    # COR — corridor / counterflow
    scene_cor1_narrow_corridor_single_oncoming,
    scene_cor2_wide_corridor_single_oncoming,
    scene_cor3_narrow_corridor_oncoming_pair,
    scene_cor4_corridor_stationary_ped_on_side,
    scene_cor5_corridor_slow_ped_ahead,
    scene_cor6_counterflow_lane_of_two,
    scene_cor7_counterflow_lane_of_three,
    scene_cor8_oncoming_plus_overtake_simultaneous,
    scene_cor9_corridor_bottleneck_stationary_pair,
    scene_cor10_human_emerges_from_side,
]


def generate_all(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    n_map = 0
    for fn in SCENES:
        name, kwargs = fn()
        if float(kwargs["has_map"]) > 0.5:
            _check_map_scene(name, kwargs["occupancy_map"], kwargs["goal"])
            n_map += 1
        out_path = os.path.join(out_dir, f"sample_{name}.npz")
        np.savez(out_path, **kwargs)
        n_obs = kwargs["obstacles"].shape[0]
        goal = kwargs["goal"]
        has_map = bool(kwargs["has_map"] > 0.5)
        print(f"  saved {out_path}  (goal={goal}, n_obs={n_obs}, has_map={has_map})")
        paths.append(out_path)
    print(f"\n{len(paths)} test scenes written to {out_dir}/ "
          f"({len(paths) - n_map} open-space, {n_map} with a static map)")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate 100 canonical style-test scenes (50 open-space + 50 style-focused)")
    parser.add_argument("--out_dir", default="test_scenes3",
                        help="Output directory for .npz scene files")
    args = parser.parse_args()
    generate_all(args.out_dir)


if __name__ == "__main__":
    main()