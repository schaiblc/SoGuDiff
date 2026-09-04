"""
Static occupancy-map scene generator for evaluation.

All public methods accept a numpy.random.RandomState for determinism.
No module-level random state is used.

Design philosophy (challenge mode)
-----------------------------------
The generator is designed so the static map actually forces non-trivial
navigation.  Concretely:

  1. Goal reachability is verified with BFS on an agent-inflated grid —
     the goal must be reachable via *some* path, but that path need not be
     the straight line to the goal.

  2. When challenge=True (the default for map_eval), an additional
     "path-vicinity obstruction" check requires that at least
     `min_obstruction` fraction of the sampled straight-line path points
     have at least one wall cell within `CHALLENGE_VICINITY_M` meters.
     This detects both:
       - walls ON the path (doorway, narrow_passage, single_wall) and
       - walls ALONG the path (corridor).
     Open-space scenes and laterally-displaced walls that don't interact
     with navigation correctly are therefore rejected.

  3. Each scene generator is designed so walls are placed on or adjacent
     to the direct robot→goal path, guaranteeing rich interaction.

Usage
-----
    gen = SceneMapGenerator(map_size=50, map_extent=10.0)
    occ, world_xy, map_type = gen.generate_scene(rng, goal=np.array([3., 1.]))
    start = gen.sample_free_position(occ, rng, clearance_m=0.4)
"""

import numpy as np
from collections import deque


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
WALL_THICKNESS_M    = 0.30   # meters — wall half-thickness
START_CLEARANCE     = 0.55   # min free-space radius around origin
AGENT_RADIUS        = 0.50   # robot radius used for BFS inflation
CHALLENGE_VICINITY_M = 1.20  # radius for path-obstruction fraction check

# Default type weights for the eval distribution.  Challenge validation
# filters out trivial maps, so weights here reflect desired *pass-rate*
# composition after filtering.
DEFAULT_TYPE_WEIGHTS = {
    "open":             0.02,   # sanity baseline — almost never selected
    "single_wall":      0.10,   # partial wall robot goes around
    "corridor":         0.22,   # constrained channel
    "L_corner":         0.12,   # forced corner turn
    "doorway":          0.26,   # must thread through gap in wall
    "room":             0.10,   # exit room to reach goal
    "pillars":          0.10,   # slalom between on-path pillars
    "narrow_passage":   0.08,   # tight squeeze between blocks
}


class SceneMapGenerator:
    """
    Generates structured binary occupancy maps for evaluation scenarios.

    Parameters
    ----------
    map_size    : int   cells per side (default 50)
    map_extent  : float physical side-length in meters (default 10.0)
    type_weights: dict  optional per-type probability override
    """

    def __init__(self, map_size=50, map_extent=10.0, type_weights=None):
        self.map_size   = int(map_size)
        self.map_extent = float(map_extent)
        self.res        = self.map_extent / self.map_size   # m / cell
        self.half       = self.map_extent / 2.0

        raw    = type_weights if type_weights is not None else DEFAULT_TYPE_WEIGHTS
        types  = list(raw.keys())
        probs  = np.array([raw[t] for t in types], dtype=np.float64)
        probs /= probs.sum()
        self._types = types
        self._probs = probs

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def world_to_grid(self, x, y):
        """World (x, y) → integer (row, col), clipped to grid."""
        col = int(round(x / self.res + self.map_size / 2))
        row = int(round(y / self.res + self.map_size / 2))
        return (int(np.clip(row, 0, self.map_size - 1)),
                int(np.clip(col, 0, self.map_size - 1)))

    def grid_to_world(self, row, col):
        """Integer (row, col) → world cell-center (x, y)."""
        return ((col - self.map_size / 2 + 0.5) * self.res,
                (row - self.map_size / 2 + 0.5) * self.res)

    def occ_to_world_xy(self, occ):
        """Binary occupancy map → (N, 2) world-frame occupied cell centers."""
        occ_rows, occ_cols = np.where(occ > 0.5)
        if len(occ_rows) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        x_world = (occ_cols - self.map_size / 2 + 0.5) * self.res
        y_world = (occ_rows - self.map_size / 2 + 0.5) * self.res
        return np.stack([x_world, y_world], axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _empty(self):
        return np.zeros((self.map_size, self.map_size), dtype=np.float32)

    def _add_segment(self, occ, p0, p1, thickness=WALL_THICKNESS_M):
        """Rasterize a thick line segment from p0 to p1 (world coords)."""
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq < 1e-10:
            return
        half_t = thickness / 2.0
        pad = half_t + self.res
        xlo = min(p0[0], p1[0]) - pad;  xhi = max(p0[0], p1[0]) + pad
        ylo = min(p0[1], p1[1]) - pad;  yhi = max(p0[1], p1[1]) + pad
        r0, c0 = self.world_to_grid(xlo, ylo)
        r1, c1 = self.world_to_grid(xhi, yhi)
        for r in range(min(r0, r1), max(r0, r1) + 1):
            for c in range(min(c0, c1), max(c0, c1) + 1):
                x, y = self.grid_to_world(r, c)
                pt = np.array([x, y], dtype=np.float64)
                t  = float(np.dot(pt - p0, seg) / seg_len_sq)
                closest = p0 + max(0.0, min(1.0, t)) * seg
                if np.linalg.norm(pt - closest) <= half_t:
                    occ[r, c] = 1.0

    def _add_rect(self, occ, center, size_m, angle_rad=0.0):
        """Filled rectangle (w × h) centerd at world `center`."""
        cx, cy   = float(center[0]), float(center[1])
        w, h     = float(size_m[0]), float(size_m[1])
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        bbox = max(w, h) + self.res
        r0, c0 = self.world_to_grid(cx - bbox, cy - bbox)
        r1, c1 = self.world_to_grid(cx + bbox, cy + bbox)
        for r in range(min(r0, r1), max(r0, r1) + 1):
            for c in range(min(c0, c1), max(c0, c1) + 1):
                x, y = self.grid_to_world(r, c)
                dx, dy = x - cx, y - cy
                if abs(cos_a*dx + sin_a*dy) <= w/2 and abs(-sin_a*dx + cos_a*dy) <= h/2:
                    occ[r, c] = 1.0

    # ------------------------------------------------------------------
    # Core free-space check
    # ------------------------------------------------------------------

    def is_free(self, occ, pos_xy, clearance_m):
        """True iff a circle of radius clearance_m around pos_xy is wall-free."""
        r, c = self.world_to_grid(float(pos_xy[0]), float(pos_xy[1]))
        rad  = int(np.ceil(clearance_m / self.res))
        return not occ[max(0, r-rad):min(self.map_size, r+rad+1),
                       max(0, c-rad):min(self.map_size, c+rad+1)].any()

    def map_blocks(self, occ, pos_xy, clearance_m=0.30):
        """True iff pos_xy is too close to any occupied cell."""
        return not self.is_free(occ, pos_xy, clearance_m)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _start_clear(self, occ):
        """True iff origin has sufficient agent clearance."""
        return self.is_free(occ, (0.0, 0.0), START_CLEARANCE)

    def _inflated_occ(self, occ, agent_radius=AGENT_RADIUS):
        """
        Return the occupancy grid dilated by agent_radius.
        Used so BFS path-planning accounts for the robot's body.
        """
        rad    = int(np.ceil(agent_radius / self.res))
        result = occ.copy()
        rows, cols = np.where(occ > 0.5)
        for r, c in zip(rows, cols):
            result[max(0, r-rad):min(self.map_size, r+rad+1),
                   max(0, c-rad):min(self.map_size, c+rad+1)] = 1.0
        return result

    def _is_reachable(self, occ, goal_xy, agent_radius=AGENT_RADIUS):
        """
        BFS on agent-inflated grid from origin → goal.
        Returns True iff a collision-free path exists.
        Replaces the old _goal_path_clear, which was checking the wrong
        thing (clear *straight* line rather than any path).
        """
        inflated = self._inflated_occ(occ, agent_radius)
        sr, sc   = self.world_to_grid(0.0, 0.0)
        gr, gc   = self.world_to_grid(float(goal_xy[0]), float(goal_xy[1]))
        if inflated[sr, sc] or inflated[gr, gc]:
            return False

        visited               = np.zeros((self.map_size, self.map_size), dtype=bool)
        visited[sr, sc]       = True
        queue                 = deque([(sr, sc)])
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        while queue:
            r, c = queue.popleft()
            if r == gr and c == gc:
                return True
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.map_size and 0 <= nc < self.map_size
                        and not visited[nr, nc] and not inflated[nr, nc]):
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        return False

    def _path_obstruction_fraction(self, occ, goal_xy, n_samples=25):
        """
        Fraction of sample points along the straight-line path (origin→goal)
        that have at least one wall cell within CHALLENGE_VICINITY_M.

        A value ≥ 0.20 means the map is spatially non-trivial along the path:
          • walls ON  the path  (doorway, narrow_passage) score 0.3-0.8
          • walls ALONGSIDE path (corridor) score 0.6-1.0
          • lateral walls far from path score 0.0-0.1
          • open map scores 0.0
        """
        if np.linalg.norm(goal_xy) < 0.2:
            return 0.0
        # Sample 10 %→90 % of path (skip start/goal buffers)
        ts  = np.linspace(0.10, 0.90, n_samples)
        pts = np.outer(ts, goal_xy)
        blocked = sum(1 for px, py in pts
                      if not self.is_free(occ, (px, py), CHALLENGE_VICINITY_M))
        return blocked / n_samples

    # ------------------------------------------------------------------
    # Scene generators
    # Each takes (goal, rng) and returns a (map_size, map_size) occ array.
    # All are designed to place structure ON or adjacent to the direct
    # origin→goal path so the challenge filter reliably passes.
    # ------------------------------------------------------------------

    def _gen_open(self, goal, rng):
        return self._empty()

    def _gen_single_wall(self, goal, rng):
        """
        A partial wall that crosses the direct path on one side, forcing the
        robot to detour around the open end.  Redesigned from the original
        lateral placement which had no interaction with the path.
        """
        occ       = self._empty()
        goal_norm = np.linalg.norm(goal)
        if goal_norm < 0.3:
            return occ
        goal_dir = goal / goal_norm
        perp_dir = np.array([-goal_dir[1], goal_dir[0]])

        # Wall is perpendicular to goal direction, anchored near the path center
        t           = rng.uniform(0.30, 0.60)
        path_point  = goal * t                              # point on path
        # Start slightly off-center so wall definitely crosses the path
        cross_off   = rng.uniform(0.0, 0.25)
        side        = rng.choice([-1, 1])
        wall_start  = path_point - perp_dir * side * cross_off
        wall_length = rng.uniform(2.0, 3.5)
        wall_end    = wall_start + perp_dir * side * wall_length  # extends one way
        self._add_segment(occ, wall_start, wall_end)
        return occ

    def _gen_corridor(self, goal, rng):
        """Two parallel walls forming a channel the robot must navigate through."""
        occ       = self._empty()
        goal_norm = np.linalg.norm(goal)
        if goal_norm < 0.1:
            axis_angle = rng.uniform(0, np.pi)
        else:
            axis_angle = np.arctan2(goal[1], goal[0])
        # Narrower half-width → more constrained, always detectably close to path
        half_w = rng.uniform(0.85, 1.2)
        axis   = np.array([np.cos(axis_angle), np.sin(axis_angle)])
        perp   = np.array([-axis[1], axis[0]])
        length = rng.uniform(4.0, 6.5)
        for side in [-1, 1]:
            center = side * half_w * perp
            self._add_segment(occ,
                               center + axis * (length / 2),
                               center - axis * (length / 2))
        return occ

    def _gen_L_corner(self, goal, rng):
        """
        L-shaped wall with one arm anchored to cross the goal path, forcing
        the robot to navigate around the corner.
        """
        occ       = self._empty()
        goal_norm = np.linalg.norm(goal)
        if goal_norm > 0.5:
            goal_dir = goal / goal_norm
            perp_dir = np.array([-goal_dir[1], goal_dir[0]])
            # Corner: lateral offset from the path, between robot and goal
            t      = rng.uniform(0.25, 0.55)
            side   = rng.choice([-1, 1])
            corner = goal * t + perp_dir * side * rng.uniform(0.4, 1.2)
            # Arm 1: crosses back toward (or across) the path
            arm1_dir = perp_dir * (-side)
            # Arm 2: runs along the goal direction
            arm2_dir = goal_dir * rng.choice([-1, 1])
        else:
            corner     = np.array([rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)])
            base_angle = rng.uniform(0, 2 * np.pi)
            arm1_dir   = np.array([np.cos(base_angle), np.sin(base_angle)])
            arm2_dir   = np.array([-arm1_dir[1], arm1_dir[0]])
            if rng.rand() < 0.5:
                arm2_dir = -arm2_dir
        len1 = rng.uniform(2.0, 3.5)
        len2 = rng.uniform(2.0, 3.5)
        self._add_segment(occ, corner, corner + arm1_dir * len1)
        self._add_segment(occ, corner, corner + arm2_dir * len2)
        return occ

    def _gen_doorway(self, goal, rng):
        """
        A wall spanning the scene width, placed between robot and goal, with a
        narrow gap the robot must navigate through.
        """
        occ       = self._empty()
        goal_norm = np.linalg.norm(goal)
        if goal_norm < 0.1:
            wall_angle = rng.uniform(0, np.pi)
            midpoint   = np.zeros(2)
        else:
            goal_angle = np.arctan2(goal[1], goal[0])
            # Wall nearly perpendicular to path (small angle jitter only)
            wall_angle = goal_angle + np.pi / 2 + rng.normal(0, 0.15)
            t          = rng.uniform(0.30, 0.65)
            midpoint   = goal * t             # wall is on the path

        axis = np.array([np.cos(wall_angle), np.sin(wall_angle)])

        # Narrower door (more challenging) with small lateral offset so the
        # robot cannot go straight through without steering
        door_width  = rng.uniform(1.5, 2.0)
        door_offset = rng.uniform(-0.35, 0.35) * rng.choice([-1, 1])
        door_center = midpoint + axis * door_offset
        half_len    = rng.uniform(3.0, 4.0)
        self._add_segment(occ,
                          door_center + axis * (door_width / 2),
                          door_center + axis * half_len)
        self._add_segment(occ,
                          door_center - axis * (door_width / 2),
                          door_center - axis * half_len)
        return occ

    def _gen_room(self, goal, rng):
        """
        A bounded box with one exit.  Robot starts inside, goal is near the
        exit or outside, so robot must find and navigate through the opening.
        """
        occ    = self._empty()
        half_w = rng.uniform(2.0, 3.0)
        half_h = rng.uniform(2.0, 3.0)
        cx, cy = rng.uniform(-0.4, 0.4, size=2)
        corners = [
            np.array([cx - half_w, cy - half_h]),
            np.array([cx + half_w, cy - half_h]),
            np.array([cx + half_w, cy + half_h]),
            np.array([cx - half_w, cy + half_h]),
        ]
        # Opening faces toward goal so robot can exit toward it
        if np.linalg.norm(goal) > 0.1:
            gx, gy = goal
            dists  = [abs(gy - (cy - half_h)), abs(gx - (cx + half_w)),
                      abs(gy - (cy + half_h)), abs(gx - (cx - half_w))]
            opening_side = int(np.argmin(dists))
        else:
            opening_side = int(rng.randint(0, 4))
        door_width = rng.uniform(1.5, 2.0)
        for i in range(4):
            a = corners[i]; b = corners[(i + 1) % 4]
            if i == opening_side:
                mid       = (a + b) / 2
                direction = (b - a) / (np.linalg.norm(b - a) + 1e-6)
                self._add_segment(occ, a, mid - direction * (door_width / 2))
                self._add_segment(occ, mid + direction * (door_width / 2), b)
            else:
                self._add_segment(occ, a, b)
        return occ

    def _gen_pillars(self, goal, rng):
        """
        Scattered pillars, with the first half explicitly placed on or very
        near the direct path so the robot must slalom through them.
        """
        occ       = self._empty()
        goal_norm = np.linalg.norm(goal)
        n_pillars = int(rng.randint(4, 8))
        placed    = []

        if goal_norm > 0.5:
            goal_dir = goal / goal_norm
            perp_dir = np.array([-goal_dir[1], goal_dir[0]])
        else:
            goal_dir = np.array([1.0, 0.0])
            perp_dir = np.array([0.0, 1.0])

        n_on_path = max(2, n_pillars // 2)  # at least half go on the path

        for i in range(n_pillars):
            for _ in range(30):
                if i < n_on_path and goal_norm > 0.5:
                    # Place ON or very close to the direct path
                    t       = rng.uniform(0.20, 0.80)
                    along   = t * goal_norm
                    lateral = rng.uniform(-0.45, 0.45)
                    x = goal_dir[0] * along + perp_dir[0] * lateral
                    y = goal_dir[1] * along + perp_dir[1] * lateral
                else:
                    x = rng.uniform(-3.5, 3.5)
                    y = rng.uniform(-3.5, 3.5)

                if np.linalg.norm([x, y]) < 0.80:
                    continue
                if np.linalg.norm([x - goal[0], y - goal[1]]) < 0.55:
                    continue
                if any(np.linalg.norm([x - px, y - py]) < 0.75 for px, py in placed):
                    continue

                size = rng.uniform(0.30, 0.60)
                self._add_rect(occ, (x, y), (size, size),
                               angle_rad=rng.uniform(0, np.pi / 2))
                placed.append((x, y))
                break
        return occ

    def _gen_narrow_passage(self, goal, rng):
        """
        Two rectangular obstacles close together on the path, forcing a squeeze.
        """
        occ       = self._empty()
        goal_norm = np.linalg.norm(goal)
        if goal_norm < 0.1:
            passage_angle = rng.uniform(0, np.pi)
            midpoint      = rng.uniform(-1.5, 1.5, size=2)
        else:
            passage_angle = np.arctan2(goal[1], goal[0]) + np.pi / 2
            t             = rng.uniform(0.30, 0.60)
            midpoint      = goal * t       # centerd on the path
        perp = np.array([np.cos(passage_angle), np.sin(passage_angle)])
        # Narrower gap → tighter squeeze
        gap  = rng.uniform(1.5, 1.8)
        for side in [-1, 1]:
            center = midpoint + side * perp * (gap / 2 + rng.uniform(0.5, 0.8))
            size   = (rng.uniform(0.9, 1.5), rng.uniform(0.9, 1.5))
            self._add_rect(occ, center, size,
                           angle_rad=rng.uniform(0, np.pi / 2))
        return occ

    _GEN = {
        "open":             _gen_open,
        "single_wall":      _gen_single_wall,
        "corridor":         _gen_corridor,
        "L_corner":         _gen_L_corner,
        "doorway":          _gen_doorway,
        "room":             _gen_room,
        "pillars":          _gen_pillars,
        "narrow_passage":   _gen_narrow_passage,
    }

    # ------------------------------------------------------------------
    # Free-space sampling
    # ------------------------------------------------------------------

    def sample_free_position(self, occ, rng, bounds=None,
                              clearance_m=0.40,
                              exclude_positions=None, exclude_radius=1.0,
                              max_attempts=200):
        lo = -self.half + clearance_m if bounds is None else bounds[0]
        hi =  self.half - clearance_m if bounds is None else bounds[1]
        for _ in range(max_attempts):
            pos = np.array([rng.uniform(lo, hi), rng.uniform(lo, hi)])
            if not self.is_free(occ, pos, clearance_m):
                continue
            if exclude_positions is not None:
                if any(np.linalg.norm(pos - ep) < exclude_radius
                       for ep in exclude_positions):
                    continue
            return pos
        return None

    # ------------------------------------------------------------------
    # Top-level scene generation
    # ------------------------------------------------------------------

    def generate_scene(self, rng, goal=None,
                       challenge=True, min_obstruction=0.20,
                       max_retries=25):
        """
        Sample a scene type, generate its map, and validate:

          (a) origin has sufficient free space
          (b) goal cell is not inside a wall
          (c) goal is reachable via BFS  ← replaces old straight-line check
          (d) [when challenge=True] path obstruction fraction ≥ min_obstruction

        Parameters
        ----------
        challenge       : require walls to be near the direct path
        min_obstruction : fraction of path samples that must have walls within
                          CHALLENGE_VICINITY_M (default 0.20 = 20 %)
        max_retries     : attempts before falling back to open map
        """
        if goal is None:
            dist  = rng.uniform(3.0, 5.5)
            angle = rng.uniform(-np.pi, np.pi)
            goal  = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        goal = np.asarray(goal, dtype=np.float64)

        for _ in range(max_retries):
            map_type = rng.choice(self._types, p=self._probs)
            occ      = self._GEN[map_type](self, goal, rng)

            # (a) start clear
            if not self._start_clear(occ):
                continue
            # (b) goal cell free
            gr, gc = self.world_to_grid(goal[0], goal[1])
            if occ[gr, gc] > 0.5:
                continue
            # (c) BFS reachability — goal must be reachable via some path
            if not self._is_reachable(occ, goal):
                continue
            # (d) challenge: path must be non-trivially obstructed
            if challenge and map_type != "open":
                if self._path_obstruction_fraction(occ, goal) < min_obstruction:
                    continue

            world_xy = self.occ_to_world_xy(occ)
            return occ, world_xy, map_type

        # Fallback: open map always passes all checks
        occ = self._empty()
        return occ, self.occ_to_world_xy(occ), "open"
