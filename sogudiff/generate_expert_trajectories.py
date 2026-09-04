"""
Expert demonstration generator FINAL — norm-grounded social cost
=============================================================

Design principles (these ARE the justification; §IV-D of the paper):

  P1. Every social term is  mean_t max_j (severity in [0,1]).  One aggregation,
      no exceptions.  A term reads as "fraction of the plan spent in violation,
      weighted by severity", so all six share one currency and can be summed.
      A peak (max_t) aggregation was tried and is WRONG here: in a receding
      horizon the closest approach of a plan that starts near someone is fixed
      by the FIRST step, which every candidate shares and none can change, so
      max_t is a sunk cost -- it punishes closing a gap but never REWARDS
      opening one, and the robot freezes instead of keeping space.  Measured
      with a static pedestrian abeam at 1.0 m: staying put, opening the gap to
      1.53 m, and driving past to 1.61 m all scored an IDENTICAL 0.306 under
      max_t, versus 0.306 / 0.178 / 0.145 under mean_t.  mean_t is also the form
      in which the Karamouzas interaction ENERGY is integrated.  Peak safety is
      owned by the hard constraint d >= R_c, not by the social terms.
      g_cut additionally conditions its mean on the steps where its gate is
      ACTIVE (see _active_mean); g_side deliberately does not (see its comment).

  P2. Every length is a Hall (1966) zone boundary.  Every time is a length
      divided by the free-flow walking speed (Weidmann '93 / Helbing & Molnar
      '95, v_bar = 1.34 m/s).  No free geometry.

  P3. Style moves ONE scale, the Hall ladder D(s), which every axis reads once:
        s = -1  ->  R_c   = 0.50 m   contact  (== Hall intimate 0.46 m to 4 cm)
        s =  0  ->  D_per = 1.22 m   personal
        s = +1  ->  D_soc = 3.66 m   social
      log-linear each side.  rho_dn = D_per/R_c = 2.44 vs Hall's own
      D_per/D_int = 2.65 (8% apart); rho_up = D_soc/D_per = 3.00 exactly.
        prox  reads D(s) as a zone RADIUS around a point (a person)
        yield reads D(s)/v_bar as an anticipation TIME
        group reads D(s) as a zone radius around a SEGMENT (a formation)
        pass  has no scale to move: |s_pass| weights, sign(s_pass) selects side

  P4. All six social terms carry EQUAL weight.  The style vector is the only
      thing that differentiates them.  The social block as a whole is scaled by
      W_SOCIAL against J_goal (see P5) -- that is an exchange rate, not a
      per-term weight, so nothing distinguishes the six from each other.

  P5. J_goal is normalized so an ideal max-speed run costs 0 and STALLING COSTS
      EXACTLY 1.  This fixes the social/efficiency exchange rate with no knob:
      maximal violation of any single norm -- at its worst instant for the
      spatial terms, sustained for the temporal ones -- costs W_SOCIAL times as
      much as making no progress at all."  W_SOCIAL = 4 encodes "stopping is
      preferable to a maximal norm violation".  W_SOCIAL = 1 (equal) was
      measurably too weak: inside a 1.6 s / 1.6 m window ANY maneuver that
      changes the social geometry costs 0.5-0.65 of J_goal, more than a single
      social term can swing, so the straight-through candidate won at EVERY
      style and no social detour was ever affordable.

  P6. Safety is a hard constraint, never a weight.  No style can buy a
      collision.  s = -1 means the norm carries no PREFERENCE: g_prox, g_rear
      and g_group vanish identically because D(-1) = R_c and every feasible
      trajectory has d >= R_c.  Assertive, never antisocial.

  The cost has two free numbers: LAM_SM = 0.05 (legibility) and W_SOCIAL = 4.0
  (the social/efficiency exchange rate, P5).  W_SOCIAL is not free-floating --
  it encodes one statable norm, "stopping is preferable to a maximal norm
  violation" -- but it IS the number to sweep for the Pareto figure.

Terms and their axes
--------------------
  g_prox   clearance ............ prox    zone radius D(s_prox)
  g_rear   blind zone ........... prox    same zone, angular gate
  g_side   passing convention ... pass    |s_pass| weight, sign selects side
  g_ttc    temporal deference ... yield   Karamouzas '14 tau^-2, scale D(s)/v
  g_cut    spatial right-of-way . yield   forward capsule, length D(s)
  g_group  formation integrity .. group   segment zone radius D(s_group)
  g_smooth legibility ........... none    lambda = 0.05, unconditional

Fixes vs V6.5 (all load-bearing)
--------------------------------
  * seeding: zlib.crc32, not hash().  PYTHONHASHSEED randomizes per spawned
    worker, so V6.5 produced a DIFFERENT dataset on every run.
  * static feasibility: >= ROBOT_R (was >= 0.0, i.e. robot CENTER in a free
    cell), so V6.5 demonstrations clipped walls that acados then refuses --
    a direct mechanical cause of "it stops instead of detouring".
  * presort: stratified (see process_scene).  V6.5 kept the top-K by distance
    to goal, silently deleting every slowing/stopping trajectory BEFORE the
    social cost was evaluated -- i.e. exactly what the yield axis needs.
  * p(s | scene) is now uniform.  V6.5's complexity gate chose styles from the
    scene's dominant cost term, making the style label partly predictable from
    the geometry -- a conditioning confound that lets the model read style off
    the scene and ignore the token.
  * geodesic: for out-of-crop goals the progress field is built from a subgoal
    projected to the crop border along the true geodesic.  The Euclidean
    fallback (which drove straight at walls) is DELETED.  The SAVED goal is
    always the original global goal.
  * multimodality: n_modes=3 within a cost TOLERANCE (interpretable now that
    cost is normalized).  V6.5's n_modes=1 made p(tau|c,s) a delta -- an
    expensive regressor, not a distribution.
  * stopping is a legitimate demonstration, not a rejection (see run_style).
  * equal MPPI budget for all styles.  V6.5 gave non-neutral styles 25% fewer
    samples and 33% fewer iterations -- a quality asymmetry between neutral and
    styled demonstrations that biases the measured style response.
  * deleted: violation machinery, complexity gate, MAX_EXPERT_COST /
    CEILING_CONSERV_BONUS, SOCIAL_SCALE and the xH rescale, the Dijkstra
    fallback, c2_head_on, all style-dependent thresholds.
"""
from __future__ import annotations

import argparse, glob, os, time, zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import skfmm
from numba import njit, prange
from scipy.ndimage import distance_transform_edt

# The visualizer imports names from THIS module, so importing it here at module
# level is circular: it would see a half-initialized generator, fail, and print
# a spurious "generator module not importable" warning on every run of either
# script. Deferring the import to first use breaks the cycle.
_visualize_scene = None


def _vis_available():
    """True if the visualizer can be imported. The result is cached."""
    global _visualize_scene
    if _visualize_scene is None:
        try:
            from visualize_expert_trajectories import visualize_scene
            _visualize_scene = visualize_scene
        except ImportError:
            _visualize_scene = False
    return _visualize_scene is not False

# =============================================================================
# Constants
# =============================================================================
DT, HORIZON = 0.05, 32
T_H = HORIZON * DT                                   # 1.6 s plan
V_MAX, A_MAX = 1.0, 1.5
W_MAX, ALPHA_MAX = np.pi, np.pi

ROBOT_R = HUMAN_R = 0.25
R_C = ROBOT_R + HUMAN_R                              # 0.50 m contact

D_INT, D_PER, D_SOC = 0.46, 1.22, 3.66               # Hall 1966
RHO_DN = D_PER / R_C                                 # 2.44 (Hall's own: 2.65)
RHO_UP = D_SOC / D_PER                               # 3.00 (exact)

V_BAR = 1.34             # Weidmann '93 / Helbing & Molnar '95 free-flow speed
COS_PSI = np.sqrt(2) / 2   # 45 deg encounter cone -- SA-CADRL (Chen '17)
# Co-motion gate for group membership: a pair is a formation only if their
# relative speed ||v_i - v_j|| <= DV_MAX.  This is a VECTOR difference, so it
# already folds in heading (equal-speed pair at angle phi -> |dv| = 2 v sin(phi/2))
# AND speed mismatch.  At V_bar it admits ~29 deg of instantaneous heading spread
# -- room for real gait / tracking noise inside a walking group -- while a
# crossing pair (|dv| ~ 1.9) or opposing pair (~2.7) is rejected by a wide
# margin.  Grounded (P2) as half the free-flow walking speed: "moving together"
# = relative motion small vs walking.  The old D_int/T_h = 0.29 tied the gate to
# the ROBOT's plan horizon (not a property of a pedestrian group) and, at ~18 deg,
# dropped genuine walking pairs.  Standing groups (dv = 0) pass at any threshold.
DV_MAX = V_BAR / 2       # 0.67 m/s
V_MIN = 0.05             # stationary threshold (numerical)
LAM_SM = 0.05            # legibility weight

# Bounded social lookahead.  The social terms are evaluated over the plan PLUS a
# constant-velocity tail out to T_LOOK (the robot coasts at its terminal speed
# and heading; pedestrians continue at constant velocity).  Without the tail the
# cost is INVERTED for a closing encounter: braking does not resolve the
# encounter, it defers it past the window boundary, so stopping accumulates no
# exposure and scores BEST while actually producing the WORST outcome.  Measured
# on an oncoming pedestrian at prox+1 -- cost rank vs eventual clearance:
#     brake     rank 1 (best)  ->  0.15 m   (identical to the head-on collision)
#     w = -1.40 rank 2         ->  1.06 m   (actually the safest)
# i.e. the two orderings were exact reverses.  Only LATERAL offset changes the
# passing clearance, and only a window containing the whole encounter can see
# that.  T_LOOK = D_soc / v_bar keeps P2 (a Hall length over the free-flow
# speed) and lands at the Karamouzas interaction timescale (~3 s), beyond which
# pedestrian interaction is negligible AND constant-velocity prediction stops
# being trustworthy -- so the tail is bounded there, not run open-ended.
# Feasibility (the hard constraints) is deliberately NOT extended: the robot is
# only committed to the plan, never to the coasting tail.
T_LOOK = D_SOC / V_BAR                        # 2.73 s total lookahead
K_TAIL = int(round((T_LOOK - T_H) / DT))      # 23 coasting steps past the plan

# Social-vs-goal exchange rate.  J_goal is normalized so a stall costs exactly 1
# (P5); W_SOCIAL says what a maximal norm violation costs in that same currency.
# P4 is untouched: all six social terms stay EQUAL to each other, this scales the
# block as a whole.  W_SOCIAL = 1 ("a maximal violation costs the same as making
# no progress") is measurably too weak -- inside a 1.6 s / 1.6 m window every
# maneuver that changes the social geometry (detour, slow, side-shift) costs
# 0.5-0.65 of J_goal, more than any one weight-1 term can swing, so the
# straight-through candidate wins at every style.  W_SOCIAL = 4 encodes
# "stopping is preferable to a maximal norm violation", which is what makes a
# social detour affordable at all.  Sweep this for the Pareto figure.
W_SOCIAL = 4.0

# Minimum penetration band for the zone terms (g_prox, g_rear, g_group).  Their
# severity ramp is normalized by the ZONE DEPTH (D(s) - R_c), so R_c (contact)
# is the =1 bound and the ramp spans the WHOLE zone at every style -- it never
# saturates deep inside.  (A fixed band saturates once the zone is deeper than
# the band: at s=+1 a straight approach and a meter-wider swerve both clip to 1,
# and the term loses all gradient to detour on.  A d=0 reference instead gives a
# slope 1/D that COLLAPSES as the axis rises.  Zone-depth normalization avoids
# both.)  W_ZONE FLOORS that denominator so sub-neutral styles (D - R_c < W_ZONE)
# do not become an unsamplable razor wall and s=-1 stays non-singular.  It is the
# neutral Hall personal band (personal minus contact), so P2 ("no free geometry")
# holds and at neutral prox the denominator IS W_ZONE exactly.
W_ZONE = D_PER - R_C     # 0.72 m  (minimum band / neutral personal band)

PROG_MAX = V_MAX * DT * HORIZON * (HORIZON + 1) / 2  # 26.4 m-steps

MAP_SIZE, MAP_EXTENT = 50, 10
MAP_RES = MAP_EXTENT / MAP_SIZE                      # 0.2 m/cell
HALF_EXTENT = MAP_EXTENT / 2

STYLE_AXES = ["prox", "pass", "yield", "group"]
NEUTRAL = np.zeros(4, np.float32)
SOCIAL = ("g_prox", "g_rear", "g_side", "g_ttc", "g_cut", "g_group")

N_UNIFORM, N_SMOOTH = 4096, 12288
SMOOTH_TAU_A, SMOOTH_TAU_ALPHA = 6, 4
PRESORT_K, N_SEEDS = 4096, 8
MPPI_N, MPPI_ITER = 2048, 12
MPPI_SIG_A, MPPI_SIG_AL, MPPI_DECAY, MPPI_TOP = 0.8, 2.0, 0.92, 0.10


def seed_of(*parts) -> int:
    """Deterministic across spawned processes.  hash() is NOT (PYTHONHASHSEED)."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0xFFFFFFFF


def ladder(s: float) -> float:
    """Hall ladder.  s=-1 -> R_c (contact ~ intimate), 0 -> personal, +1 -> social."""
    s = float(np.clip(s, -1.0, 1.0))
    return D_PER * (RHO_DN ** s if s <= 0.0 else RHO_UP ** s)


# =============================================================================
# Map / geodesic
# =============================================================================
def _to_rc(p):
    return (int(round(p[1] / MAP_RES + MAP_SIZE / 2)),
            int(round(p[0] / MAP_RES + MAP_SIZE / 2)))


def _to_xy(r, c):
    return np.stack([(np.asarray(c) - MAP_SIZE / 2) * MAP_RES,
                     (np.asarray(r) - MAP_SIZE / 2) * MAP_RES], -1).astype(np.float32)


def _clearance_field(occ):
    """EDT of the RAW occupancy, meters.  Feasibility compares against ROBOT_R."""
    b = occ > 0.5
    if not b.any():
        return np.full(occ.shape, np.inf, np.float32)
    return (distance_transform_edt(~b) * MAP_RES).astype(np.float32)


def _fmm(blocked, rc):
    phi = np.ones(blocked.shape)
    phi[rc] = 0.0
    d = skfmm.distance(np.ma.MaskedArray(phi, mask=blocked), dx=MAP_RES)
    out = np.asarray(d, dtype=np.float32)
    out[blocked] = np.inf
    if np.ma.is_masked(d):
        out[np.ma.getmaskarray(d)] = np.inf
    return out


def _ped_spacetime_blocked(obs, base_blocked, start_rc):
    """Space-time footprint of the humans, for the PROGRESS FIELD only.

    A cell is blocked iff some human is within R_C of it AT THE TIME THE ROBOT
    COULD FIRST GET THERE.  Arrival time is estimated from the human-free
    geodesic out of the robot's cell divided by V_MAX -- the earliest the robot
    could possibly arrive -- and each human is propagated to that time at
    constant velocity.  This is what makes the treatment correct for MOVING
    people and not just standing ones: a cell a walker has already vacated by
    the time the robot reaches it is NOT blocked (so passing behind someone
    stays free), while the cell they will be occupying IS.  Blocking the t=0
    snapshot instead would falsely reject perfectly good trajectories through
    space the walker has long left.

    R_C = ROBOT_R + HUMAN_R exactly -- the same radius the hard feasibility
    constraint uses, i.e. the physical footprint and nothing more.  No social
    zone D(s) enters here, so the progress field never encodes style and never
    double-counts the social terms, which own only the clearance BEYOND R_C.
    The purpose is narrow: give J_goal a gradient that goes AROUND a person
    instead of straight through them.  Without it the geodesic points through
    the human, the social term pushes back, and the robot sits at the standoff
    and times out.

    Cells the robot cannot reach inside T_LOOK are left unblocked -- past that
    horizon the constant-velocity prediction is not trustworthy (the same bound
    the social lookahead uses).
    """
    if obs is None or len(obs) == 0:
        return None
    o = np.asarray(obs, np.float32).reshape(-1, 4)
    if not o.size:
        return None
    d_r = _fmm(base_blocked, start_rc)              # meters, inf where unreachable
    t_arr = d_r / max(V_MAX, 1e-6)                  # earliest possible arrival, s
    valid = np.isfinite(t_arr) & (t_arr <= T_LOOK)
    t_safe = np.where(valid, t_arr, 0.0).astype(np.float32)
    rr, cc = np.mgrid[0:MAP_SIZE, 0:MAP_SIZE]
    xy = _to_xy(rr, cc)                             # (S,S,2)
    m = np.zeros((MAP_SIZE, MAP_SIZE), bool)
    for q, vq in zip(o[:, :2], o[:, 2:4]):
        pq = q[None, None, :] + vq[None, None, :] * t_safe[..., None]
        m |= valid & (np.linalg.norm(xy - pq, axis=-1) < R_C)
    return m


def build_geodesic(occ, goal, has_map, obs=None):
    """
    Returns (subgoal_xy, geo_field, clearance_field) or (None, None, clear).

    With `obs`, humans are treated as physical blockers at their SPACE-TIME
    footprint (see _ped_spacetime_blocked) so the progress gradient routes
    around them.  If that disconnects the goal (someone is standing in the only
    gap) we fall back to the human-free field rather than dropping the scene --
    the social terms and the hard constraint still apply there.

    The geodesic field respects obstacles, so a goal reachable ONLY via a long
    detour IS reachable and the field's gradient points along the detour.  We
    reject only when no path exists at all (the goal is in a disconnected
    component).  This is a connectivity test, not a line-of-sight test.

    The robot is a LOCAL planner (1.6 s / ~1.6 m of trajectory).  For a goal
    outside the +-5 m crop we cannot know the path beyond the border, so the
    progress field is built from the reachable border cell minimizing
        (geodesic robot -> border) + (straight line border -> goal),
    i.e. exactly the exit point a global planner would produce given only this
    crop.  The SAVED goal is always the original global goal -- the subgoal is
    internal to the progress term.

    With no map, `blocked` is empty and FMM degenerates to Euclidean, so the
    no-map case needs no special handling.
    """
    clear = (_clearance_field(occ) if (has_map > 0.5 and occ.any())
             else np.full((MAP_SIZE, MAP_SIZE), np.inf, np.float32))
    base = clear < ROBOT_R
    sr, sc = _to_rc((0.0, 0.0))
    if base[sr, sc]:
        return None, None, clear

    def _solve(blocked):
        """Progress field for one blocked-mask; None if the goal is unreachable."""
        if blocked[sr, sc]:
            return None
        if abs(goal[0]) < HALF_EXTENT and abs(goal[1]) < HALF_EXTENT:
            gr, gc = _to_rc(goal)
            gr = int(np.clip(gr, 0, MAP_SIZE - 1))
            gc = int(np.clip(gc, 0, MAP_SIZE - 1))
            if blocked[gr, gc]:
                return None
            f = _fmm(blocked, (gr, gc))
            return (np.asarray(goal, np.float32), f) if np.isfinite(f[sr, sc]) else None

        geo_r = _fmm(blocked, (sr, sc))
        border = np.zeros((MAP_SIZE, MAP_SIZE), bool)
        border[0], border[-1], border[:, 0], border[:, -1] = True, True, True, True
        cand = border & ~blocked & np.isfinite(geo_r)
        if not cand.any():
            return None
        rr, cc = np.where(cand)
        xy = _to_xy(rr, cc)
        k = int(np.argmin(geo_r[rr, cc] +
                          np.linalg.norm(xy - np.asarray(goal), axis=1)))
        f = _fmm(blocked, (int(rr[k]), int(cc[k])))
        return (xy[k], f) if np.isfinite(f[sr, sc]) else None

    ped = _ped_spacetime_blocked(obs, base, (sr, sc))
    if ped is not None:
        ped[sr, sc] = False                 # never block the robot's own cell
        r = _solve(base | ped)
        if r is not None:
            return r[0], r[1], clear        # human-aware field (preferred)
    r = _solve(base)                        # fallback: human-free
    return (r[0], r[1], clear) if r is not None else (None, None, clear)


def sample_geo(xy, geo):
    cf = xy[..., 0] / MAP_RES + MAP_SIZE / 2
    rf = xy[..., 1] / MAP_RES + MAP_SIZE / 2
    inb = (rf >= 0) & (rf <= MAP_SIZE - 1) & (cf >= 0) & (cf <= MAP_SIZE - 1)
    r0 = np.clip(np.floor(rf).astype(np.int32), 0, MAP_SIZE - 1)
    c0 = np.clip(np.floor(cf).astype(np.int32), 0, MAP_SIZE - 1)
    r1, c1 = np.minimum(r0 + 1, MAP_SIZE - 1), np.minimum(c0 + 1, MAP_SIZE - 1)
    fr, fc = (rf - r0).astype(np.float32), (cf - c0).astype(np.float32)
    v00, v01, v10, v11 = geo[r0, c0], geo[r0, c1], geo[r1, c0], geo[r1, c1]
    bad = ~(np.isfinite(v00) & np.isfinite(v01) & np.isfinite(v10) & np.isfinite(v11))
    with np.errstate(invalid="ignore"):
        out = (v00 * (1 - fr) * (1 - fc) + v01 * (1 - fr) * fc +
               v10 * fr * (1 - fc) + v11 * fr * fc)
    return np.where(bad | ~inb, np.inf, out).astype(np.float32)


def map_clearances(xy, clear):
    if not np.isfinite(clear).any():
        return np.full(xy.shape[:-1], np.inf, np.float32)
    c = np.round(xy[..., 0] / MAP_RES + MAP_SIZE / 2).astype(np.int32)
    r = np.round(xy[..., 1] / MAP_RES + MAP_SIZE / 2).astype(np.int32)
    inb = (r >= 0) & (r < MAP_SIZE) & (c >= 0) & (c < MAP_SIZE)
    v = clear[np.clip(r, 0, MAP_SIZE - 1), np.clip(c, 0, MAP_SIZE - 1)]
    return np.where(inb, v, np.inf).astype(np.float32)


# =============================================================================
# Dynamics
# =============================================================================
_PI, _2PI = np.float32(np.pi), np.float32(2 * np.pi)
_AM, _VM, _WM, _ALM = map(np.float32, (A_MAX, V_MAX, W_MAX, ALPHA_MAX))


@njit(parallel=True, fastmath=True, cache=True)
def _rollout(x0, u, dt):
    N, H, _ = u.shape
    st = np.empty((N, H + 1, 5), dtype=np.float32)
    for i in prange(N):
        x, y, th, v, w = x0[0], x0[1], x0[2], x0[3], x0[4]
        st[i, 0, 0], st[i, 0, 1], st[i, 0, 2] = x, y, th
        st[i, 0, 3], st[i, 0, 4] = v, w
        for t in range(H):
            a = min(max(u[i, t, 0], -_AM), _AM)
            al = min(max(u[i, t, 1], -_ALM), _ALM)
            vn = min(max(v + a * dt, np.float32(0.0)), _VM)
            wn = min(max(w + al * dt, -_WM), _WM)
            vm = np.float32(0.5) * (v + vn)
            tm = th + np.float32(0.5) * wn * dt
            x += vm * np.cos(tm) * dt
            y += vm * np.sin(tm) * dt
            th = (th + wn * dt + _PI) % _2PI - _PI
            v, w = vn, wn
            st[i, t + 1, 0], st[i, t + 1, 1], st[i, t + 1, 2] = x, y, th
            st[i, t + 1, 3], st[i, t + 1, 4] = v, w
    return st


def rollout(x0, u, dt=DT):
    sq = u.ndim == 2
    r = _rollout(x0.astype(np.float32), (u[None] if sq else u).astype(np.float32),
                 np.float32(dt))
    return r[0] if sq else r


def obs_positions(obs, horizon=HORIZON, dt=DT):
    if obs is None or len(obs) == 0:
        return np.zeros((horizon + 1, 0, 2), np.float32)
    o = np.asarray(obs, np.float32)
    t = np.arange(horizon + 1, dtype=np.float32)[:, None, None] * dt
    return (o[None, :, 0:2] + o[None, :, 2:4] * t).astype(np.float32)


def brake_controls():
    """Decelerate at the limit, then hold.  Standing still is a legitimate
    yield, so this must survive the presort (see process_scene)."""
    u = np.zeros((HORIZON, 2), np.float32)
    u[:4, 0] = -A_MAX
    return u


# =============================================================================
# Social geometry -- style-INVARIANT, computed once per (scene, pool)
# =============================================================================
def geometry(traj, obs_traj, obs_vel):
    """
    Everything that does not depend on s.  Formation membership is a threshold
    on sep0, applied in terms(), so the candidate pair list is style-invariant
    too -- every style in a scene shares one geometry pass.
    """
    N, Hp1, _ = traj.shape
    H, M = Hp1 - 1, obs_traj.shape[1]

    # g_smooth: normalized jerk + angular jerk.  a_t = dv/dt and alpha_t = dw/dt
    # come from the rolled-out states; the largest change either can undergo in
    # one step is 2*A_MAX / 2*ALPHA_MAX, so each is in [0,1] by construction.
    v, w = traj[:, :, 3], traj[:, :, 4]
    a, al = np.diff(v, axis=1) / DT, np.diff(w, axis=1) / DT
    JERK_MAX  = 2 * A_MAX / DT          # 60
    AJERK_MAX = 2 * ALPHA_MAX / DT      # ~125
    g_sm = (0.5*np.clip(np.abs(np.diff(a,axis=1))/JERK_MAX, 0,1).mean(1) +
            0.5*np.clip(np.abs(np.diff(al,axis=1))/AJERK_MAX,0,1).mean(1))
    G = {"N": N, "H": H, "M": M, "g_sm": g_sm.astype(np.float32)}
    if M == 0:
        return G

    # ---- bounded social lookahead (see T_LOOK) ------------------------------
    # Steps 1..H are the PLAN; the next K_TAIL steps are a constant-velocity
    # coast used only to let the encounter finish inside the evaluation window.
    # g_smooth above is computed on the plan alone -- the tail carries no
    # controls -- and feasibility in total_cost likewise stays plan-only, since
    # the robot is not committed to the coast.
    rob = traj[:, 1:, 0:2]
    th, sp = traj[:, 1:, 2], traj[:, 1:, 3]
    hum = obs_traj[1:H + 1].astype(np.float32)                          # (H,M,2)
    if K_TAIL > 0:
        k = (np.arange(1, K_TAIL + 1, dtype=np.float32) * DT)           # (K,)
        thT, vT = traj[:, -1, 2], traj[:, -1, 3]                        # terminal
        dirT = np.stack([np.cos(thT), np.sin(thT)], -1).astype(np.float32)
        # robot coasts: heading held, zero angular rate (a spiral would be a
        # fiction; straight-line coast is the conservative continuation)
        rob = np.concatenate(
            [rob, traj[:, -1:, 0:2] + (vT[:, None] * k[None])[..., None] * dirT[:, None]],
            axis=1)
        th = np.concatenate([th, np.repeat(thT[:, None], K_TAIL, 1)], axis=1)
        sp = np.concatenate([sp, np.repeat(vT[:, None], K_TAIL, 1)], axis=1)
        # pedestrians continue at constant velocity from their last plan position
        hum = np.concatenate(
            [hum, obs_traj[H][None] + k[:, None, None] * obs_vel[None]], axis=0)
    rd = np.stack([np.cos(th), np.sin(th)], -1).astype(np.float32)      # robot heading
    vr = (sp[..., None] * rd).astype(np.float32)                        # robot velocity
    hum = hum.astype(np.float32)                                        # (H+K,M,2)
    vj = obs_vel.astype(np.float32)
    hs = np.linalg.norm(vj, axis=1)
    mov = hs > V_MIN
    hd = (vj / np.maximum(hs, 1e-6)[:, None]).astype(np.float32)        # garbage if !mov
    hl = np.stack([-hd[:, 1], hd[:, 0]], -1).astype(np.float32)         # left normal

    rel = (rob[:, :, None, :] - hum[None]).astype(np.float32)           # p_r - p_j
    d = np.linalg.norm(rel, axis=3).clip(min=1e-4).astype(np.float32)
    vrel = (vr[:, :, None, :] - vj[None, None]).astype(np.float32)
    movf = mov.astype(np.float32)[None, None]
    G["d"], G["hs"], G["movf"] = d, hs.astype(np.float32), movf

    # b_rear: 1 directly astern, 0 at the flank and ahead.  Purely radial x
    # angular -- no lateral component -- which is why keeping this on the prox
    # axis cannot compete with s_pass.
    G["b_rear"] = np.clip(-(rel * hd[None, None]).sum(-1) / d, 0, 1) * movf

    # sigma: signed lateral offset in the PEDESTRIAN's frame (>0 = robot on
    # their left).  This frame is why one formula covers head-on and overtake.
    sig = (rel * hl[None, None]).sum(-1)
    G["sig"] = sig

    # side_gate: "is this an encounter where a side convention applies?"  The
    # head-on / overtake classification is made ONCE from the INITIAL heading and
    # then HELD for the whole encounter.  It must not be recomputed per step from
    # the robot's current heading, and must not require `closing`, because both
    # let a candidate switch the term OFF by doing the very thing being judged:
    # turning to commit to a side rotates the robot ~62 deg in 1.2 s, dropping
    # |cos psi| under the threshold.  Measured on a head-on pass: the old gate
    # was ON for steps 0-24 while |sigma| <= 0.17 (no side information yet) and
    # OFF by the actual closest approach at step 37, where |sigma| reaches 1.43 --
    # i.e. it evaluated the convention entirely during the ambiguous approach and
    # switched itself off before the pass it exists to judge.  Holding the
    # classification raises the left/right differential from 0.034 to 0.525 (15x)
    # and closes the loophole.  The initial heading is shared by every candidate
    # (they all start from x0), so this is a scene property, not a per-candidate
    # one, and cannot be gamed.
    rd0 = np.stack([np.cos(traj[:, 0, 2]), np.sin(traj[:, 0, 2])], -1).astype(np.float32)
    aligned0 = np.abs((rd0[:, None, :] * hd[None]).sum(-1)) >= COS_PSI   # (N,M)
    G["side_gate"] = ((d < D_SOC) & mov[None, None] &
                      aligned0[:, None, :]).astype(np.float32)

    G["s_par"] = (rel * hd[None, None]).sum(-1)                         # lead along path
    G["lat"] = np.clip(1 - np.abs(sig) / D_PER, 0, 1).astype(np.float32)
    G["block"] = np.clip(1 - (vr[:, :, None, :] * hd[None, None]).sum(-1) /
                         np.maximum(hs, 1e-6)[None, None], 0, 1).astype(np.float32)

    # tau: exact time until ||p_j - p_r|| = R_c under constant relative
    # velocity.  With w = p_j - p_r and u = v_j - v_r,
    #   ||w + u t||^2 = R_c^2  ->  A t^2 + B t + C = 0
    #   A = |v_rel|^2,  B = 2 (rel . v_rel),  C = d^2 - R_c^2.
    # C <= 0: already touching -> tau = 0.  (V6.5 returned the EXIT time here,
    # scoring the worst possible violation as perfectly safe.)
    # B >= 0: separating -> both roots negative -> tau = inf.
    A = (vrel ** 2).sum(-1)
    B = 2.0 * (rel * vrel).sum(-1)
    C = d ** 2 - R_C ** 2
    disc = B ** 2 - 4 * A * C
    ok = (A > 1e-8) & (B < 0) & (disc >= 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = np.where(ok, (-B - np.sqrt(np.maximum(disc, 0))) /
                       (2 * np.maximum(A, 1e-8)), np.inf)
    G["tau"] = np.where(C <= 0, 0.0, tau).astype(np.float32)

    # Formations.  The ONLY style-invariant physical fact is co-motion:
    # ||v_i - v_j|| <= DV_MAX (half the free-flow walking speed; see the DV_MAX
    # definition for the heading/speed reading).  This covers walking groups
    # (small Delta v) and standing groups (Delta v = 0) with no orientation --
    # so obs_dim = 4 is sufficient by construction and no F-formation detector
    # is needed or claimed.  Candidates are gated at D_soc, the widest possible
    # membership radius; the style-dependent radius is applied in terms().
    if M >= 2:
        p0 = obs_traj[0]
        sep0 = np.linalg.norm(p0[:, None] - p0[None], axis=2)
        dv = np.linalg.norm(vj[:, None] - vj[None], axis=2)
        pair = (sep0 < D_SOC) & (dv <= DV_MAX) & np.triu(np.ones((M, M), bool), 1)
        ii, jj = np.where(pair)
        if ii.size:
            seg = (hum[:, jj] - hum[:, ii]).astype(np.float32)          # (H,P,2)
            L2 = np.maximum((seg ** 2).sum(-1), 1e-6)
            ri = (rob[:, :, None, :] - hum[:, ii][None]).astype(np.float32)
            # Distance to the SEGMENT (capsule spine), clamped to the endpoints
            # so the zone is the members' connecting bar, not an infinite line.
            u = np.clip((ri * seg[None]).sum(-1) / L2[None], 0.0, 1.0)
            G["useg"] = u.astype(np.float32)   # where along the bar we project
            G["dseg"] = np.linalg.norm(ri - u[..., None] * seg[None], axis=3).astype(np.float32)
            G["sep0"] = sep0[ii, jj].astype(np.float32)                 # (P,)
    
    
        if ii.size and os.environ.get("LOG_GROUPS"):
            print(f"[grp] pairs={ii.size} sep0={np.round(sep0[ii,jj],2).tolist()} "
                f"(D_grp: neutral={D_PER:.2f}, +1={D_SOC:.2f})", flush=True)
            
    return G


def _active_mean(sev, active):
    """Mean of per-step severity over the steps where the term is ACTIVE.

    The GATED terms (g_side, g_cut) fire only during an encounter.  Averaging
    them over all H steps makes a term's attainable value scale with the
    fraction of the plan the encounter happens to occupy -- an artifact of where
    the 1.6 s window falls relative to the encounter, not of the behavior.  A
    gate active for 25% of the window caps the term at 0.25 no matter how badly
    the norm is violated, which is why g_side could never compete for a side.
    Conditioning on the active steps restores a true [0,1] reading of "how badly
    did you violate this DURING the encounter".  Zero when never active.
    """
    n = active.sum(1)
    return np.where(n > 0, sev.sum(1) / np.maximum(n, 1.0), 0.0).astype(np.float32)


def terms(G, s):
    """
    The six social terms.  P1: each is mean_t max_j (severity in [0,1]).
    P3: each reads the Hall ladder exactly once.  P4: all equally weighted.
    """
    N, M = G["N"], G["M"]
    out = {"g_smooth": G["g_sm"]}
    if M == 0:
        for k in SOCIAL:
            out[k] = np.zeros(N, np.float32)
        return out

    D_prox = ladder(s[0])            # prox  : zone radius around a person
    tau_a = ladder(s[2]) / V_BAR     # yield : anticipation time
    D_grp = ladder(s[3])             # group : zone radius around a formation

    # ---- g_prox  (prox) -----------------------------------------------------
    # Penetration of the proxemic zone, normalized by the zone DEPTH so R_c
    # (contact) is the =1 bound:
    #     clip( (D_prox - d) / max(D_prox - R_c, W_ZONE) )
    # 0 at the wall d = D_prox, 1 at contact d = R_c, linear between.  Two simpler
    # forms each fail at one end: a d = 0 reference gives slope 1/D_prox that
    # COLLAPSES as the axis rises (0.82/m at s=0 -> 0.27/m at +1), so higher style
    # widens the zone but softens the wall; a FIXED band gives slope 1/W_ZONE that
    # SATURATES once the zone is deeper than the band (at s=+1 a straight approach
    # and a meter-wider swerve both clip to 1 -- no gradient to detour on).
    # Normalizing by the zone depth (D - R_c) keeps a graded ramp across the WHOLE
    # zone at every style, so widening clearance ALWAYS lowers cost; W_ZONE only
    # floors the denominator near s=-1 (keeping it non-singular there).  Vanishes
    # IDENTICALLY at s=-1 (D_prox = R_c, every feasible trajectory has d >= R_c:
    # P6).  Nothing here is reduced by stopping -- only by distance -- which
    # removes V6.5's stall attractor (c4 = speed x proximity, minimized by v = 0).
    # Aggregation (P1): worst neighbor (max_j), then MEAN over time.  mean_t --
    # not max_t.  A peak measure is fatally wrong here in a receding-horizon
    # setting: the closest approach of a plan that starts near someone is set by
    # the FIRST step, which every candidate shares and none can change, so it is
    # a sunk cost.  Measured with a static pedestrian abeam at 1.0 m: staying
    # put, opening the gap to 1.53 m, and driving past to 1.61 m all score
    # max_t = 0.306 -- IDENTICAL.  max_t punishes closing the gap but never
    # rewards widening it, so "stop and don't make it worse" becomes optimal and
    # the robot freezes instead of keeping space.  mean_t ranks them 0.306 /
    # 0.178 / 0.145, i.e. it actively rewards moving away, which is the whole
    # point of the term.  The transient-pass dilution that motivated trying
    # max_t is instead handled where it belongs -- the zone-depth severity (no
    # saturation, no gradient collapse) and W_SOCIAL (the exchange rate).
    zone = np.clip((D_prox - G["d"]) / max(D_prox - R_C, W_ZONE), 0, 1)
    out["g_prox"] = zone.max(2).mean(1)

    # ---- g_rear  (prox) -----------------------------------------------------
    # Visibility / blind-spot term: the same proxemic zone, gated by how
    # directly astern the robot is.  g_rear <= g_prox pointwise; the overlap is
    # deliberate -- together they ARE an anisotropic personal space, and being
    # 1 m behind someone is worse than being 1 m beside them.  No closing gate:
    # occupying a blind spot is rude whether or not you are approaching.
    # Moving pedestrians only -- a stationary person has an orientation, but we
    # do not observe it (obs_dim = 4), and the cost must not use what the
    # network cannot see.  Stated as a limitation in the paper.
    out["g_rear"] = (G["b_rear"] * zone).max(2).mean(1)  # mean_t, as g_prox

    # ---- g_side  (pass) -----------------------------------------------------
    # Pass is the one axis with no scale to move: for a proxemic zone the
    # neutral stance is a DISTANCE, but for a CONVENTION the neutral stance is
    # NO convention.  So |s_pass| is the weight and sign(s_pass) picks the side
    # -- continuous through zero, and PSB(0) = 0 by construction.
    #
    # Hinge over sigma in [-D_per, +D_per], sigma > 0 = robot on the ped's left:
    #     sigma = +D_per -> 0     correct side, one personal distance clear
    #     sigma =  0     -> 0.5   on their line: maximum ambiguity
    #     sigma = -D_per -> 1     wrong side, committed (saturates out to D_soc)
    # The residual on the correct side is deliberate: a pure sign test has zero
    # gradient for any sigma >= 0, so the robot would graze them and leave
    # clearance to g_prox.  The hinge instead pulls toward an early, committed,
    # legible pass.  Saturation at +D_per means no reward for going wider --
    # g_prox owns clearance (radial), g_side owns side (lateral, signed).
    #
    # Under right-hand traffic the robot ends on the pedestrian's LEFT in a
    # head-on pass AND in an overtake: no sign flip, one formula.  SA-CADRL's
    # own reward confirms this -- its two sets carry opposite p~y signs
    # (S_pass: -2 < p~y < 0; S_ovtk: 0 < p~y < 1) precisely because they are
    # written in the ROBOT's frame.
    #
    # Gate (in geometry()): social zone (d < D_soc; SA-CADRL's own 1-4 m and
    # 0-3 m ranges sit inside it), |cos psi| >= cos 45 deg -- opposing OR
    # aligned, matching SA-CADRL's |phi-psi| > 3pi/4 and < pi/4 partition.  A
    # 90 deg crossing has NO side convention: it is a right-of-way problem and
    # falls through to g_cut on the yield axis.  Moving only (a stationary
    # person has no heading, hence no convention; g_prox covers them), and
    # closing.
    if abs(float(s[1])) < 1e-6:
        out["g_side"] = np.zeros(N, np.float32)
    else:
        # Hinge over sigma in [-R_c, +R_c], NOT [-D_per, +D_per].  R_c is the
        # lateral offset at which the robot physically CLEARS the pedestrian, so
        # it is the natural scale for "committed to a side": the convention is
        # resolved once you are unambiguously past them on one side, and going
        # wider than that is CLEARANCE, which g_prox owns.  The D_per span
        # (2.44 m) was far wider than the sigma a 1.6 s plan can actually produce
        # (~0.3 m), so the left/right differential collapsed to 0.014 -- below
        # the noise from every other term, which is why the pass side flipped at
        # random even in a simple head-on.  R_c raises it to 0.034 (x2.4).
        # PSB(0) = 0.5 is preserved, so behavior through sigma = 0 is unchanged.
        hinge = np.clip((R_C - np.sign(s[1]) * G["sig"]) / (2 * R_C), 0, 1)
        # NOTE mean over the WHOLE plan, deliberately NOT _active_mean (unlike
        # g_cut).  The hinge is 0.5 at sigma = 0 by design -- "on their line,
        # undecided" -- and that residual is only meaningful as a GRADIENT
        # pulling toward a side.  Conditioning on active steps promotes it into a
        # large, nearly SIDE-INDEPENDENT constant (a brief encounter went from
        # 0.5/32 = 0.016 to 0.5), so every short encounter paid a big offset that
        # carried no side information and noise chose the side -- observed as an
        # inconsistent, sometimes-wrong pass side.  g_cut has no such residual
        # (its severity is genuinely 0 when not violating), so it keeps
        # _active_mean.
        out["g_side"] = abs(float(s[1])) * (G["side_gate"] * hinge).max(2).mean(1)

    # ---- g_ttc  (yield) -----------------------------------------------------
    # Severity = (1 + tau/tau_a)^-2.
    #  * Karamouzas et al. (PRL 2014) MEASURED the pedestrian interaction energy
    #    to fall as tau^-2 (R^2 = 0.92-0.94 across two very different datasets).
    #    The exponent is theirs, not a choice.
    #  * The +1 regularizes tau -> 0; their data saturates below ~200 ms, the
    #    human reaction time.  A hard clip at tau_a instead would be flat for
    #    all tau < tau_a -- a 2.7 s dead zone at s = +1 across exactly the
    #    encounters the axis exists to shape, and no gradient for MPPI.
    #  * tau_a(s) = D(s)/v_bar is NOT a second ladder.  Karamouzas's own
    #    screening relation gives the interaction time scale as the
    #    nearest-neighbor distance over the mean walking speed, and Hall's
    #    D_per IS a nearest-neighbor distance.  tau_a = 0.37/0.91/2.73 s at
    #    s = -1/0/+1: the upper end lands within 9% of their independently
    #    estimated intrinsic range tau_0 ~ 3 s, the lower end stays above their
    #    ~200 ms reaction floor.  Two literatures, one number.
    #  * Unlike prox/rear/group this does not vanish at s = -1: the scale is a
    #    HORIZON, not a magnitude.  s_yield sets how EARLY you react, not
    #    whether.  A residual at imminent contact is correct at every style.
    with np.errstate(invalid="ignore", divide="ignore"):
        ttc = 1.0 / (1.0 + G["tau"] / tau_a) ** 2
    out["g_ttc"] = np.nan_to_num(ttc, nan=0.0, posinf=0.0).max(2).mean(1)

    # ---- g_cut  (yield) -----------------------------------------------------
    # The pedestrian's forward space = their personal zone extruded along their
    # path by one anticipation time: a capsule of half-width D_per and length
    # L_j = ||v_j|| * tau_a(s).  At the free-flow speed the capsule is exactly
    # D(s) long -- the personal zone swept forward by one style-scaled zone.
    # One parameter (tau_a) therefore drives BOTH yield terms, temporal
    # collision and spatial priority, making them complementary rather than the
    # redundant pair V6.5 had (TTC and PCR measured the same thing).
    # Three zero-cost escapes, all correct: swerve out laterally (lat -> 0),
    # drop behind (s_par < 0), or out-run them (block -> 0).
    # blocking = 1 - (v_r . d_j)/||v_j|| is MAXIMAL when the robot is stationary
    # in their path, so standing still is not an escape from this term.
    # Head-on needs no separate term: it fires g_ttc AND g_cut, and the escapes
    # are exactly "swerve or slow".  V6.5's c2_head_on is deleted.
    L = G["hs"][None, None] * tau_a
    inside = ((G["s_par"] > 0) & (G["s_par"] < L)).astype(np.float32)
    out["g_cut"] = _active_mean((inside * G["lat"] * G["block"] * G["movf"]).max(2),
                                (inside * G["movf"]).max(2) > 0)

    # ---- g_group  (group) ---------------------------------------------------
    # Formation = a co-moving pair the robot regards as together.  The pair is
    # treated as ONE extended obstacle: the segment between the members,
    # inflated by D(s_group).  Severity = penetration of that capsule by the same
    # zone-depth rule as g_prox,
    #     clip((D_grp - d_seg) / max(D_grp - R_c, W_ZONE))
    # exactly as g_prox is the D(s)-dilation of a POINT, this is the D(s)-
    # dilation of the SEGMENT, read through the identical severity so the group
    # wall is graded across the whole zone at every style -- no gradient collapse
    # at the edge, and crucially NO saturation deep inside.  The latter is what
    # made group=+1 useless: with a fixed band the zone (3.66 m) is far deeper
    # than the band (0.72 m), so a straight approach and a meter-wider swerve both
    # clipped to 1 and the term had no gradient to detour on.  It fires when the
    # robot approaches the shared
    # space from ANY angle -- which is what "respect the formation" means --
    # rather than only when it threads between the members (which for a tight
    # pair the robot physically cannot do: it needs 2*R_c = 1.0 m of gap, so the
    # old interior gate 4u(1-u) was near-unsatisfiable and the term was dead).
    # Membership scales too: at s=-1, D_grp=R_c, only touching pairs the robot
    # cannot split anyway, so the term vanishes (P6); at s=+1, D_grp=D_soc, loose
    # and multi-person formations qualify (via chained pairs) and get a wide
    # shared zone, so the robot detours around the whole group.
    # Separates from g_prox: prox is per-person and radial; this is proximity to
    # the connecting BAR.  Near the middle of a wide pair only this fires (you
    # are >D_per from both people but inside their shared space).
    if "dseg" in G:
        # INTERIOR weight 4u(1-u): 1 at the middle of the connecting bar, 0 at the
        # members themselves.  Without it the term has NO lateral gradient and so
        # cannot express "don't split": for a line-abreast group approached
        # head-on, distance-to-segment is dominated by the approach gap and is
        # IDENTICAL whether the robot aims at the middle (where it would cut the
        # formation) or at the edge (where it would round it) -- measured flat at
        # 2.80 m across the whole span.  The robot therefore had no reason to
        # prefer going around, and went straight, i.e. through.  Weighting by the
        # projection restores exactly that signal, and generalizes: a robot
        # between two members of ANY qualifying pair projects near u=0.5 and is
        # penalized; one beyond the outermost member projects to u=0 or 1 and is
        # not.  Clearance to the end members stays owned by g_prox (radial,
        # per-person) -- this term owns formation INTEGRITY only.
        member = (G["sep0"] < D_grp).astype(np.float32)[None, None]     # (1,1,P)
        interior = 4.0 * G["useg"] * (1.0 - G["useg"])
        pen = np.clip((D_grp - G["dseg"]) / max(D_grp - R_C, W_ZONE), 0, 1)
        out["g_group"] = (member * interior * pen).max(2).mean(1)  # mean_t, as g_prox
    else:
        out["g_group"] = np.zeros(N, np.float32)
    return out


def total_cost(traj, obs_traj, obs_vel, geo, clear, s, G=None, breakdown=False):
    """
    J = J_goal + W_SOCIAL * sum(six equally-weighted social terms)
        + LAM_SM * g_smooth

    J_goal = 1 - sum_t (delta_0 - delta_t) / (v_max dt T(T+1)/2), delta =
    geodesic distance to the subgoal.  = 0 for an ideal max-speed run along the
    geodesic, = 1 for stalling, = 2 for retreating at max speed; provably in
    [0, 2] since geodesic distance is 1-Lipschitz.  delta is the GEODESIC, so a
    detour around a STATIC obstacle is already free (progress is measured along
    the path that goes around it); only a detour around PEOPLE has to be bought,
    and W_SOCIAL is the exchange rate that buys it (P5).

    Hard constraints (P6, style-independent, matching the acados layer):
    d >= R_c to every pedestrian, map clearance >= ROBOT_R.  Infeasible -> +inf.
    """
    single = traj.ndim == 2
    if single:
        traj = traj[None]
    xy = traj[:, :, 0:2]

    gs = sample_geo(xy, geo)
    feas_geo = np.isfinite(gs).all(axis=1)
    gsf = np.where(np.isfinite(gs), gs, np.float32(1e6))
    J_goal = (1.0 - (gsf[:, 0:1] - gsf[:, 1:]).sum(1) / PROG_MAX).astype(np.float32)

    nxt = xy[:, 1:]
    if obs_traj.shape[1] > 0:
        dmin = np.linalg.norm(nxt[:, :, None, :] - obs_traj[1:HORIZON + 1][None],
                              axis=3).min((1, 2))
    else:
        dmin = np.full(traj.shape[0], np.inf, np.float32)
    feas = feas_geo & (dmin >= R_C) & (map_clearances(nxt, clear).min(1) >= ROBOT_R)

    G = geometry(traj, obs_traj, obs_vel) if G is None else G
    tm = terms(G, s)
    social = sum(tm[k] for k in SOCIAL)
    total = np.where(feas, J_goal + W_SOCIAL * social + LAM_SM * tm["g_smooth"],
                     np.inf).astype(np.float32)

    if breakdown:
        bd = dict(tm)
        bd.update(J_goal=J_goal, social=social, feasible=feas)
        return (float(total[0]), bool(feas[0]), bd) if single else (total, feas, bd)
    return (float(total[0]), bool(feas[0])) if single else (total, feas)


# =============================================================================
# Search
# =============================================================================
def sample_uniform(n, H, rng):
    return np.stack([rng.uniform(-A_MAX, A_MAX, (n, H)),
                     rng.uniform(-ALPHA_MAX, ALPHA_MAX, (n, H))], -1).astype(np.float32)


@njit(fastmath=True, cache=True)
def _ar1(rho, sig, noise):
    out = np.empty_like(noise)
    out[:, 0] = noise[:, 0]
    for t in range(1, noise.shape[1]):
        out[:, t] = rho * out[:, t - 1] + sig * noise[:, t]
    return out


def sample_smooth(n, H, rng):
    def ar1(tau):
        rho = np.float32(np.exp(-1.0 / tau))
        return _ar1(rho, np.float32(np.sqrt(1 - float(rho) ** 2)),
                    rng.normal(0, 1, (n, H)).astype(np.float32))

    def scale(raw, m):
        raw = raw / raw.std(1, keepdims=True).clip(min=1e-6)
        return np.clip(raw * rng.uniform(0.3, 1.0, (n, 1)).astype(np.float32) * m +
                       rng.uniform(-0.2, 0.2, (n, 1)).astype(np.float32) * m, -m, m)

    return np.stack([scale(ar1(SMOOTH_TAU_A), A_MAX),
                     scale(ar1(SMOOTH_TAU_ALPHA), ALPHA_MAX)], -1).astype(np.float32)


def farthest_point(feats, n, min_sep):
    """Greedy farthest-point over (midpoint, endpoint).  Used for both seed
    diversity and mode selection -- V6.5 had two copies of this routine."""
    sel = [0]
    md = np.linalg.norm(feats - feats[0], axis=1)
    while len(sel) < n and len(sel) < feats.shape[0]:
        k = int(np.argmax(md))
        if md[k] < min_sep:
            break
        sel.append(k)
        md = np.minimum(md, np.linalg.norm(feats - feats[k], axis=1))
        md[k] = 0.0
    return sel


def _feats(trajs):
    m = (trajs.shape[1] - 1) // 2
    return np.concatenate([trajs[:, -1, 0:2], trajs[:, m, 0:2]], 1)


def mppi(x0, obs_traj, obs_vel, geo, clear, seed_u, rng, s):
    H = seed_u.shape[0]
    mean = seed_u.copy().astype(np.float32)
    lo = np.array([-A_MAX, -ALPHA_MAX], np.float32)
    hi = np.array([A_MAX, ALPHA_MAX], np.float32)
    sa, sal = MPPI_SIG_A, MPPI_SIG_AL
    best = (None, None, np.inf, False)
    for _ in range(MPPI_ITER):
        noise = rng.standard_normal((MPPI_N, H, 2)).astype(np.float32)
        noise[..., 0] *= sa
        noise[..., 1] *= sal
        noise[0] = 0.0
        smp = np.clip(mean[None] + noise, lo, hi)
        tj = rollout(x0, smp)
        c, f = total_cost(tj, obs_traj, obs_vel, geo, clear, s)
        fin = np.where(np.isfinite(c))[0]
        if fin.size:
            b = fin[np.argmin(c[fin])]
            if c[b] < best[2]:
                best = (tj[b].copy(), smp[b].copy(), float(c[b]), bool(f[b]))
            ne = min(max(2, int(MPPI_TOP * MPPI_N)), fin.size)
            el = fin[np.argpartition(c[fin], ne - 1)[:ne]]
            ec = c[el]
            sd = ec.std()
            wgt = np.exp(-((ec - ec.mean()) / sd if sd > 1e-6 else ec - ec.mean()))
            wgt /= wgt.sum() + 1e-12
            mean = (smp[el] * wgt[:, None, None]).sum(0).astype(np.float32)
        sa *= MPPI_DECAY
        sal *= MPPI_DECAY
    return best


# =============================================================================
# Style sampling -- p(s | scene) must NOT depend on the scene
# =============================================================================
def sample_styles(rng, n, p_axis, p_corner, holdout):
    """Draw style vectors for demonstration generation.

    Four arms, all independent of the scene, each covering a region of style
    space that a later experiment evaluates in:

      neutral  [0,0,0,0], always included. Serves as the unstyled reference
               that styled behavior is measured against.

               Note that neutral is not the same as null. Neutral means "no
               preference declared, defend personal space" and is a real
               conditioning value. Null is the learned empty embedding that
               conditioning dropout substitutes during training, and means "no
               style information at all". Both have to be trained: guidance
               extrapolates away from null, so the model needs it, while
               neutral is what an unstyled request actually asks for.

      axis     one axis ~ U(-1,1), the rest exactly 0. This manifold has
               measure zero under joint sampling, so without a dedicated arm a
               single-axis request at s = +-0.5 would be evaluated entirely off
               the trained support. It is also the "train on single concepts,
               compose at inference" setting that per-axis guidance relies on.

      joint    U(-1,1)^4. The interior of style space, for composed styles.

      corner   (+-1,+-1,+-1,+-1). The extreme compositions. Under uniform
               sampling P(all |s_i| > 0.9) is about 6e-6, so a vector such as
               [1,1,1,1] would otherwise be out of distribution.

    --holdout_pass_group rejects |s_pass| > 0.5 AND |s_group| > 0.5 from every
    arm, corners included, leaving that region empty so a model can be trained
    without it and asked to compose those two axes at inference. The axis arm
    never violates the holdout by construction.
    """
    def bad(v):
        return holdout and abs(v[1]) > 0.5 and abs(v[3]) > 0.5

    out = [(NEUTRAL.copy(), "neutral")]
    for j in range(n - 1):
        r = rng.random()
        v, tag = None, None
        for _ in range(64):
            if r < p_axis:
                i = int(rng.integers(4))
                v = np.zeros(4, np.float32)
                v[i] = rng.uniform(-1, 1)
                tag = f"axis_{STYLE_AXES[i]}{j}"
            elif r < p_axis + p_corner:
                v = rng.choice(np.array([-1.0, 1.0], np.float32), 4).astype(np.float32)
                tag = f"corner{j}"
            else:
                v = rng.uniform(-1, 1, 4).astype(np.float32)
                tag = f"joint{j}"
            if not bad(v):
                break
        out.append((v, tag))
    return out


# =============================================================================
# Scene processing
# =============================================================================
def load_scene(path):
    with np.load(path, allow_pickle=True) as d:
        v0, w0 = float(d["start_state"][0]), float(d["start_state"][1])
        goal = np.asarray(d["goal"], np.float32)
        raw = d["obstacles"]
        obs = (np.asarray(raw, np.float32).reshape(-1, 4) if raw.size
               else np.zeros((0, 4), np.float32))
        extra = {k: np.array(d[k]) for k in ("threat_type", "map_type") if k in d.files}
        occ = (np.asarray(d["occupancy_map"], np.float32) if "occupancy_map" in d.files
               else np.zeros((MAP_SIZE, MAP_SIZE), np.float32))
        has_map = float(d["has_map"]) if "has_map" in d.files else 0.0
        orig = np.array(d["start_state"])
    if obs.shape[0]:
        obs = obs[np.all(np.isfinite(obs), axis=1)]
    return np.array([0, 0, 0, v0, w0], np.float32), goal, obs, occ, has_map, orig, extra


def run_style(ctx, s, labels, neutral_xy):
    """
    labels: list of (style_vec, tag) to write this result under.  Normally one;
    for pedestrian-free scenes the cost is style-independent, so we solve once
    and emit under every sampled label (see process_scene).
    """
    tag0 = labels[0][1]
    try:
        c, f = total_cost(ctx["pool_traj"], ctx["obs_traj"], ctx["obs_vel"],
                          ctx["geo"], ctx["clear"], s, G=ctx["G"])
        ok = np.where(f & np.isfinite(c))[0]
        if ok.size == 0:
            return tag0, ([], np.nan, "no_feasible_seeds", 0.0, 0, False)
        order = ok[np.argsort(c[ok])]
        seeds = farthest_point(_feats(ctx["pool_traj"][order]), N_SEEDS, 0.4)

        rng = np.random.default_rng(seed_of(os.path.basename(ctx["path"]), tag0))
        rt, ru, rk = [], [], []
        for i in seeds:
            tj, u, cst, feas = mppi(ctx["x0"], ctx["obs_traj"], ctx["obs_vel"],
                                    ctx["geo"], ctx["clear"],
                                    ctx["pool_u"][order[i]], rng, s)
            if tj is not None and feas:
                rt.append(tj); ru.append(u); rk.append(cst)
        if not rt:                                   # MPPI degenerate: use the raw pool
            for i in seeds:
                rt.append(ctx["pool_traj"][order[i]])
                ru.append(ctx["pool_u"][order[i]])
                rk.append(float(c[order[i]]))
        rt, ru, rk = np.stack(rt), np.stack(ru), np.array(rk, np.float32)
        o = np.argsort(rk)
        rt, ru, rk = rt[o], ru[o], rk[o]

        # Standing still is a legitimate DEMONSTRATION, not a failure.  Yielding
        # at s_yield = +1 IS slowing or stopping to let someone pass, so
        # rejecting "stopping is optimal" would delete exactly the
        # demonstrations that axis needs, and the model would never learn to
        # yield by waiting.  The cost defines the target: if it says stop, we
        # teach stop.  Stopping is not free (J_goal = 1; g_ttc still fires;
        # g_cut's blocking factor is MAXIMAL when stationary in someone's path;
        # no term is minimized by v = 0), so it only wins when moving would
        # require sustained maximal violation -- which is when a person would
        # stop too.  We only RECORD the rate: if it is high, especially at
        # s = [1,1,1,1], the social budget is over-weighted against J_goal --
        # a finding for the Pareto figure, not a reason to censor the data.
        bc, _ = total_cost(rollout(ctx["x0"], brake_controls()), ctx["obs_traj"],
                           ctx["obs_vel"], ctx["geo"], ctx["clear"], s)
        stopping = bool(np.isfinite(bc) and not (rk[0] < bc - 1e-4))

        # Modes: every retained mode is within `tol` of the optimum in
        # NORMALIZED cost units (0.10 = within 10% of achievable progress), so
        # all are genuinely near-optimal.  This is a statement, unlike V6.5's
        # arbitrary top-15%-by-rank.  Fewer than n_modes is correct when the
        # scene is genuinely unimodal.
        elig = np.where(rk <= rk[0] + ctx["tol"])[0]
        sel = [int(elig[i]) for i in
               farthest_point(_feats(rt[elig]), ctx["n_modes"], ctx["sep"])]

        l2 = 0.0
        if neutral_xy is not None:
            n_ = min(rt[sel[0]].shape[0], neutral_xy.shape[0])
            l2 = float(np.linalg.norm(rt[sel[0], :n_, 0:2] - neutral_xy[:n_],
                                      axis=1).mean())

        # in run_style(), after computing `sel`, before saving — behind ctx["log_terms"]:
        if ctx.get("log_terms"):
            best = rt[sel[0]]
            _, _, bd = total_cost(best[None], ctx["obs_traj"], ctx["obs_vel"],
                                ctx["geo"], ctx["clear"], s, breakdown=True)
            parts = " ".join(f"{k[2:]}={float(bd[k][0]):.3f}" for k in SOCIAL)
            print(f"[term] {os.path.basename(ctx['path'])[:24]:24s} "
                f"s=[{','.join(f'{v:+.1f}' for v in s)}] "
                f"J={float(bd['J_goal'][0]):+.2f} {parts} "
                f"sm={float(bd['g_smooth'][0]):.3f} "
                f"nmode={len(sel)} stop={int(stopping)} "
                f"npair={'dseg' in ctx['G']}", flush=True)                             

        base = ctx.get("out_prefix", "") + os.path.splitext(
            os.path.basename(ctx["path"]))[0]
        saved = []
        for sv, st in labels:
            for k, i in enumerate(sel):
                p = os.path.join(ctx["out_dir"], f"{base}__{st}" +
                                 (f"_mode{k}" if ctx["n_modes"] > 1 else "") + ".npz")
                np.savez(p, start_state=ctx["orig"], goal=ctx["goal"],
                         subgoal=ctx["subgoal"], obstacles=ctx["obs"],
                         traj_xy=rt[i, :-1, 0:2].astype(np.float32),
                         traj_full=rt[i, :-1].astype(np.float32),
                         traj_xy_with_start=rt[i, :, 0:2].astype(np.float32),
                         controls=ru[i].astype(np.float32),
                         cost=np.float32(rk[i]), reward=np.float32(-rk[i]),
                         feasible=np.bool_(True), mode_idx=np.int32(k),
                         n_modes_in_scene=np.int32(len(sel)),
                         occupancy_map=ctx["occ"].astype(np.float32),
                         has_map=np.float32(ctx["has_map"]),
                         style_values=np.clip(sv, -1, 1).astype(np.float32),
                         style_axes=np.array(STYLE_AXES, dtype=object),
                         traj_l2_vs_neutral=np.float32(l2),
                         stopping_optimal=np.bool_(stopping), **ctx["extra"])
                saved.append(p)
        return tag0, (saved, float(rk[0]), "ok", l2, len(sel), stopping)
    except Exception as e:
        import traceback
        return tag0, ([], np.nan, f"exception: {e}\n{traceback.format_exc()}", 0.0, 0, False)


def _arm(tag):
    return "axis_" + tag.split("_")[1].rstrip("0123456789") if tag.startswith("axis") \
        else tag.rstrip("0123456789")


def _run_visualize(path, vis_dir, out_dir, x0, goal, obs, obs_traj, obs_vel,
                   occ, clear, geo, full_pool_traj, styles, style_results_raw):
    """Best-effort visualization call. Never raises -- a plotting failure
    must not take down scene generation (mirrors the old generator's
    try/except around its visualize_scene call)."""
    if not _vis_available():
        return
    scene_name = os.path.splitext(os.path.basename(path))[0]
    try:
        _visualize_scene(
            scene_name, vis_dir,
            x0, goal, obs, obs_traj, obs_vel,
            occ, clear, geo,
            full_pool_traj,
            styles, style_results_raw,
            out_dir=out_dir,
        )
    except Exception as ve:
        import traceback
        print(f"[vis] WARNING: visualization failed for {scene_name}: {ve}", flush=True)
        traceback.print_exc()


def process_scene(path, out_dir, n_styles, p_axis, p_corner, n_modes, sep, tol,
                  holdout, threads, vis_dir=None, log_terms=False,
                  out_prefix=""):
    x0, goal, obs, occ, has_map, orig, extra = load_scene(path)
    subgoal, geo, clear = build_geodesic(occ, goal, has_map, obs)
    if geo is None:
        return [], "goal_unreachable", [], {}

    rng = np.random.default_rng(seed_of(os.path.basename(path), "styles"))
    styles = sample_styles(rng, n_styles, p_axis, p_corner, holdout)

    rp = np.random.default_rng(seed_of(os.path.basename(path), "phase1"))
    nw = max(512, N_UNIFORM // 2)
    ws = sample_smooth(nw, HORIZON, rp)
    ws[:, :4] *= 0.2
    pool_u = np.concatenate([sample_uniform(N_UNIFORM, HORIZON, rp),
                             sample_smooth(N_SMOOTH, HORIZON, rp), ws,
                             brake_controls()[None]], 0)
    pool_traj = rollout(x0, pool_u)
    brake_i = pool_u.shape[0] - 1
    # Kept only for --visualize (the cost-landscape plot wants the full,
    # untrimmed Phase-1 pool). pool_traj/pool_u below get reassigned to the
    # stratified-presort subset; this reference is unaffected by that and
    # nothing downstream of it reads from full_pool_traj/full_pool_u, so
    # this changes no generation logic.
    full_pool_traj, full_pool_u = pool_traj, pool_u

    obs_traj = obs_positions(obs)
    obs_vel = (obs[:, 2:4].astype(np.float32) if obs.shape[0]
               else np.zeros((0, 2), np.float32))

    nxt = pool_traj[:, 1:, 0:2]
    cheap = map_clearances(nxt, clear).min(1) >= ROBOT_R
    if obs_traj.shape[1]:
        cheap &= np.linalg.norm(nxt[:, :, None, :] - obs_traj[1:HORIZON + 1][None],
                                axis=3).min((1, 2)) >= R_C
    idx = np.where(cheap)[0]
    if idx.size == 0:
        return [], "no_feasible_seeds", [], {}

    # Stratified presort.  The social geometry on ~18k trajectories x M
    # pedestrians does not fit in memory across workers, so the pool must be
    # trimmed -- but V6.5 trimmed by distance-to-goal, which is exactly the
    # anti-social criterion we removed from the runtime selector: it deletes
    # every slowing or stopping candidate BEFORE the social cost is ever
    # evaluated, i.e. precisely the yield axis's positive end.  Half the budget
    # goes to goal-directed candidates (so good seeds survive), half is drawn at
    # random (so slow/waiting candidates survive), and the braking rollout is
    # force-included.
    if idx.size > PRESORT_K:
        gd = ((nxt[idx, -1] - subgoal[None]) ** 2).sum(1)
        k = PRESORT_K // 2
        top = idx[np.argpartition(gd, k - 1)[:k]]
        rest = np.setdiff1d(idx, top)
        extra_i = (rp.choice(rest, size=min(k, rest.size), replace=False)
                   if rest.size else rest)
        idx = np.concatenate([top, extra_i])
    if cheap[brake_i] and brake_i not in idx:
        idx = np.append(idx, brake_i)
    pool_traj, pool_u = pool_traj[idx], pool_u[idx]

    ctx = dict(path=path, out_dir=out_dir, out_prefix=out_prefix, x0=x0, orig=orig, goal=goal,
               subgoal=subgoal, obs=obs, obs_traj=obs_traj, obs_vel=obs_vel,
               occ=occ, has_map=has_map, geo=geo, clear=clear, extra=extra,
               pool_traj=pool_traj, pool_u=pool_u,
               G=geometry(pool_traj, obs_traj, obs_vel),
               n_modes=n_modes, sep=sep, tol=tol, log_terms=log_terms)

    # Pedestrian-free scene: every social term is identically zero, so the cost
    # -- and therefore the demonstration -- is style-independent.  Solve once,
    # emit under every sampled label.  A neutral-only shortcut (V6.5's "low
    # complexity" tier) would leak "no pedestrians" into the style prior; this
    # keeps p(s) scene-independent at ~1/n_styles the cost, and teaches the true
    # fact that style is irrelevant with nobody around.
    if obs_traj.shape[1] == 0:
        _, r = run_style(ctx, NEUTRAL.copy(), styles, None)
        if vis_dir and _vis_available():
            _run_visualize(path, vis_dir, out_dir, x0, goal, obs, obs_traj, obs_vel,
                           occ, clear, geo, full_pool_traj,
                           styles, {tag: r for _, tag in styles})
        return r[0], ("ok" if r[0] else r[2]), [("empty", bool(r[0]), r[2])], \
            {"modes": r[4], "stop": int(r[5])}

    _, rn = run_style(ctx, NEUTRAL.copy(), [(NEUTRAL.copy(), "neutral")], None)
    neutral_xy = None
    if rn[0]:
        with np.load(rn[0][0]) as d:
            neutral_xy = np.array(d["traj_xy_with_start"])
    res = [("neutral", rn)]
    tasks = [(v, t) for v, t in styles if t != "neutral"]
    if tasks:
        with ThreadPoolExecutor(max_workers=min(threads, len(tasks))) as ex:
            futs = [ex.submit(run_style, ctx, v, [(v, t)], neutral_xy) for v, t in tasks]
            res += [f.result() for f in as_completed(futs)]

    saved, per, l2, nm, ns = [], [], {}, [], 0
    for t, (sp, _, why, v, m, stop) in res:
        per.append((_arm(t), bool(sp), why))
        if sp:
            saved += sp
            l2[_arm(t)] = v
            nm.append(m)
            ns += int(stop)

    if vis_dir and _vis_available():
        _run_visualize(path, vis_dir, out_dir, x0, goal, obs, obs_traj, obs_vel,
                       occ, clear, geo, full_pool_traj, styles, dict(res))

    return saved, ("ok" if saved else "all_styles_failed"), per, \
        {"l2": l2, "modes": float(np.mean(nm)) if nm else 0.0, "stop": ns}


def _worker(a):
    # a = (path, out_dir, ...). A scene is "done" once process_scene returns
    # WITHOUT an exception — including the legitimate empty outcomes
    # (goal_unreachable, no_feasible_seeds), which correctly produce no files but
    # must not be retried forever. We write an atomic <scene>.done marker only on
    # a clean return, so a job killed mid-scene (SLURM timeout) leaves NO marker
    # and the scene is fully regenerated on requeue rather than half-skipped.
    path, out_dir, out_prefix = a[0], a[1], a[-1]
    try:
        result = process_scene(*a)
        if not (isinstance(result, tuple) and len(result) >= 2
                and str(result[1]).startswith("exception")):
            base = out_prefix + os.path.splitext(os.path.basename(path))[0]
            marker = os.path.join(out_dir, f"{base}.done")
            tmp = marker + f".tmp{os.getpid()}"
            with open(tmp, "w") as fh:
                fh.write(str(result[1]))          # record the outcome reason
            os.replace(tmp, marker)               # atomic: never a partial marker
        return result
    except Exception as e:
        import traceback
        return [], f"exception: {e}\n{traceback.format_exc()}", [], {}


# =============================================================================
# CLI
# =============================================================================
def main():
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    p = argparse.ArgumentParser(description="Expert demonstration generator FINAL")
    p.add_argument("--in_dir", default="data/scenes/eval_500")
    p.add_argument("--out_dir", default="data/expert_trajectories")
    p.add_argument("--n_styles_per_scene", type=int, default=6)
    p.add_argument("--p_axis", type=float, default=0.40,
                   help="Fraction of non-neutral draws on the single-axis "
                        "manifold, which has measure zero under joint sampling.")
    p.add_argument("--p_corner", type=float, default=0.10,
                   help="Fraction drawn from the 16 (+-1,+-1,+-1,+-1) corners.")
    p.add_argument("--n_modes", type=int, default=3)
    p.add_argument("--mode_separation", type=float, default=0.5)
    p.add_argument("--mode_tol", type=float, default=0.10,
                   help="Keep modes within this much of the optimum, in normalized "
                        "cost units (0.10 = within 10%% of achievable progress).")
    p.add_argument("--holdout_pass_group", action="store_true",
                   help="Exclude |s_pass|>0.5 AND |s_group|>0.5 (composition study).")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    p.add_argument("--style_threads", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--out_prefix", default="",
                   help="Prepended to every output filename and .done marker. "
                        "Scene files are named sample_<N>.npz in every source "
                        "directory, so give each source its own prefix "
                        "(e.g. --out_prefix mp_) when writing several sources "
                        "into one --out_dir. Without it the second source "
                        "would overwrite the first's demonstrations and be "
                        "skipped by its resume markers.")
    p.add_argument("--force_redo", action="store_true")
    p.add_argument("--log_terms", action="store_true",
               help="Print per-term cost breakdown for the best trajectory of each style.")
    p.add_argument("--visualize", action="store_true",
                   help="Emit cost-landscape / style-comparison / cost-breakdown / "
                        "radar / per-term cost-field PNGs for every processed scene")
    p.add_argument("--vis_dir", default="vis_output/",
                   help="Directory for visualization PNGs (used with --visualize)")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.in_dir, "sample_*.npz")))[a.offset:]
    if a.limit:
        files = files[:a.limit]
    if a.force_redo:
        # Wipe markers so every scene regenerates. (Output .npz are overwritten
        # in place by run_style, so they need not be deleted here.)
        for m in glob.glob(os.path.join(a.out_dir, f"{a.out_prefix}*.done")):
            os.remove(m)
    else:
        # A scene is skipped ONLY if its atomic .done marker exists — i.e. it was
        # FULLY processed (all styles + modes written, or a legitimate empty
        # outcome). A scene whose job died mid-way has no marker and is redone,
        # overwriting its partial files. This is the fix for the "any-file-exists
        # => skip" bug, which silently dropped styles when a job hit the wall
        # mid-scene.
        done = {os.path.splitext(os.path.basename(m))[0]
                for m in glob.glob(os.path.join(a.out_dir, "*.done"))}
        files = [f for f in files
                 if a.out_prefix + os.path.splitext(os.path.basename(f))[0]
                 not in done]

    print(f"FINAL norm-grounded.  {len(files)} scenes.", flush=True)
    print(f"  R_c={R_C:.2f} (Hall intimate {D_INT})   ladder "
          f"{ladder(-1):.2f}/{ladder(0):.2f}/{ladder(1):.2f} m   "
          f"rho {RHO_DN:.2f}/{RHO_UP:.2f}", flush=True)
    print(f"  tau_a {ladder(-1)/V_BAR:.2f}/{ladder(0)/V_BAR:.2f}/"
          f"{ladder(1)/V_BAR:.2f} s   predicted g_prox^max "
          + "/".join(f"{min(1.0, max(0.0, (ladder(x) - R_C) / W_ZONE)):.2f}"
                     for x in (-1, -0.5, 0, 0.5, 1)),
          flush=True)
    print(f"  arms: axis={a.p_axis} corner={a.p_corner} "
          f"joint={1-a.p_axis-a.p_corner:.2f}   holdout={a.holdout_pass_group}   "
          f"n_modes={a.n_modes} tol={a.mode_tol}", flush=True)
    if a.visualize and not _vis_available():
        print("  WARNING: --visualize set but visualize_expert_trajectories.py is not "
              "importable; skipping visualization.", flush=True)
    vis_dir_arg = a.vis_dir if (a.visualize and _vis_available()) else None
    if vis_dir_arg:
        print(f"  Visualizing to: {vis_dir_arg}", flush=True)

    n_ok = n_f = n_e = n_stop = 0
    why, st, ss = Counter(), Counter(), Counter()
    l2a, modes = defaultdict(list), []
    t0 = time.time()
    args = [(f, a.out_dir, a.n_styles_per_scene, a.p_axis, a.p_corner, a.n_modes,
             a.mode_separation, a.mode_tol, a.holdout_pass_group, a.style_threads,
             vis_dir_arg, a.log_terms, a.out_prefix)
            for f in files]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(_worker, x) for x in args]), 1):
            saved, reason, per, ex_ = fut.result()
            if reason.startswith("exception"):
                n_e += 1
                if n_e == 1:
                    print(f"\n!!! FIRST ERROR:\n{reason}\n", flush=True)
            for t, ok, r in per:
                st[t] += 1
                ss[t] += int(ok)
                if not ok:
                    why[r.split(":")[0]] += 1
            for t, v in ex_.get("l2", {}).items():
                l2a[t].append(v)
            if ex_.get("modes"):
                modes.append(ex_["modes"])
            n_stop += ex_.get("stop", 0)
            if saved:
                n_ok += 1
                n_f += len(saved)
            else:
                why[reason] += 1
            if i % 100 == 0 or i == len(files):
                el = time.time() - t0
                print(f"  [{i}/{len(files)}] ok={n_ok} files={n_f} "
                      f"({n_f/max(n_ok,1):.2f}/scene) "
                      f"modes={np.mean(modes) if modes else 0:.2f} stop={n_stop} "
                      f"err={n_e}  {el/i:.2f}s/scene  "
                      f"ETA={(len(files)-i)*el/i/3600:.1f}h", flush=True)

    print(f"\nDone. {n_ok}/{len(files)} scenes, {n_f} trajectories.")
    print(f"  mean modes per (scene,style) = {np.mean(modes) if modes else 0:.2f}   "
          f"(if ~1.0, raise --mode_tol)")
    print(f"  stopping_optimal = {n_stop} (scene,style) pairs   "
          f"(if high, the social budget is over-weighted vs J_goal)")
    print("\nPer-arm success / mean L2 vs neutral:")
    for t in sorted(st):
        v = l2a.get(t, [])
        print(f"  {t:>16s}  {ss[t]:>6d}/{st[t]:<6d} ({100*ss[t]/max(st[t],1):5.1f}%)"
              + (f"  L2={np.mean(v):.3f}" if v else ""))
    if why:
        print("\nFailure reasons:")
        for r, c in why.most_common():
            print(f"  {r:35s} {c:>6d}")


if __name__ == "__main__":
    main()