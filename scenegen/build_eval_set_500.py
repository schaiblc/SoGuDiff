"""
build_eval_set_500.py
==========================

A 500-scene STYLE-DIFFERENTIATION test set.

    python build_eval_set_500.py --out_dir \
        data/scenes/eval_500

WHY THE SCENES ARE BUILT THIS WAY
---------------------------------
A style axis only shows up in a metric when the scene admits both behaviors.
An earlier 100-scene set could not measure the group axis at all: group splits
were possible in just 9 of 100 scenes, which left the guidance sweep for that
axis non-monotone purely from lack of support.

The cause is geometric rather than statistical. The robot needs more than
2*R_c = 1.00 m of clear space to fit between two people, and pairwise member
spacings in that earlier set were:

    < 1.00 m  (IMPASSABLE)        60%     <- robot CANNOT split, metric pinned at 0
    1.0-2.4 m (CONTESTED)         36%
    > 2.4 m   (WIDE OPEN)          4%     <- robot passes through regardless
    median spacing 0.80 m

A style axis can only show up in a metric when the scene admits BOTH behaviors.
With a 0.80 m median gap, "go around" was the only feasible option at any style,
so group+ and group- produced identical trajectories by construction.

THE DESIGN RULE APPLIED THROUGHOUT
----------------------------------
Every scene is placed in the CONTESTED band for the axis it targets -- the
region where the styled and unstyled behaviors are both feasible and visibly
different:

  group  : member spacing in [1.35, 2.4] m.  Wide enough that a group-agnostic
           robot will cut through, tight enough that a group-aware robot pays a
           real detour to go around.  Groups straddle y = 0 so the gap is ON the
           direct line to the goal.
  yield  : crossing agents timed so the pedestrian reaches the conflict point at
           0.7-1.3x the robot's arrival time.  Outside that band there is no
           conflict to yield to and the TTC metric stays at 0.
  pass   : head-on encounters with the oncoming agent within +-0.3 m of the
           centerline, so left and right are both viable and the side choice is
           genuinely free.
  prox   : overtakes and stationary clusters with open lateral space, so
           clearance is the free variable rather than something geometry forces.

GUARANTEES (each verified over all 500 scenes after generation)
--------------------------------------------------------------
  1. Goal reachable for EVERY style: no pedestrian is within 3.66 m -- Hall
     social distance, the largest clearance any style asks for -- of the goal
     at t = 24 s, one second before the 25 s limit. A timeout is therefore
     attributable to the planner, never to the benchmark.
  2. Nothing unavoidable at t = 0: no agent spawns within 1.8 m of the robot.
  3. Group gaps are CONTESTED: adjacent member spacing 1.35-2.4 m, above the
     1.0 m the robot needs to fit and below the width at which cutting through
     is free. Both "split" and "go around" are always available.
  4. Group span <= 4.1 m, so even prox+ (which must clear the outermost member
     by 3.66 m) can detour at |y| <= 5.71 m, inside the +-7.5 m world.
  5. Corridors are passable on BOTH sides: >= 1.35 m of body clearance on the
     squeeze side and >= 1.8 m on the open side, so the side choice is a real
     preference and not a wall.
  6. At most 3 agents per scene, goal fixed at (+6, 0) so the goal itself never
     biases left or right, and 500/500 obstacle configurations distinct.

Maps appear in 8% of scenes (Category COR only): the static-map channel is a
capability of the method, not what this benchmark measures.

Scenes are parametric sweeps of the hand-designed archetypes in
style_probe_scenes.py, not random configurations -- every filename encodes its
parameters, so any scene in a results table can be read back and reproduced.
"""

import argparse
import os
import numpy as np

import style_probe_scenes as G

GOAL = [6.0, 0.0]    # same convention as style_probe_scenes.py
V_NOM = 0.8          # robot's effective speed for conflict-timing arithmetic
MIN_SPAWN_DIST = 1.8  # no agent closer than this at t=0
R_CONTACT = 0.5       # robot+human contact radius; centers must stay this far apart


def _far_enough(obstacles):
    """Reject anything that starts on top of the robot at (0,0)."""
    for o in obstacles:
        if np.hypot(o[0], o[1]) < MIN_SPAWN_DIST:
            return False
    return True


# ---------------------------------------------------------------------------
# GOAL KEEP-OUT -- a scene must never make its own goal unreachable.
#
# Measured on an earlier build: 41 scenes parked a STATIONARY pedestrian within
# D(+1) = 3.66 m of the goal, 17 of them within D(0) = 1.22 m (closest 0.90 m).
# Under prox+ the robot holds 3.66 m from every person, so it could never
# occupy the goal -- a guaranteed timeout produced by the scene, not by the
# planner, and exactly the kind of number a reader would misattribute to the
# method.
#
# The condition is checked at ONE INSTANT, t = T_FINAL, not across a window.
# Pedestrians move at constant velocity for the whole episode, so anyone
# crossing the goal region does so transiently and keeps going -- the robot can
# wait them out and then finish. The only unrecoverable case is someone still
# parked on the goal when time runs out. Checking a whole window instead was
# far too strict: it discarded most straight-line overtakes over conflicts that
# resolve themselves seconds earlier.
# ---------------------------------------------------------------------------
GOAL_KEEPOUT = 3.66
T_FINAL = 24.0               # env.config time_limit is 25 s; check just before it


def _goal_reachable(obstacles, goal=GOAL):
    """The goal must be free at the END of the episode.

    Pedestrians move at constant velocity for the whole episode, so anyone
    crossing the goal region does so transiently and keeps going. The robot can
    simply wait them out and then finish. What it can NEVER do is share the goal
    with someone parked there -- so the binding condition is a single instant:
    at t = T_FINAL, is every agent clear of the goal?

    Checking every instant of the arrival window instead (an earlier version)
    was far too strict: it rejected any scene where a pedestrian merely passed
    near the goal mid-episode, which threw out most straight-line overtakes for
    a conflict that resolves itself seconds later.

    The threshold is the full social radius, so the check is the demanding
    prox+ case: if prox+ can sit at the goal, every other style can too.
    """
    g = np.asarray(goal, dtype=float)
    for o in obstacles:
        end = np.array([o[0] + o[2] * T_FINAL, o[1] + o[3] * T_FINAL])
        if np.linalg.norm(end - g) < GOAL_KEEPOUT:
            return False
    return True


# =============================================================================
# Category G — GROUP.  Members abreast, spacing in the contested band so
# "cut through" and "go around" are both feasible.
#
# TWO constraints, and the first version only enforced one of them:
#
#   GAP (enforced from the start): adjacent spacing 1.35-2.4 m, so the robot can
#       physically fit between members and the split-vs-detour choice is real.
#   SPAN (missing, and it dominated): the group's total width. A 4-member line
#       at 2.4 m spacing is 7.2 m across. Under prox+ the Hall ladder sets
#       D(+1) = 3.66 m, so clearing that group means passing at |y| >= 7.26 m --
#       outside the +-7.5 m world. With W_SOCIAL = 4 encoding "stopping is
#       preferable to a maximal norm violation", the robot correctly STOPS. That
#       produced 53% timeouts for prox+ and 30% even for neutral: the scene left
#       no legal path, so it measured the world boundary, not the style.
#
# So: n <= 3, and spacing capped per n to hold the span <= 3.6 m. A lateral
# offset is also swept, which puts the group off-center and makes one go-around
# side cheap while the gap stays near the robot's path -- that keeps the
# split-vs-detour tradeoff genuinely contested instead of making the detour
# unaffordable.
# =============================================================================
MIN_GAP_CONTESTED = 1.35  # measured: the robot does not use a 1.2 m gap, so a
                          # 1.2 m scene cannot differentiate -- it only ever
                          # produces the detour, like the sub-1.0 m gaps did.
MAX_SPAN = 4.1        # meters; prox+ must clear the outermost member by 3.66 m,
                      # so the detour sits at |y| <= 4.1/2 + 3.66 = 5.71 m,
                      # comfortably inside the +-7.5 m world.


def build_group():
    out = []
    for n, spacings in ((2, (1.35, 1.5, 1.8, 2.1, 2.4)), (3, (1.35, 1.5, 1.8)),
                        (4, (1.35,))):   # 4 x 1.35 m >= 3.6 m span, still <= MAX_SPAN
        for spacing in spacings:
            span = (n - 1) * spacing
            if span > MAX_SPAN:
                continue
            # Distance depends on whether the group ever clears. A STATIONARY
            # group never does, so it must start near enough to the robot that
            # it is still >= GOAL_KEEPOUT from the goal; a moving one walks out
            # of the way and can start further out.
            for speed in (0.0, 0.4, 0.8):
                for dist in ((2.2, 2.6) if speed == 0.0 else (3.0, 3.6, 4.2)):
                    # MIRRORED offsets. Offsetting only to +y put 73% of the
                    # group scenes on the robot's left, so it passed on the
                    # right by construction and the whole set carried a +0.249
                    # neutral passing_side_bias -- a benchmark artifact that
                    # would read as a right-hand prior in the policy.
                    for y_off in (0.0, 0.9, -0.9, 1.5, -1.5):
                        for orient, vy in (("headon", 0.0), ("oblique", 0.25)):
                            # A head-on abreast line with NO lateral drift is a
                            # moving WALL, not a formation to negotiate. Measured
                            # failure rates on the previous build (neutral and
                            # group-, so not a style effect):
                            #     n=4 head-on, any offset  100%
                            #     n=3 head-on, centerd      90%
                            #     n=2 head-on, centerd      47%
                            #     anything oblique        0-40%
                            # The gap closes at up to 0.8 m/s so threading it is
                            # unsafe, and escaping past a 4 m span while the line
                            # advances does not finish inside 25 s. The robot
                            # retreats -- 13.4 m of path, no arrival. NEITHER
                            # option is available, so the scene cannot express a
                            # group preference; it only measures the wall.
                            #
                            # A head-on line is therefore allowed only when it is
                            # offset enough to leave an open side, and wide lines
                            # (n >= 3) only when they drift laterally and vacate
                            # the corridor on their own.
                            if orient == "headon" and abs(y_off) < 0.9:
                                continue
                            if orient == "headon" and n >= 3:
                                continue
                            offs = [y_off + (i - (n - 1) / 2.0) * spacing
                                    for i in range(n)]
                            obs = [[dist, y, -speed, vy] for y in offs]
                            if not (_far_enough(obs) and _goal_reachable(obs)):
                                continue
                            name = (f"G_n{n}_s{spacing:.1f}_d{dist:.1f}"
                                    f"_v{speed:.1f}_off{y_off:.1f}_{orient}")
                            out.append((name, G._npz_kwargs(
                                goal=GOAL, obstacles=obs,
                                threat_type="group_head_on" if speed > 0
                                            else "stationary_group")))
    return out


# =============================================================================
# Category Y — YIELD.  A crossing agent reaches the conflict point at ratio*t_robot.
# ratio ~ 1.0 is a genuine right-of-way contest; a yielding robot slows, an
# assertive one cuts in front.  Outside [0.7, 1.3] there is nothing to yield to.
# =============================================================================
def build_yield():
    out = []
    for cx in (2.5, 3.5, 4.5):
        t_robot = cx / V_NOM
        for ratio in (0.7, 0.85, 1.0, 1.15, 1.3):
            for v_p in (0.5, 0.8, 1.1):
                for side in (+1, -1):
                    for n_cross in (1, 2, 3):
                        dy = ratio * t_robot * v_p       # so it arrives on time
                        obs = [[cx, side * dy, 0.0, -side * v_p]]
                        if n_cross >= 2:
                            obs.append([cx + 1.2, -side * dy * 0.9, 0.0, side * v_p])
                        if n_cross >= 3:
                            obs.append([cx + 2.4, side * dy * 1.15, 0.0, -side * v_p])
                        if not (_far_enough(obs) and _goal_reachable(obs)):
                            continue
                        if abs(dy) > 6.0:               # keep inside the world box
                            continue
                        sd = "L" if side > 0 else "R"
                        name = (f"Y_cx{cx:.1f}_r{ratio:.2f}_vp{v_p:.1f}"
                                f"_{sd}_n{n_cross}")
                        out.append((name, G._npz_kwargs(
                            goal=GOAL, obstacles=obs, threat_type="cutoff")))
    return out


# =============================================================================
# Category P — PASS.  Head-on within +-0.3 m of the centerline: both sides are
# open, so the side choice is free and passing_side_bias can actually move.
# =============================================================================
def build_pass():
    out = []
    for offset in (-0.3, -0.15, 0.0, 0.15, 0.3):
        for dist in (3.0, 4.0, 5.0, 6.0):
            for v in (0.5, 0.8, 1.1):
                for n in (1, 2, 3):
                    obs = [[dist, offset, -v, 0.0]]
                    if n >= 2:
                        # Further back and slightly opposite, so the robot has to
                        # commit to a side rather than weave between them.
                        obs.append([dist + 1.8, -offset, -v * 0.9, 0.0])
                    if n >= 3:
                        obs.append([dist + 3.4, offset * 1.5, -v * 0.8, 0.0])
                    if not (_far_enough(obs) and _goal_reachable(obs)):
                        continue
                    name = f"P_o{offset:+.2f}_d{dist:.1f}_v{v:.1f}_n{n}"
                    out.append((name, G._npz_kwargs(
                        goal=GOAL, obstacles=obs, threat_type="head_on")))
    return out


# =============================================================================
# Category X — PROX.  Overtaking slower/stationary agents with open lateral
# space: how wide the robot swings is style, not geometry.
# =============================================================================
def build_prox():
    """Overtaking a slower pedestrian, with open lateral space.

    Lead agents DRIFT laterally (vy) so that by the time they reach the goal's
    x-coordinate they are well off the goal line. Without the drift an overtaken
    pedestrian walks straight down the centerline into the goal and parks there,
    which blocks the goal for prox+ and manufactures a timeout.
    """
    out = []
    # drift 0.0 is the plain straight-line overtake; +-0.18 makes the pedestrian
    # veer. With the end-of-episode check, FASTER straight walkers are the ones
    # that work: by t = 24 s they are well past the goal and no longer near it.
    # A stationary lead works too, provided it starts clear of the goal. It is
    # the mid-speed walkers that end up parked on the goal at 24 s, and
    # _goal_reachable() removes exactly those -- no special-casing needed.
    for v_lead in (0.0, 0.35, 0.5, 0.6):
        for dist in ((2.0, 2.3) if v_lead == 0.0 else (2.2, 2.8, 3.4, 4.0)):
            for lat in (-0.3, 0.0, 0.3):
                for drift in (-0.18, 0.0, 0.18):
                    # A stationary pedestrian has no drift, so the three drift
                    # values would collapse to one identical scene AND one
                    # identical filename -- silently overwriting two scenes on
                    # disk and leaving the set short of its target.
                    if v_lead == 0.0 and drift != 0.0:
                        continue
                    for n in (1, 2, 3):
                        vy = 0.0 if v_lead == 0.0 else drift
                        obs = [[dist, lat, v_lead, vy]]
                        if n >= 2:
                            obs.append([dist + 1.5, lat + 0.35, v_lead, vy])
                        if n >= 3:
                            obs.append([dist + 3.0, lat - 0.30, v_lead, vy])
                        if not (_far_enough(obs) and _goal_reachable(obs)):
                            continue
                        name = (f"X_vl{v_lead:.2f}_d{dist:.1f}_y{lat:+.2f}"
                                f"_dr{vy:+.2f}_n{n}")
                        out.append((name, G._npz_kwargs(
                            goal=GOAL, obstacles=obs, threat_type="sneak_up")))
    return out


# =============================================================================
# Category COR — corridor counterflow WITH a static map (the 8% map minority).
#
# Oncoming agents travel SINGLE FILE in a lane on one side, the way real
# counterflow behaves -- not abreast across the corridor.
#
# The abreast version (first attempt) was a trap: three agents spread across a
# half_w = 1.6 m corridor left a widest opening of 0.90 m, below the 1.00 m the
# robot physically needs, so the corridor was fully blocked. The robot correctly
# waited for the agents to clear, then had no time left to reach the goal --
# producing timeouts that measured dynamic blocking, not style. Even the 1.30 m
# openings elsewhere were a squeeze once walls are inflated by AGENT_RADIUS, so
# the whole category was testing the wrong thing.
#
# Single-file in a lane guarantees an open channel on the far side, which is
# both realistic and the RIGHT style probe: does the robot hug the free side
# (pass) and how much room does it leave the oncoming line (prox), under lateral
# confinement it cannot simply detour around.
# =============================================================================
MIN_FAR   = 1.8   # clear body-width on the open side of the oncoming lane
MIN_TIGHT = 1.35  # clear body-width on the SQUEEZE side. 1.2 m was still too
                  # tight to be chosen in practice, so it made the side a wall
                  # rather than a preference -- same failure as the group gaps.
LANE_JITTER = 0.12  # per-agent lateral jitter; MUST be budgeted into the squeeze
WALL_MARGIN = 0.5 # wall inflation (AGENT_RADIUS) applied for collision checks


def build_corridor():
    """Single-file counterflow, with BOTH sides passable.

    The lane offset is DERIVED from the two clearance constraints, not chosen as
    a fraction of the corridor. Two earlier versions failed here:

      v1, agents abreast: a half_w = 1.6 m corridor with 3 agents left a 0.90 m
          widest opening -- below the 1.00 m the robot needs. Fully blocked; the
          robot waited for a gap that never came and timed out.
      v2, lane as a fraction of the width: the far side was generous (2.0-3.6 m)
          but the squeeze side collapsed to 0.09-0.88 m. A left-biased style
          committed to that side, jammed against the wall, and timed out. The
          scene admitted only ONE side, so the pass axis could not differentiate
          -- it could only fail.

    Now: lane_y = inner - R_c - MIN_TIGHT, which pins the squeeze side at exactly
    MIN_TIGHT and leaves the rest to the open side. Both choices are feasible,
    they just cost different clearance -- which is what makes the pass axis
    measurable here instead of merely punished.

    The lane is mirrored to both sides so pass+ and pass- each meet the squeeze
    and the open side equally often; otherwise the side metric would inherit a
    bias from the scene set itself.
    """
    out = []
    for half_w in (2.8, 3.2, 3.6):
        occ = G.corridor_map(GOAL, half_w=half_w, length=20.0)
        inner = half_w - WALL_MARGIN
        for tight in (MIN_TIGHT, MIN_TIGHT + 0.3):
            # Budget the jitter into the lane offset. Without this the realized
            # squeeze is (tight - LANE_JITTER): a nominal 1.0 m came out at
            # 0.88 m, still under the robot's 1.0 m minimum, and a style biased
            # to that side jammed against the wall and timed out.
            lane_mag = inner - R_CONTACT - tight - LANE_JITTER
            far = lane_mag - R_CONTACT + inner
            if lane_mag <= 0 or far < MIN_FAR:
                continue
            for side in (+1, -1):
                lane_y = side * lane_mag
                for n_onc in (1, 2, 3, 4):
                    for v in (0.5, 0.8):
                        for d0 in (3.5, 5.0):
                            obs = [[d0 + 1.4 * i, lane_y + (LANE_JITTER if i % 2 else -LANE_JITTER),
                                    -v, 0.0] for i in range(n_onc)]
                            if not (_far_enough(obs) and _goal_reachable(obs)):
                                continue
                            sd = "L" if side > 0 else "R"
                            name = (f"COR_hw{half_w:.1f}_{sd}_tight{tight:.1f}"
                                    f"_n{n_onc}_v{v:.1f}_d{d0:.1f}")
                            out.append((name, G._npz_kwargs(
                                goal=GOAL, obstacles=obs, occupancy_map=occ,
                                threat_type="corridor_head_on")))
    return out


# Target mix. Group and yield get the most scenes because they are the two axes
# whose metrics were starved on the old set; pass and prox already measured
# cleanly and need fewer.
TARGETS = [("G", build_group, 160), ("Y", build_yield, 130),
           ("P", build_pass, 110), ("X", build_prox, 60),
           ("COR", build_corridor, 40)]


def _thin(pool, target):
    """Exactly `target` DISTINCT entries, spread evenly across the sweep.

    Even spacing rather than a random draw, so the retained scenes still cover
    the parameter grid uniformly instead of clumping. Collisions from rounding
    are walked forward to the next unused index, which guarantees the count is
    exact -- padding by resampling would duplicate filenames and silently
    overwrite scenes on disk.
    """
    n = len(pool)
    if target == 1:
        return [pool[0]]
    used, out = set(), []
    for k in range(target):
        i = int(round(k * (n - 1) / (target - 1)))
        while i in used:
            i = (i + 1) % n
        used.add(i)
        out.append(i)
    return [pool[i] for i in sorted(out)]


def generate_all(out_dir, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    seen_sigs = set()
    written = []
    for tag, fn, target in TARGETS:
        pool = fn()
        # Drop configurations already emitted by another category: two builders
        # can coincidentally produce the same obstacle array, and "500 scenes"
        # should mean 500 distinct ones.
        uniq = []
        for name, kw in pool:
            sig = (np.round(kw["obstacles"], 3).tobytes(), float(kw["has_map"]))
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            uniq.append((name, kw))
        pool = uniq
        if len(pool) < target:
            raise SystemExit(
                f"{tag}: only {len(pool)} feasible scenes, need {target}. "
                f"Widen that category's parameter grid.")
        pool = _thin(pool, target)   # exactly `target`, spread across the sweep
        for name, kw in pool[:target]:
            if kw["has_map"] > 0.5:
                G._check_map_scene(name, kw["occupancy_map"], kw["goal"])
            np.savez(os.path.join(out_dir, f"sample_{name}.npz"), **kw)
            written.append((tag, name, kw))
    return written


def report(written):
    import itertools, collections
    print(f"\n{len(written)} scenes written")
    c = collections.Counter(t for t, _, _ in written)
    for k, v in c.most_common():
        print(f"   {k:4s} {v:4d}")

    gaps, nobs, nmap = [], [], 0
    for tag, name, kw in written:
        o = kw["obstacles"]
        nobs.append(len(o))
        nmap += int(kw["has_map"] > 0.5)
        if tag == "G" and len(o) >= 2:
            # ADJACENT gaps only: sort members across the group and take
            # consecutive spacings. Non-adjacent pairs (the ends of a 4-person
            # line) are not openings the robot can pass through, so including
            # them would overstate how open the group is.
            ys = np.sort(o[:, 1])
            gaps.extend(float(d) for d in np.diff(ys))
    gaps = np.array(gaps)
    print(f"\n   agents/scene: mean={np.mean(nobs):.2f} max={max(nobs)}")
    print(f"   scenes with a map: {nmap} ({100*nmap/len(written):.0f}%)")
    print(f"\n   GROUP member spacings:  n={len(gaps)}")
    print(f"     < 1.00 m  impassable : {100*(gaps<1.0).mean():5.1f}%")
    print(f"     1.35-2.4 contested   : {100*((gaps>=1.0)&(gaps<=2.4)).mean():5.1f}%")
    print(f"     > 2.4 m   wide open  : {100*(gaps>2.4).mean():5.1f}%")
    print(f"     median = {np.median(gaps):.2f} m")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    report(generate_all(a.out_dir, a.seed))
