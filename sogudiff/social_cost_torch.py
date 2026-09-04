"""
social_cost_torch.py -- batched Torch port of the norm-grounded expert cost.
===========================================================================

This is the SAME cost that generates the demonstrations
(`generate_expert_trajectories.py`), re-expressed in Torch so
the trainer can use it for best-of-K selection and checkpoint selection.

Why this file exists
--------------------
The trainer previously scored candidates with the *V6.5* cost
(`exp(3 ln2 . s)` intensity weights, a `c2_head_on` term the expert deleted,
`_time_blend = 0.5 max + 0.5 mean` instead of `mean_t`, a 0.65 m collision
radius instead of 0.50 m).  That put a DIFFERENT objective between the
diffusion model and every reported number: the selector picked the candidate
that best satisfied an obsolete cost, and checkpoints were chosen by it.  Any
claim of the form "the requested style survives the pipeline" is a claim about
this function, so it has to be the real one.

Fidelity to the expert
----------------------
Every term, gate, constant and aggregation matches
`generate_expert_trajectories.py` (P1-P6 in its docstring).
`test_social_cost_torch.py` checks this numerically against the expert's own
`geometry()` / `terms()` on random scenes.

Two DELIBERATE deviations, both documented at their use sites:

  1. J_goal uses EUCLIDEAN distance-to-goal, not the geodesic (FMM) distance.
     Computing an FMM field per batch element inside the training loop is not
     affordable.  For best-of-K this is benign: all K candidates share one
     scene, one start and one 1.6 s horizon, so the geodesic/Euclidean gap is
     a near-constant offset that cancels in the argmin.  It is NOT benign if
     you ever use this to compare ACROSS scenes -- use the expert for that.
     Pass `goal=None` to drop J_goal entirely and score social terms only.

  2. The robot's (theta, v, omega) are recovered by finite-differencing the
     position sequence, because the diffusion model emits positions only.  The
     expert has them exactly from its rollout.  This affects g_smooth (jerk /
     angular jerk) and the coasting tail's terminal heading, both mildly.

Shapes
------
  traj_xy      (B, T, 2)   ego-frame positions, index 0 == the start pose
  obs          (B, M, 4)   pedestrians at t=0 as (x, y, vx, vy)
  obs_mask     (B, M)      1 = real pedestrian, 0 = padding
  style        (B, 4)      (s_prox, s_pass, s_yield, s_group) in [-1, 1]
  goal         (B, 2)      optional, ego frame
"""
from __future__ import annotations

import math
import torch

# =============================================================================
# Constants -- MUST match generate_expert_trajectories.py.
# Do not "tune" anything here.  If the expert changes, change it there first
# and re-run test_social_cost_torch.py.
# =============================================================================
DT, HORIZON = 0.05, 32
T_H = HORIZON * DT                      # 1.6 s plan
V_MAX, A_MAX = 1.0, 1.5
W_MAX, ALPHA_MAX = math.pi, math.pi

ROBOT_R = HUMAN_R = 0.25
R_C = ROBOT_R + HUMAN_R                 # 0.50 m contact

D_INT, D_PER, D_SOC = 0.46, 1.22, 3.66  # Hall 1966
RHO_DN = D_PER / R_C                    # 2.44
RHO_UP = D_SOC / D_PER                  # 3.00

V_BAR = 1.34                            # Weidmann '93 free-flow walking speed
COS_PSI = math.sqrt(2) / 2              # 45 deg encounter cone (SA-CADRL)
DV_MAX = V_BAR / 2                      # 0.67 m/s co-motion gate
V_MIN = 0.05                            # stationary threshold
LAM_SM = 0.05                           # legibility weight
W_SOCIAL = 4.0                          # social / efficiency exchange rate

T_LOOK = D_SOC / V_BAR                  # 2.73 s bounded social lookahead
K_TAIL = int(round((T_LOOK - T_H) / DT))  # 23 coasting steps past the plan

W_ZONE = D_PER - R_C                    # 0.72 m minimum penetration band
PROG_MAX = V_MAX * DT * HORIZON * (HORIZON + 1) / 2   # 26.4 m-steps

JERK_MAX = 2 * A_MAX / DT               # 60
AJERK_MAX = 2 * ALPHA_MAX / DT          # ~125

STYLE_AXES = ["prox", "pass", "yield", "group"]
SOCIAL_TERMS = ("g_prox", "g_rear", "g_side", "g_ttc", "g_cut", "g_group")

_BIG = 1.0e6


def ladder(s: torch.Tensor) -> torch.Tensor:
    """Hall ladder D(s).  s=-1 -> R_c (contact), 0 -> personal, +1 -> social.

    Log-linear on each side of 0, so the axis is continuous and monotone
    through neutral.  This is the ONE scale every axis reads (P3); it replaces
    the V6.5 intensity law `w = w0 * 2^(beta s)`, which multiplied
    non-negative penalty terms and therefore made the whole negative half of
    every axis a behavioral no-op.
    """
    s = s.clamp(-1.0, 1.0)
    return D_PER * torch.where(s <= 0, RHO_DN ** s, RHO_UP ** s)


def _active_mean(sev: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Mean of per-step severity over the steps where the term is ACTIVE.

    Used by g_cut only.  A gate that fires for 25% of the window would
    otherwise cap the term at 0.25 no matter how badly the norm is violated --
    an artifact of where the horizon falls relative to the encounter.  g_side
    deliberately does NOT use this (see its comment).
    """
    n = active.sum(dim=1)
    return torch.where(n > 0, sev.sum(dim=1) / n.clamp(min=1.0),
                       torch.zeros_like(n))


# =============================================================================
# Geometry -- style-INVARIANT, computed once per (scene, candidate set)
# =============================================================================
def _geometry(traj_xy, obs, obs_mask, start_theta=None):
    """Everything in the expert's geometry() that does not depend on s.

    Returns a dict of tensors on the plan+tail time axis (length H + K_TAIL).
    """
    B, T, _ = traj_xy.shape
    M = obs.shape[1]
    dev, dt = traj_xy.device, DT
    H = T - 1                                     # plan steps (index 0 = start)

    p0 = obs[:, :, 0:2]                           # (B,M,2) at t=0
    vj = obs[:, :, 2:4]                           # (B,M,2)
    valid = obs_mask.bool()                       # (B,M)

    # ---- robot state recovered from positions --------------------------------
    vel = (traj_xy[:, 1:] - traj_xy[:, :-1]) / dt          # (B,H,2)
    sp = vel.norm(dim=-1)                                   # (B,H)
    rd = torch.nn.functional.normalize(vel, dim=-1, eps=1e-6)
    th = torch.atan2(vel[..., 1], vel[..., 0])              # (B,H)

    rob = traj_xy[:, 1:]                                    # (B,H,2) t = 1..H

    # ---- bounded social lookahead (T_LOOK): robot coasts, peds continue ------
    # Without the tail the cost is INVERTED for a closing encounter: braking
    # defers the encounter past the window boundary instead of resolving it, so
    # stopping accumulates no exposure and scores best while producing the worst
    # outcome.  The tail is bounded at the Karamouzas interaction timescale.
    if K_TAIL > 0:
        k = torch.arange(1, K_TAIL + 1, device=dev, dtype=traj_xy.dtype) * dt
        dirT, vT = rd[:, -1], sp[:, -1]                     # terminal heading/speed
        tail = (traj_xy[:, -1:, :]
                + (vT[:, None] * k[None, :])[..., None] * dirT[:, None, :])
        rob = torch.cat([rob, tail], dim=1)
        rd = torch.cat([rd, dirT[:, None, :].expand(-1, K_TAIL, -1)], dim=1)
        sp = torch.cat([sp, vT[:, None].expand(-1, K_TAIL)], dim=1)

    Ht = rob.shape[1]                                        # H + K_TAIL
    tt = torch.arange(1, Ht + 1, device=dev, dtype=traj_xy.dtype) * dt
    hum = p0[:, None, :, :] + tt[None, :, None, None] * vj[:, None, :, :]  # (B,Ht,M,2)

    vr = sp[..., None] * rd                                  # (B,Ht,2) robot velocity
    hs = vj.norm(dim=-1)                                     # (B,M)
    mov = hs > V_MIN
    hd = vj / hs.clamp(min=1e-6)[..., None]                  # (B,M,2) garbage if !mov
    hl = torch.stack([-hd[..., 1], hd[..., 0]], dim=-1)      # left normal

    # rel = p_robot - p_ped  (NOTE the sign: the old trainer cost used the
    # opposite convention, which silently flips every signed quantity below)
    rel = rob[:, :, None, :] - hum                           # (B,Ht,M,2)
    d = rel.norm(dim=-1).clamp(min=1e-4)                     # (B,Ht,M)
    vrel = vr[:, :, None, :] - vj[:, None, :, :]             # (B,Ht,M,2)

    vmask = valid[:, None, :].to(traj_xy.dtype)              # (B,1,M)
    movf = mov[:, None, :].to(traj_xy.dtype) * vmask         # (B,1,M)

    # Padding must never win a max_j reduction: push it out of every zone.
    d = torch.where(valid[:, None, :], d, torch.full_like(d, _BIG))

    G = {"B": B, "Ht": Ht, "M": M, "d": d, "hs": hs, "movf": movf,
         "vmask": vmask, "valid": valid, "dtype": traj_xy.dtype, "dev": dev}

    # b_rear: 1 directly astern, 0 at the flank and ahead.  Purely radial x
    # angular -- no lateral component.
    G["b_rear"] = (-(rel * hd[:, None, :, :]).sum(-1) / d).clamp(0, 1) * movf

    # sigma: signed lateral offset in the PEDESTRIAN's frame (>0 = robot on
    # their left).  This frame is why ONE formula covers head-on and overtake.
    sig = (rel * hl[:, None, :, :]).sum(-1)                  # (B,Ht,M)
    G["sig"] = sig

    # side_gate: classification held from the INITIAL heading for the whole
    # encounter.  Recomputing it per step lets a candidate switch the term off
    # by doing the very thing being judged (turning to commit to a side).
    if start_theta is None:
        # ego frame: the robot starts at the origin facing +x
        rd0 = torch.zeros(B, 2, device=dev, dtype=traj_xy.dtype)
        rd0[:, 0] = 1.0
    else:
        rd0 = torch.stack([torch.cos(start_theta), torch.sin(start_theta)], -1)
    aligned0 = ((rd0[:, None, :] * hd).sum(-1).abs() >= COS_PSI)          # (B,M)
    G["side_gate"] = ((d < D_SOC) & mov[:, None, :] & aligned0[:, None, :]
                      & valid[:, None, :]).to(traj_xy.dtype)

    G["s_par"] = (rel * hd[:, None, :, :]).sum(-1)                        # lead along path
    G["lat"] = (1 - sig.abs() / D_PER).clamp(0, 1)
    G["block"] = (1 - (vr[:, :, None, :] * hd[:, None, :, :]).sum(-1)
                  / hs.clamp(min=1e-6)[:, None, :]).clamp(0, 1)

    # tau: exact time until ||p_j - p_r|| = R_c under constant relative
    # velocity.  C <= 0 (already touching) -> tau = 0.  V6.5 returned the EXIT
    # time here, scoring the worst possible violation as perfectly safe.
    A = (vrel ** 2).sum(-1)
    Bq = 2.0 * (rel * vrel).sum(-1)
    C = d ** 2 - R_C ** 2
    disc = Bq ** 2 - 4 * A * C
    ok = (A > 1e-8) & (Bq < 0) & (disc >= 0)
    tau = torch.where(
        ok, (-Bq - disc.clamp(min=0).sqrt()) / (2 * A.clamp(min=1e-8)),
        torch.full_like(A, float("inf")))
    tau = torch.where(C <= 0, torch.zeros_like(tau), tau)
    G["tau"] = torch.where(valid[:, None, :], tau, torch.full_like(tau, float("inf")))

    # ---- formations: co-motion is the ONLY style-invariant physical fact ----
    # ||v_i - v_j|| <= DV_MAX covers walking groups (small dv) and standing
    # groups (dv = 0) with NO orientation, which is why obs_dim = 4 suffices and
    # no F-formation detector is claimed.  Membership radius is applied later
    # (it is style-dependent); candidates are gated at D_soc here.
    if M >= 2:
        sep0 = (p0[:, :, None, :] - p0[:, None, :, :]).norm(dim=-1)       # (B,M,M)
        dv = (vj[:, :, None, :] - vj[:, None, :, :]).norm(dim=-1)
        triu = torch.triu(torch.ones(M, M, dtype=torch.bool, device=dev), 1)
        both = valid[:, :, None] & valid[:, None, :]
        pair = (sep0 < D_SOC) & (dv <= DV_MAX) & triu[None] & both        # (B,M,M)

        seg = hum[:, :, None, :, :] - hum[:, :, :, None, :]               # (B,Ht,M,M,2) j - i
        L2 = (seg ** 2).sum(-1).clamp(min=1e-6)
        ri = rob[:, :, None, None, :] - hum[:, :, :, None, :]             # robot - i
        u = ((ri * seg).sum(-1) / L2).clamp(0.0, 1.0)
        G["useg"] = u
        G["dseg"] = (ri - u[..., None] * seg).norm(dim=-1)                # (B,Ht,M,M)
        G["sep0"] = sep0
        G["pair"] = pair

    # ---- g_smooth: on the PLAN only, the tail carries no controls -----------
    om = torch.atan2(torch.sin(th[:, 1:] - th[:, :-1]),
                     torch.cos(th[:, 1:] - th[:, :-1])) / dt              # (B,H-1)
    a = (sp[:, 1:H] - sp[:, :H - 1]) / dt if H >= 2 else sp[:, :0]
    jerk = (a[:, 1:] - a[:, :-1]) / dt if a.shape[1] >= 2 else a[:, :0]
    al = (om[:, 1:] - om[:, :-1]) / dt if om.shape[1] >= 2 else om[:, :0]
    ajerk = (al[:, 1:] - al[:, :-1]) / dt if al.shape[1] >= 2 else al[:, :0]
    zj = torch.zeros(B, device=dev, dtype=traj_xy.dtype)
    gj = (jerk.abs() / JERK_MAX).clamp(0, 1).mean(1) if jerk.shape[1] else zj
    ga = (ajerk.abs() / AJERK_MAX).clamp(0, 1).mean(1) if ajerk.shape[1] else zj
    G["g_sm"] = 0.5 * gj + 0.5 * ga
    return G


# =============================================================================
# The six social terms.  P1: each is mean_t max_j (severity in [0,1]).
# P3: each reads the Hall ladder exactly once.  P4: all equally weighted.
# =============================================================================
def _terms(G, s):
    B, M = G["B"], G["M"]
    out = {"g_smooth": G["g_sm"]}
    if M == 0:
        z = torch.zeros(B, device=G["dev"], dtype=G["dtype"])
        for k in SOCIAL_TERMS:
            out[k] = z.clone()
        return out

    D_prox = ladder(s[:, 0])[:, None, None]          # prox : zone radius, point
    tau_a = (ladder(s[:, 2]) / V_BAR)[:, None, None]  # yield: anticipation time
    D_grp = ladder(s[:, 3])[:, None, None]           # group: zone radius, segment

    # ---- g_prox (prox) ------------------------------------------------------
    # Penetration normalized by the ZONE DEPTH, so the ramp spans the whole zone
    # at every style: 0 at the wall d = D_prox, 1 at contact d = R_c.  W_ZONE
    # floors the denominator so sub-neutral styles stay non-singular.  Vanishes
    # identically at s=-1 (D_prox = R_c and every feasible traj has d >= R_c).
    # mean_t, NOT max_t: in a receding horizon the closest approach is fixed by
    # the first step, so max_t punishes closing a gap but never rewards opening
    # one and the robot freezes instead of keeping space.
    zone = ((D_prox - G["d"]) / (D_prox - R_C).clamp(min=W_ZONE)).clamp(0, 1)
    out["g_prox"] = zone.max(dim=2).values.mean(dim=1)

    # ---- g_rear (prox) ------------------------------------------------------
    # Same zone, gated by how directly astern the robot is.  g_rear <= g_prox
    # pointwise; together they ARE an anisotropic personal space.
    out["g_rear"] = (G["b_rear"] * zone).max(dim=2).values.mean(dim=1)

    # ---- g_side (pass) ------------------------------------------------------
    # Pass is the one axis with no scale to move: for a CONVENTION the neutral
    # stance is NO convention.  |s_pass| weights, sign(s_pass) picks the side,
    # continuous through zero, PSB(0) = 0 by construction.
    # Hinge over sigma in [-R_c, +R_c] (not [-D_per, +D_per]): R_c is the offset
    # at which the robot physically clears the pedestrian; anything wider is
    # CLEARANCE, which g_prox owns.
    sp_ = s[:, 1][:, None, None]
    hinge = ((R_C - torch.sign(sp_) * G["sig"]) / (2 * R_C)).clamp(0, 1)
    gs = (G["side_gate"] * hinge).max(dim=2).values.mean(dim=1)
    # NOTE mean over the WHOLE plan, deliberately NOT _active_mean: the hinge is
    # 0.5 at sigma = 0 by design ("on their line, undecided") and that residual
    # is only meaningful as a gradient.  Conditioning on active steps promotes it
    # into a large, nearly side-INDEPENDENT constant and noise picks the side.
    out["g_side"] = torch.where(s[:, 1].abs() < 1e-6,
                                torch.zeros_like(gs), s[:, 1].abs() * gs)

    # ---- g_ttc (yield) ------------------------------------------------------
    # Severity = (1 + tau/tau_a)^-2.  The tau^-2 exponent is MEASURED
    # (Karamouzas PRL 2014, R^2 = 0.92-0.94); the +1 regularizes tau -> 0 at the
    # human reaction floor.  Unlike prox/rear/group this does not vanish at
    # s=-1: the scale is a HORIZON, not a magnitude.  s_yield sets how EARLY you
    # react, not whether.
    ttc = 1.0 / (1.0 + G["tau"] / tau_a) ** 2
    ttc = torch.nan_to_num(ttc, nan=0.0, posinf=0.0, neginf=0.0)
    out["g_ttc"] = ttc.max(dim=2).values.mean(dim=1)

    # ---- g_cut (yield) ------------------------------------------------------
    # The pedestrian's forward space: their personal zone extruded along their
    # path by one anticipation time.  ONE parameter (tau_a) drives both yield
    # terms, so temporal collision and spatial priority are complementary rather
    # than the redundant pair V6.5 had.  Three zero-cost escapes, all correct:
    # swerve out (lat -> 0), drop behind (s_par < 0), out-run them (block -> 0).
    L = G["hs"][:, None, :] * tau_a
    inside = ((G["s_par"] > 0) & (G["s_par"] < L)).to(G["dtype"])
    sev = (inside * G["lat"] * G["block"] * G["movf"]).max(dim=2).values
    act = (inside * G["movf"]).max(dim=2).values > 0
    out["g_cut"] = _active_mean(sev, act.to(G["dtype"]))

    # ---- g_group (group) ----------------------------------------------------
    # The pair is one extended obstacle: the segment between the members,
    # inflated by D(s_group), read through the same zone-depth severity as
    # g_prox.  The interior weight 4u(1-u) is what expresses "don't SPLIT":
    # without it, distance-to-segment for a line-abreast group approached
    # head-on is identical whether the robot aims at the middle or the edge.
    # Membership scales with style: at s=-1 only touching pairs (term vanishes),
    # at s=+1 loose formations qualify and get a wide shared zone.
    if "dseg" in G:
        member = (G["sep0"] < D_grp).to(G["dtype"])                 # (B,M,M)
        gate = (member * G["pair"].to(G["dtype"]))[:, None, :, :]
        interior = 4.0 * G["useg"] * (1.0 - G["useg"])
        pen = ((D_grp[..., None] - G["dseg"])
               / (D_grp[..., None] - R_C).clamp(min=W_ZONE)).clamp(0, 1)
        out["g_group"] = (gate * interior * pen).amax(dim=(2, 3)).mean(dim=1)
    else:
        out["g_group"] = torch.zeros(B, device=G["dev"], dtype=G["dtype"])
    return out


# =============================================================================
# Public entry point
# =============================================================================
@torch.no_grad()
def social_cost(traj_xy, obs, obs_mask, style=None, goal=None,
                start_theta=None, return_terms=True):
    """J = J_goal + W_SOCIAL * sum(six equal social terms) + LAM_SM * g_smooth.

    `style=None` means the NEUTRAL style s = 0 (Hall personal zone), which is
    the correct fixed reference under this cost -- there is no separate
    "neutral weight" table any more, because style no longer scales weights, it
    moves one shared Hall ladder.

    Returns (cost, terms) with cost shape (B,) and terms a dict of (B,) tensors.
    """
    B = traj_xy.shape[0]
    dev, dtype = traj_xy.device, traj_xy.dtype
    if style is None:
        style = torch.zeros(B, 4, device=dev, dtype=dtype)
    style = style.to(dtype).clamp(-1, 1)

    G = _geometry(traj_xy, obs.to(dtype), obs_mask, start_theta)
    tm = _terms(G, style)

    social = sum(tm[k] for k in SOCIAL_TERMS)
    cost = W_SOCIAL * social + LAM_SM * tm["g_smooth"]

    if goal is not None:
        # J_goal normalized so an ideal max-speed run costs 0 and STALLING costs
        # exactly 1 (P5).  EUCLIDEAN substitute for the expert's geodesic -- see
        # the module docstring; valid for ranking candidates within one scene.
        delta = (traj_xy - goal[:, None, :]).norm(dim=-1)          # (B,T)
        prog = (delta[:, :1] - delta[:, 1:]).sum(dim=1)
        j_goal = 1.0 - prog / PROG_MAX
        cost = cost + j_goal
        tm["J_goal"] = j_goal

    tm["social_sum"] = social
    return (cost, tm) if return_terms else cost