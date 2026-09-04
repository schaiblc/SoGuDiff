"""
Expert Trajectory Visualizer — V7 (norm-grounded cost) port
=============================================================
Companion to generate_expert_trajectories.py.

This is a straight port of visualize_expert_trajs.py (the V6.5-fast
visualizer) onto the V7 cost model. The panel layout, color scheme, and
figure set are UNCHANGED. What changed is the data plumbing, because V7's
cost function is structurally different from V6.5's:

  * V6.5 had 8 named social terms (c1..c8) evaluated through per-style
    "effective weights" (style_to_effective_weights), so raw magnitudes
    were NOT comparable across styles — hence the old visualizer's whole
    "_reference_eff / VIS_USE_NEUTRAL_COST_SCALE / shared scale" apparatus,
    which recomputed every trajectory under one fixed weight set just to
    make styles comparable, and zeroed out c1_keep_side in the process
    (a bug the old file's docstring calls out).

  * V7 has 6 social terms + smoothness (g_prox, g_rear, g_side, g_ttc,
    g_cut, g_group, g_smooth), ALL at weight 1 (design principle P4 in the
    generator's docstring). Style only rescales the Hall-ladder zone/time
    each term reads, so a term's raw value is already on one shared
    [0, ~1]-ish scale for every style — there is nothing to rescale. The
    "shared scale" machinery is therefore deleted outright rather than
    ported: every plot here evaluates each trajectory at its OWN style
    vector and the numbers are directly comparable as-is.

  * g_side (V7's analogue of c1_keep_side) is `|s_pass| * hinge`, i.e.
    already non-negative by construction — unlike c1_keep_side it carries
    no sign, so the old radar chart's "+/-" annotation at that vertex is
    dropped.

  * obs_headings doesn't exist in V7 (pedestrian direction comes from
    obs_vel only), so it's removed from every function signature.

  * V7 has no complexity-tier gate (P(s|scene) is uniform — see the
    generator's docstring), so `complexity_tier` is dropped from titles
    and no longer threaded through.

Figures produced per scene (same five categories as before):
  1. cost_landscape.png          — Phase-1 pool colored by neutral cost
  2. style_comparison.png        — one panel per style, all saved modes
  3. cost_breakdown.png          — stacked bar + progress/total comparison
  4. style_radar.png             — radar of the 7 cost terms per style
  5. cost_fields__<tag>.png      — per-term cost-field plots per style,
                                   each evaluated at that style's own weights

Usage
-----
  # Live generation (add --visualize to your run):
  python generate_expert_trajectories.py \\
      --in_dir raw/ --out_dir expert/ --limit 5 --visualize

  # Standalone on already-generated .npz outputs:
  python visualize_expert_trajs_v7.py \\
      --in_dir  raw_scenes/ \\
      --out_dir expert_v7/ \\
      --vis_dir vis_output/ \\
      --limit   10
"""

import argparse
import os
import glob
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

# Cost machinery is shared with the expert generator, so visualizations are
# scored by exactly the same functions that produced the demonstrations.
try:
    from generate_expert_trajectories import (
        total_cost, geometry, ladder, rollout,
        HORIZON, DT, MAP_SIZE, MAP_RES, MAP_EXTENT, R_C,
        STYLE_AXES, NEUTRAL, SOCIAL, LAM_SM, W_SOCIAL,
        load_scene, build_geodesic, sample_geo, map_clearances,
        sample_uniform, sample_smooth, brake_controls,
        N_UNIFORM, N_SMOOTH, sample_styles, seed_of,
        obs_positions,
    )
    _GEN_AVAILABLE = True
except ImportError:
    _GEN_AVAILABLE = False
    print("[visualizer] generator module not importable — "
          "standalone mode (reading saved .npz outputs only).")
    # Minimal stubs so the file stays importable
    HORIZON = 32; DT = 0.05; MAP_SIZE = 50; MAP_RES = 0.2
    MAP_EXTENT = 10; R_C = 0.50
    STYLE_AXES = ["prox", "pass", "yield", "group"]
    NEUTRAL = np.zeros(4, dtype=np.float32)
    SOCIAL = ("g_prox", "g_rear", "g_side", "g_ttc", "g_cut", "g_group")
    LAM_SM = 0.05
    W_SOCIAL = 2.0
    N_UNIFORM, N_SMOOTH = 4096, 12288

N_STYLE_AXES = len(STYLE_AXES)

# ── aesthetic constants (unchanged from the V6.5 visualizer) ──────────────
COST_CMAP = "plasma"
MAP_CMAP  = "Greys"
STYLE_COLORS = [
    "#4FC3F7", "#81C784", "#FFB74D", "#F06292",
    "#CE93D8", "#80DEEA", "#FFCC02", "#A5D6A7",
    "#FF8A65", "#90CAF9", "#EF9A9A", "#A1887F",
]
# V7's seven cost terms: six social (equal to each other, the block scaled by
# W_SOCIAL against J_goal) + smoothness (LAM_SM).
TERM_COLORS = {
    "progress": "#4FC3F7",
    "g_prox":   "#26A69A",
    "g_rear":   "#EF5350",
    "g_side":   "#FF7043",
    "g_ttc":    "#FFA726",
    "g_cut":    "#AB47BC",
    "g_group":  "#5C6BC0",
    "g_smooth": "#78909C",
}
TERM_LABELS = {
    "progress": "Progress",
    "g_prox":   "Proxemic Clearance",
    "g_rear":   "Rear / Blind Zone",
    "g_side":   "Passing Side",
    "g_ttc":    "Time-to-Collision",
    "g_cut":    "Cut-Off",
    "g_group":  "Group Split",
    "g_smooth": "Smoothness",
}
# All seven cost terms besides progress (mirrors old SOCIAL_TERMS, which
# also had 7 entries: c1..c5, c7, c8). g_smooth is included here even though
# it carries LAM_SM=0.05 rather than the social block's W_SOCIAL.
SOCIAL_TERMS = list(SOCIAL) + ["g_smooth"]

FIG_DPI = 150
DARK_BG = "#0D1117"
PANEL_BG = "#161B22"


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _world_extent():
    half = MAP_EXTENT / 2.0
    return [-half, half, -half, half]


def _draw_map(ax, occ, clear=None, alpha_occ=0.55):
    ext = _world_extent()
    if occ is not None and occ.any():
        ax.imshow(occ, origin="lower", extent=ext,
                  cmap=MAP_CMAP, vmin=0, vmax=1, alpha=alpha_occ, zorder=1)
    if clear is not None:
        finite = clear[np.isfinite(clear)]
        if finite.size:
            vmax = np.percentile(finite, 90)
            ax.imshow(clear, origin="lower", extent=ext,
                      cmap="YlOrRd_r", vmin=0, vmax=vmax, alpha=0.18, zorder=2)


def _draw_obstacles(ax, obs, obs_traj):
    if obs is None or len(obs) == 0:
        return
    for m in range(len(obs)):
        xs, ys = obs_traj[:, m, 0], obs_traj[:, m, 1]
        ax.plot(xs, ys, color="#FF6B6B", lw=0.8, alpha=0.45, zorder=4)
        circle = mpatches.Circle(
            (xs[0], ys[0]), radius=0.50,
            color="#FF6B6B", alpha=0.20, zorder=5,
            linewidth=0.8, edgecolor="#FF6B6B", fill=True,
        )
        ax.add_patch(circle)
        ax.scatter(xs[0], ys[0], s=18, c="#FF6B6B",
                   edgecolors="white", lw=0.4, zorder=6)
        vx, vy = obs[m, 2], obs[m, 3]
        if abs(vx) + abs(vy) > 1e-3:
            ax.annotate("", xy=(xs[0]+vx*0.5, ys[0]+vy*0.5),
                        xytext=(xs[0], ys[0]),
                        arrowprops=dict(arrowstyle="->", color="#FF6B6B",
                                        lw=1.2, mutation_scale=10), zorder=7)


def _draw_goal(ax, goal):
    ax.scatter(goal[0], goal[1], marker="*", s=200, c="#FFD700",
               edgecolors="black", lw=0.7, zorder=10)


def _draw_start(ax, start_state):
    x, y = start_state[0], start_state[1]
    theta = start_state[2] if len(start_state) > 2 else 0.0
    v = start_state[3] if len(start_state) > 3 else 0.0
    size = 0.22
    tip   = np.array([x + size*np.cos(theta),       y + size*np.sin(theta)])
    left  = np.array([x + size*np.cos(theta+2.4),   y + size*np.sin(theta+2.4)])
    right = np.array([x + size*np.cos(theta-2.4),   y + size*np.sin(theta-2.4)])
    tri   = plt.Polygon([tip, left, right], color="#00E5FF", ec="black", lw=0.6, zorder=10)
    ax.add_patch(tri)
    arrow_len = 0.45
    ax.annotate("",
        xy=(x + arrow_len*np.cos(theta), y + arrow_len*np.sin(theta)),
        xytext=(x, y),
        arrowprops=dict(arrowstyle="->", color="#00E5FF", lw=1.6, mutation_scale=12),
        zorder=11)
    ax.text(x+0.07, y+0.07, f"v={v:.1f}", fontsize=5, color="#00E5FF", zorder=12,
            path_effects=[pe.Stroke(linewidth=1.5, foreground="black"), pe.Normal()])


def _styled_traj_line(ax, xy, color, lw=2.2, alpha=1.0, zorder=8, label=None):
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, alpha=alpha,
            zorder=zorder, label=label,
            path_effects=[pe.Stroke(linewidth=lw+1.5, foreground="black", alpha=alpha*0.55),
                          pe.Normal()])


def _set_scene_ax(ax, title, start_state, goal, title_color="white"):
    half = MAP_EXTENT / 2.0
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8, pad=3, color=title_color)
    ax.tick_params(colors="#888", labelsize=6)
    ax.grid(True, lw=0.2, alpha=0.3)
    _draw_goal(ax, goal)
    _draw_start(ax, start_state)


def _style_label(tag, vec):
    if tag == "neutral":
        return "Neutral  [0,0,0,0]"
    names = ["Pr", "Pa", "Yi", "Gr"]
    parts = [f"{n}{'↑' if v > 0 else '↓'}{abs(v):.1f}"
             for n, v in zip(names, vec) if abs(v) > 0.05]
    vec_str = "[" + ",".join(f"{v:+.1f}" for v in vec) + "]"
    return f"{tag}  {vec_str}" if not parts else f"{','.join(parts)}  {vec_str}"


# ══════════════════════════════════════════════════════════════════════════
# Cost breakdown for a single trajectory, at ITS OWN style's weights.
#
# There is no per-style weight rescaling (P4: the six social terms are equal to
# each other; style only moves the Hall-ladder scale each term reads), so unlike
# the V6.5 visualizer no separate "shared reference scale" is needed to make
# styles comparable — total_cost(..., breakdown=True) at the trajectory's own
# style vector already returns numbers on one common scale for every style.
#
# We return WEIGHTED CONTRIBUTIONS, not raw severities: each social term is
# multiplied by W_SOCIAL and g_smooth by LAM_SM, exactly as total_cost sums
# them.  That is what makes the stacked bars / radar / inset add up to the
# reported total.  (total_cost's `bd` holds raw [0,1] severities; the per-term
# COST FIELD panels plot those raw values instead, since each has its own
# colorbar and no cross-term summation is implied there.)
# ══════════════════════════════════════════════════════════════════════════

def _cost_breakdown(traj_states, obs_traj, obs_vel, geo, clear, style):
    """style=None is treated as neutral.  Values are weighted contributions."""
    if not _GEN_AVAILABLE:
        return {}
    s = NEUTRAL if style is None else np.asarray(style, dtype=np.float32)
    _, _, bd = total_cost(traj_states[None], obs_traj, obs_vel, geo, clear, s,
                          breakdown=True)
    out = {"progress": float(bd["J_goal"][0])}
    for term in SOCIAL_TERMS:
        w = LAM_SM if term == "g_smooth" else W_SOCIAL
        out[term] = float(bd[term][0]) * w
    return out


# ══════════════════════════════════════════════════════════════════════════
# Output 1 - Phase-1 cost landscape (neutral coloring)
# ══════════════════════════════════════════════════════════════════════════

def plot_cost_landscape(
    scene_name, vis_dir,
    start_state, goal, obs, obs_traj, obs_vel,
    occ, clear, geo,
    all_traj, all_costs,
    best_traj_neutral=None,
    max_trajs_shown=2000,
):
    os.makedirs(vis_dir, exist_ok=True)
    finite_mask = np.isfinite(all_costs)
    fin_idx = np.where(finite_mask)[0]
    inf_idx = np.where(~finite_mask)[0]

    rng_vis = np.random.default_rng(42)
    show_fin = min(len(fin_idx), max_trajs_shown)
    show_inf = min(len(inf_idx), max_trajs_shown // 5)
    if show_fin < len(fin_idx):
        fin_idx = rng_vis.choice(fin_idx, size=show_fin, replace=False)
    if show_inf < len(inf_idx):
        inf_idx = rng_vis.choice(inf_idx, size=show_inf, replace=False)

    fin_costs = all_costs[fin_idx]
    vmin = np.percentile(fin_costs, 5)  if fin_costs.size else 0
    vmax = np.percentile(fin_costs, 95) if fin_costs.size else 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(COST_CMAP)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=FIG_DPI)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    _draw_map(ax, occ, clear)
    _draw_obstacles(ax, obs, obs_traj)

    for i in inf_idx:
        xy = all_traj[i, :, 0:2]
        ax.plot(xy[:, 0], xy[:, 1], color="#444", lw=0.35,
                alpha=0.22, zorder=3, linestyle="--")

    for i in fin_idx:
        col = cmap(norm(all_costs[i]))
        xy  = all_traj[i, :, 0:2]
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=0.65, alpha=0.50, zorder=4)

    if best_traj_neutral is not None:
        _styled_traj_line(ax, best_traj_neutral[:, 0:2], "#00FF88",
                          lw=2.5, zorder=9, label="Best neutral")

    _set_scene_ax(ax, f"Phase-1 Cost Landscape  |  {scene_name}", start_state, goal)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Neutral total cost (J_goal + social + smooth)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    leg = [
        mpatches.Patch(color="#444",    alpha=0.5, label=f"Infeasible ({len(inf_idx)} shown)"),
        mpatches.Patch(color=cmap(0.1), label=f"Low cost  (p5={vmin:.2f})"),
        mpatches.Patch(color=cmap(0.9), label=f"High cost (p95={vmax:.2f})"),
        mpatches.Patch(color="#00FF88", label="Best neutral"),
        mpatches.Patch(color="#FF6B6B", label="Pedestrians"),
        mpatches.Patch(color="#FFD700", label="Goal"),
    ]
    ax.legend(handles=leg, loc="upper right", fontsize=6,
              framealpha=0.35, labelcolor="white",
              facecolor="#1A1E26", edgecolor="#444")

    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    out = os.path.join(vis_dir, f"{scene_name}__cost_landscape.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [vis] saved {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# Output 2 - Per-term cost field plots  (one PNG per active style)
# ══════════════════════════════════════════════════════════════════════════

def plot_traj_cost_fields(
    scene_name, vis_dir,
    start_state, goal, obs, obs_traj, obs_vel,
    occ, clear, geo,
    all_traj,
    style=None,
    style_tag="neutral",
    saved_mode_trajs=None,    # list of (H+1,5) arrays — the actually saved modes
    max_trajs=1500,
):
    """
    9-panel figure: total, progress (J_goal), and the 7 V7 cost terms
    (g_prox, g_rear, g_side, g_ttc, g_cut, g_group, g_smooth).
    Saved mode trajectories are overlaid in white (mode-0) / yellow (mode-1+).

    Always evaluated at this style's OWN vector (V7 has no per-style weight
    rescaling to toggle — see the module docstring), via a single
    total_cost(..., breakdown=True) call that returns every term at once.
    """
    os.makedirs(vis_dir, exist_ok=True)
    if not _GEN_AVAILABLE:
        return None

    N = all_traj.shape[0]
    rng_v = np.random.default_rng(99)
    idx   = rng_v.choice(N, size=min(N, max_trajs), replace=False)
    trajs = all_traj[idx]

    eval_style = NEUTRAL if style is None else np.asarray(style, dtype=np.float32)
    total, feas, bd = total_cost(trajs, obs_traj, obs_vel, geo, clear, eval_style,
                                 breakdown=True)

    terms = [
        ("total",    total,          "RdYlGn_r", "TOTAL cost"),
        ("progress", bd["J_goal"],   "RdYlGn_r", "Progress (J_goal)"),
    ]
    for k in SOCIAL_TERMS:
        cmap_name = "plasma" if k == "g_smooth" else "inferno"
        terms.append((k, bd[k], cmap_name, TERM_LABELS[k]))

    ncols = 3
    nrows = (len(terms) + ncols - 1) // ncols
    half  = MAP_EXTENT / 2.0

    fig = plt.figure(figsize=(6.5*ncols, 5.5*nrows + 1.2), dpi=FIG_DPI)
    fig.patch.set_facecolor(DARK_BG)

    mode_overlay_colors = ["white", "#FFD700", "#00FF88", "#FF6EC7", "#7DF9FF"]

    for i, (key, costs_raw, cmap_name, label) in enumerate(terms):
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.set_facecolor(DARK_BG)

        costs  = np.asarray(costs_raw, dtype=np.float32)
        finite = np.isfinite(costs)

        _draw_map(ax, occ, alpha_occ=0.30)
        _draw_obstacles(ax, obs, obs_traj)

        if finite.any():
            vmin = np.percentile(costs[finite], 2)
            vmax = np.percentile(costs[finite], 98)
            if key in ("progress", "total"):
                bound = max(abs(vmin), abs(vmax))
                vmin, vmax = -bound, bound
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap_obj = cm.get_cmap(cmap_name)

            for j in np.where(~finite)[0]:
                ax.plot(trajs[j, :, 0], trajs[j, :, 1],
                        color="#333", lw=0.3, alpha=0.15, zorder=3)

            fin_order = np.where(finite)[0]
            fin_order = fin_order[np.argsort(costs[fin_order])]
            for j in fin_order:
                col = cmap_obj(norm(costs[j]))
                ax.plot(trajs[j, :, 0], trajs[j, :, 1],
                        color=col, lw=0.65, alpha=0.55, zorder=4)

            sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.02)
            cbar.ax.yaxis.set_tick_params(color="white", labelsize=5)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

            # Annotate best/worst for total panel
            if key == "total":
                best_j  = fin_order[0]
                worst_j = fin_order[-1]
                ax.plot(trajs[best_j, :, 0], trajs[best_j, :, 1],
                        color="#00FF88", lw=1.8, alpha=0.85, zorder=6,
                        path_effects=[pe.Stroke(linewidth=3.0, foreground="black",
                                               alpha=0.5), pe.Normal()])
                ax.text(0.02, 0.02,
                        f"best={costs[fin_order[0]]:.2f}  worst={costs[fin_order[-1]]:.2f}",
                        transform=ax.transAxes, fontsize=5.5, color="white",
                        path_effects=[pe.Stroke(linewidth=1.5, foreground="black"),
                                      pe.Normal()])

        # ── Overlay saved mode trajectories ───────────────────────────────
        if saved_mode_trajs:
            for mi, mode_traj in enumerate(saved_mode_trajs):
                mc  = mode_overlay_colors[mi % len(mode_overlay_colors)]
                mlw = 2.5 - mi * 0.3
                alpha_m = max(0.5, 1.0 - mi * 0.15)
                xy = mode_traj[:, 0:2]
                ax.plot(xy[:, 0], xy[:, 1], color=mc, lw=mlw, alpha=alpha_m,
                        zorder=10 + mi,
                        path_effects=[pe.Stroke(linewidth=mlw+1.5,
                                               foreground="black", alpha=0.5),
                                      pe.Normal()])
                ax.scatter(xy[-1, 0], xy[-1, 1], s=25, c=mc,
                           edgecolors="black", lw=0.5, zorder=12+mi)
                ax.text(xy[-1, 0]+0.05, xy[-1, 1]+0.05, f"m{mi}",
                        fontsize=5, color=mc, zorder=13+mi,
                        path_effects=[pe.Stroke(linewidth=1.2, foreground="black"),
                                      pe.Normal()])

        _draw_goal(ax, goal)
        _draw_start(ax, start_state)

        border_lw  = 2.0 if key == "total" else 0.5
        border_col = "#FFD700" if key == "total" else "#333"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(border_lw)

        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_title(label, color="white", fontsize=8, pad=3,
                     fontweight="bold" if key == "total" else "normal")
        ax.tick_params(colors="#888", labelsize=6)
        ax.grid(True, lw=0.2, alpha=0.25)

    style_vec_str = "neutral" if style is None else str(np.round(eval_style, 2).tolist())
    n_modes = len(saved_mode_trajs) if saved_mode_trajs else 0
    fig.suptitle(
        f"Per-term cost fields (RAW severities, unweighted)  |  {scene_name}  |  style: {style_tag}  "
        f"|  {style_vec_str}  |  {n_modes} mode(s) saved  "
        f"|  fields evaluated at this style's own weights",
        fontsize=10, color="white", fontweight="bold", y=0.995,
    )

    safe_tag = style_tag.replace("+", "plus").replace("-", "minus").replace(" ", "_")
    out = os.path.join(vis_dir, f"{scene_name}__cost_fields__{safe_tag}.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [vis] saved {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# Output 3 - Style comparison grid  (all modes per style)
# ══════════════════════════════════════════════════════════════════════════

def plot_style_comparison(
    scene_name, vis_dir,
    start_state, goal, obs, obs_traj, obs_vel,
    occ, clear, geo,
    style_results,
    # dict: tag -> {
    #   "vec": ndarray, "modes": list[ndarray (H+1,5)],
    #   "costs": list[float], "breakdown": dict|None, "reason": str
    # }
    all_traj=None, all_costs=None,
    max_bg_trajs=600,
):
    os.makedirs(vis_dir, exist_ok=True)
    tags = list(style_results.keys())
    n    = len(tags)
    if n == 0:
        return None

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig   = plt.figure(figsize=(6*ncols, 5.5*nrows + 1.4), dpi=FIG_DPI)
    fig.patch.set_facecolor(DARK_BG)

    gs = GridSpec(nrows, ncols, figure=fig,
                  hspace=0.45, wspace=0.28,
                  top=0.93, bottom=0.05, left=0.04, right=0.97)

    bg_cmap    = cm.get_cmap(COST_CMAP)
    bg_norm    = None
    bg_fin_idx = np.array([], dtype=int)
    bg_inf_idx = np.array([], dtype=int)
    if all_traj is not None and all_costs is not None:
        fin_mask = np.isfinite(all_costs)
        fin_idx  = np.where(fin_mask)[0]
        inf_idx  = np.where(~fin_mask)[0]
        rng_bg   = np.random.default_rng(7)
        bg_n     = min(len(fin_idx), max_bg_trajs)
        bg_fin_idx = (rng_bg.choice(fin_idx, size=bg_n, replace=False)
                      if bg_n < len(fin_idx) else fin_idx)
        bg_inf_n  = min(len(inf_idx), max_bg_trajs // 5)
        bg_inf_idx = (rng_bg.choice(inf_idx, size=bg_inf_n, replace=False)
                      if bg_inf_n < len(inf_idx) else inf_idx)
        if fin_idx.size:
            fc = all_costs[fin_idx]
            bg_norm = mcolors.Normalize(vmin=np.percentile(fc, 5),
                                        vmax=np.percentile(fc, 95))

    mode_colors = ["white", "#FFD700", "#00FF88", "#FF6EC7", "#7DF9FF"]

    for i, tag in enumerate(tags):
        row, col = divmod(i, ncols)
        ax  = fig.add_subplot(gs[row, col])
        ax.set_facecolor(DARK_BG)
        res = style_results[tag]
        accent_color = STYLE_COLORS[i % len(STYLE_COLORS)]

        if bg_norm is not None:
            for bi in bg_inf_idx:
                xy = all_traj[bi, :, 0:2]
                ax.plot(xy[:, 0], xy[:, 1], color="#3A3A3A", lw=0.3,
                        alpha=0.18, zorder=2, linestyle="--")
            for bi in bg_fin_idx:
                col_bg = bg_cmap(bg_norm(all_costs[bi]))
                xy = all_traj[bi, :, 0:2]
                ax.plot(xy[:, 0], xy[:, 1], color=col_bg, lw=0.45,
                        alpha=0.30, zorder=3)

        _draw_map(ax, occ, alpha_occ=0.30)
        _draw_obstacles(ax, obs, obs_traj)

        modes   = res.get("modes", [])
        costs_l = res.get("costs", [])
        reason  = res.get("reason", "?")

        if not modes:
            ax.text(0.5, 0.5, f"FAILED\n{reason}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="#FF6B6B",
                    fontweight="bold",
                    path_effects=[pe.Stroke(linewidth=2, foreground="black"),
                                  pe.Normal()])
            title_str = f"FAILED: {reason}"
        else:
            for mi, (mode_traj, mode_cost) in enumerate(zip(modes, costs_l)):
                mc    = mode_colors[mi % len(mode_colors)]
                mlw   = 2.8 - mi * 0.35
                alpha_m = max(0.45, 1.0 - mi * 0.18)
                xy    = mode_traj[:, 0:2]
                _styled_traj_line(ax, xy, mc, lw=mlw, alpha=alpha_m, zorder=8+mi,
                                  label=f"mode{mi} cost={mode_cost:.2f}")
                prog_idx = int(0.6 * (xy.shape[0]-1))
                if prog_idx > 0:
                    dx = xy[prog_idx, 0] - xy[prog_idx-1, 0]
                    dy = xy[prog_idx, 1] - xy[prog_idx-1, 1]
                    ax.annotate("",
                        xy=(xy[prog_idx,0]+dx*4, xy[prog_idx,1]+dy*4),
                        xytext=(xy[prog_idx,0], xy[prog_idx,1]),
                        arrowprops=dict(arrowstyle="->", color=mc,
                                        lw=1.4, mutation_scale=11), zorder=11+mi)
                ax.scatter(xy[-1,0], xy[-1,1], s=30, c=mc,
                           edgecolors="black", lw=0.5, zorder=12+mi)
                ax.text(xy[-1,0]+0.06, xy[-1,1]+0.06,
                        f"m{mi}: {mode_cost:.2f}", fontsize=5.5, color=mc,
                        zorder=13+mi,
                        path_effects=[pe.Stroke(linewidth=1.5, foreground="black"),
                                      pe.Normal()])

            best_cost = costs_l[0] if costs_l else float("nan")
            title_str = f"{len(modes)} mode(s)  cost={best_cost:.3f}"

            if res.get("breakdown"):
                _draw_cost_inset(ax, res["breakdown"], accent_color)

        lbl = _style_label(tag, res.get("vec", np.zeros(N_STYLE_AXES)))
        _set_scene_ax(ax, f"{lbl}\n{title_str}", start_state, goal)

        ax.text(0.02, 0.97, f"● {tag}", transform=ax.transAxes,
                fontsize=7, color=accent_color, va="top", fontweight="bold",
                path_effects=[pe.Stroke(linewidth=2, foreground="black"),
                              pe.Normal()])

        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.tick_params(colors="#888")
        ax.title.set_color("white")

    fig.suptitle(f"Style Comparison (saved modes)  |  {scene_name}",
                 fontsize=12, color="white", fontweight="bold", y=0.975)

    out = os.path.join(vis_dir, f"{scene_name}__style_comparison.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [vis] saved {out}")
    return out


def _draw_cost_inset(ax, breakdown, accent_color):
    """Tiny horizontal bar chart inset in bottom-right of ax."""
    terms  = [t for t in SOCIAL_TERMS if t in breakdown]
    values = [breakdown[t] for t in terms]
    prog   = breakdown.get("progress", 0.0)
    all_t  = ["progress"] + terms
    all_v  = [prog]       + values
    colors = [TERM_COLORS.get(t, "#888") for t in all_t]
    labels = [TERM_LABELS.get(t, t) for t in all_t]

    inset = ax.inset_axes([0.62, 0.02, 0.36, 0.36])
    inset.set_facecolor("#1A1E26CC")
    inset.patch.set_alpha(0.75)
    ys   = np.arange(len(all_t))
    bars = inset.barh(ys, all_v, color=colors, height=0.7, edgecolor="none")
    for bar, val in zip(bars, all_v):
        inset.text(bar.get_width() + (0.02 if val >= 0 else -0.02),
                   bar.get_y() + bar.get_height()/2,
                   f"{val:.2f}", va="center",
                   ha="left" if val >= 0 else "right",
                   fontsize=4.0, color="white")
    inset.set_yticks(ys)
    inset.set_yticklabels(labels, fontsize=4.0, color="white")
    inset.set_xticks([])
    inset.axvline(0, color="#555", lw=0.6)
    inset.tick_params(left=False)
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.set_title("cost terms (own style weights)", fontsize=4.0, color="#AAAAAA", pad=1.5)


# ══════════════════════════════════════════════════════════════════════════
# Output 4 - Cross-style cost breakdown comparison
# ══════════════════════════════════════════════════════════════════════════

def plot_cost_breakdown_comparison(scene_name, vis_dir, style_results):
    os.makedirs(vis_dir, exist_ok=True)
    valid = {tag: r for tag, r in style_results.items()
             if r and r.get("breakdown") and r.get("costs")}
    if not valid:
        return None

    tags = list(valid.keys())
    n    = len(tags)
    social_terms_used = [t for t in SOCIAL_TERMS
                         if any(t in r["breakdown"] for r in valid.values())]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n*1.5), 6), dpi=FIG_DPI)
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        ax.set_facecolor(DARK_BG)

    x     = np.arange(n)
    width = 0.6

    # Left: stacked social costs (each style at its OWN weights — V7's terms
    # are already weight-1 and comparable across styles, see module docstring)
    ax = axes[0]
    bottoms = np.zeros(n)
    for term in social_terms_used:
        vals = np.array([valid[tag]["breakdown"].get(term, 0.0) for tag in tags])
        ax.bar(x, vals, width, bottom=bottoms,
               label=TERM_LABELS.get(term, term),
               color=TERM_COLORS.get(term, "#888"),
               edgecolor=DARK_BG, linewidth=0.4)
        bottoms += vals
    for i, tag in enumerate(tags):
        ax.text(i, bottoms[i]+0.02, f"{bottoms[i]:.2f}",
                ha="center", va="bottom", fontsize=6.5, color="white",
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tags],
                       fontsize=7, color="white")
    ax.set_ylabel(f"Weighted cost contribution (W_SOCIAL={W_SOCIAL:g}, "
                  f"LAM_SM={LAM_SM:g})", color="white", fontsize=8)
    ax.set_title("Cost breakdown by style (sums to total)", color="white", fontsize=10)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.4,
              facecolor="#1A1E26", edgecolor="#444", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_edgecolor("#555")
    ax.axhline(0, color="#555", lw=0.8)

    # Right: progress vs total
    ax = axes[1]
    prog_vals  = [valid[tag]["breakdown"].get("progress", 0.0) for tag in tags]
    total_vals = [valid[tag]["costs"][0] if valid[tag]["costs"] else 0.0
                  for tag in tags]
    bar_colors = [STYLE_COLORS[i % len(STYLE_COLORS)] for i in range(n)]
    ax.bar(x-width/4, prog_vals,  width/2, label="Progress (J_goal)",
           color="#4FC3F7", edgecolor=DARK_BG, alpha=0.85)
    ax.bar(x+width/4, total_vals, width/2, label="Total cost",
           color=bar_colors, edgecolor=DARK_BG, alpha=0.85)
    ax.axhline(0, color="#888", lw=0.8, linestyle="--")
    ax.axhline(1, color="#FFD700", lw=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tags],
                       fontsize=7, color="white")
    ax.set_ylabel("Cost", color="white", fontsize=9)
    ax.set_title("Progress vs total cost by style", color="white", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.4,
              facecolor="#1A1E26", edgecolor="#444", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_edgecolor("#555")

    fig.suptitle(f"Cost breakdown comparison  |  {scene_name}  |  "
                 f"J_goal=0 max-speed run, =1 stalling (dotted line)",
                 fontsize=11, color="white", fontweight="bold")
    out = os.path.join(vis_dir, f"{scene_name}__cost_breakdown.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [vis] saved {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# Output 5 - Radar plot: 7 cost terms per style
# ══════════════════════════════════════════════════════════════════════════

def plot_style_radar(scene_name, vis_dir, style_results, breakdowns):
    """
    One radar chart, one polygon per style that has a valid breakdown.
    Axes = the 7 V7 cost terms (SOCIAL_TERMS), each style evaluated at its
    OWN vector — directly comparable because V7 keeps every social term at
    weight 1 (P4). Unlike c1_keep_side in the old cost, g_side is
    |s_pass| * hinge and therefore already non-negative, so there is no
    sign to annotate on that vertex.
    """
    os.makedirs(vis_dir, exist_ok=True)

    tags = [t for t in style_results.keys() if breakdowns.get(t) is not None]
    if not tags:
        return None

    terms  = SOCIAL_TERMS
    labels = [TERM_LABELS.get(t, t) for t in terms]
    n      = len(terms)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    all_vals = []
    for tag in tags:
        bd = breakdowns[tag]
        all_vals.extend(abs(bd.get(t, 0.0)) for t in terms)
    vmax = max(all_vals) if all_vals else 1.0
    vmax = vmax * 1.1 if vmax > 0 else 1.0

    fig = plt.figure(figsize=(8, 8), dpi=FIG_DPI)
    fig.patch.set_facecolor(DARK_BG)
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(PANEL_BG)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, color="white")
    ax.set_ylim(0, vmax)
    ax.tick_params(axis="y", colors="#888", labelsize=7)
    ax.grid(color="#444", alpha=0.4)
    ax.spines["polar"].set_color("#444")

    for i, tag in enumerate(tags):
        bd = breakdowns[tag]
        vals = [bd.get(t, 0.0) for t in terms]
        vals_plot = vals + vals[:1]

        color = STYLE_COLORS[i % len(STYLE_COLORS)]
        ax.plot(angles, vals_plot, color=color, lw=2.0, alpha=0.85,
                label=tag,
                path_effects=[pe.Stroke(linewidth=3.0, foreground="black",
                                         alpha=0.4), pe.Normal()])
        ax.fill(angles, vals_plot, color=color, alpha=0.08)

    fig.suptitle(f"Cost terms by style (weighted contributions)  |  {scene_name}",
                  fontsize=12, color="white", fontweight="bold", y=0.97)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8,
              framealpha=0.4, facecolor="#1A1E26", edgecolor="#444",
              labelcolor="white")

    out = os.path.join(vis_dir, f"{scene_name}__style_radar.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [vis] saved {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# Saved .npz loader — finds all mode files for a given scene + style tag.
# Field names (traj_full, traj_xy_with_start, reward) are unchanged from
# V6.5, so this is ported verbatim.
# ══════════════════════════════════════════════════════════════════════════

def _load_saved_modes(out_dir, scene_base, style_tag):
    """
    Return list of (traj_full array, cost float) for every saved mode file
    matching scene_base + style_tag, sorted by mode index.
    """
    pattern_single = os.path.join(out_dir, f"{scene_base}__{style_tag}.npz")
    pattern_multi  = os.path.join(out_dir, f"{scene_base}__{style_tag}_mode*.npz")

    files = sorted(glob.glob(pattern_multi))
    if not files:
        if os.path.exists(pattern_single):
            files = [pattern_single]

    modes = []
    for fpath in sorted(files):
        try:
            d = np.load(fpath, allow_pickle=True)
            if "traj_full" in d:
                traj = d["traj_full"].astype(np.float32)      # (H, 5)
                if "traj_xy_with_start" in d:
                    xy_ws = d["traj_xy_with_start"]            # (H+1, 2)
                    full_hp1 = np.zeros((traj.shape[0]+1, traj.shape[1]), dtype=np.float32)
                    full_hp1[1:, :] = traj
                    full_hp1[:, 0:2] = xy_ws
                    traj = full_hp1
                else:
                    start_row = np.zeros((1, traj.shape[1]), dtype=np.float32)
                    traj = np.concatenate([start_row, traj], axis=0)
            elif "traj_xy_with_start" in d:
                xy_ws = d["traj_xy_with_start"]
                traj  = np.zeros((xy_ws.shape[0], 2), dtype=np.float32)
                traj[:, 0:2] = xy_ws
            else:
                continue

            cost = float(d["cost"]) if "cost" in d else (
                -float(d["reward"]) if "reward" in d else 0.0)
            modes.append((traj, cost))
        except Exception as e:
            print(f"  [vis] warning: could not load {fpath}: {e}")

    return modes


# ══════════════════════════════════════════════════════════════════════════
# Master entry point — called from process_scene (FINAL generator)
# ══════════════════════════════════════════════════════════════════════════

def visualize_scene(
    scene_name, vis_dir,
    start_state, goal, obs, obs_traj, obs_vel,
    occ, clear, geo,
    all_traj,
    active_styles,        # list[(vec, tag)] — every style sampled for the scene
    style_results_raw,    # dict tag -> (saved_paths, cost, reason, l2, n_modes, stopping)
                           # i.e. exactly what run_style() returns, keyed by tag
    out_dir=None,          # if provided, load saved .npz modes from here
):
    """
    Produces five categories of PNG for the scene:
      1. cost_landscape.png
      2. style_comparison.png  (saved modes per style)
      3. cost_breakdown.png
      4. style_radar.png
      5. cost_fields__<tag>.png for every style in active_styles

    V7 has no complexity-tier gate (every sampled style is run — see the
    generator's docstring), so `active_styles` is simply the full styles
    list for the scene and there's no tier to report.
    """
    os.makedirs(vis_dir, exist_ok=True)

    if not _GEN_AVAILABLE:
        print("[vis] generator not importable — skipping cost field plots")
        return

    # ── Phase-1 neutral cost landscape ────────────────────────────────────
    neutral_costs, _ = total_cost(all_traj, obs_traj, obs_vel, geo, clear, NEUTRAL)
    fin_mask = np.isfinite(neutral_costs)
    best_neutral_traj = None
    if fin_mask.any():
        best_neutral_traj = all_traj[int(np.argmin(
            np.where(fin_mask, neutral_costs, np.inf)))]

    plot_cost_landscape(
        scene_name, vis_dir,
        start_state, goal, obs, obs_traj, obs_vel,
        occ, clear, geo,
        all_traj, neutral_costs,
        best_traj_neutral=best_neutral_traj,
    )

    # ── Build style_results dict: load actual saved modes ─────────────────
    style_results = {}
    for vec, tag in active_styles:
        raw = style_results_raw.get(tag)
        # run_style's tuple is (saved, cost, reason, l2, n_modes, stopping);
        # success == non-empty saved list (there is no separate "ok" flag).
        saved0 = raw[0] if raw else []
        ok = bool(saved0)

        if not ok:
            style_results[tag] = {
                "vec": vec, "modes": [], "costs": [],
                "breakdown": None,
                "reason": raw[2] if raw else "not_run",
            }
            continue

        modes_data = []
        if out_dir is not None:
            modes_data = _load_saved_modes(out_dir, scene_name, tag)

        if modes_data:
            mode_trajs  = [m[0] for m in modes_data]
            mode_costs  = [m[1] for m in modes_data]
        else:
            # Fallback: re-evaluate the Phase-1 pool with this style
            style_costs, style_feas = total_cost(all_traj, obs_traj, obs_vel,
                                                 geo, clear, vec)
            fm = style_feas & np.isfinite(style_costs)
            if not fm.any():
                style_results[tag] = {
                    "vec": vec, "modes": [], "costs": [],
                    "breakdown": None,
                    "reason": "no_feasible_in_pool",
                }
                continue
            best_idx  = int(np.argmin(np.where(fm, style_costs, np.inf)))
            best_traj = all_traj[best_idx]
            mode_trajs = [best_traj]
            mode_costs = [float(style_costs[best_idx])]

        bd = None
        if mode_trajs and mode_trajs[0].shape[1] >= 5:
            bd = _cost_breakdown(mode_trajs[0], obs_traj, obs_vel, geo, clear, vec)

        style_results[tag] = {
            "vec": vec,
            "modes": mode_trajs,
            "costs": mode_costs,
            "breakdown": bd,
            "reason": "ok",
        }

    # ── Style comparison (all modes per style) ─────────────────────────────
    plot_style_comparison(
        scene_name, vis_dir,
        start_state, goal, obs, obs_traj, obs_vel,
        occ, clear, geo,
        style_results,
        all_traj=all_traj, all_costs=neutral_costs,
    )

    # ── Cost breakdown bar chart ────────────────────────────────────────────
    plot_cost_breakdown_comparison(scene_name, vis_dir, style_results)

    # ── Radar plot: 7 cost terms per style ──────────────────────────────────
    breakdowns = {tag: r.get("breakdown") for tag, r in style_results.items()}
    plot_style_radar(scene_name, vis_dir, style_results, breakdowns)

    # ── Per-term cost fields for every style, each at ITS OWN weights ──────
    for vec, tag in active_styles:
        saved_mode_trajs = style_results.get(tag, {}).get("modes", [])
        plot_traj_cost_fields(
            scene_name, vis_dir,
            start_state, goal, obs, obs_traj, obs_vel,
            occ, clear, geo,
            all_traj=all_traj,
            style=vec if tag != "neutral" else None,
            style_tag=tag,
            saved_mode_trajs=saved_mode_trajs,
        )


# ══════════════════════════════════════════════════════════════════════════
# Standalone mode — mirrors process_scene()'s pool construction in the V7
# generator (load_scene -> build_geodesic -> sample pool -> rollout), minus
# the presort/MPPI refinement, since visualization only needs the Phase-1
# pool and (optionally) the already-saved .npz outputs.
# ══════════════════════════════════════════════════════════════════════════

def _run_standalone(args):
    if not _GEN_AVAILABLE:
        raise RuntimeError(
            "Generator module (generate_expert_trajectories.py) "
            "must be importable.")

    files = sorted(glob.glob(os.path.join(args.in_dir, "sample_*.npz")))
    if args.limit:
        files = files[:args.limit]

    print(f"Standalone visualizer: {len(files)} scenes -> {args.vis_dir}")
    out_dir = getattr(args, "out_dir", None)

    for in_path in files:
        scene_name = os.path.splitext(os.path.basename(in_path))[0]
        print(f"\nScene: {scene_name}")

        try:
            x0, goal, obs, occ, has_map, orig, extra = load_scene(in_path)
        except Exception as e:
            print(f"  [skip] load error: {e}")
            continue

        subgoal, geo, clear = build_geodesic(occ, goal, has_map)
        if geo is None:
            print("  [skip] goal_unreachable")
            continue

        rng = np.random.default_rng(seed_of(os.path.basename(in_path), "styles"))
        styles = sample_styles(rng, args.n_styles_per_scene, args.p_axis,
                               args.p_corner, args.holdout_pass_group)

        rp = np.random.default_rng(seed_of(os.path.basename(in_path), "phase1"))
        n_uniform = getattr(args, "n_uniform", N_UNIFORM)
        n_smooth  = getattr(args, "n_smooth", N_SMOOTH)
        nw = max(512, n_uniform // 2)
        ws = sample_smooth(nw, HORIZON, rp)
        ws[:, :4] *= 0.2
        pool_u = np.concatenate([sample_uniform(n_uniform, HORIZON, rp),
                                 sample_smooth(n_smooth, HORIZON, rp), ws,
                                 brake_controls()[None]], 0)
        all_traj = rollout(x0, pool_u)

        obs_traj = obs_positions(obs)
        obs_vel = (obs[:, 2:4].astype(np.float32) if obs.shape[0]
                  else np.zeros((0, 2), np.float32))

        # Mark all as "ok, no saved paths known yet" — visualize_scene will
        # try to load the real saved .npz modes from out_dir if given, and
        # fall back to a Phase-1-pool best-trajectory re-evaluation if not.
        style_results_raw = {tag: ([f"{scene_name}__{tag}"], 0.0, "ok", 0.0, 1, False)
                             for _, tag in styles}

        visualize_scene(
            scene_name, args.vis_dir,
            x0, goal, obs, obs_traj, obs_vel,
            occ, clear, geo,
            all_traj,
            styles, style_results_raw,
            out_dir=out_dir,
        )


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualize V7 norm-grounded expert trajectory cost landscapes")
    parser.add_argument("--in_dir",  default="data/scenes/eval_500",
                        help="Directory of raw scene .npz files")
    parser.add_argument("--out_dir", default=None,
                        help="Directory of expert .npz outputs (to load saved modes). "
                             "If omitted, falls back to pool re-evaluation.")
    parser.add_argument("--vis_dir", default="vis_output",
                        help="Where to write PNG files")
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--n_uniform", type=int, default=N_UNIFORM)
    parser.add_argument("--n_smooth",  type=int, default=N_SMOOTH)
    parser.add_argument("--n_styles_per_scene", type=int, default=6)
    parser.add_argument("--p_axis", type=float, default=0.40)
    parser.add_argument("--p_corner", type=float, default=0.10)
    parser.add_argument("--holdout_pass_group", action="store_true")
    args = parser.parse_args()
    _run_standalone(args)

if __name__ == "__main__":
    main()
