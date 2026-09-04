#
# projection_solver.py
#
# Builds and configures the acados OCP solver used as a feasibility +
# safety projection layer for diffusion-generated trajectories.
#
# Public API:
#
#   build_projection_solver(N, Tf, n_obs_max)
#       -> (solver, model, constraint)
#       Builds the solver once. Reuse across many solve calls.
#
#   pack_obstacle_params(obstacles_xy, obstacles_v, radii, n_obs_max,
#                        N, dt, n_extrapolate=True)
#       -> np.ndarray of shape (N+1, 3*n_obs_max)
#       Packs per-node obstacle parameters for set("p", ...).
#       Propagates each obstacle by its constant velocity to each horizon
#       node. Pads unused slots with r = 0.
#
#   solve_projection(solver, x0, ref_xy, params_per_node,
#                    warm_start=None) -> dict
#       Sets references, parameters, initial state, optional warm start;
#       runs the solver (single SQP iteration in RTI mode); returns a dict
#       with the projected trajectory, controls, status, slacks, and timing.
#
# Design notes:
#   - Position weight w_pos = 100, terminal w_pos_e = 1: heavy stage tracking
#     of the diffusion path; small terminal weight so the endpoint is not
#     pinned (the projection should follow the whole path, not the goal).
#   - L1 slacks on obstacle constraints with large linear penalty (1e4) keep
#     the QP feasible even when the warm start is in collision. Slack values
#     are reported back so the caller can detect "soft violations" and fall
#     back if needed.
#   - SQP_RTI: one SQP iteration per call. Fast and well-suited to the
#     receding-horizon / per-tick replanning pattern.
#

import os
import time
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from projection_unicycle_model import projection_unicycle_model


# ---------------------------------------------------------------------- #
#  Solver build
# ---------------------------------------------------------------------- #

def build_projection_solver(
    N,
    Tf,
    n_obs_max,
    *,
    w_pos        = 1e2,    # stage position tracking weight
    w_a          = 1e-2,   # linear-acceleration regularization
    w_alpha      = 1e-2,   # angular-acceleration regularization
    w_pos_e      = 1e0,    # terminal position weight (small — no pinning)
    slack_lin    = 1e4,    # L1 slack penalty on obstacle constraints
    slack_quad   = 1e1,    # L2 slack penalty (curvature for QP solver)
    solver_mode  = "RTI",  # "RTI" (1 SQP iter, fastest) | "SQP" (iterate to converge)
    sqp_max_iter = 5,      # used only when solver_mode == "SQP"
    sqp_tol      = 1e-3,   # used only when solver_mode == "SQP"
    code_export_dir = "c_generated_proj",
    json_file       = "acados_proj_ocp.json",
):
    """
    Build the acados OCP solver for diffusion trajectory projection.

    Parameters
    ----------
    N         : int    number of shooting intervals (= horizon - 1 if your
                       diffusion outputs T points, i.e. N = T - 1)
    Tf        : float  prediction horizon length [s]  (= (T-1) * dt_diffusion)
    n_obs_max : int    maximum number of obstacle slots

    Returns
    -------
    solver, model, constraint
    """
    dt_horizon = Tf / N

    ocp = AcadosOcp()

    model, constraint = projection_unicycle_model(n_obs_max, dt_horizon)

    # ------------------------------------------------------------------ #
    #  AcadosModel
    # ------------------------------------------------------------------ #
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x    = model.x
    model_ac.xdot = model.xdot
    model_ac.u    = model.u
    model_ac.z    = model.z
    model_ac.p    = model.p
    model_ac.name = model.name

    if n_obs_max > 0:
        model_ac.con_h_expr   = constraint.expr
        model_ac.con_h_expr_e = constraint.expr   # also enforced at terminal node

    ocp.model = model_ac

    # ------------------------------------------------------------------ #
    #  Dimensions
    # ------------------------------------------------------------------ #
    nx   = model.x.rows()    # 5
    nu   = model.u.rows()    # 2
    ny   = 2 + nu            # [px, py, a, alpha]
    ny_e = 2                 # [px, py]

    np_param = 3 * n_obs_max # (ox, oy, r) per obstacle
    ocp.solver_options.N_horizon = N

    # Required: register parameter dimension via parameter_values default
    ocp.parameter_values = np.zeros(np_param)

    # ------------------------------------------------------------------ #
    #  Cost: linear LS on (px, py, a, alpha) at stages, (px, py) at terminal
    # ------------------------------------------------------------------ #
    W   = np.diag([w_pos, w_pos, w_a, w_alpha])
    W_e = np.diag([w_pos_e, w_pos_e])

    ocp.cost.cost_type   = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    Vx = np.zeros((ny, nx)); Vx[0, 0] = 1.0; Vx[1, 1] = 1.0
    Vu = np.zeros((ny, nu)); Vu[2, 0] = 1.0; Vu[3, 1] = 1.0
    Vx_e = np.zeros((ny_e, nx)); Vx_e[0, 0] = 1.0; Vx_e[1, 1] = 1.0

    ocp.cost.W    = W
    ocp.cost.W_e  = W_e
    ocp.cost.Vx   = Vx
    ocp.cost.Vu   = Vu
    ocp.cost.Vx_e = Vx_e

    # Default refs (overwritten every solve)
    ocp.cost.yref   = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # ------------------------------------------------------------------ #
    #  Slack penalties on obstacle constraints (stage + terminal)
    # ------------------------------------------------------------------ #
    if n_obs_max > 0:
        nsh = n_obs_max
        ocp.cost.zl   = slack_lin  * np.ones(nsh)
        ocp.cost.zu   = slack_lin  * np.ones(nsh)
        ocp.cost.Zl   = slack_quad * np.ones(nsh)
        ocp.cost.Zu   = slack_quad * np.ones(nsh)
        ocp.cost.zl_e = slack_lin  * np.ones(nsh)
        ocp.cost.zu_e = slack_lin  * np.ones(nsh)
        ocp.cost.Zl_e = slack_quad * np.ones(nsh)
        ocp.cost.Zu_e = slack_quad * np.ones(nsh)

    # ------------------------------------------------------------------ #
    #  State bounds (v and omega)
    # ------------------------------------------------------------------ #
    ocp.constraints.lbx   = np.array([model.v_min,  model.omega_min])
    ocp.constraints.ubx   = np.array([model.v_max,  model.omega_max])
    ocp.constraints.idxbx = np.array([3, 4])

    # ------------------------------------------------------------------ #
    #  Input bounds
    # ------------------------------------------------------------------ #
    ocp.constraints.lbu   = np.array([model.a_min,     model.alpha_min])
    ocp.constraints.ubu   = np.array([model.a_max,     model.alpha_max])
    ocp.constraints.idxbu = np.array([0, 1])

    # ------------------------------------------------------------------ #
    #  Nonlinear obstacle constraints (stage + terminal)
    # ------------------------------------------------------------------ #
    if n_obs_max > 0:
        nsh = n_obs_max
        lh = constraint.lh_obs
        uh = constraint.uh_obs

        ocp.constraints.lh    = lh
        ocp.constraints.uh    = uh
        ocp.constraints.lsh   = np.zeros(nsh)
        ocp.constraints.ush   = np.zeros(nsh)
        ocp.constraints.idxsh = np.arange(nsh)

        ocp.constraints.lh_e    = lh
        ocp.constraints.uh_e    = uh
        ocp.constraints.lsh_e   = np.zeros(nsh)
        ocp.constraints.ush_e   = np.zeros(nsh)
        ocp.constraints.idxsh_e = np.arange(nsh)

    # ------------------------------------------------------------------ #
    #  Initial condition (overwritten every solve)
    # ------------------------------------------------------------------ #
    ocp.constraints.x0 = np.zeros(nx)

    # ------------------------------------------------------------------ #
    #  Solver options
    # ------------------------------------------------------------------ #
    ocp.solver_options.tf                    = Tf
    ocp.solver_options.qp_solver             = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx        = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type       = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 2

    if solver_mode == "RTI":
        # Single SQP iteration per call. Fastest (~1 ms).
        # Suitable when the warm start is close to the new optimum,
        # which is true once we warm-start from the diffusion path.
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
    elif solver_mode == "SQP":
        # Iterate to convergence (or sqp_max_iter, whichever first).
        # Slower (~3-10 ms) but far more robust to bad warm starts.
        # Recommended for safety-critical deployment.
        ocp.solver_options.nlp_solver_type   = "SQP"
        ocp.solver_options.nlp_solver_max_iter = sqp_max_iter
        ocp.solver_options.tol               = sqp_tol
    else:
        raise ValueError(f"solver_mode must be 'RTI' or 'SQP', got {solver_mode!r}")

    # Code export dir keeps generated C separate from any other acados solver
    ocp.code_export_directory = code_export_dir

    solver = AcadosOcpSolver(ocp, json_file=json_file)

    return solver, model, constraint


# ---------------------------------------------------------------------- #
#  Parameter packing (dynamic obstacle propagation)
# ---------------------------------------------------------------------- #

def pack_obstacle_params(obstacles_xy, obstacles_v, radii, n_obs_max, N, dt):
    """
    Build the per-node parameter vector for the projection OCP.

    Each obstacle is propagated by its (assumed-constant) velocity:
        ox_k = ox_0 + vx * k * dt
        oy_k = oy_0 + vy * k * dt

    Static obstacles correspond to vx = vy = 0.

    Parameters
    ----------
    obstacles_xy : (M, 2)  current obstacle positions
    obstacles_v  : (M, 2)  current obstacle velocities (zero for static)
    radii        : (M,)    clearance radii — the minimum allowed
                           distance from ROBOT CENTER to OBSTACLE CENTER.
                           The caller is responsible for choosing this
                           value's meaning (e.g. a single global
                           safety_radius, or a per-obstacle inflated
                           radius like r_obs + r_robot + margin). This
                           function just packs them into the parameter
                           vector — it does not interpret them.
    n_obs_max    : int     number of slots in the model (must match build)
    N            : int     number of shooting intervals
    dt           : float   horizon time step [s]

    Returns
    -------
    params : (N+1, 3*n_obs_max) array
    """
    M = obstacles_xy.shape[0] if obstacles_xy.size > 0 else 0
    M_used = min(M, n_obs_max)

    params = np.zeros((N + 1, 3 * n_obs_max), dtype=np.float64)

    if M_used == 0:
        return params

    obs_xy = obstacles_xy[:M_used]
    obs_v  = obstacles_v[:M_used]
    obs_r  = radii[:M_used]

    for k in range(N + 1):
        t = k * dt
        ox_k = obs_xy[:, 0] + obs_v[:, 0] * t
        oy_k = obs_xy[:, 1] + obs_v[:, 1] * t

        # Layout: [ox_0, oy_0, r_0,  ox_1, oy_1, r_1,  ...]
        slot = np.zeros(3 * n_obs_max)
        for i in range(M_used):
            slot[3 * i + 0] = ox_k[i]
            slot[3 * i + 1] = oy_k[i]
            slot[3 * i + 2] = obs_r[i]
        params[k] = slot

    return params


# ---------------------------------------------------------------------- #
#  Single solve call
# ---------------------------------------------------------------------- #

def solve_projection(
    solver,
    x0,
    ref_xy,
    params_per_node,
    *,
    warm_start_x=None,
    warm_start_u=None,
    N=None,
):
    """
    Run one projection solve.

    Parameters
    ----------
    x0              : (5,) initial state  [px, py, theta, v, omega]
    ref_xy          : (N+1, 2) reference (px, py) sequence (the diffusion
                      trajectory to be projected)
    params_per_node : (N+1, n_p) obstacle parameters from pack_obstacle_params
    warm_start_x    : optional (N+1, 5) state trajectory warm start
    warm_start_u    : optional (N, 2)   input trajectory warm start
    N               : optional int      horizon length (inferred if None)

    Returns
    -------
    result : dict with keys
        'status'      : acados status code (0 = success, 2 = max iter,
                         other = failure)
        'x_traj'      : (N+1, 5) solved state trajectory
        'u_traj'      : (N, 2)  solved input trajectory
        'n_iter'      : int — SQP iterations actually taken (1 in RTI mode),
                         or None if this acados build exposes neither the
                         'sqp_iter' nor the 'nlp_iter' stat
        'max_slack'   : float — largest slack across all obstacle nodes
                        (proxy for soft constraint violation)
        'solve_ms'    : float — wall time for the solve [ms]
    """
    if N is None:
        N = ref_xy.shape[0] - 1
    nx = 5
    nu = 2

    # -- Set per-node obstacle parameters ---------------------------------
    for k in range(N + 1):
        solver.set(k, "p", params_per_node[k])

    # -- Set per-node references ------------------------------------------
    # Stage k: yref = [px_ref, py_ref, 0, 0]  (zero accelerations)
    for k in range(N):
        yref = np.array([ref_xy[k, 0], ref_xy[k, 1], 0.0, 0.0])
        solver.set(k, "yref", yref)
    solver.set(N, "yref", np.array([ref_xy[N, 0], ref_xy[N, 1]]))

    # -- Initial condition -------------------------------------------------
    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    # -- Warm start (optional) --------------------------------------------
    if warm_start_x is not None:
        for k in range(N + 1):
            solver.set(k, "x", warm_start_x[k])
    if warm_start_u is not None:
        for k in range(N):
            solver.set(k, "u", warm_start_u[k])

    # -- Solve -------------------------------------------------------------
    t0 = time.time()
    status = solver.solve()
    solve_ms = (time.time() - t0) * 1e3

    # Iterations actually taken. In RTI this is 1 by construction; in SQP mode
    # it is what separates a converged solve from one that ran out of budget
    # (status 2), and it is the only way to read cost-per-iteration off the
    # latency logs instead of inferring it. The stats field is named
    # "sqp_iter" on older acados and "nlp_iter" on newer ones — try both and
    # degrade to None rather than failing a solve over instrumentation.
    n_iter = None
    for _field in ("sqp_iter", "nlp_iter"):
        try:
            n_iter = int(solver.get_stats(_field))
            break
        except Exception:
            continue

    # -- Extract solution --------------------------------------------------
    x_traj = np.zeros((N + 1, nx))
    u_traj = np.zeros((N, nu))
    for k in range(N + 1):
        x_traj[k] = solver.get(k, "x")
    for k in range(N):
        u_traj[k] = solver.get(k, "u")

    # -- Slack inspection (soft obstacle violation) -----------------------
    max_slack = 0.0
    try:
        # Stage slacks: 'sl' / 'su' available per node when soft constraints are set
        for k in range(N + 1):
            try:
                sl = solver.get(k, "sl")
                su = solver.get(k, "su")
                local_max = float(max(np.max(sl) if sl.size else 0.0,
                                      np.max(su) if su.size else 0.0))
                if local_max > max_slack:
                    max_slack = local_max
            except Exception:
                # Some acados builds expose slacks only at certain nodes — skip silently
                pass
    except Exception:
        pass

    return dict(
        status     = status,
        x_traj     = x_traj,
        u_traj     = u_traj,
        max_slack  = max_slack,
        solve_ms   = solve_ms,
        n_iter     = n_iter,
    )


# ---------------------------------------------------------------------- #
#  Helpers: build a warm start from a position trajectory
# ---------------------------------------------------------------------- #

def kinematic_warm_start_from_xy(ref_xy, dt, theta0=0.0, v0=0.0, omega0=0.0):
    """
    Build a feasible-ish state trajectory from a sequence of (px, py)
    waypoints by finite-differencing position and heading.

    Convention: ref_xy[0] is assumed to be the CURRENT robot position
    (i.e. (0,0) in ego frame). Indices 1..T-1 are the future targets.
    This matches how we feed the OCP: x0 is the initial-condition node,
    and stages 1..N are tracked references.

    Heading at node 0 is set to the supplied theta0 (the robot's actual
    current heading in the chosen frame — typically 0 in ego frame), since
    the segment 0->1 may be very short or zero and finite-differencing it
    yields a meaningless heading.

    Returns
    -------
    x_ws : (T, 5)   state warm start  [px, py, theta, v, omega]
    u_ws : (T-1, 2) input warm start  [a, alpha]
    """
    T = ref_xy.shape[0]
    x_ws = np.zeros((T, 5))
    x_ws[:, 0:2] = ref_xy

    # Segment-wise displacements
    dx = np.diff(ref_xy[:, 0])
    dy = np.diff(ref_xy[:, 1])
    seg_len = np.sqrt(dx**2 + dy**2)

    # Headings from finite differences. Use a length threshold to skip
    # tiny or zero segments (would give numerically meaningless angles).
    # If a segment is too short, inherit heading from previous node.
    eps_len = 1e-4
    headings = np.zeros(T)
    headings[0] = theta0
    for k in range(1, T):
        if seg_len[k - 1] > eps_len:
            headings[k] = np.arctan2(dy[k - 1], dx[k - 1])
        else:
            headings[k] = headings[k - 1]
    x_ws[:, 2] = headings

    # Speeds: node 0 = v0 (current robot speed), node k>=1 = ||seg_{k-1}|| / dt
    x_ws[0, 3]  = v0
    if T >= 2:
        x_ws[1:, 3] = seg_len / dt

    # Omega: finite difference of unwrapped heading; node 0 = omega0
    x_ws[0, 4] = omega0
    if T >= 2:
        dtheta = np.diff(np.unwrap(x_ws[:, 2]))
        x_ws[1:, 4] = dtheta / dt

    # Inputs: finite difference of v and omega
    u_ws = np.zeros((T - 1, 2))
    if T >= 2:
        u_ws[:, 0] = np.diff(x_ws[:, 3]) / dt
        u_ws[:, 1] = np.diff(x_ws[:, 4]) / dt

    return x_ws, u_ws
