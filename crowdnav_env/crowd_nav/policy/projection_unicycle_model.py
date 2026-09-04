#
# projection_unicycle_model.py
#
# Unicycle kinematic model for feasibility projection of diffusion-generated
# trajectories with DYNAMIC obstacle avoidance.
#
# Differences vs the static-obstacle course-project version:
#   - Obstacles are passed as runtime PARAMETERS (per-node), not baked into
#     the model at build time. This lets us update obstacle positions
#     between solver calls and propagate constant-velocity humans across the
#     prediction horizon.
#   - Each obstacle slot stores (ox, oy, r). The propagated position
#     ox + vx*k*dt is computed in Python and set per-node before solving.
#   - n_obs_max is fixed at build time. Unused slots are deactivated by
#     setting r = 0 (the constraint becomes (p-o)^2 >= 0, trivially true).
#   - v_min = 0 (no reversing) — prevents the solver from "detouring" by
#     backing up around obstacles, which is unphysical for our use case.
#
# State:    x = [px, py, theta, v, omega]^T
# Controls: u = [a, alpha]^T   (linear and angular acceleration)
#

import numpy as np
import types
from casadi import MX, vertcat, cos, sin


def projection_unicycle_model(n_obs_max, dt_horizon):
    """
    Build the CasADi unicycle model with PARAMETERIZED obstacle constraints.

    Parameters
    ----------
    n_obs_max : int
        Maximum number of obstacles (static + dynamic, padded with r=0
        for empty slots). Must be fixed at build time.
    dt_horizon : float
        Sample time of the prediction horizon [s]. Used only for documenting
        the time-step convention; obstacle propagation is done in the caller.

    Returns
    -------
    model      : SimpleNamespace  acados-compatible model struct
    constraint : SimpleNamespace  obstacle constraint expression and bounds
    """
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "diffusion_projection_unicycle"

    # ------------------------------------------------------------------ #
    #  States
    # ------------------------------------------------------------------ #
    px    = MX.sym("px")
    py    = MX.sym("py")
    theta = MX.sym("theta")
    v     = MX.sym("v")
    omega = MX.sym("omega")
    x = vertcat(px, py, theta, v, omega)

    # ------------------------------------------------------------------ #
    #  Controls (acceleration commands)
    # ------------------------------------------------------------------ #
    a     = MX.sym("a")
    alpha = MX.sym("alpha")
    u = vertcat(a, alpha)

    # ------------------------------------------------------------------ #
    #  State derivatives (for implicit form)
    # ------------------------------------------------------------------ #
    pxdot    = MX.sym("pxdot")
    pydot    = MX.sym("pydot")
    thetadot = MX.sym("thetadot")
    vdot     = MX.sym("vdot")
    omegadot = MX.sym("omegadot")
    xdot = vertcat(pxdot, pydot, thetadot, vdot, omegadot)

    # ------------------------------------------------------------------ #
    #  Parameters: per-node obstacle positions and radii.
    #  Layout per slot i:  [ox_i, oy_i, r_i]  (3 numbers per obstacle)
    #  Time-shifted positions are computed in Python before solving.
    # ------------------------------------------------------------------ #
    p_list = []
    h_obs_list = []
    for i in range(n_obs_max):
        oxi = MX.sym(f"ox_{i}")
        oyi = MX.sym(f"oy_{i}")
        ri  = MX.sym(f"r_{i}")
        p_list += [oxi, oyi, ri]

        # Constraint expression:
        #     h_i = (px - ox)^2 + (py - oy)^2 - r^2
        # Imposed as h_i >= 0  (i.e.  lh = 0,  uh = +inf)
        # With r = 0 the slot is effectively disabled (always feasible).
        h_i = (px - oxi)**2 + (py - oyi)**2 - ri**2
        h_obs_list.append(h_i)

    p = vertcat(*p_list) if n_obs_max > 0 else MX.sym("p_dummy", 0)

    # ------------------------------------------------------------------ #
    #  Kinematics
    # ------------------------------------------------------------------ #
    f_expl = vertcat(
        v * cos(theta),     # px_dot
        v * sin(theta),     # py_dot
        omega,              # theta_dot
        a,                  # v_dot
        alpha,              # omega_dot
    )

    # ------------------------------------------------------------------ #
    #  Physical bounds (tuned for crowd-nav scale, low-speed ground robot)
    # ------------------------------------------------------------------ #
    # Speed: forward only (v_min = 0). This prevents pathological reversing
    # solutions and matches typical crowd-nav simulator conventions.
    model.v_min   =  0.0
    model.v_max   =  1.0    # m/s — typical for indoor / crowd settings

    # Angular rate
    model.omega_min = -np.pi
    model.omega_max =  np.pi

    # Acceleration limits (control bounds)
    model.a_min     = -1.5
    model.a_max     =  1.5
    model.alpha_min = -np.pi
    model.alpha_max =  np.pi

    # ------------------------------------------------------------------ #
    #  Obstacle constraint vector: h_i = dist_sq - r^2 >= 0
    # ------------------------------------------------------------------ #
    if n_obs_max > 0:
        h_expr = vertcat(*h_obs_list)
        constraint.expr   = h_expr
        constraint.lh_obs = np.zeros(n_obs_max)              # h >= 0
        constraint.uh_obs = 1e6 * np.ones(n_obs_max)         # upper unbounded
    else:
        constraint.expr   = vertcat()
        constraint.lh_obs = np.array([])
        constraint.uh_obs = np.array([])

    constraint.n_obs_max     = n_obs_max
    constraint.params_per_obs = 3   # (ox, oy, r) — vx, vy handled outside

    # ------------------------------------------------------------------ #
    #  Initial condition (overridden at runtime)
    # ------------------------------------------------------------------ #
    model.x0 = np.zeros(5)

    # ------------------------------------------------------------------ #
    #  Pack model struct
    # ------------------------------------------------------------------ #
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl
    model.x    = x
    model.xdot = xdot
    model.u    = u
    model.z    = vertcat()
    model.p    = p
    model.name = model_name

    return model, constraint
