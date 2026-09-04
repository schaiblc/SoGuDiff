import numpy as np


def project_velocity_to_unicycle(vx_des, vy_des, theta_now, prev_v, prev_omega, dt,
                                  max_vel, max_wrot, max_accel, max_w_accel,
                                  enforce_lims, k_omega=2.0):
    """
    Project a desired holonomic (vx, vy) command onto a unicycle (v, omega)
    action: forward speed v in [0, max_vel] and angular rate omega in
    [-max_wrot, max_wrot], respecting per-step acceleration bounds derived
    from (prev_v, prev_omega) when enforce_lims=True.

    Shared by ORCAUnicycle and SFMUnicycle so both classical baselines apply
    the exact same heading-tracking + limit-clamping logic as the unicycle
    RL policies' own _clamp_and_update() — any future correction to this
    projection only has to be made once.

    Heading error is wrapped to (-pi, pi]; forward speed is scaled by
    max(0, cos(heading_err)) so a large heading error causes the agent to
    turn in place rather than drive forward in the wrong direction (same
    trick used in unicycle pure-pursuit controllers).

    Returns (v_cmd, omega_cmd) with omega_cmd in rad/s — NOT yet multiplied
    by dt. Callers building ActionRot must do ActionRot(v_cmd, omega_cmd * dt).
    """
    v_des = float(np.hypot(vx_des, vy_des))

    # If the desired motion is near-zero, hold heading — avoids atan2 noise.
    if v_des < 1e-3:
        theta_des = theta_now
    else:
        theta_des = float(np.arctan2(vy_des, vx_des))

    heading_err = (theta_des - theta_now + np.pi) % (2 * np.pi) - np.pi
    omega_des = k_omega * heading_err

    speed_scale = max(0.0, np.cos(heading_err))
    v_cmd = v_des * speed_scale

    # Hard velocity bounds first.
    v_cmd = float(np.clip(v_cmd, 0.0, max_vel))
    omega_cmd = float(np.clip(omega_des, -max_wrot, max_wrot))

    # Acceleration bounds (only if enforce_lims=True).
    if enforce_lims:
        v_cmd = float(np.clip(v_cmd, prev_v - max_accel * dt, prev_v + max_accel * dt))
        v_cmd = float(np.clip(v_cmd, 0.0, max_vel))  # re-clip after accel-clip

        omega_cmd = float(np.clip(omega_cmd, prev_omega - max_w_accel * dt, prev_omega + max_w_accel * dt))
        omega_cmd = float(np.clip(omega_cmd, -max_wrot, max_wrot))

    return v_cmd, omega_cmd
