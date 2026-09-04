"""
MPPIExpert — online sampling planner as a drop-in for the diffusion policy
================================================================================

EXPERIMENT #1 (the load-bearing one): does distilling the expert into a diffusion
planner buy anything over just RUNNING the expert online? This policy answers it
by running the SAME norm-grounded MPPI planner that generated the training data,
online, inside the SAME evaluation harness the diffusion policy uses — identical
scenes, identical style vectors, identical feasibility projection, identical
scoring, identical action extraction. The ONLY thing that differs is how the
candidate trajectory is produced: online MPPI sampling here vs. amortized
denoising in the diffusion policy.

Because it subclasses SoGuDiff and overrides only the
candidate-generation half of predict(), every downstream stage — projection,
selection, ActionRot extraction, predicted_traj bookkeeping — is byte-for-byte
the code path the diffusion policy takes. So a metric difference is attributable
to (sampling vs denoising), not to harness drift.

WHAT TO READ OFF THE RESULTS
----------------------------
  * self.timing accumulates wall-clock per predict() call (candidate generation
    only, excluding projection which both share). The frontier figure is
    social-style-fidelity (PSI/PSB/GSR/TTC separation vs style) on the y-axis
    against this latency on the x-axis, with the diffusion policy as one point.
  * The MPPI budget is swept via config (mppi_samples, mppi_iters, mppi_restarts).
    Shrinking it toward real-time is the experiment: the hypothesis is that
    style fidelity collapses before success rate does, because the wide-berth /
    correct-side / go-around-group modes are low-density regions of control space
    that a small-budget sampler cannot reliably find — whereas the diffusion
    policy holds full-budget fidelity at ~20 ms. If a 500-sample online MPPI
    still tracks the styles, the distillation argument weakens and you learn it
    here, cheaply, before writing it up.

USAGE (env.config / policy factory)
-----------------------------------
  [mppi_expert]
  # reuse the diffusion block's projection + scoring keys verbatim so the two
  # policies are configured identically:
  safety_radius = 0.5
  use_projection = true
  proj_max_dev_thresh = 5.0
  proj_max_penetration_thresh = 0.1
  # these three MUST match the diffusion block exactly — a different solver
  # mode means a different feasibility layer, which breaks the "only the
  # candidate generator differs" claim. Both build lines print the mode, so a
  # mismatch is visible at the top of the log.
  proj_solver_mode = SQP
  proj_sqp_max_iter = 5      # SQP mode only
  proj_sqp_tol = 1e-3        # SQP mode only
  k_static = 20
  use_map = true
  map_size = 50
  map_extent = 10
  style_vector = 0.0, 0.0, 0.0, 0.0
  collision_coef = 10
  smooth_pen_coef = 1
  goal_reward_coef = 5
  control_effort_coef = 0.5
  # MPPI budget (THE swept knob for the frontier):
  mppi_samples = 2048
  mppi_iters = 12
  mppi_restarts = 8
  mppi_seed_pool = 4096
  horizon = 32
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from datetime import datetime

import numpy as np

try:
    from crowd_sim.envs.utils.action import ActionRot, ActionXY  # noqa: F401
except Exception:  # pragma: no cover — allows import outside the eval env
    ActionRot = ActionXY = None

# The diffusion policy we subclass. Import via the package path so we subclass
# the SAME class object policy_factory loads (a bare `import sogudiff`
# would create a second, distinct module instance and split the class identity).
from crowd_nav.policy.sogudiff import SoGuDiff
from crowd_nav.paths import resolve_path


# ---------------------------------------------------------------------------
# Import the EXACT sampler + cost the training data was generated with, so the
# online planner is provably the same instrument as the offline expert. We load
# it by file path (it is a script, not a package) to avoid a rename.
# ---------------------------------------------------------------------------
_EXPERT_MODULE_NAME = "expert_sampler"


def _load_expert_module(path):
    """Load the generator by file path (it is a script, not a package).

    The module MUST be registered in sys.modules under a STABLE name before
    exec_module. The generator's hot kernels are @njit(cache=True), and numba's
    on-disk cache pickles a reference to the defining module; if that module
    is not importable by name, loading a cached kernel dies with
    "ModuleNotFoundError: No module named '<dynamic>'". Registering it also
    keeps the cache key stable across processes, so we reuse the compiled
    kernels instead of paying JIT cost on every evaluation run — which matters
    here because JIT time would otherwise pollute the latency measurement.
    """
    if _EXPERT_MODULE_NAME in sys.modules:
        return sys.modules[_EXPERT_MODULE_NAME]

    # Isolate numba's on-disk cache from the generator's.
    #
    # The generator runs as a script, so ITS cached kernels are pickled with a
    # reference to module "__main__". We load the same source file under a
    # different module name, and numba keys the cache by (source file,
    # function, signature) — NOT by module name — so our entries would
    # OVERWRITE the generator's in sogudiff/__pycache__. A generator
    # process then loading our entry does `import expert_sampler`, which does
    # not exist in that process, and dies with ModuleNotFoundError.
    #
    # That is a live hazard whenever an evaluation runs while a data-generation
    # chain is active. A private cache dir keeps the two completely separate;
    # each pays its own one-time JIT and neither can corrupt the other.
    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "numba_mppi_baseline"),
    )
    try:  # picked up at numba import; force a re-read if it is already loaded
        import numba.core.config as _nbcfg
        _nbcfg.reload_config()
    except Exception:
        pass

    spec = importlib.util.spec_from_file_location(_EXPERT_MODULE_NAME, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_EXPERT_MODULE_NAME] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(_EXPERT_MODULE_NAME, None)
        raise
    return mod


class MPPIExpert(SoGuDiff):
    """Online MPPI expert, drop-in for the diffusion policy.

    Overrides configure() (to load the sampler, not a checkpoint) and predict()
    (to generate the candidate by online MPPI). Everything else — projection,
    scoring, action extraction, predicted_traj — is inherited unchanged.
    """

    def __init__(self):
        super().__init__()
        self.name = "MPPIExpert"
        self._E = None                 # expert sampler module
        self.mppi_samples = 2048
        self.mppi_iters = 12
        self.mppi_restarts = 8
        # Full-parity default: the generator's Phase-1 pool is
        # N_UNIFORM + N_SMOOTH + max(512, N_UNIFORM//2) = 4096+12288+2048 = 18432
        # control sequences (+1 brake). Set below full to shrink the search.
        self.mppi_seed_pool = 18432
        # Base RNG seed. Offline the expert seeds per (scene, style) so
        # generation replays exactly; online there is no scene identity to
        # hash, so we seed from base + per-call counter. Set < 0 for genuinely
        # nondeterministic behavior.
        self.mppi_seed = 0
        self._call_idx = 0
        # self.timing is inherited from the parent and holds the SAME
        # quantity the diffusion policy records: full per-call
        # candidate-generation latency (conditioning + geodesic + sampling +
        # selection), EXCLUDING the shared projection back-end. Measuring the
        # same span on both sides is what makes the frontier x-axis
        # comparable.
        #
        # timing_sampler additionally isolates just the MPPI refinement call,
        # so we can attribute cost between the FMM/geodesic setup and the
        # sampling itself when the budget is swept. It is appended ONCE PER
        # predict() call, index-aligned with self.timing (nan on ticks where
        # the search never reached the sampler), so the same warmup `skip`
        # applied to both selects the same calls.
        self.timing_sampler = []

        # Online-search failures, i.e. ticks where MPPI produced no candidate
        # and the robot fell back to the emergency brake. Split by cause:
        #   geo  — FMM could not reach the goal from the current crop
        #   cand — the search ran but nothing feasible survived
        # These are the budget sweep's real degradation mechanism, so they are
        # reported per episode alongside latency rather than left to surface
        # indirectly as a lower success rate.
        self.n_search_fail_geo = 0
        self.n_search_fail_cand = 0

        # Per-call flag, index-aligned with self.timing: True iff the search
        # actually produced a candidate on that tick.
        #
        # WHY THIS EXISTS. Failed ticks are CHEAP — a geodesic failure bails
        # after the FMM build, and the common candidate failure bails as soon
        # as the prefilter empties. Folding them into the headline mean would
        # hand the baseline a latency DISCOUNT FOR GIVING UP, and the discount
        # grows as the budget shrinks (failures get more frequent), i.e.
        # exactly the regime the frontier makes a claim about. The diffusion
        # policy has no such path — denoising always produces its K samples at
        # full cost — so the discount would be available to exactly one of the
        # two policies being compared.
        #
        # Excluding the ticks outright is equally wrong: they are real control
        # ticks the robot paid for. So self.timing keeps EVERY tick (what a
        # control cycle costs) and this mask derives a second statistic over
        # planning ticks only (what producing a plan costs). Report the
        # success-only number as the frontier x-axis with the failure rate
        # annotated, and the all-ticks number beside it.
        self._timing_ok = []

    # -- configuration --------------------------------------------------------
    def configure(self, config):
        """Configure exactly like the diffusion policy for every SHARED stage
        (projection, scoring, map, action limits, style), then load the sampler
        instead of a network checkpoint."""
        section = "mppi_expert"

        # 1) Reuse the diffusion policy's configure for all shared machinery.
        #    We temporarily point it at our section but must avoid its
        #    checkpoint/model load. The clean way: call the shared setters
        #    directly rather than the whole diffusion configure(). We replicate
        #    only the projection/scoring/map/limits/style setup here so there is
        #    no network dependency.
        gc = lambda k, d=None, t=str: self._cfg_get(config, section, k, d, t)

        # Kinematic limits — MUST match the action_space block the baselines use.
        self.kinematics   = "unicycle"
        self.enforce_lims = True
        self.max_vel      = gc("max_vel", 1.0, float)
        self.max_wrot     = gc("max_wrot", np.pi, float)
        self.max_accel    = gc("max_accel", 1.5, float)
        self.max_w_accel  = gc("max_w_accel", np.pi, float)

        # Horizon / timing — identical to the diffusion policy.
        self.horizon   = gc("horizon", 32, int)
        self.k_max     = gc("k_max", 10, int)
        self.proj_dt   = 0.05
        self.time_step = gc("time_step", 0.25, float)

        # Style vector (the axis whose fidelity we measure).
        sv = gc("style_vector", "0.0,0.0,0.0,0.0", str)
        self.style_vector = np.array([float(x) for x in sv.replace(" ", "").split(",")],
                                     dtype=np.float32)
        self.n_style_axes = len(self.style_vector)

        # Map config — identical rasterization to the diffusion policy.
        self.use_map    = gc("use_map", "true", str).lower() == "true"
        self.map_size   = gc("map_size", 50, int)
        self.map_extent = gc("map_extent", 10, float)

        # Scoring coefficients — IDENTICAL to the diffusion policy so selection
        # among candidates is the same rule. (Used by inherited score path only
        # if we chose to score; see predict() note on single-candidate return.)
        self.collision_coef      = gc("collision_coef", 10.0, float)
        self.smooth_pen_coef     = gc("smooth_pen_coef", 1.0, float)
        self.goal_reward_coef    = gc("goal_reward_coef", 5.0, float)
        self.control_effort_coef = gc("control_effort_coef", 0.5, float)

        # Projection — IDENTICAL to the diffusion policy. This is the whole point:
        # both policies feed their chosen candidate through the same OCP.
        self.use_projection = gc("use_projection", "true", str).lower() == "true"
        self.safety_radius  = gc("safety_radius", 0.5, float) * 1.3
        self.proj_max_dev_thresh         = gc("proj_max_dev_thresh", 5.0, float)
        self.proj_max_penetration_thresh = gc("proj_max_penetration_thresh", 0.1, float)
        self.proj_solver_mode = gc("proj_solver_mode", "RTI", str)
        assert self.proj_solver_mode in ("RTI", "SQP"), (
            f"proj_solver_mode must be 'RTI' or 'SQP', got {self.proj_solver_mode!r}"
        )
        # SQP budget — must be readable here too, or the baseline would build a
        # solver with different iteration limits than the diffusion policy and
        # the shared-back-end claim would stop holding.
        self.proj_sqp_max_iter = gc("proj_sqp_max_iter", 5, int)
        self.proj_sqp_tol      = gc("proj_sqp_tol", 1e-3, float)
        self.k_static   = gc("k_static", 20, int)
        # Total acados obstacle slots = dynamic (k_max) + static map cells.
        # _init_projection() (re)derives proj_N/proj_Tf from horizon/proj_dt.
        self.proj_n_obs = self.k_max + self.k_static

        # Build the acados projection solver via the parent's factored builder,
        # so the OCP is byte-for-byte identical to the diffusion policy's.
        if self.use_projection:
            self._init_projection()

        # Warm-state used both by the projection x0 and by the online MPPI x0,
        # so the sampler and the projector agree on the current (v, omega).
        # _reset_unicycle_state() (inherited, called by the explorer at each
        # episode start) refreshes these; initialize here for safety.
        self._reset_unicycle_state()

        # MPPI budget — the swept knob.
        # Load the sampler module FIRST (same cost + MPPI the data was
        # generated with) so the budget below can be derived from the
        # generator's own constants rather than duplicated literals.
        expert_path = resolve_path(
            gc("expert_module", "sogudiff/generate_expert_trajectories.py", str))
        self._E = _load_expert_module(expert_path)
        E = self._E

        # ---- MPPI budget: ONE knob (mppi_budget) ---------------------------
        # mppi_budget is a fraction of the expert's full search. It scales the
        # two components that carry essentially all the compute AND have wide
        # dynamic range:
        #     seed_pool  (Phase-1 exploration)  ~18.4k rollouts at b=1
        #     samples    (MPPI refinement)      8*12*2048 = ~197k rollouts at b=1
        # Total work is ~linear in b with NO FLOOR, which is the point: sweeping
        # samples alone asymptotes at the fixed Phase-1 pool cost and can never
        # reach real-time latency, so it cannot trace the frontier.
        #
        # restarts (N_SEEDS) and iters (MPPI_ITER) are deliberately NOT scaled.
        # They encode the expert's DESIGN, not its budget: restarts is the
        # homotopy-mode discovery mechanism (8 seeds >= 0.4 m apart), and iters
        # is the convergence depth of each refinement. Scaling them would floor
        # out at 1 almost immediately and change what the planner IS rather than
        # how hard it searches. Override them explicitly only for the
        # breadth-vs-depth mechanism ablation.
        self.mppi_budget = gc("mppi_budget", 1.0, float)
        b = max(1e-3, float(self.mppi_budget))
        full_pool = E.N_UNIFORM + E.N_SMOOTH + max(512, E.N_UNIFORM // 2)

        self.mppi_samples   = gc("mppi_samples",
                                 max(8, int(round(E.MPPI_N * b))), int)
        self.mppi_seed_pool = gc("mppi_seed_pool",
                                 max(64, int(round(full_pool * b))), int)
        self.mppi_iters     = gc("mppi_iters",    E.MPPI_ITER, int)
        self.mppi_restarts  = gc("mppi_restarts", E.N_SEEDS,   int)
        self.mppi_seed      = gc("mppi_seed", 0, int)
        self._call_idx      = 0

        print(f"MPPI budget b={b:g} -> seed_pool={self.mppi_seed_pool} "
              f"samples={self.mppi_samples} restarts={self.mppi_restarts} "
              f"iters={self.mppi_iters}"
              + ("  [FULL EXPERT PARITY]" if abs(b - 1.0) < 1e-9 else ""))
        # Sanity: the sampler's horizon/dt must match the eval harness's.
        assert self._E.HORIZON == self.horizon, \
            f"sampler HORIZON {self._E.HORIZON} != policy horizon {self.horizon}"
        assert abs(self._E.DT - self.proj_dt) < 1e-9, \
            f"sampler DT {self._E.DT} != policy proj_dt {self.proj_dt}"

    def _cfg_get(self, config, section, key, default, cast):
        try:
            if config.has_option(section, key):
                raw = config.get(section, key)
                return cast(raw) if cast is not str else raw
        except Exception:
            pass
        return default

    # -- the one method that differs: candidate generation --------------------
    def predict(self, state):
        """Same contract as the diffusion policy's predict(): returns an
        ActionRot, populates self.predicted_traj. The ONLY difference is that the
        candidate (T,2) ego trajectory comes from online MPPI, not denoising."""
        # Opens the end-to-end (state → action) window and zeros the
        # visualizer-overhead accumulator that _finish_from_candidate
        # subtracts from it — same call, same definition, as the diffusion
        # policy, which is what keeps the two totals comparable.
        t_predict_start = self._begin_predict_timing()

        if self.reach_destination(state):
            return ActionRot(0, 0)

        # Candidate-generation clock — starts at the SAME point the diffusion
        # policy starts its own (right after reach_destination), so both
        # measure conditioning + planning up to a committed candidate.
        _t_cand0 = time.perf_counter()

        # --- Physical-unit conditioning (no norm round-trip) ------------------
        # The sampler works in PHYSICAL ego units, so we pull obstacles/goal
        # straight from the raw state. We deliberately do NOT call
        # build_condition_from_state (which needs a diffusion norm file this
        # policy never loads) — the sampler and the shared projection back-end
        # both operate in physical units.
        obs_phys = self._physical_obstacles_ego(state)      # (M,4) [px,py,vx,vy]
        goal_phys = self._physical_goal_ego(state)          # (2,)

        # Initial state for the online rollout — the SAME warm-state
        # (_v0/_w0_from_last_step) the projection uses for its x0, so the online
        # sampler and the projector are consistent (and match the offline
        # expert, which rolls out from the current (v, omega)).
        v0 = float(getattr(self, "_v0_from_last_step", 0.0) or 0.0)
        w0 = float(getattr(self, "_w0_from_last_step", 0.0) or 0.0)
        x0 = np.array([0.0, 0.0, 0.0, v0, w0], dtype=np.float32)

        # Map + geodesic — identical rasterization to the diffusion policy, then
        # the SAME build_geodesic the expert used offline.
        if self.use_map:
            occ_np, has_map_f = self._rasterize_ego_map(state)
        else:
            occ_np, has_map_f = np.zeros((self.map_size, self.map_size), np.float32), 0.0
        # obs_phys is passed so the progress field carries the SAME human
        # space-time footprint the offline expert used; omitting it would give
        # the online planner a geodesic that points straight through people
        # while the offline demonstrations routed around them -- exactly the
        # offline/online mismatch this baseline exists to rule out.
        subgoal, geo, clear = self._E.build_geodesic(occ_np, goal_phys, has_map_f,
                                                     obs_phys)
        if geo is None:
            # Goal unreachable from the current crop → controlled brake, exactly
            # as the diffusion policy does on projection fallback.
            #
            # TIMED AND COUNTED, not skipped. This is a real control tick: the
            # robot brakes and the sim advances, so its latency belongs in the
            # frontier x-axis. Dropping it would bias the sweep exactly where
            # it matters — search failures are FAST and get MORE common as the
            # budget shrinks, so excluding them would report the low-budget
            # planner's latency over only the ticks where it happened to
            # succeed. nan into timing_sampler keeps that list index-aligned
            # with self.timing (the sampler never ran on this tick).
            self.timing.append(time.perf_counter() - _t_cand0)
            self.timing_sampler.append(np.nan)
            self._timing_ok.append(False)
            self.n_search_fail_geo += 1
            self._log_predict_total(t_predict_start, tag="SEARCH-FAIL(geo) ")
            return self._emergency_brake_action()

        obs_traj = self._E.obs_positions(obs_phys) if obs_phys.shape[0] else \
            np.zeros((self.horizon + 1, 0, 2), np.float32)
        obs_vel = (obs_phys[:, 2:4].astype(np.float32) if obs_phys.shape[0]
                   else np.zeros((0, 2), np.float32))

        # --- ONLINE MPPI: the candidate generator being benchmarked -----------
        t0 = time.perf_counter()
        cand_xy_ego = self._run_online_mppi(x0, obs_traj, obs_vel, geo, clear,
                                            subgoal, occ_np)
        self.timing_sampler.append(time.perf_counter() - t0)

        if cand_xy_ego is None:
            # The search ran and found nothing feasible (empty prefilter, or no
            # feasible MPPI refinement). Timed and counted for the same reason
            # as the geodesic failure above — and this is THE degradation mode
            # the budget sweep is trying to expose, so it must be a reported
            # number rather than something visible only as a worse success
            # rate downstream.
            self.timing.append(time.perf_counter() - _t_cand0)
            self._timing_ok.append(False)
            self.n_search_fail_cand += 1
            self._log_predict_total(t_predict_start, tag="SEARCH-FAIL(cand) ")
            return self._emergency_brake_action()

        # Stop the candidate-generation clock at the same point the diffusion
        # policy stops its own: one committed candidate in hand, projection
        # not yet run. (Pure CPU/numba here — nothing async to synchronize.)
        self.timing.append(time.perf_counter() - _t_cand0)
        self._timing_ok.append(True)

        # Per-tick, flushed, so a SLURM job streams progress live instead of
        # buffering the whole eval set. One glance shows the swept budget and
        # this tick's candidate-generation latency (the frontier x-axis);
        # `sampler` isolates the MPPI refinement from the FMM/geodesic setup.
        print(f"[mppi] b={self.mppi_budget:g} samples={self.mppi_samples} "
              f"pool={self.mppi_seed_pool} restarts={self.mppi_restarts} "
              f"cand={self.timing[-1]*1e3:.1f}ms "
              f"sampler={self.timing_sampler[-1]*1e3:.1f}ms", flush=True)

        # --- SHARED back-end: identical projection + action extraction ---------
        # Hand the candidate to the EXACT same back-end the diffusion policy
        # runs (_finish_from_candidate): projection, feasibility fallback,
        # predicted_traj bookkeeping, action extraction and enforce_lims. The
        # only difference between the two policies is how cand_xy_ego was
        # produced (online MPPI here vs. denoising there).
        return self._finish_from_candidate(
            cand_xy_ego, state, all_samples_world=None,
            t_predict_start=t_predict_start,
        )

    def _make_rng(self):
        """RNG for one online search.

        Offline the expert seeds per (scene, style) via seed_of(...), so
        generation replays exactly. Online there is no scene identity to hash,
        so we seed from a configured base plus a per-call counter: the draw
        still varies tick to tick (as a real deployment would), but an entire
        evaluation run replays identically — which a benchmark needs, and which
        matters here because latency is being compared against a deterministic
        diffusion forward pass. Set mppi_seed < 0 for a genuinely
        nondeterministic sampler.
        """
        self._call_idx += 1
        if self.mppi_seed is None or int(self.mppi_seed) < 0:
            return np.random.default_rng()
        return np.random.default_rng(int(self.mppi_seed) + self._call_idx)

    # -- online MPPI over the same cost ---------------------------------------
    def _run_online_mppi(self, x0, obs_traj, obs_vel, geo, clear, subgoal, occ_np):
        """Run the expert's MPPI at the configured (real-time-ish) budget and
        return the best (T,2) ego trajectory, or None if nothing feasible.

        This mirrors run_style()'s search but at DEPLOYMENT budget and for a
        SINGLE style (self.style_vector), because online we commit to one action.
        The budget knobs are what the frontier sweeps."""
        E = self._E
        H = self.horizon
        s = np.clip(self.style_vector, -1, 1).astype(np.float32)

        rng = self._make_rng()

        # ---- Phase 1: seed pool, mirroring the generator's composition ------
        # The expert draws N_UNIFORM uniform + N_SMOOTH smooth + a damped
        # `ws` block (+ the brake rollout) = 22% / 67% / 11%. That ratio is a
        # design choice, not an accident: the AR(1) `smooth` samples are the
        # realistic low-jerk trajectories, and the expert over-weights them 3:1
        # over near-white `uniform` noise. `ws` is smooth with its first 4
        # steps damped to 0.2, i.e. candidates that start gently.
        #
        # mppi_seed_pool scales all three components TOGETHER, preserving that
        # ratio, so shrinking the budget slides along the expert's own design
        # rather than silently changing what the sampler is. At the default
        # (18432) this pool is the generator's pool exactly.
        n_full = E.N_UNIFORM + E.N_SMOOTH + max(512, E.N_UNIFORM // 2)
        frac   = float(self.mppi_seed_pool) / float(n_full)
        nu = max(1, int(round(E.N_UNIFORM * frac)))
        ns = max(1, int(round(E.N_SMOOTH  * frac)))
        nw = max(1, int(round(max(512, E.N_UNIFORM // 2) * frac)))

        ws = E.sample_smooth(nw, H, rng)
        ws[:, :4] *= 0.2                       # damped start — expert's `ws`
        pool_u = np.concatenate([
            E.sample_uniform(nu, H, rng),
            E.sample_smooth(ns, H, rng),
            ws,
            E.brake_controls()[None],
        ], 0)
        brake_i   = pool_u.shape[0] - 1
        pool_traj = E.rollout(x0, pool_u)

        # Cheap feasibility prefilter (same as offline): drop pool trajs that
        # hit the static map or a pedestrian, before the social cost.
        nxt = pool_traj[:, 1:, 0:2]
        cheap = E.map_clearances(nxt, clear).min(1) >= E.ROBOT_R
        if obs_traj.shape[1]:
            cheap &= np.linalg.norm(
                nxt[:, :, None, :] - obs_traj[1:H + 1][None], axis=3
            ).min((1, 2)) >= E.R_C
        idx = np.where(cheap)[0]
        if idx.size == 0:
            return None

        # ---- Stratified presort (identical rule to the generator) -----------
        # Trimming by distance-to-goal alone would delete every slowing or
        # stopping candidate before the social cost is ever evaluated — i.e.
        # precisely the positive end of the yield axis. So half the budget goes
        # to goal-directed candidates (good seeds survive), half is drawn at
        # random (slow/waiting candidates survive), and the brake rollout is
        # force-included. Without this the online search would be biased
        # against exactly the demonstrations the yield axis is made of.
        if subgoal is not None and idx.size > E.PRESORT_K:
            gd   = ((nxt[idx, -1] - np.asarray(subgoal, np.float32)[None]) ** 2).sum(1)
            k    = E.PRESORT_K // 2
            top  = idx[np.argpartition(gd, k - 1)[:k]]
            rest = np.setdiff1d(idx, top)
            extra_i = (rng.choice(rest, size=min(k, rest.size), replace=False)
                       if rest.size else rest)
            idx = np.concatenate([top, extra_i])
        if cheap[brake_i] and brake_i not in idx:
            idx = np.append(idx, brake_i)

        pool_traj, pool_u = pool_traj[idx], pool_u[idx]
        G = E.geometry(pool_traj, obs_traj, obs_vel)

        # Rank the pool by the SAME total_cost, pick diverse seeds, MPPI-refine.
        c, f = E.total_cost(pool_traj, obs_traj, obs_vel, geo, clear, s, G=G)
        ok = np.where(f & np.isfinite(c))[0]
        if ok.size == 0:
            return None
        order = ok[np.argsort(c[ok])]
        n_seeds = min(int(self.mppi_restarts), order.size)
        seeds = E.farthest_point(E._feats(pool_traj[order]), n_seeds, 0.4)

        # Temporarily set the sampler's MPPI budget to the DEPLOYMENT budget.
        saved = (E.MPPI_N, E.MPPI_ITER)
        E.MPPI_N, E.MPPI_ITER = int(self.mppi_samples), int(self.mppi_iters)
        try:
            best_traj, best_cost = None, np.inf
            for i in seeds:
                tj, u, cst, feas = E.mppi(x0, obs_traj, obs_vel, geo, clear,
                                          pool_u[order[i]], rng, s)
                if tj is not None and feas and cst < best_cost:
                    best_traj, best_cost = tj, cst
        finally:
            E.MPPI_N, E.MPPI_ITER = saved

        if best_traj is None:
            # fall back to the best raw pool trajectory
            best_traj = pool_traj[order[0]]

        # Drop the final state: rollout() returns HORIZON+1 states, but the
        # generator saves rt[:-1] as `traj_xy` — the (HORIZON, 2) target the
        # diffusion model was actually trained on (rt[:] is stored separately as
        # `traj_xy_with_start`). Returning all HORIZON+1 points would hand the
        # shared projection a trajectory one node longer than the OCP was built
        # for (N = horizon - 1), and — more importantly — would mean the two
        # policies emit trajectories in DIFFERENT representations, which is
        # exactly the confound this baseline exists to avoid.
        cand = best_traj[:-1, 0:2].astype(np.float32)      # (HORIZON, 2) ego
        assert cand.shape[0] == self.horizon, (
            f"MPPI candidate has {cand.shape[0]} points, expected "
            f"horizon={self.horizon} (must match the generator's traj_xy "
            f"convention and the diffusion policy's output length)"
        )
        return cand

    # -- physical-unit conditioning (no norm round-trip) ----------------------
    def _physical_obstacles_ego(self, state):
        pos = np.array([state.self_state.px, state.self_state.py])
        th = state.self_state.theta
        R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        out = []
        for h in state.human_states:
            p = R @ (np.array([h.px, h.py]) - pos)
            v = R @ np.array([h.vx, h.vy])
            out.append([p[0], p[1], v[0], v[1]])
        return np.array(out, dtype=np.float32) if out else np.zeros((0, 4), np.float32)

    def _physical_goal_ego(self, state):
        pos = np.array([state.self_state.px, state.self_state.py])
        th = state.self_state.theta
        R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        return (R @ (np.array([state.self_state.gx, state.self_state.gy]) - pos)
                ).astype(np.float32)

    # NOTE: the projection + action-extraction back-end is inherited unchanged
    # from SoGuDiff._finish_from_candidate() — see the call
    # in predict(). It is deliberately NOT re-implemented here so the baseline
    # and the diffusion policy share the exact same code past candidate
    # generation.

    # -- reporting ------------------------------------------------------------
    def timing_summary(self, skip=10):
        """Parent's candidate-generation stats (identical definition and
        warmup handling as the diffusion policy) plus MPPI-specific context:
        the swept budget, the thread count that budget was realized on, and
        the sampler-only sub-timing."""
        out = super().timing_summary(skip=skip)
        if not out:
            return out
        # nanmedian: entries are nan on ticks where the geodesic failed before
        # the sampler ran. Those ticks are still counted in self.timing (they
        # are real control ticks) — the nan only says "no sampler time to
        # attribute here", and dropping the whole tick instead would put the
        # two lists out of alignment.
        ts = np.asarray(self.timing_sampler[skip:], dtype=np.float64) * 1e3
        ts_valid = ts[np.isfinite(ts)] if ts.size else ts
        n_fail = self.n_search_fail_geo + self.n_search_fail_cand

        # Latency over PLANNING ticks only — the cost of actually producing a
        # candidate, with no credit for the cheap ticks where the search gave
        # up. The inherited mean_ms/median_ms cover EVERY tick; both are
        # reported so neither definition can be accused of doing the other's
        # work. The mask is index-aligned with self.timing, so the same warmup
        # `skip` selects the same calls in both.
        tall = np.asarray(self.timing[skip:], dtype=np.float64) * 1e3
        mask = np.asarray(self._timing_ok[skip:], dtype=bool)
        if tall.size and mask.size == tall.size and mask.any():
            tok = tall[mask]
            out.update({
                "plan_n_calls":   int(tok.size),
                "plan_mean_ms":   float(tok.mean()),
                "plan_median_ms": float(np.median(tok)),
                "plan_p95_ms":    float(np.percentile(tok, 95)),
                "plan_max_ms":    float(tok.max()),
            })
        out.update({
            "sampler_only_median_ms": (float(np.median(ts_valid))
                                       if ts_valid.size else None),
            # Search failures are over the WHOLE run, not the post-warmup
            # window, so the fraction is reported against total predict()
            # calls seen by the latency stats for scale only.
            "mppi_search_fail_geo":  self.n_search_fail_geo,
            "mppi_search_fail_cand": self.n_search_fail_cand,
            "mppi_search_fail":      n_fail,
            "mppi_search_fail_frac": n_fail / float(len(self.timing) or 1),
            "mppi_budget":    self.mppi_budget,
            "mppi_samples":   self.mppi_samples,
            "mppi_iters":     self.mppi_iters,
            "mppi_restarts":  self.mppi_restarts,
            "mppi_seed_pool": self.mppi_seed_pool,
            # Latency here is meaningless without the thread count it was
            # measured at — numba parallelizes the rollout kernels.
            "numba_threads":  os.environ.get("NUMBA_NUM_THREADS", "<unset:all cores>"),
        })
        return out

    def reset_timing(self):
        super().reset_timing()
        self.timing_sampler = []
        self._timing_ok = []
        self.n_search_fail_geo = 0
        self.n_search_fail_cand = 0