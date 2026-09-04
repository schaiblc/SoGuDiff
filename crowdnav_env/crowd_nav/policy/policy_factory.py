"""Registry mapping ``--policy`` names to implementations.

Every name below corresponds to a row in the paper's comparison table. Learned
baselines (DSRNN, NaviSTAR, HEIGHT) are evaluation-only adapters: they load a
checkpoint trained by the corresponding fork under ``baselines/``, using that
fork's own training pipeline, and expose it through this repository's policy
interface. Their checkpoint and source-tree locations come from the policy
config, so retraining means pointing the config at a new checkpoint rather than
editing code.

Baselines are evaluated under the same unicycle action space as the proposed
method. Where an author's published model is holonomic and could not be
retrained unicycle-native, the plain name enforces the unicycle limits at
evaluation time and is the reported configuration. The matching
``*_holonomic`` name runs the same checkpoint under its native dynamics: a
diagnostic variant that quantifies what the conversion costs, not reported and
not comparable to the other rows.
"""

from crowd_sim.envs.policy.policy_factory import policy_factory

# --- Reactive / classical ---------------------------------------------------
from crowd_sim.envs.policy.linear import Linear
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.orca_unicycle import ORCAUnicycle
from crowd_sim.envs.policy.social_force import SFM
from crowd_sim.envs.policy.sfm_unicycle import SFMUnicycle

# --- Learned baselines ------------------------------------------------------
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.rgl import RGL
from crowd_nav.policy.dsrnn import DSRNNHolonomic, DSRNN
from crowd_nav.policy.navistar import NaviSTAR, NaviSTARHolonomic
from crowd_nav.policy.height import HEIGHT, HEIGHTHolonomic
from crowd_nav.policy.sicnav import SICNav

# --- Proposed method and its ablation ---------------------------------------
# Both reach acados through projection_solver.py. acados is a heavy optional
# dependency and is deliberately absent from some environments -- notably
# sicnav_np_env, which the SICNav baseline needs for its CasADi/IPOPT stack
# (see crowd_nav/policy/sicnav.py). Importing lazily keeps every other policy
# usable there; only requesting one of these two by name fails, and then with a
# message naming the missing package.
# The name bound by ``except ... as`` is deleted when the clause exits, so the
# original error is stashed in a module-level variable first: a closure that
# referred to the clause name directly would raise NameError when finally
# called, hiding the message it exists to deliver.
_cfg_err = None
try:
    from crowd_nav.policy.sogudiff import SoGuDiff
except ImportError as _err:
    _cfg_err = _err

    def SoGuDiff(*args, **kwargs):
        raise ImportError(
            "policy 'sogudiff' requires acados_template, "
            "which is not installed in this environment. See docs/INSTALL.md."
        ) from _cfg_err

_mppi_err = None
try:
    from crowd_nav.policy.mppi_expert import MPPIExpert
except ImportError as _err:
    _mppi_err = _err

    def MPPIExpert(*args, **kwargs):
        raise ImportError(
            "policy 'mppi_expert' requires acados_template, which is not "
            "installed in this environment. See docs/INSTALL.md."
        ) from _mppi_err


# --- Reactive / classical ---------------------------------------------------
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['orca_unicycle'] = ORCAUnicycle
policy_factory['sfm'] = SFM
policy_factory['sfm_unicycle'] = SFMUnicycle

# --- Learned baselines ------------------------------------------------------
policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['rgl'] = RGL

# DSRNN: the published model is holonomic. 'dsrnn_holonomic' retrains the author's
# recipe on this repository's evaluation task; 'dsrnn' projects
# that checkpoint's (vx, vy) output onto the shared unicycle envelope at
# evaluation time, and is the reported configuration.
policy_factory['dsrnn_holonomic'] = DSRNNHolonomic
policy_factory['dsrnn'] = DSRNN

# NaviSTAR and HEIGHT: retrained with the authors' own recipes on this
# evaluation task (see baselines/navistar and baselines/height).
policy_factory['navistar'] = NaviSTAR
policy_factory['navistar_holonomic'] = NaviSTARHolonomic
policy_factory['height'] = HEIGHT
policy_factory['height_holonomic'] = HEIGHTHolonomic

policy_factory['sicnav'] = SICNav

# --- Proposed method and its ablation ---------------------------------------
# 'sogudiff' is SoGuDiff. 'mppi_expert' runs the offline
# expert's own sampling-based planner online, sharing the projection back-end,
# so a metric gap is attributable to the planner rather than the feasibility
# layer.
policy_factory['sogudiff'] = SoGuDiff
policy_factory['mppi_expert'] = MPPIExpert
