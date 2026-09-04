"""Repository-relative path resolution.

Config files ship paths written relative to the repository root, optionally
using the ``${SOGUDIFF_ROOT}`` placeholder. This module turns those into
absolute paths at load time so a clone works from any location without editing
configs.

Resolution order for the repository root:

1. ``$SOGUDIFF_ROOT``, if set. Set this when running from a directory outside
   the repository, or when the package is installed rather than used in place.
2. Otherwise, the third parent of this file (``<root>/crowdnav_env/crowd_nav``).

Typical use, at each site that reads a path out of a ``.config`` file::

    from crowd_nav.paths import resolve_path
    checkpoint = resolve_path(config.get('height', 'checkpoint'))
"""

import os
import shutil

# <root>/crowdnav_env/crowd_nav/paths.py -> <root>
_DEFAULT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)


def repo_root():
    """Absolute path to the repository root."""
    return os.path.abspath(os.environ.get('SOGUDIFF_ROOT', _DEFAULT_ROOT))


def ffmpeg_binary():
    """The ffmpeg executable to shell out to when concatenating episode clips.

    Prefers ``$SOGUDIFF_FFMPEG``, which is how a cluster points at a
    module-provided build, then whatever is on ``PATH``. Falls back to the bare
    name so the failure surfaces as ffmpeg's own error rather than here.
    """
    return os.environ.get('SOGUDIFF_FFMPEG') or shutil.which('ffmpeg') or 'ffmpeg'


def resolve_path(path):
    """Expand a config-supplied path to an absolute one.

    Handles ``${SOGUDIFF_ROOT}`` and other environment variables, ``~``, and
    plain relative paths (interpreted against the repository root, not the
    current working directory). Absolute paths pass through unchanged, so a
    user pointing a config at an asset stored outside the repository still
    works. Empty input is returned as-is for callers that treat it as "unset".
    """
    if not path:
        return path
    os.environ.setdefault('SOGUDIFF_ROOT', repo_root())
    path = os.path.expanduser(os.path.expandvars(str(path).strip()))
    if not os.path.isabs(path):
        path = os.path.join(repo_root(), path)
    return os.path.normpath(path)
