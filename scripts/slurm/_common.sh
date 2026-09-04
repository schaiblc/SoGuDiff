# Shared preamble for the job scripts in this directory. Not run directly.
set -euo pipefail

SOGUDIFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SOGUDIFF_ROOT
cd "$SOGUDIFF_ROOT"

# Local overrides win, and are gitignored.
# shellcheck disable=SC1091
source "$SOGUDIFF_ROOT/scripts/slurm/cluster_env.sh"
if [ -f "$SOGUDIFF_ROOT/scripts/slurm/cluster_env.local.sh" ]; then
    # shellcheck disable=SC1091
    source "$SOGUDIFF_ROOT/scripts/slurm/cluster_env.local.sh"
fi

mkdir -p "$SOGUDIFF_ROOT/logs"
