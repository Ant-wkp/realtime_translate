#!/usr/bin/env bash
# Launch the real-time subtitle pipeline.
#
# Works on both the x86 development host (conda env) and the Jetson (venv),
# because it activates whichever environment it finds instead of hard-coding
# one. `exec python -u` matters: without -u the stdout block buffer swallows
# the [首字]/[耗时] lines once output is redirected to a file.
set -euo pipefail

# The package directory, so `python main.py` works regardless of where the
# script is invoked from (main.py injects its own package name for this).
cd "$(dirname "$(readlink -f "$0")")"

CONDA_ENV="${CONDA_ENV:-funasr}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/subtitle}"

if [ -n "${VIRTUAL_ENV:-}" ]; then
    : # already inside a venv
elif [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$VENV_DIR/bin/activate"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    echo "run.sh: no venv at $VENV_DIR and no conda; using system python" >&2
fi

# Device and capture channel are not configurable: the XVF3800 is detected by
# name fragment and channel 0 (the conference processed output) is fixed.
exec python -u main.py "$@"