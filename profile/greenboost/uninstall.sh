#!/bin/bash
# Remove GreenBoost. Safe to run at any time -- the module holds no state
# that survives, and the shim is only active in processes that LD_PRELOAD it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${GREENBOOST_SRC:-/opt/greenboost/src}"

printf '\n\033[1;34m== unloading\033[0m\n'
if lsmod | grep -q '^greenboost'; then
    # Anything still holding a pinned buffer would block this; there should be
    # nothing, because the module releases on process exit.
    rmmod greenboost && echo "module unloaded"
else
    echo "not loaded"
fi

printf '\n\033[1;34m== removing shim\033[0m\n'
if [ -f /usr/local/lib/libgreenboost_cuda.so ]; then
    rm -f /usr/local/lib/libgreenboost_cuda.so && echo "shim removed"
else
    echo "shim not installed"
fi

printf '\n\033[1;34m== restoring pristine sources\033[0m\n'
if [ -f "$SRC/greenboost_cuda_shim.c.orig" ]; then
    mv "$SRC/greenboost_cuda_shim.c.orig" "$SRC/greenboost_cuda_shim.c"
    echo "shim source restored"
else
    echo "no .orig to restore"
fi

printf '\n\033[1;34m== done\033[0m\n'
echo "Restart any container that sets LD_PRELOAD=libgreenboost_cuda.so."
echo "The test instance: docker compose -f compose-greenboost.yaml down"
echo