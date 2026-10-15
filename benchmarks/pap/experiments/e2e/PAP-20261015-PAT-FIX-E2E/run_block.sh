#!/usr/bin/env bash
# Run block A or B in its recorded order; failed runs are kept and reported.
#
# Usage: bash run_block.sh <A|B>
set -uo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${EXPERIMENT_DIR}/experiment.env"
case "${1:?block must be A or B}" in
  A) order="${BLOCK_A}" ;;
  B) order="${BLOCK_B}" ;;
  *) echo "ERROR: block must be A or B" >&2; exit 2 ;;
esac
for item in ${order}; do
  bash "${EXPERIMENT_DIR}/run.sh" "${item%%:*}" "${item##*:}"
done
