#!/usr/bin/env bash
# Run every point of ROUND_ORDER (or the given variants) once as repetition <k>.
#
# Usage: bash run_round.sh <repetition> [variant ...]
set -uo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${EXPERIMENT_DIR}/experiment.env"
repetition="${1:?repetition is required}"
shift
if (($# == 0)); then
  # shellcheck disable=SC2086  # ROUND_ORDER is a word list
  set -- ${ROUND_ORDER}
fi
for variant in "$@"; do
  bash "${EXPERIMENT_DIR}/run.sh" "${variant}" "${repetition}"
done
