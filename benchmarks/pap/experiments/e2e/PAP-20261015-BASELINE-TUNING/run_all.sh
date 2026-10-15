#!/usr/bin/env bash
# Run every variant once per round, REPETITIONS rounds; failed runs are kept.
#
# Usage: bash run_all.sh [first_round]
set -uo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${EXPERIMENT_DIR}/experiment.env"
for ((round = ${1:-1}; round <= REPETITIONS; round++)); do
  for variant in ${VARIANTS}; do
    bash "${EXPERIMENT_DIR}/run.sh" "${variant}" "${round}"
  done
done
