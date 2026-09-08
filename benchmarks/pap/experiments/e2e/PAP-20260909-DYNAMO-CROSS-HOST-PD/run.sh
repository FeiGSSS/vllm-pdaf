#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../../.." && pwd)"
RUN_ID="${DYNAMO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_cross_host_pd_7p1d}"

set -a
source "${HERE}/experiment.env"
set +a

export DYNAMO_RUN_ID="${RUN_ID}"
export DYNAMO_RUN_ROOT="${HERE}/runs/${RUN_ID}"
export DYNAMO_REMOTE_RUN_ROOT="${ROOT}/benchmarks/pap/experiments/_staging/${RUN_ID}_remote"
exec bash "${ROOT}/benchmarks/pap/scripts/run_cross_host_dynamo_7p1d_aiperf.sh"
