#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_CONFIG="${EXPERIMENT_DIR}/experiment.env"
# shellcheck source=/dev/null
source "${EXPERIMENT_CONFIG}"

RUN_ID="${PAP_ATTENTION_SCALING_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export PAP_ATTENTION_SCALING_EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG}"
export PAP_ATTENTION_SCALING_OUTPUT_ROOT="${EXPERIMENT_DIR}/runs/${RUN_ID}"
exec bash "${EXPERIMENT_DIR}/driver.sh"
