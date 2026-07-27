#!/usr/bin/env bash
set -euo pipefail

# Canonical eight-GPU Qwen3-8B capacity scan. The matrix runner owns dataset
# generation and architecture lifecycle; AIPerf owns each concurrency sweep.

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
MATRIX_RUNNER="${ROOT_DIR}/benchmarks/pap/aiperf/run_capacity_matrix.sh"

export PAP_CAPACITY_MATRIX_ID="${PAP_CAPACITY_MATRIX_ID:-$(date +%Y%m%d_%H%M%S)_8gpu_goodput_scan}"
export PAP_CAPACITY_ARCHITECTURES="${PAP_CAPACITY_ARCHITECTURES:-pap_7pa1p,pap_6pa2p,pd_4p4d,pd_6p2d,dp_8}"

# PAP's earlier boundaries were C16 for Strict and C24-C32 for the higher
# tiers. The intermediate points distinguish a real boundary from one noisy
# measurement without repeating the low-capacity region.
export PAP_CAPACITY_PAP_7PA1P_POINTS="${PAP_CAPACITY_PAP_7PA1P_POINTS:-16,20,24,28,32}"
export PAP_CAPACITY_PAP_6PA2P_POINTS="${PAP_CAPACITY_PAP_6PA2P_POINTS:-16,20,24,28,32}"

# Current NIXL materially changes PD, so refresh its lower Strict boundary and
# its C24-C32 saturation region.
export PAP_CAPACITY_PD_4P4D_POINTS="${PAP_CAPACITY_PD_4P4D_POINTS:-12,16,20,24,28,32}"
export PAP_CAPACITY_PD_6P2D_POINTS="${PAP_CAPACITY_PD_6P2D_POINTS:-12,16,20,24,28,32}"

# DP is unaffected by PAP/NIXL changes, but remains in the canonical full scan
# so a fresh campaign can be reproduced without importing historical results.
export PAP_CAPACITY_DP_8_POINTS="${PAP_CAPACITY_DP_8_POINTS:-8,12,16,20,24,28,32}"

export PAP_CAPACITY_REPETITIONS="${PAP_CAPACITY_REPETITIONS:-1}"
export PAP_CAPACITY_PAP_ROUTING_POLICY="${PAP_CAPACITY_PAP_ROUTING_POLICY:-conversation_affinity}"
export PAP_CAPACITY_PAP_MIGRATION_MIN_PEAK_GAIN_RATIO="${PAP_CAPACITY_PAP_MIGRATION_MIN_PEAK_GAIN_RATIO:-0.30}"

exec "${MATRIX_RUNNER}"
