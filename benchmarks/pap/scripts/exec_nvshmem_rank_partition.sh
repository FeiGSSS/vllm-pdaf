#!/usr/bin/env bash
set -euo pipefail

rank="${PMI_RANK:-${PMI_ID:-${PMIX_RANK:-}}}"
case "${rank}" in
  0)
    export CUDA_VISIBLE_DEVICES="${PAP_NVSHMEM_GPU_0:?}"
    export CUDA_MPS_SM_PARTITION="${PAP_NVSHMEM_PARTITION_0:?}"
    ;;
  1)
    export CUDA_VISIBLE_DEVICES="${PAP_NVSHMEM_GPU_1:?}"
    export CUDA_MPS_SM_PARTITION="${PAP_NVSHMEM_PARTITION_1:?}"
    ;;
  *)
    echo "ERROR: unresolved NVSHMEM rank: ${rank:-unset}" >&2
    exit 2
    ;;
esac

exec "$@"
