#!/usr/bin/env bash
# Same-host IB loopback variant.
#
# Both NIXL agents run on THIS host, but the UCX transport list is forced to
# the production cross-host one (`rc,cuda_copy` on the local HCA), so the
# receiving GPU is still the target of NIC PCIe posted writes. This removes
# cross-host networking as an explanation for the fatal.
#
# Run on the host that should crash (the sink). Usage:
#   bash run_loopback.sh --where cuda --sink-gpu 0 --compute-gemm 8192
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${EXP_DIR}/../../../../.." && pwd)"

LOCAL_IP="${LOCAL_IP:-10.128.7.24}"
LOCAL_UCX_DEVICE="${LOCAL_UCX_DEVICE:-ibp49s0:1}"
WITNESS_HOST="${WITNESS_HOST:-L20T}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-dynamo/bin/python}"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
SCP_OPTS=(-q -o BatchMode=yes -o StrictHostKeyChecking=accept-new)

WHERE="cuda"
CHUNK_MB="1400"
MAX_INFLIGHT="8"
DURATION_S="180"
SINK_GPU="0"
SRC_GPU="1"
COMPUTE_GEMM="0"
TARGET_GBPS="0"
UCX_DEBUG="0"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_gdr_loopback"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --where) WHERE="$2"; shift 2 ;;
    --chunk-mb) CHUNK_MB="$2"; shift 2 ;;
    --max-inflight) MAX_INFLIGHT="$2"; shift 2 ;;
    --duration-s) DURATION_S="$2"; shift 2 ;;
    --sink-gpu) SINK_GPU="$2"; shift 2 ;;
    --src-gpu) SRC_GPU="$2"; shift 2 ;;
    --compute-gemm) COMPUTE_GEMM="$2"; shift 2 ;;
    --target-gbps) TARGET_GBPS="$2"; shift 2 ;;
    --ucx-debug) UCX_DEBUG="1"; shift ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${WHERE}" == "cuda" || "${WHERE}" == "cpu" ]] || {
  echo "--where must be cuda or cpu" >&2; exit 2;
}
[[ "${SINK_GPU}" != "${SRC_GPU}" ]] || {
  echo "sink and source must use different GPUs" >&2; exit 2;
}

# shellcheck source=../../../scripts/configure_cross_host_nixl.sh
source "${ROOT_DIR}/benchmarks/pap/scripts/configure_cross_host_nixl.sh"
if ! pap_configure_cross_host_nixl "${ROOT_DIR}" "${LOCAL_IP}" "${LOCAL_UCX_DEVICE}"; then
  echo "NIXL/UCX preflight failed" >&2
  exit 1
fi
[[ "${UCX_DEBUG}" == "1" ]] && export UCX_LOG_LEVEL=debug

RUN_DIR="${EXP_DIR}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

{
  echo "mode=same_host_ib_loopback"
  echo "hostname=$(hostname)"
  echo "local_ip=${LOCAL_IP}"
  echo "ucx_device=${LOCAL_UCX_DEVICE}"
  echo "ucx_tls=${UCX_TLS}"
  echo "ucx_version=${PAP_NIXL_UCX_VERSION}"
  echo "where=${WHERE}"
  echo "chunk_mb=${CHUNK_MB}"
  echo "max_inflight=${MAX_INFLIGHT}"
  echo "duration_s=${DURATION_S}"
  echo "sink_gpu=${SINK_GPU}"
  echo "src_gpu=${SRC_GPU}"
  echo "compute_gemm=${COMPUTE_GEMM}"
  echo "target_gbps=${TARGET_GBPS}"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "kernel=$(uname -r)"
  echo "cmdline=$(cat /proc/cmdline)"
  echo "git_commit=$(cd "${ROOT_DIR}" && git rev-parse HEAD 2>/dev/null || echo unknown)"
} > "${RUN_DIR}/effective_config.txt"
sync

sink_idle="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
[[ "${sink_idle}" == "0" ]] || {
  echo "refusing to run: ${sink_idle} GPU compute processes" >&2
  exit 1
}

echo "run_dir=${RUN_DIR}"
echo "loopback on $(hostname): sink=gpu${SINK_GPU} src=gpu${SRC_GPU} where=${WHERE} rate<=${TARGET_GBPS:-unlimited}"

cleanup() {
  [[ -n "${SRC_PID:-}" ]] && kill "${SRC_PID}" 2>/dev/null || true
  ssh "${SSH_OPTS[@]}" "${WITNESS_HOST}" "pkill -f 'witness_liveness.p[y]' 2>/dev/null; true" >/dev/null 2>&1 || true
}
trap cleanup EXIT

"${PYTHON_BIN}" "${EXP_DIR}/probe_gdr_p2p.py" \
  --role source --agent-name loop-src --gpu "${SRC_GPU}" --where cuda \
  --chunk-mb "${CHUNK_MB}" --duration-s "$((DURATION_S + 240))" \
  --meta-file "${RUN_DIR}/meta.json" \
  --heartbeat-file "${RUN_DIR}/source-heartbeat.jsonl" \
  > "${RUN_DIR}/source.log" 2>&1 &
SRC_PID=$!

for _ in $(seq 1 90); do
  [[ -f "${RUN_DIR}/meta.json" ]] && break
  sleep 2
done
[[ -f "${RUN_DIR}/meta.json" ]] || { echo "source did not publish metadata" >&2; exit 1; }

# Independent liveness witness on the peer host (it stays up even when this one dies).
ssh "${SSH_OPTS[@]}" "${WITNESS_HOST}" "mkdir -p '${RUN_DIR}'"
scp "${SCP_OPTS[@]}" "${EXP_DIR}/witness_liveness.py" "${WITNESS_HOST}:${RUN_DIR}/"
ssh "${SSH_OPTS[@]}" "${WITNESS_HOST}" \
  "nohup setsid '${ROOT_DIR}/.venv-dynamo/bin/python' '${RUN_DIR}/witness_liveness.py' \
   --target '${LOCAL_IP}' --ports 22 --out '${RUN_DIR}/witness.jsonl' \
   --seconds '$((DURATION_S + 300))' > '${RUN_DIR}/witness.log' 2>&1 < /dev/null & echo \$!" \
  > "${RUN_DIR}/witness.pid" 2>/dev/null || true

SINK_EXIT=0
"${PYTHON_BIN}" "${EXP_DIR}/probe_gdr_p2p.py" \
  --role sink --agent-name loop-snk --gpu "${SINK_GPU}" --where "${WHERE}" \
  --chunk-mb "${CHUNK_MB}" --max-inflight "${MAX_INFLIGHT}" \
  --duration-s "${DURATION_S}" --compute-gemm "${COMPUTE_GEMM}" \
  --target-gbps "${TARGET_GBPS}" \
  --meta-file "${RUN_DIR}/meta.json" \
  --heartbeat-file "${RUN_DIR}/sink-heartbeat.jsonl" \
  > "${RUN_DIR}/sink.log" 2>&1 || SINK_EXIT=$?

kill "${SRC_PID}" 2>/dev/null || true
SRC_PID=""

last_heartbeat="$(tail -n 1 "${RUN_DIR}/sink-heartbeat.jsonl" 2>/dev/null || echo '{}')"
verdict="sink_completed"
[[ "${SINK_EXIT}" -eq 0 ]] || verdict="sink_exit_${SINK_EXIT}"
cat > "${RUN_DIR}/verdict.json" <<EOF
{
  "verdict": "${verdict}",
  "mode": "same_host_ib_loopback",
  "where": "${WHERE}",
  "compute_gemm": ${COMPUTE_GEMM},
  "target_gbps": ${TARGET_GBPS},
  "sink_exit_code": ${SINK_EXIT},
  "last_sink_heartbeat": ${last_heartbeat},
  "run_dir": "${RUN_DIR}"
}
EOF
sync
echo "verdict=${verdict}"
echo "artifacts=${RUN_DIR}"
