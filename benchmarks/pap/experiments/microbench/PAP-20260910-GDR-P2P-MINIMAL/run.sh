#!/usr/bin/env bash
# Minimal GPUDirect P2P RDMA repro: orchestration entry point.
#
# Run this on the SINK peer (L20B). The sink is the victim: it may hard-reset
# when the platform fault triggers. Only the L20B -> L20T SSH direction is
# required. Everything that must survive a reset (source process, liveness
# witness) is launched detached on L20T, and the sink artifacts are written to
# local disk, which survives the reset.
#
# Usage:
#   bash run.sh --where cpu  --duration-s 60    # control arm
#   bash run.sh --where cuda --duration-s 180   # treatment arm
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${EXP_DIR}/../../../../.." && pwd)"

PEER_HOST="${PEER_HOST:-L20T}"
PEER_IP="${PEER_IP:-10.128.7.25}"
PEER_UCX_DEVICE="${PEER_UCX_DEVICE:-mlx5_0:1}"
LOCAL_IP="${LOCAL_IP:-10.128.7.24}"
LOCAL_UCX_DEVICE="${LOCAL_UCX_DEVICE:-ibp49s0:1}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-dynamo/bin/python}"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
SCP_OPTS=(-q -o BatchMode=yes -o StrictHostKeyChecking=accept-new)

WHERE="cuda"
CHUNK_MB="1400"
NUM_DESCS="1"
MAX_INFLIGHT="8"
DURATION_S="180"
SINK_GPU="0"
SRC_GPU="0"
COMPUTE_GEMM="0"
COMPUTE_H2D_MB="8"
TARGET_GBPS="0"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_gdr_p2p"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --where) WHERE="$2"; shift 2 ;;
    --chunk-mb) CHUNK_MB="$2"; shift 2 ;;
    --num-descs) NUM_DESCS="$2"; shift 2 ;;
    --max-inflight) MAX_INFLIGHT="$2"; shift 2 ;;
    --duration-s) DURATION_S="$2"; shift 2 ;;
    --sink-gpu) SINK_GPU="$2"; shift 2 ;;
    --src-gpu) SRC_GPU="$2"; shift 2 ;;
    --compute-gemm) COMPUTE_GEMM="$2"; shift 2 ;;
    --compute-h2d-mb) COMPUTE_H2D_MB="$2"; shift 2 ;;
    --target-gbps) TARGET_GBPS="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${WHERE}" == "cuda" || "${WHERE}" == "cpu" ]] || {
  echo "--where must be cuda or cpu" >&2; exit 2;
}

# shellcheck source=../../../scripts/configure_cross_host_nixl.sh
source "${ROOT_DIR}/benchmarks/pap/scripts/configure_cross_host_nixl.sh"

LOCAL_HOSTNAME="$(hostname)"
PEER_HOSTNAME="$(ssh "${SSH_OPTS[@]}" "${PEER_HOST}" hostname)"
[[ "${LOCAL_HOSTNAME}" != "${PEER_HOSTNAME}" ]] || {
  echo "refusing to run: local host equals source peer (${LOCAL_HOSTNAME})" >&2
  exit 1
}

if ! pap_configure_cross_host_nixl "${ROOT_DIR}" "${LOCAL_IP}" "${LOCAL_UCX_DEVICE}"; then
  echo "sink-side NIXL/UCX preflight failed" >&2
  exit 1
fi
SINK_ENV_LD="${LD_LIBRARY_PATH:-}"
SINK_ENV_PREFIX="${PAP_UCX_PREFIX}"
SINK_ENV_PLUGIN="${PAP_NIXL_PLUGIN_DIR}"
SINK_ENV_MODULE="${UCX_MODULE_DIR}"
SINK_ENV_RCACHE="${UCX_RCACHE_MAX_UNRELEASED}"
SINK_ENV_UCXV="${PAP_NIXL_UCX_VERSION}"

RUN_DIR="${EXP_DIR}/runs/${RUN_ID}"
PEER_RUN_DIR="${PEER_RUN_DIR:-${EXP_DIR}/runs/${RUN_ID}}"
mkdir -p "${RUN_DIR}"

cleanup_remote_probes() {
  ssh "${SSH_OPTS[@]}" "${PEER_HOST}" \
    "pkill -f 'source_launc[h].sh' 2>/dev/null; \
     pkill -f 'probe_gdr_p2p.py --role sourc[e]' 2>/dev/null; \
     pkill -f 'witness_launc[h].sh' 2>/dev/null; \
     pkill -f 'witness_liveness.p[y]' 2>/dev/null; true" || true
}
pkill -f 'probe_gdr_p2p.py --role sin[k]' 2>/dev/null || true
cleanup_remote_probes

# Source environment: same stack, source-local device and IP.
cat > "${RUN_DIR}/source.env" <<EOF
export PAP_NIXL_RUNTIME_MODE=cross_host_ucx122_rdma
export PAP_UCX_PREFIX=${SINK_ENV_PREFIX}
export PAP_NIXL_PLUGIN_DIR=${SINK_ENV_PLUGIN}
export NIXL_PLUGIN_DIR=${SINK_ENV_PLUGIN}
export UCX_MODULE_DIR=${SINK_ENV_MODULE}
export LD_LIBRARY_PATH=${SINK_ENV_PREFIX}/lib:${SINK_ENV_LD}
export UCX_PROTO_EMULATION_ENABLE=n
export UCX_TLS=rc,cuda_copy
export UCX_NET_DEVICES=${PEER_UCX_DEVICE}
export UCX_RCACHE_MAX_UNRELEASED=${SINK_ENV_RCACHE}
export VLLM_NIXL_SIDE_CHANNEL_HOST=${PEER_IP}
export PYTHONUNBUFFERED=1
EOF

{
  echo "local_hostname=${LOCAL_HOSTNAME}"
  echo "peer_hostname=${PEER_HOSTNAME}"
  echo "where=${WHERE}"
  echo "chunk_mb=${CHUNK_MB}"
  echo "num_descs=${NUM_DESCS}"
  echo "max_inflight=${MAX_INFLIGHT}"
  echo "duration_s=${DURATION_S}"
  echo "sink_gpu=${SINK_GPU}"
  echo "src_gpu=${SRC_GPU}"
  echo "compute_gemm=${COMPUTE_GEMM}"
  echo "compute_h2d_mb=${COMPUTE_H2D_MB}"
  echo "target_gbps=${TARGET_GBPS}"
  echo "sink_ucx_device=${LOCAL_UCX_DEVICE}"
  echo "source_ucx_device=${PEER_UCX_DEVICE}"
  echo "ucx_version=${SINK_ENV_UCXV}"
  echo "sink_boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "sink_cmdline=$(cat /proc/cmdline)"
  echo "sink_kernel=$(uname -r)"
  echo "git_commit=$(cd "${ROOT_DIR}" && git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "peer_boot_id=$(ssh "${SSH_OPTS[@]}" "${PEER_HOST}" 'cat /proc/sys/kernel/random/boot_id')"
  echo "peer_uptime_since=$(ssh "${SSH_OPTS[@]}" "${PEER_HOST}" 'uptime -s')"
} > "${RUN_DIR}/effective_config.txt"
# A panic does not flush dirty page cache, so persist the run metadata now:
# after a reset these files are the only record of the effective configuration.
sync

# Fail closed on contaminated hardware state: no GPU compute processes anywhere.
sink_idle="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
peer_idle="$(ssh "${SSH_OPTS[@]}" "${PEER_HOST}" \
  "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")"
[[ "${sink_idle}" == "0" && "${peer_idle}" == "0" ]] || {
  echo "refusing to run: sink=${sink_idle} peer=${peer_idle} GPU compute processes" >&2
  exit 1
}

echo "run_dir=${RUN_DIR}"
echo "sink=${LOCAL_HOSTNAME} where=${WHERE} duration=${DURATION_S}s source=${PEER_HOSTNAME}"

ssh "${SSH_OPTS[@]}" "${PEER_HOST}" "mkdir -p '${PEER_RUN_DIR}'"

# Remote launchers are files, not inline strings: a backgrounded compound
# command leaves an unredirected subshell holding the SSH channel open, so the
# launch never returns. A plain script behind nohup/setsid does not.
cat > "${RUN_DIR}/source_launch.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
set -a
. "${PEER_RUN_DIR}/source.env"
set +a
exec "${ROOT_DIR}/.venv-dynamo/bin/python" "${EXP_DIR}/probe_gdr_p2p.py" \\
  --role source --agent-name pap-src --gpu ${SRC_GPU} --where cuda \\
  --chunk-mb ${CHUNK_MB} --duration-s $((DURATION_S + 240)) \\
  --meta-file "${PEER_RUN_DIR}/meta.json" \\
  --heartbeat-file "${PEER_RUN_DIR}/source-heartbeat.jsonl"
EOF

cat > "${RUN_DIR}/witness_launch.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec "${ROOT_DIR}/.venv-dynamo/bin/python" "${PEER_RUN_DIR}/witness_liveness.py" \\
  --target "${LOCAL_IP}" --ports 22 --out "${PEER_RUN_DIR}/witness.jsonl" \\
  --seconds $((DURATION_S + 300))
EOF
chmod +x "${RUN_DIR}/source_launch.sh" "${RUN_DIR}/witness_launch.sh"
sync

scp "${SCP_OPTS[@]}" "${RUN_DIR}/source.env" "${RUN_DIR}/source_launch.sh" \
  "${RUN_DIR}/witness_launch.sh" "${EXP_DIR}/witness_liveness.py" \
  "${PEER_HOST}:${PEER_RUN_DIR}/"

# Source side: publish the buffer the sink will READ from, detached from this
# SSH channel so the launch returns immediately.
ssh "${SSH_OPTS[@]}" "${PEER_HOST}" \
  "nohup setsid '${PEER_RUN_DIR}/source_launch.sh' > '${PEER_RUN_DIR}/source.log' 2>&1 < /dev/null & echo \$!" \
  > "${RUN_DIR}/source.pid"

for _ in $(seq 1 90); do
  ssh "${SSH_OPTS[@]}" "${PEER_HOST}" \
    "test -f '${PEER_RUN_DIR}/meta.json'" 2>/dev/null && break
  sleep 2
done
scp "${SCP_OPTS[@]}" "${PEER_HOST}:${PEER_RUN_DIR}/meta.json" "${RUN_DIR}/meta.json"
echo "source_pid=$(cat "${RUN_DIR}/source.pid")"

# Independent liveness witness for the sink host, running on the peer.
ssh "${SSH_OPTS[@]}" "${PEER_HOST}" \
  "nohup setsid '${PEER_RUN_DIR}/witness_launch.sh' > '${PEER_RUN_DIR}/witness.log' 2>&1 < /dev/null & echo \$!" \
  > "${RUN_DIR}/witness.pid"
echo "witness_pid=$(cat "${RUN_DIR}/witness.pid")"

# Sink side: the process under test.
SINK_EXIT=0
"${PYTHON_BIN}" "${EXP_DIR}/probe_gdr_p2p.py" \
  --role sink \
  --agent-name pap-sink \
  --gpu "${SINK_GPU}" \
  --where "${WHERE}" \
  --chunk-mb "${CHUNK_MB}" \
  --num-descs "${NUM_DESCS}" \
  --max-inflight "${MAX_INFLIGHT}" \
  --duration-s "${DURATION_S}" \
  --compute-gemm "${COMPUTE_GEMM}" \
  --compute-h2d-mb "${COMPUTE_H2D_MB}" \
  --target-gbps "${TARGET_GBPS}" \
  --meta-file "${RUN_DIR}/meta.json" \
  --heartbeat-file "${RUN_DIR}/sink-heartbeat.jsonl" \
  > "${RUN_DIR}/sink.log" 2>&1 || SINK_EXIT=$?

# Surviving-sink path only: a reset skips all of this.
cleanup_remote_probes
scp "${SCP_OPTS[@]}" "${PEER_HOST}:${PEER_RUN_DIR}/witness.jsonl" \
  "${RUN_DIR}/witness.jsonl" 2>/dev/null || true

last_heartbeat="$(tail -n 1 "${RUN_DIR}/sink-heartbeat.jsonl" 2>/dev/null || echo '{}')"
verdict="sink_completed"
[[ "${SINK_EXIT}" -eq 0 ]] || verdict="sink_exit_${SINK_EXIT}"
cat > "${RUN_DIR}/verdict.json" <<EOF
{
  "verdict": "${verdict}",
  "where": "${WHERE}",
  "chunk_mb": ${CHUNK_MB},
  "duration_s": ${DURATION_S},
  "sink_host": "${LOCAL_HOSTNAME}",
  "sink_gpu": ${SINK_GPU},
  "compute_gemm": ${COMPUTE_GEMM},
  "compute_h2d_mb": ${COMPUTE_H2D_MB},
  "target_gbps": ${TARGET_GBPS},
  "sink_exit_code": ${SINK_EXIT},
  "last_sink_heartbeat": ${last_heartbeat},
  "run_dir": "${RUN_DIR}"
}
EOF

scp "${SCP_OPTS[@]}" "${RUN_DIR}/sink.log" "${RUN_DIR}/sink-heartbeat.jsonl" \
  "${RUN_DIR}/verdict.json" "${RUN_DIR}/effective_config.txt" \
  "${PEER_HOST}:${PEER_RUN_DIR}/" 2>/dev/null || true

echo "verdict=${verdict}"
echo "artifacts=${RUN_DIR}"
