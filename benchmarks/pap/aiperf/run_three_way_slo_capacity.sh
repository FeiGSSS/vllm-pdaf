#!/usr/bin/env bash
set -euo pipefail

# End-to-end DP/PD/PAP SLO-aware capacity scan driven by AI Perf.

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
MATRIX_RUNNER="${ROOT_DIR}/benchmarks/pap/aiperf/run_capacity_matrix.sh"
SUMMARY_SCRIPT="${ROOT_DIR}/benchmarks/pap/aiperf/summarize_capacity_matrix.py"
COMPARE_SCRIPT="${ROOT_DIR}/benchmarks/pap/aiperf/compare_three_way_slo.py"

export PAP_CAPACITY_POINTS="${PAP_CAPACITY_POINTS:-16,24,32,48}"
export PAP_CAPACITY_MATRIX_ID="${PAP_CAPACITY_MATRIX_ID:-$(date +%Y%m%d_%H%M%S)_dp_pd_pap_slo_scan}"
export PAP_CAPACITY_ARCHITECTURES="${PAP_CAPACITY_ARCHITECTURES:-dp_8,pd_4p4d,pd_6p2d,pd_7p1d,pap_6pa2p,pap_7pa1p}"
export PAP_CAPACITY_DP_8_POINTS="${PAP_CAPACITY_DP_8_POINTS:-${PAP_CAPACITY_POINTS}}"
export PAP_CAPACITY_PD_6P2D_POINTS="${PAP_CAPACITY_PD_6P2D_POINTS:-${PAP_CAPACITY_POINTS}}"
export PAP_CAPACITY_PD_4P4D_POINTS="${PAP_CAPACITY_PD_4P4D_POINTS:-${PAP_CAPACITY_POINTS}}"
export PAP_CAPACITY_PD_7P1D_POINTS="${PAP_CAPACITY_PD_7P1D_POINTS:-${PAP_CAPACITY_POINTS}}"
export PAP_CAPACITY_PAP_6PA2P_POINTS="${PAP_CAPACITY_PAP_6PA2P_POINTS:-${PAP_CAPACITY_POINTS}}"
export PAP_CAPACITY_PAP_7PA1P_POINTS="${PAP_CAPACITY_PAP_7PA1P_POINTS:-${PAP_CAPACITY_POINTS}}"
export PAP_CAPACITY_DOCUMENT_TOKENS_MEAN="${PAP_CAPACITY_DOCUMENT_TOKENS_MEAN:-4096}"
export PAP_CAPACITY_DOCUMENT_TOKENS_MEDIAN="${PAP_CAPACITY_DOCUMENT_TOKENS_MEDIAN:-4000}"
export PAP_CAPACITY_DOCUMENT_TOKENS_MIN="${PAP_CAPACITY_DOCUMENT_TOKENS_MIN:-2048}"
export PAP_CAPACITY_DOCUMENT_TOKENS_MAX="${PAP_CAPACITY_DOCUMENT_TOKENS_MAX:-5632}"
export PAP_CAPACITY_APPEND_TOKENS_MEAN="${PAP_CAPACITY_APPEND_TOKENS_MEAN:-1100}"
export PAP_CAPACITY_APPEND_TOKENS_MEDIAN="${PAP_CAPACITY_APPEND_TOKENS_MEDIAN:-400}"
export PAP_CAPACITY_APPEND_TOKENS_MIN="${PAP_CAPACITY_APPEND_TOKENS_MIN:-4}"
export PAP_CAPACITY_APPEND_TOKENS_MAX="${PAP_CAPACITY_APPEND_TOKENS_MAX:-2125}"
export PAP_CAPACITY_OUTPUT_TOKENS_MEAN="${PAP_CAPACITY_OUTPUT_TOKENS_MEAN:-${PAP_CAPACITY_OUTPUT_TOKENS:-16}}"
export PAP_CAPACITY_OUTPUT_TOKENS_MEDIAN="${PAP_CAPACITY_OUTPUT_TOKENS_MEDIAN:-$((PAP_CAPACITY_OUTPUT_TOKENS_MEAN * 15 / 16))}"
export PAP_CAPACITY_OUTPUT_TOKENS_MIN="${PAP_CAPACITY_OUTPUT_TOKENS_MIN:-$((PAP_CAPACITY_OUTPUT_TOKENS_MEAN / 2))}"
export PAP_CAPACITY_OUTPUT_TOKENS_MAX="${PAP_CAPACITY_OUTPUT_TOKENS_MAX:-$((PAP_CAPACITY_OUTPUT_TOKENS_MEAN * 2))}"
export PAP_CAPACITY_THINK_TIME_MS="${PAP_CAPACITY_THINK_TIME_MS:-1000}"
export PAP_CAPACITY_TOOL_TIME_MS="${PAP_CAPACITY_TOOL_TIME_MS:-300}"
export PAP_CAPACITY_TOOL_EVERY="${PAP_CAPACITY_TOOL_EVERY:-3}"
export PAP_CAPACITY_RANDOM_SEED="${PAP_CAPACITY_RANDOM_SEED:-42}"
export PAP_CAPACITY_SESSIONS="${PAP_CAPACITY_SESSIONS:-128}"
export PAP_CAPACITY_TURNS="${PAP_CAPACITY_TURNS:-5}"
export PAP_CAPACITY_MODEL_PATH="${PAP_CAPACITY_MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
export PAP_CAPACITY_CORPUS_PATH="${PAP_CAPACITY_CORPUS_PATH:-/home/fei/research/PD/refer_codes/vllm/benchmarks/sonnet_4x.txt}"
export PAP_CAPACITY_SLO_STRICT="${PAP_CAPACITY_SLO_STRICT:-TTFT<=5000ms,ITL<=50ms,good>=0.95}"
export PAP_CAPACITY_SLO_STANDARD="${PAP_CAPACITY_SLO_STANDARD:-TTFT<=10000ms,ITL<=75ms,good>=0.95}"
export PAP_CAPACITY_SLO_RELAXED="${PAP_CAPACITY_SLO_RELAXED:-TTFT<=20000ms,ITL<=100ms,good>=0.95}"

export PAP_CAPACITY_REPETITIONS="${PAP_CAPACITY_REPETITIONS:-1}"
export PAP_CAPACITY_WAIT_FOR_GPUS="${PAP_CAPACITY_WAIT_FOR_GPUS:-1}"
export PAP_CAPACITY_SKIP_RUN="${PAP_CAPACITY_SKIP_RUN:-0}"

export PAP_CAPACITY_RESUME="${PAP_CAPACITY_RESUME:-1}"
export PAP_CAPACITY_GOODPUT_SLO="${PAP_CAPACITY_GOODPUT_SLO:-time_to_first_token:10000 inter_token_latency:75}"

export PAP_CAPACITY_OUTPUT_TOKENS="${PAP_CAPACITY_OUTPUT_TOKENS:-16}"

export PAP_CAPACITY_MATRIX_ROOT="${PAP_CAPACITY_MATRIX_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments/_staging/capacity/${PAP_CAPACITY_MATRIX_ID}}"

echo "Running matrix: ${PAP_CAPACITY_MATRIX_ID}"
echo "Output root: ${PAP_CAPACITY_MATRIX_ROOT}"

if [[ "${PAP_CAPACITY_SKIP_RUN}" == "1" ]]; then
  echo "Scan mode: SKIP existing matrix (recompute only summary)"
  if [[ ! -f "${PAP_CAPACITY_MATRIX_ROOT}/matrix_config.env" ]]; then
    echo "ERROR: skip-run requested but matrix_config.env is missing in ${PAP_CAPACITY_MATRIX_ROOT}" >&2
    exit 2
  fi
  if [[ "${PAP_CAPACITY_SKIP_MISMATCH:-1}" == "0" ]]; then
    echo "Skip-run compatibility check disabled (PAP_CAPACITY_SKIP_MISMATCH=0)"
  else
    "${ROOT_DIR}/.venv/bin/python" - <<'PY'
import os
import json
from pathlib import Path


def unescape_env(value: str) -> str:
    return value.replace("\\", "")


def canonical_csv(value: str) -> str:
    values = sorted({token.strip() for token in value.split(",") if token.strip()})
    return ",".join(values)


def check_csv(name: str, got: str, want: str | None) -> None:
    if not want:
        return
    if canonical_csv(got) != canonical_csv(want):
        raise SystemExit(
            f"{name} mismatch: matrix {got}, requested {want}. "
            "Set PAP_CAPACITY_SKIP_RUN=0 or align requested values."
        )


def check_scalar(name: str, got: str, want: str | None) -> None:
    if want is None or want == "":
        return
    if got != want:
        raise SystemExit(f"{name} mismatch: matrix {got}, requested {want}")


root = Path(os.environ["PAP_CAPACITY_MATRIX_ROOT"])
config_path = root / "matrix_config.env"
results_path = root / "capacity_results.json"
rows = []
if results_path.exists():
    results_payload = json.loads(results_path.read_text(encoding="utf-8"))
    if isinstance(results_payload, dict):
        rows = results_payload.get("rows", [])

requested_arch = os.environ["PAP_CAPACITY_ARCHITECTURES"]
requested_points = {
    "dp_8": os.environ.get("PAP_CAPACITY_DP_8_POINTS"),
    "pd_4p4d": os.environ.get("PAP_CAPACITY_PD_4P4D_POINTS"),
    "pd_6p2d": os.environ.get("PAP_CAPACITY_PD_6P2D_POINTS"),
    "pd_7p1d": os.environ.get("PAP_CAPACITY_PD_7P1D_POINTS"),
    "pap_6pa2p": os.environ.get("PAP_CAPACITY_PAP_6PA2P_POINTS"),
    "pap_7pa1p": os.environ.get("PAP_CAPACITY_PAP_7PA1P_POINTS"),
}
raw = {}
for line in config_path.read_text(encoding="utf-8").splitlines():
    if "=" not in line:
        continue
    key, value = line.split("=", 1)
    raw[key] = unescape_env(value.strip('"'))

matrix_arch = raw.get("ARCHITECTURES", "")
if rows:
    if canonical_csv(matrix_arch) != canonical_csv(requested_arch):
        print(
            f"matrix_config ARCHITECTURES mismatch: matrix has {matrix_arch}, "
            f"requested {requested_arch}. Falling back to capacity_results.json."
        )
else:
    if canonical_csv(matrix_arch) != canonical_csv(requested_arch):
        raise SystemExit(
            f"architecture mismatch: matrix has {matrix_arch}, requested {requested_arch}"
        )


def requested_topology(requested: str) -> str:
    if requested.startswith("dp_"):
        return f"{requested.split('_', 1)[1]}dp"
    return requested.split('_', 1)[1]


def request_arch(requested: str) -> str:
    return requested.split("_", 1)[0]

for key in requested_arch.split(","):
    architecture = key.strip()
    if not architecture:
        continue
    want = requested_points.get(architecture, raw.get("DEFAULT_POINTS"))
    got = raw.get(f"POINTS_{architecture.upper()}", raw.get("DEFAULT_POINTS", ""))
    if not got and rows:
        topo = requested_topology(architecture)
        row_points = {
            str(row.get("concurrency"))
            for row in rows
            if str(row.get("topology", "")) == topo
            and row.get("concurrency") is not None
        }
        if row_points:
            got = ",".join(sorted(row_points, key=lambda item: int(item)))
    check_csv(f"point mismatch for {architecture}", got, want)

scalar_signature = [
    ("MODEL_PATH", os.environ["PAP_CAPACITY_MODEL_PATH"]),
    ("CORPUS_PATH", os.environ["PAP_CAPACITY_CORPUS_PATH"]),
    ("TOTAL_SESSIONS", os.environ["PAP_CAPACITY_SESSIONS"]),
    ("TURNS", os.environ["PAP_CAPACITY_TURNS"]),
    ("RANDOM_SEED", os.environ["PAP_CAPACITY_RANDOM_SEED"]),
    ("DOCUMENT_TOKENS_MEAN", os.environ["PAP_CAPACITY_DOCUMENT_TOKENS_MEAN"]),
    ("DOCUMENT_TOKENS_MEDIAN", os.environ["PAP_CAPACITY_DOCUMENT_TOKENS_MEDIAN"]),
    ("DOCUMENT_TOKENS_MIN", os.environ["PAP_CAPACITY_DOCUMENT_TOKENS_MIN"]),
    ("DOCUMENT_TOKENS_MAX", os.environ["PAP_CAPACITY_DOCUMENT_TOKENS_MAX"]),
    ("APPEND_TOKENS_MEAN", os.environ["PAP_CAPACITY_APPEND_TOKENS_MEAN"]),
    ("APPEND_TOKENS_MEDIAN", os.environ["PAP_CAPACITY_APPEND_TOKENS_MEDIAN"]),
    ("APPEND_TOKENS_MIN", os.environ["PAP_CAPACITY_APPEND_TOKENS_MIN"]),
    ("APPEND_TOKENS_MAX", os.environ["PAP_CAPACITY_APPEND_TOKENS_MAX"]),
    ("OUTPUT_TOKENS_MEAN", os.environ["PAP_CAPACITY_OUTPUT_TOKENS_MEAN"]),
    ("OUTPUT_TOKENS_MEDIAN", os.environ["PAP_CAPACITY_OUTPUT_TOKENS_MEDIAN"]),
    ("OUTPUT_TOKENS_MIN", os.environ["PAP_CAPACITY_OUTPUT_TOKENS_MIN"]),
    ("OUTPUT_TOKENS_MAX", os.environ["PAP_CAPACITY_OUTPUT_TOKENS_MAX"]),
    ("THINK_TIME_MS", os.environ["PAP_CAPACITY_THINK_TIME_MS"]),
    ("TOOL_TIME_MS", os.environ["PAP_CAPACITY_TOOL_TIME_MS"]),
    ("TOOL_EVERY", os.environ["PAP_CAPACITY_TOOL_EVERY"]),
    ("SLO_STRICT", os.environ["PAP_CAPACITY_SLO_STRICT"]),
    ("SLO_STANDARD", os.environ["PAP_CAPACITY_SLO_STANDARD"]),
    ("SLO_RELAXED", os.environ["PAP_CAPACITY_SLO_RELAXED"]),
    ("AIPERF_GOODPUT_SLO", os.environ["PAP_CAPACITY_GOODPUT_SLO"]),
]
for key, value in scalar_signature:
    check_scalar(f"{key} mismatch", raw.get(key, ""), value)

print(
    "Compatibility check for reusable matrix passed: "
    f"{os.environ['PAP_CAPACITY_MATRIX_ROOT']}"
)
PY
  fi
else
  echo "Scan mode: execute matrix for architectures=${PAP_CAPACITY_ARCHITECTURES}"
  echo "Concurrency sweep plan by topology:"
  IFS=, read -r -a __archs <<< "${PAP_CAPACITY_ARCHITECTURES}"
  for __a in "${__archs[@]}"; do
    if [[ -n "$(printf '%q' "$(eval echo \${PAP_CAPACITY_${__a^^}_POINTS})")" ]]; then
      echo "  - ${__a}: $(eval echo \${PAP_CAPACITY_${__a^^}_POINTS})"
    else
      echo "  - ${__a}: ${PAP_CAPACITY_POINTS:-16,24,32,48} (default)"
    fi
  done
  unset __archs __a
fi

if [[ "${PAP_CAPACITY_SKIP_RUN}" == "1" ]]; then
  if [[ ! -f "${PAP_CAPACITY_MATRIX_ROOT}/capacity_results.json" ]]; then
    echo "ERROR: PAP_CAPACITY_SKIP_RUN=1 but capacity results missing in ${PAP_CAPACITY_MATRIX_ROOT}" >&2
    exit 2
  fi
  echo "Skipping runner; using existing matrix ${PAP_CAPACITY_MATRIX_ROOT}"
else
  "${MATRIX_RUNNER}"
fi

if [[ ! -f "${SUMMARY_SCRIPT}" ]]; then
  echo "ERROR: summary script missing: ${SUMMARY_SCRIPT}" >&2
  exit 2
fi

if ! "${ROOT_DIR}/.venv/bin/python" -m py_compile "${SUMMARY_SCRIPT}"; then
  echo "ERROR: summary script is not valid python" >&2
  exit 2
fi

if [[ ! -f "${COMPARE_SCRIPT}" ]]; then
  echo "ERROR: compare script missing: ${COMPARE_SCRIPT}" >&2
  exit 2
fi

if ! "${ROOT_DIR}/.venv/bin/python" -m py_compile "${COMPARE_SCRIPT}"; then
  echo "ERROR: compare script is not valid python" >&2
  exit 2
fi

"${ROOT_DIR}/.venv/bin/python" "${SUMMARY_SCRIPT}" "${PAP_CAPACITY_MATRIX_ROOT}"
"${ROOT_DIR}/.venv/bin/python" "${COMPARE_SCRIPT}" "${PAP_CAPACITY_MATRIX_ROOT}"

echo "SLO summary:"
cat <<'PY' | "${ROOT_DIR}/.venv/bin/python"
import os
import json
from pathlib import Path

matrix_root = Path(os.environ["PAP_CAPACITY_MATRIX_ROOT"])
envelope = json.loads((matrix_root / "capacity_envelope.json").read_text(encoding="utf-8"))


def _fmt_gput(value):
    return "-" if value is None else value

for tier in ("strict", "standard", "relaxed"):
  item = envelope["compliant_goodput_by_slo"][tier]
  pap = item["best_pap"]
  pd = item["best_pd"]
  dp = item["best_dp"]
  def _fmt_pct(value):
    return "-" if value is None else f"{value:.2f}%"
  print(f"\n{tier.upper()}:")
  print(
      f"  PAP: topology={pap['topology'] or '-'} C={pap['concurrency'] or '-'} "
      f"goodput={_fmt_gput(pap['requests_per_second'])}"
  )
  print(
      f"  PD:  topology={pd['topology'] or '-'} C={pd['concurrency'] or '-'} "
      f"goodput={_fmt_gput(pd['requests_per_second'])}"
  )
  print(
      f"  DP:  topology={dp['topology'] or '-'} C={dp['concurrency'] or '-'} "
      f"goodput={_fmt_gput(dp['requests_per_second'])}"
  )
  print(f"  PAP over PD: {_fmt_pct(item['pap_over_pd_percent'])}")
  print(f"  PAP over DP: {_fmt_pct(item['pap_over_dp_percent'])}")
PY


echo ""
echo "Re-run command (summary for another matrix):"
echo "PAP_CAPACITY_MATRIX_ROOT=${PAP_CAPACITY_MATRIX_ROOT}"
echo "PAP_CAPACITY_ARCHITECTURES=${PAP_CAPACITY_ARCHITECTURES}"
echo "PAP_CAPACITY_SKIP_RUN=1 PAP_CAPACITY_SKIP_MISMATCH=0 PAP_CAPACITY_WAIT_FOR_GPUS=0 bash ${ROOT_DIR}/benchmarks/pap/aiperf/run_three_way_slo_capacity.sh"
