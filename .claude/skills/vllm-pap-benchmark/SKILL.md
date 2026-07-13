---
name: vllm-pap-benchmark
description: Run and compare vLLM PAP, PAP-local, PAP-NIXL, and PD/NIXL serving benchmarks in this repository. Use when benchmarking PAP performance, rerunning the PD baseline, tracing PAP overhead, comparing PD vs PAP, or choosing a standard workload for PAP experiments.
---

# vLLM PAP Benchmark

Use this skill for PAP/PD benchmark and tracing work in
`/home/fei/research/PD/vllm-pap`. Keep benchmark parameters explicit and do
not choose a new workload unless the user asks for one.

## Canonical Comparison Workload

Default to the existing PD baseline workload:

- Model: `/data/ssd1/llm-models/Qwen3-8B`
- Dataset: `sonnet`
- Dataset path: `/home/fei/research/PD/refer_codes/vllm/benchmarks/sonnet_4x.txt`
- Input length: `128`
- Output length: `32`
- Prefix length: `50`
- Request rate: `16`
- Number of prompts: `128`
- Warmups: `0`
- Max model length: `512`
- Max sequences: `64`

Canonical PD baseline result:

```text
/home/fei/research/PD/test/baseline/nixl_disaggregated/results/runs/20260701_171300/1P1D_i128_o32_q16.json
```

PD baseline metrics from that run:

- Successful/failed: `128/0`
- Duration: `8.98 s`
- Request throughput: `14.26 req/s`
- Output throughput: `456.22 tok/s`
- Total token throughput: `2198.47 tok/s`
- Mean/median/p99 TTFT: `246.82 / 212.53 / 470.69 ms`
- Mean/median/p99 TPOT: `24.69 / 24.28 / 26.27 ms`
- Mean/p99 ITL: `24.69 / 26.97 ms`
- Peak concurrent requests: `42`

## Required Environment Rules

Always run from the repository root:

```bash
cd /home/fei/research/PD/vllm-pap
```

Use the repo virtualenv tools. Do not use system `python3` or bare `pip`.

```text
VLLM_BIN=/home/fei/research/PD/vllm-pap/.venv/bin/vllm
PYTHON_BIN=/home/fei/research/PD/vllm-pap/.venv/bin/python
```

For PAP benchmark runs, always disable FlashInfer sampler until the sampler JIT
failure is fixed:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0
```

Without this, PAP startup can fail with a FlashInfer/CUB compile error like:

```text
flashinfer/sampling.cuh ... BlockAdjacentDifference ... has no member "FlagHeads"
```

Benchmark clients use environment proxy settings. If `HTTP_PROXY` or
`HTTPS_PROXY` is set, ensure local requests bypass it:

```text
NO_PROXY=127.0.0.1,localhost
no_proxy=127.0.0.1,localhost
```

The bundled PAP runner appends these entries automatically and records them in
`effective_config.env`.

## PAP NIXL Mailbox Run

Use only bundled scripts from this skill for runnable shell workflows. Do not
call existing project benchmark shell scripts from this skill; the benchmark
environment must be self-contained here.

Run the default PAP-vs-PD comparison workload with:

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh
```

The same self-contained runner supports arbitrary positive `xPAyP` TP1
topologies. Set one GPU per PA group and one GPU per Projection instance:

```bash
PAP_TOPOLOGY=3pa2p \
PAP_PREFILL_GPUS=1,2,3 \
PAP_PROJECTION_GPUS=4,5 \
bash .claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh
```

The default routing policy is `round_robin`, which exercises the lazy
Projection-to-Attention crossbar and uses every configured Projection.
`projection_affinity` remains available for static pairing.

For a canonical baseline, require a clean tracked worktree:

```bash
PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE=1 \
bash .claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh
```

The runner contains the full service startup, benchmark command, environment
defaults, FlashInfer sampler workaround, run metadata writing, and cleanup
logic. It may call repository Python entrypoints and `.venv/bin/vllm` because
those are the implementation under test, but it must not delegate to another
project shell script.

The runner records the full Git commit, tracked dirty state, `git_status.txt`,
`tracked_worktree.patch`, and `topology_manifest.json`. It also fails closed by
default when the client result is incomplete or service logs contain
decode-commit, lease-release, or unified-KV consistency errors. For x:y runs,
`routing_audit.json` verifies the routing policy's expected PA/Projection
distribution plus per-PA request, commit, and release counts. It waits for all
Attention instances' active session counts to reach zero and records
`session_drain.env`, so ACK flush and lease release must finish before services
are stopped. Set
`PAP_BENCH_STRICT_CORRECTNESS_AUDIT=0` only for diagnostic failure capture, not
for a baseline result.

The initial x:y implementation was validated with Qwen3-8B on `1PA2P`,
`2PA1P`, `2PA2P`, and non-divisible `3PA2P`; fixed 70/30 MPS was additionally
validated on `2PA1P`. These are correctness smoke runs with shorter output and
lower QPS, not canonical PD performance comparisons:

```text
/home/fei/research/PD/vllm-pap/test/baseline/pap/results/runs/20260710_xpayp_1pa2p_smoke_v1
/home/fei/research/PD/vllm-pap/test/baseline/pap/results/runs/20260710_xpayp_2pa1p_smoke_v1
/home/fei/research/PD/vllm-pap/test/baseline/pap/results/runs/20260710_xpayp_2pa2p_smoke_v1
/home/fei/research/PD/vllm-pap/test/baseline/pap/results/runs/20260710_xpayp_3pa2p_smoke_v1
/home/fei/research/PD/vllm-pap/test/baseline/pap/results/runs/20260710_xpayp_2pa1p_mps_smoke_v1
```

The correctness-valid clean baseline for `e9044a88c` consists of three runs:

```text
/home/fei/research/PD/vllm-pap/test/baseline/pap/results/runs/20260710_e904_nixl_rep1_clean/1PA1P_i128_o32_q16.json
/home/fei/research/PD/vllm-pap/test/baseline/pap/results/runs/20260710_e904_nixl_rep2_clean/1PA1P_i128_o32_q16.json
/home/fei/research/PD/vllm-pap/test/baseline/pap/results/runs/20260710_e904_nixl_rep3_clean/1PA1P_i128_o32_q16.json
```

All three runs used a clean tracked worktree, completed `128/0`, and passed the
strict correctness audit with zero matches. Across-run medians are:

- Successful/failed: `128/0`
- Duration: `16.58 s`
- Request throughput: `7.72 req/s`
- Output throughput: `247.01 tok/s`
- Total token throughput: `1190.33 tok/s`
- Mean/median/p99 TTFT: `1554.57 / 837.53 / 6993.54 ms`
- Mean/median/p99 TPOT: `79.89 / 82.96 / 92.33 ms`
- Mean/p99 ITL: `79.89 / 112.38 ms`
- Peak concurrent requests: `102`

The ACK-watermark and reliable lease protocol was validated end to end at:

```text
/home/fei/research/PD/vllm-pap/test/baseline/pap/results/runs/20260710_ack_watermark_e2e_drain
```

It completed `128/0`, passed strict correctness and session-drain checks, and
recorded `128` registrations, `128` session DELETEs, and `128` successful lease
releases. Its tracked worktree was dirty, so use it as protocol validation, not
as a replacement performance baseline.

The older `20260707_090030` run used commit `328384e90`, predates
`e9044a88c`, and contains decode-commit failures despite reporting `128/0`.
Treat it as a performance-only historical run.

## PD NIXL Run

Prefer the canonical PD baseline result above for comparison unless the user
asks to rerun PD. A self-contained 1P1D runner and proxy are bundled in this
skill; do not call existing project benchmark shell scripts:

```bash
QPS=8 RUN_ID=my_pd_run \
bash .claude/skills/vllm-pap-benchmark/scripts/run_pd_same_workload.sh
```

The runner defaults to the canonical PD engine configuration and GPUs 1/2.
Override `PD_PREFILL_GPU` and `PD_DECODE_GPU` only after checking occupancy. It
records effective configuration, Git state, completeness, and a correctness
log audit. Its cleanup is scoped to process groups created by that run.

## Fixed Multi-turn North-star

For 1P1D PD versus 1PA1P PAP multi-turn TTFT/TPOT work, use the frozen
`qwen3_8b_chat_16k_2turn_o256_c1_v1` profile documented in
`test/baseline/pap/README.md`. The runner explicitly uses the same-node
`local_fast` CUDA-IPC/P2P ring for PAP OFFLOAD_EXEC. Run PAP candidates with:

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/run_multiturn_north_star.sh quick
bash .claude/skills/vllm-pap-benchmark/scripts/run_multiturn_north_star.sh formal
```

The north-star uses last-output-token TTFT/TPOT timing and records HTTP EOF
cleanup separately. Formal repetitions must share one clean Git commit and
implementation fingerprint, and the aggregate requires embedded cache,
routing, lifecycle, and fatal-log gates.

For the five-turn 16K C4 PAP optimization lane, use
`run_pap_multiturn_load.sh`. On the local R595/L20 host its accepted development
default is asynchronous decode-token delivery without the Projection step
barrier plus static MPS `64/28` SM partitions:

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/run_pap_multiturn_load.sh quick c4
```

Set `PAP_ASYNC_DECODE_TOKEN=0` to restore the synchronous descriptor-token
path, or `PAP_LOAD_MPS_PROFILE=baseline_70_30` to restore dynamic MPS. Static
MPS is host-specific and must pass the recorded visible-SM and cleanup audits.

Refresh PD only through `bootstrap_pd_multiturn_reference.sh`. That script uses
the unchanged official streaming proxy and validates the current effective
semantics from P/D token-source metrics: exact local cache boundaries,
prompt-source conservation, and second-turn P-to-D NIXL transfer. Streaming
chat currently does not return the Decode KV handle, so this lane uses the
default one-way connector mode and proxy-level D-to-P lookup remains a miss;
do not patch PD for this benchmark. Reference writes are always explicit.

## Result Comparison

Compare a PAP result against the canonical PD baseline with:

```bash
.venv/bin/python .claude/skills/vllm-pap-benchmark/scripts/compare_pd_pap_results.py \
  --pap /path/to/1PA1P_i128_o32_q16.json
```

Omit `--pap` to compare against the latest same-workload PAP result recorded in
this skill.

## Before Running

Check and record:

- Current commit: `git rev-parse --short HEAD`
- Worktree state: `git status --short`
- Existing benchmark processes:
  `ps -ef | rg -n "pap|vllm|run_benchmark|benchmark|9460|8100|8200|8300"`
- GPU occupancy if relevant:
  `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader`
- Proxy environment:
  `env | sort | rg -i "proxy|no_proxy"`

If ports or GPUs are occupied by unrelated work, do not kill them without user
approval. Choose new PAP ports only if the user agrees that the result remains
comparable.

## After Running

Verify the generated run directory:

- Read `effective_config.env`.
- Read `run_metadata.json`.
- Read `topology_manifest.json`.
- Read `routing_audit.env` and require `STATUS=passed`.
- Read `correctness_audit.env` and require `STATUS=passed`.
- Read `session_drain.env` and require `STATUS=passed` and `ACTIVE_SESSIONS=0`.
- Confirm `git_tracked_worktree_dirty` is false for canonical baselines.
- Confirm input length, output length, prefix length, qps, prompts, warmups,
  model path, and transport match the intended comparison.
- Confirm `NO_PROXY` and `no_proxy` include `127.0.0.1` and `localhost`.
- Confirm there are no failed requests.

If all benchmark requests fail with `Forbidden`, check whether local requests
were sent through an HTTP proxy. A proxy-caused failure typically has no
`/v1/completions` entries in `service_logs/proxy.log`, only `/health`; rerun
with local `NO_PROXY` entries before investigating PAP internals.

Compare at least these JSON fields:

- `completed`
- `failed`
- `duration`
- `request_throughput`
- `output_throughput`
- `total_token_throughput`
- `mean_ttft_ms`, `median_ttft_ms`, `p99_ttft_ms`
- `mean_tpot_ms`, `median_tpot_ms`, `p99_tpot_ms`
- `mean_itl_ms`, `p99_itl_ms`
- `max_concurrent_requests`

Interpretation rule:

- If PAP has much higher `max_concurrent_requests` than PD, TTFT includes queue
  buildup. Say this explicitly instead of attributing all TTFT regression to a
  single attention or communication operation.
- Do not compare runs with different `NUM_PROMPTS`, `PREFIX_LEN`,
  `BENCH_NUM_WARMUPS`, or `MAX_MODEL_LEN` as same-workload results.

## Known Historical Non-Comparable Run

Do not use this as the same-workload PD comparison:

```text
/home/fei/research/PD/test/baseline/pap/results/runs/20260707_024400/1PA1P_i128_o32_q16.json
```

It used `NUM_PROMPTS=64`, `PREFIX_LEN=0`, `BENCH_NUM_WARMUPS=16`, and
`MAX_MODEL_LEN=256`, so it is only historical context.
