# PAP Runner 3-Way Microbatch Pipeline

Updated: 2026-05-26

## Purpose

This note records the runner-level 3-way microbatch experiment for PAP 4PA4P on
Qwen3-8B. The goal was to split one Projection decode batch into three smaller
ubatches so Projection can send QKV for one ubatch, let Attention compute another
ubatch, and continue Projection work on a third ubatch instead of forcing a fully
serial Projection/Attention alternation at every layer.

## Implementation Summary

The final implementation is runner-level, not scheduler-level:

- `PAP_RUNNER_MICROBATCH_COUNT` enables PAP-specific ubatching on Projection
  workers without enabling vLLM's generic DBO CLI flags.
- `PAP_RUNNER_MICROBATCH_DECODE_THRESHOLD` controls when decode batches are large
  enough to split. The default is `12` because splitting very small batches is a
  net regression.
- `ModelRunner` creates `UBatchSlices`, slices attention metadata, slot mappings,
  and PAP forward-context request metadata, then executes the model through
  `UBatchWrapper`.
- Qwen3 PAP attention yields after sending QKV when ubatching is active, allowing
  the next ubatch thread to run before the current ubatch waits for the Attention
  output.
- V2 attention metadata builder reuse is handled for the PAP eager path: V2
  normally has one metadata builder, so ubatches reuse builder 0 when per-ubatch
  builders are not available.

The earlier layer-internal microbatch attempt remained opt-in and was not the
winning path. It split QKV and output projection inside Qwen3 attention, but the
extra Python/threading and per-chunk work outweighed overlap.

## Common Benchmark Environment

All runs below used:

```bash
MODEL_PATH=/data/ssd1/llm-models/Qwen3-8B
BENCH_NUM_WARMUPS=0
BENCH_TIMEOUT=900
SERVER_START_TIMEOUT=900
CLUSTER_READY_WAIT_SECONDS=15
PAP_MODE=pap
PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox
PAP_DIRECT_MAILBOX_OUTPUT=1
PAP_OFFLOAD_EXEC_MICROBATCH_COUNT=0
PAP_NIXL_MAILBOX_SLOT_COUNT=8
PAP_NIXL_MAILBOX_RECV_SLOT_COUNT=8
PAP_NIXL_MAILBOX_ASYNC_SEND_SLOTS=4
PAP_Q_FIRST_KV_LATER=0
PAP_Q_FIRST_PROJECTION=0
PAP_ATTENTION_Q_FIRST_PARTIAL=0
PAP_PREFILL_MPS_PERCENT=30
PAP_ATTENTION_MPS_PERCENT=70
PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.60
PAP_PROJECTION_GPU_MEMORY_UTILIZATION=0.80
```

Benchmark command shape:

```bash
bash /home/fei/research/PD/test/baseline/run_benchmark.sh \
  --mode pap \
  --topology 4pa4p \
  --input-lens 1024 \
  --output-lens 16 \
  --qps 80 \
  --num-prompts 64 \
  --model /data/ssd1/llm-models/Qwen3-8B \
  --proxy-port 9000
```

## Results

| Config | Run directory | Successful | Failed | Mean TTFT | Mean TPOT | P99 TPOT | Total tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Serial 4PA4P, current code | `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_135807` | 64 | 0 | 5214.05 ms | 69.82 ms | 82.62 ms | 7142.84 |
| 3-way, decode threshold 12 | `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_135517` | 64 | 0 | 5189.36 ms | 63.81 ms | 82.00 ms | 7165.59 |
| 3-way, decode threshold 8 | `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_135644` | 64 | 0 | 5196.41 ms | 66.06 ms | 81.37 ms | 7183.55 |
| 3-way, decode threshold 1 | `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_135339` | 64 | 0 | 5208.48 ms | 75.86 ms | 93.48 ms | 6988.29 |

Compared with the same-code serial run, the chosen `threshold=12` configuration
improves mean TPOT from `69.82 ms` to `63.81 ms`, an `8.6%` reduction. Total
throughput moves from `7142.84 tok/s` to `7165.59 tok/s`, a `0.32%` increase.

## Interpretation

The 3-way design works only when the decode batch is large enough. With
`threshold=1`, the first Projection shard can split a 3-token decode batch into
three 1-token ubatches, and the synchronization cost dominates the overlap. With
`threshold=12`, the split happens later, when each ubatch has enough work to
amortize the thread/context overhead.

For this workload, TPOT improves meaningfully while total throughput is nearly
flat. The total-token throughput includes the large prompt-token denominator and
is less sensitive than TPOT to decode-stage overlap. The most useful signal for
this change is therefore mean TPOT and ITL, not aggregate token throughput.

## Verification

Fresh verification before recording this note:

```bash
pre-commit run ruff-format --files tests/pap/test_pap_contract.py tests/pap/test_pap_launch_files.py vllm/v1/worker/gpu/model_runner.py
pre-commit run ruff-check --files tests/pap/test_pap_contract.py tests/pap/test_pap_launch_files.py vllm/v1/worker/gpu/model_runner.py
.venv/bin/python -m py_compile tests/pap/test_pap_contract.py tests/pap/test_pap_launch_files.py vllm/v1/worker/gpu/model_runner.py vllm/v1/worker/gpu_ubatch_wrapper.py vllm/v1/worker/gpu/attn_utils.py
.venv/bin/python -m pytest tests/pap/test_pap_contract.py tests/pap/test_pap_launch_files.py -q
bash -n examples/pap/launch_pap_nixl.sh /home/fei/research/PD/test/baseline/pap/launch_service.sh
```

Observed results: ruff format passed, ruff check passed, py_compile passed,
`44 passed, 16 warnings`, and launcher syntax passed.

## Follow-Up

- Re-run the threshold sweep with more prompts to reduce variance and include
  `threshold=10`, `12`, `16`, and `24`.
- Add timeline/trace instrumentation for ubatch yields to prove the intended
  Projection-send / Attention-compute / Projection-compute overlap, rather than
  relying only on endpoint metrics.
- Avoid using `PAP_RUNNER_MICROBATCH_DECODE_THRESHOLD=1` as a default; it is a
  measured regression on this workload.
