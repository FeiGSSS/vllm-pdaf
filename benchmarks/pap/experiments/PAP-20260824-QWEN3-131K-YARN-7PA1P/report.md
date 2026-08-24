# Qwen3-8B YaRN 131K on PAP 7PA1P

## Decision

The direct PAP runner now defaults to Qwen's official static-YaRN factor-4
configuration and `max_model_len=131072`. Historical frozen comparison scripts
that explicitly set 32K remain unchanged.

## Configuration

```json
{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}
```

The current vLLM branch receives this through `--hf-overrides`; its removed
legacy `--rope-scaling` interface is not used.

## Workload

- topology: 7PA1P on eight NVIDIA L20 GPUs;
- seven conversations at concurrency seven, one conversation per PA;
- two turns per conversation;
- measured input lengths: 125,018 and 129,058 tokens;
- output: 16 tokens per request;
- total requests: 14;
- dataset SHA-256:
  `4a01e7a0d1e57f2cf92efeee02913dc02e67b120b15f46f44dce9758909bf28c`.

## Results

- 14/14 requests completed with zero errors;
- all seven PA routes were used;
- all seven Attention whole-step 36-layer CUDA Graphs captured;
- Prefill CUDA Graph, Projection outer Graph, routing, decode-token join,
  gateway drain, and session drain audits passed;
- mean TTFT: 40,210.54 ms;
- first-turn 125K TTFT range: approximately 75.0--75.9 seconds;
- second-turn 129K TTFT minimum: 4,871.16 ms;
- mean inter-token latency: 63.05 ms;
- maximum inter-token latency across requests: 66.15 ms;
- benchmark duration: 82.88 seconds.

The first launch failed before serving because Projection had 17.91 GiB of
available validation KV capacity while 18.00 GiB was required. Adding a fixed
512 MiB runtime headroom to the Projection memory planner raised its planned
utilization from 0.8071 to 0.8182 and resolved the startup boundary. Prefill
workers reported 159,888--163,760 KV tokens of capacity, or 1.22--1.25
concurrent 131,072-token requests per PA.

## Artifacts

- passing run: `runs/yarn131k_7pa1p_7s2t_full`;
- single-conversation functional run: `runs/yarn131k_7pa1p_1s2t_r2`;
- initial Projection-capacity failure: `runs/yarn131k_7pa1p_1s2t_smoke`;
- workload: `artifacts/longctx_7s2t.jsonl`;
- model-history note: `history/model-pr-history-notes.md`.

## Default short-context regression

After changing the direct PAP defaults, a second 7PA1P run omitted both
`MAX_MODEL_LEN` and `PAP_HF_OVERRIDES`. Its effective configuration recorded
`max_model_len=131072` and the factor-4 YaRN parameters automatically.

- workload: seven conversations, concurrency seven, two turns each;
- input lengths: 8,210 and 8,762 tokens;
- 14/14 requests completed with zero errors;
- every correctness, routing, Graph, decode-token, gateway-drain, and
  session-drain audit passed;
- mean TTFT: 1,002.89 ms;
- mean inter-token latency: 27.61 ms;
- request throughput: 4.76 requests/s;
- artifact: `runs/yarn131k_default_short_7s2t`.
