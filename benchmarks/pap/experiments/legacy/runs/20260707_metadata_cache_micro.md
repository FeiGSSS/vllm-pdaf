# PAP Metadata Cache Microbenchmark - 2026-07-07

## Scope

This is an operation-level validation for Rank 1 in
`docs/design/pap-nixl-nvlink-optimization-idea-book-20260707.md`.

It tests repeated unified paged FlashAttention metadata construction for the
same decode batch signature. It does not replace a full warmed NIXL `1pa1p`
serving benchmark.

## Command Shape

Environment:

- Python: `.venv/bin/python`
- Device: CPU microbenchmark, because this validates Python-side repeated
  metadata construction and cache hit behavior
- Batch size: `64`
- Blocks per request: `9`
- Repeated builds: `360`

Measured loop:

```text
build_unified_paged_flash_metadata(states=states, device=torch.device("cpu"))
```

## Result

| Mode | Total for 360 builds | Per build |
|---|---:|---:|
| Cache disabled with `PAP_UNIFIED_MD_CACHE_LIMIT=0` | `513.235 ms` | `1.425653 ms` |
| Cache enabled with `PAP_UNIFIED_MD_CACHE_LIMIT=256` | `19.347 ms` | `0.053741 ms` |

Cache counters after the enabled run:

```text
hits=359 misses=1 entries=1
```

## Interpretation

The operation-level repeated-build cost drops by about `26.5x` for an identical
decode signature. This targets the traced attention-side
`metadata_build_ms = 1.84 ms/layer` overhead. The expected full-service win
depends on how often the NIXL run reuses the same request order, block IDs, and
sequence lengths across the 36 decoder layers.

## Next Validation

Run a warmed NIXL `1pa1p` benchmark on the same Qwen3-8B `i128/o32/q16/c64`
workload and compare:

- `metadata_build_ms`
- per-layer `remote_total_ms`
- median/p99 TPOT

Keep the cache if the operation-level reduction is visible in the service trace
and does not regress correctness.
