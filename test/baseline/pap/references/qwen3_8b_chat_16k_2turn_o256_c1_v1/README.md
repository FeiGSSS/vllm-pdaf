# Qwen3-8B 16K two-turn north-star references

This directory is the Git-tracked control plane for the fixed 1P1D PD versus
1PA1P PAP multi-turn comparison. Raw service logs and metrics remain under
`test/baseline/pap/results/runs/`.

The frozen workload is described by `profile.json`. Each formal reference is
the cross-run median of three serial repetitions with a full service restart.
The primary optimization metric is round-two TPOT; TTFT and round-one metrics
remain independent regression signals.

The tracked references use `last_output_token_v2`: TPOT ends at the final
output token, while HTTP EOF cleanup is reported separately. The superseded
`http_stream_eof_v1` results remain in the raw run archive and design history;
schema-v2 comparison rejects them.

## PD reference

`pd_reference.json` was created from:

```text
test/baseline/pap/results/runs/
  20260712_161402_7e81e2d10_pd_multiturn_formal/
```

It uses the unchanged official multi-turn proxy and the default one-way NIXL
producer/consumer path. The current streaming Chat API cannot return a
Decode-side KV handle, so both proxy lookups are expected to miss. Every
repetition proves the effective reuse path from exact token-source counters:

- Prefill: `local_compute=16420`, `local_cache_hit=16016`, external `0`;
- Decode: local cache `16272`, external NIXL `16164`, local compute `0`;
- both nodes conserve the two-round total of `32436` prompt tokens;
- Decode local cache includes all `256` Decode-derived tokens in the true LCP.

The formal medians are:

| Round | TTFT (ms) | TPOT (ms) |
| --- | ---: | ---: |
| 1 | 8483.474 | 25.101 |
| 2 | 267.273 | 25.183 |

## PAP local-fast reference

`pap_reference.json` was created from:

```text
test/baseline/pap/results/runs/
  20260712_201947_0727ed946_pap_multiturn_formal/
```

All three repetitions hit the exact `16272`-token second-turn cache boundary,
computed only `146` new prompt tokens, drained every Attention session, and
passed the strict service-log audit. The formal medians are:

| Round | TTFT (ms) | TPOT (ms) | PAP/PD TTFT | PAP/PD TPOT |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5460.711 | 30.196 | 0.644x | 1.203x |
| 2 | 224.491 | 30.449 | 0.840x | 1.209x |

The round-two north-star boundary is `< 50.366 ms/token`; PAP is
`19.917 ms/token` below it. The remaining absolute PAP/PD gap is
`5.266 ms/token`. Round-two TPOT was `30.449 / 30.385 / 30.474 ms` across
the three repetitions; the output signatures and exact `16272`-token hit were
identical in all three.

The first controlled quick A/B at
`results/runs/20260712_north_star_local_fast_quick` changed the PAP
OFFLOAD_EXEC bundle from NIXL mailbox/direct-output-off to same-node
`local_fast`/direct-output-on. It reduced
round-two TTFT/TPOT from `278.483/55.967 ms` to `246.587/38.603 ms`, preserved
the exact `16272`-token cache hit and PAP output digests, and passed lifecycle
gates. The runner now fixes `local_fast`; promotion still requires a clean
three-repetition formal result. The v2 formal above completes that promotion.

The original v2 local-fast control at commit `7e81e2d10` remains archived at
`results/runs/20260712_162130_7e81e2d10_pap_multiturn_formal`. It measured
round-two TTFT/TPOT `235.388/39.128 ms`. Stage A commit `6bc383dab` replaced
per-element CUDA writes on paged-FA metadata misses with bulk construction,
preserving key/LRU/padding semantics. Its clean formal result remains archived
at `results/runs/20260712_171755_6bc383dab_pap_multiturn_formal`; it improved
round-two TTFT by `7.28%` and TPOT by `21.83%` over the original v2 control.

Stage B commit `c134bc3d9` made slot-plan keys generation-aware. Legal
`4096 -> 8192 -> 12288 -> 16018` chunked-Prefill growth now completes four
separate activations, while same-activation topology conflicts remain
fail-closed. Every formal repetition changed the exact slot-plan counters from
`hits/misses/mismatch=8925/255/1` to `17850/510/0`, with zero fallback. Relative
to Stage A, round-one TPOT improved `14.25%`, round-two TPOT changed only
`+0.64%`, and two-turn conversation latency improved `5.91%` to `0.985x` PD.
The comparator classified this as neutral because its primary round-two TPOT
change stayed inside the 3% noise band; it is promoted as the clean baseline
for that accepted implementation, not claimed as a round-two win.

Stage C commit `0727ed946` replaces metadata cache-hit block-table scans with
process-unique topology tokens. Unknown or mixed state still falls back to the
exact full key. It also serializes the process-global metadata LRU to remove a
real peer-thread `get -> move_to_end` eviction race, while keeping tensor
construction outside the lock. In the alternating three-pair controlled A/B,
round-two TPOT improved `1.39%` and conversation latency improved `1.03%`;
all three paired TPOT deltas were between `-1.33%` and `-1.41%`.

Each clean formal repetition recorded `17920` fast-key hits and only `512`
full scans. Block IDs examined fell from `18994176` in the disabled control to
`527616`, exactly `36x` fewer, without changing the `17920/512` metadata
hit/miss split. Relative to Stage B formal, round-one/round-two TPOT improved
`1.06%/1.08%`. This remains below the comparison policy's 3% improvement band,
so it is promoted as the current correct default and reference, not described
as a statistically large performance win.

Exact-token signatures are also tracked. All three repetitions within each
architecture are deterministic, all prompt digests match, and round-one PD/PAP
outputs match exactly. Round-two PAP is deterministic but its output digest
differs from PD. The comparator reports this as a correctness warning rather
than hiding it or invalidating timing; performance promotion requires stable
PAP output, while cross-architecture numerical parity remains a separate item
to investigate.

Schema-v2 results additionally bind every repetition to its Git commit,
transport/direct-output implementation fingerprint, and architecture-specific
external gates. Candidate output-token and assistant-text signatures must
match the PAP reference exactly; a stable PD/PAP numerical-path difference is
reported separately as a warning.

## Reference policy

Daily PAP runs never update these files. Refresh PD with
`bootstrap_pd_multiturn_reference.sh`, then use the comparison CLI's explicit
`write-reference --allow-reference-write` operation. Promote a PAP reference
only from a valid three-repetition formal aggregate. Any profile fingerprint,
hardware, or effective cache-semantics change requires a new reference rather
than an in-place mixed comparison.
