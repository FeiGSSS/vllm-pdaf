# Qwen3-8B 16K two-turn north-star references

This directory is the Git-tracked control plane for the fixed 1P1D PD versus
1PA1P PAP multi-turn comparison. Raw service logs and metrics remain under
`test/baseline/pap/results/runs/`.

The frozen workload is described by `profile.json`. Each formal reference is
the cross-run median of three serial repetitions with a full service restart.
The primary optimization metric is round-two TPOT; TTFT and round-one metrics
remain independent regression signals.

## PD reference

`pd_reference.json` was created from:

```text
test/baseline/pap/results/runs/
  20260712_031855_d341f7e3e_pd_multiturn_formal/
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
| 1 | 8250.232 | 25.083 |
| 2 | 269.013 | 25.163 |

## Reference policy

Daily PAP runs never update these files. Refresh PD with
`bootstrap_pd_multiturn_reference.sh`, then use the comparison CLI's explicit
`write-reference --allow-reference-write` operation. Promote a PAP reference
only from a valid three-repetition formal aggregate. Any profile fingerprint,
hardware, or effective cache-semantics change requires a new reference rather
than an in-place mixed comparison.
