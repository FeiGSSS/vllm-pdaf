# PAP benchmark datasets

This directory is the immutable workload registry. Dataset directories use a
semantic ID rather than an experiment date. Once referenced by a benchmark,
the bytes at that ID must not change; a changed workload receives a new ID.

Every dataset records its AIPerf schema and SHA-256 in a local manifest. The
repository-wide checksums are in `SHA256SUMS` and can be verified with:

```bash
cd benchmarks/pap/datasets
sha256sum --check SHA256SUMS
```

## Active workloads

| Dataset ID | AIPerf format | Purpose |
| --- | --- | --- |
| `agentic-code/full-131k-osl10k-no-delay-seed42` | `mooncake-trace` | Fixed-duration, long-running serving experiments |
| `agentic-code/s60-t3-seed42` | `mooncake-trace` | Three-turn, 60-session control workload |
| `agentic-code/s60-t3-half-seed42` | `mooncake-trace` | Short QPS scans: 60 sessions, three turns, halved lengths |
| `agentic-code/s60-t3-native32k-stratified-seed42` | `mooncake-trace` | Native-32K QPS scans: 60 sessions, three turns, mean output below 100 tokens, and 15 final contexts in each 4K stratum from 16K through 32K |
| `long-context/qwen3-8b-yarn131k-shared-prefix` | `multi-turn` | Qwen3 YaRN 131K validation; cross-session prefix sharing |

New workloads allow identical prefixes to be reused across conversations by
default: the generator does not inject a per-session `cache_salt`. Conversation
IDs still define turn ordering, not KV isolation. Explicit caller-requested salt
is never silently removed; PAP's Dynamo integration rejects that unsupported mode.

The historical `long-context/qwen3-8b-yarn131k` fixtures remain immutable,
session-isolated controls, not the default Dynamo validation workload. The new
`-shared-prefix` version removes only `turn.extra.cache_salt`; its manifest records
source hashes. All text, turn order, delays and output lengths are unchanged.

Generation reports and configuration belong beside their source dataset as
provenance. Runtime logs and benchmark results never belong here.

Deterministic dataset construction and filtering utilities live in `tools/`.
