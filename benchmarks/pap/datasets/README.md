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
| `long-context/qwen3-8b-yarn131k` | `multi-turn` | Focused Qwen3 YaRN 131K validation fixtures |

Generation reports and configuration belong beside their source dataset as
provenance. Runtime logs and benchmark results never belong here.
