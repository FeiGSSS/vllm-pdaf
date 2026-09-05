# Agentic Coding architecture QPS matrix

This experiment compares six eight-GPU serving architectures with one fixed
AIPerf workload and launch protocol:

- Dynamo DP8;
- Dynamo 2P6D, 4P4D, and 6P2D;
- PAP 7PA1P and 6PA2P with 2K and 32K Prefill token budgets.

The configured request rates are `0.6`, `0.9`, `1.2`, `1.5`, and `1.8` req/s,
all using Poisson arrivals. Every point replays the same 60 conversations and
180 sequential turns with concurrency 60. There is no warmup or timed cutoff;
the point ends after all 180 requests complete. Dynamo DP8 and the three PD
splits use a 32,768-token budget. Both PAP topologies are measured with
2,048-token and 32,768-token Prefill budgets. All architectures use
`max_num_seqs=256`.

Start a new complete matrix with:

```bash
bash benchmarks/pap/experiments/e2e/PAP-20260903-AGENTIC-CODE-QPS-MATRIX/run.sh
```

`experiment.env` is the requested experiment configuration and is sourced only
by this directory's `run.sh`. The colocated `driver.sh` contains this matrix's
orchestration logic.

New runs use `runs/<timestamp>/<architecture>/qps_<rate>/attempt_<number>/`.
The existing `results/` directory is a historical record, not the output of a
new invocation. Each run stores `matrix.env`, a copy of `experiment.env`,
`driver.snapshot.sh`, and `source.patch`. The manifest records the Git commit
and tracked diff checksum, dataset checksum, and launcher checksums. Each
attempt additionally records the topology and process configuration observed
after service startup. `provenance/` adds a source archive (including nonignored
new files), environment package inventories, hardware details and model-file
checksums. The manifest fingerprints untracked files as well as the tracked diff.

To resume, explicitly select the existing run directory:

```bash
PAP_QPS_SCAN_RUN_ROOT=/absolute/path/to/existing/run \
bash benchmarks/pap/experiments/e2e/PAP-20260903-AGENTIC-CODE-QPS-MATRIX/run.sh
```

Resume rejects changed source, experiment configuration, GPU identity or driver,
and refuses to reuse an incomplete environment snapshot. It skips valid
points and gives failed or incomplete points a new attempt number. Set
`PAP_QPS_SCAN_VALIDATE_ONLY=1` to check configuration without writing any files
or launching services. A configuration check does not prove GPU availability,
runtime dependency compatibility, or model correctness.

The complete run directory, including `provenance/`, is the reproduction unit;
an individual attempt folder is not a standalone bundle. Reproduction still
requires model weights, compatible GPUs and reconstructed dependencies. Restore
the recorded checkout, apply `source.patch` including deletions, then unpack the
source archive to restore new files. See the shared
[runtime requirements](../../../scripts/RUNTIMES.md). Historical results that
lack these snapshots must not be described as fully reproducible.

For an operational subset, select only canonical points without changing the
fixed protocol:

```bash
PAP_QPS_SCAN_ONLY_ARCHITECTURES=dp8,4p4d,pap_7pa1p_2k \
PAP_QPS_SCAN_ONLY_QPS=1.5,1.8 \
bash benchmarks/pap/experiments/e2e/PAP-20260903-AGENTIC-CODE-QPS-MATRIX/run.sh
```

Install the plotting dependency through the project environment:

```bash
uv pip install --python .venv/bin/python \
  -r benchmarks/pap/experiments/e2e/PAP-20260903-AGENTIC-CODE-QPS-MATRIX/requirements.txt
```

The driver refreshes the run's `summary.tsv`, `summary.json`, and one
SciencePlots figure after every point. The three panels show P99 end-to-end
request latency, mean TBT, and mean TTFT against configured QPS. Missing or
incorrect points remain visible as gaps rather than being silently discarded.
A PD point that completes correctly but falls below the 5 GB/s aggregate
same-node KV-transfer floor remains in the performance curve and is marked with
a black `x`; its summary status is `kv_transfer_below_floor`.
