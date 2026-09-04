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

Run or resume the complete matrix with:

```bash
bash benchmarks/pap/experiments/PAP-20260903-AGENTIC-CODE-QPS-MATRIX/run.sh
```

`results/matrix.env` records the dataset and AIPerf runner SHA-256 values.
Results use the layout
`results/<architecture>/qps_<rate>/attempt_<number>/`. A completed attempt is
never overwritten; rerunning the command skips valid points and gives failed
or incomplete points a new attempt number.

For an operational subset, select only canonical points without changing the
fixed protocol:

```bash
PAP_QPS_SCAN_ONLY_ARCHITECTURES=dp8,4p4d,pap_7pa1p_2k \
PAP_QPS_SCAN_ONLY_QPS=1.5,1.8 \
bash benchmarks/pap/experiments/PAP-20260903-AGENTIC-CODE-QPS-MATRIX/run.sh
```

Install the plotting dependency through the project environment:

```bash
uv pip install --python .venv/bin/python \
  -r benchmarks/pap/experiments/PAP-20260903-AGENTIC-CODE-QPS-MATRIX/requirements.txt
```

The driver refreshes `results/summary.tsv`, `results/summary.json`, and one
SciencePlots figure after every point. The three panels show P99 end-to-end
request latency, mean TBT, and mean TTFT against configured QPS. Missing or
incorrect points remain visible as gaps rather than being silently discarded.
A PD point that completes correctly but falls below the 5 GB/s aggregate
same-node KV-transfer floor remains in the performance curve and is marked with
a black `x`; its summary status is `kv_transfer_below_floor`.
