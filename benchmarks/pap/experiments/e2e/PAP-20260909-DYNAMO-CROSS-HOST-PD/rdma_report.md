# Cross-host PD after RDMA and PSU repair

Accepted run: `runs/20260910_crosshost_pd_rdma_c30_psu_fixed_r4`.
Raw service logs, AIPerf records, launch scripts and telemetry remain in that
directory; summaries and audits are copied to `results/rdma_psu_fixed`.

Qwen3-8B FP16; seven Prefill workers on L20T GPUs 0–6, one Decode on
L20B GPU 7; the same native-32K dataset, 60 three-turn sessions, concurrency 30.
All 180 requests and 60 conversations joined successfully. Startup,
correctness, transfer and cleanup audits passed; no residual GPU processes.

| Metric | Previous cross-host TCP | Cross-host RDMA, PSU repaired |
| --- | ---: | ---: |
| Mean TTFT | 492.864 s | 7.821 s |
| P95 TTFT | 691.028 s | 12.550 s |
| Mean ITL | 27.398 ms | 58.348 ms |
| Request throughput | 0.0546 req/s | 2.0925 req/s |
| Mean NIXL transfer residence | 194.701 s | 123.597 ms |

NIXL completed 180 transfers, averaging 1911.44 MB each. Total bytes divided
by summed transfer residence time is 15465.11 MB/s, approximately 15.47 GB/s.
This is a residence-weighted transfer rate, not wall-clock NIC throughput.
The nine NIXL reporting windows have throughput P90 18238.32 MB/s and maximum
19387.36 MB/s; these are window statistics, not per-request percentiles.

The previous UCX build excluded verbs/RDMA and the deployment explicitly used
`tcp,cuda_copy` on 1 Gb/s Ethernet. The repaired deployment uses the RDMA-enabled
UCX 1.22 build, `rc,cuda_copy`, and ACTIVE 400 Gb/s InfiniBand on both hosts.
L20B's current HCA name is `ibp49s0:1`; L20T uses `mlx5_0:1`. The experiment
configuration records both names and startup fails if the devices do not match.

The prior same-host PD reference averaged 8.888 s TTFT, so this cross-host
result is back in the same latency range. These historical comparisons are
descriptive, not a controlled one-variable A/B: the host was rebooted and the
PSU repaired between runs. A lower TTFT here does not establish a cross-host
advantage. Prefill placement changed on 50/120 later-turn transitions.

Cross-host router-to-Prefill timestamp differences are negative in this run,
indicating unsynchronized host clocks; that interval is not a valid latency
measurement. Client TTFT, single-host Prefill duration and NIXL local duration
remain usable. No mechanism is inferred from the invalid cross-host interval.

The user identified PCIe fatal errors and shutdowns in the two interrupted
full-load attempts and reported the PSU repaired before this run. Earlier
attributions to controller lifetime or sandbox behavior were unsupported and
are withdrawn. Those incomplete attempts remain invalid performance samples.
