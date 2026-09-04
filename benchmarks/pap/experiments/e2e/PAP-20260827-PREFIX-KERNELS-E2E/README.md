# Prefix-kernel end-to-end validation

This record contains the AIPerf serving validation for the PAT kernel studied
in `../../microbench/PAP-20260827-PREFIX-KERNELS/`.

It includes the 120-second PAT-versus-Cascade A/B, the historical hybrid
one-hour run, and the finalized persistent-PAT 2K and 32K one-hour summaries.
The microbenchmark kernel traces and upstream implementation pins remain in
the microbenchmark record; only client-visible serving results are stored
here.

The finalized persistent-PAT results are:

| Metric | 2K / 32K |
| --- | ---: |
| Completed requests | 740 / 651 |
| Output throughput | 198.64 / 176.88 tok/s |
| Mean TTFT | 152.73 / 196.70 s |
| Mean TBT | 77.03 / 68.49 ms |
| Mean end-to-end latency | 227.11 / 263.54 s |

See the linked microbenchmark report for the kernel implementation, isolated
measurements, and interpretation connecting both experiment classes.
