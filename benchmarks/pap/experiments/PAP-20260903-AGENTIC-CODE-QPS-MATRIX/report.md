# Agentic Coding QPS matrix results

The matrix completed all 40 points: eight architecture/configuration lanes at
five Poisson request rates. Every point completed 180/180 requests without an
AIPerf request error. The fixed protocol and replay dataset hashes are recorded
in `results/matrix.env`.

All slash-separated values below follow configured QPS
`0.6 / 0.9 / 1.2 / 1.5 / 1.8`.

| Architecture | Actual req/s | Mean TBT (ms) | Mean TTFT (s) | P99 E2E (s) |
| --- | --- | --- | --- | --- |
| DP8 | 0.517 / 0.683 / 0.754 / 0.802 / 0.905 | 34.1 / 36.0 / 41.5 / 45.6 / 50.5 | 1.16 / 1.23 / 1.32 / 1.37 / 1.36 | 66.4 / 69.6 / 75.3 / 84.9 / 94.6 |
| 2P6D | 0.546 / 0.713 / 0.777 / 0.813 / 0.890 | 30.7 / 34.0 / 37.9 / 41.3 / 43.8 | 1.36 / 1.56 / 2.08 / 3.48 / 2.94 | 55.0 / 58.0 / 67.2 / 76.3 / 80.5 |
| 4P4D | 0.545 / 0.669 / 0.738 / 0.777 / 0.799 | 34.3 / 40.2 / 47.1 / 53.5 / 58.4 | 1.30 / 1.33 / 1.48 / 1.56 / 1.68 | 61.4 / 70.8 / 80.7 / 97.6 / 102.1 |
| 6P2D | 0.498 / 0.548 / 0.566 / 0.576 / 0.570 | 50.1 / 72.9 / 95.7 / 107.2 / 114.5 | 1.26 / 1.44 / 1.56 / 1.76 / 5.82 | 89.9 / 132.0 / 171.6 / 182.6 / 195.6 |
| PAP 7PA1P-2K | 0.518 / 0.619 / 0.660 / 0.711 / 0.732 | 40.3 / 44.8 / 51.5 / 55.8 / 61.8 | 1.51 / 1.60 / 1.73 / 1.73 / 1.93 | 71.3 / 79.2 / 89.3 / 98.4 / 108.0 |
| PAP 7PA1P-32K | 0.495 / 0.609 / 0.653 / 0.680 / 0.745 | 40.7 / 45.1 / 51.8 / 57.1 / 60.9 | 1.50 / 1.58 / 1.71 / 1.76 / 1.83 | 71.4 / 79.3 / 90.0 / 99.1 / 106.1 |
| PAP 6PA2P-2K | 0.505 / 0.637 / 0.696 / 0.740 / 0.730 | 37.3 / 40.7 / 45.7 / 50.6 / 57.1 | 1.46 / 1.59 / 1.74 / 1.90 / 1.82 | 66.5 / 70.9 / 79.2 / 86.7 / 96.4 |
| PAP 6PA2P-32K | 0.507 / 0.649 / 0.683 / 0.742 / 0.757 | 37.5 / 40.5 / 45.6 / 50.4 / 56.2 | 1.44 / 1.58 / 1.67 / 1.86 / 1.93 | 66.5 / 68.4 / 80.4 / 86.8 / 94.7 |

## Main observations

- DP8 has the highest measured throughput and the lowest mean TTFT across the
  high-QPS region.
- 2P6D has the lowest TBT and P99 end-to-end latency, but its TTFT rises as two
  Prefill workers approach saturation.
- 6P2D is Decode-bound: throughput plateaus near 0.57 req/s while TBT exceeds
  100 ms at configured QPS 1.5 and 1.8.
- The second Projection consistently reduces PAP mean TBT by 7--12% and P99
  end-to-end latency by 7--14% relative to 7PA1P. Its throughput benefit is
  smaller because 6PA2P gives up one PA and its associated KV capacity.
- PAP-2K and PAP-32K remain close across the sweep. At configured QPS 1.8,
  6PA2P-32K reaches 0.757 req/s with 56.2 ms mean TBT, versus DP8's 0.905
  req/s and 50.5 ms.

## NIXL audit

The same-node runtime loaded UCX 1.22 with CUDA IPC and CUDA-copy transports.
Passing low-QPS PD points reproduce the historical 32K NIXL baseline. Four
completed points fall below the 5 GB/s aggregate KV-transfer floor:

| Point | Aggregate KV throughput |
| --- | ---: |
| 2P6D, QPS 1.5 | 3.29 GB/s |
| 2P6D, QPS 1.8 | 4.19 GB/s |
| 4P4D, QPS 1.8 | 4.47 GB/s |
| 6P2D, QPS 1.8 | 1.42 GB/s |

These points completed correctly and remain in the performance curves because
the low transfer throughput is part of the measured PD behavior. They are
marked with a black `x` in the figure and with status
`kv_transfer_below_floor` in `results/summary.tsv`.

## Artifacts

- `results/summary.tsv` and `results/summary.json`: normalized 40-point data;
- `results/figures/qps_latency.png`: SciencePlots raster figure;
- `results/figures/qps_latency.pdf`: vector figure;
- `results/<architecture>/qps_<rate>/attempt_001/`: raw AIPerf, service logs,
  effective configuration, correctness audits, and GPU telemetry.

The figure panels are P99 end-to-end request latency, mean TBT, and mean TTFT.
The x-axis is configured QPS; `actual_qps` is retained separately in the
summary because multi-turn dependencies and saturation prevent every lane from
reaching the configured arrival rate.
