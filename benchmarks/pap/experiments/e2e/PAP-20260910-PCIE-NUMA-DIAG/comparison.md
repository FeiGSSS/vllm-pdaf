# NUMA placement contrast, 2026-09-10

| Placement | Run | Client request window (CST) | Completed | Seconds | Result |
|---|---|---|---|---|---|
| Decode GPU0–3, NIC-local NUMA0 | 20260910T025543Z_4p4d_node0_r1 | 11:02:02.014–11:03:27.737 | 180/180 | 85.72 | passed |
| Decode GPU4–7, cross-NUMA1 | 20260910T030625Z_4p4d_node1_r1 | 11:08:22.117–11:09:45.277 | 180/180 | 83.16 | passed |

Same 4P4D C30 180-request dataset, remote Prefill GPUs0–3, software/scripts and recording. Both startup mappings audited and both hosts GPU-clean before/after. Each run completed 180 KV transfers. No increased Linux AER counters detected by supervisor; this is not proof of absence of hardware errors because root AER reporting is disabled and root-only endpoint/BMC state was not sampled.

The current pair does not reproduce the original 09:59 fatal. Cross-NUMA is not shown to be a sufficient trigger. Physical GPU/path and sequential cache/thermal history differ. One short run per placement does not exclude intermittent or cumulative hardware, firmware, driver or communication-stack failures. Do not interpret the small runtime difference as a measured NUMA performance advantage. Raw logs, snapshots, scripts, config and off-host mirrors are preserved in each run. Do not change BIOS/ACS based on this pair alone.
