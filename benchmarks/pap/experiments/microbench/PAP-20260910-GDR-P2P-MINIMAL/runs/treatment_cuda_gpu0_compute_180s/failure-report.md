# Minimal repro achieved: GPUDirect ingest + concurrent compute on the receiving L20

## What ran

`run.sh --where cuda --sink-gpu 0 --duration-s 180 --compute-gemm 8192`.

Source: `NF5468-L20T`, GPU 0 VRAM, NIXL agent `pap-src`.
Sink: `NF5468-L20B`, GPU 0 (`0000:01:00.0`) VRAM via dmabuf zero-copy, NIXL
agent `pap-sink`, issuing NIXL `READ` of 1400 MB chunks, 8 in flight, while a
background GEMM + host-copy loop kept the receiving GPU busy.

## Timeline (CST)

| Event | Time | Source |
| --- | --- | --- |
| last sink heartbeat | 15:24:34.145 | `sink-heartbeat.jsonl` |
| peer witness first unreachable | 15:24:34.498 | `witness.jsonl` |
| kernel panic | uptime 6787.79 s = 15:24:33 | ERST archive |
| peer witness reachable again | 15:30:46.752 | `witness.jsonl` |
| new boot | 15:30:30 | `last -x reboot` |

Last heartbeat to first unreachable: **0.35 s**. 114 transfers, 167.4 GB,
27.20 GB/s, 476 compute iterations before the machine stopped.

## Fault signature

Recovered from the new ERST archive `7683804184553979907` via the system
journal (`postmortem-pstore/`), two fatal sections in one record:

```text
[Hardware Error]: Hardware error from APEI Generic Hardware Error Source: 515
[Hardware Error]: event severity: fatal
[Hardware Error]:   section_type: PCIe error
[Hardware Error]:   port_type: 1, legacy PCI end point
[Hardware Error]:   device_id: 0000:01:00.0
[Hardware Error]:   vendor_id: 0x10de, device_id: 0x26ba
[Hardware Error]:   class_code: 030200
[Hardware Error]:   aer_cor_status: 0x00008000, aer_cor_mask: 0x0000b1c1
[Hardware Error]:   aer_uncor_status: 0x00014000, aer_uncor_mask: 0x00000000
[Hardware Error]:   aer_uncor_severity: 0x00476030
[Hardware Error]:   TLP Header: 00000000 00000000 00000000 00000000
Kernel panic - not syncing: GHES: Fatal hardware error
```

`00:01.0` is GPU 0 - the GPU this probe selected as the transfer destination.
Every field matches the field failures: same error source 515, same endpoint
`10de:26ba`, same `aer_uncor_status 0x00014000` (Completion Timeout +
Unexpected Completion), same severity, same zero TLP header, same panic.

## Contrast

| Arm | Destination | Compute on receiver | Duration reached | Volume | Result |
| --- | --- | --- | --- | --- | --- |
| control | host DRAM | no | 60 s | 1672 GB | completed |
| treatment A | GPU VRAM | no | 180 s | 5012 GB | completed |
| treatment B | GPU VRAM | **yes** | 6.0 s | 167 GB | **fatal, reproduced** |

Neither GPUDirect ingest alone nor sustained 28 GB/s RDMA alone is sufficient.
Adding concurrent compute on the receiving GPU reproduced the machine stop in
seconds, on the same GPU, with the identical signature.

Across the two compute-enabled runs the outcome differed (a 20 s run at the
same load completed, this 180 s run died after 6 s), so the trigger is
intermittent rather than deterministic.

## Scope

This establishes a sufficient trigger and a fast reproducer. It does not
identify which component (GPU firmware, GPU PCIe requester, platform
completion routing, driver) produces the completion anomaly, and it does not
establish a fix. The L20 endpoint still has no header log capability, so the
offending TLP remains unidentified.
