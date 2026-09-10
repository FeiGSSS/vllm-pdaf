# Minimal GPUDirect P2P RDMA repro: does peer DMA into GPU VRAM kill the host?

## Question

Repeated cross-host PAP PD runs fatal-stop `NF5468-L20B` with an L20 PCIe
`Completion Timeout + Unexpected Completion` (`aer_uncor_status 0x00014000`)
escalated to a GHES NMI panic. The error tracks the active Decode GPU group and
the KV receive path, where the ConnectX-7 writes KV directly into L20 VRAM
(`kv_buffer_device: cuda`, `use_host_buffer: False`) at roughly 24 GB/s.

Is the peer DMA landing in GPU VRAM sufficient to trigger it, with the model,
vLLM, Dynamo, etcd and the request router removed?

## Protocol

`probe_gdr_p2p.py` uses the production transport stack unchanged (NIXL agent +
`libplugin_UCX.so` + UCX 1.22 + RC over InfiniBand on the production devices).
The source peer publishes one registered buffer; the sink peer registers a
destination buffer and issues back-to-back NIXL `READ` transfers of the
production size, matching the Decode side's pull direction.

Two arms, identical except for the sink destination memory:

| Arm | Sink destination | Purpose |
| --- | --- | --- |
| control | host DRAM (`--where cpu`) | same RDMA volume without GPU peer DMA |
| treatment | GPU VRAM (`--where cuda`) | GPUDirect peer DMA into VRAM |

## Validity rules

- `run.sh` must execute on the source peer. The sink peer may hard-reset, so
  orchestration, the witness timeline and `verdict.json` live on the source.
- Both hosts are identified and recorded (hostname, boot id, kernel, cmdline,
  GPU compute process count) before traffic starts.
- The run is refused if either host has GPU compute processes.
- A sink that stops answering for three consecutive probes is treated as a
  reset; a single dropped probe is not.
- Absence of a failure in the control arm does not prove the treatment arm
  innocent, and a passing treatment arm does not rule out the full E2E path
  (which additionally runs GPU compute, cudagraph replay and the driver
  channel).

## Observed

2026-09-10, NF5468-L20B sink / NF5468-L20T source, same UCX 1.22 + NIXL UCX
plugin + `rc,cuda_copy` stack as production, 1400 MB chunks, 8 in flight.

| Run | Arm | Duration | Transfers | Volume | Mean rate | Result |
| --- | --- | --- | --- | --- | --- | --- |
| `control_cpu_60s_r3` | host DRAM dst | 60 s | 1139 | 1672 GB | 27.86 GB/s | completed |
| `control_cpu_30s_verify` | host DRAM dst | 30 s | 569 | 835 GB | 27.48 GB/s | completed |
| `treatment_cuda_gpu0_180s` | GPU 0 VRAM dst | 180 s | 3414 | 5012 GB | 27.78 GB/s | completed |
| `treatment_cuda_gpu0_compute_20s` | GPU 0 VRAM + compute | 20 s | 371 | 545 GB | 26.73 GB/s | completed |
| `treatment_cuda_gpu0_compute_180s` | GPU 0 VRAM + compute | 6.0 s | 114 | 167 GB | 27.20 GB/s | **fatal, reproduced** |
| `control_cpu_gpu0_compute_180s` | host DRAM + compute | 180 s | 3415 | 5013 GB | 27.79 GB/s | completed |
| `treatment_cuda_gpu0_compute_to900ms_180s` | GPU 0 VRAM + compute, GPU completion timeout raised to 260-900 ms | 11.8 s | 228 | 335 GB | 27.0 GB/s | **fatal** |
| `treatment_cuda_gpu0_compute_nohostcopy_180s` | GPU 0 VRAM + on-device GEMM only, no host copies | 6.4 s | 94 | 138 GB | 27.3 GB/s | **fatal** |

`uptime -s` on the sink stayed `2026-09-10 13:31:26` for the first four runs and
the peer witness recorded the sink reachable for every sample. The production
failure window opened 28-40 s after traffic start; the compute-free treatment
arm ran 180 s without a reset.

### What the four-cell matrix establishes

| | host DRAM destination | GPU VRAM destination |
| --- | --- | --- |
| no compute on receiver | completed (1672 GB) | completed (5012 GB) |
| compute on receiver | completed (5013 GB) | **fatal in seconds** |

Both conditions are required. Peer DMA alone is not enough; a busy GPU alone is
not enough. Together they stop the machine within seconds.

### What each follow-up ruled out

- **Raising the GPU's own completion timeout from 50 us-50 ms to 260-900 ms
  changed nothing** (same `0x00014000`, same severity, same panic). A merely
  late completion would have been absorbed. The completion is therefore lost or
  the outstanding request is no longer matchable, not just slow. "Wait longer"
  is not the fix, and suppression is now clearly unsafe: the transaction is
  genuinely disrupted, so hiding the report would mean continuing with a read
  that did not complete.
- **Removing every host-side copy from the compute loop** (pure on-device GEMM,
  so the probe issues no bulk host read at all) still reproduced the identical
  fault. The request whose completion is lost is therefore not the probe's
  H2D read; it belongs to the GPU/driver host interface itself, most plausibly
  the command-processor fetch. That puts it outside what application code can
  fix.
- In the no-host-copy run the compute loop stopped advancing for about three
  seconds before the panic (592 -> 650 iterations, then `+0` three times) while
  RDMA transfers kept completing at 27 GB/s. That is consistent with the GPU's
  execution engine stalling while its ingress stayed busy, but it appeared in
  only one of the three crashing runs, so treat it as a lead rather than a
  signature. The other loops had more slack and would not have exposed it.

### Open item: does the reproducer match the production regime?

The reproducer drives **one** GPU at 28 GB/s, saturating its PCIe Gen4 x16
ingress. Production spread roughly 24 GB/s across four GPUs, about 6 GB/s each.
The signature matches but the quantitative condition may be harsher than
production, so a multi-GPU arm is needed before treating this reproducer as
production-equivalent.

### Reproduction

Adding concurrent compute on the receiving GPU produced the field failure. The
last sink heartbeat was 15:24:34.145 CST and the peer witness recorded the sink
unreachable at 15:24:34.498 CST, 0.35 s later. The new ERST archive
`7683804184553979907` records two fatal PCIe sections on `0000:01:00.0` - the
GPU this probe selected as the destination - with `10de:26ba`,
`aer_uncor_status 0x00014000`, `aer_uncor_severity 0x00476030`, a zero TLP
header and
`Kernel panic - not syncing: GHES: Fatal hardware error`: field-identical.

See `runs/treatment_cuda_gpu0_compute_180s/failure-report.md` and the recovered
raw evidence in `runs/treatment_cuda_gpu0_compute_180s/postmortem-pstore/`.

### Transport actually used

`UCX_TLS=rc` alone fails the VRAM registration (`NIXL_ERR_BACKEND`), which only
shows that the CUDA component is required to export the buffer. The runtime
selection with the production `rc,cuda_copy` setting is direct:

```text
ucp_context.c:2139 UCX DEBUG register cuda memory on: cuda_cpy, ibp49s0
                        ... multi-frag zero-copy ... rc_mlx5/ibp49s0:1/path0
```

GPU memory is registered on the IB device itself and dispatched as multi-frag
zero-copy over `rc_mlx5`, so the treatment arm is a real dmabuf GPUDirect peer
DMA into L20 VRAM, not a host bounce-buffer path. A rate above the Gen4 x16
nominal limit is therefore not by itself evidence of staging: Gen4 x16 sustains
roughly 28-30 GB/s for large writes, which is what the treatment arm reached.

### Conclusion of this probe

The raw transport primitive alone - one RC RDMA READ stream, dmabuf zero-copy
landing directly in L20 VRAM, production chunk size, sustained past the known
failure window - does **not** reproduce the fatal. Adding concurrent compute on
the receiving GPU does, within seconds and on the same GPU, with an identical
AER/GHES signature.

The discriminator is therefore the receiving GPU being a busy requester of its
own PCIe traffic at the same time as it absorbs ~28 GB/s of inbound peer
writes. That is the state production decode GPUs are in when they die.

Remaining unknowns: which component emits the anomalous completion (GPU,
platform completion routing, driver), why the intermittency, and whether a
configuration change removes it. Not yet tried: varying the compute intensity,
four concurrent sink GPUs, and testing MPS / 10-bit tag / completion-timeout
changes against this now-fast reproducer.

### Probe defects found and fixed during bring-up

- Remote descriptors were built with the local segment type, so the control arm
  was rejected with `prepXferDlist: no available backends for mem type
  'DRAM_SEG'`. Remote descriptors must use the remote agent's segment type.
- Backgrounding a compound remote command (`cd ... && ... &`) leaves an
  unredirected subshell holding the SSH channel open, so the launch never
  returns. Remote launchers are now files invoked as
  `nohup setsid <script> >log 2>&1 </dev/null &`.
- Releasing the NIXL agent with posted-but-unreaped transfers aborts at
  interpreter exit (`free(): double free detected in tcache 2`). Transfers are
  now drained and handles released before a deliberate `os._exit`.
