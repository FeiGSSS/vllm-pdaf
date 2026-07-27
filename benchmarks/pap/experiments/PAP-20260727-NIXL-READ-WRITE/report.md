# NIXL PA migration: READ versus WRITE

Date: 2026-07-27

## Question

PAP PA-to-PA history migration currently uses `NixlConnector`, which aliases
the pull connector. The destination PA therefore initiates a NIXL `READ`.
This experiment checks whether the earlier same-node NIXL direction
asymmetry is still present.

## Runtime

- NIXL 1.3.0
- repository UCX 1.22.0 build and matching NIXL UCX plugin
- `UCX_PROTO_EMULATION_ENABLE=n`
- NVIDIA L20, CUDA IPC over PCIe
- 1 GiB transferred, three measured repetitions

The runtime must be loaded with:

```bash
source benchmarks/pap/scripts/configure_same_node_nixl.sh
pap_configure_same_node_nixl "$PWD"
```

The probe is
`benchmarks/pap/tooling/nixl_read_write_probe.py`. For the same physical
GPU1-to-GPU0 direction, WRITE uses initiator 1 and READ uses initiator 0:

```bash
.venv/bin/python benchmarks/pap/tooling/nixl_read_write_probe.py \
  --operation WRITE --initiator 1 --peer 0 \
  --regions 72 --total-mib 1024 --segment-kib 32

.venv/bin/python benchmarks/pap/tooling/nixl_read_write_probe.py \
  --operation READ --initiator 0 --peer 1 \
  --regions 72 --total-mib 1024 --segment-kib 32
```

## Results

| Physical direction | Shape | WRITE/PUT | READ/GET | WRITE speedup |
|---|---|---:|---:|---:|
| GPU1 to GPU0, same NUMA | 1 descriptor | 19.73 GB/s | 7.39 GB/s | 2.67x |
| GPU1 to GPU0, same NUMA | 32,760 x 32 KiB | 12.58 GB/s | 4.51 GB/s | 2.79x |
| GPU6 to GPU0, cross NUMA | 32,760 x 32 KiB | 12.01 GB/s | 3.50 GB/s | 3.43x |

The historical explicit GET-zcopy setting was then tested with the
cross-layer shape used by the current PA migration:

| Physical direction | Default READ | Forced GET-zcopy READ | Speedup |
|---|---:|---:|---:|
| GPU6 to GPU4, same NUMA, 1 GiB / 2 descriptors | 7.11 GB/s | 19.56 GB/s | 2.75x |
| GPU1 to GPU0, same NUMA, 512 MiB / 2 descriptors | 7.20 GB/s | 21.92 GB/s | 3.05x |
| GPU6 to GPU0, cross NUMA, 512 MiB / 2 descriptors | 6.64 GB/s | 21.95 GB/s | 3.31x |

Without loading the repository UCX 1.22 runtime, strict READ fails with
`No zero-copy protocol found for get into cuda from cuda`. That is a runtime
misconfiguration sample and is not included in the table.

## Conclusion

UCX 1.22 plus disabled emulation is not sufficient on this topology. The
runtime still selects a substantially slower CUDA IPC GET protocol unless
`UCX_CUDA_IPC_ENABLE_GET_ZCOPY=y` is explicit. The project-owned same-node
configuration now requires that setting and records it in PAP and PD
`effective_config.env` files.

A 7PA1P, C16, 16-session, five-turn E2E run confirmed correctness after the
change: 80/80 requests completed with no errors. The largest observed NIXL
READ fell from 738.5 ms to 110.9 ms. End-to-end migration latency improved
from mean/median/max `608/184/1873 ms` to `353.5/242.5/1355 ms`, though the
two runs selected 11 and 10 different migrations, so this is diagnostic
rather than a formal paired comparison. Overall mean TTFT changed from
1539.2 to 1521.6 ms; mean ITL changed from 34.94 to 35.86 ms.

The remaining long migration tail is outside the raw READ. In the slowest
new request, the target submitted NIXL 1225 ms after the migration decision,
the READ itself took 56 ms, and completion-to-install took another 198 ms.
That residual should be analyzed as standard KVConnector request scheduling
and target handoff latency, not as a private PAP transport protocol.
