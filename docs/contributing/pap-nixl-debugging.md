# PAP Same-Node NIXL Debugging

Read this before changing PAP/PD same-node NIXL launch configuration, KV
migration, connector behavior, or related benchmarks.

## Runtime Invariants

- Enter through the project-owned `configure_same_node_nixl.sh`; locate it
  with `rg --files`. Do not run a comparison against the wheel's bundled UCX
  or an unverified NIXL plugin.
- The configuration must fail closed unless the NIXL plugin resolves to the
  project UCX runtime, UCX is built with multithreading, protocol emulation is
  disabled, and CUDA IPC GET zero-copy is enabled.
- Both PAP and PD runs must record these effective settings. Check the run's
  `effective_config.env`; do not infer the runtime from the shell that launched
  it.

## Diagnosis Order

1. Verify the configured UCX version, plugin linkage, disabled emulation, and
   `UCX_CUDA_IPC_ENABLE_GET_ZCOPY=y`.
2. Run the project NIXL READ/WRITE probe before changing vLLM scheduling or
   migration code. Locate `nixl_read_write_probe.py` with `rg --files`.
3. Compare the same physical GPU direction and transfer shape. On the current
   L20 host, a cross-layer READ should be near the PCIe CUDA-P2P reference
   (about 20 GB/s), not the recurring 0.4--7 GB/s degraded range.
4. Only after the raw transfer passes, enable
   `PAP_NIXL_XFER_DIAGNOSTICS=1` in a real run. Separate decision-to-submit,
   NIXL transfer, and completion-to-install time; do not label their sum as
   NIXL bandwidth.
5. Preserve the official vLLM `NixlConnector` data plane and metadata format.
   PAP post-Prefill migration allocates target blocks through the KV-cache
   manager without creating a synthetic Prefill request, then progresses the
   standard connector receive independently of the Prefill forward queue.

## Known Recurring Failure

No NVLink does not imply that same-host GPU P2P must use TCP. This host supports
CUDA IPC over PCIe. The recurring failure has been UCX selecting a slow GET
protocol when CUDA IPC GET zero-copy was not explicit. Disabling emulation
prevents TCP fallback but does not by itself guarantee the fast GET protocol.

Do not respond to this symptom by switching to NCCL, writing a private
transport, changing descriptor layout, or blaming scheduler polling until the
runtime invariants and raw probe above have been checked.

For prior evidence, search the experiment registry for
`PAP-20260713-UCX122-GET-AB` and `PAP-20260727-NIXL-READ-WRITE`.
