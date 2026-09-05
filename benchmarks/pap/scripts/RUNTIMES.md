# PAP experiment runtime dependencies

Dependency installation and process configuration are separate. Installation
persists on disk; exported variables apply only to a process and its children.
The experiment launchers configure their own environment. Do not depend on
interactive shell startup files to select communication libraries.

The PAP runner only replays an existing `PAP_AIPERF_INPUT_FILE`. The experiment
must specify `PAP_AIPERF_SESSIONS` and `PAP_AIPERF_EXPECTED_REQUESTS`; the latter
is the expected total turns for those sessions, not a token count. Dataset
generation belongs to `datasets/tools/` and is never performed during service
startup. Old input/output-length and turn-generation controls are rejected.
Input/output lengths in a heterogeneous workload come from the dataset and
client records, not synthetic defaults in the launcher.

The current in-process Dynamo selector does not support per-request `cache_salt`
isolation. PAP rejects such requests before Prefill; replay preflight rejects
salted datasets with Dynamo routing. The source data must not be silently
rewritten to remove isolation. See the controlled
`experiments/microbench/PAP-20260905-DYNAMO-CACHE-SALT/` reproduction.

## Same-node NIXL (Dynamo PD baselines)

Install once, from the repository root:

```bash
bash benchmarks/pap/scripts/setup_same_node_nixl.sh install
```

This builds UCX 1.22.0 with CUDA and multi-thread support and the NIXL 1.3.0
UCX plugin under `.local/`. The build targets this host's CUDA installation
and SM89 GPUs. It disables verbs, RDMA CM, and GDRCopy: this is a **same-node**
runtime, not a cross-machine RDMA installation. A working project `.venv`,
CUDA development files, and the build tools listed by the script are required.
The plugin build does not install the Python NIXL package in every environment.

Check the installed runtime without rebuilding:

```bash
bash benchmarks/pap/scripts/setup_same_node_nixl.sh verify
```

The Dynamo launcher sources `configure_same_node_nixl.sh`. It selects the
plugin and UCX library/module directories and validates the UCX version,
multi-thread build and plugin library resolution. It sets:

- `UCX_PROTO_EMULATION_ENABLE=n`;
- `UCX_CUDA_IPC_ENABLE_GET_ZCOPY=y`;
- default `UCX_TLS=cuda_ipc,cuda_copy,tcp`.

TCP remains an available transport. Successful validation or agent creation
does **not** prove GPU payloads used CUDA IPC. Preserve the transfer audit and
worker logs with the result; a bandwidth floor alone also cannot identify the
transport or explain poor performance. DP8 does not transfer KV between workers.

## NVSHMEM (PAP attention–projection)

The current launcher requires NVSHMEM 3.3.24 built for CUDA 13, installed at
`.local/nvshmem-3.3.24-cuda13` or `PAP_NVSHMEM_PREFIX`. Its configuration script
checks `nvshmem-info`; it does not install NVSHMEM. The repository currently
does not provide a complete installer for this dependency.

`configure_nvshmem.sh` invokes `build_nvshmem_device_bridge.sh` for PAP's CUDA
bridge. This is a build check, not an unconditional rebuild. The compiler defaults to CUDA
13 NVCC in the project's Python 3.12 environment, and the target defaults to
`sm_89`. The cache records source/script contents, compiler identity, build
arguments, NVSHMEM headers/libraries, selected compiler environment variables,
and the output library checksum in `libpap_nvshmem_device.so.build.txt` next to
the library. A mismatch causes a rebuild. Builds are serialized and a failed
compile does not overwrite the previous library. Preserve this record with
experiments; it is not a complete lockfile for the system CUDA toolkit, driver,
or host compiler's transitive dependencies.

The launcher configures same-host P2P, disables the VMM heap for PAP's differing
visible-device/MPS mappings, and disables remote transport and IBGDA. This is
not the future cross-machine path. Runtime selection belongs to the launcher,
not to a user's `.bashrc`.

## PAT

`build_pat_attention.sh` installs the optional prefix-attention extension using
the colocated patch. Save its build output and installed version with any
performance comparison. A successful PAP launch is not proof that PAT was
available: the automatic kernel policy can use Triton instead. Confirm the
selected attention implementation from runtime evidence when comparing kernels.

## Reproduction evidence

Preserve the requested and effective configuration, source revision and local
patch, dataset checksum, model configuration, Python package versions, GPU model
and mapping, driver/CUDA versions, communication-library paths and versions,
startup logs, topology/graph/transfer audits, client output and exit status.
Do not copy the entire environment: it may contain credentials. Missing evidence
must be listed as a limitation, rather than inferred from a current installation.
