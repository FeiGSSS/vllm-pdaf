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

New multi-turn workloads omit salt by default, allowing identical prefixes to
be shared across conversations. The `qwen3-8b-yarn131k-shared-prefix` dataset
is a separately registered derivative; old salted fixtures remain unchanged.

## PAP-only Dynamo selector

This selector is mandatory for PAP. Both the launcher and Gateway reject
retired routing policies, including conversation affinity and round robin.
The official Dynamo DP/PD baseline environment is unchanged.

```bash
rustup toolchain install 1.94.1 --profile minimal
bash benchmarks/pap/scripts/build_pap_dynamo_router.sh
```

PAP imports `pap_dynamo_router` from `.local/pap-dynamo-router`, not the official
DP/PD `.venv-dynamo` runtime. The thin Python binding builds upstream
`dynamo-kv-router` at the official v1.4.1 source commit
`2112d6ba74da72e2715ae69f4b76458b7691380d`, with the colocated
`vllm/pap/gateway/dynamo_native/explicit-owner.patch` and colocated Cargo lockfile.
No routing/scoring algorithm is reimplemented. This is a source-built PAP
dependency, not asserted to be byte-identical to the PyPI wheel.

The local selector uses explicit-owner reservations: Prefill completion removes
only Prefill load; completion/cancellation frees the reservation. No arbitrary
age expires live requests. Cancellation racing a booking waits for the booking
outcome before freeing it. Shutdown stops queued selection, drains ownership
cleanup, and drops the native selector. A failed release blocks further routing
instead of silently losing load accounting. This mode is local and non-replicated;
it is not a lease design for remote selector services.

The gateway validates the lifetime contract; there is no fallback to the old
300-second runtime. `PAP_DYNAMO_PYTHON` is retired. Installation records archive,
patch, binding, lockfile and library identities in `build.txt`; experiment
provenance preserves the build record and CPU library alongside source/config.
Official DP/PD packages and their launchers remain unchanged.

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

## Manual termination caveat

The 600-second request-cancellation protocol passed its Gateway, Attention and
native-router drain audits. External termination of the shell process is a
different operation: Bash can defer its trap while waiting for the foreground
benchmark pipeline, and AIPerf workers can outlive a terminated controller and
hold its log pipe open. This occurred when stopping the duplicate validation
queue; its verified client process group and remaining workers required manual
cleanup. Final GPU/process absence was checked afterward.

Do not assume that sending TERM only to the launcher stopped the whole run.
Resolve the exact run's launcher, timeout/client process group and descendants,
then verify their termination and GPU cleanup. Never use broad name-based kills
against unrelated jobs. This operational limitation is recorded, not claimed
fixed by the Dynamo request-reservation change. Changes or tests of process
supervision require separate user approval during this closeout.
