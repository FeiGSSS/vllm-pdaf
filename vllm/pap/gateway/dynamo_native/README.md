# PAP's local Dynamo selector dependency

Build with `bash benchmarks/pap/scripts/build_pap_dynamo_router.sh` from the
repository root. `Cargo.toml` is staged beside a checksum-verified upstream
source archive; its relative dependency path is intentional, not a checked-in
vendor tree. Rust 1.94.1 and `Cargo.lock` pin the build toolchain/dependencies.

`lib.rs` only binds the upstream KV-event index, worker selection, reservation
and load APIs. `explicit-owner.patch` adds an opt-in local constructor mode
using upstream `new_without_expiry`; normal upstream selectors keep their
existing expiry behavior. PAP cannot select the old lifetime mode through an
environment override. The wheel in `.venv-dynamo` is never modified.

The ownership contract is:

1. Selection books a unique request ID once.
2. Prefill completion removes only Prefill load.
3. Completion/cancellation releases the reservation, independent of its age.
4. Gateway shutdown cancels scheduling, drains pending booking outcomes and
   releases reservations before dropping the native core. Process death also
   destroys this **in-process** selector; there is no remote orphan owner.

The Python gateway shields native booking outcomes from caller cancellation so
it can release a booking even when cancellation races successful selection.
Release failures fail closed. This is not a renewable distributed lease and
must not be used with replicated/external selection services without a separate
owner-failure protocol.

Upstream: <https://github.com/ai-dynamo/dynamo/tree/2112d6ba74da72e2715ae69f4b76458b7691380d>.
Source is Apache-2.0; the downloaded archive retains its upstream license and
notices. Build evidence lives in `.local/pap-dynamo-router/build.txt` and is
preserved with experiments.
