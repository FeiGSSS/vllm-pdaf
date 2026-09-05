# Active reservation expiry in PAP's Dynamo integration

## Observed failure

The `coding` case in `runs/20260905_114511_3015679/` completed all 180 requests,
with no client errors or output-length mismatches. However, its gateway log
contains three `Expiring stale request` events before those requests completed.
This invalidates the run as evidence of correct load accounting or comparative
performance. Inference completion alone did not detect the routing defect.

| Request ID | Selection → expiry | Expiry → client completion |
| --- | --- | --- |
| `314be973-9d0e-4c8e-ba7b-ff9613e84a5e` | 315.366 s | 1.884 s |
| `cb020145-3ad5-42d0-8850-17eb252582a8` | 334.627 s | 187.334 s |
| `e70a02a9-fb93-4c56-9828-350fd8ad43be` | 354.244 s | 40.714 s |

Join `coding/service_logs/proxy.log` selection/expiry entries by request ID with
`coding/aiperf/profile.jsonl` field `metadata.x_request_id`. Rust log timestamps
are UTC; client `metadata.request_end_ns` is Unix time in nanoseconds. These
adjacent lifecycle observations establish that live requests lost their native
router reservation; this conclusion is not inferred from a latency ratio.

A live health read also showed two PAP active requests and two wrapper-owned
reservations while the native selector's per-worker active counts were all zero.
Do not infer that the inference requests were cancelled: they continued running.

## Dependency identity and source clue

Installed PyPI packages: `ai-dynamo==1.4.1`, `ai-dynamo-runtime==1.4.1`.
The installed `_core.abi3.so` SHA-256 is
`eb8e50c53f7f1d64edab405279cbb3ba4611e99c08562b554b36dc6df782a432`.

The reference Dynamo checkout inspected at
`098f6cee01a057014a6a3a0ec96d19b25f74458b` contains an absolute 300-second request
expiry in `lib/kv-router/src/sequences/single.rs`: expiry compares `started_at`
with the current time, and `mark_prefill_completed` only removes Prefill load.
This source is a diagnostic clue consistent with the installed runtime's
observed behavior; it is not asserted to be the exact source of the PyPI binary.
That reference checkout's public `SelectionService` interface does not expose a request-lifetime
renewal operation. The separate block-cache `router_ttl_secs` must not be
confused with this request expiry.

## Required correction

PAP's live-request ownership and native reservation lifetime must agree. A
permanent correction needs an explicit-owner lifetime or a real renewable lease
contract, including cancellation, failed reservation creation and shutdown.
Merely increasing an arbitrary timeout or repeatedly freeing/rebooking requests
would not establish that contract and has not been used as a workaround.

The benchmark audit now rejects `Expiring stale request`. The queued
`coding-full` case was stopped through its owned launcher's cleanup path (exit
130); no completed measurement is claimed. GPU process cleanup was verified.
The preceding `coding-half-trace` case and its frozen raw-window checks remain
valid. The historical invalidated runs remain invalidated after the correction.

## Correction (user-approved 2026-09-05)

The official v1.4.1 source at `2112d6ba74da72e2715ae69f4b76458b7691380d`
already provides `ActiveSequencesMultiWorker::new_without_expiry`, but its
selection service did not use that ownership mode. PAP now builds an isolated
binding to that source, with a small patch selecting explicit-owner lifetime for
the local, non-replicated selector. Default upstream behavior is unchanged.
The older reference checkout above is not used as the build source.

See `vllm/pap/gateway/dynamo_native/`, `scripts/build_pap_dynamo_router.sh`, and
`experiments/microbench/PAP-20260905-DYNAMO-OWNER-LIFETIME/` (relative to the
repository/PAP benchmark roots). The CPU real-clock reproduction retains one
Prefill request and one Decode request for 370 seconds: the official runtime
loses both reservations; the patched selector retains both until explicit free.
No TTL increase or repeated rebooking is involved.

The gateway owns booking outcomes across client cancellation, releases on
completion/failure, blocks new routing after a failed release, and cancels queued
selection before draining/freeing on shutdown. The short 7PA1P E2E validation
completed with zero native or wrapper reservations left. The subsequent full
queue passed the unhalved coding replay (four requests over 300 seconds,
longest 519.783 seconds from selection to client completion) and the 600-second
large-dataset cancellation case (195 completed, 60 cancelled). Both drained
native and wrapper ownership without release failures. See `results.md` for
exact source snapshots and scope; no comparative performance claim is made.

The user separately chose default cross-session prefix sharing. New unsalted
fixtures are registered under a new ID; old salted data is not rewritten and
explicit isolation requests are still rejected by PAP's Dynamo preflight.
