# PAP Attention Runtime Relocation Design

Date: 2026-07-14

## Problem

The authoritative PAP Attention process currently lives in
`examples/pap/pap_attention_executor.py`. It is a production runtime rather
than an example: launchers execute it directly, tests import its state and
compute classes, and it owns KV-cache, CUDA IPC, mailbox, scheduling, and
service state. The package configuration only installs `vllm*`, so an
installed vLLM package does not contain this runtime.

The file began as a PAP prototype and grew in place. This change corrects that
package boundary without redesigning the runtime.

## Scope

This phase makes `vllm/pap/attention_executor.py` the authoritative module.
It does not split the module or change any PAP protocol, state transition,
kernel, scheduling policy, transport, environment variable, or default.

The change will:

1. Move the existing implementation to `vllm/pap/attention_executor.py`.
2. Add a `main()` function containing the current script startup sequence.
3. Keep `examples/pap/pap_attention_executor.py` as a thin compatibility
   launcher that calls the packaged runtime.
4. Change PAP launchers and benchmark scripts to execute
   `python -m vllm.pap.attention_executor`.
5. Change tests to import the packaged module, while retaining one contract
   test for the compatibility launcher.

The following remain in `examples/pap` in this phase:

- proxy implementations and payload helpers;
- shell launchers;
- request, multi-turn, and transport smoke clients.

Their placement is separate architecture debt and is not part of this
behavior-preserving move.

## Compatibility

The documented `examples/pap/launch_pap_nixl.sh` command remains unchanged.
Direct execution of the old Attention script path also remains supported.
The old module is not treated as a stable Python import API; repository tests
and runtime code must import `vllm.pap.attention_executor` after this change.

## Validation

Validation must demonstrate structural and runtime equivalence:

- compile the packaged implementation and compatibility launcher;
- syntax-check the PAP launch and fixed-testbed shell scripts;
- run the PAP Attention, data-plane, contract, and launcher unit suites;
- run the established combined PAP regression suite;
- run one C1 quick PAP smoke through the fixed testbed if the local GPU lane is
  available;
- run `git diff --check`.

No pre-commit hook will be run, following the project-specific user request.
No performance result will be re-frozen because this phase has no intended
data-path change; the smoke run is a startup and correctness gate.

## Commit Boundary

The runtime relocation, reference updates, compatibility launcher, and tests
form one structural commit. Benchmark output remains untracked. Future module
splitting must be a separate design and commit series.
