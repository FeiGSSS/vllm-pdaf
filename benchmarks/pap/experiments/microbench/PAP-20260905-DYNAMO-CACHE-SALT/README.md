# Dynamo cache-salt event compatibility probe

CPU-only controlled comparison. Two simulated workers publish the same 32 token
IDs as two 16-token chunks, using vLLM's real block-hash and event serializers.
Both use the same hash seed. The only difference between runs is whether workers
use the same cache salt or different cache salts. Subscription is acknowledged
before publication. Repeated overlap queries expose asynchronous application.

Run with the project Python and installed Dynamo environment:

```bash
.venv/bin/python benchmarks/pap/experiments/microbench/PAP-20260905-DYNAMO-CACHE-SALT/probe.py
.venv/bin/python benchmarks/pap/experiments/microbench/PAP-20260905-DYNAMO-CACHE-SALT/probe.py --different-salts
```

This diagnoses event-index compatibility, not model inference correctness or GPU
performance. Preserve stdout and stderr: the latter contains Dynamo's mismatch
and missing-parent diagnostics.

## Observed result (2026-09-05)

With the installed Dynamo SelectionService, `same_salt.log` ends with two
indexed blocks on both workers and no mismatch/missing-parent errors.
`different_salts.log` ends with one worker at only one indexed block and the
other at two; it contains one hash mismatch and one missing-parent error. Which
worker wins the first insertion is timing-dependent, so worker identity is not
the assertion. Both listeners report receiving both messages in both runs.

This reproduces the salted-fixture E2E failure without inference, GPU scheduling,
or transport throughput as confounding factors. The current selector cannot
safely combine these differently salted prefixes in its shared token index.
