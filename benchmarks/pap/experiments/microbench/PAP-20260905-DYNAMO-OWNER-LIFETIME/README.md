# Dynamo reservation ownership and shared-prefix regression

CPU-only, real-clock check. One local simulated PA publishes two 16-token KV
blocks through vLLM's actual MessagePack event format. Two distinct sessions
query that prefix: the first requests 32 tokens, the second 48. Both hit the
same 32 cached tokens; only the second has 16 new Prefill tokens. One request
is then marked Prefill-complete; the other stays in Prefill.

No inference request finishes during the 370-second hold. The expected native
state is two active requests, three unique KV blocks, and 16 Prefill tokens.
This tests load accounting, not inference speed or GPU cache transfer.

```bash
PYTHONPATH=. .venv/bin/python benchmarks/pap/experiments/microbench/PAP-20260905-DYNAMO-OWNER-LIFETIME/probe.py --runtime official
PYTHONPATH=. .venv/bin/python benchmarks/pap/experiments/microbench/PAP-20260905-DYNAMO-OWNER-LIFETIME/probe.py --runtime pap
```

## Observations

| Observation | Official installed Dynamo | PAP explicit-owner build |
| --- | ---: | ---: |
| Active requests initially | 2 | 2 |
| Active requests at 310 seconds | 2 | 2 |
| Active requests at 370 seconds | 0 (incorrectly expired) | 2 |
| Prefill tokens at 370 seconds | 0 | 16 |
| Unique Decode blocks at 370 seconds | 0 | 3 |
| Active requests after explicit free | Already missing | 0 |

`official.txt` contains both premature-expiry warnings around 360 seconds. The
cleanup sweep is periodic: 300 seconds is the eligibility age, not an exact
wall-clock removal deadline. `pap-installed.txt` validates the final scripted build and additionally books a
third request after 370 seconds to exercise insertion-triggered cleanup.
That booking produces three active requests, and releasing all three returns
active requests, Prefill tokens and Decode blocks to zero. Both probes passed.

The official `_core.abi3.so` SHA-256 remains
`eb8e50c53f7f1d64edab405279cbb3ba4611e99c08562b554b36dc6df782a432`.
The scripted build's identity is recorded at the start of `pap-installed.txt`
and in `build.txt`.
This is a correctness reproduction across two runtime builds, **not** a
one-variable performance A/B or proof of identical compiler output.

The gateway's mocked lifecycle tests separately cover cancellation before and
after native booking, release failure, and shutdown. Short real-GPU E2E evidence
is recorded under `PAP-20260905-REFACTOR-VALIDATION/results.md`.

Local ignored `pap.log` (an initial probe field-name error) and `pap-fixed.log`
(an initial development build) are retained for debugging, not final evidence.
The final probe uses Dynamo's documented `device_blocks` score field. The final
script additionally prints binary identity; the official control above was run
before that diagnostic print was added, and its unchanged binary hash was
verified separately.
