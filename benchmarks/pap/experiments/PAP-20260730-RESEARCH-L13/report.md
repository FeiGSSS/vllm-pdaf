# PAP research loop L13: invalid first near-limit MPS treatment

Date: 2026-07-30

## Question and decision

L13 pre-registered whether changing PAP 6PA2P from the 18-Prefill/5-Attention
MPS chunk split to 20/3 is sufficient to remove the near-model-limit TTFT
deficit observed in L12.

The first C12 treatment is **invalid performance evidence**. It completes all
192 requests and all 48 sessions, and the MPS audit verifies 80 visible
Prefill SMs and 12 visible Attention SMs on each PA. However:

- the launcher exits with code 1;
- the routing audit reports only 189 lease-release responses for 192 routed
  requests;
- Prefill logs contain decode-commit failures such as
  `expected 18, got 17`;
- the aggregate correctness summary therefore fails.

The run must not support or falsify the MPS hypothesis. L13 remains open until
the lifecycle defect is fixed and the treatment is repeated on clean committed
code.

## Diagnostic-only metrics

These values are retained to guide diagnosis, not as paper evidence:

| Configuration | Raw req/s | Mean TTFT | Mean ITL | Standard good fraction |
| --- | ---: | ---: | ---: | ---: |
| L12 18/5 control | 1.576 | 5666 ms | 34.52 ms | 95.31% |
| L13 20/3 invalid treatment | 1.674 | 4806 ms | 34.29 ms | 99.48% |

The invalid treatment is directionally consistent with more Prefill capacity:
mean TTFT decreases by 15.2%. Raw throughput increases by only 6.2%, below the
pre-registered 8% threshold. Neither number is admissible while correctness
fails.

## Correctness diagnosis

The failed run records:

- six balanced PA owner assignments of eight conversations each;
- zero decode-token join errors;
- zero active sessions after drain;
- three missing lease-release acknowledgements in the routing audit;
- repeated Prefill-side commit deltas whose token-list length is one smaller
  than the requested sequence-length advance.

The post-run working tree experiments with batched final decode commit and
submit-only EngineCore control operations. Those changes alter control-path
semantics and are not part of the registered MPS-only intervention. If they
are retained, both the 18/5 control and 20/3 treatment must be collected
contemporaneously on the same clean commit.

## Provenance

- Pre-registered baseline commit: `8b9228871`
- Treatment staging directory:
  `benchmarks/pap/experiments/_staging/capacity/20260729_l13_mps20_3_c12/`
- Dataset SHA-256:
  `8ef6c8017930b8549ba077f14c1592d683fbd69d9de3795931657ba9f9dd1e73`
- Dataset: 48 sessions, four turns, 192 requests, approximately 10K new
  input tokens per turn, randomized O16 output.

## Next action

1. Finish and test the lifecycle correction.
2. Commit the implementation before collecting comparison evidence.
3. Run a clean 18/5 control and 20/3 treatment on the same commit if the
   correction changes the control path.
4. Apply the original TTFT, throughput, ITL, Standard-good-fraction, and
   correctness thresholds without modification.
