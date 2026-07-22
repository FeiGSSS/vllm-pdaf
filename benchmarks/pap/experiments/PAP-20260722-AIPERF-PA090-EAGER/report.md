# PAP versus PD: 0.90 PA-memory eager AIPerf baseline

## Scope

This four-GPU eager-mode scan refreshes the PAP-versus-PD baseline with a 0.90
PAP Prefill memory budget and generated dataset identity independent of the
matrix ID.

- vLLM/PAP commit: `7401e8e1260a81b4f7b73d9a6857052e65149791`
- Tracked worktree at launch: clean
- AIPerf: 0.11.0 (`854ff91a4a221f899b806e7660a89b41b80d5689`)
- Model and hardware: Qwen3-8B FP16 on four NVIDIA L20 GPUs
- Work per point: 32 conversations, ten turns, 320 requests
- Timing: conversation concurrency with delays
  `0,3,3,1,3,3,1,3,3,1` seconds
- Dataset seed: 42
- Dataset SHA-256:
  `56dfe24c63fbb582f113db6e7f2ec2422bb313dcf23393ea192a062db158ea85`
- Execution mode: eager for both PAP and PD

The randomized workload uses an 8,192-token initial-input mean, 512-token
later-turn input mean, and 32-token output mean. The corresponding medians are
8,000, 500, and 30 tokens. The longest estimated request is 16,224 tokens,
leaving 3,776 tokens below `max_model_len=20000`.

## Runtime configuration

| Role | `gpu_memory_utilization` | `max_num_seqs` | `max_num_batched_tokens` |
| --- | ---: | ---: | ---: |
| PAP Prefill on PA | 0.90 | 64 | 16,384 |
| PAP Attention | colocated outside vLLM budget | - | - |
| PAP Projection | legacy explicit budget | 64 | 64 |
| PD Prefill | 0.90 | 64 | 16,384 |
| PD Decode | 0.90 | 64 | 64 |

PAP used 3PA1P with the static 72/20-SM Prefill/Attention split. PD used
one-way KV transfer and tested 1P3D, 2P2D, and 3P1D. Every matrix point
restarted all services. The default `max_num_partial_prefills=1` remained
unchanged.

Each PA Prefill executor obtained 22.97 GiB of KV cache, or 167,264 tokens.
Peak observed PA KV usage rose from 60.8% at C12 to 89.0% at C32. All four PAP
points started and completed without OOM, showing that 0.90 avoids an
artificial PA KV constraint while retaining enough memory for colocated
Attention. Automatic Projection sizing was introduced after this run; the
superseded manual value is not a current configuration input.

## Validity

The primary matrix ran nine points. A targeted repeat added one PD 3P1D C8
point after the primary run showed an extreme KV-transfer tail. Every run
completed all 320 requests and passed output-length, routing,
conversation-affinity, KV-handoff, and lifecycle-drain audits. The evidence
therefore contains 3,200 valid request records and no partial point.

This remains development evidence: the primary matrix has one repetition per
point, plus one targeted repeat. It is not a three-repetition release result.

## Primary matrix

| Architecture | Topology | C | TTFT p95 ms | ITL p95 ms | Req/s | Strict | Standard | Relaxed |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| PAP | 3PA1P | 12 | 2,844.64 | 39.52 | 2.559 | pass | pass | pass |
| PAP | 3PA1P | 20 | 8,155.34 | 51.20 | 3.416 | fail | pass | pass |
| PAP | 3PA1P | 28 | 11,254.81 | 62.87 | 3.463 | fail | fail | pass |
| PAP | 3PA1P | 32 | 14,620.49 | 74.96 | 5.000 | fail | fail | pass |
| PD | 1P3D | 8 | 16,670.34 | 30.05 | 1.081 | fail | fail | pass |
| PD | 2P2D | 10 | 7,253.60 | 32.59 | 1.867 | fail | pass | pass |
| PD | 2P2D | 16 | 13,013.98 | 36.18 | 2.612 | fail | fail | pass |
| PD | 2P2D | 20 | 26,718.25 | 39.60 | 2.363 | fail | fail | fail |
| PD | 3P1D | 8 | 32,564.13 | 32.71 | 0.432 | fail | fail | fail |

The matrix stopped 2P2D above C20 and 3P1D above C8 after their first valid
Relaxed-SLO failure, as required by the fixed lean-scan policy.

## PD 3P1D transfer variance

The primary 3P1D C8 point was complete and correct but took 740.78 seconds.
Decode KV usage stayed below capacity while individual transfer windows
reached 329.77 seconds and throughput fell to 1.22 MB/s. The same committed
code and byte-identical dataset were therefore repeated once:

| Run | TTFT p95 ms | ITL p95 ms | Req/s | Strict goodput | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Primary matrix | 32,564.13 | 32.71 | 0.432 | 0.313 | fail |
| Targeted repeat | 4,416.44 | 34.35 | 1.833 | 1.770 | pass |

The repeat's transfer throughput stayed between 201 and 443 MB/s and its
largest aggregate transfer window was 5.02 seconds. This establishes large
run-to-run variance in the current PD 3P1D transfer path. The comparison below
uses the better repeat as the PD result, which is conservative with respect to
the PAP advantage; the primary anomaly remains preserved rather than hidden.

## Capacity and compliant goodput

Using the best observed complete PD result across the primary matrix and the
targeted repeat:

| SLO | PAP capacity | Best PD capacity | PAP goodput | Best PD goodput | PAP over PD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict | C12 | C8, 3P1D | 2.455 | 1.770 | +38.7% |
| Standard | C20 | C10, 2P2D | 3.277 | 1.816, 3P1D C8 | +80.4% |
| Relaxed | C32 | C16, 2P2D | 5.000 | 2.481 | +101.5% |

The relaxed conclusion does not depend only on the unusually high C32 point:
PAP C28 still delivers 3.376 compliant req/s, 36.1% above the best PD relaxed
goodput.

## Conclusion

With the PA Prefill memory budget raised to 0.90, PAP has a clear advantage in
this fixed four-GPU testbed. It supports 50% more strict concurrency, twice the
standard concurrency, and twice the relaxed concurrency of the best observed
PD configuration. Best compliant goodput improves by 38.7%, 80.4%, and 101.5%
for the strict, standard, and relaxed tiers respectively.

The result also separates capacity from transport instability: PAP uses its
larger KV budget up to 89% without failure, while the worst PD behavior occurs
with available Decode KV and is caused by highly variable NIXL transfer time.
PD 3P1D should therefore be treated as unstable until its transfer variance is
diagnosed; this does not invalidate the conservative best-observed comparison.

The tracked PAP and PD run manifests preserve workload, topology, metrics, and
artifact identities. Machine-local raw evidence remains colocated under the
ignored `runs/` raw directory when available; this report does not link to an
untracked path.
