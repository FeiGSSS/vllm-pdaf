# A2: contribution of cross-PA Projection aggregation

Request: `paper/handoff/requests/R-004-phase3-ablations-tuning.md` (A2).

## Question

Projection is stateless, so one Projection GPU batches the decode steps of all
PAs. How much do ITL and throughput depend on that aggregation?

## Protocol

- PAP, KV migration on, fixed PAT, `s240-t3-native32k-stratified-seed42`
  (720 requests), closed-loop C30 and C50 (`pap.env`).
- Variants per concurrency: `base` (7PA1P, `PAP_PROJECTION_MAX_NUM_SEQS=256`),
  `seqs64`, `seqs32` (Projection batch cap), `6pa2p` (GPUs 0–5 PA, 6–7
  Projection), `4pa4p` (0–3 PA, 4–7 Projection).
- `run_round.sh <k>` runs all ten points once; two rounds (20 runs).
- One traced 7PA1P C30 run (`run_trace.sh`, variant `trace-c30`):
  `PAP_PROJECTION_PA_TRACE_OUTPUT` plus `capture_aligned_trace.py` freezing an
  aligned PA/Projection window while serving, for Projection step time
  versus batch size. Not used for the throughput comparison.

## Results

Pending.
