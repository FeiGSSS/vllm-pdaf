# A4: cost of carving Attention SMs out of each Prefill GPU

Request: `paper/handoff/requests/R-004-phase3-ablations-tuning.md` (A4).

## Question

How much do TTFT, ITL and throughput change when each PA gives Attention more
or fewer of the GPU's 92 SMs (static MPS partition, 4 SMs per chunk)?

## Protocol

- PAP 7PA1P, KV migration on, fixed PAT, `s60-t3-native32k-stratified-seed42`,
  closed-loop C30 (`pap.env`, the COMMON §2.3 template).
- Variants (`PAP_STATIC_{PREFILL,ATTENTION}_CHUNKS` and the expected SM
  counts): `sm84-8` (21/2), `sm80-12` (20/3, baseline), `sm76-16` (19/4),
  `sm72-20` (18/5).
- `run_round.sh <k>` runs all four once; three rounds (12 runs).

## Criterion (from R-004)

C18: 80/12 raises TTFT by at most 10% relative to 84/8.

## Results

Pending.
