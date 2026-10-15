# PAP main-result figures

All figures are vector PDF (Type-42 fonts, sized for a two-column template:
single column 3.35 in, full width 7.0 in). PNG previews (300 dpi) are in `png/`.

| File | Width | Content |
| --- | --- | --- |
| `fig_main_comparison.pdf` | ~1.35 col | Mean TTFT / ITL / throughput: PAP 7PA1P vs PD 6P2D, PD 7P1D, DP8 |
| `fig_main_comparison_all.pdf` | full | Same, all six architectures (PD 2P6D TTFT clipped at 16 s, value labelled) |
| `fig_latency_cdf.pdf` | full | Per-request CDFs of TTFT, ITL, E2E (360 requests each) |
| `fig_tradeoff.pdf` | 1 col | Mean TTFT (log) vs mean ITL, marker area = throughput |
| `fig_ttft_per_turn.pdf` | 1 col | Mean TTFT by conversation turn (log y) |
| `fig_goodput_slo.pdf` | full | Goodput at diagnostic SLOs D1–D4; filled = feasible (≥95% attainment), hollow = infeasible; attainment % on bars |
| `fig_slo_attainment_sweep.pdf` | full | SLO attainment vs TTFT SLO at ITL ≤ 50 / 75 ms (TTFT+ITL only, no E2E bound) |

## Data and provenance

Repo `FeiGSSS/vllm-pdaf`, branch `feature/pap-v026`. Qwen3-8B FP16, 8×L20, native 32K,
60 sessions × 3 turns, closed-loop concurrency 30, 2 independent starts per
architecture (360 requests).

- PAP 7PA1P + KV migration: `PAP-20260909-KV-MIGRATION-ARCHITECTURE-C30` (r7, r8)
- DP8, PD 6P2D/4P4D/2P6D: `PAP-20260907-ALIGNED-C30`
- PD 7P1D: `PAP-20260907-ALIGNED-C30-7P1D`

`data/summary.json` and `data/requests.csv` are produced from the raw AIPerf
archives by the experiment's own `analyze.py` (copied as
`scripts/analyze_upstream.py`); every mean, tail and goodput value matches
`report.md` in the KV-migration experiment.

Encoding: PAP orange; DP8 gray; PD xPyD one blue ramp (light→dark = more Decode GPUs), hatched/dashed.

## Regenerate

```bash
scripts/make_data.sh /path/to/vllm-pdaf origin/feature/pap-v026   # rebuild data/
python scripts/plot_main.py                                        # needs matplotlib, numpy
```

## Caveats

- The outline thread reports an unmerged PAT kernel split-rank fix that may change the
  main-table numbers after a rerun. Plots read only `data/`, so after a rerun,
  point `make_data.sh` at the new archives (edit the run list) and re-plot.
- The goodput figure uses the PAP + KV-migration runs (report.md table). The older
  `GOODPUT.md` uses the pre-migration PAP runs (D2 2.775 vs 3.079 req/s here).
- Only C30 is measured, with 2 runs per point: these are fixed-load results, not maximum sustainable goodput.
