# Legacy PAP evidence

Everything below this directory is historical evidence. Words such as
“current”, “default”, “next”, and “TODO” inside a dated report refer to the
state when that report was written. They do not override the
[current PAP development status](../../../../docs/design/pap/status.md).

## Retained groups

- `reports/` contains dated narrative reports that predate normalized
  experiment manifests. Each preserves unique measurements, rejected paths,
  or design rationale.
- `runs/` contains small tracked diagnostic notes and ignored raw artifacts.
- `capacity/` contains ignored early capacity artifacts.
- The shared 27-GiB archive remains under
  `/home/fei/research/PD/test/baseline/pap/results/runs`; its tracked pointer is
  [the legacy archive README](../../../../test/baseline/pap/README.md).
- Six pre-migration May/June bundles remain colocated with multi-gigabyte
  ignored logs and profiles under `benchmarks/disagg_benchmarks/results/`:

  | Bundle | Evidence state |
  | --- | --- |
  | [32B O4 projection profile](../../../disagg_benchmarks/results/pap_projection_profile_32b_ctx32_bs512_o4_mbt16384_mem076_20260528_121535/README.md) | Historical short-output diagnostic |
  | [32B O64 asynchronous profile](../../../disagg_benchmarks/results/pap_projection_profile_async_fixed_32b_ctx32_bs512_o64_mbt16384_mem076_20260528_123959/README.md) | Historical CUDA-event diagnostic |
  | [32B O64 partial timeline](../../../disagg_benchmarks/results/pap_projection_timeline_32b_ctx32_bs512_o64_mbt16384_20260528_112442/README.md) | Invalid partial run; failure evidence only |
  | [32B 3P1D/3PA1P TP=2 exploration](../../../disagg_benchmarks/results/qwen3_32b_pd_vs_pap_ctx128_bs512_20260528/README.md) | Historical mixed valid/invalid comparison |
  | [Early 32B TPOT breakdown](../../../disagg_benchmarks/results/pap_tpot_breakdown_32b_20260630/README.md) | Historical root-cause diagnostic |
  | [Consolidated 32B PAP/PD breakdown](../../../disagg_benchmarks/results/pap_pd_tpot_breakdown_32b_20260630/README.md) | Historical optimization notebook |

Three narrative reports are intentionally standalone rather than experiment
manifests: the [May progress report](reports/pap-weekly-report-2026-05-20_2026-05-28.md),
the [four-GPU pilot](reports/pap-pd-4gpu-3to1-pilot-20260716.md), and the
[bilateral trace design](reports/pap-bilateral-deferred-trace-design-20260713.md).
They are retained for project narrative or experimental rationale, not as
current conclusions.

Historical design specifications and implementation plans remain under
[`docs/superpowers/`](../../../../docs/superpowers/README.md). They explain
how a decision was reached, but they are not active implementation plans.
The former `.claude/skills/vllm-pap-benchmark/` directory has been deleted;
historical command references resolve through Git history, not an active skill.

## Retention rules

1. Do not rewrite historical metrics or conclusions to match current code.
2. Repair broken links and add archival notices when a document can be
   mistaken for a current default.
3. Do not delete a report that is the only tracked explanation of a raw run,
   failed path, or decision.
4. Put every new experiment in a normalized `PAP-*` bundle and update the
   generated [experiment index](../INDEX.md).
