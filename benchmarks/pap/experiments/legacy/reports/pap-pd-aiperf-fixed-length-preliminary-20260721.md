# PAP/PD AIPerf fixed-length comparison (archived)

## Status

These 2026-07-21 runs are archived as preliminary diagnostics. They must not
be used as the PAP/PD capacity baseline or as evidence for a performance
claim. The input document, every follow-up, and every requested output had a
fixed length, which made the workload more synchronized and less representative
than the intended AIPerf workload.

The raw artifacts are preserved locally under
[`legacy/capacity/20260721_fixed_length_preliminary`](../capacity/20260721_fixed_length_preliminary/):

| Directory | Shape | Coverage |
| --- | --- | --- |
| `o256_s96` | 96 sessions, O256 | Aborted pilot; only PD 2P2D C10 is complete and valid. |
| `o256_s32` | 32 sessions, O256 | Complete exploratory PAP/PD boundary scan. |
| `o128_s32` | 32 sessions, O128 | Partial; PAP points completed, first PD point was interrupted. |

No raw data was deleted. The directory is intentionally Git-ignored because
the preserved artifacts occupy about 420 MiB; this report is the tracked
index and decision record.

## Diagnostic result retained for provenance

The complete O256/S32 scan observed the following largest passing tested
concurrency. These values explain why the next scan region was chosen, but are
not baseline results:

| SLO | PAP 3PA1P | Best tested PD | Difference |
| --- | ---: | ---: | ---: |
| Strict | 12 | 8 (3P1D) | +4 |
| Standard | 20 | 12 (3P1D) | +8 |
| Relaxed | 24 | 14 (3P1D) | +10 |

The O128 run stopped after PAP C12/C16/C20/C24/C28. PAP passed through C20 at
the standard tier and C24 at the relaxed tier; no PD comparison completed, so
that partial run has no comparative conclusion.

## Replacement workload

The successor keeps the same ten-turn structure, mean 8,192-token initial
document, mean 512-token follow-up, and think/tool schedule. It changes all
three lengths to reproducible AIPerf-style log-normal samples parameterized by
explicit mean and median. The first fast comparison uses mean output length 32
tokens. Its manifest records configured and sampled mean, median, standard
deviation, bounds, percentiles, tokenizer-measured user lengths, cumulative
input estimates, random seed, and context headroom.

The successor is valid only if the generated dataset is non-degenerate, its
sample means remain within the configured tolerance, every request stays below
the 20K context limit, and each actual response exactly matches that request's
sampled output length.
