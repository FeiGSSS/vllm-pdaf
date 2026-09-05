# PAP refactor validation

Purpose: exercise the refactored PAP launcher, gateway, KV lifecycle, PAT/Triton
selection, NVSHMEM communication and whole-step CUDA Graph with every currently
registered dataset file. This is a correctness/regression lane, not a new fair
architecture performance comparison.

## Protocol

`experiment.env` pins 7PA1P, Dynamo routing, 2K Prefill budget, static MPS,
Qwen3-8B FP16 and YaRN 131K. `workloads.tsv` columns are case name, dataset
relative to the registry, AIPerf format, sessions, available requests, concurrency
and measured duration (zero means complete the replay).
An optional final column enables tracing (`1`); absent means disabled.

All three long-context fixtures and both 60-session coding workloads run to
completion. The full 2,092-session/16,049-request dataset runs for 600 seconds
after AIPerf starts measurement, with no warmup; it tests timed cancellation and
draining. It does **not** claim every turn of that full dataset was executed.
Use the saved client records to determine actual coverage.

```bash
bash benchmarks/pap/experiments/e2e/PAP-20260905-REFACTOR-VALIDATION/run.sh
```

Optional case names run only those cases, in argument order; a subset is not a
complete validation suite. The wrapper clears inherited experiment variables, and the
driver refuses to start if GPU processes are present or the hardware differs.
Each invocation writes a new directory under `runs/`; it never resumes or
overwrites an old result. Cases execute sequentially and stop on failure.

`coding-half-trace` is an additional diagnostic replay of the same 180 requests.
It collects PA kernel and Projection-side per-layer tensors to verify the
refactored trace recorder. Its timing must not be compared as an uninstrumented
performance result. Its protocol fixes a 2,048-step ring, a target of 512
consecutive aligned samples and a five-second export interval. A background
collector freezes matching raw PA/Projection files under `trace_capture/` and
validates the join before declaring success. Raw files retain the full ring to
allow alignment; the merged tensor has 512 samples. This capture must happen
before drain overwrites the all-PA-active window. The collector is part of the
case lifecycle and is stopped when a case fails.

## Evidence and reproduction

Each suite stores its scripts/configuration, selection, raw launch logs, actual
post-startup topology and lifecycle/Graph audits, AIPerf records, and exit codes.
`provenance/` contains a working-tree source archive (including nonignored new
files), Git revision/diff, package inventories, GPU/CPU/driver details, model
file checksums and model metadata. A `COMPLETE` marker applies only to the cases
in `selected_cases.txt`, not necessarily all six files.

To reproduce elsewhere, check out the commit in `provenance/git_commit.txt`,
apply `provenance/source.patch` (including recorded deletions), then extract
`provenance/source.tar.gz` over that checkout to restore nonignored new files.
Simply unpacking over an arbitrary checkout can leave retired files behind.
Recreate the recorded environments, obtain model weights matching
`model_files.json`, and use the archived experiment entry point. Set the model
location in a new copy of `experiment.env` if needed and preserve that change as
a new run. See `benchmarks/pap/scripts/RUNTIMES.md` for NIXL/NVSHMEM requirements.
The archive does not contain Python environments, model weights, drivers or
communication build products. These remain explicit external dependencies;
package inventories alone are not guaranteed binary-compatible lockfiles.

Status: the initial `short-context` case completed 14 requests and drained, but
was invalidated by a manual KV-event audit. See
`runs/20260905_105242_2939264/INVALIDATED.md` and the CPU reproduction under
`experiments/microbench/PAP-20260905-DYNAMO-CACHE-SALT/`. The three long-context
fixtures use per-session `cache_salt`, which the current Dynamo selector does
not support. Preflight now rejects that combination; do not remove the salt or
override routing merely to make this protocol pass. Resolving this compatibility
boundary is required before claiming completion of the all-dataset suite.

Completed checkpoints and their exact validation scope are in [results.md](results.md).
