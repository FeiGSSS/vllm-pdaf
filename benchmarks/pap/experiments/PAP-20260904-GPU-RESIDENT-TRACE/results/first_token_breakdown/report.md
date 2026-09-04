# Projection first-token pipeline latency

## Question

Why does the interval from submitting a Projection decode request to its first
streamed token approach three steady-state TBT periods?

## Controlled comparison

Both valid runs use 7PA1P, the same 60-session/3-turn Agentic Coding dataset,
2K Prefill token budget, Poisson 0.9 requests/s, and concurrency 60. All 180
requests completed without errors.

| Measurement (mean) | Async off | Async on |
| --- | ---: | ---: |
| EngineCore queue | 0.007 ms | 0.007 ms |
| First schedule to EngineCore output | 57.94 ms | 92.34 ms |
| Outside EngineCore | 41.96 ms | 39.54 ms |
| Projection request to first chunk | 99.90 ms | 131.89 ms |
| Steady-state TBT | 47.61 ms | 43.03 ms |
| First-chunk interval / TBT | 2.10x | 3.07x |
| Output throughput | 280.08 token/s | 289.16 token/s |

`Outside EngineCore` is the request-to-first-chunk interval minus the interval
from the EngineCore `QUEUED` event to creation of the first EngineCore output.
It includes Projection API input preparation, API-to-EngineCore delivery and
step-boundary admission, plus EngineCore-to-SSE output delivery.

## Root cause

The coarse A/B established that async scheduling adds 32--34 ms, but did not
explain the complete interval. A second clean async-on run recorded every
request at the API, EngineCore input, first schedule, batch-queue pop, future
completion, EngineCore output queue, API output processor, and first SSE chunk.
All 180 requests have a complete joined timeline.

| Detailed async-on segment (mean) | Time |
| --- | ---: |
| HTTP ingress and SSE egress outside `AsyncLLM.generate` | 7.079 ms |
| API request preparation and send | 0.786 ms |
| Engine input wait before handling `ADD` | 20.058 ms |
| Engine `ADD` handling to first schedule | 0.198 ms |
| First schedule to the Core's pop attempt | 47.157 ms |
| Pop attempt blocked until the future completed | 43.488 ms |
| Future completion to EngineCore output queue | 0.551 ms |
| EngineCore-to-API IPC | 0.470 ms |
| API output processing | 9.020 ms |
| Output processing to request coroutine resumption | 0.497 ms |
| **Projection request to first chunk** | **129.305 ms** |

The same run's steady-state TBT was 42.594 ms, so the first-chunk interval was
3.036 TBT periods. At the Core's first pop attempt, the future was incomplete
for all 180 requests.

This proves that the result was not complete and then held in an output queue.
The exact mechanism is:

1. A request sent while an EngineCore step is running waits an average of
   20.058 ms before the Core handles `ADD`.
2. vLLM V2 async scheduling permits two concurrent batches. The new batch is
   submitted behind an older in-flight batch. The Core comes back to pop the
   new batch after 47.157 ms.
3. At that point the new batch's future is still incomplete. The Core blocks
   another 43.488 ms for its work to finish.
4. API, IPC, output processing, and SSE account for the remaining 18.602 ms.

The exact measured sum is:

```text
20.058 ms EngineCore input wait
+ 47.157 ms older in-flight batch turnover
+ 43.488 ms current batch future completion
+ 18.602 ms API, IPC, output processing, and SSE
= 129.305 ms request-to-first-chunk

129.305 / 42.594 ms steady TBT = 3.036
```

Turning asynchronous scheduling off removes roughly one pipeline period from
first-token latency: scheduled-to-output falls by 34.41 ms and the total gap
falls by 31.98 ms. It also worsens steady-state TBT by 4.58 ms and reduces
output throughput by 3.14%, confirming that this is the latency cost of the
throughput optimization.

Once the pipeline is full, later outputs remain one period apart. The three
periods are the first-token pipeline residence of a newly joining request, not
three executions of its token and not three extra periods between later output
tokens.

## Artifacts

- `async_off_attempt_002/`: valid async-off run.
- `async_on_clean/`: valid async-on run.
- `pipeline_stages_async_on/`: valid seven-stage async-on trace.
- The failed async-off startup, the 3PA1P fallback, and the run contaminated by
  its orphan AIPerf client were invalidated and removed during cleanup.

The runner now accepts legacy effective-config topology keys, emits canonical
`PAP_TOPOLOGY`, `PAP_PA_COUNT`, and `PAP_PROJECTION_COUNT` keys, rejects
topology/count disagreement, and audits the requested Projection scheduling
mode against the mode reported by every Projection worker.
