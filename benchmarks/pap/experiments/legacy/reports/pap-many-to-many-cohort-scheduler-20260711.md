# PAP 多对多 Cohort 与 Attention 聚合调度重构计划

日期：2026-07-11

状态：Phase 0/1/2/3 与 Phase 4 等待观测已提交；active-source membership 已完成
严格 A/B 与 x:y smoke，等待提交后 clean 复跑

代码基线：`feature/pap @ 8cb6e2022b1ba3dce80a2dbc9a46b2ab3e7d7ab6` 加
active-source tracked patch

关联文档：

- `benchmarks/pap/experiments/legacy/reports/pap-tpot-attention-projection-dataplane-20260710.md`
- `benchmarks/pap/experiments/legacy/reports/pap-xpayp-multiturn-kv-affinity-20260710.md`

## 0. 结论先行

本次重构不应重写 Projection 端的 vLLM scheduler。源码已经证明，每个
Projection 实例当前就是一个 continuous-batching cohort；async scheduling 的两个
在途 `SchedulerOutput` 是同一 cohort 在相邻 token step 上的两个时间版本，不是两个
互斥 batch。

真正缺失的是以下两层：

1. **PA 端缺少全局多来源调度器。** 当前每个 Projection peer 对应一个 mailbox
   thread，线程收到一个 batch 后立即独立执行 KV append、FlashAttention 和 output
   send；不同 peer 的请求不会 combine。
2. **P 端缺少多 PA 友好的行布局。** 一个 Projection cohort 包含来自多个 PA 的请求
   时，route group 可能是不连续行，direct QKV 会退化为 gather，Attention output
   还会逐行 scatter。

推荐目标架构是：

```text
每个 Projection：一个动态 Cohort
  - 完整 batch 做 QKV/O/MLP projection
  - 按 PA 划分连续 route slices
  - fan-out QKV
  - gather Attention output

每个 PA/TP rank：一个全局 AttentionWorkScheduler
  - 多 peer ingress
  - 按兼容性 combine
  - 一次 batched KV append + FlashAttention
  - 按来源 scatter output
```

第一阶段只支持**跨 Projection、跨 request decode step、同 layer** 的 combine。
跨 layer 的任务可以同时进入全局 ready queue，但不能直接塞进当前 FlashAttention
调用。原因不是 Attention 的数学依赖，而是当前 paged-FA ABI 只接受一个 KV cache
base；不同 layer 使用不同 KV tensor。真正的跨 layer 单 kernel batching 需要
pointer-indirect KV append/Attention backend，必须作为 profiler 驱动的后续阶段。

第一功能目标固定为 **2PA2P full crossbar**：

- P0、P1 的 cohort 都能同时包含 PA0、PA1 的请求；
- PA0、PA1 都能同时接收 P0、P1 的 batch；
- 一个 request 在整个 turn 内固定绑定一个 `(PA, P)` pair；
- 新请求只在 P 的完整 forward 边界加入 cohort；
- 不创建新请求专属的 Projection forward；
- 保持 1PA1P 正确性和 TPOT 基线。

## 1. 目标与非目标

### 1.1 功能目标

1. 让 `2PA2P` 在真实 full-crossbar 请求分布下稳定运行，而不是只覆盖
   `PA0-P0`、`PA1-P1` 两条对角连接。
2. 每个 P 只维护一个动态执行 cohort；不同请求的 decode step 可以不同，但同一次
   P forward 的所有行共同执行 layer 0 到最后一层。
3. 每个 PA 对所有 P peer 维护统一 ready queue、combine policy 和 scatter path。
4. cohort membership 可以在相邻完整 forward 之间增删；在一个 forward 的 36 层内
   保持不变。
5. 保留任意正整数 `xPAyP`，不要求 `x`、`y` 整除。
6. 为后续 conversation resident KV 和多轮 attach 保留稳定 session/generation
   接口，但本轮不同时实现 resident KV。

### 1.2 性能目标

1. 1PA1P median TPOT 相对 `45c302bb3` 回退不超过 3%。
2. 1PA2P QPS 4 的 median TPOT 至少比当前 `53.67 ms` 改善 25%。
3. stretch 目标是让 1PA2P QPS 4 恢复到 1PA1P 的 1.20 倍以内。
4. 2PA2P full-crossbar 在同负载下，相对 2PA2P diagonal 的 median TPOT 回退不超过
   10%，request/output throughput 不低于 95%。
5. 不通过关闭 async scheduling、缩短输出、改变模型或扫描 MPS 比例制造表面收益。

### 1.3 本轮非目标

- 不重写 vLLM 的 token scheduler。
- 不建立跨 Projection 的全局 decode-step barrier。
- 不要求不同 P 的 request step 编号相同。
- 不在第一版实现跨 layer 单 FlashAttention kernel。
- 不恢复历史 runner ubatch/microbatch 路径。
- 不实现 conversation resident KV、final-token closure 或跨 PA KV migration。
- 不访问 Hugging Face；全部使用本地模型。
- 不做 MPS 扫描，固定使用当前验收配置。

## 2. 当前基线与实验事实

### 2.1 严格 1PA1P/1PA2P A/B

以下运行均使用：

- commit `45c302bb3`；
- clean tracked worktree；
- Qwen3-8B 本地模型；
- sonnet `i128/o32/prefix50/prompts128/warmup0`；
- `MAX_MODEL_LEN=512`、`MAX_NUM_SEQS=64`；
- local-fast、stream ordered、double slot；
- Prefill/Attention MPS `70/30`；
- strict correctness audit 与 session drain；
- async scheduling enabled。

| 负载 | 1PA1P median TPOT | 1PA2P median TPOT | 退化 |
| --- | ---: | ---: | ---: |
| QPS 4，三轮中位数 | `28.19 ms` | `53.67 ms` | `1.904x` |
| QPS 16，三轮中位数 | `32.67 ms` | `72.28 ms` | `2.212x` |

QPS 4 的六次运行全部 `128/0`。QPS 16 的 1PA2P 有一次 `126/2`，正式中位数使用
对应的 clean retry `128/0`。有效运行全部通过 correctness audit，Attention session
最终为 0。

原始结果：

```text
/home/fei/research/PD/test/baseline/pap/results/runs/
  20260711_ab_localfast_q4_1pa1p_rep{1,2,3}
  20260711_ab_localfast_q4_1pa2p_rep{1,2,3}
  20260711_ab_localfast_q16_1pa1p_rep{1,2,3}
  20260711_ab_localfast_q16_1pa2p_rep1
  20260711_ab_localfast_q16_1pa2p_rep2_retry1
  20260711_ab_localfast_q16_1pa2p_rep3
```

QPS 4 时总请求到达率没有增加，只是把 Projection 从一个拆成两个。PA 的总 Attention
工作量理论上基本不变，但 TPOT 接近翻倍。这说明主要退化来自 batch 被 peer 边界切碎、
小 kernel/metadata/doorbell 重复和两个独立 producer 的节奏竞争，而不是新增了两倍有效
Attention FLOPs。

### 2.2 当前 xPAyP smoke 的解释边界

`1PA2P`、`2PA1P`、`2PA2P`、`3PA2P` 已通过短输出 correctness smoke，证明：

- per-session Prefill control endpoint 正确；
- 一个 Attention 可以绑定多个 Projection transport；
- actor id 不冲突；
- session/lease 可以 drain。

但它们不能证明多来源调度已经完成：

- `2PA2P + round_robin` 中 `pa=n%2`、`p=n%2`，只生成对角 pair；
- 多 peer bind 测试只检查创建了两个 transport/thread；
- 没有测试两个 peer 的输入被组合成一次 FA；
- 没有测试一个 P forward 同时包含两个 PA route group；
- 没有 full-crossbar 性能基线。

## 3. 源码审计

### 3.1 Proxy：pair 选择是 request 级，但 2x2 默认不覆盖 crossbar

`examples/pap/multi_pap_proxy_server.py::select_instances()` 当前独立计算：

```text
PA index = request_number % PA_COUNT
P  index = request_number % PROJECTION_COUNT
```

当两者都是 2 时只产生 `(PA0,P0)` 和 `(PA1,P1)`。`3PA2P` 因周期不相同，反而会
自然覆盖更多 pair。这也是当前 `2PA2P` smoke 看起来正常、却没有覆盖目标业务流的原因。

选定 pair 后，proxy 会把 PA endpoint 放进 Projection request metadata；request 的
Prefill 和整个 decode 期间 pair 不再改变。这部分符合目标语义。

### 3.2 P scheduler：已经是需要的 dynamic cohort

`vllm/v1/core/sched/scheduler.py::schedule()`：

1. 先调度 RUNNING request；
2. 再从 WAITING queue 加入新 request；
3. model runner 维护 persistent `InputBatch`。

因此正常 decode budget 足够时：

```text
Plan t   = {A(step i),   B(step j)}
Plan t+1 = {A(step i+1), B(step j+1), C(step 0)}
```

不存在 `{A,B} -> {C} -> {A,B}` 的默认轮转。

`AsyncScheduler` 通过 output placeholder 允许同一 RUNNING request 在前一输出尚未回到
CPU 时进入下一 plan。PP=1 时 `max_concurrent_batches=2`，所以系统维护的是：

```text
一个正在执行的 cohort plan
+ 一个已经 committed 的下一 cohort plan
```

新请求如果错过已 committed plan，会晚一个 forward 加入，主要影响其首 token/TTFT；
它不会在已有 request 之间插入一个独占 forward。

### 3.3 Model runner：已经按 PA 构造 route group

`vllm/v1/worker/gpu_model_runner.py::_pap_offload_exec_route_groups_for_request_ids()`
在每个 scheduler/model forward 建立：

```text
(attention_endpoint, offload_exec_endpoint)
  -> req_indices, request_ids, per-request steps
```

`vllm/model_executor/models/qwen3.py::_pap_offload_exec_step_groups()` 在第一层解析一次，
后续层复用。这部分已经避免了 36 层重复展开 request route。

每个 group 的 `steps` 可以不同，所以**同 layer、跨 request step** 在当前 descriptor 和
Attention compute 中已经是合法语义。

### 3.4 P model：projection 是全 batch，但多 PA 数据搬运会退化

Qwen3 每层先对整个 hidden-state batch 执行：

- QKV projection；
- Q/K norm 与 RoPE；

然后 `_compute_pap_attention()`：

1. 为每个 PA route group 发送 QKV；
2. 所有 group 发送完成后逐 group 接收 output；
3. 重建完整 row order；
4. 对完整 batch 做一次 O projection，再继续 MLP。

这个总体方向正确，Projection GEMM 没有被拆成 PA-specific 小 GEMM。

但 `_pap_direct_qkv_batch_for_indices()` 只接受连续 `req_indices`。如果 cohort 行按到达顺序
交织 PA0/PA1，请求组会退回 `torch.cat` gather。多 group output 路径当前还会在 Python
循环里逐 row `copy_`，并禁用 single-group direct output。

### 3.5 PA executor：多 peer 连接存在，但没有 combine

commit `45c302bb3` 把：

```text
app.state.offload_exec_transport
```

扩展为：

```text
app.state.offload_exec_transports[peer_key]
```

每个 peer bind 后启动一个：

```text
run_offload_exec_mailbox_loop(registry, peer_transport)
```

该 loop 的行为是：

```text
recv one peer batch
-> compute_offload_exec_batch_output()
-> send output to the same peer
-> repeat
```

因此两个 Projection 同时服务一个 PA 时，PA 上存在两个 producer/consumer loop，但没有
共享 ready queue、batch compatibility 判断或 scatter plan。

两个线程默认都在同一 Attention GPU 上提交 append/FA 工作；registry lock 只部分串行化
准备，GPU kernel 仍以两个小 batch 分别提交。这正是 1PA2P TPOT 退化的结构根因。

### 3.6 当前 FA backend 的 layer 边界

`compute_offload_exec_batch_output()` 要求 descriptor 只有一个 `layer_name`。

`_compute_unified_paged_flash_batch()` 取：

```text
base_kv = states[0].kv_cache
```

随后整批 FA 共用这个 key/value cache base 和一张 block table。不同 request 在同一 layer
上共享该 layer 的全局 cache tensor，所以合法；不同 layer 的 cache tensor base 不同，
不能只靠合并 metadata 混入同一个现有 FA 调用。

因此第一版兼容键必须包含 `layer_ordinal`。如果未来要真正 cross-layer batching，需要：

- 每 row KV base pointer；
- pointer-indirect append；
- pointer-indirect paged Attention，或重新设计为 layer-major 统一 arena；
- 全 layer completion barrier，避免当前“首个 layer commit”过早发布。

### 3.7 现有测试缺口

当前测试覆盖：

- route group 构造和 step-level cache；
- 多 peer transport 建连；
- 单个 peer batch compute；
- local-fast slot/plan/release；
- x/y launcher 与 control endpoint。

当前未覆盖：

- 多 peer 同时 enqueue；
- combine 后只调用一次 append/FA；
- output 按原 peer/原 row order scatter；
- 一个 peer 慢或断开时的 head-of-line/fail-closed；
- async 两个 plan 下 membership generation；
- 2PA2P full Cartesian pair coverage；
- P 端 route-aware contiguous layout。

## 4. 当前与目标执行链路

### 4.1 当前 1PA2P

```text
P0 cohort batch --peer0--> PA thread0 -> append(B0) -> FA(B0) -> P0

P1 cohort batch --peer1--> PA thread1 -> append(B1) -> FA(B1) -> P1
                                      ^
                                      |
                         同一 GPU 上两个小 batch 竞争/串行
```

如果两个 P 的 layer 节奏稍有错位，PA 会持续执行交错的小 batch，后续 layer 很难重新
自然合并。

### 4.2 目标 PA combine/scatter

```text
P0 peer receiver ----+
                     |
P1 peer receiver ----+--> MPSC ready queue
                              |
                              v
                     AttentionWorkScheduler
                      bucket by compatibility
                              |
                              v
                     combined staging batch
                              |
                  one append + one FlashAttention
                              |
                      scatter output slices
                         /             \
                       P0               P1
```

### 4.3 目标 2PA2P

```text
                   +---------------- PA0 scheduler
                   |                 ^          ^
P0 dynamic cohort -+-- route PA0 ----+          |
  {PA0 rows,       |                            |
   PA1 rows}       +-- route PA1 ----+          |
                                      PA1 scheduler
                   +-- route PA0 ---------------+
P1 dynamic cohort -+
  {PA0 rows,       +-- route PA1 ---->
   PA1 rows}
```

每个 P 仍只做一个完整 Projection forward；PA route 是同一 batch 的 row partition，不是
独立 Projection cohort。

## 5. 目标数据模型

### 5.1 Projection plan

建议引入内部逻辑对象 `PAPProjectionCohortPlan`，不立即改变公共 API：

```python
@dataclass(frozen=True)
class PAPProjectionCohortPlan:
    projection_id: str
    plan_generation: int
    membership_digest: str
    request_ids: tuple[str, ...]
    attention_endpoint_by_row: tuple[str, ...]
    route_slices: tuple["PAPRouteSlice", ...]
```

`plan_generation` 是 P-local 的时间序号，不能解释为全局 decode step。不同 P 的 generation
无须对齐。

`route_slices` 应尽量连续：

```python
@dataclass(frozen=True)
class PAPRouteSlice:
    attention_endpoint: str
    start: int
    stop: int
    request_ids: tuple[str, ...]
    request_steps: tuple[int, ...]
```

### 5.2 Attention work item

每个 peer receiver 只负责把 wire message 转为：

```python
@dataclass
class PAPAttentionWorkItem:
    peer_id: str
    peer_generation: int
    transport: object
    descriptor: PAPOffloadExecBatchDescriptor
    qkv_message: object
    qkv_tensor: torch.Tensor
    layer_ordinal: int
    request_ids: tuple[str, ...]
    request_steps: tuple[int, ...]
    session_generations: tuple[int, ...]
    arrival_ns: int
```

`qkv_message` 在 combined compute 已消费相应 tensor 之前不能 release。

### 5.3 Compatibility key

第一版 combine key：

```text
(model_fingerprint,
 tp_rank,
 layer_ordinal,
 q_size,
 kv_size,
 num_heads,
 num_kv_heads,
 head_dim,
 dtype,
 scale,
 kv_layout_fingerprint)
```

request step、seq_len、block table 不需要相同；它们是 per-row metadata。

不同 layer 必须形成不同 bucket。任何 shape、scale、layout 不兼容都立即分开调度，不能
为了扩大 batch 隐式转换。

### 5.4 Scatter plan

combined batch 保留来源 slice：

```python
@dataclass(frozen=True)
class PAPAttentionScatterSlice:
    work_item: PAPAttentionWorkItem
    combined_start: int
    combined_stop: int
```

compute 完成后使用原始 peer descriptor 发送对应 output view。P 端仍按自己的 plan id、
layer 和 row count 校验，不需要知道 PA 内部是否发生 combine。

## 6. P 端 Cohort 语义

### 6.1 必须保持的不变量

1. 每个 P 实例只有一个 persistent execution cohort。
2. 一个 `SchedulerOutput` 确定后，其 membership 在完整 layer 0..N forward 内不变。
3. 所有 eligible RUNNING request 出现在后续 plan；新 request 不能把旧 request 挤出
   一个独立 forward。
4. 新 request 加入 earliest uncommitted plan；async 已预排的 plan 不回滚。
5. request 的 PA/P pair 在一个 turn 内不变。
6. 同一 P forward 的所有 projection GEMM 使用完整 cohort batch。
7. PA partition 只发生在 Attention 通信边界。

### 6.2 Route-aware row layout

推荐使用 endpoint-stable partition，让同一 PA 的 rows 在 persistent `InputBatch` 中连续。
优先复用 `InputBatch.swap_states()` 和现有 batch reorder 机制，一次 scheduler forward
重排，而不是 36 层分别 gather。

推荐顺序：

1. 保留 decode/prefill backend 的既有 region order；
2. 在 PAP decode region 内按 Attention endpoint 做 stable partition；
3. 同 endpoint 内保持原顺序，降低每 step 抖动；
4. 新 request/finished request 导致 membership 变化时增量调整；
5. request id、sampling metadata、block table 和 async previous-position mapping 必须一起
   swap。

如果该方案在 model runner 上风险过高，回退方案是 per-endpoint persistent staging；不接受
每层 Python `torch.cat` 作为最终热路径。

### 6.3 Output gather

route rows 连续后，P 端从逐 row copy 改为每 route slice 一次 `copy_`。完整 Attention
output staging 仍保持一个连续 tensor，随后执行一次 O projection。

长期可让多个 PA 直接写入 P-owned contiguous output arena 的不同 offset，但第一版保留
现有 peer-owned ring，避免同时修改传输所有权协议。

## 7. PA 端调度算法

### 7.1 线程模型

推荐：

```text
N 个 blocking peer receiver thread
  -> 只收消息、解析最小 envelope、enqueue

1 个 Attention compute dispatcher / PA TP rank
  -> 唯一拥有热路径 dispatch 顺序
  -> combine、append、FA、scatter
```

不推荐第一版就把所有 transport 改成单线程 multi-peer poll，因为 NIXL 与 local-fast 当前
都提供 blocking receive，改 transport poll API 会扩大一次提交的故障面。

### 7.2 Opportunistic combine

Phase 1 使用零人工等待的 opportunistic policy：

1. dispatcher 取 ready queue 最老任务；
2. 非阻塞 drain 当前已经 ready 且 compatibility key 相同的任务；
3. 达到 row/byte 上限立即 dispatch；
4. 没有匹配项则单独执行，不等待未来 peer。

这样先证明中心化不会引入 head-of-line，再通过 arrival-skew trace 决定是否加入小窗口。

Phase 2 可增加 bounded coalescing：

```text
deadline = first_arrival + measured_window
```

窗口不做盲目大范围扫描。先测同 layer peer arrival skew，再选择一个上限不超过
`100 us` 的候选，与 window=0 做严格 A/B。

dispatch 条件：

- 已收到所有当前 active peer 的兼容任务；或
- 达到 max rows/bytes；或
- oldest deadline 到期；或
- drain/shutdown；或
- 更早 deadline 的其他 bucket 需要服务。

不能要求“所有 peer 必须到齐”，否则一个慢 P 会阻塞其他 P 的 TPOT。

### 7.3 跨 step

同一 combined batch 可以包含：

```text
P0: request A 的 step i、request B 的 step j
P1: request C 的 step k、request D 的 step m
```

只要 layer/shape/layout 兼容即可。当前 `steps`、slot mapping 和 paged-FA seq lens 已经是
per-row，因此无需让 i、j、k、m 对齐。

### 7.4 跨 layer 的选项

当前有三个技术选项：

| 选项 | 方法 | 优点 | 风险 | 建议 |
| --- | --- | --- | --- | --- |
| A | same-layer combine，超时后单独执行 | 改动可控，复用现有 FA | layer phase 差大时命中率低 | 第一实现 |
| B | 多 stream 并发提交不同 layer 的独立 FA | 不改 FA ABI，可能提高占用 | kernel 并发收益不确定，状态/stream 更复杂 | Phase 4 实验 |
| C | pointer-indirect cross-layer append + FA | 真正跨 layer 大 batch，无全局 barrier | 需要 CUDA/backend、全层 commit、恢复语义 | 长期推荐，profiler gate |
| D | 强制不同 P 在 layer0/full-forward 边界相位对齐 | 不写新 kernel | startup/HOL、跨 PA 协调和死锁风险 | 不作为默认 |

如果 Phase 2 的 same-layer coalesce hit rate 不足，先做 B 的小原型；只有 nsys 证明小 FA
kernel/occupancy 是主瓶颈，才进入 C。不能直接用 D 把独立 P 变成全局 barrier。

## 8. GPU staging 与 slot 所有权

### 8.1 Combined QKV workspace

不同 peer 的 QKV 位于不同 receive ring slot，当前 FA 需要连续 batch。推荐建立 PA-owned
有界 workspace：

```text
workspace[dispatch_slot, max_rows, qkv_width]
```

dispatcher 把每个 work item tensor copy 到连续 slice，然后一次 split/append/FA。

要求：

- workspace 预分配，steady state 不创建 CUDA tensor；
- capacity 超限时切 batch，不临时扩容；
- copy、compute、scatter 在受控 stream 上有明确顺序；
- 每个 input message 的 release 必须排在 staging copy 消费完成之后；
- fallback 可以临时使用 `torch.cat` 做 CPU/unit correctness，但不能进入正式性能默认。

### 8.2 当前 local-fast 的安全过渡

local-fast receive 会在当前 stream 上提交 ready-generation wait；message `release()` 会在
当前 stream 上发布 slot release generation。第一中心化版本可继续使用同一 default
stream，依靠：

```text
recv wait enqueue
-> work queue handoff
-> staging/compute enqueue
-> release enqueue
```

维持顺序。

随后应把 transport message 抽象补成显式：

```text
wait_ready(target_stream)
release_after(target_stream)
```

或返回 CUDA event，避免把跨线程 default-stream 行为作为长期协议。

### 8.3 Scatter 与 output slot

combined output 按 scatter slice 依次调用各 peer transport 的 `send_output_batch()`。
原始 descriptor 必须原样保留，以继续使用 transport-local plan id 和 output 校验。

在所有 peer copy/doorbell 已正确 enqueue 前，不得复用 combined output workspace。

## 9. 状态机与错误处理

每个 work item 状态：

```text
RECEIVED
  -> QUEUED
  -> SELECTED
  -> STAGING
  -> COMPUTE_SUBMITTED
  -> SCATTER_SUBMITTED
  -> INPUT_RELEASED
  -> COMPLETED
```

错误状态：

```text
RECEIVE_FAILED
VALIDATION_FAILED
COMPUTE_PARTIAL
SCATTER_FAILED
PEER_LOST
```

不变量：

1. 每个 peer batch 恰好 enqueue 一次、compute 一次、scatter 一次。
2. input slot 恰好 release 一次，且不早于最后一次读取。
3. output 必须回到原 peer、原 descriptor、原 row order。
4. `(session_generation, layer, step)` 不匹配时 fail closed。
5. 同一 session 的同一 layer/token KV 只能 append 一次。
6. combined dispatch 在 append 后失败时，相关 session 标记 invalid；不能静默重试并重复
   append。
7. peer 失败只应终止涉及该 peer 的 work；但如果 combined kernel 已部分写入多 peer
   session，整个 dispatch 的 session 集都必须 fail closed。
8. shutdown 必须 drain ready/in-flight work，最终 queue、slot、session、lease 为 0。

第一版保留当前进程级 fail-fast 行为，但把错误分类和计数补齐。后续再扩展 transport error
frame，不能在没有 wire error protocol 时假装能透明恢复。

## 10. 2PA2P full-crossbar 路由

新增 opt-in `crossbar_round_robin` policy。推荐生成方式：

```text
pa = request_number % x
block = request_number // x
p = (block + pa) % y
```

在 `x*y` 个请求内，每个 PA 会覆盖全部 P。2x2 前四个 pair 为：

```text
(PA0,P0)
(PA1,P1)
(PA0,P1)
(PA1,P0)
```

它既包含对角 pair，也包含用户关心的交叉 pair。policy 只决定新 request 的 pair；进入
decode 后不迁移。

测试和 benchmark 必须输出 pair matrix：

```text
          P0   P1
PA0       n00  n01
PA1       n10  n11
```

full-crossbar 验收要求四格均大于 0，长期负载下差值不超过 1。

## 11. 通信性能影响

调度重构不会自动让通信变快；它会引入 staging 和排队，同时减少消息消费侧重复计算。

| 项目 | 当前 | 重构后 | 预期 |
| --- | --- | --- | --- |
| P->PA QKV 消息 | 每 peer/layer 一条 | 不变 | 连接数决定，暂不合并 wire |
| PA KV append kernel | 每 peer batch 一次 | 每 combined batch 一次 | 减少 |
| PA FlashAttention kernel | 每 peer batch 一次 | 每 combined batch 一次 | 减少、batch 变大 |
| PA metadata/slot prepare | 每 peer batch 重复 | combined plan 一次 | 减少 |
| PA output 消息 | 每 peer/layer 一条 | 不变 | 必须按来源 scatter |
| PA 本地 QKV copy | 直接消费 peer slot | 增加一次 staging copy | 新开销，需 A/B |
| P 多 PA QKV gather | 非连续时逐层发生 | route rows 连续后消失 | 减少 |
| P output scatter | 当前逐 row | 每 route slice 一次 | 减少 |
| CPU threads | 每 peer 收+算 | 每 peer 收一条后休眠，单 dispatcher 算 | 避免 GIL 抢占 |

此前微基准显示单层 QKV/output P2P copy 只有几十微秒，因此增加一次同 GPU staging copy
有可能被“少一次 append/FA/metadata/kernel launch”的收益覆盖，但必须由同代码 A/B 证明。

## 12. 代码边界

### 12.1 新增核心模块

建议新增：

```text
vllm/pap/attention_scheduler.py
```

包含：

- `PAPAttentionWorkItem`；
- compatibility key；
- ready buckets；
- opportunistic/bounded policy；
- scatter plan；
- stats；
- shutdown/drain；
- 不依赖 FastAPI 的 CPU 单元测试接口。

### 12.2 Attention executor

`examples/pap/pap_attention_executor.py`：

- bind 时注册 peer receiver，不再启动“receive + compute”完整 loop；
- 每个 receiver 最多持有一个已入队任务，随后在 completion event 上休眠，避免
  local-fast Python doorbell busy-spin 与 dispatcher 争用 GIL；
- app 生命周期启动一个 dispatcher；
- 拆分 receive、combine、compute、scatter；
- `compute_offload_exec_batch_output()` 接受内部 combined descriptor；
- 增加 workspace、session generation、dispatch trace；
- legacy mode 保留以便回滚。

### 12.3 Data plane/transport

`vllm/pap/data_plane.py`、`vllm/pap/local_fast_transport.py`、
`vllm/pap/nixl_mailbox.py`：

- 第一阶段保持 wire format；
- message 增加统一 ready/release stream 语义；
- transport stats 区分 peer receive wait、queue wait、staging、scatter；
- 不把多个 peer 强行塞进一个 transport endpoint。

### 12.4 Projection/model runner

`vllm/v1/worker/gpu_model_runner.py`：

- PAP decode row stable partition；
- 构造 `PAPProjectionCohortPlan`/route slices；
- 记录 direct/gather fallback 原因；
- 保持 standard scheduler 和 async batch queue 不变。

`vllm/model_executor/models/qwen3.py`：

- 连续 route slice direct QKV；
- output 每 slice copy；
- plan generation 与 membership stats；
- 保留旧 generic non-contiguous fallback。

### 12.5 Proxy/runner

`examples/pap/multi_pap_proxy_server.py`：

- `crossbar_round_robin`；
- pair id 日志/header；
- pair matrix stats；
- request lifetime pair binding assertion。

benchmark runner：

- 保存 pair matrix、per-peer ingress、combine/scatter stats；
- 审计每个 PA/P pair；
- 保持本地模型、strict correctness、session drain 和 clean tracked baseline。

## 13. 分阶段开发计划

### Phase 0：冻结 contract 与可观测性

行为改动仅允许 opt-in 路由 policy 和指标。

实施：

1. 新增 `crossbar_round_robin` 与 2x2 pair-matrix 单测。
2. 记录 P cohort size、route group 数、route indices 连续率。
3. 记录 PA 每 peer batch rows、layer、arrival timestamp。
4. 记录当前 `FA calls / peer batches`，建立 combine 前对照。
5. 增加 async plan membership contract test：新请求只能加入未来 plan，RUNNING request
   不被新请求独占 batch 打断。
6. 跑 2PA2P full-crossbar 短 smoke，确认当前结构的 correctness baseline。

完成标准：能够用日志证明每个 P 同时服务两个 PA、每个 PA 同时收到两个 P；不声称已经
combine。

### Phase 1：中心化 dispatcher，保持一任务一计算

实施：

1. 新增 `attention_scheduler.py` 数据模型和生命周期。
2. peer thread 只 receive/enqueue。
3. receiver 在 enqueue 后等待该 item 的 completion event；dispatcher 完成
   compute/send/input release 后唤醒它。每 peer 一个 in-flight item 足以支持下一阶段
   多 peer 合批，同时避免空转 receiver 抢占 GIL。
4. receiver stream record CUDA ready event，dispatcher stream wait event，保持跨线程
   QKV ready/release 顺序且不做 CPU synchronize。
5. 单 dispatcher 按 FIFO 取一个任务，调用现有 compute，再 scatter 回原 peer。
6. 增加 legacy/central feature flag 和完整 stats。
7. 不做 deliberate wait，不合并 GPU batch。

目的：先把线程所有权和错误语义改正确，避免把调度重构与 batching 优化混在一个提交。

完成标准：

- focused unit/contract 全过；
- 1PA1P、1PA2P、2PA2P full-crossbar correctness 通过；
- session drain 为 0；
- 1PA1P median TPOT 回退不超过 3%；
- central 与 legacy token 输出一致。

### Phase 2：opportunistic same-layer combine/scatter

实施：

1. compatibility bucket；
2. ready queue 非阻塞 drain；
3. combined descriptor/scatter slices；
4. bounded preallocated QKV/output workspace；
5. 一次 append + FA；
6. per-peer original descriptor output send；
7. partial/incompatible batch fallback。

完成标准：

- 两 peer 同 layer task 被证明只调用一次 append/FA；
- row/token/layer/step 映射逐项正确；
- slot release 顺序测试通过；
- QPS 4 的 1PA2P median TPOT 比 `53.67 ms` 至少改善 25%；
- p99 TPOT 不比 central-FIFO 退化 10% 以上。

### Phase 3：P route-aware layout 与 vectorized gather

实施：

1. PAP decode rows 按 endpoint stable partition；
2. route group 变为连续 slice；
3. direct QKV hit 覆盖多 PA；
4. output 从逐 row copy 改为逐 slice copy；
5. membership 改变、async previous positions、sampling mapping 的专项测试。

完成标准：

- 2PA2P full-crossbar 中 P0/P1 每个 forward 可出现两个 route slice；
- steady decode non-contiguous fallback 为 0；
- Projection GEMM 仍是一个完整 cohort batch；
- 1PA1P 无回退，2PA2P full-crossbar 达到性能门槛。

### Phase 4：bounded coalescing 与任意 x:y

实施：

1. 从 trace 得到同 layer peer arrival-skew 分布；
2. 只选择一个 measured window 与 window=0 A/B；
3. deadline/fairness/slow-peer policy；
4. `1PA2P`、`2PA1P`、`2PA2P`、`3PA2P`，资源允许时 `6PA2P`；
5. TP1 通过后再做 TP2 per-rank dispatcher。

完成标准：combine 收益大于主动等待成本；没有 peer starvation 或全局 barrier。

### Phase 5：cross-layer 研究门槛

只有同时满足以下条件才进入：

- same-layer combine ratio 低；
- arrival trace 证明长期 layer phase skew；
- PA GPU 上小 append/FA kernel 或低 occupancy 仍是主要 TPOT 瓶颈；
- P2P/staging/Projection 已不是主导。

先做多 stream 独立 FA 原型；若收益不足，再设计 pointer-indirect backend。实现真正
cross-layer 前必须先完成 all-layer/all-TP-rank commit barrier。

### Phase 6：多轮 resident KV 集成

在多对多单轮调度稳定后，再按
`pap-xpayp-multiturn-kv-affinity-20260710.md` 实现：

- stable conversation session；
- resident attach/detach；
- exact token ledger；
- final-token closure；
- conversation PA affinity。

Attention work item 从第一阶段就携带 session generation，以避免届时再次重构调度主键。

## 14. 测试计划

### 14.1 CPU/unit

- 2x2 crossbar policy 四个 pair 全覆盖；
- arbitrary x/y 在 `x*y` 周期覆盖 Cartesian product；
- compatibility key：step 不同可以合并，layer/shape/scale/layout 不同不能合并；
- FIFO、deadline、max-row dispatch；
- slow peer 不阻塞已到 deadline 的其他 peer；
- scatter slices 恢复原 peer 和 row order；
- duplicate/stale generation fail closed；
- shutdown drain；
- work item input release 恰好一次。

### 14.2 Transport contract

- 两个 fake peer 并发 enqueue；
- combined compute 只调用一次；
- output 各发送一次；
- local-fast ready/release 顺序；
- NIXL message release；
- peer B 断开不释放 peer A 尚在使用的 slot；
- plan id 在内部 combine 后仍使用各 peer 原始 descriptor。

### 14.3 P model-runner contract

- persistent batch endpoint stable partition；
- add/remove request 后 request/token/block/sampling metadata 对齐；
- async previous-position mapping 正确；
- 一个 cohort 含两个 PA 时 QKV route slices 连续；
- output slice gather 后 O projection 输入 row order 正确；
- RUNNING rows 在连续 scheduler plans 中保持存在。

### 14.4 GPU smoke

最小顺序：

1. 1PA1P one request；
2. 1PA1P mixed arrivals；
3. 1PA2P 两 peer 同 layer combine；
4. 1PA2P 两 peer layer skew fallback；
5. 2PA2P full-crossbar；
6. request finish + new request admission；
7. peer failure/timeout；
8. TP2。

每次要求：completed/failed、token audit、decode commit、lease release、session drain、
pair matrix、combine/scatter stats 全部保存。

## 15. 性能实验矩阵

使用仓库 canonical workload，不选择新模型/数据集：

```text
model       /data/ssd1/llm-models/Qwen3-8B
dataset     sonnet
input       128
output      32
prefix      50
prompts     128
warmups     0
max model   512
max seqs    64
transport   local_fast
MPS         Prefill/Attention 70/30
async       enabled
```

每个正式候选：

1. focused tests；
2. QPS 4 三轮，观察非饱和固定开销；
3. QPS 16 三轮，观察服务能力、TPOT 和 queue buildup；
4. strict correctness audit；
5. session drain；
6. 一次 sampled trace，不与正常 TPOT 混报；
7. clean tracked worktree 才能成为正式 baseline。

比较矩阵：

| 组 | 用途 |
| --- | --- |
| 1PA1P legacy vs central vs combine | 回归与 dispatcher 固定开销 |
| 1PA2P legacy vs central vs combine | PA 多来源核心收益 |
| 2PA2P diagonal vs full-crossbar | 路由/多向通信成本 |
| 2PA2P full-crossbar combine off/on | combine 归因 |
| async on/off 单次诊断 | 只解释 admission/TPOT 取舍，不改变默认 |

必须报告：

- completed/failed；
- mean/median/p99 TTFT、TPOT、ITL；
- request/output throughput；
- peak concurrency；
- P cohort rows、route groups、direct/fallback；
- PA peer batches、combined batches、FA calls、combine ratio；
- ingress wait、coalesce wait、staging、compute、scatter；
- per-pair request count；
- correctness/drain。

## 16. 决策门槛

```text
Phase 1:
  若 1PA1P 回退 > 3%，停止，不叠加 combine；先修 dispatcher 固定开销。

Phase 2:
  若 1PA2P TPOT 改善 >= 25%，进入 P route-layout。
  若 combine ratio 高但 TPOT无收益，检查 staging copy/slot wait/FA batch scaling。
  若 combine ratio低，先测 layer skew，不扩大等待窗口猜参数。

Phase 3:
  若 P non-contiguous fallback 归零但 TPOT无收益，保留正确性简化并重新 profile。

Phase 4:
  bounded wait 只有在三轮 TPOT/throughput均胜出且 tail 不恶化时才默认启用。

Phase 5:
  没有 nsys/operation-level 证据，不进入 cross-layer kernel。
```

## 17. 技术选项与推荐

| 决策 | 选项 | 推荐 |
| --- | --- | --- |
| P scheduler | 保留 vLLM continuous batch / 自研 cohort scheduler | 保留现有 scheduler |
| async | 保留 / 关闭 / admission 时 flush | 保留；只做诊断 A/B |
| PA ingress | peer receiver + MPSC / transport multi-poll | 第一版 MPSC |
| PA compute | 单 dispatcher / 多 peer thread 直接算 | 单 dispatcher |
| combine | opportunistic / 固定等待 / 全 peer barrier | 先 opportunistic，再 measured bounded wait |
| QKV combine | 预分配 staging / 每次 `torch.cat` | 预分配 staging |
| P 多 PA rows | stable partition / 每层 gather kernel | stable partition |
| cross-layer | 同层 fallback / 多 stream / pointer-indirect kernel | 同层先行；profiler 后决定 |
| 2x2 路由 | 独立 modulo / Cartesian policy | opt-in crossbar round-robin |
| 多轮 KV | 与调度同时实现 / 后续集成 | 后续独立 phase |

## 18. Commit 与回滚策略

每个提交只包含一个可解释变化：

1. contract/metrics/crossbar policy；
2. central dispatcher behavior-equivalent scaffold；
3. same-layer combine/scatter；
4. P route-aware layout；
5. bounded policy；
6. arbitrary x:y/TP 扩展。

保留 feature flags：

```text
PAP_ATTENTION_DISPATCH_MODE=legacy|central|combine
PAP_ATTENTION_COALESCE_WINDOW_US=0
PAP_PROJECTION_ROUTE_LAYOUT=legacy|endpoint_stable
PAP_ROUTING_POLICY=crossbar_round_robin
```

任何阶段失败都回退最近一个 flag，而不是在同一提交里保留两套隐式行为。正式提交说明
必须记录测试命令、原始 run 目录、AI assistance 和人类 review 责任。

## 19. 第一批代码范围

计划评审后第一批只做 Phase 0：

1. `crossbar_round_robin` 与 pair-matrix contract；
2. P route continuity/direct-fallback stats；
3. PA per-peer arrival/rows/layer stats；
4. combine 前 `peer_batches == FA calls` 的显式基线计数；
5. 2PA2P full-crossbar smoke runner/audit；
6. async cohort membership contract test。

这一批不改变 GPU compute 调度，不引入等待，不修改 wire format。完成后先保存 current
full-crossbar baseline，再进入 central dispatcher 重构。

## 20. 最终验收

功能：

- 2PA2P 四个 pair 均被真实请求覆盖；
- 每个 P cohort 同时包含两个 PA 的 row；
- 每个 PA 同时接收两个 P；
- request 的 PA/P pair 在 turn 内不变；
- request step 无需对齐；
- add/finish request 不破坏其他 RUNNING request 的连续 forward；
- token、KV append、commit、lease、session 全部审计通过。

性能：

- 1PA1P 回退 <= 3%；
- 1PA2P QPS 4 相对当前至少改善 25%，stretch <= 1.20x 1PA1P；
- 2PA2P full-crossbar 相对 diagonal TPOT <= 1.10x，吞吐 >= 0.95x；
- tail、queue wait、slot wait 没有用更长隐藏等待换取中位数；
- 所有结论来自三轮 clean、同参数、strict-audit 运行。

只有达到以上门槛后，才把当前“连接层支持任意 x:y”升级为“执行与调度层支持任意
x:y”，随后进入多轮 resident KV 开发。

## 21. 2026-07-11 Phase 1/2 实施结果

### 21.1 Phase 1 等价中央调度器

`d654f6011` 上完成了同代码、交替顺序、每种模式三轮的 1PA1P QPS 4
A/B：

| 模式 | 三轮 median TTFT 的中位数 | 三轮 median TPOT 的中位数 |
| --- | ---: | ---: |
| legacy | `169.799 ms` | `28.138 ms` |
| central_fifo | `170.455 ms` | `28.514 ms` |
| central 相对 legacy | `+0.39%` | `+1.34%` |

六轮均为 `128/0`，correctness、routing、session drain 全部通过。中央 ingress、
CUDA ready-event 所有权和单 compute thread 可以作为 combine 基线。

### 21.2 Phase 2 同层 combine/scatter

新增 `central_combine`：

- dispatcher 从当前 ready 集合中选择同 layer、同 QKV ABI、同 scale 的任务；
- 多来源 QKV 在 GPU 上拼接，一次 KV append + paged FlashAttention；
- output 按原 descriptor 和 transport 的 row slice scatter；
- 单来源直接复用 `central_fifo` executor；
- bounded coalescing window 只等待已连接 peer 的短到达差；
- bind 协议携带稳定的 `projection-N-rank` source id；
- 多 PA 使用相同的确定性 Projection leader，使失相 cohort 先追赶、再同层合并。

诊断与标准负载结果：

| 拓扑/策略 | median TTFT | median TPOT | 结果 |
| --- | ---: | ---: | --- |
| 1PA2P central_fifo 旧基线 | - | `53.67 ms` | 三轮中位数 |
| 1PA2P combine，200 us | `184.84 ms` | `36.75 ms` | `128/0` |
| 2PA2P full crossbar，combine 前 | `291.85 ms` | `74.29 ms` | `32/0` |
| 2PA2P leader combine，1 ms | `176.65 ms` | `40.58 ms` | `128/0` |

以相同工作负载的 PD `24.28 ms` 为参照，1PA2P 和 2PA2P 分别为约
`1.51x` 和 `1.67x`，均低于 `<2x PD` 的 `48.55 ms` 门槛。1PA2P 相对旧基线
改善 `31.5%`；2PA2P 在 leader 对齐前后改善 `45.4%`。

标准 2PA2P 运行中四个 pair 均为 32 个请求。两个 PA 的 source-batch 合并覆盖率
分别为 `91.45%` 和 `91.93%`，dispatcher failure 为 0；correctness、routing、
decode commit、lease release 和双 PA session drain 全部通过。

相关运行目录：

```text
benchmarks/pap/experiments/legacy/runs/
  20260711_d654f6011_1pa1p_{legacy,central}_q4_rep{1,2,3}
  20260711_phase2_1pa2p_combine_wait200us_q4_rep1
  20260711_phase2_2pa2p_leader_align_wait1000us_q4_rep1
```

Phase 2 的 GPU 数值来自带 tracked patch 的实现验证，提交后仍需按相同参数做 clean
复跑。下一阶段是 P 端 route-aware gather/scatter：当前 full-crossbar 的 request rows
按 PA 交织，仍会触发非连续 QKV packing 和逐 row output copy。

## 22. 2026-07-11 Phase 3 实施结果

### 22.1 实现边界

本阶段没有重排 vLLM scheduler 的 row，也没有修改 Projection GEMM 的 cohort。相比直接
实施 endpoint-stable partition，先完成风险更小、可独立归因的 vectorized fallback：

- 同一 scheduler forward 为每个非连续 route group 构造一次 CUDA `long` index；
- index tensor 缓存在 `additional_kwargs`，36 层复用；
- 非连续 QKV 从逐 row view + `torch.cat` 改为一次 `torch.index_select`；
- 非连续 Attention output 从 Python 逐 row `copy_` 改为一次 `index_copy_`；
- 连续 route 保持 view/slice，完整单 group 保持 direct QKV/direct output；
- `PAP_BATCHED_ROUTE_COPY=0|1` 提供同代码 A/B 和故障回退，默认开启；
- benchmark runner 将该开关写入 effective config 和 run metadata。

因此本阶段解决的是“交织 rows 的重复 Python/CUDA launch 开销”，不声称已经完成
endpoint-stable scheduler layout。只有 profiler 证明一次 gather/scatter 仍是主要瓶颈时，
才继续修改 persistent `InputBatch` 的 row order。

### 22.2 2PA2P full-crossbar 严格 A/B

固定 Qwen3-8B、sonnet i128/o32/prefix50、128 prompts、QPS 4、local-fast、MPS
70/30、`central_combine`、1 ms combine window，按 legacy/batched 交替运行三轮。
两组使用同一代码和 tracked patch，只改变 `PAP_BATCHED_ROUTE_COPY`：

| 指标（三轮中位数） | legacy `0` | batched `1` | 变化 |
| --- | ---: | ---: | ---: |
| mean TPOT | `44.735 ms` | `41.923 ms` | `-2.812 ms` / `-6.29%` |
| mean TTFT | `241.747 ms` | `239.018 ms` | `-2.729 ms` / `-1.13%` |
| p99 TPOT | `58.267 ms` | `56.333 ms` | `-1.934 ms` / `-3.32%` |
| combine 覆盖率 | `93.38%` | `93.13%` | 基本相同 |

三组配对 TPOT 降幅分别为 `4.279 ms`、`2.812 ms`、`1.357 ms`，方向一致。
六轮全部 `128/0`，四个 PA/P pair 各 32 个请求；correctness、routing、session drain
全部通过，dispatcher failure 为 0。收益不是通过减少 Attention combine 得到的。

batched 三轮中位数约为 PD `24.28 ms` 的 `1.73x`，低于 `<2x PD` 的
`48.55 ms` 门槛。

原始结果：

```text
benchmarks/pap/experiments/legacy/runs/
  20260711_phase3_ab_legacy_rep{1,2,3}
  20260711_phase3_ab_batched_rep{1,2,3}
```

### 22.3 拓扑回归与已识别边界

1PA1P QPS 4 单次回归为 `128/0`、TPOT `28.51 ms`，相对 Phase 2 clean
`28.19 ms` 为 `+1.1%`，处于门槛内。

1PA2P 的单次运行仍在 `38--48 ms` 间波动，且 TPOT 与 PA combine 覆盖率一起变化。
专项 trace 覆盖 2,592 个 Projection layer forward，全部满足：

```text
route_groups == contiguous_route_groups
direct_qkv_groups == route_groups
packed_qkv_groups == 0
scattered_output_rows == 0
```

也就是说 1PA2P 在该拓扑中始终使用完整连续 route 的 direct QKV/direct output，本阶段
新增的非连续 gather/scatter 没有执行。该波动属于 Phase 2 已存在的多 Projection
cohort 相位稳定性问题，不能归因于 Phase 3；下一优化阶段应针对 source arrival skew、
leader catch-up 和 bounded wait 稳定性单独设计 A/B。

相关回归/trace：

```text
benchmarks/pap/experiments/legacy/runs/
  20260711_phase3_1pa1p_batched_q4_regression
  20260711_phase3_1pa2p_batched_q4_regression
  20260711_phase3_1pa2p_legacy_q4_ab1
  20260711_phase3_1pa2p_route_trace_smoke
```

### 22.4 验证状态

- `tests/pap`: `354 passed, 3 skipped`；
- Phase 3 聚焦 contract: `70 passed`；
- Ruff check/format、runner `bash -n`、`git diff --check` 通过；
- 未运行 pre-commit；
- tracked-dirty A/B 用于归因；正式基线见下节 `bdb7a7dc7` clean 三轮。

### 22.5 提交后 clean 2PA2P 基线

提交 `bdb7a7dc7` 后，以 `PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE=1`、默认
`PAP_BATCHED_ROUTE_COPY=1` 重跑三轮 2PA2P full-crossbar：

| rep | mean TTFT | mean TPOT | p99 TPOT | combine 覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | `239.263 ms` | `41.778 ms` | `55.499 ms` | `92.88%` |
| 2 | `238.637 ms` | `43.750 ms` | `57.223 ms` | `92.91%` |
| 3 | `235.650 ms` | `39.939 ms` | `56.800 ms` | `91.86%` |
| 三轮中位数 | `238.637 ms` | `41.778 ms` | `56.800 ms` | `92.88%` |

三轮均为 `128/0`，四个 pair 各 32 个请求；tracked worktree clean；correctness、
routing、session drain 全部通过；dispatcher failure 为 0。

同 QPS 4 的 PD 三轮参照为：

```text
/home/fei/research/PD/test/baseline/nixl_disaggregated/results/runs/
  20260710_pd_qps4_rep{1,2,3}_current
```

跨轮中位数对比：

| 指标 | PD 1P1D | PAP 2PA2P | PAP / PD |
| --- | ---: | ---: | ---: |
| mean TPOT | `24.506 ms` | `41.778 ms` | `1.705x` |
| median TPOT | `24.482 ms` | `41.430 ms` | `1.692x` |
| p99 TPOT | `25.756 ms` | `56.800 ms` | `2.205x` |
| mean TTFT | `176.401 ms` | `238.637 ms` | `1.353x` |
| request throughput | `3.890 req/s` | `3.817 req/s` | `0.981x` |

mean/median TPOT 已稳定达到 `<2x PD`；p99 TPOT 尚未达到 2x，说明下一阶段不能只追求
平均 gather/scatter 开销，还要解决 cohort 相位、coalesce timeout 和 queue-wait 的尾部
波动。

正式 PAP 结果：

```text
benchmarks/pap/experiments/legacy/runs/
  20260711_bdb7a7dc7_2pa2p_batched_q4_rep{1,2,3}
```

## 23. 2026-07-11 Phase 4 等待观测与自适应窗口实验

### 23.1 可观测性

提交 `581387a51` 为 dispatcher 增加了以下统计，不改变默认 1 ms 固定窗口：

- coalesce wait 的 `compatible`、`incompatible`、`timeout`、`stopped` 结果；
- 实际等待时长直方图；
- 兼容任务相对首任务的 arrival-skew 直方图、sum、max 和 sample 数；
- identity-safe work-item 计数，避免 Tensor 值比较触发歧义或把 preferred item
  误计为自己的候选。

2PA2P QPS 4 诊断运行是 `128/0`，Mean/Median/P99 TPOT 分别为
`40.228/39.285/52.981 ms`，合并覆盖率 `91.01%`。两个 PA 合计记录 46,068 个兼容
arrival-skew 样本：`93.0% <= 100 us`、`98.0% <= 200 us`、约 `0.3% > 1 ms`。

但 arrival-skew 不能直接作为 dispatcher 的等待窗口：任务可能在 dispatcher 已经开始
等待前到达，也可能受前一个 GPU dispatch 的服务时间影响。实际等待结果中有 6,686 次
纯 timeout；排除这些 timeout 后，成功或提前终止的等待约 `1.8%` 超过 500 us。

原始结果：

```text
benchmarks/pap/experiments/legacy/runs/20260711_phase4_wait_hist_q4_rep1
```

### 23.2 两状态自适应窗口原型

实验原型使用 `aligning=1000 us`、`steady=200/500 us`：连续兼容 group 后进入短窗口，
steady 阶段看到不兼容 peer 后重新进入 aligning。该实现只用于 tracked-dirty A/B，没有
提交到代码基线。

200 us 单轮立即失败：Mean TPOT `45.079 ms`、P99 TPOT `64.618 ms`、合并覆盖率仅
`76.89%`。因此正式 A/B 只比较固定 1 ms 与 500 us 自适应窗口，并按 fixed/adaptive
交替顺序各运行三轮：

| 指标（三轮中位数） | 固定 1 ms | 自适应 1 ms/500 us | 变化 |
| --- | ---: | ---: | ---: |
| mean TPOT | `42.516 ms` | `42.880 ms` | `+0.86%` |
| median TPOT | `41.926 ms` | `42.202 ms` | `+0.66%` |
| p99 TPOT | `55.033 ms` | `53.259 ms` | `-3.22%` |
| mean TTFT | `234.649 ms` | `240.574 ms` | `+2.52%` |
| request throughput | `3.822 req/s` | `3.812 req/s` | `-0.25%` |
| combine 覆盖率 | `91.60%` | `86.15%` | `-5.45 pp` |

六轮均为 `128/0`，四个 pair 各 32 个请求；correctness、routing、session drain 全部
通过，dispatcher failure 为 0。自适应组每轮发生 552--796 次 phase transition，说明
单个 PA 上不同 layer bucket 的到达状态持续交错，二态全局窗口发生抖动。虽然跨轮中位
P99 有小幅改善，但 mean/median TPOT、TTFT 和合并覆盖率同时退化，不满足 Phase 4
“均值/吞吐胜出且 tail 不恶化”的启用门槛。

结论：回退自适应状态机，保留固定 1 ms 默认与观测指标。下一步不再扫描另一个全局
timeout，而是把 active peer set、cohort generation 和 layer bucket 纳入调度状态：只对
当前 generation 中已知活跃、预计会到达同 layer 的 peer 做有限等待；idle/finished peer
不得进入 barrier。这能把“是否值得等”从时间启发式改为显式 cohort contract。

原始结果：

```text
benchmarks/pap/experiments/legacy/runs/
  20260711_phase4_adaptive200_q4_rep1
  20260711_phase4_ab_fixed_rep{1,2,3}
  20260711_phase4_adaptive500_q4_rep1
  20260711_phase4_ab_adaptive500_rep{2,3}
```

## 24. 2026-07-11 Phase 4 active-source membership

### 24.1 设计与实现

固定窗口的主要浪费不是“所有兼容任务都晚到”，而是 PA 永久把历史上绑定过的 P 数量
作为 `expected_group_size`。一个 P 完成最后请求或暂时不再服务该 PA 后，旧 source 仍会
让每个后续 layer 最多等待 1 ms。

本阶段增加 request-cohort 边界的控制面，不修改每层 QKV/output wire 热路径：

1. 每个 Projection/TP rank 维护当前 scheduler cohort 使用的 Attention endpoint set；
2. endpoint set 只有在请求加入、结束、preempt 或恢复导致 membership 变化时，才通过
   HTTP 发布一次 `source_id + active + membership_generation`；
3. 连续 decode step 使用相同 endpoint set 时不发控制请求；
4. PA 按 source 保存单调 generation，旧更新只计数、不覆盖新状态，同 generation 的
   冲突 active 值 fail closed；
5. PA 用 active source set 原子更新 dispatcher 的 expected group size 和确定性 preferred
   peer；active set 为空时回到 size 1、preferred `None`；
6. Projection 最后一个请求结束且本轮没有 model forward 时，也必须发送 inactive。

运行中的服务使用 `vllm/v1/worker/gpu/model_runner.py`。首次 smoke 只接入了旧
`gpu_model_runner.py`，表现为 `membership_updates=0`、expected size 恒为 1；该 smoke
被判定为功能未生效。最终公共逻辑放在 `vllm/pap/peer_activity.py`，两套 runner 都只保留
薄调用。V2 调用点位于：

```text
finish/free/add/update request state
-> apply staged block-table writes
-> sync active-source membership
-> zero-token early return 或 prepare/model forward
```

因此最后一个请求结束的 empty scheduler output 也能撤销 source。逻辑同时受
`PAP_PROJECTION_KV_UNAWARE=1` 和 `PAP_ATTENTION_ACTIVE_PEER_TRACKING=1` 门控，Prefill
worker 不会上报伪 source。

benchmark runner 在 `central_combine && Projection count > 1` 时默认开启；1P、legacy、
central_fifo 保持关闭。环境变量仍可显式设为 0，作为同代码回退。

### 24.2 2PA2P 严格 A/B

固定 Qwen3-8B、sonnet i128/o32/prefix50、128 prompts、QPS 4、local-fast、MPS
70/30、1 ms combine window、full crossbar，按 off/on、on/off、off/on 的服务重启顺序
各运行三轮。两组使用同一 tracked patch，仅改变
`PAP_ATTENTION_ACTIVE_PEER_TRACKING=0|1`：

| 指标（三轮中位数） | tracking off | tracking on | 变化 |
| --- | ---: | ---: | ---: |
| mean TPOT | `44.442 ms` | `41.476 ms` | `-6.67%` |
| median TPOT | `43.968 ms` | `41.606 ms` | `-5.37%` |
| p99 TPOT | `58.158 ms` | `48.312 ms` | `-16.93%` |
| mean TTFT | `242.533 ms` | `234.766 ms` | `-3.20%` |
| request throughput | `3.807 req/s` | `3.834 req/s` | `+0.70%` |
| combine 覆盖率 | `92.91%` | `83.99%` | `-8.92 pp` |
| coalesce timeout | `4,326` | `649` | `-85.00%` |

三组 mean TPOT 配对变化为 `-3.569/-0.398/-2.966 ms`，三组 p99 TPOT 变化为
`-9.467/-10.934/-9.846 ms`，方向全部一致。active membership 降低了 source-batch
combine 覆盖率，但消除了更多不可能等到的 idle-peer timeout，因此均值和 tail 同时改善。

六轮均为 `128/0`，四个 pair 各 32 个请求；correctness、routing、session drain 全部
通过，dispatcher failure 和 stale membership update 均为 0。tracking-on 每轮两个 PA
合计 78--82 次 membership update，运行结束 active set 全部为空。

相对同 QPS4 PD 三轮参照，tracking-on 的跨轮中位数为：

| 指标 | PD 1P1D | PAP 2PA2P active-source | PAP / PD |
| --- | ---: | ---: | ---: |
| mean TPOT | `24.506 ms` | `41.476 ms` | `1.692x` |
| median TPOT | `24.482 ms` | `41.606 ms` | `1.699x` |
| p99 TPOT | `25.756 ms` | `48.312 ms` | `1.876x` |
| mean TTFT | `176.401 ms` | `234.766 ms` | `1.331x` |
| request throughput | `3.890 req/s` | `3.834 req/s` | `0.986x` |

mean、median 和 p99 TPOT 三项均达到 `<2x PD`，其中 p99 是本阶段新增达成的门槛。

原始结果：

```text
/home/fei/research/PD/test/baseline/pap/results/runs/
  20260711_phase4_active_peer_ab_off_rep{1,2,3}
  20260711_phase4_active_peer_ab_on_rep{1,2,3}
```

### 24.3 任意 x:y correctness smoke

所有 smoke 使用 local Qwen3-8B、crossbar routing、active tracking、strict audit，并固定
70/30 MPS；短输出只验证 contract，不作为性能基线：

| 拓扑 | 请求 | pair 覆盖 | membership | correctness/drain |
| --- | ---: | --- | --- | --- |
| 1PA1P | `8/0` | `1/1` | update 4，active 最终空 | passed |
| 1PA2P | `8/0` | `2/2` | 两 source generation 2，最终空 | passed |
| 2PA1P | `8/0` | `2/2` | 两 PA 均收到 update，最终空 | passed |
| 3PA2P | `12/0` | `6/6` | 三 PA、两 source 全覆盖，最终空 | passed |

四组的 routing audit、correctness audit、session drain 全部通过；stale update 和 dispatcher
failure 都为 0。结果目录：

```text
/home/fei/research/PD/test/baseline/pap/results/runs/
  20260711_active_peer_1pa1p_smoke
  20260711_active_peer_1pa2p_smoke
  20260711_active_peer_2pa1p_smoke
  20260711_active_peer_3pa2p_smoke
```

### 24.4 提交后 clean 2PA2P 正式基线

实现提交为 `54bd1a59c0bb1e1b0ad8d7c237bad1f533162cc4`。提交后在命令行显式
清除 `PAP_ATTENTION_ACTIVE_PEER_TRACKING`，由 benchmark runner 的
`central_combine && Projection count > 1` 默认策略开启 tracking，并使用
`PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE=1` 重跑三轮：

| rep | mean TTFT | mean TPOT | median TPOT | p99 TPOT | req/s | combine 覆盖率 | timeout | membership update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `227.762 ms` | `39.746 ms` | `40.490 ms` | `50.806 ms` | `3.823` | `84.27%` | `799` | `74` |
| 2 | `227.908 ms` | `39.269 ms` | `39.179 ms` | `48.569 ms` | `3.844` | `82.51%` | `671` | `88` |
| 3 | `225.056 ms` | `40.724 ms` | `41.693 ms` | `50.031 ms` | `3.842` | `82.30%` | `658` | `78` |
| 三轮中位数 | `227.762 ms` | `39.746 ms` | `40.490 ms` | `50.031 ms` | `3.842` | `82.51%` | `671` | `78` |

三轮 metadata 均确认 tracking 为 `true`、commit 一致且 tracked worktree clean；每轮均为
`128/0`，四个 pair 各 32 个请求。correctness、routing、session drain 全部通过，两个
Attention 实例的 dispatcher failure、dropped item 和 stale membership update 均为 0；
服务结束时 active source set 全部为空。各 PA 均观察到两个 Projection source 的单调
generation，说明默认开启路径和 inactive 收尾路径均实际生效。

与同 QPS 4 的 PD 三轮参照比较：

| 指标 | PD 1P1D | PAP 2PA2P active-source | PAP / PD |
| --- | ---: | ---: | ---: |
| mean TPOT | `24.506 ms` | `39.746 ms` | `1.622x` |
| median TPOT | `24.482 ms` | `40.490 ms` | `1.654x` |
| p99 TPOT | `25.756 ms` | `50.031 ms` | `1.942x` |
| mean TTFT | `176.401 ms` | `227.762 ms` | `1.291x` |
| request throughput | `3.890 req/s` | `3.842 req/s` | `0.988x` |

mean、median、p99 TPOT 的正式 clean 基线均达到 `<2x PD`。相对 Phase 3 的
`bdb7a7dc7` clean 基线，mean TPOT、p99 TPOT 和 mean TTFT 分别改善约
`4.9%`、`11.9%` 和 `4.6%`。这与严格 A/B 的方向一致，同时排除了 tracked patch 和
显式开关配置对结论的影响。

正式结果：

```text
/home/fei/research/PD/test/baseline/pap/results/runs/
  20260711_54bd1a59c_2pa2p_active_q4_rep{1,2,3}
```
