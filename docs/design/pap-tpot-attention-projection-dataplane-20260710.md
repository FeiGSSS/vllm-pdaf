# PAP TPOT 优化：Attention–Projection same-node 数据面设计

日期：2026-07-10

状态：Phase 1 热路径裁剪与 Phase 2 KV fast path 已实现并完成标准负载验证；
完整 `prepare_step`、step 原子提交和 Phase 3 同步架构尚未实现

代码基线：`feature/pap @ 7eb6c4de9`，叠加当前 worktree 中尚未提交的
stream-ordered double-slot local-fast ring

目标模型与负载：Qwen3-8B，1PA1P，`i128/o32/qps16/prompts128`，固定
MPS `70/30`，不做 MPS 扫描

## 0. 实施结果更新（2026-07-10）

在不改变模型、负载、GPU 绑定和 MPS `70/30` 的条件下，当前实现已经达到
median TPOT `33.21 ms`，即 PD 的 `1.368x`。三次正式运行全部为 `128/0`，
strict correctness audit 和 Attention session drain 全部通过。

| 阶段 | 三次 median TPOT（ms） | 跨运行中位数 | 相对前一阶段 | 相对 PD |
| --- | --- | ---: | ---: | ---: |
| 原 stream ring | `46.34 / 49.07 / 47.21` | `47.21` | — | `1.945x` |
| Phase 1 binary/descriptorless | `42.95 / 44.20 / 44.10` | `44.10` | `-6.59%` | `1.817x` |
| Phase 2 all-active append | `41.37 / 40.66 / 40.86` | `40.86` | `-7.34%` | `1.683x` |
| Phase 2 cross-layer slot plan | `33.21 / 32.99 / 34.37` | `33.21` | `-18.73%` | `1.368x` |

最终 slot-plan 三次运行的详细结果：

| 运行 | median TTFT | mean / median / p99 TPOT | request throughput |
| --- | ---: | ---: | ---: |
| `20260710_phase2_slot_plan_rep1` | `1227.32 ms` | `34.08 / 33.21 / 39.82 ms` | `8.60 req/s` |
| `20260710_phase2_slot_plan_rep2` | `1258.36 ms` | `34.08 / 32.99 / 43.56 ms` | `7.89 req/s` |
| `20260710_phase2_slot_plan_rep3` | `944.78 ms` | `35.04 / 34.37 / 42.20 ms` | `9.83 req/s` |

同代码关闭 `PAP_DECODE_SLOT_PLAN_CACHE_LIMIT=0` 的归因 A/B 为
`42.12 ms` median TPOT；cache-on 三次中位数为 `33.21 ms`，说明主要收益确实
来自跨层复用 GPU slot tensor，而不是一次运行的排队波动。QPS `1`、其余配置
不变的低并发诊断为 `128/0`、median TTFT `120.98 ms`、median TPOT
`25.60 ms`，表明剩余的 QPS `16` 差距主要与并发资源竞争和排队有关。QPS `1`
结果不能直接与 QPS `16` 的 PD 数值作严格性能对比。

最终 stats smoke 在真实 GPU 路径记录到 `slot_plan_hits=105`、
`slot_plan_misses=3`、`slot_topology_mismatches=0`，即 eligible slot-plan lookup
命中率 `97.2%`；同时记录 `fast_path_hits=108`、`fallbacks=36`，证明典型 fast
path 和保守回退路径都实际执行。runner 将这些值保存到每个 run 的
`attention_fast_path_stats.json`。

已落地的范围：

1. Qwen forward 内 step group 只解析一次；第一层发布完整 plan，后续层使用
   plan reference；
2. local-fast 使用固定二进制 hot record，后续 QKV 与 output 不再携带 JSON
   descriptor，Attention 可从 template-only descriptor 恢复 batch；
3. 全 active decode batch 直接复用 K/V view，不再 advanced-index gather；scale
   tensor 按 device 常驻；
4. session 的 `block_ids + block_size + device` 在导入时做跨层一致性验证，同一
   decode batch/step 的 slot tensor 通过有界 LRU 跨层复用；session generation
   防止 request ID 复用导致 ABA；不一致或 partial batch 自动回退；
5. benchmark runner 记录 slot-plan cache 上限，固定使用本地模型、严格审计和
   session drain。

尚未落地的范围：完整 stable integer session handle/connection epoch、覆盖 36 层
的 `StepLease/state matrix`、GPU submit 移出 registry lock、最后一层后的原子
state commit、IPC event A/B、同进程双 GPU executor 和自适应 wavefront。由于
当前结果已经优于 `36.42 ms` 的 `1.5x PD` stretch 目标，按本文决策门槛先停止
扩大架构，转向稳定性、tail 和更广负载验证。

当前完整相关测试结果为 `155 passed, 2 skipped`；按项目要求使用
`.venv/bin/python` 执行，未运行 pre-commit。

代码整理后的提交前 canonical 复验
`20260710_precommit_cleanup_qps16_rep1` 为 `128/0`，median/p99 TPOT
`35.10/40.23 ms`，strict audit 与 session drain 通过；slot-plan
`10010 hits / 286 misses`，topology mismatch 为 `0`。

## 0.1 低 QPS 的 PD/PAP 同口径比较（2026-07-10）

为验证 QPS `16` 的 TTFT 差距是否主要来自排队，保持 Qwen3-8B、sonnet、
`i128/o32/prefix50/prompts128/warmup0/max-model-len512/max-seqs64` 不变，先将
QPS 降到 `8`。PD 使用 canonical 1P1D 配置，PAP 使用当前 1PA1P local-fast、
固定 MPS `70/30`。两者均绑定 GPU1/2。

QPS `8` 的单次诊断结果：

| 指标 | PD | PAP | PAP / PD |
| --- | ---: | ---: | ---: |
| request throughput | `7.56 req/s` | `6.60 req/s` | `0.873x` |
| median TTFT | `182.94 ms` | `573.86 ms` | `3.14x` |
| median TPOT | `25.26 ms` | `30.74 ms` | `1.217x` |
| p99 TTFT | `490.89 ms` | `2657.94 ms` | `5.41x` |
| peak concurrency | `24` | `31` | `1.29x` |

PAP 的实际吞吐低于 offered QPS，且并发继续累积，因此 QPS `8` 仍不是 PAP 的
非饱和比较点。继续降到 QPS `4` 后，各运行三次，以下取跨运行中位数：

| 指标 | PD | PAP | PAP / PD |
| --- | ---: | ---: | ---: |
| request throughput | `3.890 req/s` | `3.856 req/s` | `0.991x` |
| output throughput | `124.47 tok/s` | `123.39 tok/s` | `0.991x` |
| mean TTFT | `176.40 ms` | `213.16 ms` | `1.208x` |
| median TTFT | `171.18 ms` | `163.68 ms` | `0.956x` |
| p99 TTFT | `387.54 ms` | `1029.88 ms` | `2.657x` |
| mean TPOT | `24.51 ms` | `28.19 ms` | `1.150x` |
| median TPOT | `24.48 ms` | `28.06 ms` | `1.146x` |
| p99 TPOT | `25.76 ms` | `31.29 ms` | `1.215x` |
| p99 ITL | `36.30 ms` | `49.40 ms` | `1.361x` |
| peak concurrency | `15` | `16` | `1.067x` |

QPS `4` 时两者吞吐均由请求到达率决定，没有持续 backlog；PAP median TTFT 已
与 PD 持平，证明 QPS `16` 的大部分 TTFT 回退确实来自服务能力不足导致的排队。
在非饱和区，PAP 仍有约 `14.6%` median TPOT 固定开销，且 p99 TTFT 是 PD 的
`2.66x`，说明下一优先级应是间歇性 stall/tail，而不是继续解释 median TTFT。

三次 PD 与三次 PAP 均为 `128/0`；PD correctness log audit 全部通过，PAP
strict correctness audit 和 session drain 全部通过，slot topology mismatch
全部为 `0`。这些运行使用 dirty tracked worktree，属于当前代码 A/B，不替代
提交后的 clean baseline。

原始结果：

- PD QPS `4`：`/home/fei/research/PD/test/baseline/nixl_disaggregated/results/runs/20260710_pd_qps4_rep{1,2,3}_current/`
- PAP QPS `4`：`/home/fei/research/PD/test/baseline/pap/results/runs/20260710_pap_slot_plan_qps4_rep{1,2,3}_current/`
- PD/PAP QPS `8`：`20260710_pd_qps8_rep1_current` / `20260710_pap_slot_plan_qps8_rep1_current`

## 1. 结论先行

当前链路的主要瓶颈已经不是 GPU 间的数据带宽，而是 36 层上重复发生的：

1. Projection 侧路由展开、descriptor/plan 构造、哈希和对象分配；
2. Attention 侧 session/state 查找、Python 列表构造、临时 GPU tensor 分配、
   KV append 准备；
3. 两个 Python 进程每层一次的同步交替和 CPU doorbell 参与；
4. 小 kernel、P2P copy 和同步原语被逐层提交，缺少 step 级复用。

在典型 `B=19` 时，一层 QKV 与 output 的 P2P copy 微基准合计约
`38.9 us`，36 层约 `1.40 ms/token`。而带 trace 的当前实现中，Projection
看到的远端 attention 路径中位数为 `0.873 ms/layer`。因此，继续只替换
copy API、增加 ring slot 或调 busy-poll，收益上限很低。

推荐路线是：

1. **把协议粒度从 layer 提升到 decode step**：一次发布
   `PAPDecodeStepPlan`，每层只传固定大小的 generation/layer 信号，output
   不再携带 descriptor；
2. **Attention 一次 prepare、36 层复用**：session/state、slot mapping、
   FlashAttention metadata、scale 和 workspace 都在 step 开头准备；
3. **优先消除 Attention KV append 的临时 gather/alloc**，再决定是否需要
   自定义融合 kernel；
4. **同步机制分两级演进**：跨进程先用 binary doorbell + CUDA IPC event
   建立 CUDA scheduler 可见的依赖；若仍达不到 `40 ms`，再做同进程双 GPU
   executor，让 Projection 能异步提交完整的跨 GPU layer 链；
5. wavefront 只在足够大的 macro batch 上自适应启用，不作为当前典型
   `B=19` 的第一优先级。

阶段性工程目标应设为 median TPOT `<= 40 ms`，而不仅是偶然低于
`2 * PD`。`40 ms` 相当于当前 PD median 的 `1.65x`，能给波动留出余量。

## 2. 基线、目标与稳定性问题

### 2.1 PD 基线

同模型、同标准负载的 PD 结果：

| 指标 | PD |
| --- | ---: |
| mean / median / p99 TTFT | `246.82 / 212.53 / 470.69 ms` |
| mean / median / p99 TPOT | `24.69 / 24.28 / 26.27 ms` |
| request throughput | `14.26 req/s` |
| output throughput | `456.22 tok/s` |
| peak concurrency | `42` |

因此硬目标为：

```text
2 * PD median TPOT = 2 * 24.276896 = 48.553793 ms
```

### 2.2 当前 local-fast ring

三次标准运行的 median TPOT：

| 运行 | 成功/失败 | median TPOT |
| --- | ---: | ---: |
| `20260710_stream_ring_rep1` | `128/0` | `46.340675 ms` |
| `20260710_stream_ring_rep2` | `128/0` | `49.065927 ms` |
| `20260710_stream_ring_rep3` | `128/0` | `47.212730 ms` |

三次运行的中位运行值为 `47.212730 ms`，即 `1.9448x PD`。它已经在统计上
刚好低于 `2x PD`，但余量只有 `1.34 ms`，且第二次运行仍高于硬目标
`0.51 ms`，不能视为稳定达标。

三次运行的中位统计为：

| 指标 | 当前 PAP ring | 相对 PD |
| --- | ---: | ---: |
| mean TPOT | `46.50 ms` | `1.88x` |
| median TPOT | `47.21 ms` | `1.94x` |
| p99 TPOT | `51.86 ms` | `1.97x` |
| mean TTFT | `1770.07 ms` | `7.17x` |
| median TTFT | `1249.00 ms` | `5.88x` |
| request throughput | `9.29 req/s` | `65%` |

TTFT 的大幅增加与 decode 服务速率不足、排队深度上升一致。本文以 TPOT
和 decode throughput 为主优化对象；TTFT 用作排队是否随之改善的验证指标。

### 2.3 更有余量的目标

| 目标 | TPOT | 相对 PD | 从当前需要减少 | 36 层平均需减少 |
| --- | ---: | ---: | ---: | ---: |
| 硬门槛 | `< 48.55 ms` | `< 2.00x` | 已勉强达到 | — |
| 工程目标 | `<= 40.00 ms` | `<= 1.65x` | `7.21 ms` | `0.200 ms/layer` |
| stretch | `<= 36.42 ms` | `<= 1.50x` | `10.80 ms` | `0.300 ms/layer` |

## 3. 机器拓扑与它真正能提供的能力

当前机器是 8 张 NVIDIA L20。用于 PAP 的 GPU1 与 GPU2：

- 位于同一 node/NUMA 范围；
- `cudaDeviceCanAccessPeer` 双向可用；
- `nvidia-smi topo -m` 显示为 `NODE`；
- **没有 NVLink**，数据仍通过 PCIe 路径传输。

因此 same-node 优势是 CUDA IPC、UVA、P2P 和跨 GPU event，不应把优化假设
建立在 NVLink 带宽上。CUDA 官方文档确认 peer access 可让一个 GPU 访问
另一 GPU 的内存，并且 `cudaStreamWaitEvent` 支持跨设备流依赖：
[CUDA Programming Guide: Multi-GPU Systems](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)。

后端启动时应显式检查：

1. 两个 endpoint 在同一 host；
2. 双向 P2P access 可用；
3. IPC memory/event handle 能正常导入；
4. topology 满足本地后端策略。

任何一项失败都回退到 NIXL，不允许隐式走 CPU staging。

## 4. 当前 36 层临界路径

当前 direct QKV 和 direct output 已经避免了明显的中间 materialization，
但控制与状态准备仍逐层重复：

```text
Projection process / GPU-P                    Attention process / GPU-A

QKV projection
  |
  | Python: 展开 route group
  | Python: 构造 descriptor/plan/hash/JSON
  v
P2P copy QKV -> A-owned slot
stream write ready
CPU 写 /dev/shm doorbell  ------------------> CPU spin/read/JSON decode
                                                |
                                                | 重建 descriptor/items
                                                | session/state lookup + lock
                                                | 构造 active_indices/slots
                                                | 分配 slot/scale tensors
                                                v
                                           stream wait ready
                                           KV append kernel
                                           paged FlashAttention
                                           P2P copy output -> P-owned slot
                                           stream write ready
CPU spin/read/JSON decode <------------------ CPU 写 doorbell
  |
stream wait output
o_proj 直接消费 output slot
stream write release
  |
进入下一层；上述过程重复 36 次
```

一轮 decode step、一个 endpoint 会形成 36 次 QKV 消息和 36 次 output 消息，
也就是至少 72 次逐层 CPU 控制交互。slot ring 解决了 buffer ownership，但没有
改变这种逐层同步 RPC 的协议形态。

相关源码位置：

- Projection 路由与消息构造：`vllm/model_executor/models/qwen3.py` 的
  `_compute_pap_attention`；
- route group 实际已在 scheduler step 构造：
  `vllm/v1/worker/gpu_model_runner.py`；
- ring copy、doorbell、stream memop：`vllm/pap/local_fast_transport.py`；
- Attention mailbox loop：`examples/pap/pap_attention_executor.py` 的
  `run_offload_exec_mailbox_loop`；
- KV append 与 paged FlashAttention：同文件的
  `append_decode_kv_to_unified_prefill_cache` 和
  `compute_offload_exec_batch_output`。

## 5. 定量瓶颈分解

### 5.1 trace 使用限制

`20260710_stream_ring_trace_design` 完成 `128/0`，correctness audit 和 session
drain 均通过。该运行的 median TPOT 为 `61.71 ms`，不能作为性能基线，因为
现有 trace 会在每层对 CUDA event 调用 `synchronize()`。下面只使用它做
相对路径分解。

### 5.2 每层中位耗时

| 区段 | 中位耗时 | 解释 |
| --- | ---: | --- |
| Projection layer total | `1.175 ms` | 整层 wall time |
| Projection self-attn | `1.100 ms` | 整个 self-attn section |
| Projection pre-attn local compute | `0.072 ms` | QKV/norm/RoPE 尾部 |
| Projection send | `0.175 ms` | 包含路由、descriptor 与 transport 调用 |
| Projection recv | `0.667 ms` | 主要在等 Attention 完成，不是纯传输 |
| Projection remote total | `0.873 ms` | send 到 output ready 的关键区间 |
| Projection o_proj | `0.040 ms` | 已直接消费 output slot |
| Attention recv QKV | `0.571 ms` | 大部分是提前进入 poll 后的空等 |
| Attention compute total | `0.469 ms` | prepare + append + attention |
| Attention send output | `0.107 ms` | output 发布 |
| Attention total | `1.176 ms` | recv、compute、send |

跨进程时间戳关联进一步表明：

- Attention 的 recv 比 Projection send 完成早约 `0.493 ms` 开始；
- recv 总长约 `0.573 ms`，所以 send 完成后的 recv 尾巴只有约 `0.08 ms`；
- Projection send 完成到 Attention output send 完成约 `0.639 ms`；
- 其中 Attention pre-compute 约 `0.307 ms`；
- 实际 attention compute 区间约 `0.183 ms`；
- output send 完成到 Projection recv 完成仅约 `0.026 ms`。

所以 `recv_qkv_ms=0.571 ms` 不能解释为 P2P copy 慢。它主要表达两个进程的
逐层生产者/消费者节奏。

### 5.3 P2P 数据量与传输下限

Qwen3-8B 的关键维度为：

- hidden size `4096`；
- query heads `32`，KV heads `8`，head dim `128`；
- dtype `bfloat16`；
- 每请求每层 QKV `12,288 B`；
- 每请求每层 attention output `8,192 B`。

trace 的典型 batch 为 `B=19`：

| 数据 | 大小 | P2P copy 微基准中位数 |
| --- | ---: | ---: |
| QKV | `233,472 B` | 约 `21.5 us` |
| output | `155,648 B` | 约 `17.4 us` |
| 合计 | `389,120 B` | 约 `38.9 us/layer` |

36 层合计约 `1.40 ms/token`。即使完全消灭这两次 copy，理论收益也远小于
当前 PAP 与 PD 约 `22.94 ms` 的 TPOT 差距。P2P 是需要保留的正确数据路径，
但不是现在最值得继续做标量优化的位置。

### 5.4 Projection 每层重复开销

transport 内部 trace 的 host enqueue 时间中位数：

| 方向 | copy enqueue | stream write enqueue | doorbell | 合计 |
| --- | ---: | ---: | ---: | ---: |
| QKV | `6 us` | `6 us` | `9 us` | `22 us` |
| output | `6 us` | `5 us` | `28 us` | `40 us` |

Projection 侧完整 send 为 `175 us`，比 QKV transport 内部的约 `22 us` 多
`153 us`。这部分包含 route group 重走、request/session 列表、descriptor、
plan/hash、Python 对象以及 transport dispatch，不能全部归因于 JSON。

CPU 微基准也显示，典型 `B=19` 时：

- QKV plan payload + SHA1：约 `12.5 us/layer`；
- plan-ref parse + descriptor/items 重建：约 `17.1 us/layer`；
- ref JSON encode：约 `1.7 us/layer`；
- output full metadata 构造与编码：约 `8.5 us/layer`。

在 `B=52` 时，前两项分别增加到约 `30.4 us` 和 `45.6 us`。这些单项不够
解释全部差距，但它们都是不应随 36 层重复的工作。

### 5.5 Attention prepare/KV append

Attention compute 内的中位分解：

| 子阶段 | 中位耗时 |
| --- | ---: |
| KV append | `0.233 ms` |
| FlashAttention metadata build/cache lookup | `0.025 ms` |
| session/shape lookup | `0.017 ms` |
| QKV split | `0.008 ms` |
| query move | `0.003 ms` |
| paged FlashAttention wall | `0.118 ms` |
| paged FlashAttention kernel | `0.096 ms` |

当前 KV append 每层会：

1. 在 registry lock 中逐请求查 state 和 slot；
2. 构造 Python `slots` 与 `active_indices`；
3. 使用 `key_batch[active_indices]`、`value_batch[active_indices]`，触发
   advanced-index gather；
4. 新建 GPU `slot_tensor`；
5. 新建 `k_scale`、`v_scale`；
6. 提交一次 `reshape_and_cache_flash`；
7. 再逐请求更新 state。

典型 decode step 中 request order、target seq、block table 与 slot mapping 对
36 层高度稳定。即使某些层的物理 block id 不同，也可以在 step 开头一次性
构造 `[layer, batch]` 的 slot matrix，不需要每层重新走 Python。

### 5.6 tail 问题

trace 中 Projection remote total 的 p99 为 `4.069 ms`，远高于中位数
`0.873 ms`。Attention KV append p99 为 `0.851 ms`，metadata build p99 为
`2.219 ms`。这说明 allocator、cache miss、Python 调度或控制线程抢占也在
制造 TPOT/TTFT 尾部。step 级缓存和无分配热路径既优化中位数，也应重点验证
p99。

## 6. 目标数据面：step plan + descriptorless layer ring

### 6.1 控制面与数据面分离

控制面继续负责：

- endpoint/peer 建连；
- session install/release；
- Prefill KV handle 与 lease；
- peer epoch、错误、超时、drain；
- 每个 decode step 一次的完整 plan 发布。

steady-state 数据面只负责：

- 固定 slot 中的 QKV/output；
- `step_generation + layer_ordinal + slot_id`；
- ready/release 顺序；
- 错误时 fail-closed。

### 6.2 `PAPDecodeStepPlan`

建议在 `gpu_model_runner` 已经生成 route group 的地方构造 step plan，而不是
让 Qwen 的每一层再次展开 route group。逻辑结构可为：

```text
PAPDecodeStepPlan
  protocol_version
  connection_epoch
  step_generation
  endpoint_group_id
  layer_count
  batch_size
  qkv/output layout + dtype
  rows[]:
    stable_session_handle
    session_handle_generation
    projection_row
    target_seq_len
    decode_token_id
```

关键点：

- session install 时分配稳定的整数 handle，热路径不传长 request id；
- handle 必须带 generation，防止 session id 回收后 ABA；
- plan 完整内容只发布一次并等待 ACK；
- layer 信号引用 `step_generation`，不再带 descriptor/hash；
- batch size 决定 nbytes，因此正常路径甚至不需要逐层携带 nbytes；
- output shape、row order 由 plan 唯一确定，output 完全不传 metadata；
- connection epoch 在 peer 重启后改变，旧 generation 一律拒绝。

每层热记录可以缩小为固定二进制结构：

```text
PAPLayerSignal { step_generation, layer_ordinal, slot_id, flags }
```

GPU stream memop 目前使用 32-bit value。达到 wrap 边界前必须 quiesce ring，
更新 epoch 并重新初始化 signal buffer，不能依赖自然回绕。

### 6.3 Attention 的 `prepare_step`

收到并验证 plan 后，Attention 一次完成：

1. 获取覆盖整个 step 的 `StepLease`，阻止 session/KV 在执行中被释放；
2. session handle 到 session entry 的解析；
3. 验证所有 row 的 shape、scale、unified KV 状态；
4. 建立每层 KV cache pointer/state view；
5. 建立共享或 per-layer 的 slot mapping；
6. 建立 FlashAttention block table、seq lens、cu_seqlens；
7. 从持久 workspace 切出 QKV/output/slot tensor view；
8. 建立该 step 的目标 seq_len 与 commit 记录。

如果 36 层的 block table 相同，metadata 可完全共享；如果不相同，则一次构造
`[layer, batch, blocks]`，每层只做 view/select。这里必须运行时验证，不能把
“通常相同”写成无条件假设。

### 6.4 每层热路径

完成 prepare 后，每层只做：

```text
wait QKV ready
Q/K/V view
KV append
paged FlashAttention
P2P copy output
signal output ready
```

Projection 只做：

```text
P2P copy direct QKV buffer
signal QKV ready
wait output ready
o_proj directly consumes output slot
release output slot after o_proj stream use
```

现有 direct QKV 和 direct output 应保留。本文不再引入额外 staging copy。

## 7. Attention KV append 优化

### 7.1 P0：all-active contiguous fast path

标准 decode batch 通常每行都满足：

```text
target_seq_len == state.seq_len + 1
```

此时：

- 直接使用 `key_batch`、`value_batch` view；
- 不构造 `active_indices`；
- 不做 advanced indexing gather；
- `slot_tensor` 从 step workspace 读取；
- `k_scale/v_scale` 使用进程级常量；
- registry lock 只保护解析/状态转换，不包住 GPU kernel 提交；
- 非全 active 或乱序 case 回退现有通用实现。

### 7.2 step 级状态提交

不建议在 kernel 完成前直接把 36 层 `seq_len` 全部标为完成。可采用：

1. plan 中保存 `target_seq_len`；
2. layer 执行时使用 plan 的 metadata，不依赖提前修改的 Python state；
3. 每层完成后只更新轻量 completion bitmap；
4. 最后一层成功后，在锁内原子提交该 step 的所有 layer state；
5. 中途失败则 session 进入 invalid/recovery 状态，不允许继续使用部分 KV。

这样可以把逐层 Python state loop 移出热路径，同时维持 fail-closed 语义。

### 7.3 P1：是否需要融合 kernel

先实现无 gather/无 alloc fast path，再重新测 KV append。只有当 append 仍稳定
高于约 `0.12-0.15 ms/layer` 时，才值得写自定义 kernel。

候选 kernel：

- 输入为 Attention 本地 QKV slot 和预构造 slot mapping；
- Q 保持 view 或写入持久 query workspace；
- K/V 直接 scatter 到 paged KV cache；
- 一次 launch 完成 dtype/layout 处理与 append；
- 不从 Projection GPU 做细粒度 remote load。

直接让 kernel 从 peer GPU 读取 QKV 会延长 Projection slot lease，并把 PCIe
remote-load latency 带进 kernel，风险高于先 P2P 到本地连续 slot。当前数据表明
本地 staging copy 只有约 `21.5 us`，不应优先消灭它。

## 8. 同步路线：安全地减少 CPU 参与

### 8.1 当前 stream memory-op 的边界

`cuStreamWaitValue32/cuStreamWriteValue32` 能建立低开销 GPU flag 协议，
`cuStreamBatchMemOp` 也可能减少 API/设备开销。但 NVIDIA 明确提示，stream
memory-op 形成的同步对 CUDA scheduler 不可见，间接依赖仍应使用 CUDA 可见的
同步表达，否则可能出现不正确的调度或 deadlock：
[CUDA Driver API: Stream Memory Operations](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html)。

所以不能仅因为当前双 slot 测试通过，就直接预提交一个跨两个进程、36 层、
包含环形依赖的 future-value wait 图并作为默认实现。

### 8.2 跨进程安全中间态：binary doorbell + IPC event

CUDA IPC event 可以在进程间导出/导入，并由另一个进程在 stream 上等待：
[CUDA Programming Guide: Interprocess Communication](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/inter-process-communication.html)。

推荐中间协议：

1. 每个方向、每个 slot 有 exporter-owned IPC ready event；
2. sender 在 copy/compute 后 record event；
3. sender 发布固定二进制 CPU doorbell；
4. receiver 看到 generation 后，对已 record 的 imported event 调用
   `cudaStreamWaitEvent`；
5. release 方向使用 receiver-owned IPC event 对称实现；
6. binary doorbell 只负责“哪个 generation 已 record”，不搬数据、不传 JSON。

同一个 slot 在收到对端 release 前不得再次 record ready event。否则 receiver
在 doorbell 与 `cudaStreamWaitEvent` 之间可能观察到下一 generation 的 record，
破坏当前 generation 的 slot ownership。

`cudaStreamWaitEvent` 使用调用时最近一次 `cudaEventRecord`；之后对同一 event
的 record 不会改变已经提交的 wait。因此，不能在 CPU 尚未看到新 generation
前，对复用 event 预先提交“等待未来 record”：
[CUDA Runtime API: Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)。

这个方案仍有每层一次轻量 CPU poll/submit，但：

- 删除 JSON/descriptor；
- 依赖对 CUDA scheduler 可见；
- 更容易证明 slot 生命周期；
- 可作为与当前 stream memop 的正确性/性能 A/B。

### 8.3 真正去掉逐层 CPU：两个候选

#### 候选 A：跨进程 GPU-only timeline

可研究 external timeline semaphore、NVSHMEM signal 或严格受控的
stream-memop program。但它们分别带来额外 runtime、部署依赖或 scheduler
风险。除非 binary + IPC event 路径仍明显受 CPU 控制限制，否则不优先。

#### 候选 B：同进程双 GPU executor（推荐的长期路径）

让 Projection worker 同时持有 GPU-P 与 GPU-A 的 CUDA context/stream：

- Prefill 进程仍在 GPU-A 上运行并通过 MPS 共享；
- Projection worker 导入 Prefill 的 unified KV IPC handle；
- Attention 热循环不再由独立 Python daemon 处理；
- QKV ready、P2P copy、Attention、output copy、o_proj 用跨设备 CUDA event
  串联；
- Python 可以异步提交后续层操作，而不是在每层 CPU poll output；
- 控制面服务和 NIXL fallback 保留。

同进程的关键收益不是省去一次 JSON，而是 host 能看见两条 GPU stream，并用
CUDA 原生 event 表达完整依赖。第一版不依赖 CUDA Graph；确认手工异步提交
有效后，再评估 graph capture。多设备 graph、context ownership 和 PyTorch
allocator 限制必须通过原型验证，不能预设可直接 capture。

主要代价：

- vLLM worker 原本的一进程一 GPU 假设被打破；
- current device、线程、allocator 与异常传播更复杂；
- Projection 进程会同时出现在两个 GPU 上；
- MPS/资源隔离与服务重启边界需要重新定义；
- 只适用于 same-node local backend。

因此它应是达不到 `40 ms` 后的架构升级，而不是第一份改动。

## 9. wavefront 与 Q-first 的位置

### 9.1 wavefront

历史 layer-wavefront 实验已证明：

- `B=6` 拆成 `3 x B2` 时，消息数从 240 增到 624，TPOT 退化；
- `B=24` 时 2-way `B12` 比 serial TPOT 改善约 `26.4%`；
- `B=48` 时 2-way 优于 3-way，后者被 send queue/ACK 开销抵消。

相关记录见：

- `docs/design/pap-runner-3way-microbatch.md`；
- `docs/design/pap-6pa2p-large-workload-20260526.md`。

当前标准负载的 layer batch 中位数约 `19`。在 descriptorless ring 落地前拆分
它，很可能重新放大固定成本。推荐顺序是：先降低每消息成本，再按
`macro_batch / ubatch_count` 与实测 ready/resume lag 决定 serial/2-way。

### 9.2 Q-first/KV-later

简单把 packed QKV 拆成三次 copy 的历史实验已经退化。Q-first 只有在以下条件
同时成立时才有意义：

1. Q projection/RoPE 能早于 K/V 真正 ready；
2. Attention 能用 fast backend 先计算历史 KV 的 partial output/LSE；
3. 当前 token 的 K/V 到达后能低成本合并；
4. 不把 fused QKV projection 拆成更慢的小 GEMM；
5. 新增消息没有重新放大控制开销。

Qwen3 当前是 fused QKV projection，而 P2P copy 本身只有几十微秒。为节省这点
copy latency 去拆 GEMM 或退回 Python/einsum，不是当前优先路线。

## 10. 不建议优先投入的方向

| 方向 | 原因 |
| --- | --- |
| 仅把 `Tensor.copy_` 换成 `cudaMemcpyPeerAsync` | host enqueue 约 `6 us`，收益上限太低 |
| 继续增加 ring slots | 当前不是 slot 等待主导；更多 slot 增加生命周期复杂度 |
| msgpack/缩短 JSON | descriptorless 后逐层 codec 应直接消失，不值得继续打磨 |
| CPU busy-spin 参数扫描 | 可能改善 tail，但不改变逐层同步架构 |
| MPS 比例扫描 | 本路线固定 `70/30`，且用户明确不做扫描 |
| 把 QKV projection 移到 Attention GPU | 只把每请求 `12 KB` 降为约 `8 KB`，却增加 MPS 受限 GPU 的权重与计算 |
| NCCL 替换 P2P copy | 小 payload、单 peer、逐层场景下不会消除 plan/KV prepare 开销 |
| NVSHMEM 直接作为第一版 | 可能融合 put+signal，但引入依赖，且不解决每层状态准备 |
| 无保护地预提交 36 层 memory-flag wait | CUDA scheduler 可见性与循环依赖存在正确性风险 |

## 11. 候选方案收益、风险与优先级

以下收益是基于 trace 的工作假设，必须 A/B 验证，且各项存在重叠，不能直接
相加：

| 方案 | 主要消除项 | TPOT 预估 | 风险 | 优先级 |
| --- | --- | ---: | --- | --- |
| StepPlan + descriptorless layer ring | 每层 route/descriptor/hash/JSON/object | `3-6 ms` | 中低 | P0 |
| Attention prepare_step + all-active fast path | session/state loop、gather、临时 tensor、metadata 重建 | `3-6 ms` | 中 | P0 |
| binary doorbell + IPC event | codec、部分控制抖动、不可见同步风险 | `1-3 ms` | 中 | P1 |
| KV append 融合 kernel | 仍残留的 append launch/memory overhead | `1-3 ms` | 中高 | P1，需 profiler gate |
| GPU-only cross-process timeline | 逐层 CPU submit/poll | `2-5 ms` | 高 | P2 实验 |
| same-process dual-GPU executor | 跨进程逐层 RPC 与 CPU 阻塞 | `4-10 ms` | 高、改动大 | P2 长期路线 |
| adaptive 2-way wavefront | 隐藏剩余 remote wait | workload-dependent | 中高 | P3 |

Phase 1 与 Phase 2 的联合工作假设是把 median TPOT 从 `47.21 ms` 降到
`38-42 ms`。如果只能到 `42 ms` 左右，优先检查逐层 CPU wait 是否仍在关键
路径，再决定 IPC event 或 same-process，而不是继续做 copy 微优化。

## 12. 分阶段实施路线

### Phase 0：低扰动 profiling

目标：得到可信 GPU timeline，不用带逐层 `event.synchronize()` 的 trace 做
绝对性能结论。

实施项：

- 添加 NVTX range：P-QKV、P2P-QKV、A-KV-append、A-FA、P2P-output、P-o_proj；
- CUDA event 只写入预分配 ring，在 step 结束或采样窗口结束后统一读取；
- 每 `N` 个 step 采样一次，默认关闭；
- 使用机器已有的 `nsys` 捕获 `cuda,nvtx,osrt`；
- 记录 host API count、GPU idle gap、allocator 次数、plan cache hit；
- 分 batch bucket 报告，而不只报全局中位数。

验收：profiling 默认关闭时无性能变化；开启采样时不在每层同步 GPU。

### Phase 1：StepPlan 与 descriptorless ring

实施项：

1. 增加 `PAPDecodeStepPlan`、connection epoch、stable session handle；
2. 在 model runner 每 scheduler step 构造 plan；
3. Qwen layer 只引用 plan + layer ordinal；
4. transport 增加 `publish_step_plan/prepare_step/end_step`；
5. QKV/output layer record 改为固定二进制；
6. output 不再发送 descriptor；
7. 旧 descriptor path 保留为 fallback；
8. 增加计数器，证明 full plan 为 `1/step` 而不是 `36/step`。

验收门槛：

- 三次标准运行均 `128/0`；
- strict correctness audit 与 session drain 通过；
- 每个 step 只有一次 plan build/parse；
- 每层无 SHA1、无 JSON、无 descriptor/items 重建；
- 三次 median TPOT 的中位数至少改善 `5%`；
- 三次运行各自均 `< 48.55 ms`；
- p99 TPOT 不退化超过 `10%`。

### Phase 2：Attention step preparation 与 KV fast path

实施项：

1. `StepLease` 与 state matrix；
2. slot mapping/FA metadata 一次构造；
3. all-active contiguous fast path；
4. scale、slot、workspace 持久化；
5. lock 外 GPU submit；
6. step completion bitmap + 最终原子 state commit；
7. 通用 fallback 保留；
8. 只有 profiler 证明必要时再实现 append kernel。

验收门槛：

- 热路径临时 CUDA tensor allocation 为 0；
- 典型 batch 不发生 K/V advanced-index gather；
- append median 明显下降且 p99 收敛；
- 三次运行 median TPOT 中位数 `<= 40 ms`；
- request throughput、TTFT 与 peak concurrency 同步改善；
- token 输出、KV commit、session release 全部通过审计。

### Phase 3：同步 A/B 与 CPU removal

先实现 fixed binary doorbell + IPC event，与当前 stream memop 做相同代码基线
A/B。若 `nsys` 仍显示每层 host poll/submit 是主要 idle gap，则做
same-process dual-GPU 最小原型：

- 只支持 1PA1P、同 host、P2P 双向可用；
- 不先做 CUDA Graph；
- 不改变 Prefill 控制面；
- 保留跨进程后端作为 fallback；
- 用一个固定 batch 的 36 层 synthetic pipeline 先证明无 CPU layer barrier。

验收：在 Phase 2 基线上再有稳定收益，且不能以 deadlock、错误恢复能力或
session leak 为代价。

### Phase 4：自适应 wavefront

仅在 Phase 1-3 后重新扫描 serial/2-way，不扫描 MPS。启用条件至少包括：

- 每个 ubatch 大于实测最小密度；
- 预测 overlap 收益大于新增 layer task 开销；
- ready ubatch 优先，避免 Projection resume lag；
- 2-way 先于 3-way；
- 小 batch 强制 serial。

## 13. 正确性不变量

| 不变量 | 要求 |
| --- | --- |
| QKV slot ownership | sender 只能在 Attention consumer stream 已释放后复用 |
| output slot ownership | Attention 只能在 Projection o_proj consumer stream 已释放后复用 |
| step/layer ordering | 必须匹配 epoch、step generation、layer ordinal；跳号立即失败 |
| session handle | handle + generation 匹配，禁止 ABA |
| KV append | 每 request、每 layer、每 token 恰好一次，乱序 fail-closed |
| state commit | 36 层完成后才提交整个 decode step；部分失败不得继续静默执行 |
| event reuse | receiver 只能等待 CPU 已确认 record 的 generation |
| generation wrap | quiesce + epoch rollover，不做无保护 32-bit 回绕 |
| peer death | timeout 后解除等待、标记 session invalid、清理 slot/lease |
| fallback | local 条件不满足时显式回退 NIXL，不走 CPU tensor staging |
| drain | 服务结束时 plan、slot、session、KV lease 均为 0 |

## 14. 性能验证矩阵

所有正式对比固定：Qwen3-8B、本地模型、MPS `70/30`、关闭 FlashInfer
sampler、`i128/o32/qps16/prompts128/max-model-len512/max-num-seqs64`，不访问
Hugging Face，不做 MPS 扫描。

每个候选至少执行：

1. unit/contract 测试；
2. `qps=1` 低并发诊断，隔离单 token 固定开销；
3. 标准 `qps=16` 三次运行；
4. strict correctness audit；
5. session drain；
6. 一次 sampled trace 或 nsys，不把 trace TPOT 与正常 TPOT 混报；
7. 按 batch bucket 报告 layer path；
8. 与同一 PD 基线比较。

正式报告必须列出每次运行，而不是只给最佳值：

- completed/failed；
- mean/median/p99 TPOT；
- mean/median/p99 TTFT；
- request/output throughput；
- peak concurrency；
- remote path、KV append、FA kernel、P2P copy；
- plan builds、descriptor builds、JSON bytes、CUDA alloc count；
- audit/drain 结果。

## 15. 决策门槛

```text
Phase 1 后：
  若跨运行 median < 45 ms 且所有运行 < 2x PD：进入 Phase 2。
  若改善 < 5%：先用 nsys 重新验证归因，不叠加更多协议改动。

Phase 2 后：
  若跨运行 median <= 40 ms：停止扩大架构，做稳定性与 tail 优化。
  若 40-43 ms 且 CPU layer barrier 明显：进入 Phase 3。
  若 GPU append/FA 仍主导：先做 kernel/layout 优化，不做 same-process。

Phase 3 后：
  若 <= 36.42 ms：达到 1.5x PD stretch。
  若仍 > 40 ms：重新评估 PAP 分层本身的不可隐藏下限，而不是继续微调 ring。
```

## 16. 推荐的第一批实际改动

第一批实现建议严格限制在以下范围：

1. 无同步的 sampled timeline；
2. `PAPDecodeStepPlan` 数据结构与 stable session handle；
3. runner 每 step 构造 plan；
4. transport 一次发布/解析 plan；
5. layer/output descriptorless binary record；
6. 旧协议 fallback 与完整计数器。

第一批暂不做：

- CUDA kernel；
- GPU-only future wait；
- same-process executor；
- wavefront；
- MPS 扫描。

这样可以先验证最大、最清晰的一块重复 CPU/control 开销。若收益符合预期，
第二批再实现 Attention `prepare_step` 和 all-active KV fast path。两批分别测量，
能够避免“多个优化一起落地但无法归因”的问题。

## 17. 最终判断

目前 `47.21 ms` 的 PAP median TPOT 已接近 `2x PD`，但没有稳定余量。现有
证据不支持继续把主要精力放在 P2P copy 带宽或单条 doorbell 上。更可信的
优化主线是：

```text
per-layer RPC
  -> per-step plan + descriptorless layer data plane
  -> per-step Attention state/metadata/workspace
  -> CUDA-visible synchronization
  -> 必要时 same-process dual-GPU asynchronous submission
```

前两步预计足以把 TPOT 推向 `38-42 ms` 区间；是否需要同进程 executor，应由
Phase 2 后的 nsys idle-gap 证据决定。这个顺序同时利用了 same-node P2P，减少
CPU 参与，避免重复开销，并保留了清晰的正确性和回退边界。
