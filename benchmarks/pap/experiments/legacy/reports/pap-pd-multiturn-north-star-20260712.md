# PAP/PD 1:1 多轮北极星基线与首轮优化

日期：2026-07-12

状态：Stage A/B/C 已完成 clean formal，Stage D v1 诊断已完成；当前 PAP reference
仍为 `0727ed946`

> **2026-07-13 基线校正：** 本文的 PAP 内部 Stage A/B/C/D A/B 和绝对耗时仍是有效
> 历史证据，但本文使用的 PD pull connector 后续确认走 UCX TCP software emulation，
> 因而本文中的跨架构 PAP/PD 比值已被 corrected-push 基线替代。当前五轮、C4 正式
> 比较见
> [PD Push 校正与 PAP 五轮长上下文性能报告](pd-pap-five-turn-load-results-20260713.md)。

## 1. 目的

本记录固定一个可重复、可审计的 1P1D PD 与 1PA1P PAP 多轮性能比较，并把它作为后续
TTFT/TPOT 优化的唯一北极星。长期 X:Y 扩展复用同一结果合同，本轮不扩展拓扑。

冻结 profile：`qwen3_8b_chat_16k_2turn_o256_c1_v1`。

- Qwen3-8B，FP16，TP1，GPU 1/2 均为 NVIDIA L20；
- 两轮真实 Chat messages，thinking 开启；
- 第一轮 16,000 文档 tokens，第二轮新增 120 tokens；
- 每轮固定生成 256 tokens，C1 closed loop；
- 每个 formal repetition 完整重启所有服务，正式值取三次中位数；
- `last_output_token_v2` 的 TPOT 只计算首 token 到最后 output token，HTTP EOF/cleanup
  单独记录；
- 第二轮必须精确命中 block-aligned materialized history，不能只凭 conversation ID
  推断命中。

设计合同和命令见：

- 本报告第 1 节（冻结 workload 和比较口径）；
- `test/baseline/pap/README.md`；
- `test/baseline/pap/references/qwen3_8b_chat_16k_2turn_o256_c1_v1/`。

## 2. Legacy PD reference（HTTP EOF v1）

原始目录：

```text
benchmarks/pap/experiments/legacy/runs/
  20260712_031855_d341f7e3e_pd_multiturn_formal/
```

PD 使用未修改的官方 multi-turn proxy 和 NIXL producer/consumer one-way 模式。当前
streaming Chat API 不返回 Decode KV handle，因此两轮 proxy lookup 均为 `MISS`。这并不
等于第二轮没有缓存：每次 repetition 都从 P/D `/metrics` 证明以下精确 token 来源守恒：

| 节点 | local compute | local cache | external NIXL | 总计 |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 16,420 | 16,016 | 0 | 32,436 |
| Decode | 0 | 16,272 | 16,164 | 32,436 |

Decode 的 `16,272` local hit 包含上一轮生成得到的 256 个 Decode-derived tokens；external
累计量大于首轮 boundary，证明第二轮 P→D transfer 也实际发生。

以下三次正式中位数包含最后 token 到 EOF 的尾部时间，只保留作历史证据：

| Round | TTFT (ms) | TPOT (ms) | Latency (ms) |
| --- | ---: | ---: | ---: |
| 1 | 8,250.232 | 25.083 | 14,646.414 |
| 2 | 269.013 | 25.163 | 6,685.235 |

## 3. Legacy 初始 PAP reference：NIXL mailbox（HTTP EOF v1）

原始目录：

```text
benchmarks/pap/experiments/legacy/runs/
  20260712_032326_3ec26b314_pap_multiturn_formal/
```

三次均精确命中第二轮 `16,272` cached tokens，只计算 146 个新增 prompt tokens，Attention
session 全部 drain，fatal-log audit 为零。

| Round | TTFT (ms) | TPOT (ms) | PAP/PD TTFT | PAP/PD TPOT |
| --- | ---: | ---: | ---: | ---: |
| 1 | 6,496.455 | 56.487 | 0.787x | 2.252x |
| 2 | 278.483 | 55.967 | 1.035x | 2.224x |

结论：第二轮 TTFT 已接近 PD，主要缺口是 TPOT。`<2x PD` 的边界为
`<50.327 ms/token`，初始 PAP 还需减少至少 `5.640 ms/token`。

## 4. NIXL mailbox trace 归因

诊断目录：

```text
benchmarks/pap/experiments/legacy/runs/20260712_north_star_trace_baseline/
```

Trace 会逐层同步并把第二轮 TPOT 扰动到 `65.065 ms`，因此只用于路径分解，不作为性能
结果。16K decode 每层中位数如下：

| 区段 | 中位耗时 |
| --- | ---: |
| Projection QKV/norm/RoPE | 0.084 ms |
| Projection mailbox send | 0.419 ms |
| Projection recv/wait | 0.827 ms |
| Projection o-proj | 0.032 ms |
| Attention paged-FA wall/kernel | 0.234 / 0.217 ms |
| Projection layer total | 1.517 ms |

Projection 的 QKV mailbox publish 中 `pack_ms` 中位数为 `0.247 ms/layer`。源码确认这里
主要是 direct send buffer 的 CUDA ready-event CPU synchronize，而不是 12 KB 原始 P2P
带宽；36 层累计约 8.9 ms/token。真正 paged FlashAttention kernel 约 7.8 ms/token，不能
解释 NIXL lane 的全部差距。

该结果与已有 same-node 设计结论一致：瓶颈是每层重复的 CPU 同步、publish/notify 和
Projection↔Attention RPC 边界，而不是 raw P2P copy。

## 5. 首轮优化：same-node local_fast

受控 transport bundle quick A/B：

```text
PAP_OFFLOAD_EXEC_TRANSPORT: nixl_mailbox -> local_fast
PAP_DIRECT_MAILBOX_OUTPUT: 0 -> 1（由 local_fast 默认派生）
```

原始目录：

```text
benchmarks/pap/experiments/legacy/runs/20260712_north_star_local_fast_quick/
```

`local_fast` bundle 复用现有 CUDA-IPC/P2P ring、固定二进制 doorbell、step/layer plan
cache 和 direct output，删除同节点 NIXL mailbox 的逐层通用协议开销。其余 workload、模型、GPU、
MPS 70/30、KV ownership 和 correctness Gate 不变。

| Round | 初始 PAP TPOT | local_fast TPOT | 相对变化 | local_fast/PD |
| --- | ---: | ---: | ---: | ---: |
| 1 | 56.487 ms | 43.741 ms | -22.56% | 1.744x |
| 2 | 55.967 ms | 38.603 ms | -31.02% | 1.534x |

第二轮 TTFT 同时从 `278.483` 降到 `246.587 ms`，为 PD 的 `0.917x`。缓存仍精确命中
`16,272` tokens，PAP 输出 digest 与初始 PAP reference 完全相同，session drain 和日志
审计均通过。单次 quick 已跨过旧口径的 `<2x PD` 目标，但不能替代三次 formal；修正
计时口径后，PD 和 PAP 都必须重新跑 clean 三重复才能形成正式结论。

## 6. v2 formal 基线：PD vs local-fast PAP

代码基线：`feature/pap @ 7e81e2d10`，tracked worktree clean。原始目录：

```text
benchmarks/pap/experiments/legacy/runs/
  20260712_161402_7e81e2d10_pd_multiturn_formal/
  20260712_162130_7e81e2d10_pap_multiturn_formal/
```

两边均为三次串行重启后的中位数，全部通过 artifact-backed Gate。PD 三次精确满足
Prefill `compute/cache/external=16420/16016/0`、Decode
`0/16272/16164`；PAP 三次第二轮均精确命中 `16272` tokens，只计算 146 个新增 tokens。

| Round | 指标 | PD | PAP | PAP/PD |
| --- | --- | ---: | ---: | ---: |
| 1 | TTFT | 8,483.474 ms | 5,397.499 ms | 0.636x |
| 1 | TPOT | 25.101 ms | 42.923 ms | 1.710x |
| 2 | TTFT | 267.273 ms | 235.388 ms | 0.881x |
| 2 | TPOT | 25.183 ms | 39.128 ms | 1.554x |

第二轮 PAP TPOT 三次为 `39.159 / 38.310 / 39.128 ms`，稳定低于
`2 * PD = 50.366 ms`，余量 `11.239 ms/token`。TTFT 已优于 PD，当前北极星缺口明确是
TPOT：PAP 仍比 PD 多 `13.944 ms/token`。

v2 还证明旧 EOF 口径不是主要性能差距：PD 每轮 post-token tail 约 `0.1–0.3 ms`，PAP
约 `35–41 ms/turn`，折算到 255 个 token 间隔仅约 `0.15 ms/token`。真正的 TPOT 差距仍在
decode forward 热路径，而不是 HTTP cleanup。

三次 PAP 均记录 `slot_plan_hits=8925`、`misses=255`、
`slot_topology_mismatches=1`。该 mismatch 来自首轮合法 chunked Prefill
`4096 -> 8192 -> 12288 -> 16018` 被永久误判，导致首轮完全禁用 cross-layer slot plan；
第二轮稳定 topology 才获得 255 次 layer-0 miss 和 8,925 次跨层 hit。

## 7. Exact-token 正确性边界

- 每个架构内部三次输出稳定；
- 两轮 prompt digest 在 PD/PAP 间完全一致；
- 第一轮 PD/PAP output digest 完全一致；
- 第二轮 PAP output 稳定，但与 PD output digest 不同；
- NIXL mailbox 与 local_fast 的 PAP 第二轮 output digest 相同，说明 transport 切换没有
  引入新的输出变化。

比较器将第二轮差异作为显式 warning。它不隐藏差异，也不把稳定、同 workload 的 timing
作废；跨架构 exact-token parity 作为独立正确性/数值路径问题继续调查。

## 8. Stage A：paged-FA metadata bulk build

提交：`6bc383dab`。正式原始目录：

```text
benchmarks/pap/experiments/legacy/runs/
  20260712_171755_6bc383dab_pap_multiturn_formal/
```

该优化只改变 paged FlashAttention metadata cache miss 的构造方式：把约 1,025 次
逐元素 CUDA tensor 写，替换为一次 padded rows 和一次 `seq_lens` 的 bulk tensor
构造；unpadded cache key、LRU、last-block padding、dtype/device 和 cache-hit 路径均不变。
CPU 单测另用 200 组随机 ragged 输入逐值对照旧实现；GPU1 microbenchmark 的 1,024-block
miss 从约 `7.31 ms` 降到 `0.104 ms`，约 70 倍。

三次 clean formal 均通过 exact cache、routing、session drain、fatal-log 和 artifact hash
Gate，输出 digest 与旧 PAP reference 完全一致。正式中位数如下：

| Round | 指标 | PD | 旧 PAP | Stage A PAP | Stage A/PD | 相对旧 PAP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | TTFT | 8,483.474 ms | 5,397.499 ms | 5,405.125 ms | 0.637x | +0.14% |
| 1 | TPOT | 25.101 ms | 42.923 ms | 35.593 ms | 1.418x | -17.08% |
| 2 | TTFT | 267.273 ms | 235.388 ms | 218.263 ms | 0.817x | -7.28% |
| 2 | TPOT | 25.183 ms | 39.128 ms | 30.585 ms | 1.215x | -21.83% |

第二轮 TPOT 三次为 `30.749 / 30.585 / 30.428 ms`，极差约 1.05%。绝对 PAP/PD
TPOT gap 从 `13.944` 降到 `5.402 ms/token`；两轮 conversation latency 从
`26,602.711` 降到 `22,604.784 ms`，距 PD 仅 `1.047x`。第三次第二轮的 HTTP
post-token tail 为 `105.4 ms`，但 last-token TPOT 和主 latency 仍稳定；该尾部继续作为
诊断项，不混入 v2 TPOT。

slot-plan 计数仍为每次 `hits=8925`、`misses=255`、
`slot_topology_mismatches=1`。因此 Stage A 精确消除了 metadata miss 中的标量写开销，
但未掩盖首轮 chunked-Prefill topology false mismatch；后者仍是独立 Stage B。

## 9. Stage B：generation-aware slot-plan

提交：`c134bc3d9`。正式原始目录：

```text
benchmarks/pap/experiments/legacy/runs/
  20260712_181613_c134bc3d9_pap_multiturn_formal/
```

Stage B 不改 descriptor、doorbell、P2P ring 或 Projection，只修正 Attention Registry 的
slot-plan 生命周期。旧实现把 request 的第一份 block topology 永久设为 canonical，导致
合法的 `4096 -> 8192 -> 12288 -> 16018` chunked Prefill 在第二个 chunk 的 layer 0
被误判为跨层冲突，直到 session 结束都不能恢复。

新状态机以 `prefix_len` 定义 activation，每个 session 维护单调 generation，并在首个
activation 后冻结 36 层 expected set。slot-plan key 绑定：

```text
request_id + session_epoch + prefix_len + activation_generation
  + exact-topology-id + decode_seq_lens snapshot + device
```

因此 request ID 复用、activation 推进和 topology A→B→A 都不能 ABA 命中旧 slot tensor。
新 generation 未覆盖全部 expected layers 时保守禁用；更小 prefix 的迟到 import、冻结后
出现新 layer、未完成 generation 继续推进都会 fail closed。同一 generation 内的真实
topology 冲突仍只计数一次并锁存，只有进入新 activation 才能清除。

正式中位数如下：

| Round | 指标 | PD | Stage A | Stage B | Stage B/PD | 相对 Stage A |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | TTFT | 8,483.474 ms | 5,405.125 ms | 5,451.346 ms | 0.643x | +0.86% |
| 1 | TPOT | 25.101 ms | 35.593 ms | 30.521 ms | 1.216x | -14.25% |
| 2 | TTFT | 267.273 ms | 218.263 ms | 224.747 ms | 0.841x | +2.97% |
| 2 | TPOT | 25.183 ms | 30.585 ms | 30.780 ms | 1.222x | +0.64% |

第一轮 TPOT 三次为 `30.542 / 30.521 / 30.414 ms`，第二轮为
`30.780 / 30.455 / 30.781 ms`。Stage B 把原本只覆盖第二轮的 slot-plan 扩展到两轮，
三次计数均精确从 `hits/misses/mismatch=8925/255/1` 变为
`17850/510/0`，且 `fallback=0`。第一轮获得预期的 14.25% TPOT 收益；第二轮变化
`+0.64%`，在约 1.06% 的三次极差内，比较器因此将整体分类为 `neutral` 而非
`improved`。

两轮 conversation latency 从 Stage A 的 `22,604.784 ms` 降到
`21,267.626 ms`（-5.91%），为 PD 的 `0.985x`。当前两轮 TPOT 均约为 PD 的
`1.22x`，剩余绝对 gap 约 `5.4–5.6 ms/token`。

正确性方面，三次均精确命中第二轮 `16,272` tokens、只计算 146 tokens，输出 digest
与 Stage A reference 完全一致，routing、session drain、fatal-log 和 artifact Gate 全部
通过。`PAP_PREFILL_KV_ASYNC=1` 不属于本阶段保证范围：当前 north-star 默认关闭异步
import；在 descriptor unified 字段透传、readiness failed 标记和 queue session-epoch guard
补齐前，不应宣称异步 import 已 ABA-safe。

## 10. Stage C：topology-token metadata fast key

提交：`0727ed946`。正式原始目录：

```text
benchmarks/pap/experiments/legacy/runs/
  20260712_201947_0727ed946_pap_multiturn_formal/
```

Stage A 消除了 metadata cache miss 上约 1,025 次逐元素 CUDA 写，但 Stage B 的 cache-hit
路径仍在每一层把约 1,024 个 block IDs 转成 Python tuple，再参与完整 key 查询。CPU
microbenchmark 中，该 hit 路径每层约 `0.059461 ms`。Stage C 为每个已确认 topology 分配
进程级单调、不复用的 `slot_topology_id`，并用以下轻量 key 查询稳定 activation：

```text
(device, ordered[(slot_topology_id, seq_len)])
```

只有所有 request state 都持有有效 token 时才走 fast key；unknown/mixed state 继续使用完整
block-table key，保持 fail-closed。`seq_len` 和 row order 仍属于 key，避免不同 decode
snapshot 或 batch 排列误命中。开关 `PAP_UNIFIED_MD_FAST_KEY` 默认为 1，可在严格 A/B 中
关闭。

实现审计还发现 metadata `OrderedDict` 是进程全局对象，而多个 peer worker 会并发执行未
加锁的 `get -> move_to_end`。另一个线程在两步之间淘汰 entry 时可触发 `KeyError`。Stage C
同时为 metadata/CU cache、LRU 和统计加锁；tensor 构造仍在锁外，写回时重新检查并复用
并发创建的 entry。topology ID 分配器使用独立锁，metadata cache reset 不会重置它，因而
release/re-register 后不能 ABA 命中旧 resident entry。

### 10.1 同代码严格 A/B

六次 dirty controlled runs 按 `OFF1/ON1/OFF2/ON2/OFF3/ON3` 交替执行，唯一主变量为
`PAP_UNIFIED_MD_FAST_KEY`。原始目录为
`benchmarks/pap/experiments/legacy/runs/20260712_stagec_{off,on}{1,2,3}`。三对结果都通过 exact
cache、routing、session、fatal-log 和输出 digest Gate。

| 指标 | OFF 中位数 | ON 中位数 | ON 相对 OFF | 三组 paired 变化 |
| --- | ---: | ---: | ---: | --- |
| R1 TTFT | 5,470.512 ms | 5,441.417 ms | -0.53% | -0.56%、-1.68%、+0.41% |
| R1 TPOT | 30.425 ms | 30.193 ms | -0.76% | -0.76%、-0.12%、-1.45% |
| R2 TTFT | 224.007 ms | 216.548 ms | -3.33% | -3.49%、-3.36%、-1.85% |
| R2 TPOT | 30.848 ms | 30.419 ms | -1.39% | -1.39%、-1.33%、-1.41% |
| Conversation | 21,387.490 ms | 21,167.134 ms | -1.03% | -1.08%、-1.03%、-1.14% |

OFF 每次完整扫描 `18,432` 次、共读取 `18,994,176` 个 block IDs；ON 每次只有 `512`
次完整扫描、读取 `527,616` 个 block IDs，精确减少 `36x`。fast-key lookup/hit 为
`18,432/17,920`，metadata hit/miss 仍为 `17,920/512`，说明只改变查找成本，没有改变
cache 语义。CPU hit microbenchmark 降至约 `0.001138 ms/layer`，约 `52x`。

### 10.2 clean formal 与结论

三次正式运行均绑定 clean commit、effective config 和 fast-key runtime counters；输出
digest、第二轮 `16,272` cached / `146` computed tokens、slot-plan
`17,850/510/0` 与 Stage B 完全一致。

| Round | 指标 | PD | Stage B | Stage C | Stage C/PD | 相对 Stage B |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | TTFT | 8,483.474 ms | 5,451.346 ms | 5,460.711 ms | 0.644x | +0.17% |
| 1 | TPOT | 25.101 ms | 30.521 ms | 30.196 ms | 1.203x | -1.06% |
| 2 | TTFT | 267.273 ms | 224.747 ms | 224.491 ms | 0.840x | -0.11% |
| 2 | TPOT | 25.183 ms | 30.780 ms | 30.449 ms | 1.209x | -1.08% |

Conversation latency 为 `21,228.436 ms`，相对 Stage B 降 `0.18%`，为 PD 的
`0.984x`。正式变化小于比较器 3% 阈值，因此分类仍是 `neutral`。它没有达到最初期望的
3% TPOT 收益，但严格 paired A/B 的 R2 TPOT 三次均稳定改善约 1.4%，同时删除了 36 倍
重复扫描并修复真实并发 LRU race，因此保留为默认并晋升 reference；不能把它表述为
显著性能突破。

## 11. Stage D v1：Attention 进程低扰动 GPU critical-chain trace

提交：`ad95c8c12`。诊断原始目录：

```text
benchmarks/pap/experiments/legacy/runs/
  20260712_stagec_deferred_gpu_trace_v1/
```

旧 trace 在每层调用 CUDA event `synchronize()`，会把第二轮 TPOT 从约 `30.45 ms`
扰动到 `37.13 ms`，约增加 22%，只能说明顺序，不能可靠分账剩余 gap。Stage D v1
增加默认关闭的 deferred CUDA-event collector：热路径只 record/query/reuse event，不做
同步；只有所有 Attention session drain 后的 stats capture 才允许 blocking flush。collector
按线程隔离、内部加锁，event pair 用 FIFO deque 摊销回收。

当前 scope 明确限定为 `attention_process_critical_chain`，只记录同一 Attention 进程可完整
导出的四段，不声称覆盖 Projection：

1. QKV receiver 上的 stream-ordered ready wait；
2. `reshape_and_cache_flash` KV append；
3. paged FlashAttention；
4. Attention sender 上的 output P2P copy。

finalizer 在开关启用时逐 Attention 实例 fail closed：要求 collector 存在，pending/drop/error
均为 0，四个 span 均存在，并分别精确匹配 peer batch、实际 append 和 compute counter。
其中首轮每层有一次无需追加 KV 的合法 compute，故 KV append 合同是
`fast_path_hits + fallbacks`，不是全部 compute calls。

单次 diagnostic quick 的 cache、routing、session、strict log 和输出 digest Gate 全部通过。
四段计数精确为 `18432 / 18360 / 18432 / 18432`，pending/drop/error 均为 0。相对当前
clean PAP reference 的扰动如下：

| Round | clean PAP TPOT | deferred trace TPOT | 相对扰动 |
| --- | ---: | ---: | ---: |
| 1 | 30.196 ms | 30.836 ms | +2.12% |
| 2 | 30.449 ms | 30.987 ms | +1.77% |

每层 GPU duration 分布为：

| Attention 进程区段 | mean | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: |
| QKV ready wait | 0.545 ms | 0.567 ms | 0.572 ms | 0.577 ms |
| KV append | 0.009 ms | 0.008 ms | 0.012 ms | 0.017 ms |
| paged FlashAttention | 0.192 ms | 0.191 ms | 0.191 ms | 0.218 ms |
| output P2P copy | 0.007 ms | 0.007 ms | 0.010 ms | 0.011 ms |

结论是 KV append、FA kernel 和 output raw P2P copy 都不能解释剩余约
`5.27 ms/token`；最大可见区段已经收敛到 Projection→Attention 的 QKV ready chain。
但 `0.567 ms/layer` 不能直接称为通信开销：doorbell 在 Projection enqueue copy/signal
后即可到达，Attention stream wait 还会包含源 QKV projection、norm/RoPE、源 tensor
readiness 和 MPS/CUDA 调度尾部，而这些计算的一部分在 PD 中同样存在。

因此下一步不是直接改 copy API，而是用同一种 deferred event 方法从 Projection 进程导出
`pre-attention compute`、QKV P2P copy 和 output ready wait，再把 QKV ready chain 拆成
“PD 也必须支付的计算”与“PAP 新增的 handoff/scheduling”。只有后者才是下一轮优化预算。

## 12. 当前决策与后续

1. north-star PAP runner 显式固定 `local_fast`，避免依赖手工环境变量；
2. v2 PD reference 保持 `7e81e2d10`；当前 PAP 默认实现和 reference 为 Stage C
   `0727ed946`；
3. Stage A bulk metadata、Stage B generation-aware slot-plan 和 Stage C topology-token
   fast key 均已完成 clean formal；
4. Stage D v1 已排除 KV append、FA 本体和 output raw P2P copy；下一步只拆解
   Projection→Attention QKV ready chain，并区分必要 Projection 计算与 PAP 新增等待；
5. 不做 MPS、ring slot 或 copy API 扫描；若 QKV source compute/copy 之外仍有稳定
   stream-wait residual，再做 source-stream handoff 单变量 A/B；若 residual 很小，则剩余
   gap 是 C1 下的逐层 stage bubble，应转向 cohort/pipeline 而不是继续微调通信；
6. async prefill import 的 descriptor/epoch/readiness 修复单列为正确性工作，不与同步
   north-star 性能优化混跑。
