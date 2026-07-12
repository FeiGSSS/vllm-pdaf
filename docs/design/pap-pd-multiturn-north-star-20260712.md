# PAP/PD 1:1 多轮北极星基线与首轮优化

日期：2026-07-12

状态：发现并修正旧 TPOT 的 HTTP EOF 口径；v2 formal reference 待重建

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

设计和命令见：

- `docs/superpowers/specs/2026-07-12-pap-pd-multiturn-north-star-testbed-design.md`；
- `test/baseline/pap/README.md`；
- `test/baseline/pap/references/qwen3_8b_chat_16k_2turn_o256_c1_v1/`。

## 2. Legacy PD reference（HTTP EOF v1）

原始目录：

```text
test/baseline/pap/results/runs/
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
test/baseline/pap/results/runs/
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
test/baseline/pap/results/runs/20260712_north_star_trace_baseline/
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
test/baseline/pap/results/runs/20260712_north_star_local_fast_quick/
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

## 6. Exact-token 正确性边界

- 每个架构内部三次输出稳定；
- 两轮 prompt digest 在 PD/PAP 间完全一致；
- 第一轮 PD/PAP output digest 完全一致；
- 第二轮 PAP output 稳定，但与 PD output digest 不同；
- NIXL mailbox 与 local_fast 的 PAP 第二轮 output digest 相同，说明 transport 切换没有
  引入新的输出变化。

比较器将第二轮差异作为显式 warning。它不隐藏差异，也不把稳定、同 workload 的 timing
作废；跨架构 exact-token parity 作为独立正确性/数值路径问题继续调查。

## 7. 当前决策与后续

1. north-star PAP runner 显式固定 `local_fast`，避免依赖手工环境变量；
2. 从 clean commit 按 `last_output_token_v2` 分别重建 PD/PAP 三次 formal，确认
   `<2x PD` 是否有稳定余量；
3. formal 通过后晋升 PAP reference，并保留初始 mailbox reference 数据和本记录；
4. 再用 local_fast trace 分解剩余约 `13.44 ms/token` 的 PAP/PD 差距；
5. 不做 MPS 扫描；优先分析 16K Attention compute、MPS 硬配额和跨进程逐层同步的剩余
   下限。
