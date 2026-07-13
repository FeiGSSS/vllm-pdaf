# PAP 1:1 多轮性能优化阶段总结

> **历史口径说明（2026-07-13）**：本文保留 7 月 12 日 PAP Stage A–D 的内部优化
> 演进证据，但其中 PD 使用了后来确认发生 TCP emulation 的旧 NIXL pull 路径，不再作为
> 当前公平 PAP/PD 性能基线。校正后的正式阶段结论见
> [PAP 1PA1P 五轮长上下文阶段总结与汇报](pap-1pa1p-five-turn-stage-report-20260713.md)。

日期：2026-07-12

分支：`feature/pap`

阶段状态：冻结 1PA1P 多轮性能优化成果；正式 PAP reference 为 `0727ed946`，当前
HEAD 只在其上增加默认关闭的诊断能力和文档。

## 1. 汇报结论

本阶段已经把固定 1:1、16K 两轮对话负载下的 PAP TPOT 稳定收敛到 PD 的约
`1.20–1.21x`，优于阶段目标“控制在 PD 的 2 倍以内”，也优于阶段性表述“约 1.25x”。

- Round 1：PAP/PD TPOT = `30.196 / 25.101 ms = 1.203x`；
- Round 2：PAP/PD TPOT = `30.449 / 25.183 ms = 1.209x`；
- Round 1/2 TTFT 均优于 PD，分别为 PD 的 `0.644x / 0.840x`；
- 两轮 conversation latency 为 `21,228.436 ms`，是 PD 的 `0.984x`；
- 三次 PAP formal 的 TPOT 极差分别只有 `0.69% / 0.29%`，结果稳定；
- 第二轮精确命中 `16,272` tokens，只计算 146 个新增 prompt tokens；
- 三次 formal 均通过 cache、routing、session drain、strict log 和 artifact provenance
  Gate。

因此建议把当前结果作为 1PA1P 多轮阶段性 baseline 冻结。后续不再为了追逐小幅 TPOT
数字而扫描 MPS、ring slot 或 copy API；只有业务目标要求继续压缩约 20% 的 TPOT gap
时，再启动独立的 Projection→Attention QKV readiness 研究阶段。

## 2. 固定比较口径

正式结果使用 profile `qwen3_8b_chat_16k_2turn_o256_c1_v1`：

- 模型：本地 Qwen3-8B，FP16，TP1；
- 硬件：NVIDIA L20 × 2，GPU 1/2；
- PAP：1PA1P，Prefill/Attention 固定 MPS 70/30；
- PD：1P1D，官方 one-way NIXL producer/consumer 路径；
- Round 1：16,000 document tokens；
- Round 2：追加 120 tokens，并复用第一轮 prompt 和 decode KV；
- 每轮输出 256 tokens，C1 closed loop，thinking 开启；
- 每个 formal repetition 完整重启服务，三次中位数为正式值；
- TPOT 使用 `last_output_token_v2`，不把最后 token 后的 HTTP EOF/cleanup 混入 TPOT。

PD 与 PAP 使用同一 profile fingerprint、模型、dtype、GPU 数、输入输出和 token 计时定义。

## 3. 正式性能矩阵

原始 formal 目录：

```text
PD:
test/baseline/pap/results/runs/
  20260712_161402_7e81e2d10_pd_multiturn_formal/

PAP:
test/baseline/pap/results/runs/
  20260712_201947_0727ed946_pap_multiturn_formal/
```

| Round | 指标 | PD | PAP | PAP/PD | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | TTFT | 8,483.474 ms | 5,460.711 ms | 0.644x | PAP 低 35.63% |
| 1 | TPOT | 25.101 ms | 30.196 ms | 1.203x | PAP 高 20.30% |
| 2 | TTFT | 267.273 ms | 224.491 ms | 0.840x | PAP 低 16.01% |
| 2 | TPOT | 25.183 ms | 30.449 ms | 1.209x | PAP 高 20.91% |

端到端两轮结果：

| 指标 | PD | PAP | PAP/PD |
| --- | ---: | ---: | ---: |
| Conversation latency | 21,581.358 ms | 21,228.436 ms | 0.984x |
| Conversation EOF latency | 21,581.469 ms | 21,284.156 ms | 0.986x |

PAP TPOT 与 PD 仍有 `5.095 ms/token`（Round 1）和 `5.266 ms/token`（Round 2）的
绝对差距，但 TTFT 优势抵消了这部分 decode 差距，使固定两轮总 latency 略优于 PD。

## 4. 稳定性

### 4.1 PAP 三次 formal

| Round | Rep 1 | Rep 2 | Rep 3 | Median | 极差/Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 TPOT | 30.393 ms | 30.184 ms | 30.196 ms | 30.196 ms | 0.69% |
| R2 TPOT | 30.449 ms | 30.385 ms | 30.474 ms | 30.449 ms | 0.29% |

三次 conversation latency 为 `21,228.436 / 21,196.169 / 21,252.106 ms`。三次均
精确命中第二轮缓存边界，并保持相同 PAP 输出签名。

### 4.2 PD 三次 formal

PD R1 TPOT 为 `25.101 / 25.118 / 25.093 ms`，R2 TPOT 为
`25.183 / 25.203 / 25.169 ms`。PD 参考同样稳定，因而 PAP/PD 比值不是由单次异常样本
造成。

## 5. 优化阶段与贡献

正式 v2 local-fast baseline 到 Stage C 的演进如下：

| 阶段 | 主要变化 | R1 TPOT | 相对前一阶段 | R2 TPOT | 相对前一阶段 |
| --- | --- | ---: | ---: | ---: | ---: |
| v2 baseline | 固定 local-fast 与严格 testbed | 42.923 ms | — | 39.128 ms | — |
| Stage A | paged-FA metadata bulk build | 35.593 ms | -17.08% | 30.585 ms | -21.83% |
| Stage B | generation-aware slot plan | 30.521 ms | -14.25% | 30.780 ms | +0.64% |
| Stage C | topology-token metadata fast key | 30.196 ms | -1.06% | 30.449 ms | -1.08% |

累计效果：

- R1 TPOT：`42.923 -> 30.196 ms`，降低 `29.65%`；
- R2 TPOT：`39.128 -> 30.449 ms`，降低 `22.18%`；
- Stage A 删除 metadata miss 的逐元素 CUDA 写；
- Stage B 修复 chunked-Prefill topology false mismatch，使 slot-plan 覆盖两轮；
- Stage C 将完整 block-ID 扫描减少 `36x`，同时修复 process-global LRU 并发 race。

Stage B 的 R2 变化为 `+0.64%`，属于噪声带内 neutral；它的主要价值是修复正确性边界并
把第一轮 slot-plan 覆盖补齐。Stage C 的单阶段收益约 1%，但三对严格 A/B 均同向改善，
并删除了明确的重复工作，因此保留。

## 6. 多轮缓存与正确性

PAP Round 2 的正式缓存账本为：

```text
prompt_tokens   = 16418
cached_tokens   = 16272
computed_tokens = 146
decode-derived hit tokens = 256
```

这证明第一轮 Prefill KV 和完整块内的 256 个 Decode KV 都保留在 PA-owned paged cache，
第二轮无需 KV 回传即可由原生 APC 命中。每轮结束后 PA/P pair 可以解散；下一轮是否命中
同一 PA 是上层 cache-aware routing 的职责。

正式 Gate：

- 三次 PAP session 全部 drain；
- routing audit 和 strict fatal-log audit 全部通过；
- metadata fast-key lookup/hit 为 `18,432 / 17,920`；
- full scans 只有 512 次，检查 block IDs 为 `527,616`；
- slot-plan hits/misses/mismatch 为 `17,850 / 510 / 0`；
- PAP 三次内部输出 token/text digest 完全稳定。

已知边界：两边 prompt digests 和 Round 1 输出完全一致；Round 2 PAP 输出在三次内部稳定，
但与 PD exact-token digest 不同。当前比较器将其作为明确 warning，不隐藏差异，也不把稳定
时延结果作废。若未来要求 PD/PAP 跨架构逐 token 数值完全一致，需要作为独立数值路径
问题继续调查。

## 7. Stage D 诊断结论

Stage D 增加默认关闭的 deferred CUDA-event trace。其 quick 结果不是正式性能 baseline，
但诊断扰动仅为 R1 `+2.12%`、R2 `+1.77%`，远低于旧同步 trace 的约 22%。

trace-on TPOT 为：

- R1：`30.836 ms`，为 PD 的 `1.228x`；
- R2：`30.987 ms`，为 PD 的 `1.230x`。

每层 GPU p50：

| 区段 | p50 |
| --- | ---: |
| QKV ready wait | 0.567 ms |
| KV append | 0.008 ms |
| paged FlashAttention | 0.191 ms |
| output P2P copy | 0.007 ms |

四段计数精确为 `18,432 / 18,360 / 18,432 / 18,432`，pending/drop/error 均为 0。
结果已经排除 KV append、FlashAttention 本体和 output raw P2P copy 是剩余差距的主要
来源。最大区段是 QKV ready chain，但它同时包含 PD 也必须支付的 Projection 计算与 PAP
新增 handoff，不能把全部 `0.567 ms/layer` 都称为通信开销。

基于当前阶段目标，建议在此停止微优化。若后续继续，首先应导出 Projection 侧 source
compute/QKV copy timing 做分账，而不是直接修改 copy API 或扫描 MPS。

## 8. 代码、测试与提交

关键提交：

| Commit | 内容 |
| --- | --- |
| `6bc383dab` | Stage A metadata bulk build |
| `c134bc3d9` | Stage B generation-aware slot plan |
| `0727ed946` | Stage C topology-token metadata fast key |
| `fe1a25f9b` | 晋升 Stage C PAP reference 与文档 |
| `ad95c8c12` | 默认关闭的 deferred CUDA critical-path trace |
| `5c6308658` | Stage D 结果、实验索引和 reference 说明 |

验证结果：

- `tests/pap`：`411 passed, 3 skipped`；
- Stage D/finalizer 定向测试：`21 passed`；
- runner `bash -n`、Python compile 和 `git diff --check` 通过；
- Stage C clean formal：3/3 repetitions 全部通过；
- Stage D GPU quick：正确性、缓存、路由、session drain 和 trace 完整性全部通过。

按照项目约定，本阶段没有运行 pre-commit；提交使用 `--no-verify`。未跟踪的 raw result、
历史 profile 和个人工作目录没有进入提交。

## 9. 阶段决策

1. 将 `0727ed946` 对应的 Stage C 三轮 formal 保持为正式 PAP 性能 reference；
2. 当前分支保留 `ad95c8c12` 的默认关闭诊断能力，不改变正常执行语义；
3. 将 1PA1P 多轮 TPOT `1.20–1.21x PD` 作为本阶段结论；
4. 将 arbitrary X:Y、多 PA cache-aware routing 和跨架构 Round 2 exact-token parity 保留为
   后续独立工作，不与本次 1:1 性能结论混报；
5. 下一阶段如继续性能优化，必须从 Projection 侧 QKV readiness 分账开始，并继续使用同一
   north-star testbed 做单变量 A/B。

## 10. 可直接用于汇报的摘要

本阶段完成了 PAP 1PA1P 多轮性能 testbed、Prefill-owned Decode KV 原生复用以及三轮正式
优化验证。在固定 Qwen3-8B、16K 两轮、两张 L20 的同硬件条件下，PAP Round 1/2 TPOT
分别达到 `30.20/30.45 ms`，稳定为 PD 的 `1.203x/1.209x`；TTFT 分别为 PD 的
`0.644x/0.840x`，两轮总 latency 为 PD 的 `0.984x`。相对初始 v2 local-fast baseline，
PAP 两轮 TPOT 累计降低 `29.65%/22.18%`。第二轮精确命中 `16,272` tokens，其中包含
第一轮生成的 256 个 Decode tokens，三次正式实验均通过完整正确性与生命周期审计。当前
结果已达到阶段目标，建议冻结为 1:1 多轮 baseline；后续将重点转向 arbitrary X:Y、
cache-aware routing 和必要时的 Projection→Attention QKV readiness 深度优化。
