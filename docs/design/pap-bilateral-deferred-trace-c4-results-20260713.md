# PAP/PD 双侧 Deferred Trace C4 结果

日期：2026-07-13

状态：实现与四组单次 C4 诊断实验完成。Trace 默认关闭；本报告用于确定下一轮
TPOT 优化顺序，不替代三次拉丁方冻结的正式北极星结果。

## 1. 结论摘要

固定 70:30 的同一 C4 负载下，trace-off 的 PAP/PD-twoway 稳态 TPOT 为
`51.098/42.179 ms`，PAP 为 PD 的 `1.211x`，绝对差距 `8.919 ms/token`。

双侧 trace 把剩余差距收敛到两个优先方向：

1. PAP Projection 每个 engine forward 在构造 Attention 元数据前，会把当前
   `input_ids` 从 GPU 同步复制到 CPU。该边界平均阻塞 `25.254 ms/forward`；按
   `1722` 个 forward、`5120` 个 request-token 摊销为 `8.494 ms/request-token`。
   这不是 8 字节 token 的物理拷贝时间，而是一次把 Projection GPU stream 上未完成
   工作显式结算的同步屏障。它很可能破坏 vLLM async scheduling 原本允许的跨
   iteration CPU enqueue。
2. 相同 QKV/norm/RoPE 边界下，PAP 与 PD 只差 `0.264%`；但 PAP paged
   FlashAttention 平均比 PD 慢 `28.707%`，按 36 层约多 `4.485 ms/forward`。在
   batch shape 足够可比的近似下，这相当于 clean TPOT 差距的约 `50.3%`。

raw QKV/output P2P copy 合计约 `0.583 ms/forward`，无法解释 `8.919 ms/token`
差距。下一轮不应继续优先微调 memcpy，而应先移除 token-boundary 同步，再复测
Attention 的剩余差距。

## 2. 实现范围

提交 `e115fc86f720a4d3f148072718976b1ef6d58d5c` 增加了默认关闭的双侧
deferred trace：

- PAP Projection：QKV/norm/RoPE、Q/K repack、QKV source P2P copy、output
  doorbell host wait、output-ready GPU wait、token-boundary input-ID D2H；
- PAP Attention：沿用 QKV-ready wait、KV append、paged FA、output P2P copy；
- PD-twoway Decode：与 PAP 相同代码边界的 QKV/norm/RoPE，以及标准 paged FA；
- runner 在 workload 和 session drain 后才触发 blocking flush；
- validator 对 role、scope、必需 span、count、36 层整除和
  pending/drop/error 做 fail-closed 校验。

`PAP_DEFERRED_CUDA_TRACE=0` 仍为默认。关闭时不会创建 CUDA Event、exporter
线程或结果文件。

## 3. 固定负载与实验矩阵

四组实验只改变架构和 trace 开关：

| Cell | 架构 | Trace | MPS |
| --- | --- | --- | --- |
| A | PAP local-fast 1PA1P | off | Prefill/Attention `70:30` |
| B | PAP local-fast 1PA1P | on | Prefill/Attention `70:30` |
| C | PD-twoway 1P1D，NIXL/UCX 1.22 | off | 不适用 |
| D | PD-twoway 1P1D，NIXL/UCX 1.22 | on | 不适用 |

共同配置：

- 模型：本地 Qwen3-8B，FP16，TP1；
- GPU：GPU1/GPU2；
- 第一轮 document `16000` tokens；后续每轮 append `120` tokens；
- 每轮输出 `256` tokens，共 `5` 轮；
- `4` 条 active conversations，共 `20` 个请求、`5120` 个输出 token；
- 每轮固定速率 QPS `2`，round barrier closed loop；
- `MAX_NUM_SEQS=4`、`max_model_len=20000`、
  `max_num_batched_tokens=4096`；
- temperature `0`、seed `0`、ignore EOS；
- 每个 cell 单次，按 `quick/diagnostic` 证据等级解释。

## 4. 性能结果

所有数值是 request-level median，单位为 ms。

| Cell | R1 TTFT | R1 TPOT | R2-R5 TTFT | R2-R5 TPOT |
| --- | ---: | ---: | ---: | ---: |
| PAP trace-off | 10997.429 | 39.067 | 253.742 | 51.098 |
| PAP trace-on | 10979.116 | 40.876 | 245.031 | 52.269 |
| PD-twoway trace-off | 8120.238 | 35.550 | 238.845 | 42.179 |
| PD-twoway trace-on | 8099.082 | 36.027 | 247.645 | 42.558 |

### 4.1 Clean PAP/PD 对比

| Scope | PAP - PD | PAP / PD | PAP 相对 PD |
| --- | ---: | ---: | ---: |
| R1 TTFT | +2877.192 ms | 1.354x | +35.43% |
| R1 TPOT | +3.517 ms | 1.099x | +9.89% |
| R2-R5 TTFT | +14.897 ms | 1.062x | +6.24% |
| R2-R5 TPOT | +8.919 ms | 1.211x | +21.15% |

本次 trace-off 单次结果与冻结的三次拉丁方 steady TPOT
`PAP=51.148 ms`、`PD-twoway=42.155 ms` 分别只差约 `-0.10%` 和 `+0.06%`，
说明 control cell 落在既有稳定区间。

### 4.2 Trace 扰动

| 架构 | R1 TPOT 扰动 | R2-R5 TPOT 扰动 |
| --- | ---: | ---: |
| PAP | +4.63% | +2.29% |
| PD-twoway | +1.34% | +0.90% |

PAP steady 扰动比设计中的精确预算阈值 `2%` 高 `0.29` 个百分点。因此 PAP
span 可作为强方向性证据，但不能把每段均值机械相加成精确 TPOT 预算；PD trace
扰动在阈值内。

## 5. 双侧分账

### 5.1 PAP Projection

Projection 共有 `61992` 个 layer calls，即 `1722` 个 36-layer engine forwards。
六个必需 span count 对齐，pending/drop/error 均为 `0`。

| Span | mean ms/layer 或 forward | p50 | 近似 mean × 36 |
| --- | ---: | ---: | ---: |
| QKV/norm/RoPE | 0.082849/layer | 0.082784 | 2.983/forward |
| Q/K repack | 0.006325/layer | 0.006240 | 0.228/forward |
| QKV source P2P copy | 0.008448/layer | 0.008544 | 0.304/forward |
| output doorbell host wait | 0.307348/layer | 0.209966 | 11.065/forward |
| output-ready GPU wait | 0.588901/layer | 0.637152 | 21.200/forward |
| token-boundary input-ID D2H | 25.254/forward | 27.072 | 不乘 36 |

output doorbell 的 mean 被一次 `1072.364 ms` 启动/长尾值明显拉高；其 p50 乘
36 为约 `7.559 ms/forward`。该 host wait 覆盖 Attention 端收到 QKV、构造并
enqueue 计算到发布 output doorbell 的过程，不是纯 Python poll 成本。

### 5.2 PAP Attention

| Span | mean ms/layer | 近似 mean × 36 |
| --- | ---: | ---: |
| QKV-ready GPU wait | 0.597329 | 21.504/forward |
| KV append | 0.009108 | 0.328/forward |
| paged FlashAttention | 0.558579 | 20.109/forward |
| output P2P copy | 0.007741 | 0.279/forward |

Attention trace 同样为零 pending/drop/error。QKV-ready wait 与 Projection 的
output-ready wait 是同一 producer-consumer 交替链的两侧视图，不能相加为独立开销。

### 5.3 PD-twoway Decode

PD 有 `61848` 个 layer calls，即 `1718` 个 36-layer forwards；两个必需 span count
完全一致，pending/drop/error 均为 `0`。

| Span | mean ms/layer | 近似 mean × 36 |
| --- | ---: | ---: |
| QKV/norm/RoPE | 0.082631 | 2.975/forward |
| paged FlashAttention | 0.433994 | 15.624/forward |

## 6. 瓶颈判断

### 6.1 已排除：Projection QKV 数学计算

PAP/PD 的同代码边界为 `0.082849/0.082631 ms/layer`，PAP 只慢 `0.264%`，
36 层只差约 `0.008 ms/forward`。Projection QKV、Q/K norm 和 RoPE 不是当前
TPOT 差距来源。

### 6.2 次要：raw P2P copy

QKV source copy 与 Attention output copy 合计约 `0.583 ms/forward`，只相当于
clean `8.919 ms/token` 差距的约 `6.5%`。这不代表通信链没有优化空间，而是说明
继续只改 memcpy、P2P 带宽或 copy tensor 形态不会成为主收益。

### 6.3 主要候选一：token-boundary 同步屏障

`input_batch.input_ids` 中的 decode token 由 GPU 上的
`combine_sampled_and_draft_tokens` 写入。PAP 为把 token ID 放入 Attention
descriptor，在每个 forward 开始前执行 `.to(cpu).tolist()`，因此 CPU 必须等待
Projection main stream 到达该点。

`25.254 ms/forward × 1722 forwards / 5120 request-tokens =
8.494 ms/request-token`。这个数与 clean TPOT 差距接近，但两者不是可相加的独立
区段：D2H wall time包含此前尚未完成的 GPU 计算。它真正证明的是当前实现存在一个
强制 stream barrier；删除这行 copy 后，部分时间可能只是移动到别处，只有严格 A/B
才能确定净收益。

从源码语义看，token ID 不参与 FlashAttention 数学计算，只用于
`DecodeCommitClient` 把已经追加的 decode KV 更新到 Prefill 的 token/hash/cache
状态。因此它可以从逐层计算元数据中解耦，并在计算后异步提交。

### 6.4 主要候选二：Attention paged FA

PAP/PD paged FA 为 `0.558579/0.433994 ms/layer`，PAP 慢 `28.707%`；36 层
差约 `4.485 ms/forward`。这约占 clean 差距的 `50.3%`，但当前 PAP/PD forward
数为 `1722/1718`，batch composition 也可能略有不同，所以仍需按 batch size 和
sequence-length bucket 做 matched-shape 复验。

在当前 70:30 配置中，Attention 进程受 30% MPS 上限约束，而 PD Decode 使用完整
GPU；这是一个明确的结构性解释。下一步仍应保持 70:30，通过 matched-shape 证据判断
剩余部分究竟来自 MPS 硬上限、IPC KV layout，还是 PAP Attention metadata/admission。

### 6.5 不可重复计费的等待

- Projection output doorbell host wait 包含 Attention CPU 侧准备和 enqueue；
- Projection output-ready GPU wait 包含 GPU 尚未完成的 Attention/output copy；
- Attention QKV-ready wait 是相反方向同一交替链；
- token-boundary D2H 又会结算 Projection stream 上尚未完成的工作。

这些区段大量重叠。报告只用它们定位同步点和 producer-consumer idle，不将它们相加
宣称“解释了超过 100% 的 TPOT”。

## 7. 推荐的下一轮优化顺序

### A. Deferred decode-token commit（推荐先做）

目标是从 `_pap_forward_context_kwargs()` 删除同步 input-ID D2H，同时保留当前
correctness、APC 和 lease 语义：

1. Projection 继续使用 GPU-resident `last_sampled_tokens` 驱动下一个 forward；
2. 复用 vLLM 已有 `AsyncOutput` 的异步 sample-token D2H，而不再从下一次
   `input_ids` 重复同步读取；
3. 将 `(request_id, step, token_id)` 作为独立、可滞后的 token inbox 发给 Attention；
4. Attention 在“首层 KV append 已发生”和“对应 token 已到达”两个条件都满足后，
   调用现有可靠 `DecodeCommitClient`；
5. finish/preempt/drain 必须等待该 request 的 pending token/commit 清零。

该方案直接针对 8.494 ms/request-token 的 barrier 信号，并保留 async scheduling 的
CPU enqueue 空间。若 CPU token inbox 仍形成反压，再升级为一次/step 的 GPU token
sideband，而不是在 36 层 QKV metadata 中重复携带 token。

### B. Matched-shape Attention FA 归因

在 A 的严格 C4 trace-off A/B 后，为 PAP/PD FA trace 附加低成本的 batch-size 和
sequence-length bucket；只比较同 bucket 的 FA。若约 `4.5 ms/forward` 差距仍在，
再分别检查 30% MPS ceiling、IPC KV view/layout 和 FA launch/admission。此阶段固定
70:30，不做比例扫描。

### C. Projection output descriptorless receive

Projection 已知自己发送的 batch、预期 output shape、seq 和 ring slot。可以为稳定
plan 增加 fail-closed fast path：先 enqueue GPU `wait_value32`，把 doorbell metadata
校验移出关键 CPU 路径，避免每层先 spin 到 doorbell 再 enqueue GPU wait。该方向可能
回收 p50 约 `0.210 ms/layer` 的 host enqueue bubble，但必须在 A 后重测，避免优化一个
被 token barrier 掩盖的次级症状。

## 8. 正确性、验证与原始证据

四个 cell 均满足：

- aggregate/client/cache/external validity 全部 `passed`；
- 每组 `20/20` 请求完成；
- PAP correctness、routing、session drain 全通过；
- PD correctness/cache gate 全通过；
- 四组 prompt/output token digest 一致；
- traced runs 所有必需 span count 对齐，无 pending/drop/error。

实现验证：完整聚焦回归 `60 passed, 2 skipped`；归档前再次运行 bilateral core
测试为 `30 passed`，两个实际 trace artifact 也重新通过 validator。Python
`py_compile`、两个 runner 的 `bash -n` 和 `git diff --check` 均通过。按用户要求
未运行 pre-commit。

原始目录（`repo-untracked`）：

- `test/baseline/pap/results/runs/20260713_e115fc86f_pap_bilateral_trace_off_c4`；
- `test/baseline/pap/results/runs/20260713_e115fc86f_pap_bilateral_trace_on_c4`；
- `test/baseline/pap/results/runs/20260713_e115fc86f_pd_twoway_bilateral_trace_off_c4`；
- `test/baseline/pap/results/runs/20260713_e115fc86f_pd_twoway_bilateral_trace_on_c4`。

关键 trace artifact：

- PAP Projection：`rep1/projection_deferred_trace.json`；
- PAP Attention：`rep1/attention_fast_path_stats.json`；
- PD Decode：`rep1/pd_decode_deferred_trace.json`。
