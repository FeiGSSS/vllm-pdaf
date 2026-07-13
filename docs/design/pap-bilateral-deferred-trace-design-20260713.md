# PAP/PD 双侧 Deferred Trace 设计

## 1. 目标

在固定的 Qwen3-8B、1PA1P/1P1D、16K、五轮、C4、o256、QPS 2
北极星负载下，用默认关闭、低扰动、可校验的计时把以下区段放到同一口径：

- PAP Projection 的 QKV projection、Q/K norm 和 RoPE；
- Projection 到 Attention 的 QKV source P2P copy；
- Projection 等待 Attention output ready 的 CPU doorbell 与 GPU stream wait；
- Projection 每个 decode forward 的 input-token GPU→CPU 同步等待；
- PD-twoway Decode 的同口径 QKV/norm/RoPE 和 paged FlashAttention。

本阶段只建立诊断工具和结果，不修改生产调度、MPS 比例、通信协议或数学计算。

## 2. 当前证据

冻结 C4 formal 中，PD-twoway/PAP 稳态 TPOT 分别为 `42.155/51.148 ms`，
差距为 `8.993 ms/token`。现有 Attention-only deferred trace 的 C4 quick TPOT
为 `51.581 ms`，相对 clean PAP 扰动约 `0.85%`。按每次调用均值乘 36 层：

| 区段 | 近似 ms/token | 当前归因边界 |
| --- | ---: | --- |
| QKV ready GPU wait | 21.083 | 混合 Projection compute、copy 和 CUDA 排队 |
| paged FlashAttention | 20.014 | 缺少 PD 同口径对照 |
| KV append | 0.327 | 已排除为主要瓶颈 |
| output P2P copy | 0.283 | 已排除为主要瓶颈 |

现有 trace 只在 Attention 进程可见，无法判断 QKV-ready 中有多少是 PD 也必须支付的
QKV 计算，也无法判断 C4 的 FlashAttention 是否被 30% MPS 显著拉长。

## 3. 设计原则

1. `PAP_DEFERRED_CUDA_TRACE=0` 仍是默认值，关闭时不得创建 Event、线程或文件。
2. 热路径只记录 CUDA Event 或 host monotonic duration，不调用
   `Event.synchronize()`，不逐层写日志或 JSON。
3. benchmark 完成且 session/request drain 后，runner 创建一个 `.flush` 文件；
   诊断进程的后台 exporter 才 blocking flush 并原子写 JSON。
4. 每个进程显式声明 role 和 scope，避免 Prefill、Projection、Attention 与 PD Decode
   的同名 span 混入同一结果。
5. 结果必须 fail closed：pending/drop/error 为零、必需 span 存在、逐层 span count
   一致且能被 36 层整除。
6. PAP 与 PD 的 `qkv_norm_rope_gpu_ms` 使用完全相同的 Qwen3 代码边界；PAP 专属的
   Q/K 写回 direct-QKV buffer 单列为 `projection_qk_repack_gpu_ms`。

## 4. Span 合同

### 4.1 PAP Projection

| Span | 类型 | 起止点 | 计数合同 |
| --- | --- | --- | --- |
| `qkv_norm_rope_gpu_ms` | CUDA | `qkv_proj` 前至 RoPE 后 | Attention peer batches |
| `projection_qk_repack_gpu_ms` | CUDA | 两次 Q/K `copy_` 写回 direct buffer | 同上 |
| `qkv_p2p_copy_gpu_ms` | CUDA | local-fast QKV `dst.copy_(src)` | 同上 |
| `output_doorbell_wait_wall_ms` | host | 开始 poll 至读到 output doorbell | 同上 |
| `output_ready_wait_gpu_ms` | CUDA | output `stream_wait_value32` | 同上 |
| `token_boundary_input_ids_d2h_wall_ms` | host | input IDs `.to(cpu).tolist()` | 逐层计数 / 36 |

`output_doorbell_wait_wall_ms` 与 `output_ready_wait_gpu_ms` 不能相加为一个严格的
独立区段：前者覆盖 CPU 看见通知前的等待，后者覆盖通知到达后 GPU stream 尚未满足
ready signal 的剩余等待。报告必须分别展示。

### 4.2 PAP Attention

保留现有四段和现有 HTTP stats 导出：

- `qkv_ready_wait_gpu_ms`；
- `kv_append_gpu_ms`；
- `paged_fa_gpu_ms`；
- `output_p2p_copy_gpu_ms`。

### 4.3 PD-twoway Decode

| Span | 类型 | 起止点 | 计数合同 |
| --- | --- | --- | --- |
| `qkv_norm_rope_gpu_ms` | CUDA | 与 PAP 相同的 Qwen3 边界 | 与 FA 相同且可被 36 整除 |
| `pd_paged_fa_gpu_ms` | CUDA | standard FlashAttention main paged call | 与 QKV 相同 |

PD span 只在 `max_query_len == 1` 的 Decode batch 记录，排除 Prefill 和 profiling
forward。PD Prefill 进程不启用 exporter。

## 5. 跨进程导出

诊断进程使用两个新环境变量：

- `PAP_DEFERRED_TRACE_ROLE=projection|attention|pd_decode`；
- `PAP_DEFERRED_TRACE_OUTPUT=/absolute/path/to/result.json`。

首次实际记录 span 时才启动 daemon exporter。exporter 轮询
`<output>.flush`；runner 在 workload 完成后创建该文件。检测到 trigger 后 exporter：

1. 调用 `deferred_cuda_trace_snapshot(blocking=True)`；
2. 附加 `scope`、`role` 和 PID；
3. 写入同目录临时文件；
4. 用 `os.replace()` 原子发布最终 JSON；
5. 删除 trigger。

runner 等待结果文件并执行独立 validator。超时、JSON 不完整、非零
pending/drop/error、缺 span 或 count 不一致均使实验失败。

## 6. C4 实验矩阵

全部实验固定 70:30、GPU1/GPU2、float16、16K、五轮、C4、o256、QPS 2、
`MAX_NUM_SEQS=4`，不做 MPS 扫描：

| 单元 | 架构 | Trace | 目的 |
| --- | --- | --- | --- |
| A | PAP | off | 当前代码 clean TPOT |
| B | PAP | on | 双侧分账和 PAP trace 扰动 |
| C | PD-twoway | off | 当前代码 PD control |
| D | PD-twoway | on | PD QKV/FA 分账和 PD trace 扰动 |

四个单元都必须完成 20/20 请求、strict correctness/cache/routing/drain gate，并保持
输出 digest 一致。诊断结果只在 PAP 和 PD 各自 trace 扰动不超过 `2%` 时用于精确
预算；超过 `2%` 时仍保留方向性数据，但必须先降低 trace 开销。

## 7. 结果决策

- token-boundary D2H 或逐 step host residual `>=1 ms/token`：优先设计 GPU-resident
  token side channel 和延迟 decode commit；
- PAP/PD `qkv_norm_rope_gpu_ms` 接近，而 QKV-ready residual 大：优化 handoff/control；
- PD/PAP paged FA 差异占主要差距：在固定 70:30 下优化 Attention kernel/layout/
  admission，不改变 MPS 比例；
- 每层 host/state 准备累计 `>=1.5 ms/token`：实现 per-step state matrix；
- 可优化 handoff residual `<1 ms/token`：停止 dataplane 微调，转向 MPS/cohort/kernel
  scaling。

本阶段不直接实现上述优化；先用四个实验结果与用户共同选择下一项。
