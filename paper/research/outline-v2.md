# PAP 论文大纲 v2（2026-10-14）

> 依据：`notes/pap-code-survey.md`；`feature/pap` 分支 `paper/pap/`（manuscript / claims / state / related-work，停在 2026-07-30）；`feature/pap-v026` 与 `research/pap-bottleneck-20260907` 分支下的实验报告。
> 实验 ID 均指 `benchmarks/pap/experiments/{e2e,microbench}/<ID>/`。没有标注分支的都在 `feature/pap-v026`；标 **[R]** 的在 `research/pap-bottleneck-20260907`；标 **[J]** 的是 `feature/pap` 上七月的 L 系列研究。
> 证据强度：**强**＝受控 A/B 且 ≥2 次独立启动；**中**＝2 次重复，但只有一个负载点或非单变量；**弱**＝单次、诊断性运行或旧配置。

---

## 0. 和七月旧稿相比改了什么

七月旧稿（manuscript §2.2–2.3 + claims C01–C15）的结论是：在 O16 短输出负载上，校正后的 PD 的 goodput 比 PAP 高 24–37%（L07）；KV 池化本身不足以带来优势（L12）；只在长上下文 O100、C20 这一个窄区间里观察到 PAP 的 Standard-SLO 优势（C14）。

v026 之后，系统本身和评估负载都变了，旧稿的数字不能沿用：

| 维度 | 七月（`feature/pap`） | 现在（`feature/pap-v026`） |
| --- | --- | --- |
| vLLM 版本 | 0.23/0.24 时代 | 0.26，侵入集中在少数 hook |
| PA 内 MPS 切分 | 18/5 chunk（72/20 SM） | 80/12 SM |
| A↔P 传输 | `local_fast` 批量拷贝 | NVSHMEM P2P，放在 36 层整步 CUDA Graph 里 |
| 路由 | 会话亲和 / 峰值增益规则 | Dynamo KV-aware 路由（另有 session RR 作对照） |
| 多轮放置 | 无迁移 | PA→PA KV 迁移（CUDA IPC，Snapshot 导出） |
| Attention kernel | Triton | Triton + PAT（auto 策略） |
| 主负载 | O16，约 4K 输入 | native 32K，60 会话×3 轮，平均输出约 82 token，C30 |

**新叙事的立场**：不再写“PAP 普遍优于 PD”，而是写“在长上下文、多轮的负载区间，PAP 用‘KV 不搬家’加上‘Projection 聚合 batch’，在相同 8 卡预算下取得最高吞吐和可行 goodput；短输出、Prefill 为主的负载里 PD 仍可能更好”。七月的反证（L07、L12）放进评估的“有效区间与局限”一节，正面讨论，不回避。

> ⚠️ **写之前必须先处理的一个正确性隐患**：research 分支在带宽测量中发现 PAT 的 partial-output split rank 冲突（`PAP-20260907-ATTENTION-BANDWIDTH` [R]），修复只放在隔离依赖 `.local/pat-rankfix` 中，**没有进入 `feature/pap-v026` 的 `third_party/pat`**。v026 上的主结果默认 `PAP_ATTENTION_KERNEL_POLICY=auto`，共享前缀时会走 PAT，所以主表、KV 迁移和跨机结果都可能受影响。research 报告自己也写了“尚未逐 plan 追溯”。论文的数字至少要用修复后的 PAT 或强制 Triton 重跑主表（见 §9 P0-1）。

---

## 标题与摘要（草案）

**暂定标题**：PAP: Keeping KV Where It Is Born — Prefill–Attention–Projection Disaggregation for Long-Context Multi-Turn LLM Serving

**摘要要点**（按顺序）：
1. 问题：PD 分离在长上下文多轮负载下，要把每轮新生成的大块 KV 从 P 搬到 D；Decode 的 KV 容量和 HBM 带宽集中在少数 D 卡上；而且公共前缀面临“保留还是重建”的两难。
2. 洞察：Decode 可以拆成 KV 相关、访存密集的 Attention，和 KV 无关、计算密集的 Projection。Attention 可以和 Prefill 共卡、共享同一份 KV；Projection 可以无状态、跨 PA 聚合成大 batch。
3. 设计：PA 组（MPS 80/12 SM，CUDA IPC 共享 KV，P→D 零 KV 搬运）+ 无状态 Projection（NVSHMEM 逐层 QKV/输出交换，放在整步 CUDA Graph 内）+ 多轮放置（Dynamo 负载感知路由 + allocator-safe 的 PA→PA KV 迁移）。
4. 实现：基于 vLLM 0.26，约 32K 行，PAP 关闭时上游行为不变。
5. 结果：8×L20、Qwen3-8B、native 32K 多轮负载下，吞吐比 DP8 高 8.4%、比最好的 PD（6P2D）高 19.2%，ITL 比 6P2D 低 7.7%，E2E P99 在所有架构中最低；在 D2 SLO 下可行 goodput 比 6P2D 高 20.6%；KV 迁移使 TTFT 降低 17.4%；Projection 跨机部署时 TTFT 只增加 7 ms。
   *（这些数字需要在 P0-1 重跑后更新）*

---

## 1. Introduction

**要讲的主张**
- I-1：长上下文、多轮的 agentic 负载（代码助手、工具调用）正在成为主流，单请求输入 16K–32K，跨轮前缀复用率高。
- I-2：现有部署有两种极端。DP 融合部署 TTFT 低，但 Decode 被 Prefill 干扰，ITL 高；PD 分离 ITL 低，但要付出 KV 搬运、D 侧容量集中和跨轮前缀重建的代价。
- I-3：PAP 换了一条切分线：按“KV 归属”而不是“阶段”来切。
- I-4：贡献列表（4 条：架构、整步 Graph 通信、多轮 KV 放置/迁移、vLLM 实现与评估）。

**支撑结果**
- I-2 用主表的两端：DP8 TTFT 2.21 s / ITL 77.4 ms；PD 4P4D ITL 28.4 ms / TTFT 10.98 s（`PAP-20260909-KV-MIGRATION-ARCHITECTURE-C30`，中）。
- I-3、I-4 的 headline 数字同摘要第 5 条。
- 引言图（Fig.1）：TTFT–ITL–吞吐 Pareto 散点，6 种架构，数据就是主表。报告里已经写明 Pareto 前沿是 DP8、PAP、4P4D、2P6D。

**缺口**：I-1 需要引用负载特征来源（agentic-code 数据集的出处，以及 Mooncake/Kimi 一类的 trace 统计）。

---

## 2. Background and Motivation

### 2.1 LLM serving 与 KV 归属
- 主张：Prefill 计算密集，Decode Attention 访存密集，Decode 的 QKV/O/MLP 计算密集但 batch 小。KV 的“归属”决定了谁要搬 KV、谁占显存。
- 支撑：教科书背景加引用（vLLM/PagedAttention、DistServe、Splitwise、Sarathi-Serve）。

### 2.2 PD 在长上下文多轮负载下的四个结构性矛盾
沿用旧稿 §2.2 的四条，措辞保留其中的“条件性”限定：
1. Decode 的 KV 容量和 HBM 带宽集中在 N_D 张卡上。
2. Prefill 与 Decode Attention 利用的硬件资源互补（一个吃算力，一个吃带宽）。
3. 每轮都要做 P→D 的大块 KV 搬运。
4. 公共前缀的“保留还是重建”两难。

**支撑结果（逐条）**
- 矛盾 1：
  - 主表里 Decode-heavy 和 Prefill-heavy 的 PD 都有明显短板：6P2D TTFT 5.70 s，7P1D 吞吐只有 1.82 req/s，2P6D TTFT 34.7 s（中）。
  - 七月 L12 [J] 是**反证**：KV 池化不足以转化为 goodput。所以文中只能写“容量和带宽集中会限制 Decode 侧调度空间”，不能写“PD 撞上了 KV 容量墙”。
  - `PAP-20260909-DYNAMO-CROSS-HOST-PD` 中 Decode 达到 100% KV 占用、30 个等待请求，但那是 1 GbE 下的跨机场景（弱，只能作旁证）。
- 矛盾 2：`PAP-20260907-ALIGNED-C30` 的 DCGM 数据，PAP 显存活动率 31.8%、GPU 利用率 90.1%，对比 PD 各配置（中）。**缺**：PA 卡上 Prefill 与 Attention 分开的利用率（报告明说 DCGM 无法拆分），需要 Nsight 或 CUPTI 级别的数据。
- 矛盾 3：`PAP-20260909-DYNAMO-CROSS-HOST-PD` 每请求 NIXL 搬运 2,044 MB，在 1 Gb/s 链路上 TTFT 达 493 s（弱，链路不对等）。同机 PD 的 KV 搬运开销**缺**一个干净的量化（P1-4）。
- 矛盾 4：`PAP-20260909-SESSION-ROUND-ROBIN`：Dynamo 换 PA 的 74 次转移只有 30.1% 的 cache 读，留在原 PA 的为 76.1%，换 PA 后 TTFT 6.35 s，对比留下的 4.98 s（观测性）。这说明“换位置就要重建前缀”的代价是真实的。

### 2.3 现有 Attention 分离工作为什么不够
- 主张：Adrenaline 只把一部分 Decode Attention 卸载到 P 卡，QKV/FFN 仍在 D；AFD 类工作（MegaScale-Infer、Step-3 AFD、Analytical Provisioning、How Far Can Disaggregation Go）主要针对 MoE，没有处理多轮 KV 放置；BF-IO 假设 KV 不可迁移。
- 支撑：`paper/pap/related-work.md` 的 5 行对比矩阵（**需要扩充**，见 §8）。
- 旧稿已经否掉的“新颖性”不要再用：barrier-aware 放置、A/F 比例选择、fan-in skew 主导（C01 已证伪）。

**本节图**：Fig.2 PD 与 PAP 的 KV 流向对比示意图；Fig.3 每轮 TTFT（PAP 单调下降，PD 上升），数据见主表的 per-turn 表。

---

## 3. PAP 设计

### 3.1 架构总览
- 主张 D-1：PA 组 = 同一张 GPU 上的 Prefill（80 SM）+ Attention（12 SM），通过 CUDA IPC 共享同一份分页 KV，**P→D 之间没有 KV 搬运**。
- 主张 D-2：Projection 只持有权重、不持有 KV，把 x 个 PA 的 decode 请求合成一个 batch。
- 主张 D-3：拓扑 xPAyP，主配置 7PA1P。
- 支撑：设计文档 `docs/design/pap/v026-port.md`；正确性用 v026 port 的 L3–L5 gate（GQA 数值一致、455/455 请求、1PA1P/7PA1P 审计通过，`PAP-20260824-V026-PORTING`）。
- 图：Fig.4 架构图（**缺**，需要画）。

### 3.2 PA 内资源切分
- 主张 D-4：12 SM 的 Attention 已经接近带宽受限，静态切分在 TTFT 与 ITL 之间存在取舍，80/12 是当前负载下综合最优的点。
- 支撑：
  - SM 扫描 `PAP-20260907-BOTTLENECK` [R]：72/20 相对 80/12，ITL −9.6%、TTFT +22.2%、吞吐 −7.3%（72/20 有 2 次重复，76/16 只有 1 次；中）。报告指出这里有 Decode 负载变轻的混杂因素。
  - 固定输入的 kernel 测量 `PAP-20260907-ATTENTION-BANDWIDTH` [R]：12 SM 下 Triton 449–493 GB/s，20 SM 下 529–540 GB/s（逻辑有效带宽，不是 DRAM 计数）。
  - 七月 L05 [J]：16/7 也是负向结果（作为历史佐证）。
- 写法：这一节只论证“为什么用静态 80/12”，不要声称“自适应切分”，因为没有实现。

### 3.3 Attention–Projection 逐层通信
- 主张 D-5：每层 QKV 下发、attention 输出回收，用 NVSHMEM P2P 放在 36 层整步 CUDA Graph 中，避免每层都经过 CPU。
- 主张 D-6：故障处理是 fail-closed 的（设备侧 abort 信号、`/health`）。
- 支撑：`v026-port.md` 的 runtime lifecycle 一节；`PAP-20260904-GPU-RESIDENT-TRACE` 的 TBT 分解：Projection 路径 51.5%，PA Attention kernel 25.3%，barrier 不均衡 10.8%（单次、带 trace 开销、半长度数据集；中偏弱）。
- **缺**：整步 Graph 对比 eager / 逐层 host 同步的消融（P1-1）。

### 3.4 Attention kernel（可以放进 3.3 或 Implementation）
- 主张 D-7：在 12 SM 的低算力切分下，PAP 的 Triton paged-decode 自动配置接近最优；共享前缀时 PAT 能减少重复 KV 读取。
- 支撑：
  - `PAP-20260824-ATTENTION-LATENCY-SURFACE`：374 种负载 × 19 种配置，production auto 与最优的平均差距 4.92%，P95 差距 13.6%（强，属于微基准）。
  - `PAP-20260827-PREFIX-KERNELS`：ChunkAttention 不支持 GQA，被排除；PAT 1 小时端到端吞吐 +20.9%（32K）。⚠️ 这是在 rank 冲突修复之前测的。
  - 修复后 PAT 在 B2×16K、共享 8K 的输入上比 Triton 延迟低 24.8%（[R]，固定输入）。
- 定位：PAT 是第三方 kernel（ASPLOS'26），PAP 的贡献只在“适配到 SM89、CUDA Graph 和 PAP 的 step-prepare”，不能当成核心贡献。

### 3.5 多轮 KV 放置：路由 + PA→PA 迁移
- 主张 D-8：每轮用 Dynamo 的 KV-aware 路由选 PA；如果换了 PA，就按 block hash 把源 PA 的可复用前缀经 CUDA IPC 拷到目标 PA，不重算。
- 主张 D-9：迁移是 allocator-safe 的：Snapshot 导出时，前缀查找与引用计数获取在同一把 allocator 锁下原子完成；源 block 在目标 Prefill 完成、异步 release 之前不会被回收。
- 主张 D-10：迁移把会话粘性（复用好、负载不均）和逐轮负载感知（均衡好、复用差）两种做法的优点结合起来。
- 支撑：
  - **KV 迁移 A/B**（`PAP-20260909-DYNAMO-PA-KV-MIGRATION`，交替 2+2 次，强）：mean TTFT −17.44%，吞吐 +9.57%，计算的 Prefill token −19.14%，E2E −9.01%；代价是 ITL +4.11%，P99 TTFT 只降 2.37%。
  - **Snapshot 导出 A/B**（`PAP-20260909-KV-SNAPSHOT-EXPORT-C30`，2+2 次，强）：导出 RPC 1508 ms → 11.55 ms（−99.2%），TTFT −5.36%，吞吐 +6.30%。
  - **会话粘性 RR 对照**（`PAP-20260909-SESSION-ROUND-ROBIN`，2+2 次，强）：粘性 RR 比无迁移的 Dynamo TTFT −17.8%，但 P99 TTFT +8.3%。用来说明 D-10 的动机。
  - **数据正确性**：逐字节比对 7/7 与 40/40 次迁移全部一致。
  - 拷贝本身不是瓶颈：平均 128 ms，19.8 GB/s。
- **缺**：迁移 + Dynamo 对比粘性 RR 的直接三方对比（现在是两个独立 A/B，基线 run 不同）；路由策略对端到端的影响需要放在同一张表里（P1-2）。

---

## 4. Implementation

- 主张 M-1：PAP 作为 vLLM 0.26 的插件式子系统实现，`vllm/pap/` 约 32K 行，对上游的改动集中在 `scheduler.py`、`block_pool.py`、`attention.py`、`gpu/model_runner.py`、`forward_context.py`，PAP 关闭时上游行为与性能不变。
- 主张 M-2：从旧版本移植到 v0.26 后性能非劣：ITL −9.0%，吞吐 +4.7%，TTFT +1.95%（在 5% 非劣界内）。另外发现 Prefill 的异步调度会推迟 KV 发布这个控制面交接，因此 Prefill 侧关闭了异步调度。
- 支撑：`v026-port.md`；`PAP-20260824-V026-PORTING`（3 次重复，强）；`tests/pap/` 31 个文件。
- 补充：OpenAI 兼容 Gateway、Rust 版 Dynamo 路由（`dynamo_native`）、跨机 IBGDA 支持（`PAP-20260908-CROSS-HOST-NVSHMEM-IBGDA` 修复的 6 个根因，可以写成一段“工程经验”或放进附录）。
- **缺**：上游侵入的 diff 行数统计（容易补）。

---

## 5. Evaluation：方法

- 硬件：8×NVIDIA L20（单机）；跨机：L20B + L20T，mlx5 InfiniBand。
- 模型：Qwen3-8B FP16，TP1，native 32K；另有 YaRN 131K。
- 负载：AIPerf，`s60-t3-native32k-stratified-seed42`（60 会话×3 轮，输入 16K–32K），闭环并发 C30；输入 SHA256 固定。
- 基线：DP8（Dynamo 1.4.1 + vLLM 0.26），PD 7P1D / 6P2D / 4P4D / 2P6D（NIXL/UCX 1.22 CUDA IPC，校正后的传输路径）。
- 指标：TTFT、请求级 ITL、E2E、吞吐；SLO goodput 使用 D1–D4 诊断点，以 95% 达成率作为可行性门槛。
- 方法论主张 E-0：固定并发的对比混淆了拓扑与实际吞吐，因此必须报告吞吐–延迟前沿和 SLO goodput（七月 C03/C04 [J] 的教训，可以写成方法论脚注）。
- **缺**：D1–D4 是“诊断点”而非产品 SLO，需要给出选择依据，或改用按 P50/P90 自动推导的 SLO 族（P0-2）。

## 6. Evaluation：结果

### 6.1 端到端主结果
- 主张 E-1：在 C30 下，PAP 7PA1P + 迁移的吞吐最高（3.19 req/s），比 DP8 高 8.4%、比 6P2D 高 19.2%、比 7P1D 高 75.1%。
- 主张 E-2：PAP 在 4 项均值上同时优于 6P2D 和 7P1D（对比 6P2D：TTFT −21.0%、ITL −7.7%、E2E −15.7%）。
- 主张 E-3：PAP 的 E2E P99 在 6 种架构中最低（16.27 s）。
- 诚实披露：DP8 的 TTFT 更低（PAP 比它高 104%）；4P4D/2P6D 的 ITL 更低；PAP 的 TTFT P99 比 6P2D 高 7.4%。
- 支撑：`PAP-20260909-KV-MIGRATION-ARCHITECTURE-C30`（每种架构 2 次启动、共 360 请求，中）。PAP 两次 run 的吞吐分别为 3.184 和 3.204，稳定。
- 图表：Table 2（主表 + 尾延迟）；Fig.1/5 Pareto 图。
- **缺**：只有 C30 一个负载点（P0-2），没有置信区间（P0-3），PAT 隐患（P0-1）。

### 6.2 SLO goodput
- 主张 E-4：在 D2（TTFT≤10 s、ITL≤75 ms、E2E≤20 s）下，只有 PAP 与 6P2D 可行，PAP 的 goodput 高 20.6%（3.079 vs 2.553）；D4 下 PAP 的可行 goodput 最高。
- 诚实披露：D3（ITL≤50 ms）下只有 4P4D 可行；D1 下没有任何架构可行。
- 支撑：同上，加上 `PAP-20260907-ALIGNED-C30/GOODPUT.md`（中）。
- **缺**：真正的“SLO 拐点下的最大 goodput”需要并发 / QPS 扫描（P0-2）。仓库自己也写了 C30/C60 两个点“不能定位最大可持续 goodput”。

### 6.3 多轮行为
- 主张 E-5：PAP 是唯一一个 TTFT 随轮次单调下降的分离式架构（5.69 → 4.34 → 3.48 s，第 2 轮比第 0 轮低 38.8%）；PD 各配置的 TTFT 随轮次上升。
- 支撑：主表报告的 per-turn 表（中）。图：Fig.3。

### 6.4 消融与因果分析
| 消融 | 状态 | 来源 | 强度 |
| --- | --- | --- | --- |
| KV 迁移 开/关 | 有 | `PAP-20260909-DYNAMO-PA-KV-MIGRATION` | 强 |
| Snapshot 对比 EngineCore 导出 | 有 | `PAP-20260909-KV-SNAPSHOT-EXPORT-C30` | 强 |
| 路由：Dynamo 对比粘性 RR（无迁移） | 有 | `PAP-20260909-SESSION-ROUND-ROBIN` | 强 |
| PA 内 SM 切分 80/12、76/16、72/20 | 有 | `PAP-20260907-BOTTLENECK` [R] | 中 |
| Prefill token 预算 2048 对比 4096 | 有（单次） | 同上 | 弱 |
| Attention kernel Triton 对比 PAT | 有（PAT 修复前） | `PAP-20260827-PREFIX-KERNELS(-E2E)` | 中，需重测 |
| 整步 Graph + NVSHMEM 对比 eager / 其他传输 | **缺** | — | P1-1 |
| Projection batch 聚合的贡献 | **缺** | — | P1-3 |
| PA:P 比例 7PA1P 对比 6PA2P | 只有旧配置 | `PAP-20260903-AGENTIC-CODE-QPS-MATRIX`（第二个 Projection 让 TBT 降 7–12%） | 弱 |

- 主张 E-6（TTFT/TBT 归因）：TTFT 的 86–90% 发生在 Prefill HTTP 路径内，Prefill 没有持续排队，KV 占用最高 35%，所以 TTFT 由 cache-miss 的 Prefill 计算主导，而不是 KV 容量；TBT 中 Projection 路径占约一半，PA barrier 不均衡的上限是 10.8%。
- 支撑：`PAP-20260904-GPU-RESIDENT-TRACE`（中），`PAP-20260907-BOTTLENECK` [R] 的诊断 run。

### 6.5 跨机部署
- 主张 E-7：把 Projection 放到另一台机器上（QKV/输出走 NVSHMEM IBGDA，KV 仍在 PA 本机），开启迁移后 TTFT 只增加 7.4 ms（+0.17%），吞吐 +1.1%。
- 主张 E-8：PAP 的首 token 之前不需要跨机搬运 KV；PD 跨机时每请求需要搬运约 2 GB KV。
- 支撑：`PAP-20260909-CROSS-HOST-KV-MIGRATION-C30`（2+2 次，中）；`PAP-20260908-CROSS-HOST-NVSHMEM-IBGDA`。
- 必须披露：
  - NVSHMEM 3.3 要求每个节点的 PE 数相同，跨机 world 实际占了 14 张 GPU（其中 6 个空闲 rank），和 8 卡单机不是等量资源对比（IBGDA 报告原文；迁移实验是否同样如此需确认）。
  - 跨机 PD 对照 `PAP-20260909-DYNAMO-CROSS-HOST-PD` 走的是 1 Gb/s 以太网 TCP，而 PAP 走 IB，链路不对等，**不能直接作为 PAP 与 PD 的跨机对比**。`PAP-20260910-CROSS-HOST-PD-MATRIX` 只有 env/run.sh，没有报告。
  - 跨机 PD 出现过 PCIe fatal（`PAP-20260910-PCIE-NUMA-DIAG`），需要在文中说明或排除。
- 建议：如果 P1-4 补不上 IB 上的公平 PD 对照，就把跨机部分降级为“可行性与开销”小节（只用 E-7），不写跨机的架构对比。

### 6.6 长上下文与泛化
- 主张 E-9：PAP 支持 YaRN 131K：14/14 请求成功，所有 Graph 都完成 capture，ITL 63 ms；每个 PA 可容纳 1.22–1.25 个 131K 请求（`PAP-20260824-QWEN3-131K-YARN-7PA1P`，功能性，弱）。
- **缺**：更大模型（32B 或 TP>1）、其他 GPU（6000D）、其他负载（ShareGPT、LongBench 类）（P1-5）。

### 6.7 有效区间与局限（正面处理反证）
- 主张 E-10：PAP 的优势是有条件的。以下几项要写清楚：
  1. 七月 L07 [J]：在 O16 短输出、约 4K 输入的负载下，校正后的 PD 比 PAP 的 goodput 高 24–37%。当时的系统是 18/5 切分、`local_fast`、无迁移。现在需要说明是哪些设计改变（迁移、整步 Graph、80/12）缩小了差距，还是只是负载不同，**最好用当前代码在 O16 负载上重跑一次**（P0-4）。
  2. 七月 L12 [J]：KV 池化（2.63× 的总 KV 池）不足以变成可用的 SLO 容量。v026 的 trace 也显示 KV 占用不是瓶颈，所以论文不应把“更大的 KV 池”当作主要卖点。
  3. 旧配置的 QPS 扫描 `PAP-20260903-AGENTIC-CODE-QPS-MATRIX`（开环 Poisson，C60，退役路由）：DP8 吞吐最高，2P6D 的 TBT 最低，PAP 两项都不占优。这一点必须用当前配置重做（并入 P0-2），否则审稿人会拿它当反例。
  4. DP8 的 TTFT 一直最低；PAP 的 TTFT P99 比 6P2D 高 7.4%，机制尚未解释（P1-6）。
- 写法：用一张“区间图”（负载维度 × 哪种架构胜出）来总结，比文字更有说服力。

---

## 7. Discussion
- 静态 MPS 切分的局限，以及自适应 / 时分复用的可能（作为 future work，不当成贡献）。
- Projection 是单点：容错与多 Projection 扩展（6PA2P 的旧数据）。
- MoE 模型上 Projection 会变成 FFN/expert，和 AFD 工作的关系。
- 跨机扩展的 PE 对称约束（NVSHMEM 3.3）。

## 8. Related Work
现有 `related-work.md` 只有 5 条，需要扩充到每类 3–5 篇。以下工作名称来自记忆，**引用前需要逐条核实出处**：
- **PD 分离**：DistServe、Splitwise、TetriInfer、Mooncake、Dynamo（NVIDIA）。
- **融合与 chunked prefill**：Sarathi-Serve、vLLM。
- **Attention 卸载 / 分离**：Adrenaline（最接近，必须写清差异：它是部分卸载，PAP 是完整 PA 归属 + 无状态 Projection + 多轮迁移）、Lamina 一类的 attention offloading 工作、MegaScale-Infer、Step-3 AFD、Analytical Provisioning、AFD challenges、How Far Can Disaggregation Go。
- **KV 缓存管理与迁移**：LMCache、Mooncake 的 KV store、Llumnix（实例间迁移）、CachedAttention / AttentionStore。
- **前缀共享 kernel**：PAT（ASPLOS'26）、ChunkAttention、FlashInfer Cascade、Hydragen。
- **负载均衡**：BF-IO。

## 9. Conclusion
在长上下文多轮负载区间，PAP 用“KV 留在出生地”和“Projection 跨 PA 聚合”两点，在相同 GPU 预算下取得最高吞吐与可行 goodput，并能以几乎零 TTFT 代价跨机部署 Projection。

---

## 附：主张–证据总表

| ID | 主张 | 所在节 | 主要证据 | 强度 | 状态 |
| --- | --- | --- | --- | --- | --- |
| D-1 | P→D 零 KV 搬运 | 3.1 | 设计 + v026 port 审计 | 强（设计性质） | 可写 |
| D-4 | 80/12 是当前取舍点 | 3.2 | BOTTLENECK [R]、BANDWIDTH [R] | 中 | 可写，带混杂说明 |
| D-5 | 整步 Graph 逐层通信 | 3.3 | GPU-RESIDENT-TRACE | 中 | 缺消融 |
| D-7 | 12 SM kernel 接近最优 | 3.4 | ATTENTION-LATENCY-SURFACE | 强 | 可写 |
| D-8/9 | 迁移有效且安全 | 3.5 | DYNAMO-PA-KV-MIGRATION、KV-SNAPSHOT-EXPORT | 强 | 可写（需 PAT 复核） |
| D-10 | 迁移结合了粘性与负载感知的优点 | 3.5 | SESSION-ROUND-ROBIN + 迁移 A/B | 中 | 缺同表三方对比 |
| M-2 | v0.26 移植非劣 | 4 | V026-PORTING | 强 | 可写 |
| E-1/2/3 | 主结果 | 6.1 | KV-MIGRATION-ARCHITECTURE-C30 | 中 | **P0-1/2/3 后定稿** |
| E-4 | D2 goodput +20.6% | 6.2 | 同上 + ALIGNED-C30 GOODPUT | 中 | 需负载扫描 |
| E-5 | 每轮 TTFT 单调下降 | 6.3 | 同上 | 中 | 可写 |
| E-6 | TTFT 由 Prefill 计算主导 | 6.4 | GPU-RESIDENT-TRACE、BOTTLENECK [R] | 中 | 可写 |
| E-7 | 跨机 TTFT +7 ms | 6.5 | CROSS-HOST-KV-MIGRATION-C30 | 中 | 需披露 GPU 不对等 |
| E-9 | 支持 131K | 6.6 | QWEN3-131K-YARN | 弱 | 功能性 |
| E-10 | 有效区间 | 6.7 | L07/L12 [J]、QPS-MATRIX | — | 需 P0-4 |

## 附：补实验优先级

**P0（不做无法投稿）**
1. **PAT 修复合入 v026 后重跑主表**：至少 PAP 7PA1P + 迁移重跑 2 次；同时跑一组 `PAP_ATTENTION_KERNEL_POLICY=triton` 作对照，确认结论不依赖有缺陷的 kernel。KV 迁移 A/B 视差异决定是否重跑。
2. **负载扫描**：当前配置下对 DP8 / 6P2D / 4P4D / PAP 做并发（如 C10–C50）或 QPS 扫描，在一组明确的 SLO 下求最大可行 goodput，替代只有 C30 的单点结论，并覆盖旧 QPS 矩阵的反例。
3. **重复次数与置信区间**：关键点 ≥3–5 次，报告 bootstrap CI。
4. **O16 短输出负载用当前代码重跑一次**：回答“七月 PD 领先 24–37% 的结论现在是否还成立”，用来画有效区间。

**P1（显著增强说服力）**
1. 整步 Graph + NVSHMEM 对比 eager / 逐层 host 同步的消融。
2. Dynamo + 迁移、Dynamo 无迁移、粘性 RR 在同一批 run 里的三方对比。
3. Projection batch 聚合的贡献（例如限制 Projection batch 上限，或对比每个 PA 自带 Projection 的 “PA-local” 变体）。
4. 同机 PD 的 KV 搬运开销量化；如果要保留跨机对比，补一组 IB 上的跨机 PD。
5. 泛化：至少再加一个模型（32B / TP2）或一种 GPU，再加一个公开负载。
6. TTFT P99 尾部的机制分析。

**P2（锦上添花）**
- 7PA1P 与 6PA2P 在当前配置下的比例扫描；GPU 利用率时间线图；131K 下与 PD 的对比。

## 附：图表清单
| 编号 | 内容 | 数据来源 | 状态 |
| --- | --- | --- | --- |
| Fig.1 | TTFT–ITL–吞吐 Pareto | 主表 | 有数据，待画 |
| Fig.2 | PD 与 PAP 的 KV 流向对比 | — | 待画 |
| Fig.3 | 每轮 TTFT | 主表 per-turn | 有数据，待画 |
| Fig.4 | PAP 架构 + 一个 decode step 的时序 | 设计文档 | 待画 |
| Fig.5 | goodput 与并发 / SLO 的关系 | P0-2 | 缺数据 |
| Fig.6 | KV 迁移 A/B（TTFT、计算 token） | DYNAMO-PA-KV-MIGRATION | 有数据 |
| Fig.7 | SM 切分取舍 | BOTTLENECK [R] `tradeoffs.png` | 有初版 |
| Fig.8 | TBT / TTFT 分解堆叠图 | GPU-RESIDENT-TRACE | 有数据 |
| Fig.9 | 有效区间图 | P0-4 + 主结果 | 缺数据 |
| Table 1 | 与相关工作的对比矩阵 | related-work | 需扩充 |
| Table 2 | 主表 + 尾延迟 | 主表 | 有 |

## 附：需要你拍板的两件事
1. **投稿方向**：七月 README 写的是 ICML 一类的 AI 会议；但这份材料更像系统论文（MLSys / EuroSys / ATC / OSDI 一类）。本大纲按系统论文的结构写，正文约 12 页。
2. **跨机部分的定位**：是补公平的 IB 跨机 PD 对照（P1-4），还是降级为“可行性与开销”小节。本大纲默认降级。
