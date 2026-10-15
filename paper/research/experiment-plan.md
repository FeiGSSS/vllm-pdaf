# PAP 论文补充实验计划

日期：2026-10-14。依据：`notes/pap-code-survey.md`，`feature/pap-v026`（HEAD `c565ecb9`）上的实验记录，以及 `feature/pap` 上 7 月的 L01–L13 研究记录和 `paper/pap/claims.md`、`related-work.md`。只读了代码，没有修改代码或配置。

文中带"（推断）"的内容是根据代码和报告推出来的，没有用实验验证过。

---

## 0. 结论先行

1. **最优先的是解释 7 月的反证，而不是加更多赢面。** 7 月 L07/L10/L12 的结果是：在容量边界上（max goodput），PD 6P2D 比 PAP 高 24–37%。9 月 v026 的主结果只是 **C30 的单个固定负载点**，而且负载、运行时和 PD 基线三方面同时变了。审稿人看到这两组数字后的第一个问题是"差距到底消失了没有"，这个问题现在答不了。见 §3（E2）。
2. **主结果需要从"固定点"换成"负载扫描 + SLO 拐点下的最大 goodput + 置信区间"。** 现有 60 会话 × 3 轮的数据集在 C30 下只有两波请求，启动和收尾占比很大，不适合做高并发点。见 §2（E1）。
3. **有几项计划受代码限制，需要你决定是否改代码**（详见 §9）：
   - v026 只保留了 NVSHMEM 整步 Graph 一种传输（`vllm/pap/config.py` 中 `PAPOffloadKVTransport` 只剩 `CUDA_IPC`，local_fast / NIXL mailbox 相关环境变量都在 `PAP_REMOVED_FLAGS` 里），所以"传输方式消融"要么新写微基准，要么用 `feature/pap` 的旧实现做非单变量对照。
   - TP>1 被硬性禁止（`vllm/pap/config.py:488`："PAP NVSHMEM whole-step Graph requires TP=1"；`vllm/pap/integration/runner.py:95`）。
   - 目前只在 Qwen3 dense 上验证过（`projection_routing.py` 的 key 是 `_pap_qwen3_...`），换模型架构前需要先做冒烟测试（推断）。

---

## 1. 通用实验规范（所有新实验都遵守）

| 项 | 规范 |
| --- | --- |
| 重复 | 每个点至少 3 次独立启动；主结果图的点和拐点附近 5 次。报告均值和 95% bootstrap CI（以请求为单位重采样，再按启动分层）。 |
| SLO | **在跑之前登记**一个主 SLO 和两个敏感性 SLO，避免看完数据再挑阈值。建议主 SLO 用现有的 D2（TTFT ≤ 10 s、ITL ≤ 75 ms、E2E ≤ 20 s），敏感性用 D3（ITL ≤ 50 ms）和 D4。另外画 DistServe 式的"SLO 缩放曲线"（把三个阈值同乘 0.5–2.0）。 |
| goodput 定义 | 沿用 `GOODPUT.md`：goodput = 合格请求数 / 时长；可行点要求达标率 ≥ 95%。raw goodput 和可行性分开报告。 |
| 稳态窗口 | 新数据集用更多会话（见 E1），只统计稳态窗口，丢弃前 10% 的 warm-up 和最后一波 drain。 |
| 同期运行 | 同一组对比的所有架构在同一时间段内交替运行（ABAB 顺序），不复用几周前的基线归档。 |
| 审计 | 沿用现有的拓扑、进程、设备映射、正确性、清理审计；PD 额外检查 NIXL 聚合 KV 吞吐（QPS 矩阵里已经出现过低于 5 GB/s 下限的点）。 |
| 版本 | 记录 PAP commit、Dynamo/vLLM 版本、数据集 SHA256；PAP 必须是干净的已提交代码（7 月 L06 就是因为脏工作区被作废）。 |

**成本假设**（推断）：单次 8 卡运行含启动、图捕获、审计和清理，平均约 12 分钟，即每台 8×L20 机器每小时约 5 次。下面每项给出运行次数和按此折算的机时。

---

## 1.5 E0 合入 PAT 修复并重跑主表（P0，所有实验之前）

research 分支在 9 月 7 日发现 PAT kernel 的 split-rank 写入冲突：共享 CTA 拆分时两个 CTA 写同一个 partial-output 槽位，另一个槽位没写，gather 读到未写入的数据（`research/pap-bottleneck-20260907:benchmarks/pap/experiments/microbench/PAP-20260907-ATTENTION-BANDWIDTH/README.md` "测量中发现的 PAT 正确性问题"一节）。修复只以补丁形式归档在 `results/pat-split-rank-fix.patch`（改 `third_party/pat/csrc/prefix_tree.h` 的 `balancePack` 和一个测试文件），**没有合入 `feature/pap-v026`**：v026 和 research 分支的 `third_party/pat` 完全相同，补丁里新增的检查字符串在 v026 中搜不到。

v026 默认 `PAP_ATTENTION_KERNEL_POLICY=auto`，共享前缀时会走 PAT，所以 9 月的主表、KV 迁移和跨机数字都可能受影响，影响方向未知（结果错误不一定导致变慢或变快）。

- 合入补丁（需要改代码，先问你），跑 PAT 回归测试。
- 用修复后的代码重跑 C30 主表中的 PAP 7PA1P（3 次），同时跑 `PAP_ATTENTION_KERNEL_POLICY=triton` 对照（3 次）。
- 如果修复前后差异超过 CI，KV 迁移 A/B 和跨机 PAP 结果也要重跑；如果没有差异，在附录写明已检查。
- 之后的所有实验都基于修复后的 commit。
- 规模约 6–20 次，2–4 机时。

---

## 2. E1 负载扫描（P0，主结果）

**目的**：用"最大可持续 goodput"替代 C30 单点，并给出所有主指标的置信区间。

**数据集**：现有 `s60-t3-native32k-stratified-seed42` 只有 60 个会话，C30 时只有两波，C60 时一波，结果主要由启动和收尾决定。建议用同一个生成器配置（`generator-configs/native32k-s3.json`，同样的分层抽样）新建一个 **240 会话 × 3 轮** 的数据集作为新的 ID（数据集目录不可变，需要新 ID；这是新增文件，按约定先问你）。保留旧数据集作为对照，证明新旧数据集在 C30 上结论一致。

**矩阵**：

| 维度 | 取值 |
| --- | --- |
| 架构 | DP8、PD 7P1D、6P2D、4P4D、PAP 7PA1P、PAP 6PA2P（2P6D 在所有扫描里都远离前沿，只保留一个参考点） |
| 闭环并发 | C = 6, 10, 14, 18, 22, 26, 30, 36, 42, 50, 60 |
| 开环 | Poisson QPS 扫描作为补充（对应现有 `AGENTIC-CODE-QPS-MATRIX` 的方法，但用当前配置，不用冻结的旧配置） |
| 重复 | 每点 3 次，拐点 ±1 档补到 5 次 |

**产出**：

- 图 1：吞吐–延迟曲线（x = 实际 req/s，y = TTFT P50/P99 和 ITL P50/P99），每个架构一条带 CI 的线。
- 图 2：各 SLO 下的最大可行 goodput 柱状图，带 CI。
- 图 3：SLO 缩放曲线。
- 表：每个架构的拐点并发和对应的 goodput。

**规模**：6 架构 × 11 并发 × 3 次 ≈ 200 次，加拐点补测约 40 次 → 约 240 次，约 48 机时。开环扫描另加约 60 次。

---

## 3. E2 解释 7 月的反证（P0，必须做）

### 3.1 现状

| | 7 月 L07（`feature/pap`） | 9 月主结果（`feature/pap-v026`） |
| --- | --- | --- |
| 结论 | PD 比 PAP goodput 高 24.4%–36.9% | PAP 在 D2/D4 下 goodput 最高，比 6P2D 高 20.6%（D2） |
| 度量 | 每个架构搜索容量边界，比较 max goodput | 固定 C30 一个点 |
| 负载 | 128 会话 × 5 轮；首轮输入约 4K，追加约 0.7K；输出 O16；有 think/tool 延迟 | 60 会话 × 3 轮；最终上下文 16K–32K；输出均值 < 100；无延迟 |
| PAP 运行时 | eager；18/5 MPS chunk；`local_fast`/UCX；无 KV 迁移 | 整步 CUDA Graph；20/3 MPS（80/12 SM）；NVSHMEM；Dynamo 路由；KV 迁移 |
| PD 基线 | 手工修正的同机 UCX 1.22 GET-zcopy 路径 | 官方 Dynamo 1.4.1 + vLLM 0.26.0，NIXL |
| 其他 7 月结果 | L10（长输入）PD 高 22–23%；L12（接近 40K 上限）PD 高 34% | — |

所以当前的"PAP 赢"和 7 月的"PD 赢"之间，**度量、负载、PAP 运行时、PD 基线四件事同时变了**，任何一个都可能是原因。另外，9 月 QPS 矩阵里 6P2D 在高负载下吞吐卡在约 0.57 req/s，跟 7 月 PD 6P2D 是最强架构的结论差别很大，需要先确认 PD 基线本身没有退化（7 月 C15 已经发生过一次 PD 传输退化导致结论作废）。

### 3.2 实验

**E2a：PD 基线强度审计（先做，便宜）。**
- 在 v026 的 C30 负载上，对 Dynamo PD 6P2D 检查：NIXL 聚合 KV 吞吐、每个请求的 KV 传输时间、Decode 侧 batch 大小、Prefill 侧 chunked prefill 和 `max_num_seqs` 设置。
- 对 6P2D 做一次小范围调参（Prefill token budget 2048/4096/8192，Decode `max_num_seqs` 128/256，Dynamo prefill load scale），每档 3 次。目的是保证论文里的 PD 是调过的，而不是默认值。
- 规模约 30 次，6 机时。

**E2e：实测 PD 的 KV 传输开销（"KV 不搬家"的直接证据，必须做）。**

DistServe 报告 KV 传输不到总延迟的 0.1%（NVLink），Splitwise 约 0.8%（见 `novelty-assessment.md`）。审稿人会问：在 PCIe 的 L20 上，16–32K 上下文的传输开销到底有多大。目前仓库里只有 NIXL 吞吐的健康检查（`run_dynamo_workload.sh` 要求 p90 ≥ 5000 MB/s），没有报告传输时间占 TTFT 的比例。

先算字节量（按 Qwen3-8B 的维度计算：36 层、32 个 Q 头、8 个 KV 头、head_dim 128、FP16）：

| 量 | 每单位字节 | 16K 上下文 | 32K 上下文 |
| --- | ---: | ---: | ---: |
| PD：每个请求从 P 搬到 D 的 KV | 147,456 B/token | 2.4 GB | 4.8 GB |
| PAP：每个 decode token 在 Projection 和 PA 间往返的激活（Q/K/V 去、Attention 输出回） | 737,280 B/token/step | 输出 100 token 约 74 MB | 同左（与上下文无关） |

两者相差约 5 倍/token，所以输出长度小于上下文的约 1/5 时，PAP 搬的字节更少；反过来，长输出、短上下文时 PD 搬得更少。按 7 月修正后 CUDA-IPC 路径实测的 22–24.5 GiB/s 推算（推断），32K 上下文单次传输约 0.2 s，相对 6P2D 5.7 s 的平均 TTFT 只有几个百分点。**如果实测也是这个量级，"省掉传输时间"本身不足以支撑论文，论据要转到 Decode 侧的 KV 容量、HBM 带宽和多轮放置上**。这个风险要提前知道，所以把 E2e 放进 P0。

测量分三步：

1. **离线分析已有归档（0 机时）。** vLLM 的 `NixlKVConnectorStats` 会记录每次传输的 `transfer_duration`、`post_duration`、`bytes_transferred`（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/stats.py`），`run_dynamo_workload.sh` 已经把它汇总成 `kv_transfer.throughput_mb_s`。从 `PAP-20260907-ALIGNED-C30`、`-7P1D` 和 QPS 矩阵的 PD 归档里提取每次传输的时长和字节，画传输时间随上下文长度的分布，并估算占 TTFT 的比例。
2. **点对点微基准（约 1 机时）。** 用 NIXL 自带的 nixlbench（或 UCX CUDA-IPC）测 Qwen3-8B 的 KV 布局在 4K/8K/16K/24K/32K 下的单次传输时间，分别测 1→1、3→1 扇入（对应 6P2D 中每张 D 卡同时接收 3 个 P）和跨 NUMA 两种放置（`PAP-20260910-PCIE-NUMA-DIAG` 显示拓扑会影响结果）。不改仓库代码。
3. **端到端占比（并入 E2a 和 E1 的 PD 运行，不额外加运行次数）。** 在 6P2D 的 C30 和拐点运行中，按请求拆出"Prefill 完成 → KV 到达 D → 首个 decode 步"的时间，报告传输及其排队占 TTFT 的比例（均值和 P99），以及 D 卡接收 KV 时对正在 decode 的请求 ITL 的干扰。如果 Dynamo 前端的时间戳不够细，用 nsys 对 1 次运行采 trace。

产出：一张"传输时间占 TTFT 比例 vs 上下文长度"图，和一段与 DistServe/Splitwise 数字的对比，放进动机一节。

**E2b：在 v026 上重放 7 月的 L07 负载（直接回答"差距还在不在"）。**
- 用 L07 的数据集（SHA256 `b694ba14…`，O16），在 v026 PAP（7PA1P、6PA2P）和 v026 Dynamo PD（6P2D）上各做一次容量边界搜索（每个架构 5–6 个并发点 × 3 次），按 L07 当时的 Strict/Standard/Relaxed 三档 SLO 算 max goodput。
- 注意：这个数据集只在 `feature/pap` 的报告里被引用，不在 v026 的数据集注册表里（`benchmarks/pap/datasets/` 下没有），原始文件可能在 `_staging/` 或你本机，需要先找到。
- 规模约 60 次，12 机时。

**E2c：运行时归因阶梯（如果 E2b 显示差距缩小了，用来说明是哪项设计带来的）。**
在 L07 负载上，从 7 月配置逐步走到 v026 配置，每步只变一项，每步 3 次：

| 步 | 变化 | 可行性 |
| --- | --- | --- |
| 0 | `feature/pap@163c2dfa1`（L07 的 commit），7 月配置 | 可直接跑 |
| 1 | v026，MPS 18/5，关 KV 迁移，session 粘性 round-robin 路由 | v026 启动脚本允许 chunk 之和为 23 的任意组合，18/5 可以设（`run_pap_workload.sh:314–326`）；路由和迁移都有开关 |
| 2 | 步 1 + MPS 20/3 | 可 |
| 3 | 步 2 + Dynamo 路由 | 可 |
| 4 | 步 3 + KV 迁移（= v026 默认） | 可 |

步 0 → 1 合并了"eager → 整步 Graph"和"local_fast → NVSHMEM"两件事，v026 里没有开关能拆开它们（相关旧 flag 都已删除）。论文里如实写成"运行时整体替换"，再用 E3 的微基准说明传输本身的贡献。规模约 5 × 2 个并发点 × 3 次 = 30 次，6 机时。

**E2d：负载区间图（把"什么时候 PAP 赢、什么时候 PD 赢"画清楚）。**
- 在 v026 上做 输入长度 × 输出长度 的网格，每格对 PAP 7PA1P 和 PD 6P2D 各做粗粒度边界搜索（4 个并发点 × 2 次）：
  - 输入：L07 短输入（约 4K 起）、L09 长输入（约 8K 起）、native-32K（16K–32K）
  - 输出：O16、O100、O256
- 产出图 4：PAP/PD max-goodput 比值热图，标出 1.0 等高线。这是论文"有效区间"一节的核心证据。
- 规模 9 格 × 2 架构 × 8 = 144 次，约 29 机时。可以先跑对角线 3 格判断趋势，再决定是否补满。

### 3.3 论文里怎么写（按结果分三种情况）

| E2b 结果 | 写法 |
| --- | --- |
| v026 在 L07 负载上追平或超过 PD | 7 月结果写进"设计演进"小节：说明 eager + host 侧传输的版本在短负载上输给 PD，然后用 E2c 的阶梯说明是哪些设计改变了结果。这是最好的情况，反证变成了消融。 |
| 差距缩小但仍然存在 | 用 E2d 的热图明确写出 PAP 的有效区间（预计是长上下文、多轮前缀复用、Decode KV 受限的负载），短输入短输出的区间诚实写成 PD 更优，并给出机制解释。 |
| 差距基本没变 | 主张需要收窄：只在 E2d 中 PAP 占优的区间内宣称优势；同时检查 9 月 PD 基线是否偏弱（E2a）。如果 E2a 发现 PD 调参后超过 PAP，主结果也要改。 |

机制假设（推断，需要 E5 的 trace 验证）：短上下文时每个请求的 Attention 很轻，PA 上 12 SM 的 Attention 分区大部分时间空闲，而 7PA1P 的 Decode 计算全部压在 1 张 Projection 卡上；PD 6P2D 则有 2 张完整的 Decode 卡。上下文越长，Attention 和 KV 容量越重要，PAP 的优势越明显。L09 的结果（输入翻倍后 PAP/PD 吞吐比从 0.758 升到 0.870）和这个方向一致。

---

## 4. E3 消融（P1）

| ID | 消融 | 方法 | 代码改动 | 规模 |
| --- | --- | --- | --- | --- |
| A1 | **传输方式** | (a) 新写微基准：每层 Q/K/V 发送 + attention 输出回收的往返时间，对比 NVSHMEM-in-Graph、NCCL P2P、CUDA IPC + event、经主机中转，扫 batch 和层数；(b) 引用已有跨机实验（IBGDA，TTFT 只差 +7 ms）；(c) 可选：`feature/pap` 的 `local_fast` 版本做历史对照，标明非单变量 | (a) 需要在 `benchmarks/pap/microbench/` 下新增脚本，不碰运行时 | 微基准约 4 机时 |
| A2 | **Projection batch 聚合** | (a) 用 step trace（`PAP_PROJECTION_PA_TRACE_OUTPUT`）测 Projection 单步时间随 batch 的曲线，说明聚合 7 个 PA 带来的 GEMM 效率；(b) e2e：把 `PAP_PROJECTION_MAX_NUM_SEQS` 从 256 降到 128/64/32，看吞吐和 ITL 怎么变 | 无，用现有开关 | (a) 2 机时；(b) 4 档 × 3 并发 × 3 次 = 36 次，7 机时 |
| A3 | **PA:P 比例** | 7PA1P、6PA2P、5PA3P、4PA4P，在 E1 的并发网格上做简化扫描（6 个并发点 × 3 次）；同时在 E2d 的短上下文和长上下文两种负载上各跑一次，说明最优比例随上下文长度移动。可以和 `song2026analytical` 的 A/F 比例模型对照 | 无，拓扑是任意 xPAyP | 约 120 次，24 机时 |
| A4 | MPS 切分 | 18/5、19/4、20/3、21/2（chunk 之和必须是 23），C30 和拐点各 3 次。research 分支已有部分数据，但需要在 v026 干净代码上重做 | 无 | 约 24 次，5 机时 |
| A5 | KV 迁移开/关 | 已有 A/B（`DYNAMO-PA-KV-MIGRATION`），放到 E1 的 3 个并发点上重做，各 3 次 | 无 | 18 次，4 机时 |
| A6 | 路由策略 | Dynamo KV-aware vs session 粘性 round-robin（已有 `SESSION-ROUND-ROBIN`），同 A5 的点 | 无 | 18 次，4 机时 |
| A7 | Attention kernel | PAT vs Triton（`PAP_ATTENTION_KERNEL_POLICY=auto/triton`）的 e2e 对照，补充已有的微基准 | 无 | 12 次，2 机时 |

A1 是你点名要的"传输方式"消融，但 v026 里没有可切换的替代传输，只能靠微基准说明，这一点需要你确认能接受。

---

## 5. E4 泛化（P1/P2）

### 5.1 模型

| 模型 | 显存（推断） | 可行性 | 优先级 |
| --- | --- | --- | --- |
| Qwen3-14B FP16 | 权重约 28 GB，L20 48 GB 单卡放得下，但 PA 上留给 KV 的空间明显变小 | 同架构，最可能直接跑通 | P1 |
| Qwen3-32B FP8 | 权重约 32 GB，单卡勉强；需要确认 PAP 路径支持 FP8 权重 | 需冒烟测试 | P1 |
| Llama-3.1-8B | 与 8B 规模相同，换架构 | 需冒烟测试，可能要补模型适配 | P2 |
| Qwen3-30B-A3B（MoE） | FP8 约 30 GB | MoE 的 FFN 是 Attention/FFN 分离类工作的主战场，和 E6 关系最大；但 Projection 路径是否支持 MoE 未知，可能工作量大 | P2 |

每个新模型只跑 E1 的简化版（DP8、最优 PD、PAP 最优比例，各 6 个并发点 × 3 次 ≈ 54 次，约 11 机时）。

### 5.2 TP>1

当前代码硬性要求 TP=1。支持 TP>1 需要改 NVSHMEM 整步 Graph 和 Projection 路径，属于新功能开发。建议：**本文只做 TP1，在讨论和局限性一节说明**，用更大模型的 FP8 单卡版本代替"大模型"维度。如果你希望一定要有 TP2，需要单独排一个开发线程。

### 5.3 硬件

README 提到计划用 6000D。如果能拿到，在 6000D 上重跑 E1 的简化版（同上，约 54 次）。SM 数不同，MPS 切分需要重新定（先跑 A4 的简化版）。

### 5.4 负载

| 负载 | 目的 | 说明 |
| --- | --- | --- |
| Mooncake 公开对话 trace | 真实多轮前缀复用 | AIPerf 已经支持 `mooncake-trace` 格式，接入成本最低 |
| ShareGPT（单轮短对话） | 预计是 PAP 的弱区，主动展示有效区间的边界 | 和 E2d 互相印证 |
| LongBench 类长文档 | 长输入单轮 | 和现有 131K YaRN 实验互补 |
| 现有 agentic-code 131K YaRN | 已有，补重复次数即可 | |

---

## 6. E5 TTFT 尾部和 DP8 的 TTFT 优势（P1）

现状：PAP 的 TTFT P99 比 6P2D 高 7.4%（11.50 s vs 10.70 s），DP8 的 TTFT 均值只有 PAP 的一半（2.21 s vs 4.50 s），原因都没解释。

- 在 E1 的 C30 和拐点各取一次运行，打开 GPU-resident trace（已有 `PAP-20260904-GPU-RESIDENT-TRACE` 的方法），把 P99 请求的 TTFT 拆成：网关排队、路由、Prefill 排队、Prefill 计算、KV 迁移拷贝、首个 Decode 步。
- 统计 PA 之间的负载不均（每个 PA 的排队长度随时间变化）。
- 对 DP8：说明它的 TTFT 优势来自 8 张满 SM 的卡都做 Prefill（PAP 的 Prefill 只有 7 × 80 = 560 SM，DP8 是 8 × 92 = 736 SM，推断），代价是 ITL 高 71%。这可以写成一个简单的算力账。
- 规模约 6 次带 trace 的运行，2 机时，加分析时间。

---

## 7. E6 与 Attention/FFN 分离等系统的对比（P1）

### 7.1 需要覆盖的系统

| 系统 | 和 PAP 的关系 | 能否实测 |
| --- | --- | --- |
| Adrenaline（`liang2025adrenaline`） | 最接近：把 PD 中 Decode 的 Attention 部分卸到 Prefill 卡上 | 需要先确认是否开源、能否在 vLLM 0.26 上跑。能跑的话是最有说服力的对照 |
| MegaScale-Infer、Step-3 AFD、xDeepServe 等 Attention/FFN 分离 | 都针对 MoE、大规模多机；PAP 是 dense、单机 8 卡 | 规模和模型都对不上，建议定性对比 + 分析 |
| `song2026analytical`（A/F 比例模型） | 已经形式化了 A/F 比例选择 | 用 A3 的实测数据检验它的模型预测，作为讨论 |
| `liu2026challenges`、`wu2026howfar` | 指出 AFD 的通信死区和微批 overlap 的重要性 | PAP 不做层间微批，需要用 A1/A2 说明这个取舍的代价 |
| DistServe / Splitwise | PD 分离 | 已由 Dynamo PD 覆盖 |
| Chunked prefill 聚合部署 | 聚合基线 | 确认 DP8 基线确实开了 chunked prefill，并在文中写明 |
| Mooncake | PD + 分布式 KV 池 | 定性；如果用 Mooncake trace，可以讨论 KV 池与 PAP 的 PA 间迁移的差别 |

开源状态我没有核实，以上"能否实测"是待确认项。`related-work.md` 现在只有 5 条，需要单独补。

### 7.2 可选：在 PAP 代码里构造一个"AFD 式"基线

在 8 卡上搭一个 xP + yA + zF（Prefill、独立 Attention 卡、独立 FFN 卡，KV 需要从 P 搬到 A）的对照，用来单独说明"Attention 和 Prefill 同卡、免 KV 搬运"的价值。工作量较大，需要改代码，建议放到 P2，视 E2 的结果决定要不要做。

---

## 8. 优先级和总量

| 优先级 | 实验 | 运行次数 | 机时（推断） |
| --- | --- | ---: | ---: |
| P0 | E0 合入 PAT 修复、重跑主表 + Triton 对照 | 6–20 | 2–4 |
| P0 | E2a PD 基线审计 | 30 | 6 |
| P0 | E2e PD KV 传输开销实测（离线分析 + 微基准，端到端部分并入 E2a/E1） | 少量 | 约 1 |
| P0 | E2b 重放 L07 | 60 | 12 |
| P0 | E1 负载扫描（闭环） | 240 | 48 |
| P0 | E2c 运行时归因阶梯 | 30 | 6 |
| P1 | E2d 负载区间图（先跑对角线） | 48–144 | 10–29 |
| P1 | E3 消融 A1–A7 | 约 230 | 约 52 |
| P1 | E5 TTFT 分解 | 6 | 2 |
| P1 | E4 Qwen3-14B 和 32B-FP8 | 108 | 22 |
| P2 | E1 开环 QPS 扫描 | 60 | 12 |
| P2 | E4 6000D、Llama、MoE、新负载 | 各约 54 | 各约 11 |
| P2 | E6 AFD 式基线 | 待定 | 待定 |

P0 合计约 375 次、76 机时；P0 + P1 约 815 次、176 机时。

**建议顺序**：E0 → E2a + E2e → E2b →（看结果决定论文主张）→ E1 → E2c/E2d → E3 → E4/E5。E2b 的结果会直接决定论文的主线写法，应该在投入大规模扫描之前拿到。

**可以复用的已有结果**（不必重跑，但在新扫描中会被覆盖）：C30 主结果、跨机 KV 迁移、131K YaRN、PAT 微基准、Attention 延迟面和带宽测量。

---

## 8.5 Euro-Par 2027 执行版（约 40 机时）

投稿目标已定为 Euro-Par 2027（见 `positioning-and-venue.md` §0），按 CCF-B 的约 40 机时方案执行，本文档其余部分作为完整清单保留。执行版只包括：

| 实验 | 机时（推断） | 说明 |
| --- | ---: | --- |
| E0 PAT 修复 + 主表重跑 + Triton 对照 | 2–4 | 不管投哪里都必须做 |
| E2a PD 基线审计（顺带检查 DP8 是否开了前缀缓存和 chunked prefill） | 6 | |
| E2e PD KV 传输开销实测 | 约 1 | 离线分析 + 微基准；端到端部分用 E2a/E1 的 PD 运行 |
| E2b 重放 L07 负载 | 12 | 结果决定论文主线 |
| E1 简化扫描：DP8、6P2D、4P4D、PAP 7PA1P × 6 个并发点 × 3 次 | 约 14 | |
| E5 TTFT 尾部分解 | 2 | |
| **合计** | **约 37–39** | |

`novelty-assessment.md` §5 还提出了 AMPD 式基线（会话绑定到持有 KV 的 D 节点）。它不在这 40 机时里，是否加入需要单独决定。

---

## 9. 需要你决定的事

1. **是否允许改代码 / 新增文件。** 涉及：合入 PAT split-rank 修复（E0，改 `third_party/pat`）、新数据集 ID（240 会话版本，E1）、传输微基准脚本（A1）、新模型冒烟测试中可能需要的适配（E4）。建议合入 PAT 修复，并允许新增 `benchmarks/pap/` 下的数据集和微基准文件，其他运行时代码暂不动。
2. **TP>1 是否在本文范围内。** 建议不做，写进局限性。
3. **主 SLO。** 建议用 D2 作为主 SLO，D3/D4 作为敏感性分析，在开跑前写进实验 README。
4. **L07 数据集在哪里。** SHA256 `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`，v026 的数据集注册表里没有，可能在 `feature/pap` 的 `_staging/` 或你本机。
5. **6000D 是否能用、什么时候能用**，决定 E4 硬件部分是否排进来。

---

## 附：证据位置

- 主结果和多轮 TTFT：`feature/pap-v026:benchmarks/pap/experiments/e2e/PAP-20260909-KV-MIGRATION-ARCHITECTURE-C30/report.md`
- Goodput D1–D4 和 C60 敏感性：`…/e2e/PAP-20260907-ALIGNED-C30/GOODPUT.md`
- QPS 矩阵（旧配置）：`…/e2e/PAP-20260903-AGENTIC-CODE-QPS-MATRIX/report.md`
- 7 月反证：`feature/pap:benchmarks/pap/experiments/PAP-20260729-RESEARCH-L07/report.md`，以及 L08–L13 同目录
- 主张台账：`feature/pap:paper/pap/claims.md`（C06–C15）
- 相关工作：`feature/pap:paper/pap/related-work.md`
- 传输选项：`feature/pap-v026:vllm/pap/config.py`（`PAPOffloadKVTransport`、`PAP_REMOVED_FLAGS`）
- TP 限制：`feature/pap-v026:vllm/pap/config.py:488`，`vllm/pap/integration/runner.py:95`
- MPS 和 Projection 开关：`feature/pap-v026:benchmarks/pap/scripts/run_pap_workload.sh:307–331`
