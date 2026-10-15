# PAP 论文定位与投稿级别评估

日期：2026-10-14。依据：`notes/pap-code-survey.md`、`paper/experiment-plan.md`、`paper/outline-v2.md`、`feature/pap-v026`（HEAD `c565ecb9`，2026-09-10，确认 `third_party/pat` 仍未合入 split-rank 修复），以及网上检索的相关工作和会议日期。机时沿用实验计划的估算（每次 8 卡运行约 12 分钟），属于推断。

会议日期中，标"官方"的来自会议官网 CFP；标"聚合站"的来自 myhuiban / PaperPilot / beri.net 等第三方，需要以官网为准；标"推断"的是按往年周期推出来的，CFP 尚未发布。

---

## 0. 结论

**已决定（2026-10-14，更新）：按 CCF-B 准备，首投 Euro-Par 2027（预计 2027 年 3 月截稿，近两年录取率约 26–30%）；被拒后转 ICPP 2027（预计 5 月），C 类 ICA3PP / ICPADS 兜底。HPDC（2026 年录取率约 12.5%）和 ICML（偏 ML 创新）不投。**

**推荐：按 CCF-B 系统会议的标准准备，首选 HPDC 2027（预计 2027 年 1 月底到 2 月初截稿，推断），约 40 机时的补充实验。** E2b（在 v026 上重放 7 月 L07 负载）出结果后再做一次判断：如果 PAP 在短输出负载上追平 PD，故事明显变强，可以用同一套数据加泛化实验冲 USENIX ATC '27（预计 2027 年 1 月截稿，推断）。

理由：

1. **降低会议级别并不能省掉最关键的两件事。** PAT kernel bug 可能让现有所有 PAP 数字失效，这是正确性问题，不是严谨度问题；7 月"PD goodput 高 24–37%"的反证在仓库里有完整记录，任何审稿人只要看过 arXiv 版或旧稿就会问。这两项不管投哪里都得做（见 §3）。
2. **但降级确实能省掉大头。** 实验计划 P0+P1 约 175 机时，其中负载扫描全矩阵（48 机时）、消融 A1–A7（约 52 机时）、泛化（22 机时起）占了绝大部分。CCF-B 会议只需要其中约 40 机时。
3. **顶会的风险不在实验量，而在故事强度。** 现有优势是有条件且幅度中等的：吞吐比 DP8 高 8.4%，但 DP8 的 TTFT 只有 PAP 的一半；只有 Qwen3-8B、TP1、L20 单机；最接近的 Adrenaline 和 semi-PD 已经覆盖了一部分思想（见 §1）。这些在 OSDI/EuroSys 级别会被当成主要缺陷，补实验也未必能弥补。
4. **时间上 MLSys 2027（10 月 30 日）赶不上**：E0 需要先合入修复再重跑，E2 还没开始，16 天内做完并写完不现实。

---

## 1. 论文定位

### 1.1 建议的一句话定位

> 面向长上下文、多轮对话负载的单机 dense 模型服务：把 Decode 的 Attention 留在生成 KV 的 Prefill 卡上（KV 不搬家），把 KV 无关的 Projection 放到无状态的卡上跨 PA 聚合 batch，并用 PA 间 KV 迁移解决多轮放置问题。优势是有条件的，论文明确给出有效区间。

这和 `outline-v2.md` 的"新叙事立场"一致，不需要改大纲结构。

### 1.2 相关工作和差异（审稿人最可能的质疑点）

| 工作 | 做了什么 | 和 PAP 的差异 | 风险 |
| --- | --- | --- | --- |
| **Adrenaline**（[arXiv 2503.20552](https://arxiv.org/abs/2503.20552)，中山大学 + 华为云） | PD 分离下，把**一部分** Decode Attention 卸载到 Prefill 实例，用 Prefill 卡的空闲显存和带宽，dense 模型 | PAP 是**全部** Attention 都在 PA 上，KV 从不离开生成它的卡；Decode 的非 Attention 部分在无状态 Projection 上跨 PA 聚合；另有多轮迁移 | **最高**。审稿人很可能说"PAP = Adrenaline 推到极端"。必须写清楚差异，最好有定量或分析对比 |
| **semi-PD**（[arXiv 2504.19867](https://arxiv.org/abs/2504.19867)）、**Nexus**（[arXiv 2507.06608](https://arxiv.org/abs/2507.06608)） | 同一张卡上按 SM 切分 Prefill 和 Decode，存储统一，不搬 KV | PAP 的 PA 组也是同卡 SM 切分，但只切出 Attention，Projection 拆到别的卡；semi-PD 每张卡都要放全部权重并跑完整 Decode | 中。PA 内的 MPS 切分不能当成新贡献，要引用它们 |
| **Lamina**（[arXiv 2405.01814](https://arxiv.org/abs/2405.01814)）、Infinite-LLM / DistAttention | 把 Attention 卸到便宜的大显存设备 | 动机相近（Attention 访存密集），但 KV 仍需从 Prefill 搬到 Attention 设备 | 中 |
| **AFD 系列**：MegaScale-Infer（[arXiv 2504.02263](https://arxiv.org/abs/2504.02263)）、Step-3（[arXiv 2507.19427](https://arxiv.org/abs/2507.19427)）、A/F 比例理论（[arXiv 2601.21351](https://arxiv.org/abs/2601.21351)）、How Far Can Disaggregation Go（[arXiv 2605.28302](https://arxiv.org/abs/2605.28302)） | Attention 与 FFN 分离，主要针对 MoE、多机大规模 | PAP 是 dense、单机，且 Attention 与 Prefill 共卡 | 低到中。审稿人会问"为什么不用 MoE"，在讨论里说明 |
| DistServe / Splitwise / Mooncake / Dynamo | PD 分离 | 已用 Dynamo PD 作基线 | 低 |

结论：**新颖性可以成立，但必须把"KV 归属不变 + 无状态 Projection 聚合 + 多轮迁移"作为一个整体来讲**，任何单独一项都有先例。

---

## 2. 候选会议（截稿在 2026-10-14 之后）

| 级别 | 会议 | 截稿 | 来源 | 匹配度 |
| --- | --- | --- | --- | --- |
| 顶会（CCF-A） | OSDI '27 | 摘要 2026-12-01，全文 2026-12-08 | 官方（[CFP](https://www.usenix.org/node/321091)） | 门槛最高，8 周内补完顶会级实验不现实 |
| 顶会（CCF-A） | USENIX ATC '27 | 预计 2027 年 1 月 | 推断（ATC '27 回到 1 月截稿；CFP 未发布） | 系统实现型论文的最佳顶会选项 |
| 顶会（CCF-A） | EuroSys 2028 春季轮 | 预计 2027 年 5 月 | 推断（EuroSys 2027 秋季轮 9 月 24 日已过，[聚合站](https://www.getpaperpilot.com/deadlines/eurosys-2027.html)） | 时间宽裕，可作为 ATC 被拒后的下一站 |
| 顶会（非 CCF） | MLSys 2027 | 2026-10-30 | 聚合站（[mldeadlines](https://mldeadlines.com/conference/mlsys/)、[beri.net](https://www.beri.net/events/mlsys-2027)） | 主题最贴合，但 16 天来不及 |
| 已过 | ASPLOS '27、NSDI '27 秋季轮、EuroSys '27、IPDPS '27、SoCC '26 | — | 聚合站 | — |
| CCF-B | ICS 2027 | 约 2026-12-10 | 二手列表（[来源](https://www.queensu.ca/academia/afsahi/pprl/events.html)），需核实 | HPC 取向，LLM serving 能投但不是主场 |
| CCF-B | **HPDC 2027** | 预计 2027 年 1 月底到 2 月初 | 推断（HPDC '26 是 2026-02-05，[来源](https://hpdc.sci.utah.edu/2026/calls.html)） | 分布式 / 异构系统，匹配好 |
| CCF-B | Euro-Par 2027 | 预计 2027 年 2 月 | 推断 | 范围较广，匹配一般 |
| CCF-B | ICPP 2027、CLUSTER 2027 | 预计 2027 年 4 月 | 推断 | 时间最宽裕 |
| CCF-B | SoCC 2027、Middleware 2027 | 预计 2027 年夏 | 推断 | 云 / 服务系统，匹配好但较晚 |
| 期刊 | IEEE TPDS（CCF-A 期刊）、软件学报 / 计算机学报（CCF-A 中文期刊） | 滚动投稿 | — | 审稿周期长（数月起），篇幅不受限，适合实验更完整之后 |
| Workshop | EuroMLSys 2027（与 EuroSys 同期）、MLSys 2027 workshops | 预计 2027 年 2–3 月 | 推断，未找到 2027 CFP | 多为 4–6 页，常可后续再投正式会议 |

---

## 3. 无论投哪里都必须处理的两件事

### 3.1 PAT kernel split-rank 修复（实验计划 E0）

`feature/pap-v026` 的 `third_party/pat` 仍然没有合入 research 分支的 `pat-split-rank-fix.patch`（今天再次确认：v026 HEAD 仍是 9 月 10 日的 `c565ecb9`）。v026 默认 `auto` 策略在共享前缀时走 PAT，所以主表、KV 迁移 A/B 和跨机结果都可能受影响。

最低要求：合入修复（需要你批准改代码），重跑 PAP 7PA1P 主表 3 次，加 `PAP_ATTENTION_KERNEL_POLICY=triton` 对照 3 次；差异超过置信区间才需要重跑 KV 迁移 A/B 和跨机。约 2–4 机时。

如果不想改代码，退路是**全部改用 Triton 重跑 PAP 的数字**，论文里不使用 PAT。这样 PAT 只出现在相关工作里，反而更干净。

### 3.2 7 月短输出反证（L07/L10/L12）

即使是 workshop，也不能只报 C30 的胜利而不提 7 月的结果：那是同一个系统、同一个作者、记录完整的反向结论。

| 级别 | 最低处理 |
| --- | --- |
| Workshop | E2a 的低成本版（检查 PD 6P2D 的 NIXL 吞吐和配置，约 2 机时）+ 在论文里明确写"短输出、短输入时 PD 可能更优"，把主张限定在长上下文多轮区间 |
| CCF-B | E2a 完整版（6 机时）+ **E2b 重放 L07 负载**（12 机时），用结果决定写法（实验计划 §3.3 的三种情况）。需要先找到 L07 数据集（SHA256 `b694ba14…`） |
| 顶会 | 上面全部 + E2c 运行时归因阶梯（6 机时）+ E2d 负载区间热图（10–29 机时） |

---

## 4. 各级别最低证据和可以砍掉的实验

| 实验（实验计划编号） | 机时 | Workshop | CCF-B | 顶会 |
| --- | ---: | :---: | :---: | :---: |
| E0 PAT 修复 + 主表重跑 + Triton 对照 | 2–4 | 必须 | 必须 | 必须 |
| 主表 3 次重复 + bootstrap CI | 含在 E0 | 必须 | 必须 | 必须 |
| E2a PD 基线审计 | 2–6 | 简化版 | 必须 | 必须 |
| E2b 重放 L07 负载 | 12 | 可砍 | **必须** | 必须 |
| E1 负载扫描 | 14–48 | 可砍 | 简化版：4 架构（DP8、6P2D、4P4D、PAP 7PA1P）× 6 并发 × 3 次 ≈ 72 次，约 14 机时 | 全矩阵 + 拐点补测 |
| E5 TTFT 尾部分解 | 2 | 可砍 | 建议做（便宜，能回答"DP8 TTFT 为什么更好"） | 必须 |
| A7 PAT vs Triton 端到端 | 2 | 可砍 | 和 E0 合并 | 必须 |
| A5/A6 迁移、路由 | 0–8 | 用已有数据 | 用已有数据（已是 2+2 次 A/B） | 在 E1 点上重做 |
| A4 MPS 切分 | 0–5 | 用已有数据 | 用 research 分支已有数据，注明混杂 | 在干净代码上重做 |
| E2c 运行时归因阶梯 | 6 | 可砍 | 可砍（E2b 结果好时建议做） | 必须 |
| E2d 负载区间热图 | 10–29 | 可砍 | 可砍，或只跑对角线 3 格（约 10 机时） | 必须 |
| A1 传输微基准 | 4 | 可砍 | 可选 | 必须 |
| A2 Projection batch 聚合 | 9 | 可砍 | 可选，只做 (a) trace 部分（2 机时） | 必须 |
| A3 PA:P 比例 | 24 | 可砍 | 可砍 | 必须 |
| E4 更大模型（14B / 32B-FP8） | 22+ | 可砍 | 可砍，写进局限性 | 至少一个 |
| E4 其他 GPU、Llama、MoE、新负载 | 各约 11 | 可砍 | 可砍 | 至少一个公开负载 |
| E1 开环 QPS 扫描 | 12 | 可砍 | 可砍 | 建议 |
| E6 Adrenaline 对比 | 待定 | 定性 | 定性对比 + 分析模型 | 最好能实测 |
| **合计（推断）** | | **约 5–10 机时** | **约 40 机时**（加可选项约 55） | **175 机时以上** |

CCF-B 方案的总量：E0 4 + E2a 6 + E2b 12 + E1 简化 14 + E5 2 ≈ 38 机时，按单台 8×L20 每天跑 10 小时算，约 1 周机器时间，加上分析和写作，2027 年 2 月前可以完成。

---

## 5. 推荐方案的执行顺序

1. 你批准合入 PAT 修复（或决定改用 Triton），跑 E0。
2. 找到 L07 数据集，跑 E2a、E2b。**这一步决定论文主线和是否冲 ATC '27。**
3. 跑 E1 简化扫描和 E5。
4. 按 E2b 的结果改写 `outline-v2.md` 的 §6.7 和摘要数字；补相关工作（重点是 Adrenaline、semi-PD）。
5. 目标 HPDC 2027（CFP 发布后核实日期和页数）；同时可以把精简版投一个 workshop 拿反馈，前提是该 workshop 是非存档的。
