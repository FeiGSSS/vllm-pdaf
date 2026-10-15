# PAP 创新性评估（v2，对照"4. PD"分类）

日期：2026-10-14。依据：你 Zotero"IEI/内存/大模型/4. PD"分类中 30 篇最相关论文的正文（每篇只读摘要、设计、评估和相关工作部分），外加 Adrenaline、DynaServe 全文和 NVIDIA 的 Groq 3 LPX 技术博客。行号、引文和 arXiv 编号来自论文文本，部分论文只有中文译本，引文是转译；标"推断"的是我的判断。

---

## 1. 结论

**你的判断是对的：PAP 的三个支柱，单独看都已经有人做过，而且其中两个已经有顶会论文。** 剩下的是一个很窄、但确实没有人评估过的组合点，加上多轮放置里的"迁移"这一小块。

| PAP 的组成 | 已有工作 | 是否还能声称 |
| --- | --- | --- |
| **PA 节点**：同卡 MPS 切 SM，Prefill 与 Decode 共享同一份 KV，不搬 KV | **semi-PD**（2504.19867，MPS 切 SM + IPC 共享 KV，几乎就是 PA 节点）；Nexus（2507.06608，vLLM 内 Green Context）；**MuxWise（ASPLOS'26）**；RAPID-Serve（AMD）；Bullet（ASPLOS'26）；Adrenaline（MPS 在 Prefill 卡上切出 attention executor） | **不能**。而且这些工作都是动态切分，PAP 的静态 80/12 比它们弱 |
| **无状态 Projection + KV 常驻的 Attention** | **Lamina**（2405.01814，H100 跑无状态模型层、H20 存 KV 跑 attention，大 batch）；FastDecode（2024，attention 在 CPU）；**KernelFlume**（2026，权重节点 + 无权重 attention 节点，面向 agentic 长上下文）；MegaScale-Infer、Step-3、Janus、xDeepServe 等 AFD；**NVIDIA Vera Rubin + Groq 3 LPX**（GPU 做 prefill 和 decode attention，LPU 做 decode FFN） | **不能**。你在 KernelFlume 上批注的"这和 AFD 没有什么区别"，同样适用于 PAP 的这一部分 |
| **多轮：下一轮在 KV 所在处执行，不每轮搬 KV** | **AMPD（ICML 2026）**：会话绑定到持有 KV 的 D 节点，增量 prefill 在本地或远端执行，基于 Dynamo；**PPD**（2026-05）：第 2 轮起路由到持有 KV 的 D 节点 | **大部分不能**。你自己在 AMPD 上的批注"相当于将 KV 绑定在 D 节点，PAP 是绑定在 PA 节点"是准确的 |
| **组合：在 Prefill 卡上只切出 decode-attention 一小片，Projection 跨多个 PA 聚合，dense 模型，同构 GPU** | 30 篇里**没有一篇评估过**这个配置。最接近的是 How Far Can Disaggregation Go 的 Agg+AFD 模式（只做 MoE，FFN 池按副本划分），以及 When Does Disaggregation Pay 在动机里描述的 Vera Rubin 方案 | **可以**，但只是设计空间里一个没被测过的点 |
| **多轮里的 PA→PA 前缀迁移，以及"第 1 轮起就零搬运"** | AMPD、PPD 都是会话粘性绑定，没有迁移；PPD 第 1 轮仍要 P→D 搬 KV；AMPD 在本地做 prefill 时要暂停 decode batch | **可以**，这是差异最清楚的一块 |

---

## 2. 审稿人一定会拿来的三条反证

1. **When Does Disaggregation Pay**（Forys 等，2026）：在 GPU 上，"AF 在任何比例、任何模型上都没有超过不分离"，最好只有 0.96×；PDAF 只有在 attention 和 FFN 用不同硬件时才有收益。
2. **How Far Can Disaggregation Go**（2605.28302）："聚合部署仍然保持吞吐冠军"，AFD 的价值在交互延迟和显存，不在吞吐。
3. **PPD**：纯数据并行（Replica）在第 2 轮之后的 TTFT 最好。

这三条和 PAP 自己的数据一致：DP8 的 TTFT 是 PAP 的一半，PAP 的吞吐只比 DP8 高 8.4%。所以审稿人看到"PAP 吞吐最高"时，会先问是不是 DP8 没调好。

不过这里也有一个**可以反过来用的论点**（推断）：前两篇里 AF 在同构 GPU 上不划算，是因为 attention 池要额外买卡。PAP 的 attention 是从 Prefill 卡上切出来的 12 个 SM，用的是 Prefill 本来闲置的 HBM 带宽，几乎是"免费"的。这正好解释了为什么同样是同构 GPU，PAP 这个点可能划算，而它们评估的 AF 点不划算。这个论点 Adrenaline 讲过一半（单轮、部分卸载），没有人在 AFD 的框架下讲过。

---

## 3. 还能怎么讲

**不要再写"我们提出一种新的分离式架构"。** 建议的定位：

> NVIDIA 已经在 Vera Rubin + LPX 上用异构硬件实现了"Prefill 与 decode attention 留在 GPU、FFN 移到专用引擎"的分离方式。PAP 是在**同构的商用 GPU** 上实现同一切分的第一个开放系统：attention 不再需要独立的设备，而是从 Prefill 卡里切出一小片，因此 KV 从生成那一刻起就不再移动。我们回答两个问题：(1) 在同构 GPU 上，这个设计点什么时候比数据并行、PD 和 AFD 更好；(2) KV 被固定在 Prefill 卡上之后，多轮会话应该怎样放置。

注意：NVIDIA 博客只说 Dynamo 把 prefill 交给 GPU 构建 KV、decode 时 GPU 基于累积 KV 做 attention、LPU 做 FFN/MoE，没有说 prefill 和 decode attention 在同一张 GPU 上；它外置的也只是 FFN/MoE，不含 QKV/O 投影。引用时要按原文措辞，不要说成"和 PAP 完全一样"。

对应的贡献：

1. **成本模型与有效区间**：搬 KV 与搬激活的字节量对比，加上"attention 的硬件成本几乎为零"这一点。用它解释为什么 Forys、Wu 的 AF 在 GPU 上不划算而 PAP 可能划算，也解释 7 月短输出负载上的反证。可以直接借用 Song 等（A/F 比例）的闭式模型推导 7PA:1P，并引用 OpScale 的共置干扰系数论证 80/12。
2. **多轮放置与迁移**：和 AMPD、PPD 正面对比。PAP 的差异有三点：第 1 轮也不搬 KV；MPS 隔离使增量 prefill 不暂停 decode；粘性会导致负载不均，PA→PA 迁移把粘性和负载感知路由的优点结合起来（现有的 SESSION-ROUND-ROBIN 和迁移 A/B 数据正好支撑这一点）。
3. **系统实现**：vLLM 0.26、NVSHMEM 整步 Graph、开放实现。属于工程贡献，放在第三位。

**创新性的上限**：按这个讲法，CCF-B 是合理的。冲 ATC/EuroSys 需要至少在评估里实测对比一个 AMPD 式的"会话绑定 KV 持有者"基线和一个 semi-PD/Nexus 式的同卡切分基线，否则审稿人会认为最近的两类工作都没比。

---

## 4. 必须引用的论文（按优先级）

| 优先级 | 论文 | 用途 |
| --- | --- | --- |
| 必须对比 | AMPD（ICML'26）、PPD（2026） | 多轮放置，最直接的竞争者；AMPD 基于 Dynamo，可以作为实测基线 |
| 必须对比 | semi-PD、Nexus、MuxWise（ASPLOS'26） | PA 节点的先例，明确说 PAP 不声称同卡切分 |
| 必须对比 | Adrenaline、Lamina、KernelFlume | Attention 卸载 / 无状态 Projection 的先例 |
| 必须引用 | NVIDIA Vera Rubin + Groq 3 LPX 技术博客 | 工业界的同类切分，用来支撑动机 |
| 必须正面回应 | When Does Disaggregation Pay、How Far Can Disaggregation Go | 两篇"AF 在 GPU 上不划算"的反证 |
| 建议借用 | Theoretically Optimal A/F Ratios、OpScale | 推导 PA:P 比例与 SM 切分的分析模型 |
| 动机 | DualPath、ServeGen、Agentic AI Workload Characteristics、KernelFlume 用的 Codex/SWE-bench Pro trace | agentic 多轮负载的特征；DualPath 还会引出"HBM 放不下真实 agentic 会话"的质疑，需要在讨论中回应 |
| 一般引用 | MegaScale-Infer、Step-3、Janus、xDeepServe、CrossPool、FastDecode、Bullet、RAPID-Serve、POD-Attention、DistServe、Splitwise、Sarathi-Serve、TaiChi、DOPD | 相关工作各类 |

---

## 5. 对实验计划的影响

在已有计划（`experiment-plan.md`、`positioning-and-venue.md`）之外，按优先级：

1. **AMPD 式基线**（必须）：PD 部署下会话绑定到持有 KV 的 D 节点、增量 prefill 在本地执行。现有的会话粘性 RR 只是部分近似。
2. **成本模型的数字**：按 Qwen3-8B 的维度算出每轮搬 KV 与每 step 搬激活的字节量，并和实测的有效区间对上。
3. **DP8 调优审计**：确认 DP8 用了前缀缓存和合理的 chunked prefill，否则主结果站不住。
4. **实测 PD 的 KV 传输开销**（必须）：DistServe、Splitwise 都报告传输开销可以忽略（NVLink 下），PAP 需要给出 PCIe L20、16–32K 上下文下的实测占比，作为"KV 不搬家"的直接证据。
5. 可选：EcoServe 作为面向普通互连的额外基线。
6. 可选：一个 semi-PD 式的"同卡切分、完整 decode"变体，用来分离"Projection 外置"本身的贡献（和已有计划里的 A2 Projection 聚合消融是同一件事）。

---

## 附：逐篇要点

### 多轮类
- **AMPD**（He 等，ICML 2026，PMLR 306）：会话绑定到 D worker，D 持有 KV、负责全部 decode；增量 prefill 按实时 TTFT/ITL 统计决定在本地（会暂停 decode batch）或远端 P（从 D 读历史 KV，NIXL）；ILP 规划 P/D 与并行度。Qwen3-32B、Llama3.1-70B、Mixtral；4×8 H20；ToolBench、GAIA、HotpotQA、DuReader；基于 Dynamo。无迁移，迁移只在未来工作里提到。
- **PPD**（Li 等，UChicago，2026-05）：append-prefill 对 decode 的干扰比 full prefill 小约 10 倍；第 2 轮起按查找表决定走 D 还是 P；会话表粘性绑定 P/D 对；第 1 轮仍 P→D 搬 KV。Llama-3.1-8B 等，4×H100。
- **DualPath**（2602.21548，DeepSeek）：agentic 场景下 KV 从存储加载，Prefill 侧存储网卡成为瓶颈，改为经 D 侧加载再 RDMA 给 P。平均 157 轮、32.7K 上下文。重叠低，但会被用来质疑"历史 KV 全放 HBM"。

### Attention 卸载 / 无状态 Projection 类
- **Lamina**（2405.01814）：H100 无状态模型层、H20 存 KV 做 attention，KV 从 Prefill worker 逐层异步传过去；只评估 decode。与 PAP 的 decode 结构高度重叠，关键差异是 PAP 不搬 KV。
- **KernelFlume**（2026）：权重节点 + 按 token 范围存 KV 的无权重 attention 节点，可在 token 边界弹性扩展 attention；prefill 被视为正交，KV 在 prefill 后放到 attention 节点。Llama-3.1-8B，8×A6000 与 16×H100；Codex/SWE-bench Pro trace。
- **FastDecode**（ICML'24 workshop）：attention 和 KV 在远端 CPU，GPU 持有全部权重、batch 极大。PAP 结构的早期形态。
- **Tessera**（2026）：异构 GPU 对上按 kernel 粒度拆分，KV 在两边复制。重叠低。
- **Adrenaline**（2503.20552）：70–80% 的 decode attention 卸到 Prefill 卡，被卸载请求的 KV 留在 Prefill 卡；MPS 按 TTFT SLO 选 SM 比例；qkv 打包、二维 CUDA Graph。单轮 ShareGPT/OpenThoughts，8×A100。卸载到 80% 反而变差，因为远端 attention 掩盖不住；PAP 没有本地 attention，不受这个约束。

### AF 设计空间类
- **When Does Disaggregation Pay**（Forys 等，Imperial/Cambridge，2026）：屋顶线模拟器，ND/PD/AF/PDAF 四种模式，不含"prefill 与 decode attention 共卡"。动机部分描述 Vera Rubin 的切分正是 PAP 的切分。结论：GPU 上 AF 从未超过不分离。
- **How Far Can Disaggregation Go**（2605.28302）：128×B200 模拟，只做 MoE；Agg+AFD 是最接近 PAP 的模式（attention 卡同时做 prefill 与 decode attention），但 prefill 的 FFN 也外置，FFN 池按副本划分。结论：聚合部署保持吞吐冠军。
- **OpScale**（2608.13499）：以算子为单位扩缩容，Green Context 共置，含 dense 模型、真实系统；共置干扰系数可借用。
- **Theoretically Optimal A/F Ratios**（Song 等，2026）：rA-1F 的闭式最优比例；μP 越长 r* 越大；指出 attention 实例间的拖尾不均衡随 r 增大（r=32 时约 15%），与 PAP 7 对 1 的 barrier 问题对应。

### 同卡 P/D 切分类（只读摘要）
- **semi-PD**：MPS 按 SM 百分比切 prefill 与 decode 两个进程，IPC 共享统一 KV，动态调整。基本就是 PAP 的 PA 节点，差别只是 decode 进程跑完整 decode。
- **Nexus**、**MuxWise（ASPLOS'26）**：进程内 Green Context、共享 KV；MuxWise 明确批评 MPS 不够灵活。
- **Bullet（ASPLOS'26）**：libsmctrl 动态 SM 掩码，明确反对固定 MPS 切分。
- **RAPID-Serve**：AMD 上的 semi-PD 对应物。

### 其他
- **DynaServe**（2504.09285）：在任意 token 位置把请求切成两段，KV 照样搬，不重叠。
- 低相关：AlignedServe（SIGMOD'26）、POD-Attention（ASPLOS'25）、TaiChi、DOPD、SLO-aware P/D 分配、Prefill-as-a-Service、AMMA、Tessera。

### 剩余 28 篇（只读摘要，补读于同日）

分类共 77 条，去掉重复和 12 个网页、博客后约 58 篇论文，至此全部读过（30 篇读了设计与评估部分或全文，其余只读摘要）。剩余 28 篇里**没有一篇**做 PAP 的核心机制。

- **中等重叠**：FinDEP（2512.21487，MoE 的 attention/expert 分离与流水调度，自称 AFD）；EaaS（把 MoE 专家做成独立的无状态服务，经无 CPU 参与的 P2P 通信访问，和 PAP 的无状态 Projection + NVSHMEM 很像，但只针对 MoE 专家）。
- **低到中等**：EcoServe（同一实例内按时间做部分分离，面向普通互连，适合作为额外基线）、TraCT（用 CXL 共享内存搬 KV 并做机架级前缀缓存）、DisagMoE（训练中的 AF 分离）。
- **可用作动机的数字**：
  - Agentic AI Workload Characteristics（2605.26297）：缓存命中率 84.6–99.5%，decode 占 LLM 时间 91.0–98.6%，SWE-bench Pro 平均上下文 69K–80K。
  - KVCache in the Wild（ATC'25）：前缀命中率 54–62%，对话类多轮占 47–51%。
  - SplitZip：长输入和 agentic 负载下 KV 传输成为显著瓶颈。
  - EcoServe：完全分离高度依赖高带宽互连，正好对应 PAP 的 PCIe L20 场景。
  - Revisiting Disaggregated Serving（2601.08833）：PD 的收益取决于负载和 KV 传输介质，并不总是成立。
- **会被拿来反驳的**：
  - DistServe 报告 KV 传输不到总延迟的 0.1%（NVLink）；Splitwise 的逐层传输也只占端到端 0.8%。**PAP 必须自己在 PCIe L20、16–32K 上下文下实测 PD 的 KV 传输开销**，否则"KV 不搬家"的收益缺少直接证据。
  - KVCache in the Wild 的 API 类 trace 中，97% 的缓存命中来自单轮请求之间的共享前缀，不是多轮。PAP 的放置策略需要说明是否也能利用跨请求的前缀复用。
  - 真实 agentic 会话的上下文（69K–80K）和轮数远超 PAP 测试负载（16–32K、3 轮），会引出 PA 显存是否够用的问题，和 DualPath 的质疑相同。
- **基本无关**：StreamServe、TEMPO、Dual-Pool Routing、CXL-SpecKV、Seesaw、AIConfigurator、Prism、Batch Spec Decoding、Stream2LLM、RTT- or Bandwidth-Bound，以及模拟器（LLMServingSim 2.0、Frontier）、综述（Miao、From Attention to Disaggregation）和经典工作（Orca、Splitwise、DistServe，后三者作背景和基线引用）。
- 没有 PDF、未读：NanoFlow、DynamoLLM、Sarathi-Serve（均为已知工作，按需引用）。

---

## 来源

- 论文正文：Zotero `IEI/内存/大模型/4. PD` 分类中的 PDF（你的本机 iCloud 目录）。
- [NVIDIA 技术博客：Inside NVIDIA Groq 3 LPX](https://developer.nvidia.com/blog/?p=114202)；[Naddod 解读](https://www.naddod.com/ai-insights/nvidia-groq-3-lpx-introduction)
- [Adrenaline, arXiv 2503.20552](https://arxiv.org/abs/2503.20552)
- [How Far Can Disaggregation Go, arXiv 2605.28302](https://arxiv.org/pdf/2605.28302v1)
- [semi-PD, arXiv 2504.19867](https://arxiv.org/abs/2504.19867)
- [Nexus, arXiv 2507.06608](https://arxiv.org/pdf/2507.06608)
- [Lamina, arXiv 2405.01814](https://arxiv.org/html/2405.01814v2)
