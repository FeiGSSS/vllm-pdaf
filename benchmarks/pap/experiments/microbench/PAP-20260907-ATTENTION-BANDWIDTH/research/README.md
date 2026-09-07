# 小 SM Attention kernel 优化：讨论依据

日期：2026-09-07。范围：保留 Triton 与 PAT，讨论精确 FP16 Attention 的
优化和研究候选；本次没有新增 GPU 实验或修改生产 kernel。

## 计量边界与已有证据

工作单位是一个 PA 本地 batch 在一层上的 decode Attention 调用；不是整个
客户端请求，也不是全模型 decode step。调用可以包含主 kernel 和合并 kernel。
逻辑 KV 字节数按每个 query 的完整上下文累计；unique KV 字节数按物理共享块去重。
两者除以调用 GPU 时间得到有效 payload 带宽，都不等于硬件 DRAM 计数器带宽。

原始实验、正确性修复与限制见上级 README.md。固定输入、轮换 KV 池的结果：

| 输入 | 12 SM | 20 SM |
| --- | ---: | ---: |
| Triton，B1，32K | 294.0 us | 248.5 us |
| PAT，B2，每条 32K，共享 16K | 400.3 us | 326.2 us |

第二行 unique payload 带宽为 503.0 / 617.2 GB/s。若软件能让 12 SM
达到这一 20 SM 延迟，则需降低约 18.5% 延迟、提高约 22.7% 有效带宽。
这是实验目标，不能由 SM 对照推导软件一定能达成。isolated probe 的其他分区空闲，
尚不能推导 Prefill 同时运行时的收益。硬件 DRAM/L2/stall 证据仍缺失。

## 本地实现检查

- `vllm/pap/attention/triton_backend.py:50`：12/20 SM 都选
  num_splits=8、block_h=4、num_warps=4、num_stages=1；这只是候选调参空间，
  参数固定本身不是缺陷证据。
- `vllm/v1/attention/ops/triton_decode_attention.py`：已有 GQA 组内 KV
  复用、Split-KV 和 reduction。不能把加入这些机制作为新贡献。
- `third_party/pat/csrc/prefix_tree.h:591`：拆分数量阈值依据 KV head 数，
  8 KV heads 对应 27；该处不接收实际 SM 配额。不要从 27 推测设计的目标 GPU，
  也不要简单按 SM 比例缩小阈值。
- `third_party/pat/csrc/pat_fwd_kernel.h:139`：已有 cp.async、wait、下一块
  K 预取。研究对象是重叠程度、寄存器/共享内存占用及等待依赖，不是补上异步复制。

预期 tile 流程为地址解析、KV 读取、QK、softmax、PV，最后跨 split 合并。
现有实现部分重叠这些阶段。进一步预取下一 tile 的地址与数据，可能减少等待；
也可能因缓冲区和寄存器增加而降低驻留并行度。需要指令、资源和时间证据判别。

## 直接相关的公开工作

| 工作 | 已有技术和适用边界 |
| --- | --- |
| [FlashInfer](https://arxiv.org/html/2501.01005v1) | 持久化任务调度、多 tile、GQA 复用；附录 D 按 SM 和驻留能力设置 CTA 数，附录 E 明确允许指定 SM 数支持算子重叠。这是必须比较的强基线。 |
| [LeanAttention](https://arxiv.org/html/2405.10480v1) | Stream-K 分配 Attention tile，平衡工作并在单 kernel 中合并；跨 CTA 同步不能不加验证地移入受限 SM 环境。 |
| [PAT](https://arxiv.org/html/2511.22333v3) | 前缀复用、多 tile、长 KV 拆分与精确 softmax 合并。复用与负载均衡并非空白。 |
| [FlashAttention-3](https://tridao.me/blog/2024/flash3/) | 内存、矩阵计算、softmax 的异步重叠；Hopper 的 TMA/WGMMA 路径不能直接套到 L20，需用其支持的机制重新设计。 |
| [PackInfer](https://arxiv.org/html/2602.06072v1) | 异构请求分组、前缀复用与连续 KV 布局；仅提出 compute/I/O 联合 packing 不能据此主张新颖。 |
| [POD-Attention](https://arxiv.org/html/2410.18038v2) | Prefill/Decode Attention 的联合执行和 SM 调度；PAP 当前跨进程组织并不直接等价于其融合 kernel。 |

另查到 [GPU NUMA Attention](https://arxiv.org/html/2511.02132v1)：主要针对
MI300X chiplet 映射，不把其收益或拓扑机制直接外推到 L20。

## 建议的可证伪研究候选

面向固定 SM 配额、共享前缀和 Prefill 争用，联合选择 query 分组、KV split
粒度、驻留 CTA 数和预取深度；保持 FP16、全部 KV、稳定 softmax 合并及现有请求输入。
目标是在满足 Prefill 延迟约束时缩短 Attention 调用，而非单独最大化读取流量。
规划应在已有 batch 内完成，不能靠等待更多请求制造共享前缀收益。

先做两条可独立归因的工程对照：SM/shape 参数选择；保持任务划分的访存流水改造。
分别对 B1/无共享 Triton 和有共享 PAT 测试。将参数充分调优的现有实现与指定
SM 数的 FlashInfer 作为候选基线，先验证布局、精度和执行约束匹配再比较。

可能的新意需要证明“争用约束下的联合选择”超越已有资源感知调度与静态调优；
本次检索不足以确认新颖性。若最优静态配置已达到相同效果，或 isolated Attention
加速却恶化 TTFT/ITL/goodput，则该研究假设不成立或不满足 PAP 目标。

测量顺序：固定形状的 12/20 SM 对照；同配额下加入一致 Prefill 背景负载；
全系统固定工作负载比较 TTFT、ITL 分位数和同一 SLO 下 goodput。区分这三层证据，
不把 kernel 加速直接换算为 ITL 加速。需要保留每次实际配置和资源映射。

## 检索记录与范围

原始检索文件及本次实验验证日志以原文打包到
[research-and-validation.tar.gz](../results/research-and-validation.tar.gz)，
逐文件校验见 [manifest](../results/research-and-validation-manifest.json)。
在本实验目录解压即可恢复下列 `research/` 文件；原文中的姓名、断词和错误信息
保持不变。E2E 瓶颈实验的验证日志另保存在其 `results/` 下的同名归档中。

`paper-search.txt` 保存 2023–2026 两条查询的聚合检索；各源命中为
arXiv 12、DBLP 0、OpenAlex 12、OpenReview 0、Semantic Scholar 12、Crossref 12；
去重后 46 条，合并 2 条跨源重复。命中包含无关工作，不能作为 46 篇相关论文计数。
DBLP SSL、OpenReview 缺依赖，以及部分 API 限流/服务错误见
`paper-search-errors.txt`；零命中不代表不存在论文。
Tavily 结果保存在 `decode-search.json` 和 `prefix-search.json`；上述结论再以论文
和作者页面核实。此次是有界相关工作检索，不是完整新颖性审查。

按模型历史技能查阅了
`/home/fei/.codex/skills/model-pr-history-knowledge/vllm/qwen3-core/README.en.md`
和 `sglang/qwen3-core/README.en.md`。QK norm/RoPE 融合线索不直接解释本次
Attention KV 遍历测量区间，因此没有据此提出模型算子修改或性能结论。
