# PAP 12 SM Attention 有效带宽

2026-09-07，独立 worktree `research/pap-bottleneck-20260907`，8×L20，Qwen3-8B
FP16，TP1。**完整 PAP 共卡运行中，12 SM 的逻辑 KV 有效带宽约为 Triton
389 GB/s、PAT 470 GB/s；PAT 按去重后的 KV 数据量计算约 406 GB/s。**
固定输入、无 Prefill 干扰、轮换 KV 的独立实验中，12 SM 达到约
449–493 GB/s（Triton）；PAT 的逻辑带宽约 656–671 GB/s，去重 KV 带宽约
492–503 GB/s。以上均为有效数据量/时间，**不是 DRAM 硬件字节计数**。

测量过程中发现并修复了 PAT 分段编号冲突。下面的主要结果使用修复后的调度元数据；
CUDA Attention 计算内核和 Triton 参数没有修改。旧 PAT 的性能数字不能作为正确
推理的加速基线。修复仅在本 worktree 和隔离依赖目录，主工作目录及其 venv 未变。

## 带宽口径

工作单位是一个 PA 中，一层 Attention 对其当前请求子 batch 的计算。一个全局
Decode step 包含 36 层，并可同时涉及七个 PA；不能把各 PA 的时长相加当作 ITL。

Qwen3-8B 的 KV 为 8 个 KV heads、head dimension 128、FP16：

```text
每层每 token KV 字节数 = K/V 两份 × 8 heads × 128 × 2 bytes = 4096 bytes
逻辑有效带宽 = 4096 × Σ各请求 seq_len / Attention 时间
去重 KV 有效带宽 = 4096 × 按物理 block 去重的有效 token 数 / Attention 时间
```

GB/s 使用十进制。汇总为总字节数除以总时间，不平均不同大小调用的速率。
这两个分子都没有计入 Q、output、partial output、LSE、metadata、重复 cache-line
读取等全部实际流量。L2 命中也会减少 DRAM 流量。因此去重 KV 带宽仍不是实测
DRAM 带宽，不能用它直接声称达到显存峰值的某个百分比。

固定工作量下提高有效带宽与降低 latency 等价；PAT 还可以通过减少重复访问来
降低 latency，不必伴随实际 DRAM 吞吐量上升。

## 修复后，完整 PAP 的当前 trace

`bandwidth_fixed12_trace`：7PA1P，Prefill/Attention 为 80/12 SM，并发 30，
60 会话 × 3 轮，180 请求，native 32768，无 YaRN，Prefill token 预算 2048。
实际模块来自本 worktree，PAT 来自 `.local/pat-rankfix`；进程、GPU 分区、Graph、
请求、token join、commit/reclaim 和清理审计通过。

记录器在每层 KV append 后、paged Attention 前后写入 GPU 时间标记，**不把
NVSHMEM 等待、KV append 或整个 ITL 当成 Attention kernel 时间**。每个 PA 的
seq_len、backend 和 kernel 时间通过同一个 local epoch 对齐。只比较设备内时长，
没有跨 GPU 相减绝对时钟。

| Backend | PA-step 数 | 层调用数 | 平均逻辑 context tokens/PA-step | 平均每层时间 (µs) | 逻辑有效 GB/s | 去重 KV GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | 3993 | 143748 | 18962 | 199.4 | 389.5 | 389.5 |
| PAT，修复后 | 4441 | 159876 | 44805 | 390.3 | 470.2 | 405.6 |

这是 303624 个层调用的汇总，两种 backend 的 batch/context/共享情况不同，
不能根据两行速率直接判断同样输入时谁更快。
本轮客户端平均 TTFT 5.927 s、请求平均 ITL 45.212 ms、吞吐 2.764 req/s。
它是带原生记录器的单次诊断，不是修复前后服务性能的正式 A/B，也不建立模型任务
质量指标。KV 输出正确性另由下述固定输入 FP32 reference 检查支持。

证据：[汇总及原始 trace 摘要](fixed_trace_bandwidth.json)、
[逐 PA-step 记录](fixed_pa_step_samples.jsonl)、
[完整服务运行](../../e2e/PAP-20260907-BOTTLENECK/runs/bandwidth_fixed12_trace)。

## 固定工作量，12 SM 与 20 SM

同一张 GPU 0，分别使用 80/12、72/20 静态 MPS 分区，只有 Attention 客户端
执行 GPU 工作，另外一个分区没有 Prefill 负载。启动时检查所有 GPU 无计算客户，
工作进程核对实际 SM 数和 partition。输入固定 seed 42，物理 block 顺序打散；
同一对照中的 batch、context、共享 block 布局和 backend 相同。

读取设备属性得到 L2 大小 **100663296 bytes = 96 MiB**。为减少重复测单层 KV 的
L2 驻留优势，主要结果在四份数据相同、地址不同的 KV 之间轮换，实际访问工作集为
256–768 MiB，超过 L2。保持生产 KV stride。另保存热缓存实验作对照。
轮换降低跨调用复用，但不保证每次 load 都来自 DRAM。

每个 case 至少预热一秒 GPU 时间，然后七个样本，每样本 100 次调用。CUDA events
计时包括 eager launch 间隙，排除准备、reference 检查、Prefill 和传输；不是
生产 whole-step Graph 的逐层时间。未锁定 GPU 时钟，这些是探索性测量。
所有 case 在计时前通过相同的 FP32 Attention reference，`rtol=1e-2, atol=1e-4`。

| 输入与 backend | 12 SM (µs) | 20 SM (µs) | 12 SM 逻辑 GB/s | 20 SM 逻辑 GB/s | 12 SM 去重 KV GB/s | 20 SM 去重 KV GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton，B1 × 16K | 149.4 | 126.8 | 449.3 | 529.2 | 449.3 | 529.2 |
| Triton，B1 × 32K | 294.0 | 248.5 | 456.5 | 540.2 | 456.5 | 540.2 |
| Triton，B2 × 16K，共享 8K | 272.3 | 177.3 | 492.9 | 757.1 | 369.7 | 567.9 |
| PAT，B2 × 16K，共享 8K | 204.6 | 164.6 | 655.9 | 815.6 | 491.9 | 611.7 |
| PAT，B2 × 32K，共享 16K | 400.3 | 326.2 | 670.6 | 823.0 | 503.0 | 617.2 |

这些数据排除了“Prefill 变慢导致 Decode 压力更小”这一条路径：本实验没有
Prefill，且每次 Attention 处理的形状完全固定。20 SM 在这些形状上确实有更低的
执行时间。但这不代表 12 SM 一定能通过软件优化追回全部差距。

同样 B2×16K、共享 8K 的输入上，PAT 比 Triton 少约 24.8% 延迟，说明减少重复
KV 访问是实际有效的方向；不能把它解释为硬件 DRAM 带宽增大了 33%。

原始样本和最大数值误差：[12 SM](runs/fixed12_rotating/measurement.json)、
[20 SM](runs/fixed20_rotating/measurement.json)。热缓存对照保存在
`runs/fixed12_warm/` 与 `runs/fixed20_warm/`，两者不得与轮换结果混合计算收益。

## 为什么没有给 DRAM 硬件带宽

系统原有 NCU 为 2022.4。已从 NVIDIA 官方包索引下载并校验 2026.2.1，隔离解压到
`/tmp/pap-ncu-verified-2026.2.1`，没有替换系统工具。
包 SHA256：`6829651ceeb0c3f65890b9f727b74d1e550fed58c454e11c2c87442295e4eb70`。

按照 [NVIDIA MPS profiling 文档](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#multi-process-service)，
准备了单客户端 range profiling，采集 `dram__bytes_read.sum`、
`dram__bytes_write.sum`、`gpu__time_duration.sum` 和 DRAM throughput 指标。
单客户端是为了避免把其他 MPS 客户端的流量混入同一报告。

NCU 在 `runs/ncu12_triton16k/worker.log` 报告 profiling driver resource unavailable，
提示 DCGM 等工具可能占用采样资源。暂停共享 DCGM profiling 的操作被自动审批拒绝，
理由是会影响共享监控且需要明确授权。已向用户请求临时暂停并恢复的许可；获得
许可前未暂停该监控。因此当前没有可用的 `.ncu-rep`，**没有把有效带宽冒充硬件
DRAM 吞吐量**。采样命令、失败日志和配置均保留，可在授权后继续。

## 测量中发现的 PAT 正确性问题

`pat_b2_16k_shared8k` 在原依赖上未通过 reference。单个请求的两个 CTA 写入
同一 partial-output split rank 10，rank 11 没有写入，但 gather 会读取整个
`0..num_splits-1` 区间。冲突位于部分 Attention 结果槽位，不是 KV block 分配。
重复执行结果变化与这一写入冲突一致；Triton 同输入 reference 误差仅约 `2.36e-7`。

根因在 [balancePack](../../../../../third_party/pat/csrc/prefix_tree.h)：
拆分共享 CTA 时取第一个请求的下一个 rank，再递增所有请求的计数，但各请求的
计数可能已因私有分段而不同。修复后，只有所有参与请求的下一 rank 相同时才能
继续拆分该共享 CTA；仍可拆分其他合适的块。导出前再检查每个请求的 rank 是否
完整、唯一且在范围内，发现冲突时直接拒绝构建计划。

没有修改 Attention CUDA 数学内核、KV 精度或放宽误差阈值。修复后的 PAT
最大误差约 `2.4e-7`，重复执行与不同合法 KV stride 的诊断均一致。

- 新增四个 CPU 分段回归 case：旧依赖 3 failed / 1 passed，修复后 4 passed。
- 现有 Attention pytest 文件：23 passed / 1 skipped；默认沙箱没有 CUDA 驱动，
  GPU case 跳过，另在真实 12/20 SM 微基准上完成 FP32 reference 检查。
- 修复依赖只安装到 `.local/pat-rankfix`，使用 `--pat-root` 显式选择并保存二进制
  SHA256；完整 PAP 运行验证实际加载了该目录。
- 原依赖和旧 trace 均保留。`trace_bandwidth.json` / `pa_step_samples.jsonl` 是
  修复前观测，PAT 部分不能作为正确推理性能结论；当前使用 `fixed_*` 文件。

## 对下一步优化的判断

12 SM 在固定工作量下已经能提供约 450–500 GB/s 的有效 KV 数据处理率，而同形状
20 SM 仍有差距，值得研究更多 memory-level parallelism、合适的 split 数和 tile
配置。现有 Triton 在两种配额下均为 split=8、block_h=4、warps=4、stages=1、
block_n=32，尚未针对这些形状分别搜索最优参数。

但完整服务 trace 的形状分布不同，也有 Prefill 共卡，不能把服务与独立实验的
带宽差直接全归因于 Prefill 争用。下一步应固定同一 Attention 形状，对比无
Prefill 与可控 Prefill 背景，再结合 DRAM/L2/SM 计数区分访存并发不足、缓存命中、
实际显存争用和其他等待。当前带宽数字不能独立证明所有形状都纯粹受 DRAM 限制。

在 worktree 根目录复测：

```bash
.venv/bin/python benchmarks/pap/experiments/microbench/PAP-20260907-ATTENTION-BANDWIDTH/run_probe.py new12 --sms 12 --pat-root .local/pat-rankfix --cache-mode rotating
.venv/bin/python benchmarks/pap/experiments/microbench/PAP-20260907-ATTENTION-BANDWIDTH/trace_bandwidth.py --source benchmarks/pap/experiments/e2e/PAP-20260907-BOTTLENECK/runs/bandwidth_fixed12_trace --label fixed
```

`run_probe.py` 在唯一短路径创建 MPS socket，记录配置并校验 SM 数，退出时删除
分区和停止本次创建的 daemon。最初的长 socket 路径启动失败已修正，失败记录
保留。`pat-build.log`、`pat-regression-before.log`、`pat-regression-after.log`、
`pat-suite.log` 保存构建和验证证据。原始服务 trace、逐请求记录、微基准样本和
失败尝试均保留；此次没有提交或推送。

归档位于 [results](results)：完整修复后 E2E、微基准运行和修复依赖三个归档，
共 248 个普通文件，均逐文件读回校验 SHA256。清单见
[manifest.json](results/manifest.json)，源码修复见
[pat-split-rank-fix.patch](results/pat-split-rank-fix.patch)。Python、测试与文档的
pre-commit 检查通过；原生扩展已编译并完成上述验证。最终八张 GPU 无计算进程，
共享 DCGM 监控未暂停。
