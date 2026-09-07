# DP8 / PD / PAP aligned comparison, 2026-09-07

本目录保存刚完成的两轮正式实验：DP8、PD 6P2D、PD 4P4D、PD 2P6D、PAP
7PA1P 各两次，共 10 次运行、1800 个请求。每次重新启动服务；第二轮不是同一
服务上的追加请求。DP/PD 第二轮重放第一轮的有效配置，只修改输出路径和运行标识。

| 架构 | 两轮平均 TTFT (s) | 两轮平均 TBT (ms) |
| --- | ---: | ---: |
| DP8 | 3.822 | 144.372 |
| PD 6P2D | 15.084 | 45.904 |
| PD 4P4D | 30.134 | 26.368 |
| PD 2P6D | 65.443 | 25.113 |
| PAP 7PA1P | 14.537 | 43.524 |

TTFT 是客户端请求到首 token 的延迟；TBT 使用 AIPerf 的逐请求平均
`inter_token_latency`，不是 GPU step 时间。两轮等权；p95 使用排序后
`int(0.95 * N)` 的样本，与历史分析保持一致。PAP 相比 6P2D 的均值分别低
3.63% / 5.18%，但两轮范围重叠，不能据此宣称稳定胜出。

固定负载是 Qwen3-8B FP16、TP1、native 32768 上下文、无 YaRN、8 张 L20、
60 会话 × 3 轮、并发 60。Prefill/DP token 预算 2048，Decode/Projection
预算 256，最大序列数 256，Dynamo Prefill load scale 2.0。输入文件哈希和
各会话轮次的输出长度一致；后续轮次包含实际生成反馈，不能声称 prompt 全部
逐字节相同。数据集随本次提交保存于
[`s60-t3-native32k-stratified-seed42`](../../../datasets/agentic-code/s60-t3-native32k-stratified-seed42)。

DP/PD 使用隔离的官方 Dynamo 1.4.1 / vLLM 0.26.0；PAP 使用开发工作区。
显存分配与 CUDA Graph 机制保留各架构策略。本实验对齐公共参数，不代表实现
版本相同或各架构都经过最优搜索。所有正式运行均通过启动拓扑、实际进程和设备
映射、正确性及退出清理审计。失败启动、路由权重 1.0 的诊断运行和旧 PAP
测量不计入主结果；最终 PAP 是清理旧 MPS 服务后的两次新运行。

本次没有配置 goodput SLO，不能从吞吐直接推导 goodput。原始逐请求延迟和时间戳
已保留，后续可用统一阈值离线计算。

## 保存内容与校验

- [完整报告](results/REPORT.md)、[分轮统计](results/comparison.json)、
  [参数对应](results/parameter_alignment.json)。
- `results/configs/`：十次运行的 requested/effective 配置及主要审计，便于直接查看。
- `results/archives/`：逐运行无损压缩包，保留 AIPerf 原始 JSON/JSONL、服务日志、
  GPU 监控、进程快照、配置、源码补丁、依赖快照及 etcd 状态。没有删减普通文件。
- `protocol.tar.gz`：原始调度/分析脚本、测试日志、排除记录和 PAP 优化来源报告。
- [manifest](results/manifest.json)：每个归档成员的原路径、大小和 SHA256；
  `results/SHA256SUMS` 校验所有留存结果文件。

从仓库根目录离线核验全部字节，并从原始请求重新计算 TTFT/TBT mean、p50、p95：

```bash
.venv/bin/python benchmarks/pap/experiments/e2e/PAP-20260907-ALIGNED-DP-PD-PAP/verify.py
```

附加 `--extract-to /tmp/pap-aligned-restored` 可解压到一个尚不存在的目录。
解压后恢复 `runs_aligned/`、`pap_clean/`、`protocol/` 三个目录。每个运行的
`aiperf/profile.jsonl` 都可独立用于后续 SLO 分析。

## 重放边界

这是历史证据归档，不新增用于修改生产代码的 shell 文件，也不自动重新执行
GPU 实验。原始 requested/effective 配置保留当时的绝对路径以维持证据完整性。
`protocol/` 中的脚本记录当时的执行过程，含旧路径，不能直接作为当前入口运行。

新测量使用共享的
[`run_dynamo_workload.sh`](../../../scripts/run_dynamo_workload.sh) 或
[`run_pap_workload.sh`](../../../scripts/run_pap_workload.sh)，并参照
[`DYNAMO.md`](../../../scripts/DYNAMO.md)。从相应保存的 effective 配置开始，
将运行 ID、输出目录、namespace、discovery 路径改到本目录的 `runs/<timestamp>/`；
迁移机器时还需核验模型、数据、依赖和 GPU 身份。不得覆盖历史 results。
必须复用原协议的启动前后及活跃进程/设备审计，通过后才能纳入比较。

归档中的源码基线是 `001bf7fdfa` 加当时工作区补丁；
[`final_source_audit.json`](results/final_source_audit.json) 保存核心源码哈希。
本次同时提交实验所用实现和启动修复，原始补丁保留当时完整工作区状态；其中的
`AGENTS.md` 差异只是历史快照，不作为这次代码提交的修改内容。

提交前检查要求将回收边界的 `torch.cuda.synchronize()` 改为
`torch.accelerator.synchronize()`。原始测量对应前者，提交源码使用后者；没有
重新执行 10 次完整性能实验。两条 CUDA 流的实际设备验证通过，受影响的控制
测试再次通过 13 项；其余已记录核心源码哈希保持一致。变更前后哈希和验证结果
见 [提交验证记录](results/commit_validation.json)。提交前完整相关回归为 157 项通过。
