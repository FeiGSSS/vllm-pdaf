# DP / PD 与新 PAP 对齐配置的重测，2026-09-07

主结果只使用 `runs_aligned/` 与 `pap_clean/` 中通过全部审计的运行。每种架构两轮，每轮 60 会话、180 请求，共 1800 请求。下表均为客户端逐请求指标的平均值，两轮样本等权；范围是两轮均值的范围，不是置信区间。

| 架构 | 平均 TTFT | 平均 TBT | 单轮平均 TTFT 范围 | 单轮平均 TBT 范围 |
| --- | ---: | ---: | ---: | ---: |
| DP8 | 3.82 s | 144.37 ms | 3.77–3.87 s | 143.40–145.34 ms |
| PD 6P2D | 15.08 s | 45.90 ms | 14.58–15.59 s | 44.27–47.54 ms |
| PD 4P4D | 30.13 s | 26.37 ms | 29.85–30.41 s | 26.36–26.38 ms |
| PD 2P6D | 65.44 s | 25.11 ms | 65.40–65.49 s | 25.11–25.11 ms |
| PAP 7PA1P | 14.54 s | 43.52 ms | 13.65–15.42 s | 42.68–44.37 ms |

TTFT 为客户端发起请求到收到首 token 的时间。TBT 使用 AIPerf 逐请求 inter_token_latency，包含流式交付的影响，不把它等同于 GPU step 耗时。p50、p95、分轮结果和按会话轮次的 TTFT 保存于 [comparison.json](comparison.json)。

统一的负载为 Qwen3-8B FP16、TP1、native 32768 上下文、无 YaRN、8 张相同的 L20、60 并发、60 会话 × 3 轮。数据文件和 materialized input 哈希相同，每个会话轮次的输出长度一致。实际多轮反馈会追加模型生成文本，不能声称所有后续 prompt 逐字节相同。

关键参数已对齐：

- Prefill 总 token 预算 2048；PD Decode 与 PAP Projection 预算 256；最大序列数 256；block size 16。
- PD Prefill 关闭异步调度，PD Decode 与 PAP Projection 开启。DP8 混合调度使用 2048 的统一预算及异步调度。
- V2 model runner、prefix caching、chunked prefill 均开启；KV-aware Dynamo 路由，Prefill load scale 明确设为 2.0。
- 路由器 active-block tracking、Prefill-token tracking、KV reuse assumption 均为 true，output-block tracking 为 false；indexer 线程数均为 4。这些默认值已通过两边源代码核对。
- 相同 AIPerf 客户端和数据，concurrency 模式，不发送预热请求；服务与 Graph 就绪后留出 30 秒。

[parameter_alignment.json](parameter_alignment.json)保存参数名的对应关系。DP/PD 的完整模型 worker 使用 90% 显存预算；PAP Projection 使用自身推导的显存策略。DP/PD 保留原生 piecewise CUDA Graph，PAP Attention/Projection 保留 whole-step Graph。DP/PD 使用隔离的官方 Dynamo 1.4.1 / vLLM 0.26.0；PAP 使用已有开发工作区。此次是这些实现的配置对齐比较，不声称代码版本或架构专用机制完全相同，也没有对所有参数做最优搜索。

每次发请求前，核验实际 worker / EngineCore 数量、父子关系、CUDA_VISIBLE_DEVICES、EngineCore 的 GPU UUID、角色、token 预算、异步调度和前端真实路由权重。DP/PD 为八个独立 worker 与八个 EngineCore；PAP 为七个 Prefill、七个 Attention、一个 Projection，十四个分区 CUDA context 对应七组 80/12 SM，Projection 使用 GPU 7。每轮前后均检查 GPU 进程，等待 CUDA/NVML 异步释放结束；前一轮资源没有释放时不会开始下一轮。

DP/PD 第二轮直接读取第一轮保存的有效配置，只改变运行 ID、输出目录、discovery 路径和 namespace；其余配置必须一致。两轮请求、正确性日志、CUDA Graph、NIXL 传输（PD）和资源清理审计均通过。PAP 同时检查 routing、decode-token join 和 session drain。日志、实际进程快照、GPU 映射及配置都在每个 run 目录中。

本次补齐了共享 Dynamo runner 的重放配置名称、显式异步调度/V2/路由权重参数、dataset checksum 校验、namespace 前置校验以及不忽略 gitignored 日志的 fail-closed 审计。修改位于 `benchmarks/pap/scripts/run_dynamo_workload.sh`，未修改 DP/PD 模型实现或本轮 PAP 优化代码。

有以下记录被排除，未混入主表：GPU 被本任务旧 MPS 残留占用的启动、非法 namespace 启动，以及 `runs_v3/` 中仍使用路由权重 1.0 的诊断运行。遗留的七个 MPS 服务经确认没有客户端且归属本任务后已清理，PAP 在清理后的环境重新测了两轮，未沿用旧 PAP 数值。DP8 曾在退出后被 NVML 短暂列为 `[No data]`，确认进程已退出并等待列表清空后才继续；该次测量本身已经完成并通过审计。

DP/PD 可重放配置保存在各目录的 `effective_config.env` 中，所需的 `DYNAMO_*` 输入名称已经完整保存。通用参数在 [common.env](common.env)，例如：

```bash
cd /home/fei/research/PD/vllm-pap
set -a
source /tmp/pap-aligned-baselines.an6e3tbl/common.env
DYNAMO_ARCHITECTURE=6p2d
set +a
bash benchmarks/pap/scripts/run_dynamo_workload.sh
```

改为 `dp8`、`4p4d` 或 `2p6d` 即可复用该负载与参数。默认会创建新的带时间戳的运行目录。精确重放某个 `effective_config.env` 时，需要使用新的 `DYNAMO_RUN_ROOT`、`DYNAMO_AIPERF_OUTPUT_DIR` 和 discovery 路径，以免覆盖原始证据。软件环境、源代码与模型/数据路径也必须保持一致。

共享入口的 15 项测试通过，pre-commit 检查通过；见 [测试日志](final-launch-tests.log)、[检查日志](final-lint.log)、[本次入口改动补丁](launcher_changes.patch)。[源码核验](final_source_audit.json)确认正式运行后的 launcher 源码，以及此前已测试的 PAP 优化源码均未发生变化。
