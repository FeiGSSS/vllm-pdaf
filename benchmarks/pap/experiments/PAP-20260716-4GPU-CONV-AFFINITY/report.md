# PAP/PD 四卡多轮 conversation-affinity oneway milestone

> **历史阶段性证据。** 本报告冻结 12 会话、5 轮、固定长度负载下的 controlled
> 结论，不再定义当前四卡 testbed。当前对比使用 32 会话、10 轮随机长度 AIPerf
> 负载，见[审计后的 eager baseline](../PAP-20260721-AIPERF-AUDITED-CAPACITY/report.md)
> 和 [PAP 当前开发状态](../../../../docs/design/pap/status.md)。

日期：2026-07-16  
状态：accepted controlled milestone、dirty worktree、每个配置单次；不是
formal-clean release baseline

## 结论

在相同四张 NVIDIA L20、Qwen3-8B FP16/TP1 和同一套 60 请求多轮负载下，
`PAP 3PA1P` 的总完成时间、输出吞吐、median TTFT 和 median 端到端延迟均优于
PD 的 `1P3D`、`2P2D`、`3P1D` 三种配比。最快的 PD 是 `3P1D oneway`；PAP
相对它的输出吞吐高 `32.3%`，总完成时间低 `24.4%`，median TTFT 低
`30.4%`。

这个结论不应表述为“每个单项指标都更好”：PD `1P3D` 和 `2P2D` 的
median TPOT 分别是 `27.06 ms` 和 `32.82 ms`，低于 PAP 的 `41.24 ms`。
PAP 赢在把长会话 KV 分散到三个 PA、让一张不存历史 KV 的 Projection 汇聚
12 路 decode，以及更稳定的 TTFT/尾延迟，而不是在所有 PD 配比上都拥有最低
单 token 延迟。

## PD 基线决策

本 milestone 固定 PD 为 `NixlConnector` oneway。上游 vLLM 把 P→D 单向 KV
流定义为 standard disaggregated prefilling，`bidirectional_kv_xfer` 默认值也是
`false`。Bidirectional D→P 是面向 multi-turn-heavy 场景的可选优化，需要 stateful
proxy 保存 Decode block handle，并引入 TTL、失效和容量回收问题。

本 workload 每轮新增 3072 tokens、只生成 256 tokens。在 Prefill conversation
affinity 和本地 prefix cache 命中成立时，oneway 主要多算上一轮生成的约 256
tokens；twoway 则需要长期保留并反传不断增长的完整 Decode 历史。当前结果已证明后者
在 3P1D 单 Decode 容量下无法保证前进，因此后续四卡 PAP 性能开发只使用 oneway
作为 PD 主基线。twoway runner 保留为诊断工具，不进入 milestone gate，也暂不开发
pressure-LRU 私有分支。

上游依据：

- [vLLM NixlConnector usage](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)
- [vLLM bidirectional KV RFC](https://github.com/vllm-project/vllm/issues/32733)

## 固定负载

| 项目 | 值 |
| --- | --- |
| 模型/硬件 | Qwen3-8B，FP16，TP1，NVIDIA L20 × 4，GPU 0--3 |
| API | `/v1/completions`，exact token continuation |
| 会话/轮数 | 12 个 active conversations，5 轮，轮间 barrier |
| 到达 | 每轮固定 12 request/s |
| 首轮 | 4096 document tokens；实际 prompt 4109 tokens |
| 后续轮 | 上轮 prompt + 256 output IDs + 3072 corpus tokens + marker |
| 实际 prompt | 4109、7457、10805、14153、17501 tokens |
| 输出 | 每轮 256 tokens，`ignore_eos=true` |
| 调度限制 | `max_model_len=20000`、batched tokens 8192、12 sequences |
| PAP | `3PA1P`，PA GPU 0--2，Projection GPU 3，static MPS 72/20 |
| PD | `1P3D`、`2P2D`、`3P1D`，NIXL oneway |

Qwen3-8B 每个 token 的 FP16 KV 为
`2 × 36 layers × 8 KV heads × 128 × 2 bytes = 147456 bytes`。最后一轮每个
会话包含 17757 个 prompt+output token，12 个会话合计 213084 tokens。启动日志
中单张完整 PD GPU 可容纳约 173200 KV tokens；三个 PAP PA 合计可容纳
383760 tokens，且 conversation affinity 将每个 PA 固定为 4 个会话、约 71028
tokens。这个 shape 会同时施压 Prefill 算力、Decode 并发和长期 KV 容量，而不是
只测短 prompt 的瞬时吞吐。

## 路由合同

- PAP：新 `conversation_id` 按 PA 轮询，后续轮次回到同一 PA；本实验只有一张
  Projection。最终 PA assignment 为 `4/4/4`，请求为 `20/20/20`。
- PD：Prefill 与 Decode 各自维护 conversation router。新会话分别轮询，后续轮次
  固定返回原节点，避免把人工跨节点 miss 混入基线。
- 没有 conversation ID 的普通请求仍按 request round robin。

## 结果

| 配置 | 完成 | 总时间 (s) | 输出 tok/s | median TTFT (ms) | p90 TTFT (ms) | median TPOT (ms) | median latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PAP 3PA1P | 60/60 | 72.11 | 213.02 | 3012.52 | 3505.24 | 41.24 | 13315.14 |
| PD 1P3D oneway | 60/60 | 119.27 | 128.78 | 10941.97 | 15411.34 | 27.06 | 18355.47 |
| PD 2P2D oneway | 60/60 | 141.09 | 108.86 | 5921.23 | 7473.43 | 32.82 | 14236.21 |
| PD 3P1D oneway | 60/60 | 95.38 | 161.05 | 4328.80 | 6175.30 | 42.52 | 14518.73 |

相对各 PD 配比，PAP 输出吞吐分别高 `65.4%`、`95.7%`、`32.3%`；median
TTFT 分别低 `72.5%`、`49.1%`、`30.4%`。`2P2D` 在第 4 轮有四个约
68--69 秒 TTFT 的同侧尾部请求；表中 p90 未覆盖全部极端值，原始 max TTFT 为
69252.27 ms。

PAP 的 48 个跨轮 cache transition 全部精确通过，Attention/session、routing、
decode-token join 和严格日志 gate 均通过。PD 三个 oneway 配置的客户端有效性与
日志 audit 通过，但当前 PD 结果只能从请求历史推导期望 cache token，状态为
`requires_external_validation`；它没有 PAP 的 Prefill `actual_cached_tokens` 证据。

## 排除项与限制

`PD 3P1D twoway` 在第 5 轮失去进展：Decode 日志稳定在 `Running=0`、
`Waiting=3`、GPU KV usage `92.3%`、external prefix hit `100%`，四张 GPU 均为
0% utilization。该运行被终止，没有性能结果，不纳入表格。它说明在本 workload
下把生成侧 KV 反向保留并不能作为更强的 3P1D 基线；runner 后续默认把单请求超时
限制为 180 秒。

所有有效配置都只跑了一次，且代码处于 tracked-dirty 状态，因此本 milestone 冻结
的是路由、workload、PD-oneway 基线选择和阶段性容量结论，不替代三次 clean
formal。PAP v3 的原始 `result.json` 继承了旧默认值
`NVIDIA-L20x2`；实际 placement 和服务日志均为 L20×4。runner 已改为按拓扑和 TP
自动生成硬件签名，原始结果不回写。

## 原始证据

- PAP：`runs/20260716_4gpu_multiturn_pap_3pa1p_c12_4k_plus3k_v3/raw`
- PD 1P3D：`runs/20260716_4gpu_multiturn_pd_1p3d_oneway_c12_4k_plus3k/raw`
- PD 2P2D：`runs/20260716_4gpu_multiturn_pd_2p2d_oneway_c12_4k_plus3k/raw`
- PD 3P1D：`runs/20260716_4gpu_multiturn_pd_3p1d_oneway_c12_4k_plus3k/raw`
- 排除的 twoway：`runs/20260716_4gpu_multiturn_pd_3p1d_twoway_c12_4k_plus3k/raw`

复现实验使用 `benchmarks/pap/scripts/run_pap_workload.sh` 和
`benchmarks/pap/scripts/run_pd_multiturn_topology.sh`；每个 run 目录的
`effective_config.env` 是具体配置来源。
