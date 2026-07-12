# 同机 PD/NIXL KV 传输根因与校正基线

日期：2026-07-13

状态：根因已定位；官方 push 路径已完成 16K 诊断验证；5 轮并发 testbed 已实现，
正式矩阵结果待本文后续追加。

## 1. 结论

原 1P1D 基线中约 `0.42 GiB/s` 的 KV 传输不是 GPU1/GPU2 的物理 PCIe 上限，
而是当前 NIXL/UCX 组合在执行 GPU-to-GPU `GET/READ` 时选择了 TCP software
emulation。GPU1/GPU2 虽然没有 NVLink，但 CUDA P2P microbenchmark 可达到约
`24.51 GiB/s`。

校正方案不是修改 vLLM 的 PD 实现，而是采用上游已经合入的官方
`NixlPushConnector`：由 Prefill 端对 Decode 注册的显存执行 `PUT/WRITE`。
在同一 16K 请求上，它把 `2254.5 MiB` 的传输从约 `5.34 s` 降到
`91.984 ms`，NIXL 日志吞吐从约 `422 MiB/s` 提升到 `24509.697 MiB/s`，
descriptor 数从 `72144` 降到 `1`。

因此：历史 pull 结果仍可用于复盘，但不再作为同机 PD/PAP 的公平性能基线；
后续同机基线固定使用官方 push connector，并设置
`UCX_PROTO_EMULATION_ENABLE=n` 使错误路径 fail closed。

## 2. 诊断证据链

### 2.1 硬件路径

- GPU1 与 GPU2 的 `nvidia-smi topo -m` 关系是 `NODE`：同一 NUMA node、跨 PCIe
  host bridge，不是 NVLink；
- 约 `2254.5 MiB` 的双向 CUDA P2P copy 分别约 `89.83 ms`，折合约
  `24.51 GiB/s`；
- 所以“本机只能传几百 MiB/s”这个解释不成立。

### 2.2 Pull/GET 退化

原 `NixlConnector` 实际别名到 pull connector，Decode 端发起 NIXL READ：

| 变体 | R1 KV | 时间 | 日志吞吐 | descriptors | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| V2 默认 pull | 2254.5 MiB | 约 5.34 s | 约 422 MiB/s | 72144 | 异常慢 |
| V1 cross-layer pull | 2254.5 MiB | 约 4.69 s | 约 480 MiB/s | 1 | 碎片减少，但数据面仍慢 |
| pull + 禁用 emulation | 同上 | 失败 | 不适用 | 不适用 | UCX 报 `No zero-copy protocol found for get into cuda from cuda` |

`UCX_PROTO_INFO=y` 明确显示大 payload 走
`software emulation | tcp/enx...`。以下尝试都没有改变这个结论：

- 显式 `UCX_TLS=cuda_ipc,cuda_copy,tcp`；
- 让两个进程同时可见 GPU1/GPU2；
- 把 UCX error handling 从 peer 改为 none；
- 把每层/每块描述符合并成一个 cross-layer descriptor。

NIXL wheel 自带的官方 backend test 进一步复现了方向不对称：同机 GPU-to-GPU
GET 选择 TCP emulation，而 PUT 可以选择 CUDA IPC zero-copy。由此可排除模型、HTTP
proxy、KV token 记账和 vLLM scheduler 是 `0.42 GiB/s` 的主因。

### 2.3 官方 Push/PUT 校正

上游 vLLM 已提供 `NixlPushConnector` 及配套 proxy。该设计让 Decode 先发布目标
buffer metadata，再由 Prefill 发起 push。16K、两轮、o256 的 clean diagnostic 为：

| 轮次 | KV 大小 | NIXL 时间 | NIXL 吞吐 | descriptors | TTFT | TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | 2254.5 MiB | 91.984 ms | 24509.697 MiB/s | 1 | 3158.449 ms | 26.679 ms |
| R2 | 22.5 MiB | 2.105 ms | 10688.836 MiB/s | 1 | 219.661 ms | 26.804 ms |

同一旧 pull 正式参考的 R1/R2 TTFT 为 `8483.474/267.273 ms`，TPOT 为
`25.101/25.183 ms`。Push 的主要收益符合预期地落在需要搬运约 2.2 GiB KV 的 R1
TTFT；V1 cross-layer model runner 的 TPOT 比旧 V2 runner 高约 6%–7%，这是另一个
执行器变量，不能归因给 KV 传输。

本次采用的上游依据：

- [vLLM NixlConnector 使用指南](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)
- [vLLM NIXL push connector 设计](https://github.com/vllm-project/vllm/blob/main/docs/design/nixl_kv_push_connector.md)
- [vLLM PR #35264：NIXL-based Push KV Connector](https://github.com/vllm-project/vllm/pull/35264)
- [NVIDIA Dynamo NIXL](https://github.com/ai-dynamo/nixl)

## 3. 固定的 5 轮长上下文 testbed

### 3.1 Workload contract

| 项目 | 固定值 |
| --- | --- |
| 模型 | 本地 Qwen3-8B，float16，TP1 |
| 初始文档 | 16000 tokens |
| 后续每轮新增语料 | 120 tokens |
| 每轮输出 | 256 tokens，`ignore_eos=true` |
| 对话轮数 | 5 |
| 并发档 | C1 控制、C2 canary、C4 主实验 |
| 到达 | 每轮固定 2 request/s；轮内并发，轮间 barrier |
| 历史 | 下一轮包含真实的上一轮 assistant 输出 |
| 上限 | `max_model_len=20000`、`max_num_batched_tokens=4096`、`max_num_seqs=4` |
| GPU | 只使用 GPU1/GPU2 |

C4 最后一轮预计驻留约 `71552` 个 KV tokens，Qwen3-8B FP16 按约
`144 KiB/token` 估算约 `9.83 GiB`，低于实测 PAP PA KV 容量的 60%。C2 先作为
OOM、调度和证据链 canary；任何 OOM、EngineDead、请求失败或实际并发不足都会使结果
失效，不能用降低统计口径掩盖。

### 3.2 两侧固定实现

- PD：官方 `NixlPushConnector`、V1 model runner、cross-layer blocks、
  `UCX_PROTO_EMULATION_ENABLE=n`；
- PAP：1PA1P、same-node `local_fast + cuda_ipc`、MPS 70/30、固定 slot-plan 与
  metadata fast key；不做 MPS 扫描；
- 两侧：相同 prompt 构造、到达时刻、长度、GPU 型号、模型配置和完成 token 数；
- 主要指标：各轮与稳态 R2–R5 的 TTFT/TPOT median、p90、max；同时保留 latency、
  EOF delay 和实际 HTTP/decode concurrency。

### 3.3 正确性与 provenance

客户端保留 request-level token digest、prompt/completion tokens、实际到达/首 token/
末 token/EOF 时间及 prefix reuse transition。外部 finalizer 只有在以下证据同时通过时
才把 repetition 标记为可比较：

- 所有请求完成 256 tokens；
- prompt 形状和各轮请求数完整；
- PD 的 P/D cache-source 守恒、push 次数和 descriptor 计数匹配；
- PAP session drain、routing、decode commit/lease 与 Attention runtime stats 通过；
- 服务日志没有 OOM、EngineDead、Traceback 或 NIXL failure；
- Git commit、tracked dirty state、effective config 和 artifact hash 可追溯。

由于 PD 与 PAP 从 R2 起可能生成不同的 assistant tokens，比较器要求每轮 prompt token
数量形状严格相同，并把 digest 差异显式列为 warning；若 tokenizer 后长度发生分叉，
该组不能被静默宣称为 exact same-workload。

## 4. 标准运行入口

先做 C2 单次 canary：

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/run_pd_pap_multiturn_load.sh quick c2
```

再做 C1 控制和 C4 主实验：

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/run_pd_pap_multiturn_load.sh quick c1
bash .claude/skills/vllm-pap-benchmark/scripts/run_pd_pap_multiturn_load.sh quick c4
```

正式结果每侧三次、要求 tracked clean，并按
`PD, PAP, PAP, PD, PD, PAP` 交错执行以减小时间漂移：

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/run_pd_pap_multiturn_load.sh formal c4
```

每个 group root 内固定生成 `pd_aggregate.json`、`pap_aggregate.json`、
`comparison.json`、`report.md` 和 `testbed.env`；raw service logs、metrics、Git patch
及每个 request 的结果保留在各 repetition 子目录。

## 5. 仍需回答的问题

1. 在 C4 下，PD push 的 Prefill compute、KV push 与 Decode 是否形成新的排队瓶颈；
2. PAP 的 R1 TTFT 是否因 PA 上 70% MPS 的长 prefill 明显慢于独占 GPU 的 PD Prefill；
3. R2–R5 TPOT 的约 `1.2x` 差距在多会话 cohort 下会扩大还是由更大 Projection batch
   抵消；
4. V1 cross-layer runner 相对旧 V2 的约 6%–7% TPOT 代价能否在不破坏单 descriptor
   CUDA IPC push 的前提下消除。
