# 同机 PD/NIXL KV 传输根因与校正基线

日期：2026-07-13

状态：根因已定位并校正；官方 push 路径、C1/C2 canary 和 C4 三次交错正式矩阵均已
完成。完整性能解释见
[PD Push 校正与 PAP 五轮长上下文性能报告](pd-pap-five-turn-load-results-20260713.md)。

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
| 历史 | exact token continuation；下一轮包含上一轮全部 output token IDs |
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

负载通过 `/v1/completions` 直接提交 token IDs，下一轮 prompt 固定为“上一轮完整
prompt + 全部 output + 新增 user suffix”。这样避免 assistant 文本 decode 后再 tokenize
破坏 BPE 边界；上一轮最后一个 sampled token 仍保留在新 prompt 中，但因为它没有 KV，
会作为 suffix 正常重算。客户端保留 request-level token digest、prompt/completion
tokens、实际到达/首 token/
末 token/EOF 时间及 prefix reuse transition。外部 finalizer 只有在以下证据同时通过时
才把 repetition 标记为可比较：

- 所有请求完成 256 tokens；
- prompt 形状和各轮请求数完整；
- PD 的 P/D cache-source 守恒、push 次数匹配，cross-layer descriptor 保持在由实际
  并发数决定的有界区间；
- PAP session drain、routing、decode commit/lease 与 Attention runtime stats 通过；
- 服务日志没有 OOM、EngineDead、Traceback 或 NIXL failure；
- Git commit、tracked dirty state、effective config 和 artifact hash 可追溯。

由于 PD 与 PAP 从 R2 起可能生成不同的 output token IDs，比较器要求每轮 prompt token
数量形状严格相同，并把 digest 差异显式列为 warning。固定输出和 suffix 长度保证形状
一致，但报告不会把 digest 已分叉的 R2–R5 描述为逐 token 相同的请求内容。

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

## 5. 已完成矩阵与当前结论

### 5.1 Quick 容量阶梯

代码基线均为 `a646ae032`，每个请求 16K 首轮、5 轮、每轮 o256：

| 并发 | R1 TTFT PAP/PD | R1 TPOT PAP/PD | R2–R5 TTFT PAP/PD | R2–R5 TPOT PAP/PD | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| C1 | 1.729x | 1.141x | 1.024x | 1.136x | 5/5 两侧完成，Gate 通过 |
| C2 | 1.453x | 1.150x | 0.865x | 1.161x | 10/10 两侧完成，Gate 通过 |
| C4 | 1.368x | 1.105x | 0.796x | 1.214x | 20/20 两侧完成，Gate 通过 |

原始目录：

```text
test/baseline/pap/results/runs/
  20260713_025857_a646ae032_pd_pap_load_c1_quick/
  20260713_025358_a646ae032_pd_pap_load_c2_quick/
  20260713_030234_a646ae032_pd_pap_load_c4_quick/
```

### 5.2 C4 formal

三次完整重启按 `PD, PAP, PAP, PD, PD, PAP` 交错执行。每侧共 60 个请求，实际
HTTP/decode peak concurrency 均为 4，所有 correctness、cache、routing、NIXL、session
drain 和 fatal-log Gate 通过，无 OOM。

| Scope | 指标 | PD median | PAP median | PAP/PD |
| --- | --- | ---: | ---: | ---: |
| R1 | TTFT | 8140.702 ms | 11108.313 ms | 1.365x |
| R1 | TPOT | 35.456 ms | 39.218 ms | 1.106x |
| R2–R5 | TTFT | 306.166 ms | 248.321 ms | 0.811x |
| R2–R5 | TPOT | 42.115 ms | 51.375 ms | 1.220x |

每次 PD repetition 都有 20 次成功 push、9360 MiB、24 descriptors、0 failure，官方
累计吞吐为 `4095.6–4137.1 MiB/s`。这比单流 `24.5 GiB/s` 低，是四个首轮 2.2 GiB
请求在两个到达窗口内竞争链路和注册/调度资源的真实并发结果；后续小增量仍达到约
`12.4–21.9 GiB/s`，没有回到 TCP emulation。

正式目录：

```text
test/baseline/pap/results/runs/
  20260713_031215_a646ae032_pd_pap_load_c4_formal/
```

当前结论是：校正 PD 后，PAP 不再具有旧 pull 基线制造的首轮 TTFT 优势；PAP 的真实
多轮收益是稳态 TTFT，因为 PA 原地保留 decode KV。PAP 稳态 TPOT 则稳定为 PD 的
约 `1.22x`，下一阶段应继续分析 Projection→Attention QKV ready chain 和 cohort/MPS
竞争，而不是继续调整 PD 传输配置。
