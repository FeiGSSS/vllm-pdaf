# PD Push 校正与 PAP 五轮长上下文性能报告

日期：2026-07-13

代码基线：`feature/pap @ a646ae032`

## 1. 执行摘要

原 PD 基线约 `0.42 GiB/s` 的 KV 传输不是 GPU1/GPU2 的物理上限，也不是
NCCL 的正常表现。两张卡之间没有 NVLink，拓扑为同 NUMA、跨 PCIe bridge 的
`NODE`；但同尺寸 CUDA P2P copy 仍能达到约 `24.51 GiB/s`。异常来自旧
`NixlConnector` 的 pull 路径：GPU-to-GPU `READ/GET` 被 UCX 降级为 TCP software
emulation。

校正后的 PD 不修改 vLLM 源码，直接采用 2026-06-12 已合入上游的官方
`NixlPushConnector`。Prefill 对 Decode 预注册显存执行 `WRITE/PUT`，同机走
CUDA IPC；同时启用 cross-layer blocks，并用 `UCX_PROTO_EMULATION_ENABLE=n`
禁止静默回退。16K 单请求的 2254.5 MiB KV 传输由约 5.34 秒降到 91.984 ms，
日志吞吐由约 422 MiB/s 提升到 24509.697 MiB/s，约提升 58 倍。

在校正后的 PD 基线上，五轮、16K、C4 正式矩阵的核心结果是：

- 首轮 PAP median TTFT 为 PD 的 `1.365x`，median TPOT 为 `1.106x`；
- R2–R5 稳态 PAP median TTFT 为 PD 的 `0.811x`，即快 `18.9%`；
- R2–R5 稳态 PAP median TPOT 为 PD 的 `1.220x`；
- PD/PAP 各 3 次、每次 20 个请求全部完成，实际 HTTP/decode peak concurrency
  均为 4，没有 OOM、EngineDead、NIXL failure 或 session 泄漏。

因此，原来“PAP 首轮 TTFT 比 PD 更短”的主要原因已经确认是旧 PD pull 基线失真。
修复后，PAP 的真实优势出现在多轮 append prefill：PA 同时保留 prompt 和 decode KV，
而单向 PD 的 Prefill 节点必须重算上一轮由 Decode 生成的 tokens，再把增量 KV push
给 Decode。当前 PAP 的主要剩余缺口仍是稳态 TPOT，约比 PD 高 `22%`。

## 2. 根因与校正路径

### 2.1 为什么几百 MiB/s 不合理

| 证据 | 结果 | 含义 |
| --- | ---: | --- |
| `nvidia-smi topo -m` | GPU1↔GPU2 为 `NODE` | 没有 NVLink，但支持同机 PCIe P2P |
| CUDA P2P，约 2254.5 MiB | 约 89.83 ms / 24.51 GiB/s | 物理数据面不是 0.42 GiB/s |
| 旧 V2 pull | 约 5.34 s / 422 MiB/s / 72144 descriptors | UCX TCP emulation + 高碎片 |
| V1 cross-layer pull | 约 4.69 s / 480 MiB/s / 1 descriptor | 碎片消失，但 GET 数据面仍错误 |
| pull，禁用 emulation | `No zero-copy protocol found for get into cuda from cuda` | 当前 UCX 组合没有可用的同机 GPU GET 零拷贝协议 |
| 官方 push | 91.984 ms / 24509.697 MiB/s / 1 descriptor | PUT/WRITE 命中 CUDA IPC |

`UCX_PROTO_INFO=y` 和 NIXL backend test 都证明了方向不对称：GET 选择 TCP
software emulation，PUT 可以选择 CUDA IPC zero-copy。显式设置
`UCX_TLS=cuda_ipc,cuda_copy,tcp`、扩大 CUDA 可见卡集合或只减少 descriptors，都不能
修复 GET；因此不再继续用环境变量美化旧 pull 路径。

### 2.2 为什么采用官方 Push

上游设计明确区分两种模式：默认 connector 由 Decode 发起 NIXL READ；
`NixlPushConnector` 则由 Prefill 对 Decode 预分配显存发起 NIXL WRITE。对应 PR
#35264 已合入 vLLM 主线，并报告 push 相对 pull 的 TTFT 改善。本项目只改变 PD testbed
的 connector/config/proxy，不维护一份私有 PD connector patch。

固定校正配置为：

```text
kv_connector=NixlPushConnector
VLLM_USE_V2_MODEL_RUNNER=0
enable_cross_layers_blocks=True
UCX_TLS=cuda_ipc,cuda_copy,tcp
UCX_PROTO_EMULATION_ENABLE=n
kv_load_failure_policy=fail
```

上游参考：

- [vLLM NixlConnector 使用指南](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)
- [vLLM NIXL push connector 设计](https://github.com/vllm-project/vllm/blob/main/docs/design/nixl_kv_push_connector.md)
- [vLLM PR #35264](https://github.com/vllm-project/vllm/pull/35264)
- [NVIDIA Dynamo NIXL](https://github.com/ai-dynamo/nixl)

## 3. 固定五轮 Testbed

| 项目 | 固定值 |
| --- | --- |
| 模型 | 本地 Qwen3-8B，FP16，TP1 |
| GPU | 只使用 NVIDIA L20 GPU1/GPU2 |
| API/历史 | `/v1/completions`，exact token continuation |
| 第一轮 prompt | 16000 document tokens；实际 prompt 16013 tokens |
| 后续每轮 | 上轮完整 prompt + 256 output token IDs + 120-token suffix 及固定 marker |
| 输出 | 每轮 256 tokens，`ignore_eos=true` |
| 轮数 | 5 |
| 主负载 | 4 个 active conversations；每轮 2 request/s；轮间 barrier |
| 容量保护 | `max_model_len=20000`、`max_num_batched_tokens=4096`、`max_num_seqs=4` |
| PD | 1P1D，官方 push、V1 cross-layer、emulation fail closed |
| PAP | 1PA1P，`local_fast + cuda_ipc`，MPS 70/30，不扫描 MPS |
| 正式顺序 | `PD, PAP, PAP, PD, PD, PAP`，每架构 3 次完整重启 |

C4 每个 repetition 为 20 个请求。实测 HTTP peak 和 decode peak 都为 4；
time-weighted HTTP concurrency 为 PD `3.424–3.425`、PAP `3.483–3.486`，
decode concurrency 为两侧约 `3.11`。这不是把 QPS 参数调大但实际仍串行的伪并发。

## 4. 正式 C4 性能矩阵

正式原始目录：

```text
/home/fei/research/PD/vllm-pap/test/baseline/pap/results/runs/
  20260713_031215_a646ae032_pd_pap_load_c4_formal/
```

三次重复合并统计如下：

| Scope | 指标 | 统计量 | PD | PAP | PAP/PD |
| --- | --- | --- | ---: | ---: | ---: |
| R1 | TTFT | median | 8140.702 ms | 11108.313 ms | 1.365x |
| R1 | TTFT | p90 | 10509.560 ms | 16003.486 ms | 1.523x |
| R1 | TPOT | median | 35.456 ms | 39.218 ms | 1.106x |
| R1 | TPOT | p90 | 36.463 ms | 41.223 ms | 1.131x |
| R2–R5 | TTFT | median | 306.166 ms | 248.321 ms | 0.811x |
| R2–R5 | TTFT | p90 | 359.538 ms | 290.247 ms | 0.807x |
| R2–R5 | TPOT | median | 42.115 ms | 51.375 ms | 1.220x |
| R2–R5 | TPOT | p90 | 43.128 ms | 52.693 ms | 1.222x |
| R2–R5 | request latency | median | 11020.574 ms | 13299.389 ms | 1.207x |

重复间稳定性：

| 架构/重复 | R1 TTFT median | R1 TPOT median | R2–R5 TTFT median | R2–R5 TPOT median |
| --- | ---: | ---: | ---: | ---: |
| PD 1 | 8114.963 | 35.483 | 305.163 | 42.106 |
| PD 2 | 8183.174 | 35.427 | 305.304 | 42.101 |
| PD 3 | 8158.825 | 35.481 | 306.438 | 42.120 |
| PAP 1 | 11108.669 | 38.991 | 247.857 | 51.243 |
| PAP 2 | 11138.561 | 39.259 | 250.654 | 51.401 |
| PAP 3 | 11096.542 | 39.018 | 248.449 | 51.235 |

### 4.1 并发下的 PD Push 证据

每次 PD repetition 都精确记录：

- 20 次成功 push、9360 MiB、24 个 descriptors；
- 0 failed transfers、0 failed notifications、0 expired requests；
- Decode-derived cache hit 为 `4096` tokens，即 16 个轮次转换各命中 256 tokens；
- 官方累计指标的 aggregate throughput 分别为 `4137.1 / 4109.5 / 4095.6 MiB/s`。

该累计值低于单流 24.5 GiB/s，原因不是重新回到 TCP：C4 首轮 4 个请求会形成两个
并发窗口，每个请求约 2252.25 MiB，单个传输完成时间被 PCIe/注册/调度竞争拉到约
407–718 ms。后续 20.25–22.5 MiB 增量传输的日志吞吐仍为约
`12.4–21.9 GiB/s`。所以当前应把两种数字分开使用：单流诊断证明 CUDA IPC 路径正确；
C4 正式值描述真实并发下的共享链路和队列开销。

## 5. 结果解释

### 5.1 首轮 TTFT

校正后 PD 的 Prefill 节点可以使用一整张 GPU，而 PAP 的 PA 固定只获得 70% MPS，且
每层还要经过 Projection/Attention 分工。PD 的 2.2 GiB push 已缩短到百毫秒级，因此
原旧基线中由 5 秒 TCP GET 制造的 PAP TTFT 优势消失。C4 下 PAP 首轮 median TTFT
落后 `36.5%`，这是当前真实的首轮成本。

### 5.2 多轮 TTFT

PAP 在 PA 上保留第一轮 prompt 和全部已 materialize 的 decode KV；下一轮只计算新增
suffix 和未 materialize tail。单向 PD 虽然 Decode 节点保留自己的 decode KV，但
Prefill 节点没有上一轮 Decode 生成的 KV，所以 append prefill 仍需重算这些 token，
再把增量 KV push 到 Decode。C4 下 PAP 稳态 TTFT 因此比 PD 快约 `18.9%`，这正是
多轮 PAP ownership 的预期优势。

### 5.3 稳态 TPOT

PAP 稳态 TPOT 为 PD 的 `1.220x`，三次重复极稳定，说明当前差距不是偶发排队或
错误 PD 传输造成的。剩余约 `9.26 ms/token` 主要应继续从 PAP 每层
Projection→Attention QKV ready chain、跨进程调度以及 70/30 MPS 资源共享中归因；
不应再通过调慢 PD 基线来缩小比例。

## 6. 正确性边界

两侧均满足相同的请求数、每轮 prompt token 数、256 completion tokens、缓存转换和
并发形状，所有严格 Gate 通过。R1 的 PD/PAP output token digest 相同；R2 output 开始
分叉，因此 R3–R5 的 prompt 内容也随各自真实对话轨迹分叉。PAP 三次轨迹稳定；PD 在
一个 conversation 的 R4/R5 出现 batch-dependent deterministic tie 分叉。

因此本报告的后四轮结论是“相同形状、各自状态连续的真实多轮负载”性能比较，不宣称
R3–R5 是逐 token 相同输入。若后续要做严格的算子级跨架构 A/B，应新增一条
teacher-forced/shared-transcript lane，使两侧每轮都接收同一组预生成 token IDs；当前
warning 被保留，未被正确性审计静默忽略。

## 7. 产物与复现

设计、根因和标准入口：

- [同机 PD/NIXL 根因与校正基线](pd-same-node-nixl-transfer-root-cause-20260713.md)
- [PAP/PD 五轮正式报告](pd-pap-five-turn-load-results-20260713.md)
- [PAP 实验历史索引](pap-experiment-history-index.md)

正式运行：

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/run_pd_pap_multiturn_load.sh formal c4
```

每个 group root 固定生成 `pd_aggregate.json`、`pap_aggregate.json`、
`comparison.json`、`report.md` 和 `testbed.env`；每次 repetition 还保留 request-level
结果、服务日志、NIXL Prometheus 指标、effective config、Git patch/hash 和审计结果。

下一步优化以 C4 稳态 TPOT `PD 42.115 ms / PAP 51.375 ms` 为北极星，同时保留 C1
控制档，避免只优化一个并发点。
