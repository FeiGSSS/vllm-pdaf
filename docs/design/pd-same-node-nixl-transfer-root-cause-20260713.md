# 同机 PD/NIXL KV 传输根因与校正基线

日期：2026-07-13

状态：根因已定位并二次校正。旧结论用 Push 绕开 GET；新证据证明 UCX 1.22 已修复
官方 `NixlConnector` 的同机 GET 路径，因此默认基线已升级为同一 connector 下的单向、
双向 PD。新三路结果见
[PD 单向/双向与 PAP 五轮结果](pd-oneway-twoway-pap-five-turn-results-20260713.md)。

## 1. 结论

原 1P1D 基线约 `0.42 GiB/s` 不是 PCIe 上限，而是 UCX 1.21 在无 NVLink 的 L20
拓扑上把 CUDA IPC 的直接 GET 判为不可用，随后用 TCP/Active Message 在 CPU 侧模拟
GET。`nvidia_peermem` 与该问题无关：它服务 GPUDirect RDMA/verbs，本机路径使用
CUDA IPC，不经过 RDMA 网卡。

第一阶段用官方 `NixlPushConnector` 把方向改成 PUT，证明硬件和 CUDA IPC 正常；第二
阶段升级到 UCX 1.22。UCX 1.22 的 RMA rendezvous 允许“请求方发起 GET，数据拥有方用
CUDA IPC zero-copy WRITE 回填”，所以官方 `NixlConnector` 不再需要换成 Push connector。
双向 A/B 中，UCX 1.21 默认约 `500 MiB/s`，UCX 1.22 分别达到 D→P
`5957.944 MiB/s`、P→D `22205.968 MiB/s`，输出 digest 完全一致。

当前默认固定为仓库内 UCX `1.22.0` + NIXL `1.3.0`，两条 PD lane 都使用官方
`NixlConnector`；`UCX_PROTO_EMULATION_ENABLE=n` 继续 fail closed。旧 Push 结果保留
为诊断里程碑，但不再是最终统一基线。

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

### 2.4 UCX 1.22 GET 修复与严格 A/B

后续 A/B 保持模型、NIXL、请求和双向 connector 配置不变，只替换 UCX。每组均设置
`UCX_TLS=cuda_ipc,cuda_copy,tcp`；严格组额外设置
`UCX_PROTO_EMULATION_ENABLE=n`，若 GPU 数据面不能原生执行就直接失败。

| 运行时 | 方向 | KV | 时间 | 吞吐 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| UCX 1.21 默认 | D→P | 38.25 MiB | 76.490 ms | 500.065 MiB/s | TCP software emulation |
| UCX 1.21 默认 | P→D | 2277 MiB | 4488.042 ms | 507.348 MiB/s | TCP software emulation |
| UCX 1.22 strict | D→P | 38.25 MiB | 6.420 ms | 5957.944 MiB/s | CUDA IPC rendezvous |
| UCX 1.22 strict | P→D | 2277 MiB | 102.540 ms | 22205.968 MiB/s | CUDA IPC rendezvous |

UCX 1.21/1.22 的 R1/R2 output digest 完全一致。UCX 1.21 强制
`UCX_CUDA_IPC_ENABLE_GET_ZCOPY=y` 也曾达到 `23601 MiB/s`，但它覆盖了无 NVLink
拓扑下的自动策略，因此只保留为诊断对照，不作为默认。UCX 1.22 协议日志明确包含
`rendezvous data send from cuda/GPU0 to cuda/dev[0]` 和
`zero-copy flushed write to remote | cuda_ipc/cuda`。

仓库内默认运行时位于 `.local/`（不进入 Git），版本和构建方式由以下脚本固定：

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/setup_ucx122_nixl.sh install
bash .claude/skills/vllm-pap-benchmark/scripts/setup_ucx122_nixl.sh verify
```

UCX 必须显式使用 `--enable-mt`。漏掉该项时 NIXL 会报
`UCX library does not support multi-threading` 并拒绝创建 engine；验证脚本会同时检查
版本、multi-thread 配置、plugin 动态链接目标和真实 NIXL agent 实例化。

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

### 3.2 三条固定实现

- PD-oneway：官方 `NixlConnector`、`bidirectional_kv_xfer=false`；五轮均由 P 重新补齐
  Decode history，proxy 必须全 MISS；
- PD-twoway：同一 `NixlConnector`、`bidirectional_kv_xfer=true`、
  `kv_recompute_threshold=0`、`decoder_kv_blocks_ttl=480`；首轮 MISS，后四轮由 D→P
  拉取 materialized history；
- 两条 PD：同一 repo-local UCX 1.22/NIXL 1.3、V1 model runner、cross-layer blocks、
  `UCX_PROTO_EMULATION_ENABLE=n`；
- PAP：1PA1P、same-node `local_fast + cuda_ipc`、MPS 70/30、固定 slot-plan 与
  metadata fast key；不做 MPS 扫描；
- 三侧：相同 prompt 构造、到达时刻、长度、GPU 型号、模型配置和完成 token 数；
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

正式结果每侧三次、要求 tracked clean，并按三阶拉丁方交错执行以减小时间漂移：

```text
PD-oneway, PD-twoway, PAP
PD-twoway, PAP, PD-oneway
PAP, PD-oneway, PD-twoway
```

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/run_pd_pap_multiturn_load.sh formal c4
```

每个 group root 内固定生成 `pd_oneway_aggregate.json`、
`pd_twoway_aggregate.json`、`pap_aggregate.json`、
`comparison.json`、`report.md` 和 `testbed.env`；raw service logs、metrics、Git patch
及每个 request 的结果保留在各 repetition 子目录。

## 5. 历史 Push 矩阵

本节保留第一阶段 `NixlPushConnector` 校正结果，证明硬件路径并建立过渡基线。它没有
被删除，但已由 UCX 1.22 下同一 `NixlConnector` 的单向/双向三路 test bed 替代；当前
结果和结论以
[新三路报告](pd-oneway-twoway-pap-five-turn-results-20260713.md)为准。

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
