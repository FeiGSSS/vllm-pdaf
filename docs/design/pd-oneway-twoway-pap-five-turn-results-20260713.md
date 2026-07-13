# PD 单向/双向与 PAP 五轮长上下文结果

日期：2026-07-13

状态：C2 quick 与 C4 quick 已完成；三路均通过严格 Gate。结果来自 tracked-dirty
开发状态，证据等级为 `controlled/quick`，不冒充 `formal-clean`。正式三次拉丁方矩阵
必须在本次代码提交、tracked worktree clean 后执行。

## 1. 这次解决了什么

同机 PD 的旧 `NixlConnector` 在 UCX 1.21 上执行 GET 时退化到 CPU/TCP software
emulation，只有约 500 MiB/s。第一阶段曾用 `NixlPushConnector` 绕过；本次改用仓库内
UCX 1.22.0 + NIXL 1.3.0，恢复官方 `NixlConnector` 的原生 same-node CUDA IPC GET，
并把默认 test bed 固定为：

1. `PD-oneway`：P→D，`bidirectional_kv_xfer=false`；
2. `PD-twoway`：P→D + D→P，`bidirectional_kv_xfer=true`；
3. `PAP`：1PA1P local-fast、Prefill-owned unified KV。

两条 PD lane 使用同一 connector、UCX/NIXL、GPU、模型和 runner，只改变双向开关及其
必需的 threshold/TTL。这样能直接回答“双向复用 Decode KV 是否改善后续轮 TTFT”，
不再混入 Push connector 的架构变量。

## 2. 固定 workload

| 项目 | 值 |
| --- | --- |
| 模型 | 本地 `/data/ssd1/llm-models/Qwen3-8B` |
| 精度/TP | FP16 / TP1 |
| GPU | GPU1 + GPU2；GPU0 未触碰 |
| 第一轮文档 | 16000 tokens |
| 后续新增正文 | 每轮 120 tokens |
| 输出 | 每轮 256 tokens，temperature 0、seed 0、ignore EOS |
| 轮数 | 5 |
| 并发 | C2 与 C4 |
| 到达 | 每轮 2 requests/s，轮内并发、轮间 barrier |
| 上限 | max model len 20000；batched tokens 4096；seqs 4 |
| PAP | MPS 70/30，不扫描 |

三路使用 exact token-ID continuation，下一轮 prompt 包含上一轮完整 prompt、全部 256
个 output token IDs 和新 suffix。比较器要求 prompt token shape 和 prompt digest 完全
一致；output/text digest 不同只作为 correctness warning。本次 C2、C4 三路的 prompt、
output、assistant text digest 均完全一致。

## 3. UCX 1.22 修复证据

### 3.1 单会话严格 A/B

| UCX | 方向 | KV | 时间 | 吞吐 |
| --- | --- | ---: | ---: | ---: |
| 1.21 默认 | D→P | 38.25 MiB | 76.490 ms | 500.065 MiB/s |
| 1.21 默认 | P→D | 2277 MiB | 4488.042 ms | 507.348 MiB/s |
| 1.22 strict | D→P | 38.25 MiB | 6.420 ms | 5957.944 MiB/s |
| 1.22 strict | P→D | 2277 MiB | 102.540 ms | 22205.968 MiB/s |

严格组设置 `UCX_PROTO_EMULATION_ENABLE=n`。协议日志证明大块 GPU 数据选择
`cuda_ipc/cuda` rendezvous，而不是 TCP 模拟。UCX 1.21/1.22 两轮输出 digest 一致。

### 3.2 固化方式

- `.local/ucx-1.22` 与 `.local/nixl-ucx122` 保存本机二进制，不进入 Git；
- 安装脚本固定 UCX 1.22.0、NIXL 1.3.0 和 `--enable-mt`；
- runner 启动前只读验证 UCX 版本、multi-thread、plugin 动态链接和 NIXL agent；
- 缺失或链接到错误 UCX 时 fail closed，不在性能运行期间下载或编译。

## 4. 实现过程中发现并修复的缺陷

| 缺陷 | 影响 | 修复与 Gate |
| --- | --- | --- |
| 五轮客户端只给 PAP 发送 `conversation_id` | PD 双向 proxy 无法关联轮次，只会全 MISS | PD/PAP 均发送会话 ID；单向仍因无 D handle 全 MISS，双向首轮 MISS 后 HIT |
| Chat stream 有 KV metadata，Completions stream 没有 | 当前 exact-token `/v1/completions` 负载无法把 D block handle 交给 proxy | Completion 终止 chunk 同样返回 `kv_transfer_params`，新增回归测试 |
| finalizer fresh validation 未重放 Prefill/Decode 日志 | stored/fresh evidence 输入不同，封存失败 | 原始 service logs 进入 artifact，并在 finalizer 中重新读取 |
| 双向命中近似写死为 256 token | 实际 D materialized history 可包含未对齐尾部，严格守恒误报 | 使用每个 transition 的 `materialized_history_tokens - previous_prompt_block_boundary` |
| descriptor 假设每次恰好 1 个 | C2/C4 并发时 GPU region 会分片 | 保留 transfer count 下界和保守有界上限，记录实际 descriptor，不再假定等于 1 |

这些 Gate 均先在真实运行中失败，再用单元测试固化后修复；没有通过放宽请求完成数、
忽略 transfer failure 或跳过 cache-source 守恒来获得结果。

## 5. C2 quick 结果

结果目录：

```text
/home/fei/research/PD/vllm-pap/test/baseline/pap/results/runs/
  20260713_112725_a634b5bf8_pd_three_lane_c2_quick/
```

该组在开发中途按失败 Gate 续跑，最终三条 lane 都由最终审计代码重新运行并通过；统一
`comparison.json` 为 `status=valid`。

| Scope | 指标 | PD-oneway | PD-twoway | PAP | two/one | PAP/one | PAP/two |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | TTFT | 4957.207 | 4947.230 | 7254.289 | 0.998x | 1.463x | 1.466x |
| R1 | TPOT | 31.491 | 31.508 | 36.009 | 1.001x | 1.143x | 1.143x |
| R2–R5 | TTFT | 223.907 | 197.052 | 210.789 | 0.880x | 0.941x | 1.070x |
| R2–R5 | TPOT | 33.102 | 33.106 | 38.538 | 1.000x | 1.164x | 1.164x |

双向把 steady TTFT 降低约 12%，TPOT 基本不变。PAP steady TTFT 比单向 PD 快约 5.9%，
但比双向 PD 慢约 7.0%；PAP TPOT 是两条 PD 的约 1.164 倍。

### C2 NIXL 证据

| Lane/方向 | 次数 | MiB | 时间 | 聚合吞吐 | descriptors |
| --- | ---: | ---: | ---: | ---: | ---: |
| one-way D→P | 0 | 0 | 0 | 0 | 0 |
| one-way P→D | 10 | 4680.0 | 1.374760 s | 3404.231 MiB/s | 11 |
| two-way D→P | 8 | 301.5 | 0.045911 s | 6567.054 MiB/s | 122 |
| two-way P→D | 10 | 4680.0 | 1.365427 s | 3427.499 MiB/s | 16 |

Proxy 证据：one-way `10 MISS / 0 HIT / 0 send`；two-way
`2 MISS / 8 HIT / 8 send`。所有 failed transfer、failed notification、expired request
计数均为 0。

## 6. C4 quick 结果

结果目录：

```text
/home/fei/research/PD/vllm-pap/test/baseline/pap/results/runs/
  20260713_114402_a634b5bf8_pd_three_lane_c4_quick/
```

该组由最终默认入口一次端到端完成，自动产生三组 aggregate、`comparison.json`、
`report.md` 和 `testbed.env`。

| Scope | 指标 | PD-oneway | PD-twoway | PAP | two/one | PAP/one | PAP/two |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | TTFT | 8088.446 | 8109.159 | 11147.940 | 1.003x | 1.378x | 1.375x |
| R1 | TPOT | 35.512 | 35.475 | 39.099 | 0.999x | 1.101x | 1.102x |
| R2–R5 | TTFT | 279.207 | 247.077 | 252.793 | 0.885x | 0.905x | 1.023x |
| R2–R5 | TPOT | 42.165 | 42.163 | 51.261 | 1.000x | 1.216x | 1.216x |

C4 下双向仍把 steady TTFT 降低约 11.5%，且 TPOT 与单向完全相当。PAP steady TTFT
比单向 PD 快约 9.5%，与双向 PD 接近（慢约 2.3%）；PAP steady TPOT 是两条 PD 的
约 1.216 倍。

### C4 NIXL 证据

| Lane/方向 | 次数 | MiB | 时间 | 聚合吞吐 | descriptors |
| --- | ---: | ---: | ---: | ---: | ---: |
| one-way D→P | 0 | 0 | 0 | 0 | 0 |
| one-way P→D | 20 | 9360.0 | 4.046578 s | 2313.066 MiB/s | 24 |
| two-way D→P | 16 | 603.0 | 0.192206 s | 3137.259 MiB/s | 258 |
| two-way P→D | 20 | 9360.0 | 4.084413 s | 2291.639 MiB/s | 34 |

Proxy 证据：one-way `20 MISS / 0 HIT / 0 send`；two-way
`4 MISS / 16 HIT / 16 send`。吞吐低于单流 A/B 是 C4 多请求竞争 GPU/PCIe、注册和调度
资源后的聚合服务指标，但仍显著高于 UCX 1.21 的约 500 MiB/s 软件模拟路径。

## 7. 当前结论与证据边界

1. UCX 1.22 已让官方 `NixlConnector` 的同机双向 GET 可用，不是硬件缺少
   `nvidia_peermem`；
2. D→P 复用在 C2/C4 均稳定降低后续轮 TTFT 约 12%，没有增加 TPOT；
3. PAP 对单向 PD 的 steady TTFT 优势仍成立；与双向 PD 比较后，C4 steady TTFT 已接近
   持平，PAP 的主要剩余差距是 TPOT（约 1.216 倍）；
4. Round 1 不发生 D→P，所以 PD-oneway/PD-twoway 应相同，实测也基本相同；PAP R1
   TTFT 仍明显更高，是下一阶段需要分析的 prefill/调度问题；
5. C2/C4 quick 都是 dirty controlled 结果。`UCX_PROTO_INFO=y` 用于保留数据面选择证据，
   因此不把它们升级为正式发布数字；formal-clean 仍是必要的冻结步骤。

## 8. 下一步

- 用户允许提交后，先提交当前实现，使 tracked worktree clean；
- 运行 `formal c4`，按三阶拉丁方对每条 lane 做三次完整重启；
- 以 formal 结果冻结新的北极星，并把 PAP 优化目标更新为：steady TTFT 不差于
  PD-twoway，steady TPOT 继续从约 `1.216x` 向 `1.0x` 收敛；
- 输出分叉诊断继续作为独立待办。本次 quick 的三个 digest 均一致，但不能替代更广泛
  的 batch-invariance/teacher-forced 正确性实验。
