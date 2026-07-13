# PD 单向/双向与 PAP 五轮长上下文结果

日期：2026-07-13

状态：C2 quick、C4 quick 与 C4 formal 均已完成。正式结果绑定 commit `03d8da336`，
三路各重复三次并通过三阶拉丁方、strict audit 与 tracked-clean Gate；本文件第 7 节的
C4 formal 结果已冻结为当前三路多轮北极星。

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

## 7. C4 formal 冻结结果

结果目录：

```text
/home/fei/research/PD/vllm-pap/test/baseline/pap/results/runs/
  20260713_131649_03d8da336_pd_three_lane_c4_formal/
```

该组绑定 Git commit `03d8da336dfe177d878372adb41a487ffe898dd7`。三阶拉丁方顺序为：

```text
PD-oneway -> PD-twoway -> PAP
PD-twoway -> PAP -> PD-oneway
PAP -> PD-oneway -> PD-twoway
```

每个 cell 都是服务完全重启后的独立 C4 五轮运行。三条 lane 各聚合 3 个 cell、60 个
请求；全部请求完成，失败数为 0。

### 7.1 正式聚合

| Scope | 指标 | PD-oneway | PD-twoway | PAP | two/one | PAP/one | PAP/two |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | TTFT | 8112.026 | 8128.513 | 11077.283 | 1.002x | 1.366x | 1.363x |
| R1 | TPOT | 35.516 | 35.477 | 39.121 | 0.999x | 1.102x | 1.103x |
| R2-R5 | TTFT | 280.867 | 251.716 | 249.030 | 0.896x | 0.887x | 0.989x |
| R2-R5 | TPOT | 42.176 | 42.155 | 51.148 | 1.000x | 1.213x | 1.213x |

表中为 pooled request-level median。双向 PD 把 steady TTFT 从 `280.867 ms` 降到
`251.716 ms`（`-10.4%`），TPOT 不变。PAP steady TTFT 为双向 PD 的 `0.989x`，已在
本负载下持平；PAP steady TPOT 为 `1.213x`，仍慢 `21.3%`。Round 1 没有 D->P 复用，
两条 PD lane 符合预期地相同；PAP Round 1 TTFT/TPOT 分别为双向 PD 的
`1.363x/1.103x`。

### 7.2 三次重复稳定性

| Lane | Cell | R1 TTFT | R1 TPOT | R2-R5 TTFT | R2-R5 TPOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| PD-oneway | 1 | 8082.570 | 35.531 | 278.611 | 42.174 |
| PD-oneway | 2 | 8137.437 | 35.505 | 279.775 | 42.174 |
| PD-oneway | 3 | 8141.741 | 35.514 | 282.534 | 42.184 |
| PD-twoway | 1 | 8109.318 | 35.487 | 239.868 | 42.175 |
| PD-twoway | 2 | 8149.667 | 35.464 | 253.758 | 42.155 |
| PD-twoway | 3 | 8125.989 | 35.493 | 237.500 | 42.155 |
| PAP | 1 | 11023.575 | 39.023 | 250.079 | 51.100 |
| PAP | 2 | 11107.234 | 39.155 | 254.649 | 51.341 |
| PAP | 3 | 11068.515 | 39.008 | 245.693 | 50.940 |

### 7.3 正确性、复用与传输 Gate

- 9 个 cell 均记录同一 commit、`git_tracked_worktree_dirty=false`，tracked index 与
  worktree patch 均为空；
- PD-oneway 三次均为 `20 MISS / 0 HIT / 0 send`、D->P 0 次、P->D 20 次；
- PD-twoway 三次均为 `4 MISS / 16 HIT / 16 send`、D->P 16 次、P->D 20 次；
- PD-oneway P->D 每次 9360 MiB，吞吐 `2303.817-2309.516 MiB/s`；
- PD-twoway D->P 每次 603 MiB，吞吐 `3473.062-4676.630 MiB/s`、257-258 个
  descriptors；P->D 每次 9360 MiB，吞吐 `2288.021-2309.044 MiB/s`；
- 所有 NIXL failed transfer、failed notification、expired request 均为 0，且
  `UCX_PROTO_EMULATION_ENABLE=n`；
- PAP 三次的 routing、Attention stats、correctness 和 session drain 均通过，最终
  `ACTIVE_SESSIONS=0`；
- 三条 lane 的 prompt shape/digest、output token digest 和 assistant text digest 全部
  一致，统一 `comparison.json` 为 `status=valid`、warnings 为空。

原始统一报告位于同目录的 `report.md`，机器可读结果为 `comparison.json`、
`pd_oneway_aggregate.json`、`pd_twoway_aggregate.json` 和 `pap_aggregate.json`。

为防止未跟踪的大体积原始目录被静默替换，冻结时记录核心 artifact SHA256：

| Artifact | SHA256 |
| --- | --- |
| `comparison.json` | `69a1fe051c53e4ac9b60b3bf21f436abbd2cec00f69d8159bf1f393b8456062e` |
| `pd_oneway_aggregate.json` | `7db536acd1229a6fb34922d3c3b1c73964878323c82304e0e27b0778a37ba789` |
| `pd_twoway_aggregate.json` | `7724350bab386dd7d7a5a6ad4c9b6bdf6ee44873176d234fccd891b375fe2443` |
| `pap_aggregate.json` | `6d7f360583e01d11c59d625630cd9f0248fdc84c9e85ad4918c80561be59c286` |
| `report.md` | `f95173835bd662016fd78cffa127b1d53e7ba3031812cd06014f902da235ec1f` |

## 8. 当前结论与证据边界

1. UCX 1.22 已让官方 `NixlConnector` 的同机双向 GET 可用，不是硬件缺少
   `nvidia_peermem`；
2. D→P 复用在 C2/C4 quick 和 C4 formal 中都降低后续轮 TTFT，formal 降幅为
   `10.4%`，没有增加 TPOT；
3. PAP 对单向 PD 的 steady TTFT 优势仍成立；与双向 PD 比较后，formal steady TTFT
   为 `0.989x`，已经持平，主要剩余差距是 TPOT（`1.213x`）；
4. Round 1 不发生 D→P，所以 PD-oneway/PD-twoway 应相同，实测也基本相同；PAP R1
   TTFT 仍明显更高，是下一阶段需要分析的 prefill/调度问题；
5. C2/C4 quick 仍只作为开发与方向证据；当前可发布北极星仅指 commit `03d8da336`
   上的 C4 formal 三次拉丁方结果。该结论只覆盖 1:1、C4、16K 五轮固定负载，不能
   外推为任意 x:y 或其他到达分布的性能结论。

## 9. 下一步

- 以本次 formal 结果作为后续优化判断基线：保持 steady TTFT 不差于 PD-twoway，
  把 PAP steady TPOT 从 `1.213x` 继续向 `1.0x` 收敛；
- 单独分析 PAP Round 1 TTFT `1.363x` 的 Prefill/调度瓶颈，避免以牺牲 steady TPOT
  的方式换取第一轮改善；
- 输出分叉诊断继续作为独立待办。本次 quick 的三个 digest 均一致，但不能替代更广泛
  的 batch-invariance/teacher-forced 正确性实验。
