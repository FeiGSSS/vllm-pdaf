# PAP 1PA1P 五轮长上下文阶段总结与汇报

日期：2026-07-13

分支：`feature/pap`

性能代码基线：`a646ae032`

阶段状态：完成 PD 同机传输基线校正，完成 1PA1P 五轮、16K、C4 正式性能矩阵；
将该结果冻结为下一阶段 TPOT 优化的北极星基线。

## 1. 阶段汇报结论

本阶段完成了 PAP 1PA1P 多轮 KV 复用、长上下文 testbed 和同硬件 PD/PAP 公平对比。
在 Qwen3-8B、两张 L20、16K 首轮、五轮对话、4 条并发 conversation 的正式测试中：

- PAP 在 R2–R5 的 median TTFT 为 `248.321 ms`，比 PD 的 `306.166 ms` 快
  `18.9%`；
- PAP 在 R2–R5 的 median TPOT 为 `51.375 ms`，是 PD `42.115 ms` 的
  `1.220x`；
- PAP 的稳态请求延迟为 PD 的 `1.207x`，输出吞吐为 PD 的 `0.823x`；
- PAP 首轮 median TTFT 为 `11108.313 ms`，是 PD 的 `1.365x`；首轮 TPOT 为
  PD 的 `1.106x`；
- PD/PAP 各完成 3 次完整重启、每次 20 个请求，两侧均为 `60/60` 成功，实际峰值
  并发为 4，没有 OOM、EngineDead、传输失败或 PAP session 泄漏。

结果证明了两件事：第一，PAP 将 Prefill 与 Decode KV 共同保留在 PA 上的设计已经产生
可测量的多轮 TTFT 收益；第二，当前整体差距几乎完全由 decode TPOT 累积造成。下一阶段
不需要再证明“能否进入 PD 的两倍以内”，而应把目标收紧为在同一 C4 北极星负载上，将
PAP 稳态 TPOT 从 `51.38 ms` 继续逼近 PD 的 `42.12 ms`。

## 2. 本阶段完成的工作

### 2.1 多轮 KV 复用闭环

PAP 已经能够让 PA 持有首轮 Prefill KV，并将后续 Decode K/V 直接写入同一套
Prefill-owned paged blocks。请求一轮结束后，PA/P pair 可以解除；第二轮只要被上层
cache-aware router 路由回原 PA，就可以通过 vLLM 原生 APC 命中首轮 Prefill 和 Decode
完整块，不需要将 Decode KV 回传到 Prefill。

当前 1PA1P testbed 已验证该机制。多 PA conversation affinity 暂不在本地 Proxy 中重复
实现，后续由 Dynamo 等缓存感知路由框架负责。

### 2.2 PAP TPOT 热路径优化

在 7 月 12 日两轮 C1 PAP 内部 A/B 中，已完成以下优化：

- paged FlashAttention metadata 由逐元素 CUDA 写改为 bulk build；
- slot plan 由 request 级永久 latch 改为 generation/topology-aware 复用；
- cache hit 的完整 block-table 扫描改为 topology-token fast key；
- 完整 block-ID 扫描量降低 `36x`；
- PAP R1 TPOT 由 `42.923 ms` 降到 `30.196 ms`，降低 `29.65%`；
- PAP R2 TPOT 由 `39.128 ms` 降到 `30.449 ms`，降低 `22.18%`。

这些 PAP 内部优化结论仍然有效。但当时引用的 PD pull 传输基线后来被证明失真，因此
旧报告中 PAP/PD 的首轮 TTFT 和端到端优势不再作为当前公平比较结论。

### 2.3 校正 PD/NIXL 同机基线

旧 `NixlConnector` pull 路径对 GPU-to-GPU `READ/GET` 使用了 TCP software
emulation，使约 2254.5 MiB KV 的传输只有约 `422 MiB/s`，耗时约 `5.34 s`。这不是
GPU1/GPU2 的 PCIe 物理上限，也不是可接受的官方 PD 表现。

校正后采用上游官方 `NixlPushConnector`、V1 cross-layer blocks，并设置
`UCX_PROTO_EMULATION_ENABLE=n` 让错误传输路径 fail closed。Prefill 对 Decode
预注册显存执行 `WRITE/PUT`，同机命中 CUDA IPC；同一传输缩短到 `91.984 ms`，达到
约 `24509.7 MiB/s`，相对旧路径提升约 58 倍。当前正式对比没有维护私有 PD 源码 patch。

### 2.4 五轮并发 testbed

对比从两轮 C1 升级为五轮 C1/C2/C4，其中 C4 为正式主负载。testbed 固化了 exact
token continuation、请求形状、计时边界、Git/config provenance、缓存账本、PD push
审计、PAP routing/session drain 和错误日志 Gate，可作为后续性能优化的固定回归入口。

## 3. 正式比较口径

| 项目 | 固定配置 |
| --- | --- |
| 模型 | 本地 Qwen3-8B，FP16，TP1 |
| 硬件 | NVIDIA L20 × 2，只使用 GPU1/GPU2 |
| 第一轮 | 16000 document tokens；实际 prompt 16013 tokens |
| 后续轮次 | 上轮 token history + 120-token suffix 和固定 marker |
| 输出 | 每轮 256 tokens，`ignore_eos=true` |
| 对话轮数 | 5 |
| 并发 | 4 个 active conversations；每轮内 2 request/s；轮间 barrier |
| 容量限制 | `max_model_len=20000`、`max_num_batched_tokens=4096`、`max_num_seqs=4` |
| PD | 1P1D，官方 NIXL push、V1 cross-layer、UCX emulation fail closed |
| PAP | 1PA1P，`local_fast + cuda_ipc`，MPS 70/30，不扫描 MPS |
| 重复 | `PD, PAP, PAP, PD, PD, PAP`，每架构 3 次完整重启 |

每次 repetition 包含 4 条 conversation、每条 5 个请求，共 20 个请求。实际 HTTP/decode
峰值并发均为 4，因此这不是仅增大 QPS 参数但服务仍串行的伪并发测试。

## 4. 正式性能矩阵

正式原始目录：

```text
/home/fei/research/PD/vllm-pap/test/baseline/pap/results/runs/
  20260713_031215_a646ae032_pd_pap_load_c4_formal/
```

| Scope | 指标 | PD | PAP | PAP/PD | 汇报结论 |
| --- | --- | ---: | ---: | ---: | --- |
| R1 | median TTFT | 8140.702 ms | 11108.313 ms | 1.365x | PAP 慢 36.5% |
| R1 | median TPOT | 35.456 ms | 39.218 ms | 1.106x | PAP 慢 10.6% |
| R2–R5 | median TTFT | 306.166 ms | 248.321 ms | 0.811x | PAP 快 18.9% |
| R2–R5 | median TPOT | 42.115 ms | 51.375 ms | 1.220x | PAP 慢 22.0% |
| R2–R5 | median request latency | 11020.574 ms | 13299.389 ms | 1.207x | PAP 慢 20.7% |
| 全程 | median output throughput | 72.167 token/s | 59.368 token/s | 0.823x | PAP 低 17.7% |

尾延迟与中位数结论一致：R2–R5 p90 TTFT 的 PAP/PD 为 `0.807x`，p90 TPOT 为
`1.222x`，p90 request latency 为 `1.211x`。三次重复的稳态 PAP TPOT 分别为
`51.243/51.401/51.235 ms`，结果稳定，不是单次噪声或偶发排队造成。

## 5. 性能归因

### 5.1 多轮 TTFT 优势来自 KV ownership

PAP 的 PA 同时保存 prompt KV 和已 materialize 的 Decode KV，后续 append prefill 只需
处理新增 suffix 和未 materialize tail。单向 PD 的 Decode 虽保存自己的 Decode KV，
但 Prefill 没有上一轮 Decode 生成的 KV，因此仍要在 Prefill 侧重算这些 tokens，再将
增量 KV push 给 Decode。C4 下 PAP 稳态 TTFT 因而比 PD 快 `57.845 ms`。

### 5.2 TPOT 是当前端到端主导项

稳态 TPOT 的绝对差距为：

```text
51.375 - 42.115 = 9.260 ms/token
```

256-token 输出包含约 255 个 token interval，累计增加约 `2361 ms`。扣除 PAP 的
`57.845 ms` TTFT 优势，预计请求延迟增加约 `2303 ms`；实测增加 `2279 ms`。二者高度
一致，因此当前端到端差距可以主要由 TPOT 累积解释，而不是多轮 Prefill 或 KV push。

### 5.3 并发放大 PAP decode 差距

探索性 C1/C2 与正式 C4 显示：

| Active conversations | 稳态 TTFT PAP/PD | 稳态 TPOT PAP/PD | 稳态 latency PAP/PD |
| ---: | ---: | ---: | ---: |
| C1 | 1.024x | 1.136x | 1.133x |
| C2 | 0.865x | 1.161x | 1.154x |
| C4 | 0.811x | 1.220x | 1.207x |

并发越高，PAP 的多轮缓存收益越明显，但 Projection→Attention decode 链路扩展更差。
后续优化应优先解释 C4 下新增的约 `9.26 ms/token`，重点关注逐层 QKV readiness、跨进程
控制与调度、P2P copy/compute overlap、cohort/batch 形成以及 70/30 MPS 共享导致的等待，
而不是通过降低 PD 传输性能或改变负载来缩小比值。

### 5.4 首轮 TTFT 是第二优先级

校正后的 PD Prefill 可以使用一整张 GPU，2.2 GiB push 已经缩短到百毫秒级；PAP 的 PA
固定为 70% MPS，并承担 split forward 的逐层协同。因此 C4 首轮 PAP median TTFT 比 PD
多约 `2.97 s`。该差距需要后续从 Prefill 计算资源、排队和 split pipeline 归因，但本阶段
继续遵守“不扫描 MPS”的约束。

## 6. 正确性与证据边界

两侧请求数、prompt token 数、每轮 256 completion tokens、缓存转换和并发形状一致，
所有现有 routing、session、transfer 和 lifecycle Gate 通过。R1 的 PD/PAP 输出相同；
R2 在相同 prompt 下开始出现跨架构输出分叉，随后真实对话历史继承该差异。PD 三次重复
中还出现过一次相同 R4 prompt 的内部轨迹分叉。

因此，本报告支持“相同 shape、各自状态连续的真实多轮 serving 性能比较”，但暂不宣称
PD/PAP 逐 token 数值等价，也不把 R3–R5 描述为完全相同内容。该问题已经登记为
`P0-CORRECTNESS-DIVERGENCE`，后续将通过 raw token/top-k margin、PD batch-invariant
A/B、shared-transcript cold/warm R2、QKV transport/KV slot checksum 和逐层数值对比
完成诊断与修复。

这个正确性待办不否定当前 shape-controlled 性能数据，但在关闭前，任何精度或 exact-token
等价结论都必须保持保守表述。

## 7. 阶段决策与下一步

1. 将 `20260713_031215_a646ae032_pd_pap_load_c4_formal` 冻结为当前 1PA1P 五轮并发
   性能北极星；
2. 废止旧 PD pull 作为公平 PAP/PD 基线，但保留历史 PAP 内部单变量 A/B 的优化证据；
3. 当前 PAP 已达到原目标 `TPOT < 2x PD`，并稳定达到 corrected-push PD 的 `1.22x`；
4. 下一阶段性能 P0 是在固定 MPS、固定 C4 workload 下压缩约 `9.26 ms/token` 的稳态
   TPOT 差距；
5. 性能 P1 是归因并降低首轮约 `2.97 s` 的 TTFT 差距，不通过 MPS 扫描寻找偶然最优点；
6. 正确性 P0 与性能优化并行管理，先定位首个不同 token 和首个数值分叉 layer；
7. arbitrary x:y 的连接与控制面、2PA2P combine/scatter 和 1PA1P 多轮 APC 已有独立
   证据；更高拓扑的 corrected-push 五轮性能矩阵不与本次 1PA1P 结论混报。

## 8. 可直接用于领导汇报的摘要

本阶段完成了 PAP 1PA1P 多轮 KV 复用、16K 五轮并发 testbed，以及同硬件 PD/PAP
公平性能校正。排查发现旧 PD pull 路径因 UCX 将 GPU GET 降级为 TCP emulation，KV
传输仅约 422 MiB/s；切换到上游官方 NIXL push 后，同一 2.2 GiB KV 传输达到约
24.5 GiB/s，提升约 58 倍。在校正后的两张 L20、Qwen3-8B、16K、五轮、C4 正式测试中，
PAP 后四轮 TTFT 比 PD 快 18.9%，证明 Prefill 与 Decode KV 共同保留在 PA 上的多轮
收益成立；PAP 稳态 TPOT 为 PD 的 1.220 倍，达到原定低于 2 倍的阶段目标。当前 PAP
整体请求延迟仍比 PD 高 20.7%，主要由每 token 约 9.26 ms 的 TPOT 差距累积造成。
下一阶段将以该固定 testbed 为北极星，重点优化 Projection→Attention decode 热路径，
同时独立完成 PD/PAP 输出分叉的数值正确性诊断。

## 9. 证据入口

- [PD Push 校正与 PAP 五轮长上下文性能报告](pd-pap-five-turn-load-results-20260713.md)
- [同机 PD/NIXL 传输根因与校正基线](pd-same-node-nixl-transfer-root-cause-20260713.md)
- [PAP 开发与实验历史索引](pap-experiment-history-index.md)
- [7 月 12 日 PAP 内部优化阶段记录](pap-1pa1p-multiturn-stage-report-20260712.md)

正式复现入口：

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/run_pd_pap_multiturn_load.sh formal c4
```
