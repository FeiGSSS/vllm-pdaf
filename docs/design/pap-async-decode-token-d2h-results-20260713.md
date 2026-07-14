# PAP decode-token D2H 异步化实现与 C4 A/B

日期：2026-07-13

状态：`accepted-development-baseline`；TPOT 优化成立，TTFT 副作用已在后续工作关闭根因

> **2026-07-14 根因校正：** 本文第 6 节关于 sideband control-plane contention 的
> 初始假设已被后续 causal A/B 排除。进程级正交 A/B 证明，TTFT 回归由 Projection
> barrier 控制，而不是 Prefill 自身 barrier；最终 Torch trace 证明 Prefill kernel 并未
> 显著变慢，而是 Attention registry 全局锁把 Decode GPU 数据面与 Prefill 逐层 descriptor
> 串行化，造成约 `2.81 s` host-unsubmitted gap。更多小 batch 会提高 Decode append 进入
> 临界区的频率，但不是独立的 GPU 算力/HBM 根因。registry 锁分离与安全异步 Prefill
> import 已修复该问题。完整证据见
> [PAP async decode-token TTFT 回归根因](pap-async-decode-token-ttft-root-cause-20260714.md)。

## 2026-07-14 基线化决策

后续 static/dynamic MPS 2×2 A/B 证明 TTFT 退化不依赖 dynamic SM 分配；同时
async decode-token 与 static MPS 都能稳定改善 TPOT。基于“后续优化优先使用更高
decode throughput 的实现，再单独解决 TTFT”的取舍，决定默认启用：

```text
PAP_ASYNC_DECODE_TOKEN=1
PAP_LOAD_MPS_PROFILE=baseline_static_64_28
```

同步 barrier 路径和 dynamic 70:30 都继续作为显式 A/B 回退。该决策把实现接受为
后续开发基线，不把单次 dirty-worktree 2×2 结果冒充为 formal-clean 性能冻结结果。
完整矩阵见
[`pap-async-static-mps-c4-ab-results-20260714.md`](pap-async-static-mps-c4-ab-results-20260714.md)。

## 1. 结论

Projection 原路径会在每个 decode forward 的 token boundary，把下一步
`input_ids` 从 GPU 同步读回 CPU，再把 token ID 填入逐层 Attention descriptor。
这个 D2H 会让 Projection 的主线程等待 CUDA stream，因而位于下一步 forward 的
关键路径上。

本轮实现复用了 vLLM 已有的 `AsyncOutput` output-copy stream：sampled token 的 GPU→CPU
copy 完成后，通过 callback 把 token 异步投递给 Attention；Projection 不再为了构造
下一步 descriptor 同步读取 `input_ids`。Attention 以
`(session_request_id, new_seq_len)` 将 token-ready 与 KV-ready 合并，再沿用现有可靠
decode-commit 通知 Prefill。

在固定 C4 70:30 workload 上，两轮同代码 OFF/ON quick A/B 的平均结果为：

| 指标 | 同步 OFF | 异步 ON | 变化 |
| --- | ---: | ---: | ---: |
| R1 TTFT median | 11068.208 ms | 12042.239 ms | +974.032 ms / +8.80% |
| R1 TPOT median | 39.294 ms | 36.476 ms | **-2.818 ms / -7.17%** |
| R2–R5 TTFT median | 245.260 ms | 340.698 ms | +95.438 ms / +38.91% |
| R2–R5 TPOT median | 51.029 ms | 50.045 ms | **-0.984 ms / -1.93%** |
| R2–R5 TPOT p90 | 52.648 ms | 52.439 ms | -0.210 ms / -0.40% |

异步 ON 的两轮稳态 TPOT 分别为 `50.113 ms` 和 `49.977 ms`；同步 OFF 的两轮分别为
`51.047 ms` 和 `51.011 ms`。方向和幅度均可重复，因此可以确认同步 D2H 对 C4 steady
TPOT 的实际关键路径贡献约为 **0.98 ms/token**，而不是按 transformer layer 数重复
累加的数毫秒。

相对冻结的 PD-twoway trace-off steady TPOT `42.179 ms`：

- 同步 PAP：`1.210x PD`；
- 异步 PAP：`1.186x PD`；
- 本轮移除了约 `11.1%` 的 PAP–PD TPOT 剩余差距。

ON 的 TTFT 明显退化。2026-07-13 的初始判断因此只接受“D2H 异步化能降低 TPOT”
这一归因；2026-07-14 在 static MPS 排除 dynamic SM 分配主因后，用户接受这一
TTFT/TPOT 取舍并将其升级为后续开发基线。

## 2. 固定 workload

- 模型：本地 Qwen3-8B，FP16，TP1；
- 硬件：同机 NVIDIA L20 × 2，Attention/Prefill 使用 GPU 1，Projection 使用 GPU 2；
- PAP MPS：Prefill/Attention `70:30`；
- 架构：1PA1P，local-fast + Prefill-owned unified KV；
- first-turn document：16000 tokens；
- 轮数：5；后四轮各追加 120 tokens；
- 每轮输出：256 tokens，`ignore_eos=true`；
- 活跃 conversation：4，`max_num_seqs=4`；
- 每轮请求速率：2.0 request/s，fixed-rate round-barrier closed-loop；
- trace：两侧均关闭；
- 指标：`last_output_token_v2`，steady 为 R2–R5 共 16 个 request 样本/次。

唯一主变量是 `PAP_ASYNC_DECODE_TOKEN=0/1`。实验运行于同一 dirty tracked patch，属于
受控 A/B，不属于 `formal-clean`。

## 3. 实现

### 3.1 Projection

- `PAP_ASYNC_DECODE_TOKEN=1` 时，`prepare_inputs` 不再执行 descriptor token ID 所需的
  同步 `input_ids` D2H；
- `AsyncOutput` 在已有 CUDA output-copy event 完成、sampled token 完成 trim 后调用一次
  callback；
- callback 按 Attention endpoint 合并同一 forward 的请求，并把
  `request_id/new_seq_len/token_id` 放入后台可靠队列；
- `PAP_ASYNC_DECODE_TOKEN=0` 保留旧同步路径，作为同代码回滚和 A/B 开关。

### 3.2 Attention 与 Prefill

- Attention 新增单条和批量 decode-token endpoint；
- token-ready 与 unified-KV append 的 KV-ready 记录按 canonical session identity join；
- join 完成后仍使用原有 `DecodeCommitClient` 的有序、可重试、ACK 路径；
- release 时先 flush token/KV join，再 flush decode commit，然后释放 lease；
- 最后一个 sampled token 没有对应 KV，session release 时按设计丢弃；
- testbed 将 token/KV mismatch、pending、dispatch failure、非空 drain 设为 fatal gate。

高并发 canary 还发现并修复了两个实现问题：

1. 最初每个 request-token 单独 POST，C4 产生 5120 次请求并抵消 D2H 收益；现在按
   forward/endpoint batch，第二轮 ON 实际为 1858 次批量 POST，平均 2.76 tokens/POST；
2. decode-commit 与 lease-release 在 Prefill 共用顺序锁，原 `0.2 s` release timeout 会在
   C4 误判失败；release client 默认 timeout 调整为 `5.0 s`，严格 routing gate 恢复
   20/20。

## 4. 正确性证据

C2 canary：

- 目录：
  `test/baseline/pap/results/runs/20260713_async_d2h_batch_c2_canary`；
- 10/10 requests、8/8 transitions；
- 每个多轮 transition 的 256 个 decode-derived tokens 均命中；
- steady TTFT/TPOT median：`254.653/38.013 ms`；
- client、cache、routing、decode-token join、session drain 全部 passed。

两个最终 C4 ON run 均满足：

- 20/20 requests、16/16 transitions；
- `decode_token_received=5120`、`decode_token_matched=5100`；
- 20 个最后 token 按设计 token-only drop；
- pending token/KV、mismatch、dispatch failure、active session 均为 0；
- cache、digest、routing、lease release、join、drain 全部 passed。

相关单元/contract 回归：`233 passed`。完整 `test_pap_launch_files.py` 仍有两个与本改动无关
的历史 stale expectation（`--dtype` 数量和旧 `REPETITIONS=3` 文本），未在本轮扩展处理。

## 5. A/B 原始目录

| 配置 | steady TTFT median | steady TPOT median | steady TPOT p90 | strict gate | 目录 |
| --- | ---: | ---: | ---: | --- | --- |
| OFF rep1 | 242.297 ms | 51.047 ms | 52.613 ms | passed | `test/baseline/pap/results/runs/20260713_async_d2h_ab_c4_off_rep1` |
| OFF rep2 | 248.222 ms | 51.011 ms | 52.684 ms | passed | `test/baseline/pap/results/runs/20260713_async_d2h_batch_c4_off_rep2` |
| ON rep1 | 338.437 ms | 50.113 ms | 52.373 ms | passed | `test/baseline/pap/results/runs/20260713_async_d2h_batch_c4_on_rep1` |
| ON rep2 | 342.958 ms | 49.977 ms | 52.505 ms | passed | `test/baseline/pap/results/runs/20260713_async_d2h_batch_c4_on_rep2` |

每个目录都包含 `aggregate.json`、`effective_config.env`、
`decode_token_join_audit.env`、`routing_audit.json`、`session_drain.env`、Attention stats 和
完整 service logs。

## 6. TTFT 退化与下一步

以下内容保留 2026-07-13 当时的初始假设，已被 2026-07-14 的 sync-only barrier 和
Projection single-batch A/B **推翻**，不再代表当前结论。

异步路径虽然移除了 Projection 主线程的 CUDA barrier，但新增了一条 CPU control plane：
Projection 后台 worker → Attention batch HTTP endpoint → token/KV join → 原 decode-commit
worker。它不会阻塞 Projection 的下一个 forward，却会和 Attention/Prefill API、调度与
长 Prefill 共享 CPU/event loop。C4 中新增的 1858 次 sideband HTTP 请求与 ON 的
R1/steady TTFT 退化同时稳定复现，因此当前最合理的判断是 control-plane contention，
而不是 GPU D2H 本身变慢。

下一轮优先级应是：

1. 把 sampled-token sideband 从逐 forward HTTP 改为长连接 mailbox/ring，或进一步按
   时间窗批量，消除 endpoint/event-loop 放大；
2. unified-KV 当前每层都会观察同一 `(request, seq_len)` readiness，committer 虽会去重，
   但 C4 仍记录大量 duplicate；应改为整次 forward 仅发布一次 all-layer-ready；
3. 在以上 control-plane 开销消除后，再做 clean、三次交错 OFF/ON，决定是否正式默认
   启用。

当前状态为：**TPOT 路线与端到端实现均接受为开发基线；TTFT 退化继续作为下一阶段
优化目标。formal-clean 北极星结果仍需在提交后的干净工作树上重新生成。**
