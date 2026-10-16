# R-004: 阶段 3：机制、消融与配置调优

status: done
priority: P1
claims: C17, C18, C19, G1
opened: 2026-10-15
from: research agent
common settings: `paper/handoff/requests/COMMON.md`（本请求的所有运行都遵守）

depends: R-001 done（不依赖 G1，可以和 R-003 交错；它使用 R-003 的 s240 数据集，所以在该数据集注册之后开始）

## Hypothesis

- A2（C17）：Projection 跨 PA 聚合是吞吐的来源之一：限制 Projection batch 上限，或者把 Projection 拆到更多卡上，吞吐会下降。
- A5/A6（C19）：在相同负载下，"Dynamo 路由 + KV 迁移"的 TTFT 低于"只用 Dynamo 路由"和"粘性 RR"。
- A4（C18）：从 Prefill 卡切出 12 个 SM 做 Attention，代价不大：和 84/8 相比，80/12 的 TTFT 增加不超过 10%。
- E5（G1）：TTFT 尾部可以拆解成可识别的阶段（排队、路由、迁移、Prefill、首个 decode step）。
- O1–O3：只改环境变量就能改善 PAP 的 TTFT。

## Why it matters for the paper

A2 和 A4 分别支撑"跨 PA 聚合 Projection"和"Attention 切片几乎免费"两条论点；A5/A6 支撑多轮放置的主张，也是最接近 AMPD/PPD 的对照；E5 回答 G1 中的 TTFT 尾部问题。

## Expected result and falsification

每个小节的判据都在下面。O1–O3 中如果有一档更好，只写建议，不重跑 R-003；改不改默认值由 research agent 和用户决定。

## Suggested setup

### A2 Projection 跨 PA 聚合的贡献

对应论点：Projection 无状态，所以能把 7 个 PA 的 decode 合成一个大 batch。s240，C30 和 C50 各 2 次，迁移开。

| 变量 | 取值 |
| --- | --- |
| `PAP_PROJECTION_MAX_NUM_SEQS` | 256（基线），64，32 |
| 拓扑（同时改 GPU 列表） | 7PA1P（基线），6PA2P，4PA4P |

共 (2 + 2) × 2 并发 × 2 次 = 16 次，加上和基线的交错。另外在 C30 7PA1P 跑 1 次开 trace：设 `PAP_PROJECTION_PA_TRACE_OUTPUT=<run_root>/projection_trace.pt`，用 `tooling/merge_detailed_projection_trace.py` 和 `capture_aligned_trace.py` 合并，得到 Projection 单步时间随 batch 大小的曲线。

**产出**：`e2e/PAP-<date>-PROJECTION-AGGREGATION/`。
**判据**：给出"batch 上限减半 → ITL / 吞吐变化多少"，以及每个拓扑的吞吐和 ITL。

### A5/A6 多轮放置三方对比

对应论点：KV 固定在 PA 之后，"迁移 + 负载感知路由"比"粘性"和"不迁移"都好。这也是和 AMPD、PPD 这类"会话绑定到 KV 持有者"方案最接近的对照。

| 变体 | 怎么跑 |
| --- | --- |
| `dynamo+migration` | 当前代码，`PAP_ENABLE_PA_KV_MIGRATION=1` |
| `dynamo-only` | 当前代码，`PAP_ENABLE_PA_KV_MIGRATION=0` |
| `sticky-rr` | 用 `c565ecb9` 的 worktree，按 `CLEANUP-REGRESSION` 的 `base` 做法：`git worktree add .worktrees/rr-base c565ecb904`，共用 `.venv` 里的修复版 PAT（不在该 worktree 里重新编译），用 ABI 10 的 bridge 前缀 `.local/nvshmem-3.3.24-cuda13-bridge-v10`，`PYTHONPATH=<worktree>`；路由用 `PAP_ROUTING_POLICY=session_round_robin`，设置 `PAP_SESSION_ROUTER_MAX_CONVERSATIONS=<会话数>`，并且必须设 `PAP_ENABLE_PA_KV_MIGRATION=0`（旧代码中迁移要求 Dynamo 路由，见该版本 `run_pap_workload.sh:396`）。参数名以该 worktree 里的 `run_pap_workload.sh` 和 `PAP-20260909-SESSION-ROUND-ROBIN` 的记录为准 |

**问题**：PAT 是共享 `.venv` 里的扩展。`sticky-rr` 用的是旧代码 + 修复版 PAT，可以共用同一个安装，只要 PAT 的 Python 接口没有变。冒烟时确认一次；如果不兼容，就三个变体都改用 `PAP_ATTENTION_KERNEL_POLICY=triton`，在 README 写明。

**矩阵**：s240，C30 和 C50，3 变体 × 3 次，按变体轮流，共 18 次，约 2 机时。
**产出**：`e2e/PAP-<date>-PLACEMENT-3WAY/`，包括每轮 TTFT、每个 PA 的负载不均（每个 PA 的请求数和 Prefill token 数的变异系数）、迁移次数和迁移字节数、重算的 prefill token 数。
**判据**：所有运行有效。`sticky-rr` 来自旧代码，README 里要写明"非单变量：代码版本不同，只有路由是有意改变的"。

### A4 Attention 切片的代价

对应论点：从 Prefill 卡切出 12 个 SM 做 Attention 几乎是免费的。s60，C30，3 次。

| Prefill/Attention chunk | SM |
| --- | --- |
| 21/2 | 84/8 |
| 20/3（基线） | 80/12 |
| 19/4 | 76/16 |
| 18/5 | 72/20 |

每组修改 `PAP_STATIC_PREFILL_CHUNKS`、`PAP_STATIC_ATTENTION_CHUNKS` 和两个 `EXPECTED_SMS`。共 12 次，约 0.7 机时。另外从 R-003 E1 的 DP8 数据里取同负载下的 Prefill 吞吐（92 SM 满卡），换算成"每 SM 的 prefill token/s"，和 PAP 80 SM 的值对比。

**产出**：`e2e/PAP-<date>-MPS-SLICE/`，每组的 TTFT/ITL/吞吐，以及"Attention 每多给 4 个 SM，TTFT 和 ITL 各变多少"。

### E5 TTFT 尾部分解

在 R-003 E1 中 PAP 的 C30 和拐点各跑 2 次，设置 `PAP_PREFILL_IPC_PROFILE=1` 和 `PAP_DYNAMO_ROUTING_TRACE_OUTPUT=<run_root>/routing_trace.jsonl`。在实验目录里写 `analyze.py`，解析 Gateway 日志中的三类行：

- `PAP proxy prefill IPC profile request_id=… register_ms tokenization_ms routing_ms migration_ms prefill_payload_ms prefill_ms readiness_ms projection_payload_ms pre_projection_ms`（`vllm/pap/gateway/request_pipeline.py:374`）
- `PAP proxy projection admission profile … admission_ms request_to_admission_ms`（同文件 :409）
- `PAP proxy projection stream profile … first_chunk_ms request_to_first_chunk_ms`（`gateway/handoff.py:100`）

再用 `PAP Dynamo placement request_id=… selected_pa=…`（`gateway/dynamo_routing.py:278`）关联请求被放到哪个 PA，与 AIPerf 的逐请求 TTFT 按 request_id 对齐。

**profile 开关本身可能有开销**：同时用 `PAP_PREFILL_IPC_PROFILE=0` 跑 2 次作对照。开销超过 3% 时在 README 写明。

**产出**：`e2e/PAP-<date>-TTFT-BREAKDOWN/`，包括 P50/P99 请求的 TTFT 分段堆叠表、每个 PA 的排队长度和请求数，以及对比 DP8 的算力账（PAP Prefill 共 7 × 80 = 560 SM，DP8 共 8 × 92 = 736 SM）。共 6 次，约 0.6 机时。

### 优化方向

#### 配置调优（P1，约 2 机时，可以直接做）

目标是缩小 PAP 相对 DP8 的 TTFT 差距（9 月 PAP 4.50 s，DP8 2.21 s），并改善 TTFT P99。每档 3 次，s240，C30 和 R-003 E1 的 PAP 拐点。

| ID | 变量 | 取值 | 预期（推断） |
| --- | --- | --- | --- |
| O1 | `PAP_PREFILL_MAX_NUM_BATCHED_TOKENS` | 2048（基线），4096，8192 | 更大的 chunk 让 Prefill 的 GEMM 效率更高，TTFT 下降；代价是同卡 Attention 受干扰，ITL 上升。research 分支只测过 2048 和 4096 各 1 次 |
| O2 | `PAP_DYNAMO_PREFILL_LOAD_SCALE` | 1.0，2.0（基线），4.0 | 路由更看重 Prefill 负载，可能降低 PA 之间的不均衡和 TTFT 尾部 |
| O3 | `PAP_MIGRATION_MAX_CONVERSATIONS` | 会话数（基线），会话数/2 | 检查迁移表上限会不会成为瓶颈，预计没有影响 |

一次只改一个变量。如果某档在 D2 goodput 上比基线好且 CI 不包含 0，就在回报里**建议**把它作为新默认值，但已经跑完的 R-003 不重跑。是否改默认值由我们决定。

**产出**：`e2e/PAP-<date>-CONFIG-TUNING/`。

#### 需要改代码的方向（不要做，只在报告里评估可行性和预计收益）

| 方向 | 动机 | 和论文的关系 |
| --- | --- | --- |
| 动态 MPS / Green Context 切分 | semi-PD、Bullet、MuxWise 都批评静态切分；E5 和 A4 会显示静态 80/12 损失多少 | 放进讨论和 future work |
| "PA-local Projection" 变体（每个 PA 自带 Projection，不跨 PA 聚合） | 直接隔离"跨 PA 聚合"本身的贡献 | 比 A2 更干净的消融；只有在 A2 结论不清楚时才考虑 |
| AMPD 式 PD 基线（会话绑定到持有 KV 的 D，增量 prefill 在 D 上本地执行） | `paper/research/novelty-assessment.md` §5 列为冲更高级别会议时的必做项 | Euro-Par 预算内不做，用 §6.2 的 `sticky-rr` 近似 |
| 多轮更深的负载（≥ 6 轮） | 真实 agentic 会话远超 3 轮；当前 native-32K 只能放 3–4 轮 | 可以只加数据集、不改代码；先评估 PA 显存够不够 |

## Deliverables

- `paper/handoff/reports/R-004.md`：每个小节一段结论，外加对"需要改代码的方向"的可行性评估（不实施）。
- `paper/figures/`：Projection 单步时间 vs batch、放置三方对比的每轮 TTFT、MPS 切分表、TTFT 分段堆叠表。
- 更新 `claims.md` 中 C17–C19 和 G1。
