# COMMON: R-001 ~ R-004 共用设置

opened: 2026-10-15
from: research agent

本文件由 `claude/paper-experiment-plan-6llagm` 上的 Euro-Par 2027 runbook（`89e60acf`）迁移而来（回应 X-001），该分支和 runbook 已作废，以本目录为准。协议见 `paper/README.md`；与 runbook 冲突的地方一律以协议为准：

- 唯一交换分支是 `feature/pap-v026`。实验目录、报告、`claims.md` 状态都提交到这里；不再有 `results/europar27` 分支。
- 每个请求的报告写到 `paper/handoff/reports/R-NNN.md`（格式见 `TEMPLATES.md`）。runbook 里的"阶段回报"即对应请求的报告。
- 门控：G0 = R-001 判据触发时设 `status: blocked`；G1 = R-002 完成后，R-003 开始前要等 research agent 在 R-003 里写 `cleared:` 行（48 小时没有回复就按 R-002 报告里的默认分支继续，并在 R-003 报告里写明）。R-004 不依赖 G1，可以和 R-003 交错。
- 背景文档都在 `paper/research/`：`experiment-plan.md`（旧计划，§3.3 和 §8.5）、`novelty-assessment.md`、`outline-v2.md`、`positioning-and-venue.md`。作图脚本和输入格式在 `paper/figures/`。

## 0. 执行规则

1. 实验代码、脚本、数据集都在 exp agent 的写权限内（协议"everything outside `paper/`"）。但本轮请求**不需要**改 `vllm/`、`third_party/`、`benchmarks/pap/scripts/`、`benchmarks/pap/tooling/`。如果确实要改，在报告里说明原因和 diff。
2. 新实验目录放在 `benchmarks/pap/experiments/{e2e,microbench}/PAP-2026MMDD-<NAME>/`，遵守 `benchmarks/pap/experiments/README.md` 的约定。新数据集只能新增 ID，不能改已有 ID。
3. 每次运行之前先提交该实验目录的 `experiment.env`/`run.sh`，然后用 `PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE=1` 跑。
4. 同一张图/表里比较的架构，必须在同一天按交错顺序（ABAB…）跑，不复用 9 月归档。原因：同样的代码，今天比 9 月慢 6–8%（`PAP-20261015-CLEANUP-REGRESSION`）。
5. 失败的运行保留目录，在 README 里记一行原因，不计入统计，补跑一次。

## 1. 背景：自 9 月以来的代码变化

### 1.1 新提交（`c565ecb9..e39d8058`）

| 提交 | 内容 | 对实验的影响 |
| --- | --- | --- |
| `f61f1859` | 删除 7 个无人引用的分析/数据集脚本：`analyze_decode_load_confound`、`analyze_gateway_phase_trace`、`analyze_projection_pa_trace`、`chat_prefix_cache_diagnostic`、`prefix_cache_diagnostic`、`plot_qps_matrix`、`build_synthetic_longctx_dataset` | E5 的 TTFT 分解需要在实验目录里写新的解析脚本。旧 `analyze_gateway_phase_trace.py` 的正则已经和当前日志不匹配（当前日志在 `routing_ms` 后多了 `migration_ms`，见 `vllm/pap/gateway/request_pipeline.py:374`），不能直接恢复使用 |
| `d908915d` | 精简运行时：只保留 Dynamo PA 路由（删 session round-robin 和 `--routing-policy`，`PAP_ROUTING_POLICY≠dynamo` 直接报错）；删 Gateway load tracker（少一次整 prompt 的 block-hash）；每个 PA 只支持 1 个 Attention endpoint；删 GPU-resident dispatch、device graph launch；NVSHMEM bridge ABI 升到 11；配置解析统一 | ① 旧计划 E2c 第 1 步和 A6 的"粘性 RR 路由"在当前代码上**无法运行**，只能用 `c565ecb9` 的 worktree 做历史对照（R-004）。② **每台机器都要用 `build_nvshmem_device_bridge.sh` 重新编译 bridge**，否则加载时被拒。③ TP>1 和多 Attention rank 已明确不在计划内，写进局限性 |
| `e39d8058` | `PAP-20261015-CLEANUP-REGRESSION`：C30、7PA1P、迁移关闭，清理后/清理前各 2 次交错运行 | 清理本身无可测影响（TTFT −1.94%，95% CI −6.99%～+3.76%；ITL +0.88%，CI −1.03%～+2.86%）。但两个版本的 TTFT 都比 9 月参考值慢 6–8%，期间驱动从 595.84 升到 595.99 |

### 1.2 PAT 修复（PR #1）已于 10-15 合入

- PR #1（`ae727afc`）已按用户决定合入 `feature/pap-v026`，合并提交 `9b814d13`。它只改了 `third_party/pat/csrc/prefix_tree.h` 和一个测试，Python 侧没有改动。**合入后必须重新编译 PAT**（R-001），否则 `.venv` 里装的仍是旧扩展。
- **PAT 在当前主路径上确实被用到。** `PAP-20261015-CLEANUP-REGRESSION` 的 `current_r1` 归档里，7 个 PA 的 `attention_fast_path_stats_<i>.json` 显示 `attention_kernel_pat_rebuilds` = 16/38/29/27/32/28/25，`triton_selections` = 18/12/16/17/20/14/18，其余约 1100–1290 步复用上一次的计划。也就是说，9 月以来所有 `auto` 策略下的 PAP 数字都可能受这个 bug 影响。所以 E0 仍然是 P0。

### 1.3 旧计划条目的处理（相对 `paper/research/experiment-plan.md` §8.5）

| 旧条目 | 现在的处理 | 原因 |
| --- | --- | --- |
| E0 PAT 修复 + 主表重跑 + Triton 对照 | **保留，P0**；修复已合入，未修复版从 `e39d8058` 编译作对照（R-001） | PAT 确实在用 |
| E2a PD 基线审计 | **缩小**。DP8/PD 已开 prefix caching 和 chunked prefill（`run_dynamo_workload.sh:512–572`），decode 的 CUDA Graph capture 已到 128（`ALIGNED-C30/results/common.env`）。只剩调参（§4.2） | 审计项一半已经能从代码确认 |
| E2e PD KV 传输开销 | **保留，P0**。现成工具：`tooling/analyze_dynamo_ttft.py` 能把 TTFT 拆成路由/Prefill/handoff/Decode 并汇总 NIXL 传输；`tooling/nixl_read_write_probe.py` 可以直接做点对点微基准（§4.1） | 无需写代码 |
| E2b 重放 L07 | **保留，P0**，补上数据集的生成方法（R-002） | L07 原文件带 `cache_salt`，而当前 PAP 的 Dynamo 集成会拒绝显式 salt，必须先派生无 salt 版本 |
| E1 简化扫描 | **扩大**：机时比原来估计便宜得多（见 1.4），改成 4 架构 × 6 并发 × 3 次，用 240 会话的新数据集（R-003） | |
| E5 TTFT 尾部分解 | **保留**，换成当前日志格式的新解析器（R-004） | 旧解析脚本已删除且不匹配 |
| E2c 运行时归因阶梯 | **删除** | 粘性路由已删，eager/local_fast 早已不存在，阶梯走不通；论文里写成"运行时整体替换" |
| A6 路由对比 | **改成**三方对比，粘性 RR 用 `c565ecb9` worktree（R-004，P1） | 这是多轮放置主张的直接证据 |
| 新增 E2d-lite 负载区间 2×2 | **新增，P0.5** | 新颖性主张是"什么时候划算"，必须有区间图（R-003） |
| 新增 A2 Projection 聚合、A4 Attention 切片成本 | **新增，P1** | 分别对应"跨 PA 聚合 Projection"和"Attention 切片几乎免费"两条论点（R-004） |
| 新增 O 系列配置调优 | **新增，P1** | 只改环境变量，不改代码（R-004） |

### 1.4 成本校准

`CLEANUP-REGRESSION` 的 4 次有效运行，从建目录到下一次运行开始约 3 分钟。其中服务启动约 1.7 分钟，AIPerf 180 个请求约 65 秒，清理约 0.5 分钟。下文按以下公式估算机时（推断）：

> 单次 ≈ 2.5 min（启动 + 审计 + 清理）+ 请求数 / 吞吐

旧计划按每次 12 分钟估算，偏高约 3–4 倍。下面的总量在 38 机时以内，同时做了更多实验。

---


## 2. 前置检查 X0（R-001 的第一步）

```bash
# 1. 驱动与 GPU
nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id --format=csv > /tmp/x0_gpus.csv
#   预期：8×L20，驱动 595.99（如果不同，在回报里写明，后续所有数字都以这个驱动为准）
# 2. 重新编译 NVSHMEM bridge（ABI 11）
bash benchmarks/pap/scripts/build_nvshmem_device_bridge.sh
# 3. PAT 身份
.venv/bin/python -c "import importlib.metadata as m; print(m.version('prefix_attn'))"
find .venv -name '*prefix_attn*.so' -exec sha256sum {} \;
# 4. 数据集校验
(cd benchmarks/pap/datasets && sha256sum --check SHA256SUMS)
# 5. 单元测试
.venv/bin/python -m pytest tests/pap -q            # 预期 309 passed
# 6. 冒烟：用 COMMON.md"启动模板"跑一次 PAP 7PA1P C30（迁移开），和一次 Dynamo 6P2D C30
```

**通过判据**：bridge 编译成功；pytest 309 passed；两次冒烟都是 180/180 请求，所有审计 `STATUS=passed`。PAP 冒烟的 TTFT 均值在 4.2–6.2 s 之间（9 月迁移开为 4.50 s，今天的代码慢 6–8%，再留出余量）。
**结果写到**：`experiments/e2e/PAP-<date>-X0-SMOKE/`。


## 3. 固定参数

| 项 | 值 |
| --- | --- |
| 模型 | Qwen3-8B FP16，`MAX_MODEL_LEN=32768`，`PAP_HF_OVERRIDES=''` / `DYNAMO_HF_OVERRIDES=''`（9 月 r5/r6 就是因为漏了这一项作废的） |
| PAP 默认 | 7PA1P，Dynamo 路由，**KV 迁移开**（`PAP_ENABLE_PA_KV_MIGRATION=1`，`PAP_KV_MIGRATION_EXPORT_MODE=snapshot`），MPS 20/3（80/12 SM），Prefill budget 2048，Projection budget 256 / max_num_seqs 256 |
| Dynamo 基线 | 完全沿用 `PAP-20260907-ALIGNED-C30/results/common.env`（Prefill 2048、Decode 256、`DYNAMO_ROUTER_PREFILL_LOAD_SCALE=2.0`、CUDA Graph capture 到 128、v2 model runner） |
| PAT | E0 之后所有实验都用**修复版 PAT**（R-001） |
| 重复 | 每点 3 次独立启动；E0 和主图拐点 ±1 档为 5 次 |
| 客户端 | AIPerf 闭环并发，`timing_mode=concurrency` |

## 4. 启动模板

**PAP**：`experiments/e2e/<ID>/requested.env` 写入下面的内容，按实验改其中标 `#VAR` 的行。

```bash
MODEL_PATH=/data/ssd1/llm-models/Qwen3-8B
PAP_TOPOLOGY=7pa1p                       #VAR
PAP_PREFILL_GPUS=0,1,2,3,4,5,6           #VAR 与拓扑一致
PAP_PROJECTION_GPUS=7                    #VAR
PAP_ROUTING_POLICY=dynamo
PAP_ENABLE_PA_KV_MIGRATION=1             #VAR
PAP_KV_MIGRATION_EXPORT_MODE=snapshot
PAP_MIGRATION_MAX_CONVERSATIONS=60       #VAR = 会话数
PAP_ATTENTION_KERNEL_POLICY=auto         #VAR auto|triton
PAP_STATIC_PREFILL_CHUNKS=20             #VAR 与下一行之和必须是 23
PAP_STATIC_ATTENTION_CHUNKS=3            #VAR
PAP_STATIC_PREFILL_EXPECTED_SMS=80       #VAR = chunks×4
PAP_STATIC_ATTENTION_EXPECTED_SMS=12     #VAR = chunks×4
PAP_VLLM_DTYPE=float16
MAX_MODEL_LEN=32768
PAP_HF_OVERRIDES=''
PAP_PREFILL_MAX_NUM_BATCHED_TOKENS=2048  #VAR
PAP_PREFILL_MAX_NUM_SEQS=256
PAP_PROJECTION_MAX_NUM_BATCHED_TOKENS=256
PAP_PROJECTION_MAX_NUM_SEQS=256          #VAR
PAP_PREFILL_ASYNC_SCHEDULING=off
PAP_PROJECTION_ASYNC_SCHEDULING=on
PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.90
PAP_BLOCK_SIZE=16
PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS=256
PAP_DYNAMO_PREFILL_LOAD_SCALE=2.0        #VAR
PAP_AIPERF_INPUT_FILE=<绝对路径>/dataset.jsonl   #VAR
PAP_AIPERF_SESSIONS=60                   #VAR
PAP_AIPERF_EXPECTED_REQUESTS=180         #VAR = 会话数×轮数
PAP_AIPERF_CONCURRENCY=30                #VAR 单个整数；不要用逗号列表
PAP_AIPERF_TIMING_MODE=concurrency
PAP_AIPERF_CUSTOM_DATASET_TYPE=mooncake-trace   #VAR L07 系列用 multi-turn
PAP_PREFILL_IPC_PROFILE=0                #VAR E5 用 1
PAP_RUNTIME_CUDA_CONTEXT_AUDIT=1
BENCH_TIMEOUT=3600
PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE=1
PAP_BENCH_STRICT_CORRECTNESS_AUDIT=1
```

`run.sh` 照抄 `experiments/e2e/PAP-20261015-CLEANUP-REGRESSION/run.sh` 的 `current` 分支，即 `env -i` + `launch_env` + `source requested.env` + `exec run_pap_workload.sh`。改两处：参数改成 `<variant> <repetition>`，每个 variant 对应 `variants/<variant>.env`，在 `requested.env` 之后 source 进去覆盖；去掉 `base` 分支（§6.2 除外）。

> **不要用 `PAP_AIPERF_CONCURRENCY=10,20,30` 一次启动跑多个点**：同一个服务实例里，后一个点会命中前一个点留下的前缀缓存，结果偏乐观。正式数据每个点都单独启动一次。

**Dynamo（DP8 / xPyD）**：`requested.env` = `ALIGNED-C30/results/common.env` 全文，加上 `DYNAMO_ARCHITECTURE=<dp8|7p1d|6p2d|4p4d|2p6d>`，并按实验覆盖 `DYNAMO_AIPERF_INPUT_FILE / _INPUT_SHA256 / _SESSIONS / _CONCURRENCY / _EXPECTED_REQUESTS / _CUSTOM_DATASET_TYPE`。每次运行设置独立的 `DYNAMO_RUN_ID`、`DYNAMO_RUN_ROOT=<exp>/runs/<ts>_<variant>_r<k>`、`DYNAMO_AIPERF_OUTPUT_DIR=$DYNAMO_RUN_ROOT/aiperf`、`DYNAMO_NAMESPACE`、`DYNAMO_FILE_KV=$DYNAMO_RUN_ROOT/discovery`，然后 `bash benchmarks/pap/scripts/run_dynamo_workload.sh`。

**每次 PD 运行之后**都要额外做：

```bash
.venv/bin/python benchmarks/pap/tooling/analyze_dynamo_ttft.py "$DYNAMO_RUN_ROOT" \
  --block-size 16 --output "$DYNAMO_RUN_ROOT/ttft_breakdown.json"
```

这一步是 E2e 端到端部分的数据来源，不增加运行次数。

## 5. 指标、SLO 与统计（开跑前写进每个实验的 README）

- **逐请求表**（每个实验的 `results/requests.csv`）：列至少包括 `architecture,variant,concurrency,run,conversation_id,turn,input_tokens,output_tokens,cached_tokens,ttft_ms,itl_ms,e2e_ms,start_ns,end_ns`。前 10 列与论文作图脚本 `paper/figures/data/requests.csv` 的格式一致。可以以 `experiments/e2e/PAP-20260909-KV-MIGRATION-ARCHITECTURE-C30/analyze.py` 为模板。
- **ITL**：AIPerf 的请求级 `inter_token_latency`，口径与 9 月相同。
- **SLO（预登记）**：主 SLO 用 **D2**（TTFT ≤ 10 s、ITL ≤ 75 ms、E2E ≤ 20 s），敏感性分析用 D1、D3、D4（定义见 `ALIGNED-C30/GOODPUT.md:46–49`）。L07 系列另外用当年的 Strict/Standard/Relaxed：TTFT ≤ 5/10/20 s，ITL ≤ 50/75/100 ms，不限 E2E。
- **goodput**：达标请求数 ÷ 运行时长（第一个请求开始到最后一个请求结束）。达标率 ≥ 95% 的点算可行。**最大可持续 goodput** = 所有可行并发点里 goodput 的最大值。另外报告"稳态窗口"版本作为敏感性：去掉开始时间落在前 10% 时长内的请求，以及最后一个请求开始之后才结束的请求。
- **置信区间**：按运行分层、在请求层面做 bootstrap（10,000 次），给出 95% 百分位 CI。架构之间的比值，两边各自独立重采样。
- **每个点的 `summary.json`**：吞吐（req/s、output tok/s）、TTFT/ITL/E2E 的均值和 P50/P90/P95/P99、各 SLO 的 goodput 和达标率、CI，以及运行有效性清单。

---


## 6. 总量

| 阶段 | 实验 | 运行次数 | 机时（推断） |
| --- | --- | ---: | ---: |
| 0 | X0 前置检查 + 冒烟 | 2 | 1 |
| 0 | E0 PAT 修复 | 18 | 1.5 |
| 1 | E2e KV 传输（离线 + 微基准） | 微基准 | 1 |
| 1 | E2a 基线调参 | 21 | 1.5 |
| 1 | E2b L07 重放 | 约 56 | 3.5 |
| 2 | E1 负载扫描（含一致性检查和拐点补测） | 约 92 | 11 |
| 2 | E2d-lite 2×2 | 约 72 | 5 |
| 3 | A2 Projection 聚合 | 约 17 | 2 |
| 3 | A5/A6 放置三方对比 | 18 | 2 |
| 3 | A4 MPS 切片 | 12 | 0.7 |
| 3 | E5 TTFT 分解 | 6 | 0.6 |
| 3 | O1–O3 配置调优 | 约 24 | 2 |
| | **合计** | **约 340** | **约 32**（留约 6 机时给重跑和补测） |


## 7. 证据位置

- 清理回归：`benchmarks/pap/experiments/e2e/PAP-20261015-CLEANUP-REGRESSION/README.md`
- 9 月主表：`…/e2e/PAP-20260909-KV-MIGRATION-ARCHITECTURE-C30/report.md`；Dynamo 基线参数：`…/e2e/PAP-20260907-ALIGNED-C30/results/common.env`
- SLO D1–D4：`…/e2e/PAP-20260907-ALIGNED-C30/GOODPUT.md:46–49`
- L07：`feature/pap:benchmarks/pap/experiments/PAP-20260729-RESEARCH-L07/report.md`；生成参数：`feature/pap:benchmarks/pap/aiperf/run_capacity_matrix.sh:20–68`
- PAT 选择逻辑：`vllm/pap/attention/backend.py`（`PAPPATOrTritonSelector`）、`vllm/pap/attention/peers.py:163`；统计输出：`run_pap_workload.sh:998` 起
- PD 传输分析：`benchmarks/pap/tooling/analyze_dynamo_ttft.py`；微基准：`benchmarks/pap/tooling/nixl_read_write_probe.py`
- 运行时可调参数：`benchmarks/pap/scripts/run_pap_workload.sh:80–331`，`run_dynamo_workload.sh:9–60`
