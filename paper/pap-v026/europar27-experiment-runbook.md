# PAP 论文实验与优化执行手册（Euro-Par 2027）

日期：2026-10-15。基线代码：`feature/pap-v026` @ `e39d8058`（2026-10-15 清理后）。本分支 `claude/paper-experiment-plan-6llagm` 只在其上增加本文档，**不改任何运行时代码、脚本、配置或数据集**。

读者：在 8×L20 服务器上执行实验的 agent。读完本文档即可开跑；每个实验都给出命令、矩阵、产出位置和通过/失败判据。执行完一个阶段后，按 §9 的格式回报。

文中标"（推断）"的是根据代码和已有报告推出来的，没有用实验验证过。

---

## 0. 给执行 agent 的总规则

1. **拉取**：
   ```bash
   git fetch origin
   git checkout -B claude/paper-experiment-plan-6llagm origin/claude/paper-experiment-plan-6llagm
   ```
   本分支的运行时代码与 `feature/pap-v026@e39d8058` 字节相同（`git diff e39d8058 -- vllm benchmarks third_party tests` 应为空），以此为准记录 source identity。
2. **不许改的东西**（要改先回报，等人确认）：`vllm/`、`third_party/`、`benchmarks/pap/scripts/`、`benchmarks/pap/tooling/`、已注册的 `benchmarks/pap/datasets/*`、`tests/`。不许合并任何 PR，不许 push 到 `feature/*` 分支。
3. **可以新增的东西**（不需要再问）：
   - `benchmarks/pap/experiments/{e2e,microbench}/PAP-2026MMDD-<NAME>/` 下的新实验目录，遵守 `benchmarks/pap/experiments/README.md` 的约定（`README.md`、`experiment.env`/`requested.env`、薄 `run.sh`、`analyze.py`、`results/`；`runs/` 进 `.gitignore`）。
   - `benchmarks/pap/datasets/` 下**新的**数据集 ID（新目录 + manifest + 追加 `SHA256SUMS` 行），只能新增，不能改已有 ID。
   - 本地临时 git worktree（例如 E0 用来编译修复版 PAT），不 push。
4. **结果分支**：所有实验目录和回报提交到新分支 `results/europar27`（从本分支切出），push 到 origin。不要往本分支 push，以免和文档修订冲突。
5. **每次运行之前**提交该实验目录的 `experiment.env`/`run.sh`，然后用 `PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE=1` 跑（已跟踪文件必须干净；7 月 L06 因脏工作区作废过）。
6. **同期运行**：同一张图/表里比较的架构，必须在同一天按交错顺序跑（ABAB…），不复用 9 月归档做对比。原因见 §1.3：同样的代码，今天比 9 月慢 6–8%。
7. 任何一次运行失败（审计不过、请求数不足、进程残留），保留目录、在 README 里记一行原因，**不计入统计、不删除**，补跑一次。

---

## 1. 自 9 月 10 日以来的代码变化，以及对旧计划的影响

### 1.1 新提交（`c565ecb9..e39d8058`）

| 提交 | 内容 | 对实验的影响 |
| --- | --- | --- |
| `f61f1859` | 删除 7 个无人引用的分析/数据集脚本：`analyze_decode_load_confound`、`analyze_gateway_phase_trace`、`analyze_projection_pa_trace`、`chat_prefix_cache_diagnostic`、`prefix_cache_diagnostic`、`plot_qps_matrix`、`build_synthetic_longctx_dataset` | E5 的 TTFT 分解需要在实验目录里写新的解析脚本。旧 `analyze_gateway_phase_trace.py` 的正则已经和当前日志不匹配（当前日志在 `routing_ms` 后多了 `migration_ms`，见 `vllm/pap/gateway/request_pipeline.py:374`），不能直接恢复使用 |
| `d908915d` | 精简运行时：只保留 Dynamo PA 路由（删 session round-robin 和 `--routing-policy`，`PAP_ROUTING_POLICY≠dynamo` 直接报错）；删 Gateway load tracker（少一次整 prompt 的 block-hash）；每个 PA 只支持 1 个 Attention endpoint；删 GPU-resident dispatch、device graph launch；NVSHMEM bridge ABI 升到 11；配置解析统一 | ① 旧计划 E2c 第 1 步和 A6 的"粘性 RR 路由"在当前代码上**无法运行**，只能用 `c565ecb9` 的 worktree 做历史对照（§6.2）。② **每台机器都要用 `build_nvshmem_device_bridge.sh` 重新编译 bridge**，否则加载时被拒。③ TP>1 和多 Attention rank 已明确不在计划内，写进局限性 |
| `e39d8058` | `PAP-20261015-CLEANUP-REGRESSION`：C30、7PA1P、迁移关闭，清理后/清理前各 2 次交错运行 | 清理本身无可测影响（TTFT −1.94%，95% CI −6.99%～+3.76%；ITL +0.88%，CI −1.03%～+2.86%）。但两个版本的 TTFT 都比 9 月参考值慢 6–8%，期间驱动从 595.84 升到 595.99 |

### 1.2 PAT 修复（PR #1）仍未合入

- PR #1（`claude/project-thread-5pniu5`，`ae727afc`）仍是 open 状态，base 是旧的 `c565ecb9`。我在本地验证过它可以**无冲突地合并**到 `e39d8058`（`git merge-tree` 干净）。PR 只改了 `third_party/pat/csrc/prefix_tree.h` 和一个测试，Python 侧没有改动。
- **PAT 在当前主路径上确实被用到。** `PAP-20261015-CLEANUP-REGRESSION` 的 `current_r1` 归档里，7 个 PA 的 `attention_fast_path_stats_<i>.json` 显示 `attention_kernel_pat_rebuilds` = 16/38/29/27/32/28/25，`triton_selections` = 18/12/16/17/20/14/18，其余约 1100–1290 步复用上一次的计划。也就是说，9 月以来所有 `auto` 策略下的 PAP 数字都可能受这个 bug 影响。所以 E0 仍然是 P0。

### 1.3 旧计划条目的处理（相对 `paper/experiment-plan.md` §8.5）

| 旧条目 | 现在的处理 | 原因 |
| --- | --- | --- |
| E0 PAT 修复 + 主表重跑 + Triton 对照 | **保留，P0**，改成不合并 PR 也能跑（§3） | PAT 确实在用；合并 PR 需要你决定 |
| E2a PD 基线审计 | **缩小**。DP8/PD 已开 prefix caching 和 chunked prefill（`run_dynamo_workload.sh:512–572`），decode 的 CUDA Graph capture 已到 128（`ALIGNED-C30/results/common.env`）。只剩调参（§4.2） | 审计项一半已经能从代码确认 |
| E2e PD KV 传输开销 | **保留，P0**。现成工具：`tooling/analyze_dynamo_ttft.py` 能把 TTFT 拆成路由/Prefill/handoff/Decode 并汇总 NIXL 传输；`tooling/nixl_read_write_probe.py` 可以直接做点对点微基准（§4.1） | 无需写代码 |
| E2b 重放 L07 | **保留，P0**，补上数据集的生成方法（§4.3） | L07 原文件带 `cache_salt`，而当前 PAP 的 Dynamo 集成会拒绝显式 salt，必须先派生无 salt 版本 |
| E1 简化扫描 | **扩大**：机时比原来估计便宜得多（见 1.4），改成 4 架构 × 6 并发 × 3 次，用 240 会话的新数据集（§5.1） | |
| E5 TTFT 尾部分解 | **保留**，换成当前日志格式的新解析器（§6.4） | 旧解析脚本已删除且不匹配 |
| E2c 运行时归因阶梯 | **删除** | 粘性路由已删，eager/local_fast 早已不存在，阶梯走不通；论文里写成"运行时整体替换" |
| A6 路由对比 | **改成**三方对比，粘性 RR 用 `c565ecb9` worktree（§6.2，P1） | 这是多轮放置主张的直接证据 |
| 新增 E2d-lite 负载区间 2×2 | **新增，P0.5** | 新颖性主张是"什么时候划算"，必须有区间图（§5.2） |
| 新增 A2 Projection 聚合、A4 Attention 切片成本 | **新增，P1** | 分别对应"跨 PA 聚合 Projection"和"Attention 切片几乎免费"两条论点（§6.1、§6.3） |
| 新增 O 系列配置调优 | **新增，P1** | 只改环境变量，不改代码（§7） |

### 1.4 成本校准

`CLEANUP-REGRESSION` 的 4 次有效运行，从建目录到下一次运行开始约 3 分钟。其中服务启动约 1.7 分钟，AIPerf 180 个请求约 65 秒，清理约 0.5 分钟。下文按以下公式估算机时（推断）：

> 单次 ≈ 2.5 min（启动 + 审计 + 清理）+ 请求数 / 吞吐

旧计划按每次 12 分钟估算，偏高约 3–4 倍。下面的总量在 38 机时以内，同时做了更多实验。

---

## 2. 通用设置（所有实验共用）

### 2.1 前置检查 X0（约 1 机时，必须先做）

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
# 6. 冒烟：用 §2.3 的模板跑一次 PAP 7PA1P C30（迁移开），和一次 Dynamo 6P2D C30
```

**通过判据**：bridge 编译成功；pytest 309 passed；两次冒烟都是 180/180 请求，所有审计 `STATUS=passed`。PAP 冒烟的 TTFT 均值在 4.2–6.2 s 之间（9 月迁移开为 4.50 s，今天的代码慢 6–8%，再留出余量）。
**结果写到**：`experiments/e2e/PAP-<date>-X0-SMOKE/`。

### 2.2 固定参数

| 项 | 值 |
| --- | --- |
| 模型 | Qwen3-8B FP16，`MAX_MODEL_LEN=32768`，`PAP_HF_OVERRIDES=''` / `DYNAMO_HF_OVERRIDES=''`（9 月 r5/r6 就是因为漏了这一项作废的） |
| PAP 默认 | 7PA1P，Dynamo 路由，**KV 迁移开**（`PAP_ENABLE_PA_KV_MIGRATION=1`，`PAP_KV_MIGRATION_EXPORT_MODE=snapshot`），MPS 20/3（80/12 SM），Prefill budget 2048，Projection budget 256 / max_num_seqs 256 |
| Dynamo 基线 | 完全沿用 `PAP-20260907-ALIGNED-C30/results/common.env`（Prefill 2048、Decode 256、`DYNAMO_ROUTER_PREFILL_LOAD_SCALE=2.0`、CUDA Graph capture 到 128、v2 model runner） |
| PAT | E0 之后所有实验都用**修复版 PAT**（§3.3） |
| 重复 | 每点 3 次独立启动；E0 和主图拐点 ±1 档为 5 次 |
| 客户端 | AIPerf 闭环并发，`timing_mode=concurrency` |

### 2.3 启动模板

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

### 2.4 指标、SLO 与统计（开跑前写进每个实验的 README）

- **逐请求表**（每个实验的 `results/requests.csv`）：列至少包括 `architecture,variant,concurrency,run,conversation_id,turn,input_tokens,output_tokens,cached_tokens,ttft_ms,itl_ms,e2e_ms,start_ns,end_ns`。前 10 列与论文作图脚本 `paper/figures/data/requests.csv` 的格式一致。可以以 `experiments/e2e/PAP-20260909-KV-MIGRATION-ARCHITECTURE-C30/analyze.py` 为模板。
- **ITL**：AIPerf 的请求级 `inter_token_latency`，口径与 9 月相同。
- **SLO（预登记）**：主 SLO 用 **D2**（TTFT ≤ 10 s、ITL ≤ 75 ms、E2E ≤ 20 s），敏感性分析用 D1、D3、D4（定义见 `ALIGNED-C30/GOODPUT.md:46–49`）。L07 系列另外用当年的 Strict/Standard/Relaxed：TTFT ≤ 5/10/20 s，ITL ≤ 50/75/100 ms，不限 E2E。
- **goodput**：达标请求数 ÷ 运行时长（第一个请求开始到最后一个请求结束）。达标率 ≥ 95% 的点算可行。**最大可持续 goodput** = 所有可行并发点里 goodput 的最大值。另外报告"稳态窗口"版本作为敏感性：去掉开始时间落在前 10% 时长内的请求，以及最后一个请求开始之后才结束的请求。
- **置信区间**：按运行分层、在请求层面做 bootstrap（10,000 次），给出 95% 百分位 CI。架构之间的比值，两边各自独立重采样。
- **每个点的 `summary.json`**：吞吐（req/s、output tok/s）、TTFT/ITL/E2E 的均值和 P50/P90/P95/P99、各 SLO 的 goodput 和达标率、CI，以及运行有效性清单。

---

## 3. 阶段 0：E0 PAT 修复验证（P0，约 2 机时）

### 3.1 目的

确认 PR #1 修复的 split-rank 冲突对 9 月以来的 PAP 数字有没有影响。确认之后，所有后续实验都在修复版 PAT 上跑。

### 3.2 离线预检查（0 机时，先做）

对现有 PAP 归档逐个统计 `attention_kernel_pat_rebuilds` 和 `triton_selections`，覆盖 `CLEANUP-REGRESSION` 的 4 次、`KV-MIGRATION-ARCHITECTURE-C30` 的 r7/r8，以及 `ALIGNED-C30` 的 `run_pap1/2`：

```bash
for t in <归档>.tar.gz; do d=$(mktemp -d); tar xzf "$t" -C "$d"
  for f in "$d"/attention_fast_path_stats_[0-9].json; do
    jq -c '{pat:.attention_kernel_pat_rebuilds,tri:.attention_kernel_triton_selections}' "$f"; done; done
```

把结果写成一张表放进 E0 的 README，说明每次运行里 PAT 被选中的程度。

### 3.3 编译修复版 PAT（不合并 PR、不 push）

```bash
git worktree add .worktrees/patfix HEAD
git -C .worktrees/patfix merge --no-edit origin/claude/project-thread-5pniu5   # 只在本地生成合并提交
# 先记录当前（未修复）PAT 的 .so 的 sha256，然后：
PAP_PAT_SOURCE=$PWD/.worktrees/patfix/third_party/pat bash benchmarks/pap/scripts/build_pat_attention.sh
.venv/bin/python -m pytest .worktrees/patfix/tests/pap/test_pap_attention_kernels.py -v
#   预期：全部通过，包括新增的 test_pat_balancing_writes_each_query_split_exactly_once
```

如果你（用户）之后合并了 PR #1，就直接用合并后的分支重新编译，这一节的 worktree 可以删掉。

**注意**：`build_pat_attention.sh` 会把扩展重装进共享的 `.venv`。修复版和未修复版不能同时存在，所以 E0 要按 §3.4 的顺序分块跑。每次 PAT 重装后都要记录 `.so` 的 sha256。

### 3.4 矩阵

数据集 `s60-t3-native32k-stratified-seed42`，C30，7PA1P，迁移开（这是 9 月主表的 headline 配置）。

| 块 | 已安装的 PAT | 运行顺序 |
| --- | --- | --- |
| A | 未修复（当前安装） | `unfixed-auto r1, triton r1, unfixed-auto r2, triton r2, unfixed-auto r3, triton r3` |
| B | 修复版 | `fixed-auto r1, triton r4, fixed-auto r2, triton r5, fixed-auto r3, triton r6` |

共 18 次，约 1 机时。`triton` 不受 PAT 影响，用来检测块 A 和块 B 之间的时间漂移。

**另外跑一次 kernel 级对比**：在修复版和未修复版上分别运行 `tests/pap/test_pap_attention_kernels.py`，记录失败用例。如果未修复版在某些 shape 上失败，就说明这个 bug 在实际测试里能触发。

### 3.5 判据与分支

- 用块 A 和块 B 的 triton 均值之差估计漂移。漂移超过 3% 时，块 B 补一组 A 再测。
- `fixed-auto` 对比 `unfixed-auto` 的 TTFT/ITL/吞吐比值，95% CI 都包含 0 → **PAT bug 对性能无可测影响**。9 月的 KV 迁移 A/B 和跨机结果不用重跑，在附录里写明已检查。
- 任何一个比值的 CI 不包含 0 → 回报，暂停 §4 之后的实验，等我们决定哪些旧结果要重跑。
- `fixed-auto` 对比 `triton`：给出 PAT 在 e2e 上的真实收益（对应旧计划 A7）。如果 PAT 不比 Triton 好，论文里就不把 PAT 当作贡献，后续实验改用 `triton`。这一点写进回报。

**产出**：`experiments/e2e/PAP-<date>-PAT-FIX-E2E/`，内容包括 README、`requests.csv`、`summary.json`、每块 PAT 的 `.so` 哈希、预检查表。
**通过**：18 次运行都有效；pytest 结果已记录。

---

## 4. 阶段 1：决定论文主线的三个实验（P0，约 6 机时）

阶段 1 做完后**先回报**（§9），再进阶段 2。E2b 的结果会决定论文主张的写法。

### 4.1 E2e PD 的 KV 传输开销（"KV 不搬家"的直接证据）

**(a) 离线分析已有归档（0 机时）**：对 `ALIGNED-C30` 的 `6p2d_r1/r2`、`4p4d_r1/r2`、`2p6d_r*`，以及 `ALIGNED-C30-7P1D` 的两次运行，解压后执行：

```bash
.venv/bin/python benchmarks/pap/tooling/analyze_dynamo_ttft.py <解压后的 run_root> --block-size 16 --output <exp>/results/<arch>_<run>_ttft.json
```

输出：每个请求的 handoff 段（Prefill 完成 → Decode 开始）、加权平均传输时间、MB/次、吞吐分布。把 handoff 时间按上下文长度分成 4 档（16–20K、20–24K、24–28K、28–32K），给出 handoff ÷ TTFT 的均值和 P99。

**(b) 点对点微基准（约 1 机时）**：Qwen3-8B 的 KV 布局是 36 层 × K/V = 72 个 region，block 16 token，每个 segment 32 KiB，每 token 147,456 B。上下文长度对应的 `--total-mib`：4K=576、8K=1152、16K=2304、24K=3456、32K=4608。

```bash
source benchmarks/pap/scripts/configure_same_node_nixl.sh && pap_configure_same_node_nixl "$PWD"   # 与 Dynamo 基线相同的 UCX 1.22 设置
for mib in 576 1152 2304 3456 4608; do
  # 1→1，同一 PCIe switch：initiator 为 D 卡，READ（与 vLLM NIXL connector 方向一致）
  .venv/bin/python benchmarks/pap/tooling/nixl_read_write_probe.py --initiator 7 --peer 6 \
     --operation READ --regions 72 --total-mib $mib --segment-kib 32 --repetitions 5
  # 跨 NUMA：从 nvidia-smi topo -m 选一对 SYS 连接的卡
  .venv/bin/python benchmarks/pap/tooling/nixl_read_write_probe.py --initiator 7 --peer <跨 NUMA 卡> \
     --operation READ --regions 72 --total-mib $mib --segment-kib 32 --repetitions 5
done
# 3→1 扇入：三个 probe 同时启动，peer 分别为 0、1、2，initiator 都是 7，各自记录时长
```

`nvidia-smi topo -m` 的输出要保存下来。注意 `PAP-20260910-PCIE-NUMA-DIAG` 记录过 PCIe fatal reset；如果再出现，立即停下并回报。

**(c) 端到端部分**：不单独安排运行，用阶段 2 所有 PD 运行的 `ttft_breakdown.json`。

**产出**：`microbench/PAP-<date>-PD-KV-TRANSFER/`，包括：一张"传输时间与传输占 TTFT 比例 vs 上下文长度"的表（离线 + 微基准两组数），以及字节账（每请求 KV 字节，对比 PAP 每 decode step 的激活 737,280 B/token）。
**判据**（只决定怎么写，不是成败）：

- 16–32K 下 handoff 占 TTFT 的均值 **< 5%** → 论文不能把"省掉传输时间"当主论据，改为强调 Decode 侧 KV 容量、HBM 带宽和多轮放置。这一条回报时要放在第一行。
- **≥ 10%**，或者 3→1 扇入时显著变慢 → 可以作为动机里的主论据。

### 4.2 E2a 基线调参（约 2 机时）

数据集 s60，C30，每档 3 次。目的是证明论文里的 PD 和 DP8 都是调过的，而不是默认值。

| 架构 | 变量 | 取值 |
| --- | --- | --- |
| 6P2D | `DYNAMO_PREFILL_MAX_NUM_BATCHED_TOKENS` | 2048（基线），4096，8192 |
| 6P2D | `DYNAMO_ROUTER_PREFILL_LOAD_SCALE` | 1.0，2.0（基线），4.0 |
| DP8 | `DYNAMO_AGG_MAX_NUM_BATCHED_TOKENS` | 2048（基线），8192 |

一次只改一个变量，共 (2 + 2 + 1) 档 × 3 次 + 基线 6P2D、DP8 各 3 次 = 21 次。

**判据**：某一档在 D2 goodput 或吞吐上比基线好，且 CI 不包含 0 → 之后所有实验的这个架构改用该档，并在回报里写明。否则保持基线不变。

**产出**：`e2e/PAP-<date>-BASELINE-TUNING/`。

### 4.3 E2b 在当前代码上重放 7 月 L07 负载（约 4 机时）

**数据集准备**：L07 的数据文件不在 v026 仓库里。原文件 SHA256 为 `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`，128 会话 × 5 轮 = 640 请求，AIPerf `multi-turn` 格式，每轮都带 `cache_salt`。**当前 PAP 的 Dynamo 集成会拒绝显式 cache_salt**（`benchmarks/pap/datasets/README.md`），所以要新注册一个无 salt 的版本：

1. 在服务器上查找原文件：`find / -name 'multiturn_s128_longtail_random_o16_t5_seed42.jsonl' 2>/dev/null`，然后比对 sha256。L07 当时的路径在 `feature/pap` 的 `run_capacity_matrix.sh` 中是 `<matrix_root>/dataset/` 下。
2. 用当前生成器按原参数重新生成。参数来自 `feature/pap:benchmarks/pap/aiperf/run_capacity_matrix.sh` 第 20–68 行；生成器与当时相比只是不再写 `cache_salt`。
   ```bash
   .venv/bin/python benchmarks/pap/datasets/tools/generate_multiturn_dataset.py \
     --model /data/ssd1/llm-models/Qwen3-8B \
     --corpus /home/fei/research/PD/refer_codes/vllm/benchmarks/sonnet_4x.txt \
     --output benchmarks/pap/datasets/multiturn/l07-s128-t5-o16-seed42-shared-prefix/dataset.jsonl \
     --sessions 128 --turns 5 \
     --document-tokens 4096 --document-tokens-median 4000 --document-tokens-min 2048 --document-tokens-max 5632 \
     --append-tokens 1100 --append-tokens-median 400 --append-tokens-min 4 --append-tokens-max 2125 \
     --output-tokens 16 --output-tokens-median 15 --output-tokens-min 8 --output-tokens-max 32 \
     --random-seed 42 --sampled-mean-tolerance 0.40 --max-model-len 32768 \
     --think-time-ms 1000 --tool-time-ms 300 --tool-every 3 \
     --session-prefix pap-pd-dp-s128-t5-seed42
   ```
3. 如果第 1 步找到了原文件：把原文件去掉每个 `turn.extra.cache_salt` 之后，应该和第 2 步的输出**逐字节相同**。相同就说明是精确重放；不同就在 manifest 里写明差异，并把它作为"同分布重放"。注意去掉 salt 后，不同会话之间可以共享前缀，这和 7 月的会话隔离不同，也要在 manifest 里写明。
4. 写 `dataset.manifest.json`（来源、命令、sha256、去 salt 说明），并在 `SHA256SUMS` 里追加一行。

**矩阵**：`PAP_AIPERF_CUSTOM_DATASET_TYPE=multi-turn`，`SESSIONS=128`，`EXPECTED_REQUESTS=640`。PAP 使用修复版 PAT，迁移开。

| 架构 | 并发点（以 L07 的边界为中心） |
| --- | --- |
| PAP 7PA1P | 24, 30, 34, 40, 48 |
| PAP 6PA2P（`PAP_TOPOLOGY=6pa2p`，`PREFILL_GPUS=0..5`，`PROJECTION_GPUS=6,7`） | 24, 32, 40, 48, 56 |
| PD 6P2D | 31, 38, 44, 48, 56 |
| DP8 | 8, 18, 28, 36, 44 |

每点先跑 2 次（与 L07 相同）。按 Strict/Standard/Relaxed 找到每个架构的可行边界后，边界点和它的上一档各补到 3 次。总计约 20 × 2 + 16 = 56 次；L07 量级的负载吞吐为 10–16 req/s，每次约 3.5 分钟，合计约 3.5 机时。如果某个架构在最高点仍然可行，就往上加一档，直到不可行为止。

**产出**：`e2e/PAP-<date>-L07-REPLAY/`，包括一张和 L07 原表同格式的表（每档最佳 PAP、最佳 PD、DP 和比值），外加我们自己的 D2 结果。
**判据**（决定论文写法，见 `paper/experiment-plan.md` §3.3）：

- PAP/PD 的 Standard goodput 比值 ≥ 0.95 → "反证已消除"。7 月结果写进设计演进一节。
- 0.75–0.95 → "差距缩小"，短输出负载划为 PD 的优势区间。
- < 0.75（7 月是 0.63）→ "差距仍在"，论文主张要收窄到长上下文区间。

这三种结果都算实验成功。回报时要写明属于哪一种。

---

## 5. 阶段 2：主结果（P0，约 15 机时）

### 5.1 E1 负载扫描（主图）

**新数据集** `agentic-code/s240-t3-native32k-stratified-seed42`，生成方法与 s60 完全相同，只是会话数变成 240，每个上下文分层 60 个。原来的 2000 会话候选池里，最高的 28–32K 分层只有 29 个候选，不够分，所以候选池要扩大到 10,000：

```bash
# 1. 候选池（与 s60 manifest 中的 source_generation 相同，只改 num_sessions）
.venv-aiperf/bin/aiperf synthesize agentic-code \
  --config benchmarks/pap/datasets/agentic-code/generator-configs/native32k-s3.json \
  --num-sessions 10000 --seed 42 --output /tmp/pap-native32k-pool-10000
#   如果 CLI 参数名与此不同，以 `aiperf synthesize agentic-code --help` 为准，把实际命令写进 manifest
# 2. 分层抽样
.venv/bin/python benchmarks/pap/datasets/tools/filter_agentic_code_dataset.py \
  --input /tmp/pap-native32k-pool-10000/dataset.jsonl \
  --output benchmarks/pap/datasets/agentic-code/s240-t3-native32k-stratified-seed42/dataset.jsonl \
  --sessions 240 --min-turns 3 --take-first-turns 3 --hash-block-size 512 --seed 42 \
  --final-context-min 16384 --final-context-max 32768 --final-context-strata 4
```

**验收**：manifest 中 4 个分层各选中 60 个；每请求输出均值 < 100 token；上下文分布与 s60 接近（均值差 < 10%）。

**一致性检查**（4 次）：PAP 7PA1P 和 6P2D 在 s240 上各跑 C30 × 2 次，和阶段 0/1 在 s60 上的 C30 结果比较。两者的 PAP/6P2D 吞吐比值方向要一致；不一致就回报。

**矩阵**：s240（720 请求），每点单独启动，3 次，按架构交错顺序：

| 架构 | 并发 |
| --- | --- |
| DP8、PD 6P2D、PD 4P4D、PAP 7PA1P（迁移开，修复版 PAT） | 10, 20, 30, 40, 50, 60 |

共 72 次。低并发点慢（C10 时 720 个请求约 12–15 分钟，推断），平均每次约 8 分钟，合计约 10 机时。找到每个架构 D2 下的拐点（最后一个可行点）后，拐点和下一档补到 5 次，再加约 16 次。E2a 选出更好的档位时，6P2D 和 DP8 就用那一档。

**产出**：`e2e/PAP-<date>-LOAD-SWEEP/`，包括：
- 吞吐–延迟曲线数据（x = 实际 req/s，y = TTFT 和 ITL 的 P50/P99，带 CI）；
- D1–D4 下的最大可行 goodput 表，带 CI；
- SLO 缩放曲线：D2 三个阈值同乘 0.5–2.0，每个架构的最大可行 goodput；
- 每轮 TTFT 表（用来复核"PAP 每轮 TTFT 单调下降"）；
- 所有 PD 运行的 `ttft_breakdown.json` 汇总（E2e 的端到端部分）。

**判据**：每个点 720/720 请求、审计通过。PAP 是否胜出不是判据。结论按实际结果写，包括 PAP 输的点。

### 5.2 E2d-lite 负载区间 2×2（"什么时候划算"，约 5 机时）

两个维度：输入（短：L07 约 4K + 0.7K 追加；长：native-32K）× 输出（短：O16 或约 O80；长：O256）。其中两格已经有数据：短输入短输出 = E2b，长输入短输出 = E1。只需要补两格长输出：

| 新数据集 ID | 生成方法 |
| --- | --- |
| `multiturn/l07-s128-t5-o256-seed42-shared-prefix` | §4.3 的命令，只改 `--output-tokens 256 --output-tokens-median 240 --output-tokens-min 128 --output-tokens-max 512`，输出路径相应改名 |
| `agentic-code/s120-t3-native32k-o256-stratified-seed42` | 复制 `native32k-s3.json` 为新的生成配置 `native32k-s3-o256.json`（只改 `generation_length`：mean 256、median 240、min 128、max 512），用 §5.1 的流程生成候选池并分层抽 120 个会话（每层 30） |

新的生成配置文件属于数据集 provenance，放在 `generator-configs/` 下，属于新增文件，可以直接加。

**矩阵**：每格 PAP 7PA1P、PD 6P2D、DP8 各 4 个并发点 × 2 次。长输出格的并发点要先跑一个粗扫：在 C8/16/32/48 各跑 1 次找到 D2 边界，然后在边界附近取 4 个点。合计约 2 格 × 3 架构 × 4 点 × 2 次 = 48 次，加粗扫约 24 次，约 5 机时。

**产出**：`e2e/PAP-<date>-REGIME-2X2/`，一张 2×2 的 PAP/PD 和 PAP/DP8 最大 goodput 比值表（带 CI，四格都填），以及每格的 bytes 账：PD 每请求搬的 KV 字节 vs PAP 每请求搬的激活字节（= 737,280 × 输出 token 数）。
**判据**：四格都要有有效数据。分析假设（推断）：输出越长，PAP 的激活字节越接近 PD 的 KV 字节，PAP 的优势会缩小或反转；输入越长，PAP 的优势越大。不管实测是否符合这个假设，都照实回报。

---

## 6. 阶段 3：机制与消融（P1，约 8 机时）

### 6.1 A2 Projection 跨 PA 聚合的贡献

对应论点：Projection 无状态，所以能把 7 个 PA 的 decode 合成一个大 batch。s240，C30 和 C50 各 2 次，迁移开。

| 变量 | 取值 |
| --- | --- |
| `PAP_PROJECTION_MAX_NUM_SEQS` | 256（基线），64，32 |
| 拓扑（同时改 GPU 列表） | 7PA1P（基线），6PA2P，4PA4P |

共 (2 + 2) × 2 并发 × 2 次 = 16 次，加上和基线的交错。另外在 C30 7PA1P 跑 1 次开 trace：设 `PAP_PROJECTION_PA_TRACE_OUTPUT=<run_root>/projection_trace.pt`，用 `tooling/merge_detailed_projection_trace.py` 和 `capture_aligned_trace.py` 合并，得到 Projection 单步时间随 batch 大小的曲线。

**产出**：`e2e/PAP-<date>-PROJECTION-AGGREGATION/`。
**判据**：给出"batch 上限减半 → ITL / 吞吐变化多少"，以及每个拓扑的吞吐和 ITL。

### 6.2 A5/A6 多轮放置三方对比

对应论点：KV 固定在 PA 之后，"迁移 + 负载感知路由"比"粘性"和"不迁移"都好。这也是和 AMPD、PPD 这类"会话绑定到 KV 持有者"方案最接近的对照。

| 变体 | 怎么跑 |
| --- | --- |
| `dynamo+migration` | 当前代码，`PAP_ENABLE_PA_KV_MIGRATION=1` |
| `dynamo-only` | 当前代码，`PAP_ENABLE_PA_KV_MIGRATION=0` |
| `sticky-rr` | 用 `c565ecb9` 的 worktree，按 `CLEANUP-REGRESSION` 的 `base` 做法：`git worktree add .worktrees/rr-base c565ecb904`，在该 worktree 里打上修复版 PAT（同 §3.3，用合并后的 `third_party/pat` 编译），用 ABI 10 的 bridge 前缀 `.local/nvshmem-3.3.24-cuda13-bridge-v10`，`PYTHONPATH=<worktree>`；路由用 `PAP_ROUTING_POLICY=session_round_robin`，设置 `PAP_SESSION_ROUTER_MAX_CONVERSATIONS=<会话数>`，并且必须设 `PAP_ENABLE_PA_KV_MIGRATION=0`（旧代码中迁移要求 Dynamo 路由，见该版本 `run_pap_workload.sh:396`）。参数名以该 worktree 里的 `run_pap_workload.sh` 和 `PAP-20260909-SESSION-ROUND-ROBIN` 的记录为准 |

**问题**：PAT 是共享 `.venv` 里的扩展。`sticky-rr` 用的是旧代码 + 修复版 PAT，可以共用同一个安装，只要 PAT 的 Python 接口没有变。冒烟时确认一次；如果不兼容，就三个变体都改用 `PAP_ATTENTION_KERNEL_POLICY=triton`，在 README 写明。

**矩阵**：s240，C30 和 C50，3 变体 × 3 次，按变体轮流，共 18 次，约 2 机时。
**产出**：`e2e/PAP-<date>-PLACEMENT-3WAY/`，包括每轮 TTFT、每个 PA 的负载不均（每个 PA 的请求数和 Prefill token 数的变异系数）、迁移次数和迁移字节数、重算的 prefill token 数。
**判据**：所有运行有效。`sticky-rr` 来自旧代码，README 里要写明"非单变量：代码版本不同，只有路由是有意改变的"。

### 6.3 A4 Attention 切片的代价

对应论点：从 Prefill 卡切出 12 个 SM 做 Attention 几乎是免费的。s60，C30，3 次。

| Prefill/Attention chunk | SM |
| --- | --- |
| 21/2 | 84/8 |
| 20/3（基线） | 80/12 |
| 19/4 | 76/16 |
| 18/5 | 72/20 |

每组修改 `PAP_STATIC_PREFILL_CHUNKS`、`PAP_STATIC_ATTENTION_CHUNKS` 和两个 `EXPECTED_SMS`。共 12 次，约 0.7 机时。另外从 E1 的 DP8 数据里取同负载下的 Prefill 吞吐（92 SM 满卡），换算成"每 SM 的 prefill token/s"，和 PAP 80 SM 的值对比。

**产出**：`e2e/PAP-<date>-MPS-SLICE/`，每组的 TTFT/ITL/吞吐，以及"Attention 每多给 4 个 SM，TTFT 和 ITL 各变多少"。

### 6.4 E5 TTFT 尾部分解

在 E1 中 PAP 的 C30 和拐点各跑 2 次，设置 `PAP_PREFILL_IPC_PROFILE=1` 和 `PAP_DYNAMO_ROUTING_TRACE_OUTPUT=<run_root>/routing_trace.jsonl`。在实验目录里写 `analyze.py`，解析 Gateway 日志中的三类行：

- `PAP proxy prefill IPC profile request_id=… register_ms tokenization_ms routing_ms migration_ms prefill_payload_ms prefill_ms readiness_ms projection_payload_ms pre_projection_ms`（`vllm/pap/gateway/request_pipeline.py:374`）
- `PAP proxy projection admission profile … admission_ms request_to_admission_ms`（同文件 :409）
- `PAP proxy projection stream profile … first_chunk_ms request_to_first_chunk_ms`（`gateway/handoff.py:100`）

再用 `PAP Dynamo placement request_id=… selected_pa=…`（`gateway/dynamo_routing.py:278`）关联请求被放到哪个 PA，与 AIPerf 的逐请求 TTFT 按 request_id 对齐。

**profile 开关本身可能有开销**：同时用 `PAP_PREFILL_IPC_PROFILE=0` 跑 2 次作对照。开销超过 3% 时在 README 写明。

**产出**：`e2e/PAP-<date>-TTFT-BREAKDOWN/`，包括 P50/P99 请求的 TTFT 分段堆叠表、每个 PA 的排队长度和请求数，以及对比 DP8 的算力账（PAP Prefill 共 7 × 80 = 560 SM，DP8 共 8 × 92 = 736 SM）。共 6 次，约 0.6 机时。

---

## 7. 优化方向（只改环境变量；涉及代码的放在 §7.2，需要先批准）

### 7.1 配置调优（P1，约 2 机时，可以直接做）

目标是缩小 PAP 相对 DP8 的 TTFT 差距（9 月 PAP 4.50 s，DP8 2.21 s），并改善 TTFT P99。每档 3 次，s240，C30 和 E1 的 PAP 拐点。

| ID | 变量 | 取值 | 预期（推断） |
| --- | --- | --- | --- |
| O1 | `PAP_PREFILL_MAX_NUM_BATCHED_TOKENS` | 2048（基线），4096，8192 | 更大的 chunk 让 Prefill 的 GEMM 效率更高，TTFT 下降；代价是同卡 Attention 受干扰，ITL 上升。research 分支只测过 2048 和 4096 各 1 次 |
| O2 | `PAP_DYNAMO_PREFILL_LOAD_SCALE` | 1.0，2.0（基线），4.0 | 路由更看重 Prefill 负载，可能降低 PA 之间的不均衡和 TTFT 尾部 |
| O3 | `PAP_MIGRATION_MAX_CONVERSATIONS` | 会话数（基线），会话数/2 | 检查迁移表上限会不会成为瓶颈，预计没有影响 |

一次只改一个变量。如果某档在 D2 goodput 上比基线好且 CI 不包含 0，就在回报里**建议**把它作为新默认值，但已经跑完的 E1 不重跑。是否改默认值由我们决定。

**产出**：`e2e/PAP-<date>-CONFIG-TUNING/`。

### 7.2 需要改代码的方向（不要做，只在回报里评估可行性和预计收益）

| 方向 | 动机 | 和论文的关系 |
| --- | --- | --- |
| 动态 MPS / Green Context 切分 | semi-PD、Bullet、MuxWise 都批评静态切分；E5 和 A4 会显示静态 80/12 损失多少 | 放进讨论和 future work |
| "PA-local Projection" 变体（每个 PA 自带 Projection，不跨 PA 聚合） | 直接隔离"跨 PA 聚合"本身的贡献 | 比 A2 更干净的消融；只有在 A2 结论不清楚时才考虑 |
| AMPD 式 PD 基线（会话绑定到持有 KV 的 D，增量 prefill 在 D 上本地执行） | `novelty-assessment.md` §5 列为冲更高级别会议时的必做项 | Euro-Par 预算内不做，用 §6.2 的 `sticky-rr` 近似 |
| 多轮更深的负载（≥ 6 轮） | 真实 agentic 会话远超 3 轮；当前 native-32K 只能放 3–4 轮 | 可以只加数据集、不改代码；先评估 PA 显存够不够 |

---

## 8. 总量与顺序

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

**门控点**：

- **G0**（E0 之后）：PAT bug 有可测影响 → 停下回报。
- **G1**（阶段 1 之后）：一定要回报，等我们确认论文主线后再开始阶段 2。如果 48 小时内没有回复，就按"差距缩小"的写法默认继续。
- 阶段 2 和阶段 3 之间不需要等，但每个阶段结束都要回报一次。

---

## 9. 回报格式

每个阶段结束时，在 `results/europar27` 分支上提交并 push：

1. 各实验目录：`README.md`（问题、协议、有效性、结论）、`experiment.env`/`requested.env`、`run.sh`、`analyze.py`、`results/summary.json`、`results/requests.csv`、`results/SHA256SUMS`、`results/archives/*.tar.gz`。打包方式沿用 `CLEANUP-REGRESSION`：配置、启动环境、审计、runner log、AIPerf 导出放进归档，完整服务日志留在本地 `runs/`。
2. 一个阶段汇总文件 `paper/pap-v026/reports/phase<N>-<date>.md`，格式如下：

```markdown
# Phase N 回报（日期，source commit，驱动版本，PAT .so sha256）

## 需要你决定的事（没有就写"无"）

## 结论（每个实验一行：结论 + 关键数字 + 95% CI + 判据结果）

## 有效性（每个实验：计划次数 / 有效次数 / 作废原因）

## 和预期不符的地方

## 下一步（按本手册，或建议的调整）
```

3. push 之后，在项目对话里贴出分支名和汇总文件路径。

---

## 附：需要用户决定的事

1. **是否合并 PR #1**（PAT 修复）到 `feature/pap-v026`。它和新 HEAD 无冲突。本手册默认不合并、在本地 worktree 编译修复版 PAT（§3.3）。合并之后流程更简单，E0 的结果也不受影响。
2. **G1 之后的论文主线**：根据 E2b 和 E2e 的结果，三选一（反证消除 / 差距缩小 / 差距仍在）。
3. §7.2 中有哪些代码方向要进入本轮。默认都不做。

## 附：证据位置

- 清理回归：`benchmarks/pap/experiments/e2e/PAP-20261015-CLEANUP-REGRESSION/README.md`
- 9 月主表：`…/e2e/PAP-20260909-KV-MIGRATION-ARCHITECTURE-C30/report.md`；Dynamo 基线参数：`…/e2e/PAP-20260907-ALIGNED-C30/results/common.env`
- SLO D1–D4：`…/e2e/PAP-20260907-ALIGNED-C30/GOODPUT.md:46–49`
- L07：`feature/pap:benchmarks/pap/experiments/PAP-20260729-RESEARCH-L07/report.md`；生成参数：`feature/pap:benchmarks/pap/aiperf/run_capacity_matrix.sh:20–68`
- PAT 选择逻辑：`vllm/pap/attention/backend.py`（`PAPPATOrTritonSelector`）、`vllm/pap/attention/peers.py:163`；统计输出：`run_pap_workload.sh:998` 起
- PD 传输分析：`benchmarks/pap/tooling/analyze_dynamo_ttft.py`；微基准：`benchmarks/pap/tooling/nixl_read_write_probe.py`
- 运行时可调参数：`benchmarks/pap/scripts/run_pap_workload.sh:80–331`，`run_dynamo_workload.sh:9–60`
