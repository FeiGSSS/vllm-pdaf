# R-003: 阶段 2：负载扫描（E1）与负载区间 2×2（E2d-lite）

status: open
priority: P0
claims: C15, C16
opened: 2026-10-15
from: research agent
common settings: `paper/handoff/requests/COMMON.md`（本请求的所有运行都遵守）

depends: R-002 done；research agent 在本文件写 `cleared:` 行之后开始（48 小时没有回复就按 R-002 报告里的默认分支继续）
cleared:

## Hypothesis
- E1（C15）：在 native-32K 多轮负载上，PAP 7PA1P（迁移开）在 D2 SLO 下的最大可持续 goodput 高于 DP8、PD 6P2D 和 PD 4P4D。
- E2d（C16）：PAP 相对 PD 的优势随输入变长而增大，随输出变长而缩小。

## Why it matters for the paper
E1 用"负载扫描 + 最大可持续 goodput + CI"取代 9 月 C30 单点的主结果（C1–C3 将由它替换），并补上重复次数和置信区间（G2 的一部分）。E2d 是新颖性主张"什么时候划算"的核心证据（`paper/research/novelty-assessment.md` §3）。

## Expected result and falsification
E1 的判据是数据有效，不是 PAP 获胜；PAP 输的点照实报告。E2d 的假设不成立时也要照实回报，论文会按实测的区间写。

## Suggested setup

### E1 负载扫描（主图）

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

**一致性检查**（4 次）：PAP 7PA1P 和 6P2D 在 s240 上各跑 C30 × 2 次，和 R-001/R-002 在 s60 上的 C30 结果比较。两者的 PAP/6P2D 吞吐比值方向要一致；不一致就回报。

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

### E2d-lite 负载区间 2×2（"什么时候划算"，约 5 机时）

两个维度：输入（短：L07 约 4K + 0.7K 追加；长：native-32K）× 输出（短：O16 或约 O80；长：O256）。其中两格已经有数据：短输入短输出 = R-002 的 E2b，长输入短输出 = E1。只需要补两格长输出：

| 新数据集 ID | 生成方法 |
| --- | --- |
| `multiturn/l07-s128-t5-o256-seed42-shared-prefix` | R-002"E2b"一节的数据集生成命令，只改 `--output-tokens 256 --output-tokens-median 240 --output-tokens-min 128 --output-tokens-max 512`，输出路径相应改名 |
| `agentic-code/s120-t3-native32k-o256-stratified-seed42` | 复制 `native32k-s3.json` 为新的生成配置 `native32k-s3-o256.json`（只改 `generation_length`：mean 256、median 240、min 128、max 512），用 本请求 E1 一节的流程生成候选池并分层抽 120 个会话（每层 30） |

新的生成配置文件属于数据集 provenance，放在 `generator-configs/` 下，属于新增文件，可以直接加。

**矩阵**：每格 PAP 7PA1P、PD 6P2D、DP8 各 4 个并发点 × 2 次。长输出格的并发点要先跑一个粗扫：在 C8/16/32/48 各跑 1 次找到 D2 边界，然后在边界附近取 4 个点。合计约 2 格 × 3 架构 × 4 点 × 2 次 = 48 次，加粗扫约 24 次，约 5 机时。

**产出**：`e2e/PAP-<date>-REGIME-2X2/`，一张 2×2 的 PAP/PD 和 PAP/DP8 最大 goodput 比值表（带 CI，四格都填），以及每格的 bytes 账：PD 每请求搬的 KV 字节 vs PAP 每请求搬的激活字节（= 737,280 × 输出 token 数）。
**判据**：四格都要有有效数据。分析假设（推断）：输出越长，PAP 的激活字节越接近 PD 的 KV 字节，PAP 的优势会缩小或反转；输入越长，PAP 的优势越大。不管实测是否符合这个假设，都照实回报。



## Deliverables
- `paper/handoff/reports/R-003.md`。
- `paper/figures/`：吞吐–延迟曲线数据、D1–D4 最大可行 goodput 表、SLO 缩放曲线、每轮 TTFT 表、2×2 区间比值表（都是小 CSV/JSON）。
- 更新 `claims.md` 中 C15、C16，以及因 E1 而变化的 C1–C3、C6、C9。

