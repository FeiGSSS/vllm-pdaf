# R-001: 阶段 0：环境检查（X0）与 PAT 修复验证（E0）

status: accepted
priority: P0
claims: C10, C11
opened: 2026-10-15
from: research agent
common settings: `paper/handoff/requests/COMMON.md`（本请求的所有运行都遵守）

## Hypothesis

PAT 的 split-rank 冲突（PR #1，已合入 `9b814d13`）对 PAP 的端到端性能没有可测影响：修复前后 TTFT、ITL、吞吐比值的 95% CI 都包含 0。

## Why it matters for the paper

9 月以来所有 PAP 数字（C1–C3、C7、C8）都在 `auto` 策略下跑，PAT 确实被选中（每个 PA 有 16–38 次 PAT 计划重建，见 COMMON §1.2）。这个请求决定这些旧数字是否还能用，也决定 PAT 能不能当作贡献（C11）。之后所有请求都在修复版 PAT 上运行。

## Expected result and falsification

支持：`fixed-auto` vs `unfixed-auto` 的三个比值 CI 都包含 0。证伪：任何一个不包含 0 → 设 `status: blocked`，报告哪些旧结果需要重跑，暂停 R-002 及之后的请求。

## Suggested setup

### X0

先按 COMMON §2 做前置检查和冒烟，结果写进 `e2e/PAP-<date>-X0-SMOKE/`。

### E0 目的

确认 PR #1 修复的 split-rank 冲突对 9 月以来的 PAP 数字有没有影响。确认之后，所有后续实验都在修复版 PAT 上跑。

### E0 离线预检查（0 机时，先做）

对现有 PAP 归档逐个统计 `attention_kernel_pat_rebuilds` 和 `triton_selections`，覆盖 `CLEANUP-REGRESSION` 的 4 次、`KV-MIGRATION-ARCHITECTURE-C30` 的 r7/r8，以及 `ALIGNED-C30` 的 `run_pap1/2`：

```bash
for t in <归档>.tar.gz; do d=$(mktemp -d); tar xzf "$t" -C "$d"
  for f in "$d"/attention_fast_path_stats_[0-9].json; do
    jq -c '{pat:.attention_kernel_pat_rebuilds,tri:.attention_kernel_triton_selections}' "$f"; done; done
```

把结果写成一张表放进 E0 的 README，说明每次运行里 PAT 被选中的程度。

### 编译两版 PAT

修复已合入当前分支。未修复版只作为 E0 的对照，从合并前的 `e39d8058` 编译：

```bash
# 未修复版（块 A 之前）
git worktree add .worktrees/pat-unfixed e39d8058
PAP_PAT_SOURCE=$PWD/.worktrees/pat-unfixed/third_party/pat bash benchmarks/pap/scripts/build_pat_attention.sh
find .venv -name '*prefix_attn*.so' -exec sha256sum {} \;
.venv/bin/python -m pytest tests/pap/test_pap_attention_kernels.py -v   # 记录失败用例（预期新增的 split-rank 用例可能失败）

# 修复版（块 B 之前，以及之后所有实验）
bash benchmarks/pap/scripts/build_pat_attention.sh                      # 默认用当前分支的 third_party/pat
find .venv -name '*prefix_attn*.so' -exec sha256sum {} \;
.venv/bin/python -m pytest tests/pap/test_pap_attention_kernels.py -v   # 预期全部通过，含 test_pat_balancing_writes_each_query_split_exactly_once
```

E0 完成后**必须保持修复版安装**，删除 `.worktrees/pat-unfixed`。

**注意**：`build_pat_attention.sh` 会把扩展重装进共享的 `.venv`。修复版和未修复版不能同时存在，所以 E0 要按下面"矩阵"一节的顺序分块跑。每次 PAT 重装后都要记录 `.so` 的 sha256。

### 矩阵

数据集 `s60-t3-native32k-stratified-seed42`，C30，7PA1P，迁移开（这是 9 月主表的 headline 配置）。

| 块 | 已安装的 PAT | 运行顺序 |
| --- | --- | --- |
| A | 未修复（`e39d8058` 编译） | `unfixed-auto r1, triton r1, unfixed-auto r2, triton r2, unfixed-auto r3, triton r3` |
| B | 修复版 | `fixed-auto r1, triton r4, fixed-auto r2, triton r5, fixed-auto r3, triton r6` |

共 18 次，约 1 机时。`triton` 不受 PAT 影响，用来检测块 A 和块 B 之间的时间漂移。

**kernel 级对比**："编译两版 PAT"一节里两次 pytest 的结果对照。如果未修复版在某些 shape 上失败，就说明这个 bug 在实际测试里能触发。

### 判据与分支

- 用块 A 和块 B 的 triton 均值之差估计漂移。漂移超过 3% 时，块 B 补一组 A 再测。
- `fixed-auto` 对比 `unfixed-auto` 的 TTFT/ITL/吞吐比值，95% CI 都包含 0 → **PAT bug 对性能无可测影响**。9 月的 KV 迁移 A/B 和跨机结果不用重跑，在附录里写明已检查。
- 任何一个比值的 CI 不包含 0 → 回报，把本请求设为 `blocked`，暂停 R-002 及之后的请求，等我们决定哪些旧结果要重跑。
- `fixed-auto` 对比 `triton`：给出 PAT 在 e2e 上的真实收益（对应旧计划 A7）。如果 PAT 不比 Triton 好，论文里就不把 PAT 当作贡献，后续实验改用 `triton`。这一点写进回报。

**产出**：`experiments/e2e/PAP-<date>-PAT-FIX-E2E/`，内容包括 README、`requests.csv`、`summary.json`、每块 PAT 的 `.so` 哈希、预检查表。
**通过**：18 次运行都有效；pytest 结果已记录。

## Deliverables

- `paper/handoff/reports/R-001.md`：X0 结果、预检查表、两版 PAT 的 `.so` 哈希和 pytest 结果、`fixed/unfixed` 和 `fixed/triton` 的比值与 CI、结论（是否 blocked）。
- 更新 `claims.md` 中 C10、C11 的 Status/Evidence；如果 C1–C9 因此需要降级，一并更新。
