# R-002: 阶段 1：PD 的 KV 传输开销（E2e）、基线调参（E2a）、L07 重放（E2b）

status: accepted
priority: P0
claims: C12, C13, C14
opened: 2026-10-15
from: research agent
common settings: `paper/handoff/requests/COMMON.md`（本请求的所有运行都遵守）

depends: R-001 不是 blocked

## Hypothesis

三个独立的假设：

- E2e（C12）：在 PCIe L20、16–32K 上下文下，PD 的 P→D KV handoff 占 TTFT 的 10% 以上。
- E2a（C13）：论文中的 PD 6P2D 和 DP8 基线已经调好，单个参数改动不会在 D2 goodput 上带来超出 CI 的提升。
- E2b（C14）：在 7 月 L07 的短输出负载上，当前 PAP 的 Standard SLO 最大 goodput 至少是 PD 的 0.95 倍（7 月是 0.63）。

## Why it matters for the paper

这三个结果决定论文主线（G1）。E2e 决定"KV 不搬家"能不能作为动机的主论据；E2a 决定基线能否经得起"PD/DP 没调好"的质疑；E2b 回答"7 月的反证还在不在"。R-003 要等 research agent 读完本报告、在 R-003 写下 `cleared:` 后才开始。

## Expected result and falsification

各小节的判据都写在下面。三种 E2b 结果（反证消除 / 差距缩小 / 差距仍在）都算有效结果，报告里写明属于哪一种。

### 对 X-001 中 E2b 公平性问题的回应

去掉 salt 之后，不同会话之间可以共享前缀，这和 7 月的会话隔离不同。为了让"反证消除"这个结论站得住，增加两项检查：

1. 报告第 0 轮请求（每个会话的首轮）的 `cached_tokens` 分布。L07 的文档是从 sonnet 语料随机截取的，预计跨会话命中很少（推断）。如果首轮的平均缓存命中率超过 5%，在报告中把 E2b 的结论降为"同分布、非严格重放"。
2. **PD 复现检查**：PD 6P2D 在新数据集上的 Standard SLO 最大 goodput 应该在 7 月数值（15.305 good req/s，C44）的 ±15% 以内。如果超出这个范围，说明负载或基线已经变了，E2b 只能写成"在当前负载上的对比"，不能写成"7 月反证被消除"。

## Suggested setup

### E2e PD 的 KV 传输开销（"KV 不搬家"的直接证据）

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

**(c) 端到端部分**：不单独安排运行，用 R-003 所有 PD 运行的 `ttft_breakdown.json`。

**产出**：`microbench/PAP-<date>-PD-KV-TRANSFER/`，包括：一张"传输时间与传输占 TTFT 比例 vs 上下文长度"的表（离线 + 微基准两组数），以及字节账（每请求 KV 字节，对比 PAP 每 decode step 的激活 737,280 B/token）。
**判据**（只决定怎么写，不是成败）：

- 16–32K 下 handoff 占 TTFT 的均值 **< 5%** → 论文不能把"省掉传输时间"当主论据，改为强调 Decode 侧 KV 容量、HBM 带宽和多轮放置。这一条回报时要放在第一行。
- **≥ 10%**，或者 3→1 扇入时显著变慢 → 可以作为动机里的主论据。

### E2a 基线调参（约 2 机时）

数据集 s60，C30，每档 3 次。目的是证明论文里的 PD 和 DP8 都是调过的，而不是默认值。

| 架构 | 变量 | 取值 |
| --- | --- | --- |
| 6P2D | `DYNAMO_PREFILL_MAX_NUM_BATCHED_TOKENS` | 2048（基线），4096，8192 |
| 6P2D | `DYNAMO_ROUTER_PREFILL_LOAD_SCALE` | 1.0，2.0（基线），4.0 |
| DP8 | `DYNAMO_AGG_MAX_NUM_BATCHED_TOKENS` | 2048（基线），8192 |

一次只改一个变量，共 (2 + 2 + 1) 档 × 3 次 + 基线 6P2D、DP8 各 3 次 = 21 次。

**判据**：某一档在 D2 goodput 或吞吐上比基线好，且 CI 不包含 0 → 之后所有实验的这个架构改用该档，并在回报里写明。否则保持基线不变。

**产出**：`e2e/PAP-<date>-BASELINE-TUNING/`。

### E2b 在当前代码上重放 7 月 L07 负载（约 4 机时）

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
**判据**（决定论文写法，见 `paper/research/experiment-plan.md` §3.3）：

- PAP/PD 的 Standard goodput 比值 ≥ 0.95 → "反证已消除"。7 月结果写进设计演进一节。
- 0.75–0.95 → "差距缩小"，短输出负载划为 PD 的优势区间。
- < 0.75（7 月是 0.63）→ "差距仍在"，论文主张要收窄到长上下文区间。

这三种结果都算实验成功。回报时要写明属于哪一种。

## Deliverables

- `paper/handoff/reports/R-002.md`：第一行写 E2e 的判据结果（< 5% 还是 ≥ 10%），然后写 E2a 选中的档位，以及 E2b 属于哪种结果和两项公平性检查。
- `paper/figures/` 下的小数据文件：handoff 占 TTFT 比例 vs 上下文长度（CSV）；L07 重放表（CSV）。
- 更新 `claims.md` 中 C12–C14。
