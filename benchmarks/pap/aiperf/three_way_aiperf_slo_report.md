# 三架构（DP/PD/PAP）AI Perf SLO 对比扫描报告

更新时间（本地扫描基线）：`2026-07-28`

## 1. 数据源与目标

本报告用于“以 SLO 为导向”的三类架构对比：
- DP：8 副本 fused（`8dp`）
- PD：`4p4d`, `6p2d`（`7p1d` 已在本轮扫描命令加入，但当前复用矩阵尚未包含该变体的结果）
- PAP：`6pa2p`, `7pa1p`

数据来源：
- `benchmarks/pap/experiments/_staging/capacity/20260725_8gpu_capacity_fullscan_c16_48`
- 统一 AIPerf 8 卡 workload：128 会话，5 轮对话，`O16` 长尾随机输入配置
- SLO 门限：
  - Strict：`TTFT<=5000ms, ITL<=50ms, good>=0.95`
  - Standard：`TTFT<=10000ms, ITL<=75ms, good>=0.95`
  - Relaxed：`TTFT<=20000ms, ITL<=100ms, good>=0.95`

## 2. 一键生成入口

```bash
PAP_CAPACITY_MATRIX_ID=20260728_dp_pd_pap_slo_fullscan \
PAP_CAPACITY_ARCHITECTURES=dp_8,pd_4p4d,pd_6p2d,pd_7p1d,pap_6pa2p,pap_7pa1p \
PAP_CAPACITY_DP_8_POINTS=12,16,20,24,28,32 \
PAP_CAPACITY_PD_4P4D_POINTS=12,16,20,24,28,32 \
PAP_CAPACITY_PD_6P2D_POINTS=12,16,20,24,28,32 \
PAP_CAPACITY_PD_7P1D_POINTS=12,16,20,24,28,32 \
PAP_CAPACITY_PAP_6PA2P_POINTS=12,16,20,24,28,32,40,48 \
PAP_CAPACITY_PAP_7PA1P_POINTS=12,16,20,24,28,32,40,48 \
PAP_CAPACITY_REPETITIONS=1 \
PAP_CAPACITY_WAIT_FOR_GPUS=1 \
bash benchmarks/pap/aiperf/run_three_way_slo_capacity.sh
```

当前环境 `nvidia-smi` 不可用时，先用复用模式跑汇总：

```bash
PAP_CAPACITY_SKIP_RUN=1 \
PAP_CAPACITY_MATRIX_ROOT=benchmarks/pap/experiments/_staging/capacity/20260725_8gpu_capacity_fullscan_c16_48 \
PAP_CAPACITY_ARCHITECTURES=dp_8,pd_4p4d,pd_6p2d,pd_7p1d,pap_6pa2p,pap_7pa1p \
PAP_CAPACITY_SKIP_MISMATCH=0 \
PAP_CAPACITY_WAIT_FOR_GPUS=0 \
bash benchmarks/pap/aiperf/run_three_way_slo_capacity.sh
```

## 3. SLO 最优结果（正确运行 + 通过阈值）

- **Strict**
  - PAP：`7pa1p @ C16`，3.347 req/s（TTFTp95=2896.8ms，ITLp95=40.2ms，good=97.97%）
  - PD：无合格点（在该矩阵内）
  - DP：无合格点（在该矩阵内）

- **Standard**
  - PAP：`6pa2p @ C32`，4.339 req/s（TTFTp95=8561.5ms，ITLp95=39.6ms，good=97.7%）
  - PD：`6p2d @ C24`，3.596 req/s（TTFTp95=9762.4ms，ITLp95=38.7ms，good=95.0%）
  - DP：`8dp @ C16`，4.006 req/s（TTFTp95=1800.7ms，ITLp95=67.9ms，good=96.4%）

- **Relaxed**
  - PAP：`7pa1p @ C32`，4.894 req/s（TTFTp95=5979.4ms，ITLp95=82.6ms，good=97.8%）
  - PD：`6p2d @ C24`，3.785 req/s（TTFTp95=9762.4ms，ITLp95=38.7ms，good=100.0%）
  - DP：`8dp @ C24`，5.181 req/s（TTFTp95=2205.2ms，ITLp95=93.6ms，good=96.2%）

## 4. 结构化对比（标准化）

| 架构 | Strict | Standard | Relaxed |
| --- | --- | --- | --- |
| PAP（最佳） | 7pa1p @ C16，3.347 | 6pa2p @ C32，4.339 | 7pa1p @ C32，4.894 |
| PD（最佳） | 无（未合格） | 6p2d @ C24，3.596 | 6p2d @ C24，3.785 |
| DP（最佳） | 无（未合格） | 8dp @ C16，4.006 | 8dp @ C24，5.181 |

- PAP 相对 PD：Standard +20.65%，Relaxed +29.27%
- PAP 相对 DP：Standard +8.30%，Relaxed -5.55%

## 5. 注意与后续

这个对比基线是基于历史已完成矩阵复现得到的，不是本次新跑完的实时结果；主要用于验证“扫描链路 + SLO 计算口径”正确性与复用。

若要得到新一轮“完整三架构 12~48 并发点位”的最终扫描，请在 GPU 可用后直接执行上节“一键生成入口”。
