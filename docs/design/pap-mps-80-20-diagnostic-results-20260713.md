# PAP MPS 80:20 诊断结果

日期：2026-07-13

状态：完成一轮 C4 quick。该结果用于判断静态 MPS 配比方向，不是新的 formal
baseline。PD-oneway、PD-twoway 以及 PAP 70:30 继续使用已冻结的三次拉丁方结果。

## 1. 问题与单变量

假设：PAP Round 1 的 16K Prefill 只获得 70% MPS，可能是 R1 TTFT 显著落后 PD 的
原因之一。处理组只把同一 PA GPU 上的 Prefill/Attention MPS 从 `70/30` 改为
`80/20`；模型、GPU、请求 token IDs、五轮 C4 到达、KV ownership、transport、调度与
正确性 Gate 均不变。

后续 TPOT 优化仍固定在 70:30。本实验不扫描更多 MPS 比例，也不会改变三路 formal
runner 的默认值。

## 2. 结果

所有数值为 request-level median，单位为 ms。

| Scope | PD-oneway frozen | PD-twoway frozen | PAP 70:30 frozen | PAP 80:20 quick | 80:20 vs 70:30 | 80:20 / PD-twoway |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 TTFT | 8112.026 | 8128.513 | 11077.283 | 9887.638 | -10.74% | 1.216x |
| R1 TPOT | 35.516 | 35.477 | 39.121 | 48.611 | +24.26% | 1.370x |
| R2-R5 TTFT | 280.867 | 251.716 | 249.030 | 269.488 | +8.22% | 1.071x |
| R2-R5 TPOT | 42.176 | 42.155 | 51.148 | 63.471 | +24.09% | 1.506x |

80:20 把 R1 TTFT 降低 `1189.645 ms`，但仍比 PD-twoway 慢 `21.6%`。与此同时，
R1 和稳态 TPOT 都恶化约 `24%`；稳态 TTFT 也从与 PD-twoway 持平退化到慢 `7.1%`。

## 3. 机制证据

三次 frozen PAP 70:30 cell 的 R1 `prefill_ms` 为：

```text
run1: 5035, 8981, 12810, 15469
run2: 5029, 9064, 12874, 15522
run3: 5025, 9029, 12851, 15485
```

80:20 quick 为：

```text
4530, 8077, 11437, 13780
```

pooled 70:30 median 为 `10937 ms`，80:20 median 为 `9757 ms`，下降 `10.79%`，与
R1 TTFT 的 `10.74%` 降幅几乎一致。因此“70% Prefill 份额限制首轮长上下文计算”得到
直接支持。

反面证据同样明确：Attention 份额从 30% 降到 20% 后，R1 与 R2-R5 TPOT 同时增加约
24%。这说明静态 80:20 只是把共享 GPU 时间从逐层 Decode Attention 转给 Prefill，
没有消除 PAP 数据面或调度开销。

## 4. 正确性与 provenance

- commit：`ba17ea18cc999b94709200900dd294f15122b519`；tracked worktree clean；
- workload：Qwen3-8B FP16、1PA1P、16K、5 turns、C4、o256、轮内 q2；
- 完成/失败：`20/0`；peak HTTP/decode concurrency：`4/4`；
- cache validation：20 requests、16 transitions，全部通过；
- strict correctness：`STATUS=passed`、`MATCH_COUNT=0`；
- routing：`STATUS=passed`、20 requests；
- session drain：`STATUS=passed`、`ACTIVE_SESSIONS=0`；
- aggregate validity：`passed`，warnings 为空；
- 原始目录：
  `test/baseline/pap/results/runs/20260713_ba17ea18c_pap_mps_80_20_c4_quick`。

## 5. 决策

拒绝把静态 80:20 晋升为默认配置。它证明 Prefill MPS 是 R1 TTFT 差距的一部分，但以
不可接受的 TPOT 回退换取收益。PAP 默认和下一阶段 TPOT 优化基线保持 70:30。本次
quick 的变化远大于 frozen 70:30 三次重复的离散范围，且方向已经构成明确 rejection
evidence，因此不继续消耗 GPU 做 80:20 formal 三次重复。
