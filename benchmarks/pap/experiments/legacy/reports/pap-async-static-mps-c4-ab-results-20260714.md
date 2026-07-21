# PAP Async Decode-Token × Static MPS C4 A/B

日期：2026-07-14

状态：`accepted-development-baseline`

## 1. 结论

在完全相同的 C4 多轮负载和代码 patch 上完成 dynamic/static MPS 与 async
decode-token off/on 的 2×2 A/B。结果表明：

- static MPS **不能解决** async 路径的 TTFT 退化；R1 async 代价在 dynamic 和
  static 下分别为 `+888.6 ms` 和 `+883.9 ms`，几乎相同；
- async 路径在 static 下把 steady TPOT 降低 `0.899 ms`（`-1.77%`）；
- static MPS 在 async 开启时进一步把 steady TPOT 降低 `0.419 ms`
  （`-0.83%`）；
- 因此接受 async 无 barrier 路径和 static 64/28 MPS 作为后续 PAP 优化默认基线，
  同时保留同步路径与 dynamic 70:30 的显式回退。

## 2. 固定负载与实验控制

- 模型：本地 Qwen3-8B，FP16，TP1；
- 架构：1PA1P，Prefill/Attention 在 GPU 1，Projection 在 GPU 2；
- 4 个活跃 conversation，5 轮；
- 首轮 16000-token document，后四轮各追加 120 tokens；
- 每轮输出 256 tokens，request rate `2.0/s`；
- trace 关闭，`max_num_seqs=4`；
- 四格都使用同一 tracked worktree patch，SHA-256 为
  `45d4c8feb71396ae22f38c55a22582addaad0883e27bd1e005b3be3137e0f89a`；
- static client 实测 Prefill/Attention 分别只可见 `64/28` 个 SM。

## 3. 结果矩阵

| MPS | async | R1 TTFT median | R1 TPOT median | R2–R5 TTFT median | R2–R5 TPOT median |
| --- | --- | ---: | ---: | ---: | ---: |
| dynamic 70:30 | off | 11012.007 ms | 39.224 ms | 247.406 ms | 51.161 ms |
| dynamic 70:30 | on | 11900.620 ms | 36.525 ms | 350.104 ms | 50.292 ms |
| static 64/28 | off | 11135.140 ms | 39.223 ms | 259.277 ms | 50.773 ms |
| static 64/28 | on | 12019.072 ms | 36.447 ms | 344.426 ms | 49.873 ms |

同一 MPS 模式内的 async 增量：

| MPS | R1 TTFT | R2–R5 TTFT | R1 TPOT | R2–R5 TPOT |
| --- | ---: | ---: | ---: | ---: |
| dynamic 70:30 | +888.612 ms | +102.698 ms | -2.699 ms | -0.869 ms |
| static 64/28 | +883.931 ms | +85.149 ms | -2.775 ms | -0.899 ms |

## 4. 因果判断

async 开启后，第一个长 Prefill 基本不变；后续三个请求的 Prefill/排队时间逐渐
增加。dynamic 与 static 的增长曲线几乎重合，因此“Attention 在 dynamic MPS 下
抢占 Prefill SM”不是主要原因。

相同的 Attention 总计算量为 `184320` rows，但 dynamic async off/on 的每层
Attention 调用次数从 `1739` 增加到 `1838`，static 则从 `1742` 增加到 `1836`。
async 消除 Projection barrier 后，decode cadence 更紧、batch 更碎，同时产生
decode-token sideband 控制流量。static MPS 只隔离 SM，不隔离 HBM、L2、copy
engine、launch front-end 或 CPU/HTTP 控制路径，因此无法消除这一 TTFT 副作用。

这说明 TTFT 问题属于 async 实现引发的调度/共享资源副作用，而不是 sampled-token
D2H copy 自身变慢。后续优化应保留 async TPOT 收益，优先减少 sideband 与 batch
碎片化。

## 5. 正确性与运行目录

四格均满足：20/20 请求完成、16/16 多轮 cache transition 命中、strict correctness
通过、decode-token join 通过、routing 通过、session drain 后 active session 为 0。

- dynamic off：
  `benchmarks/pap/experiments/legacy/runs/20260714_async_mps_ab_c4_dynamic_off`
- dynamic on：
  `benchmarks/pap/experiments/legacy/runs/20260714_async_mps_ab_c4_dynamic_on`
- static off：
  `benchmarks/pap/experiments/legacy/runs/20260714_async_mps_ab_c4_static_off`
- static on：
  `benchmarks/pap/experiments/legacy/runs/20260714_async_mps_ab_c4_static_on`

这些是同一 dirty patch 上的一次严格因果 A/B，足以决定开发默认值；提交后的
formal-clean 北极星仍需单独运行，不能用本结果替代。

## 6. 默认值与回退

PAP 多轮 test bed 默认：

```text
PAP_ASYNC_DECODE_TOKEN=1
PAP_LOAD_MPS_PROFILE=baseline_static_64_28
```

消融或兼容性回退：

```text
PAP_ASYNC_DECODE_TOKEN=0
PAP_LOAD_MPS_PROFILE=baseline_70_30
```

底层通用 runner 的 `PAP_MPS_MODE` 默认仍为 `dynamic`，只有 PAP 多轮 wrapper 在
本机北极星负载中默认选择 static profile；不支持 R595 static partition 的环境可
显式使用 dynamic profile。
