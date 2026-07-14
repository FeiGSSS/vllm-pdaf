# PAP async decode-token TTFT 回归因果诊断

日期：2026-07-14

状态：`root-cause-closed-development-validated`；clean commit 后的 formal freeze 待执行

## 1. 最终结论

早期 barrier、MPS 和 Decode gate A/B 正确定位了触发条件，但把 Prefill CUDA event 的
首尾跨度误当成了“GPU 一直在慢算”。后续 bounded Torch Profiler 和逐层 IPC profile
已经关闭微观根因：主因不是 HBM/L2 或 SM 算力竞争，而是 **Attention registry 全局锁
把 GPU 数据面工作与 Prefill 控制面串行化**。

完整因果链为：

```text
Projection 去掉 token-boundary barrier
  -> decode forward/call 更碎、Attention append 临界区进入更频繁
  -> append 在 registry 全局锁内执行跨 GPU K/V 搬运、slot tensor 构造和
     reshape_and_cache_flash launch
  -> Prefill 每层发布 CUDA-IPC KV descriptor 时等待同一 registry 锁
  -> Prefill host 线程无法提交下一层 GPU kernel
  -> GPU 出现 host-unsubmitted 空洞，长 Prefill 与后续请求逐层累积
  -> Prefill HTTP/TTFT 增加
```

关键证据如下：

1. pure-Prefill 与正常 Decode 的 32 个 profiler iteration 中，主 kernel 数量完全相同；
   正常 Decode 的 GPU busy 只增加 `158.270 ms`，但 GPU span 增加
   `2966.765 ms`；其中 `2809.416 ms` 被归类为 host-unsubmitted gap，已经排除
   “kernel 已提交但被 MPS/HBM 拖慢”。
2. 同步 Prefill IPC profile 中，R1 四个请求的逐层 TCP response wait 总和为
   `42.8 / 366.1 / 930.8 / 1687.6 ms`，后到请求累计约 `3.03 s`，与 profiler 的
   host gap 同量级。
3. 把 IPC install 放到后台仍无收益；async profile 证明 R1 第 3、4 个请求仅
   `enqueue_prefill_paged_kv_descriptor()` 等 registry 锁就分别累计约
   `863 / 1626 ms`，实际 IPC open/install 总和只有约 `14--15 ms`。
4. 将 GPU copy/kernel launch 移出 registry 锁后，默认同步 import 的 R1
   `prefill_ms` 从 `5031/9437/14401/18007` 降到
   `5036/9038/12853/15489`；无 Decode 理想轨迹为
   `5047/8739/12438/14993`。第 3、4 个请求分别恢复约 `1.55 s/2.52 s`。

最终修复采用两层同步：专用 decode-append 锁只保持 decode KV 写入顺序；registry
全局锁只在读取状态快照和提交 `seq_len` 时短暂持有。GPU tensor 构造、跨 GPU copy 和
kernel launch 全部在 registry 锁外；提交时重新验证 state identity 和旧 `seq_len`，避免
并发覆盖。

在此基础上，Prefill CUDA-IPC descriptor 默认异步安装。异步 worker 透传 unified-KV
lease/capacity/writable-range 元数据，并绑定 session epoch 防 ABA。Decode 还必须等待
对应 layer 的 `state.seq_len >= session.prefix_len`；没有这个完整-prefix readiness 门时，
首个后续轮请求会偶发读取早期 chunk 并产生输出分叉。

三次 dirty-development C4 重复的逐请求 prompt/output digest 映射完全一致；R1
`prefill_ms` 分别为：

```text
5025/9017/12812/15424
5031/9023/12820/15433
5018/8928/12708/15321
```

三次总输出吞吐为 `60.33/60.15/60.57 token/s`，整体 TPOT 中位数为
`50.12/50.20/49.86 ms`。旧正常 async control 为 `58.54 token/s` 和
`49.52 ms`；吞吐提升约 `2.7%--3.5%`，TPOT 中位数变化约在 `±1.4%`。R1 单请求
TPOT 变大主要因为后续请求更早完成 Prefill、更早进入并发 Decode；总完成时间和吞吐
反而改善。所有重复均为 20/20 requests、16/16 cache transitions、0 join error、0
残留 session。

这三次因 tracked worktree dirty 只属于 `controlled` 证据，不能冒充 formal-clean
freeze。提交后仍需按同一 C4 workload 跑三次 clean formal。

## 2. 负载语义校正

本轮 C4 中的 `C4` 表示 **4 个 active conversations**，不是 QPS 4。固定配置为：

- 模型：Qwen3-8B，FP16，TP1；
- 拓扑：1PA1P；GPU1 为 Prefill + Attention，GPU2 为 Projection；
- static MPS：Prefill/Attention 为 `64/28` SM；
- first turn：16000 document tokens；后四轮各追加 120 tokens；
- 每请求输出 256 tokens，共 5 轮；
- 每轮 active conversations 为 4；
- `request_rate_per_round=2.0`，请求约在 `0/0.5/1.0/1.5 s` 发起；
- 每轮末有 round barrier，所以下一轮不会在上一轮完成前发起；
- `max_num_seqs=4`，chunked-prefill 上限 4096 tokens。

因此这里的 `2 req/s` 只是每轮内部的发起节奏，不是全程持续吞吐。以无 barrier 这一格
为例，20 个请求运行约 87.5 秒，端到端平均吞吐只有约 `0.228 req/s`。

## 3. Projection / Prefill barrier 正交 A/B

四格都启用 async sampled-token sideband，只改变 sync-only barrier 所在进程。trace 只在
Prefill 记录低频 model/sample CUDA event，并在 Attention 记录 layer-0 调用时间；四格的
诊断开关完全相同。

| Projection barrier | Prefill barrier | R1 TTFT | steady TTFT | R1 TPOT | steady TPOT | steady Prefill | calls/layer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | off | 12025.106 ms | 335.194 ms | **36.607 ms** | **49.925 ms** | 216.0 ms | 1835 |
| on | off | 11147.056 ms | **246.510 ms** | 39.523 ms | 50.502 ms | **131.0 ms** | 1738 |
| off | on | 12001.909 ms | 334.654 ms | **36.590 ms** | **49.930 ms** | 211.0 ms | 1836 |
| on | on | **11112.424 ms** | 246.571 ms | 39.516 ms | 50.684 ms | 132.0 ms | 1739 |

每格均为 20/20 requests、16/16 cache transitions；cache、digest、routing、token/KV join、
lease release 和 session drain 全部通过。以上是 trace-on causal run，不替代 trace-off formal
北极星。

交互效应非常清楚：Projection barrier 决定结果，Prefill barrier 的主效应接近零。因此
早期“全局环境变量同时污染 Prefill 与 Projection，可能直接改变 Prefill”的担忧已经由
正交实验排除；旧实验虽然作用域不严谨，但恢复 TTFT 的实际来源仍是 Projection 侧。

## 4. Prefill 到底有没有变化

需要区分逻辑工作和物理执行。

### 4.1 逻辑工作没有变化

- R1 每请求均计算 `16013` tokens；
- R2–R5 分别计算 `153/149/145/141` tokens，其余历史由 APC 命中；
- 每请求的 cached/computed tokens 和 cache validation 四格一致；
- chunked-prefill 的 model-runner 调用形状一致；
- steady 请求 admission queue 约为 `0.005–0.061 ms`，远小于 TTFT 差异；
- 每轮请求发起间隔仍约 0.5 秒，负载生成器没有因 barrier 改变。

R1 后三个长 Prefill 会因前一个 Prefill 尚未完成而排队；后四轮的 append Prefill 小于
0.3 秒，短于 0.5 秒到达间隔，Prefill 之间不重叠。因此 steady 的增长不能解释为
Prefill-Prefill 排队。

### 4.2 GPU 首尾 span 变化，但实际 kernel busy 基本不变

steady R2–R5 按每轮请求序号统计：

| 配置 | conv0 Prefill | conv1 | conv2 | conv3 |
| --- | ---: | ---: | ---: | ---: |
| Projection barrier off | 126.0 ms | 198.5 ms | 243.5 ms | 281.0 ms |
| Projection barrier on | 125.5 ms | 132.0 ms | 131.5 ms | 134.0 ms |
| Prefill barrier only | 125.0 ms | 195.5 ms | 222.0 ms | 266.5 ms |

每轮开始前有 round barrier，所以 conv0 Prefill 时没有 active decode；conv1/2/3 Prefill 时
约有 1/2/3 个已有请求在 decode。无 Projection barrier 时，Prefill 随 active decode 数
单调变慢；恢复 Projection barrier 后基本保持平坦。

早期 CUDA event 表明模型首尾 GPU span 发生变化：

| 配置 | conv0 model GPU | conv1 | conv2 | conv3 |
| --- | ---: | ---: | ---: | ---: |
| Projection barrier off | 108.14 ms | 181.35 ms | 228.02 ms | 265.60 ms |
| Projection barrier on | 108.06 ms | 114.72 ms | 115.71 ms | 117.42 ms |
| Prefill barrier only | 108.28 ms | 178.46 ms | 204.67 ms | 249.34 ms |

该 event 只测量模型区间内第一个和最后一个 GPU event 的距离，包含中间 GPU idle，不能
证明 kernel 本身一直在执行。后续 Torch Profiler 将同一区间拆成 GPU busy 与 gap：正常
Decode 相对 pure-Prefill 的 busy 只增加 `158.270 ms`，而 host-unsubmitted gap 增加
`2809.416 ms`。因此早期表格仍是有效的墙钟现象，但“物理 GPU 算慢”这一解释已被推翻；
它实际反映 Prefill host 线程被 Attention registry 锁阻塞后没有提交下一层 kernel。

### 4.3 TTFT 增量不是一个 Projection step

将 steady TTFT 近似拆成 `Prefill wall + Prefill 返回后到 first token`，无 barrier 相对
Projection barrier 的增量为：

| 每轮请求序号 | TTFT 增量 | Prefill wall 增量 | Prefill 后增量 |
| --- | ---: | ---: | ---: |
| conv0 | 0.3 ms | 0.5 ms | 0.2 ms |
| conv1 | 74.8 ms | 66.5 ms | 9.6 ms |
| conv2 | 125.2 ms | 112.0 ms | 17.8 ms |
| conv3 | 161.9 ms | 147.0 ms | 15.6 ms |

因此“新请求错过已提交的 `step i+1`，多等约 0–1 个 TPOT”只适用于最后一列；实测
该部分确实小于一个约 50 ms 的 TPOT。较大的 66–147 ms 在请求进入 Projection 前就已
发生。

后续请求没有等待前一请求 first token 才发起。它们仍按约 0.5 秒固定到达；由于 append
Prefill 小于 0.3 秒，steady Prefill 之间也没有排队。看似逐请求累计，是因为 conv1/2/3
做 Prefill 时分别已有约 1/2/3 个 decode 在共享 PA GPU 上运行，系统负载逐步增加，而
不是请求 TTFT 形成串行依赖。

R1 的 16K Prefill 则不同：单次 Prefill 约 5 秒，远大于 0.5 秒到达间隔，后续长 Prefill
会在同一 Prefill server 上排队或 chunk-interleave。R1 的无 barrier 相对 Projection
barrier TTFT 增量约为 conv1/2/3 的 `322/1434/2425 ms`，对应 Prefill wall 增量约
`346/1458/2416 ms`；这部分会通过 Prefill backlog 向后传播，但仍不是等待前一请求
first token。

## 5. Attention 调度证据如何解释

四格总 Attention rows 都是每层 `5120`，但：

- 无 Projection barrier：`1835 calls/layer`，平均 `2.790 rows/call`；
- 有 Projection barrier：`1738 calls/layer`，平均 `2.946 rows/call`。

因此 barrier 改变了 forward 快照的聚合程度。已有的无 barrier + 关闭 Projection async
scheduling 诊断也把 calls/layer 降到 `1722`，steady Prefill 恢复到约
`128/133.5/133.5/139 ms`；但 steady TPOT 退化到 `52.656 ms`。这说明 async queue 深度与
Projection 节奏确实参与因果链，但关闭 async scheduling 只适合作为诊断，不是默认修复。

在 append Prefill wall window 内，layer-0 调用数与窗口长度近似同比增长，单看调用率无法
区分争用资源。更重要的是 conv1 只有一个 active decode，也不存在“多个请求被拆成小
batch”的空间，但无 barrier 仍把 Prefill model GPU 从约 108 ms 拉到约 181 ms。因此：

- 更小 batch / 更多 launch 是事实；
- 但它不能单独解释全部 slowdown；
- Projection queue-ahead 改变 Attention append 进入全局 registry 临界区的频率，是最终
  解释；
- static MPS 固定 SM 集合，但不隔离 CPU 锁；这里无需诉诸 HBM/L2/P2P 资源饱和即可解释
  主回归。

## 6. 已排除与仍未解决

| 假设 | 判定 | 证据 |
| --- | --- | --- |
| C4 实际是 QPS4、每 0.25 秒发一个 Prefill | 排除 | profile 固定为 4 conversations、2 req/s、约 0.5 秒间隔 |
| Prefill cache 命中或 computed tokens 变化 | 排除 | 四格 shape/cache 完全一致 |
| steady Prefill scheduler queue 增加 | 排除 | admission queue 基本均小于 0.1 ms |
| Prefill 进程自身少了 barrier | 排除 | Prefill-only barrier 对指标和 model GPU span 基本无效 |
| sampled-token payload / CPU tuple 是主因 | 排除 | Projection sync-only 不做 token D2H 也恢复结果 |
| 两个独立 Projection cohort 相互抢占 | 排除 | 只有一个逻辑 cohort；两份是在途 forward 快照 |
| batch 碎片化是唯一微观根因 | 尚未证明 | conv1 单 active decode 仍出现大幅 model GPU slowdown |
| 最终争用是 HBM/L2、P2P copy 还是 work-queue 相位 | 排除为主因 | Torch trace 显示 kernel busy 仅增约 1%，`2.81 s` 来自 host-unsubmitted gap；IPC profile 将 gap 对齐到 registry 锁等待 |

## 7. 三个严格隔离实验

以下诊断均保持同一 C4 负载、同一提交 `cbcbfabcbd91`、trace-off、async sampled-token
sideband 和 `64/28` static MPS。每个完整 run 的 correctness、cache、routing、token/KV
join、lease release 和 session drain 都通过。诊断开关默认关闭，不改变生产默认语义。

### 7.1 活跃 CUDA context 审计

审计不再只启动独立 probe context，而是在实际 Prefill worker 和 Attention executor 完成
CUDA 初始化后写出 device UUID、partition UUID 和可见 SM 数；runner 同时用 `lspart`
确认这两个 partition 有真实 client。

四个严格对照 run 均得到：

- Prefill：`64` SM，Prefill partition UUID 匹配，`clients=Yes`；
- Attention：`28` SM，Attention partition UUID 匹配，`clients=Yes`。

因此“服务实际落回整卡或落入错误 partition”已经排除。static MPS 的确生效，但它只
隔离 SM，不隔离 HBM、L2、copy engine/P2P 和 GPU work queue。

### 7.2 R1 Projection gate：排除 barrier 对 Prefill 的直接作用

Proxy 让 R1 四个请求各自完成 Prefill 后停在 Projection 入口，直到四个 Prefill 全部
完成才统一放行。因此整个测量窗口中没有任何 Projection/Attention Decode。只切换
Projection sync-only barrier，R1 `prefill_ms` 为：

| 配置 | conv0 | conv1 | conv2 | conv3 |
| --- | ---: | ---: | ---: | ---: |
| 无 Decode，Projection barrier off | 5030 | 8717 | 12408 | 14955 |
| 无 Decode，Projection barrier on | 5029 | 8710 | 12394 | 14939 |
| on - off | -1 | -7 | -14 | -16 |

1--16 ms 只占总时长约 `0.02%--0.11%`，属于实验噪声。由此可以确认：barrier 不会
直接改变 Prefill；它必须先改变正在运行的 Decode，才会间接影响同卡 Prefill。由于 gate
人为让 first token 等待四个 Prefill，以上两组的 TTFT/TPOT 不用于性能比较。

### 7.3 Decode commit gate：区分 GPU 数据面与 CPU/KV 控制反馈

这一组保留请求完成 Prefill 后的真实 Projection、QKV P2P 和 Attention，只让异步
Decode KV commit worker 暂存 commit，等第四个 R1 Prefill 完成后再放行。它切断
`Attention -> HTTP decode-commit -> Prefill EngineCore/KV manager` 反馈，但不阻止
Decode GPU 数据面。

为避免与旧 trace-on 数据混用，另外补跑了同一代码、同一 trace-off 配置的正常 async
control：

| R1 `prefill_ms` | conv0 | conv1 | conv2 | conv3 |
| --- | ---: | ---: | ---: | ---: |
| 正常 async | 5031 | 9437 | 14401 | 18007 |
| Decode 正常、commit 暂存 | 5032 | 9317 | 14255 | 17865 |
| 完全无 Decode | 5030 | 8717 | 12408 | 14955 |
| 正常 - 完全无 Decode | 1 | 720 | 1993 | 3052 |
| 正常 - commit 暂存 | -1 | 120 | 146 | 142 |
| commit 暂存 - 完全无 Decode | 2 | 600 | 1847 | 2910 |

对 conv1/2/3 做一阶分账：阻断 commit 控制反馈只消除了约
`120/146/142 ms`，即总 Decode 干扰的约 `16.7%/7.3%/4.7%`；保留 GPU Decode
数据面后仍有约 `83.3%/92.7%/95.3%` 的增量。单次 run 的差分不能解释所有交互项，
但数量级已经足以排除“HTTP commit 或 Prefill KV manager 是主要根因”。这里切断的是
Decode 完成后的 commit 反馈，并未切断 Prefill 每层 descriptor 在 Attention 侧获取
registry 锁的路径；因此该实验不能排除、也没有排除最终发现的 Attention registry 锁。

## 8. 修复设计与验证

### 8.1 缩短 registry 临界区

`append_decode_kv_to_unified_prefill_cache()` 原来在 `self._lock` 内完成状态校验、slot
tensor 构造、QKV 跨设备转换、`reshape_and_cache_flash` launch 和 `seq_len` 更新。
修复后：

1. 获取专用 `_decode_append_lock`，串行化同一 Attention registry 的 decode append；
2. 短暂获取 `self._lock`，读取 state、position、slot-plan 和 scale cache 快照；
3. 释放 `self._lock` 后构造 tensor、搬运 K/V 并 launch GPU kernel；
4. 再次短暂获取 `self._lock`，验证 state object 未替换且 `seq_len` 仍等于快照值；
5. 验证通过后提交 `seq_len += 1`，否则 fail closed。

这样既保持 decode 写入顺序，也允许 Prefill descriptor、session/readiness 和统计查询在
GPU 数据面执行期间进入 registry。单测用阻塞的 fake GPU op 证明 GPU 工作期间 stats
查询不会被全局锁阻塞。

### 8.2 安全异步 Prefill descriptor

仅把 import 放到 worker 不足以修复锁竞争；缩锁完成后，异步 import 才用于清除剩余的
descriptor open/install 和 TCP ACK 开销。安全条件包括：

- worker 透传 `lease_id`、`leased_block_ids`、capacity、prefix 和 writable range；
- queue item 绑定 `session_request_id + session_epoch`，旧 session 的迟到 work item 直接
  丢弃，不能标记新 session failed；
- unified Decode 在每层检查 `state.seq_len >= session.prefix_len`，未达到完整 prefix 时
  在 condition 上等待并释放 registry 锁；
- worker failure 设置 per-layer failed readiness，Decode fail closed。

未加完整-prefix 门的反例 run
`20260714_registry_lock_narrow_async_c4` 中，conversation 0 的 R2–R5 输出 digest 从
`5252...` 分叉为 `a0fc...`；加门后单次 C4 和三次重复都恢复逐项一致。因此该门属于
正确性条件，不是可选性能优化。

### 8.3 当前默认与剩余工作

benchmark test bed 默认使用 `PAP_PREFILL_KV_ASYNC=1`，并可显式设为 `0` 回退同步
import；`PAP_PREFILL_IPC_PROFILE=0` 默认关闭。两个值均写入 effective config 和 run
metadata。通用代码在没有 launcher 显式配置时仍保持保守的 async-off 语义。

未显式设置上述两个变量的 16K/5-turn/C1 quick smoke
`20260714_registry_lock_default_async_fingerprint_c1` 已确认 effective config 为 async `1`、
profile `0`，且两个值进入 implementation fingerprint；
5/5 requests、4/4 cache transitions、digest、routing、token/KV join 和 session drain 全部
通过，R2--R5 均命中上一轮 decode 派生的 256 tokens。该 smoke 只验证默认接线与正确性，
不替代 C4 性能结果。

Prefill-window pacing 不再是当前首选修复，因为根因可在不牺牲已有 Decode TPOT 的情况
下通过锁分离解决。提交后应运行三次 clean formal C4；随后再用同样的 lock/readiness
语义验证 1PA2P 和 2PA2P，避免把 1PA1P 的单 worker 假设外推到任意 x:y。

## 9. 原始证据

| 实验 | 目录 | 用途 |
| --- | --- | --- |
| 两侧无 barrier | `test/baseline/pap/results/runs/20260714_prefill_scope_ab_none_c4` | 当前 async 行为 |
| 仅 Projection barrier | `test/baseline/pap/results/runs/20260714_prefill_scope_ab_projection_only_c4` | Projection 侧因果干预 |
| 仅 Prefill barrier | `test/baseline/pap/results/runs/20260714_prefill_scope_ab_prefill_only_c4` | 排除 Prefill 侧直接作用 |
| 两侧 barrier | `test/baseline/pap/results/runs/20260714_prefill_scope_ab_both_c4` | 复现旧全局 sync-only 语义 |
| 旧 Projection single batch | `test/baseline/pap/results/runs/20260714_async_token_projection_single_batch_c4` | async queue 深度诊断 |
| 当前 clean async/static | `test/baseline/pap/results/runs/20260714_cbcbfabcb_async_static_clean_c4` | trace-off 参考 |
| R1 gate、无 Projection barrier | `test/baseline/pap/results/runs/20260714_strict_isolation_projection_gate_async_c4` | 完全排除 Decode 的 Prefill 窗口 |
| R1 gate、有 Projection barrier | `test/baseline/pap/results/runs/20260714_strict_isolation_projection_gate_sync_c4` | 验证 barrier 无直接 Prefill 作用 |
| Decode commit gate | `test/baseline/pap/results/runs/20260714_strict_isolation_commit_gate_async_c4` | 只切断 Decode 到 Prefill 的控制反馈 |
| 正常 async 严格 control | `test/baseline/pap/results/runs/20260714_strict_isolation_normal_async_c4` | commit gate 的同版本 trace-off 对照 |
| pure-Prefill Torch trace | `test/baseline/pap/results/runs/20260714_prefill_torch_profile_no_decode_c4` | kernel count/busy 和 host gap 基准 |
| 正常 Decode Torch trace | `test/baseline/pap/results/runs/20260714_prefill_torch_profile_normal_async_c4` | 证明 `2.81 s` 增量为 host-unsubmitted gap |
| 同步 IPC profile | `test/baseline/pap/results/runs/20260714_prefill_ipc_profile_sync_c4` | 将 host gap 对齐到逐层 TCP/registry 等待 |
| 异步 IPC profile（缩锁前） | `test/baseline/pap/results/runs/20260714_prefill_ipc_profile_async_fixed_c4` | 分离 registry queue wait 与 IPC open/install |
| 缩锁、同步 import | `test/baseline/pap/results/runs/20260714_registry_lock_narrow_sync_c4` | 单独验证主修复的因果收益 |
| 缩锁、异步但缺 readiness | `test/baseline/pap/results/runs/20260714_registry_lock_narrow_async_c4` | 输出分叉反例，禁止作为正确候选 |
| 缩锁、安全异步单次 | `test/baseline/pap/results/runs/20260714_registry_lock_narrow_async_ready_c4` | 完整-prefix readiness 正确性验证 |
| 缩锁、安全异步三次 | `test/baseline/pap/results/runs/20260714_registry_lock_narrow_async_ready_c4_reps3` | dirty-development 可重复性；三份 raw 通过，聚合器按设计拒绝 dirty formal |
| test bed 默认接线 C1 | `test/baseline/pap/results/runs/20260714_registry_lock_default_async_fingerprint_c1` | 未显式传 async/profile；验证默认值、implementation fingerprint、五轮 APC 和严格 audit |

四格所用的进程级 barrier 和 gate 都是默认关闭的诊断能力；当前开发候选在
`cbcbfabcb` 的 async decode-token + static MPS 基线上增加 registry 缩锁和安全异步
Prefill descriptor。本文不宣称 clean formal，最终冻结必须在提交后执行。
