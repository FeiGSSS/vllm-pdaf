# PAP/PD 1:1 长上下文多轮性能对比设计

日期：2026-07-11

状态：已完成讨论，待用户审阅书面规格

目标分支：`feature/pap`

## 1. 目标与核心问题

在相同 2-GPU 预算下，对比三种 1:1 分离式部署：

1. `PAP-native`：1PA1P，使用 PA 上的原生 APC 复用 Prompt 和 Decode KV；
2. `PD-oneway`：1P1D，只允许 P→D KV 传输；
3. `PD-bidir`：1P1D，允许下一轮 P 从 D 拉取上一轮 KV。

测试回答以下问题：

- PAP 相对当前单向 PD 的多轮 TTFT、会话延迟和容量优势从哪里开始出现；
- PD 双向 D→P KV 传输何时优于在 P 上重新计算 Decode 历史；
- PAP 相对当前 workload 下最优的 PD 模式是否仍有优势；
- 基础长文档、累计 Decode 历史、对话深度和并发如何改变上述结论；
- 在相同 GPU 数量下，各架构能承载多少 live context 和 active conversations。

短 Prompt 只用于协议与正确性 Gate。正式性能结论来自 4K、16K 和 32K
基础上下文，不再以 128-token Prefill 代表分离式部署的目标场景。

## 2. 已确认的设计决策

- 采用分阶段矩阵，不做一次性全因子扫描；
- 同时保留 PD 单向和 PD 双向，最终与 workload 下的 `best_PD` 比较；
- Qwen3-8B 是 Test 1 的唯一模型，使用本地模型文件；
- 使用 exact-token lane 做可归因的长度实验，使用 Chat lane 做真实多轮实验；
- 三种架构必须统一输出 local reuse、remote load 和 recompute token 分账；
- correctness/smoke 可以并行，正式性能是否并行由干扰 Gate 决定；
- 三个长期 sub-agent 分别负责三个架构 lane，主 agent 负责资源与证据仲裁；
- 不扫描 MPS；PAP 保持当前 70/30 Prefill/Attention 配置；
- 主矩阵 inter-turn think time 为 0，TTL/长思考时间留给后续独立实验。
- 不照搬短 Prompt PAP benchmark 的 concurrency、`max_num_seqs` 或 memory-utilization；
  长上下文按实际 KV token capacity 和安全余量分级 admission。

## 3. 非目标

Test 1 不包含：

- 多 PA cache-aware routing 或 Dynamo 集成；
- 任意 x:y 多轮性能；
- 其他模型、TP2 或跨节点部署；
- MPS 百分比、NIXL backend 或 GPU pair 的性能调参；
- partial-block attach、final-token closure 或跨 PA KV migration；
- 30 秒以上 think time、TTL 过期和缓存保持性扫描；
- 在同一实验提交中修改 PAP/PD 性能实现并宣称提升；
- 用 CUDA OOM 探测容量边界，或在 OOM 后临时降低单个 cell 的 workload 来制造可比较
  结果。

若实验暴露 correctness 缺陷，应先单独修复和验证，再用新 commit 重跑受影响的
完整 comparison group。

## 4. 三条架构线

| Lane | 拓扑 | Round 2+ 历史 KV 来源 | 主要成本 |
| --- | --- | --- | --- |
| `PAP-native` | 1PA1P | PA 原生 APC 中的 Prompt + Decode blocks | prefix attach、新 suffix Prefill、PA MPS 竞争 |
| `PD-oneway` | 1P1D | P 本地 APC 中原来由 P 计算的 blocks | Decode 历史重算、新 suffix Prefill、P→D transfer |
| `PD-bidir` | 1P1D | P 本地 APC + 从 D remote load 的 blocks | D→P remote load、新 suffix Prefill、随后 P→D transfer |

PD 两条 lane 使用同一个多轮 Proxy 实现和相同 conversation payload，只改变显式的
`bidirectional reuse` 开关。这样避免把不同 Proxy 调度、JSON 处理或 streaming 行为
误认为 KV 复用差异。

PD 服务使用非 deprecated 的 NIXL roles：Prefill 为 `kv_producer`，Decode 为
`kv_consumer`。`PD-bidir` 在两端显式设置：

```json
{
  "bidirectional_kv_xfer": true,
  "decoder_kv_blocks_ttl": 480,
  "kv_recompute_threshold": 64
}
```

`PD-oneway` 显式关闭 `bidirectional_kv_xfer`。主矩阵保留默认 recompute threshold；
只有当 Gate 无法证明 D→P 路径实际执行时，才增加 threshold 0 的诊断运行，诊断结果
不替代正式默认配置。

## 5. 公共硬件与服务合同

### 5.1 模型和 KV 规模

本地模型：`/data/ssd1/llm-models/Qwen3-8B`

模型配置为 36 层、32 Attention heads、8 KV heads、head dimension 128、原生最大
位置 40,960。FP16 KV 的理论大小约为：

```text
36 layers * 2(K/V) * 8 KV heads * 128 * 2 bytes
= 147456 bytes/token
= 144 KiB/token
```

| Context | 单份 KV 约占用 |
| ---: | ---: |
| 4K | 0.56 GiB |
| 16K | 2.25 GiB |
| 32K | 4.50 GiB |

容量矩阵必须按 context 分级，不能把短上下文的 8/16 active conversations 直接套到
32K。

### 5.2 固定配置

- GPU：NVIDIA L20，每条 lane 使用两张独占 GPU；
- TP=1；
- dtype=`float16`；
- `--enforce-eager`；
- `max_model_len=40960`；
- `max_num_batched_tokens=4096`，在 32K/C1 preflight 证明 8192 具有同等 OOM
  headroom 后才允许整组统一提升；
- `max_num_seqs` 按 context profile 固定为 4K:8、16K:4、32K:2；
- chunked prefill 开启；
- prefix caching 显式开启；
- KV block size=16；
- `temperature=0`、固定 seed；
- `ignore_eos=true`，输出长度固定；
- `VLLM_USE_FLASHINFER_SAMPLER=0`；
- 本地请求的 `NO_PROXY/no_proxy` 包含 `127.0.0.1,localhost`；
- 不访问 Hugging Face；
- PAP MPS 固定 70/30，不做扫描。

PAP 和 PD 可以使用不同的、已经验证过的 lane-specific memory-utilization，因为两种
架构的进程和权重布局不同；但每条 lane 只能在正式矩阵前做一次 32K/C1 preflight，
随后冻结配置。所有数值必须写入 effective config，不能按 cell 临时调整。

### 5.3 OOM 预防与 admission Gate

长上下文不能复用短负载的 scale knobs。每条 lane 启动后必须从服务日志或控制接口取得
实际 KV block/token capacity。对一个待运行 cell，先计算：

```text
required_live_tokens =
    active_conversations
    * max_rendered_context_tokens_per_conversation
```

只有满足以下条件才允许发请求：

```text
required_live_tokens <= 0.70 * reported_usable_kv_token_capacity
```

30% 余量用于 block fragmentation、PAP reserved decode slots、Prefill chunk workspace、
临时 tensors 和观测误差。若任一 lane 不满足，同一 comparison group 不启动；该 point
记录为 `admission-limited`，而不是通过实际 CUDA OOM 测量容量。

分级顺序固定为：

1. 4K/C1；
2. 16K/C1；
3. 32K/C1；
4. 对应 context 的下一 concurrency point；
5. 只有前一点的峰值 KV 使用仍低于 70% 才进入更高点。

32K/C4 需要单独把三条 lane 的 `max_num_seqs` 从 2 统一提升到 4，并重新运行
32K/C1 preflight；任一 lane 不满足 70% Gate 时，三条 lane 都不运行 C4。

运行中监控 peak allocated/free KV blocks 和 GPU memory。出现 `CUDA out of memory` 后
不得继续使用同一服务进程；保存失败证据、停止该 lane 的更高点并重新评估 admission
计算。OOM run 一律为 `invalid`，不提供 latency 或 throughput 结论。

### 5.4 Conversation 语义

本文的一个 `round` 指一次 user 请求和一次 assistant 回复。一个 conversation 内：

- 下一 round 只在上一 round 完整结束后发起；
- `conversation_id` 和 `cache_salt` 跨 round 保持不变；
- 不同 conversation 使用不同的 ID 和 salt；
- 不同 cell/repetition 使用新的 namespace，禁止跨实验命中；
- Chat lane 使用 `enable_thinking=true`，完整保留 assistant thinking/content；
- 性能 lane 使用 streaming 采集 TTFT/TPOT；
- correctness lane 使用 token-returning 非 streaming 请求做逐 token 对账。

`active conversations` 是 closed-loop 会话并发，不等同于独立 turn QPS。Test 1 不同时
引入 open-loop conversation arrival rate，以免把两个负载模型混为一谈。

## 6. Workload 生成

### 6.1 Exact-token 数据

exact lane 使用本地 tokenizer 和本地语料生成确定性 token IDs。每个 conversation 的
文档内容不同，但三种架构获得完全相同的 token IDs。禁止通过简单重复同一段文本制造
跨 conversation 的公共 prefix；cache salt 进一步隔离不同 conversation。

### 6.2 Chat 数据

Chat lane 使用本地长文本构造 system/user document context，并由 Qwen3 Chat Template
渲染。4K/16K/32K 指**渲染后的实际 Prompt token bucket**：

- 目标允许最多一个 block，即 16 tokens 的偏差；
- 每个请求保存服务端返回的真实 Prompt token IDs/digest；
- 同一 cell 的三种架构必须获得相同 token IDs；
- 若模板或 assistant history 造成跨 lane token discontinuity，cell 判 invalid。

## 7. 分阶段测试矩阵

### 7.1 Gate 0：短请求正确性

配置：128-token Prompt、48-token first Decode、两轮请求。

三种架构分别执行：

1. exact-token audit；
2. Qwen3 Chat Completions audit。

共 6 个 Gate cells。Gate 只证明缓存/传输/输出正确，不产生正式长上下文性能结论。

必须观察：

- `PAP-native` 命中至少一个 decode-derived block；
- `PD-oneway` 的 D→P remote load 为 0；
- `PD-bidir` 实际完成 D→P transfer；
- 三种架构逐轮输出 token IDs 一致；
- token 分账、PAP commit/lease/session 和 PD transfer lifecycle 全部闭合。

### 7.2 Matrix 1：基础 Context × Decode 历史

目的：定位 recompute、D→P transfer 和 PAP local attach 的 break-even。

| 变量 | 取值 |
| --- | --- |
| API | exact-token completion |
| 基础 Prompt | 4096 / 16384 / 32768 tokens |
| 第一轮真实 Decode | 128 / 512 / 2048 tokens |
| 第二轮新增 suffix | 128 tokens |
| 第二轮输出 | 128 tokens |
| active conversations | 1 |
| think time | 0 |

规模：

```text
3 architectures * 3 context lengths * 3 decode-history lengths
= 27 logical cells
```

每个 logical cell 包含独立的 Warm 和 Cold subrun。Cold 使用相同第二轮 token IDs、
新的 salt/conversation namespace 和 reset 后的 cache。Warm/Cold 不能共享会导致 32K
副本互相 eviction 的服务状态。

第一遍每 cell 跑 1 次用于 fail-fast；整组有效后补齐到 3 repetitions。

### 7.3 Matrix 2：真实长上下文 Chat

| 变量 | 取值 |
| --- | --- |
| API | Chat Completions |
| 渲染后基础 Prompt | 16K / 32K |
| rounds | 4 / 8 |
| 每轮新增 user tokens | 约 128，记录真实值 |
| 每轮 assistant output | 256 tokens |
| active conversations | 1 / 2 |
| think time | 0 |

规模：

```text
3 architectures * 2 contexts * 2 depths * 2 concurrencies
= 24 logical cells
```

每个 cell 先运行 32 conversations，正式运行 3 repetitions。主报告使用 per-round p50、
p90 和跨运行中位数；单 cell 样本不足时，p99 只能标记为 diagnostic。

完整 Warm/Cold Chat 归因只在以下 C1 profiles 运行：

- 16K / 4 rounds；
- 32K / 4 rounds。

其他 Chat cells 依靠 token 分账和 transfer counters 证明路径，不重复 Cold workload。

### 7.4 Matrix 3：容量补点

固定 4 rounds、每轮 user append 约 128、assistant output 256。Matrix 2 已包含
16K/32K 的 C1/C2，因此只补以下 points：

| 基础 Context | 新增 active-conversation points |
| ---: | --- |
| 4K | 1 / 4 / 8 |
| 16K | 4 |
| 32K | 4，仅在 70% KV-capacity admission Gate 通过后 |

mandatory 新增规模：

```text
3 architectures * (4K:C1/C4/C8 + 16K:C4)
= 12 logical cells
```

32K/C4 是 3 个 conditional cells。容量边界优先由 token-budget admission Gate 给出；
意外 OOM 只作为失败证据保留，其 latency/throughput 一律 `invalid`。

### 7.5 Tail 确认点

选定 `16K / 4 rounds / C4` 作为公共 tail profile。三种架构各获得 5 个完整
repetitions，每次 128 conversations。Matrix 3 中已有的 repetitions 计入 5 次，不重复
执行。

若 32K/C4 三条 lane 均通过，可另把 `32K / 4 rounds / C4` 作为 stress-tail 附录，
但不改变主报告的公共 profile。

### 7.6 实验规模

mandatory 正式矩阵包含：

```text
Matrix 1: 27 cells
Matrix 2: 24 cells
Matrix 3: 12 cells
Total:    63 architecture-specific cells
```

每个 cell 的 fail-fast rep1 是正式三次中的第一次，不额外计数。补齐后共有 189 个
formal repetitions。Matrix 1 的每个 repetition 还包含相互隔离的 Warm/Cold subruns。
Gate 0 有 6 个 smoke cells；32K/C4 最多增加 3 个 conditional cells；tail profile 把
已有 3 次补到 5 次，而不是重新增加 5 次。

## 8. Token 复用分账

每轮 Prompt token 必须进入三个互斥 bucket：

```text
local_reused_tokens
+ remote_loaded_tokens
+ recomputed_tokens
= prompt_tokens
```

若底层 connector 把 remote tokens 计入通用 cached tokens，adapter 必须先拆分，再输出
上述互斥字段，禁止重复计数。

定义：

```text
total_reuse_ratio =
    (local_reused_tokens + remote_loaded_tokens) / prompt_tokens

decode_history_reuse_ratio =
    reused_decode_origin_tokens / reusable_decode_history_tokens
```

`reusable_decode_history_tokens` 按实际 materialized token LCP、完整 block 对齐和
最后 sampled token 无 KV 的规则计算，不使用文本长度或 conversation ID 推断。

预期路径不是硬编码的性能结论，但 correctness Gate 要求：

- PAP 的 decode history 主要进入 local reuse；
- PD-oneway 的 decode-origin reuse 为 0；
- PD-bidir 在 transfer 被选择时产生 remote loaded tokens；
- recompute + reuse 必须覆盖真实 Prompt，不允许 silent gap。

## 9. 指标体系

### 9.1 客户端 per-round 指标

- TTFT mean/p50/p90/p99；
- TPOT mean/p50/p90/p99；
- ITL p90/p99；
- turn latency；
- input/output token 数；
- request success/failure；
- round index、conversation ID digest 和 cell ID。

TTFT 从 client send 到第一个非空输出 token。TPOT 使用实际 output token count 计算：

```text
TPOT = (turn_latency - TTFT) / max(output_tokens - 1, 1)
```

不能把 SSE chunk count 当成 token count。只有当服务能提供 token-level timestamps 或
确认一个 chunk 对应一个 token 时才报告 ITL；否则 ITL 为 `null + reason`。

Round 1 冷 Prefill与 Round 2+ Warm 指标必须分开，不能只给全轮平均。

### 9.2 Conversation 指标

- conversation completion latency；
- completed conversations/s；
- completed turns/s；
- output tokens/s；
- 每个 round 的存活 conversation 数；
- 最大 active requests/conversations；
- 因 OOM、TTL、eviction 或服务错误中止的 conversation 数。

### 9.3 服务端时间分解

三种 lane 输出统一字段：

```text
queue_wait_ms
local_cache_lookup_ms
d2p_remote_load_ms
prefill_compute_ms
p2d_transfer_ms
decode_start_wait_ms
```

无法直接观测的字段必须是 `null + reason`，不能伪造 0。

### 9.4 Cache、transfer 和资源指标

- local APC hit blocks/tokens；
- remote loaded blocks/tokens；
- recomputed tokens；
- D→P 和 P→D transfer bytes、count、duration、failure；
- NIXL effective bandwidth；
- Prefill chunk 数量和每 chunk tokens/time；
- APC eviction 和 cache occupancy；
- PAP decode commit count/ACK/queue drain；
- PAP lease release、Attention session drain；
- GPU memory used/free、GPU utilization；
- CPU utilization、host memory 和必要时的 PCIe counters；
- actual GPU pair 和 topology snapshot。
- reported usable KV blocks/tokens、required-live-token estimate 和 70% Gate decision。

## 10. 正确性和有效性 Gate

一个 formal cell 必须满足：

- 所有预期请求和 conversations 完成；
- 三种架构在同输入下逐轮输出 token IDs 一致；
- 要求 paired Cold 的 Matrix 1 和选定 Chat profiles 中，Warm/Cold 输出一致；
- 本地 tokenizer 与服务端 Prompt token IDs 一致；
- token reuse 分账精确闭合；
- PD-oneway D→P load 为 0；
- PD-bidir 记录实际 transfer 或明确记录 threshold recompute decision；
- PAP 所有 commit ACK、lease release 和 session drain 通过；
- PD transfer lifecycle 无 timeout、stale block 或 request mismatch；
- 无隐藏 server traceback、EngineDeadError 或 correctness audit match；
- tracked worktree clean；
- model、workload、GPU、ports、proxy bypass 和 effective config 完整记录。

容量 cell 应在 OOM 前由 admission Gate 拒绝。若仍出现 OOM，该 run 为 `invalid`，不能
被引用为 latency、throughput 或正常容量结果。
如果输出 token 不一致、KV 分账不闭合或 remote blocks 错配，则是 correctness failure，
整个 comparison group 停止，先修复再重跑。

## 11. 比较与决策方法

每个 workload 同时报告：

```text
PAP-native / PD-oneway
PAP-native / PD-bidir
PD-bidir / PD-oneway
```

对于每个 comparison group 和 primary metric，`best_PD` 按指标方向选择：

```text
latency/TTFT/TPOT: best_PD = min(PD-oneway, PD-bidir)
throughput/capacity: best_PD = max(PD-oneway, PD-bidir)
```

主结论报告：

- PAP Round 2+ TTFT / best_PD；
- PAP turn/conversation latency / best_PD；
- PAP conversation throughput / best_PD；
- PAP maximum admitted live-context tokens / best_PD；
- PD-bidir 相对 PD-oneway 的 D→P transfer break-even；
- PAP 相对 PD 两种模式的优势区域，而不是只给一个全局平均。

一次差异只有同时满足以下条件才称为稳定收益：

- 三个 repetitions 方向一致；
- 跨运行中位数差异至少 5%；
- correctness、分账和 lifecycle Gate 全部通过；
- 不是 GPU pair、parallel interference、OOM 或 eviction 差异造成；
- 失败率、TPOT、显存容量等回退被显式披露。

低于 5% 的差异标记为 parity/noise band。单次最好结果不用于决策。

## 12. 并行干扰 Gate 与 GPU 轮换

### 12.1 当前能力

协作环境最多同时运行 4 个 agent，包括主 agent，因此可配置主 agent + 三个长期 lane
agents。GPU 空闲状态是运行时事实，不能写死；每批实验前重新检查 GPU、进程和端口。

### 12.2 干扰 Gate

选定 `16K / 4 rounds / C2` 代表 cell：

1. 三种架构分别单独运行；
2. 三种架构使用不重叠 GPU pairs 同时运行；
3. 比较 Round 2+ TTFT、TPOT、conversation throughput、transfer time 和 CPU/PCIe 状态。

只有所有 primary metrics 的 parallel-vs-solo 偏差绝对值不超过 2%，且没有共享资源
饱和或错误，正式性能才允许三路并行。否则 formal runs 串行执行；sub-agents 仍可并行
准备配置、审计上一 cell 和整理结果。

### 12.3 GPU pair 轮换

候选三轮映射采用 Latin square：

| Rep | PAP-native | PD-oneway | PD-bidir |
| --- | --- | --- | --- |
| 1 | pair A | pair B | pair C |
| 2 | pair B | pair C | pair A |
| 3 | pair C | pair A | pair B |

运行前记录 `nvidia-smi topo -m`，选择拓扑可比较的 pair。不能触碰无关用户/服务占用的
GPU，也不能杀死不属于当前 run process group 的进程。

## 13. Sub-agent 编排

### 13.1 主 agent

- 冻结 tested commit 和 manifest；
- 检查 GPU/port/process/proxy；
- 分配 lane、GPU pair、端口和 run root；
- 决定 parallel Gate；
- 审核 invalid/failure 分类；
- 汇总三条 lane，产出最终报告和实验账本更新。

### 13.2 三个 lane agents

每个 agent 长期负责一个架构，不按单次 run 反复创建 agent。它只能：

- 启动和停止自己 process group；
- 写入自己的独立 run root；
- 运行 manifest 分配的 cells；
- 完成结果完整性、correctness、cache/transfer 和 drain audit；
- 生成标准化 cell summary；
- 将异常报告给主 agent，不擅自修改共同代码或降低 workload。

代码实现与 formal run 不并行。只有 tracked code 冻结并提交后，lane agents 才进入
正式性能阶段。

当前 sub-agent 接口不提供 per-agent AI model 选择；三个 agents 使用相同能力配置。
被测 serving model 也固定为 Qwen3-8B。

## 14. 执行顺序与停止条件

1. 实现统一 client、三条 self-contained runner、观测 adapter 和 aggregator；
2. 单元测试 token accounting、Prompt/LCP、manifest 和 summary schema；
3. 提交代码并要求 tracked worktree clean；
4. 每条 lane 完成 32K/C1 memory preflight 并冻结配置；
5. 保存三条 lane 的 KV capacity，并验证所有 manifest cells 的 70% admission decision；
6. 执行 Gate 0；
7. 执行 parallel interference Gate；
8. Matrix 1 全部 cell 先跑 rep1，审计后补 rep2/rep3；
9. Matrix 2 同样按 rep1→审计→补齐执行；
10. Matrix 3 从低 context/concurrency 向高点递增，每一点重新检查 peak headroom；
11. 补齐 tail repetitions；
12. 生成三路 comparison tables、break-even 图和容量图；
13. 更新 PAP 实验历史索引并提交结果摘要。

停止条件：

- Gate 0 任一 correctness failure；
- 同 comparison group 输出不一致；
- token 分账不闭合；
- PD remote block/token identity 错配；
- PAP commit/lease/session 不闭合；
- runner 发现非本 run 进程/端口冲突；
- 代码或配置在三条 lane 之间发生未记录变化。

意外 OOM 立即销毁该服务进程，停止对应 context 的更高 concurrency points，并把
admission estimate 缺陷作为待修问题；它不抹去此前独立、已通过的低点。

## 15. Artifact 与目录合同

建议外部 raw root：

```text
/home/fei/research/PD/test/baseline/multiturn_pd_pap/results/runs
```

稳定 cell ID：

```text
MT-<matrix>-<lane>-ctx<context>-d<decode>-r<rounds>-c<active>-rep<N>
```

每个 run directory 至少包含：

```text
run_metadata.json
effective_config.env
git_status.txt
tracked_worktree.patch
topology_snapshot.txt
conversation_metrics.jsonl
round_summary.json
cache_accounting.json
transfer_summary.json
resource_samples.csv
capacity_admission.json
correctness_audit.env
lifecycle_audit.env
service_logs/
```

`run_metadata.json` 记录 tested commit、lane、cell ID、GPU pair、model、dtype、完整
workload、proxy mode、APC/bidirectional flags、MPS、memory utilization、parallel/solo 和
agent lane。缺失任何关键 provenance 的 run 只能作为 diagnostic。

Git 追踪：

- 本设计、实施计划、manifest/schema、runner/client/aggregator 和最终摘要；
- 不把 raw logs、profiles 或大 JSONL 结果提交进 Git；
- 最终摘要把 logical cell 映射到 raw run roots，并更新 PAP 实验历史索引。

## 16. 最终报告结构

报告至少包含：

1. correctness 与 token-accounting Gate；
2. Round 1 cold TTFT 表；
3. Round 2+ TTFT、TPOT 和 reuse-source 表；
4. Context × Decode-history break-even heatmap；
5. 4/16/32K conversation throughput 和容量曲线；
6. PAP vs PD-oneway、PD-bidir 和 best_PD 比率；
7. D→P/P→D bytes、time 和 effective bandwidth；
8. memory/OOM/eviction 边界；
9. p90/p99 tail profile；
10. invalid、failed、inconclusive 和环境干扰记录；
11. 当前结论、适用区间和下一步实验。

## 17. 参考实现与现有证据

- [PAP 多轮原生 APC 设计](../../design/pap-xpayp-multiturn-kv-affinity-20260710.md)；
- [PAP 实验历史索引](../../design/pap-experiment-history-index.md)；
- [NixlConnector 双向 KV 使用指南](../../features/nixl_connector_usage.md)；
- `examples/pap/pap_multiturn_prefix_cache.py`；
- `examples/pap/pap_multiturn_chat_prefix_cache.py`；
- `examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py`；
- `benchmarks/multi_turn/benchmark_serving_multi_turn.py`；
- `.codex/skills/vllm-pap-benchmark/`。

现有 128/48 exact 和 Chat clean results 只作为 Gate 先验：它们证明 PAP 能命中
decode-derived blocks，但尚未构成 PAP 与两种 PD 在长上下文多轮负载下的性能矩阵。
