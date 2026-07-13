# PAP 开发与实验历史索引

更新日期：2026-07-13

覆盖分支：`feature/pap`

覆盖时间：2026-05-22 至 2026-07-13

用途：从时间、模块或指标出发，逐层回溯 PAP 的设计动机、实现、实验和决策。

## 0. 如何使用本索引

本文件是 PAP 历史的导航和决策层，不替代已有设计文档、Git diff 或原始实验日志。
推荐按问题选择入口：

| 想回答的问题 | 第一入口 | 下一层 |
| --- | --- | --- |
| 某一天到某一天发生了什么？ | [全局时间线](#3-全局时间线) | 阶段对应的模块档案和提交 |
| 某个模块为什么存在？ | [模块地图](#4-模块地图) | [模块档案](#5-模块档案)中的动机、机制和边界 |
| 某个 TPOT/TTFT 数字来自哪里？ | [实验账本](#6-实验账本) | workload、commit、run 目录和 audit |
| 某个优化为什么没有继续？ | [负结果登记](#7-负结果回滚与被替代路线) | rejecting A/B、回滚开关和替代方案 |
| 当前哪些能力已经闭合？ | [当前状态](#1-当前状态与证据边界) | 正式 clean baseline 和未完成边界 |
| 如何追加新实验？ | [记录模板](#10-新增实验记录模板) | 复制模板并同步更新模块和时间线 |

本索引采用三层渐进披露：

1. **阶段层**回答“为什么改变方向”；
2. **模块层**回答“如何设计和实现”；
3. **证据层**回答“哪个 commit、哪个 A/B、哪个原始目录证明了结论”。

结论标签统一为：`接受`、`保留为可选实验`、`拒绝`、`回滚`、`被替代`、
`无结论`。证据等级统一见[第 2 节](#2-路径与存储类别)。

## 1. 当前状态与证据边界

### 1.1 当前接受的架构

- Prefill 进程拥有 prompt 和 decode 的物理 paged KV blocks；
- Projection 采用 KV-unaware 调度，不持有历史 KV；
- Attention 通过 CUDA IPC 访问 Prefill-owned KV，并把 decode K/V 直接追加到这些
  blocks；
- Projection 与 Attention 的同机 decode 热路径采用 local-fast/slot-plan 等复用机制；
- Proxy、连接层和控制面支持任意正整数 `xPAyP`；
- 执行层已经验证 2PA2P full crossbar、same-layer combine/scatter、vectorized route
  copy 和 active-source membership；
- 1PA1P 多轮请求已经通过 vLLM 原生 APC 复用第一轮 prompt 和 decode 完整块；
- 多 PA 多轮的 cache-aware routing 尚未在当前 Proxy 中实现，后续交给 Dynamo 等外部
  路由框架。

### 1.2 当前正式证据摘要

| 能力 | 正式/最高等级结果 | 结论 |
| --- | --- | --- |
| 1PA1P、QPS 4 | PAP median TPOT `28.06 ms`，PD `24.48 ms`，PAP/PD `1.146x` | 非饱和区 TPOT 接近 PD，TTFT 中位数基本持平 |
| 2PA2P full crossbar | clean 三轮 median TPOT `40.49 ms`、p99 TPOT `50.03 ms` | 相对 PD 分别 `1.654x/1.942x`，均低于 `2x` |
| 任意 x:y correctness | 1PA1P、1PA2P、2PA1P、3PA2P active-peer smoke | pair、membership、routing、correctness、drain 全部通过 |
| 多轮 exact-token | expected/actual hit `160/160`，decode-derived `32`，cold `0` | 第一轮 decode KV 可被第二轮原生 APC 命中 |
| 多轮 Qwen3 Chat | expected/actual hit `176/176`，decode-derived `48`，cold `0` | thinking 模板下真实 messages 保持完整 materialized LCP |
| 1PA1P 16K 两轮历史优化 reference | Round2 old-pull PD/PAP TTFT `267.27/224.49 ms`，TPOT `25.18/30.45 ms` | PAP Stage C/D 内部优化证据保留；旧 PD pull 已被 P10 校正，不再作公平 PD 基线 |
| 1PA1P 16K 五轮 C4 正式矩阵 | corrected-push PD/PAP 稳态 TTFT `306.17/248.32 ms`，TPOT `42.12/51.38 ms` | PAP 稳态 TTFT 为 PD `0.811x`，TPOT 为 `1.220x`；两侧各三次、实际并发 4、严格 Gate 全通过 |

以上各数字的工作负载和原始证据分别记录在 `M6`、`M9`、`M10`、P10 和实验账本中，
不能脱离 workload 直接互相比较。

面向阶段汇报的聚合结论见
[PAP 1:1 多轮性能优化阶段总结](pap-1pa1p-multiturn-stage-report-20260712.md)。

### 1.3 证据边界

- 正式 clean、受控 dirty A/B、trace 诊断、短 smoke 和历史记录不是同一证据等级；
- trace 会改变执行时序，trace TPOT 不得冒充正常性能；
- HTTP 请求完成不等于正确性通过，必须同时检查 token、routing、decode commit、lease
  release 和 session drain；
- 旧实验分布在两个 PAP result roots；路径存在不代表它受 Git 保护；
- 本索引无法恢复从未写入文件的终端输出；
- 记录为 `missing` 的历史路径仍保留引用，避免静默抹去一次已做过的探索。

## 2. 路径与存储类别

### 2.1 符号路径

| 符号 | 当前展开 | 内容 |
| --- | --- | --- |
| `$PAP_REPO` | `/home/fei/research/PD/vllm-pap` | 源码、Git-tracked 文档、repo-local 结果 |
| `$PAP_RESULTS` | `/home/fei/research/PD/test/baseline/pap/results/runs` | 当前标准 PAP raw runs |
| `$PAP_REPO_RESULTS` | `$PAP_REPO/test/baseline/pap/results/runs` | 早期和 2026-07-11 多对多 repo-local runs |
| `$PD_RESULTS` | `/home/fei/research/PD/test/baseline/nixl_disaggregated/results/runs` | PD/NIXL 对照结果 |
| `$PAP_PROFILES` | `$PAP_REPO/profile_output` | profiler、trace summary 和诊断报告 |
| `$PAP_HANDOFF` | `/tmp/pap-handoff-20260707.md` | 临时 handoff，易随 `/tmp` 清理丢失 |

### 2.2 存储类别

| 标签 | 含义 | 能否随 Git clone 获得 |
| --- | --- | --- |
| `tracked` | Git 管理的源码、设计、计划或小型摘要 | 是 |
| `repo-untracked` | 位于仓库目录但没有进入 Git 的 raw result/profile | 否 |
| `external` | 仓库之外的标准结果目录 | 否 |
| `temporary` | `/tmp` 等临时路径 | 否，且最易丢失 |
| `missing` | 文档引用但 2026-07-11 清点未找到 | 否 |

### 2.3 证据等级

| 等级 | 要求 | 允许的结论 |
| --- | --- | --- |
| `formal-clean` | 已提交代码、tracked clean、受控 workload、通常多轮、strict audit | 可作为正式 baseline 或启用依据 |
| `controlled` | 同代码 A/B，只改变一个主变量；可带 tracked patch | 可做归因，但要注明 dirty/provenance |
| `diagnostic` | trace、profile、单请求或特殊低负载 | 只解释瓶颈，不作正常性能基线 |
| `smoke` | 短输出、小请求数、拓扑/协议覆盖 | 只证明功能和 contract |
| `historical` | 已有记录但当前 metadata 不完整 | 可重建方向，不夸大精度 |
| `invalid` | OOM、0 完成、正确性失败、配置错误或方法学不成立 | 只记录失败原因，禁止引用性能值 |

## 3. 全局时间线

| 阶段 | 时间 | 核心问题 | 关键提交 | 阶段结论 | 证据 |
| --- | --- | --- | --- | --- | --- |
| `P1` 原型与 KV ownership | 05-22 至 05-25 | PAP 是否能真正拆开 Projection/Attention，并让 Projection 不拥有 KV？ | `e3901b7a2`、`5d10f3e89`、`d5464a6b7`、`312ae6fbb`、`a62895ef7` | true-split、KV-unaware Projection、CUDA IPC 和 Prefill-owned KV 路线成立；conversation placement 绑定随后移除 | `historical/smoke`，见 M1/M3/M4 |
| `P2` Mailbox 与 scale-out | 05-26 | NIXL mailbox 能否替代旧触发链路，4PA4P/6PA2P 是否随节点扩展？ | `998965825`、`ec99da84c`、`937b56049`、`6e912dbbc` | Mailbox 功能闭合，但 4PA4P/6PA2P 暴露逐层交替和 PA/Projection 资源分配瓶颈；3-way 没有改善该 workload | `historical/controlled`，见 M2/M6 |
| `P3` MoE、wavefront 与 TP | 05-27 至 05-28 | 大模型、大 batch、wavefront 和更高并发能否释放 Projection 算力？ | `a2fb19376`、`7009a1cac`、`a500ba08a`、`821f21cb6`、`7b1e04b1e`、`f7052fb84` | 6PA2P 是 30B workload 中最好 PAP 点，但仍远落后 PD；更高 concurrency 和更多 P 不单调；TP2 多 PA 稳定性得到修复 | `controlled/invalid`，见 M2/M6/M8 |
| `P4` 方法学与热路径拆解 | 06-30 至 07-01 | 性能差距来自 raw NIXL、路由、Python 控制还是 Attention 计算？ | `4a7737647`、`e826128d1`、`a017bc7d4`、`401ce1425`、`61848acf5` | 固化 PD/PAP 同口径方法；逐层 Projection/Attention alternation 和重复控制开销远大于单次 raw transfer | `controlled/diagnostic`，见 M2/M6/M7 |
| `P5` Local-fast 与统一 KV | 07-02 至 07-07 | 能否去掉旧 NCCL/ubatch/local KV 分叉，并把 decode KV 直接写入 Prefill blocks？ | `e89346dc8`、`014f33b3d`、`169066c78`、`214dff673`、`960d3ab7d`、`24361dd67` | local-fast 与 unified KV/lease/remote append 成为单一主路径；batched slot mapping 明显优于 per-row；decode commit 进入正确性主线 | `controlled/smoke`，见 M4/M5/M7 |
| `P6` 正确性加固与 TPOT 收敛 | 07-10 | ACK、lease、async send 正确性闭合后，same-node 热路径能否把 TPOT 降到 `2x PD` 内？ | `86a7c1273`、`72b0c1598`、`87bb1061f` | slot-plan/doorbell/metadata 复用使 QPS16 median TPOT 到 `33.21 ms`；QPS4 PAP/PD 为 `28.06/24.48 ms` | `formal-clean/controlled`，见 M5/M6/M7 |
| `P7` 任意 x:y 与多对多执行 | 07-10 至 07-11 | 连接层任意 x:y 能否升级为 2PA2P full-crossbar 的真正 combine/scatter 执行？ | `45c302bb3`、`d654f6011`、`12b689d1b`、`bdb7a7dc7`、`54bd1a59c`、`d8bce2e6c` | central dispatcher、same-layer combine、vectorized route copy 和 active peer 闭合；2PA2P clean median/p99 TPOT 都低于 `2x PD` | `formal-clean/controlled/smoke`，见 M8/M9 |
| `P8` 多轮原生 APC | 07-11 | 第二轮能否在同一 PA 直接命中第一轮 prompt 和 decode KV，而不做 KV 回传？ | `6a7094c3b`、`fd723d2e2`、`c71ccc9df`、`043339691`、`558db3cdd`、`848f321ab` | exact-token 命中 2 个 decode blocks，真实 Chat 命中 3 个；pair 每轮解散，APC LRU 保存完整 hashed blocks | `formal-clean`，见 M10 |
| `P9` 多轮北极星与 metadata | 07-12 | 固定 16K 多轮后，metadata 构造、chunk generation 和 cache-hit block 扫描如何影响 TPOT？ | `7e81e2d10`、`7d0fd13cb`、`6bc383dab`、`c134bc3d9`、`0727ed946` | bulk build 将 Round2 TPOT `39.13 -> 30.59 ms`；generation-aware slot-plan 将 Round1 降到 `30.52 ms`；topology-token fast key 将全扫描减少 `36x`，两轮约 `1.21x PD` | `formal-clean/controlled`，见 M6/M7/M10 |
| `P10` PD push 校正与五轮负载 | 07-13 | 同机 PD 为何只有约 0.42 GiB/s；校正后长上下文、多轮、C4 下 PAP/PD 的真实差距是多少？ | `131e1dfa2`、`e8ab4ab23`、`340c11abc`、`a646ae032` | 旧 pull GET 命中 TCP emulation；官方 push PUT 单流约 24.5 GiB/s；C4 formal 稳态 PAP/PD TTFT `0.811x`、TPOT `1.220x` | `diagnostic/formal-clean`，见 M6/M10 和 P10 报告 |

时间线不是 commit 全表。更细的里程碑见[第 8 节](#8-关键提交时间线)，完整 patch 以
Git 历史为准。

## 4. 模块地图

| 模块 | 主要职责 | 上游依赖 | 下游影响 | 当前状态 |
| --- | --- | --- | --- | --- |
| `M1` PAP 分离架构与控制流 | 定义 PA、Projection、Attention 角色和请求生命周期 | vLLM scheduler/model runner | 所有 transport、KV、x:y 和多轮能力 | 接受 |
| `M2` OFFLOAD_EXEC 与 NIXL Mailbox | QKV/Attention output 的跨进程数据与通知 | M1 | M7 热路径、M9 多来源聚合 | Mailbox 保留为 NIXL 能力；same-node 默认走更轻路径 |
| `M3` Projection KV-unaware 调度 | Projection 只做权重相关计算，不分配历史 KV | M1 | M4 KV ownership、M8/M9 调度 | 接受 |
| `M4` Prefill-owned Shared/Unified KV | prompt/decode 共享一套 Prefill paged blocks | M1/M3 | M5 commit/lease、M10 多轮 | 接受 |
| `M5` Decode Commit、Lease 与正确性闭环 | 远端写入后的 token/hash/cache 状态一致性 | M4 | APC、多轮、可靠释放 | 接受，fail-closed |
| `M6` Benchmark、Tracing 与审计 | 同口径 workload、Git/config provenance、严格 audit | 全模块 | 所有性能和正确性结论 | 接受 |
| `M7` Same-node Local-fast 与 TPOT 热路径 | 降低逐层 CPU、分配、slot 和 metadata 开销 | M2/M4/M6 | 1PA1P/多对多 TPOT | 接受；更深 GPU-only timeline 未实现 |
| `M8` 任意 x:y 拓扑与路由 | 任意 PA/P 数量、pair 选择和 full-crossbar 覆盖 | M1/M3 | M9 执行聚合、多 PA 路由 | 连接层接受；多轮 affinity 外置 |
| `M9` 多对多 Cohort/Combine/Scatter | 多来源 ingress 聚合、一次 Attention、结果 scatter | M7/M8 | 2PA2P 性能和任意 x:y correctness | 2PA2P 正式闭合，其他 x:y 为 smoke |
| `M10` 多轮原生 Prefix Cache | 复用 Prefill-owned prompt/decode hashed blocks | M4/M5/M6 | 多轮 TTFT/Prefill、未来 Dynamo | 1PA1P 闭合；多 PA cache-aware routing 未实现 |

## 5. 模块档案

### M1. PAP 分离架构与控制流

#### 问题与动机

PD 把 Prefill 和 Decode 按请求阶段拆到不同实例；PAP 进一步把 decode layer 内的
Projection 与 Attention 拆开，希望 Projection 聚合较大 batch 的权重相关 GEMM，而 PA
节点负责 Prefill 和 KV-dependent Attention。最初需要证明的不是性能，而是三件事：

1. Projection 可以在不知道 prompt KV 的情况下继续完整 transformer forward；
2. Attention 能定位正确请求、layer、step 和 KV owner；
3. 请求结束时 PA/P pair、Attention session 和 KV lease 可以独立解除。

#### 设计与机制

一次请求先在 PA 的 Prefill 进程生成首 token 和 KV 描述符，Proxy 再把 request-scoped
Attention endpoint、remote prefix length 和 mailbox/transport 元数据交给 Projection。
Projection 对每层计算 Q/K/V 和后续线性层；Attention 接收 Q/K/V，访问 Prefill-owned
KV，返回 Attention output。绑定是**一轮请求级**，不是 conversation 级永久绑定。

#### 关键实现提交

- `e3901b7a2` Add PAP NIXL prototype experiment；
- `5d10f3e89` Add PAP true-split attention prototype；
- `f38b24351` Parameterize PAP benchmark topology；
- `5dc263af0` 移除旧 NCCL P2P transport classes，并引入 OffloadComm/Agent；
- `fd7b00453` Implement PAP attention ready flow；
- `8f84bbf61` Clean PAP stage-one branch；
- `45c302bb3` Support arbitrary PAP x:y topologies。

#### 关键实验与证据

- `PAP-20260522-PROTO-NIXL`：原型证明 true split 和基本请求链路可运行；
- `PAP-20260524-PROJECTION-KVUNAWARE`：Projection 不拥有 prompt KV 仍可调度；
- `PAP-20260526-MAILBOX`：NIXL mailbox 成为完整 OFFLOAD_EXEC checkpoint；
- `PAP-20260710-ARBITRARY-XY`：连接和 Proxy 层支持任意正整数 x:y；
- `PAP-20260711-ACTIVE-PEER`：请求 cohort 生命周期能驱动 active source membership。

#### 负结果与被替代方案

- 原型 TCP trigger 和 NCCL performance path 只保留了阶段性证据，后续被 mailbox 和
  same-node local-fast 路径替代；
- 早期 conversation placement/handle reuse 提交 `7ae51f27b`、`400eae351`、
  `349b890fe` 没有成为当前跨轮架构，`c12e99edc` 移除了 placement binding；
- persistent resident session 方案在 2026-07-11 被 native APC 方案替代，原因见 M10。

#### 当前结论与边界

PAP 角色边界已经稳定：PA 持有 KV，Projection 持有权重相关计算，Attention 是 PA 侧
KV-dependent 服务。当前执行层正式性能闭合到 2PA2P；更高 x:y 主要有 correctness
smoke。多轮每轮重新选 pair，多 PA conversation affinity 不是 Proxy 当前职责。

#### 深入阅读与原始证据

- [NIXL mailbox handoff](pap-nixl-mailbox-handoff.md)（`tracked`）；
- [多对多 Cohort 调度设计](pap-many-to-many-cohort-scheduler-20260711.md)（`tracked`）；
- [多轮原生 APC 设计](pap-xpayp-multiturn-kv-affinity-20260710.md)（`tracked`）；
- `$PAP_RESULTS/202605*` 和 `$PAP_REPO_RESULTS/20260710_xpayp_*`
  （`external/repo-untracked`）。

### M2. OFFLOAD_EXEC 与 NIXL Mailbox

#### 问题与动机

PAP 每个 decode layer 都经过 Projection→Attention→Projection。如果每层都建立动态
buffer、传长控制消息、等待线程唤醒和 ACK，即使 raw P2P/NIXL READ 很快，36 层累计
开销仍会主导 TPOT。Mailbox 的目标是把 buffer 注册、地址和路由元数据移到启动/bind
阶段，让热路径只传固定 slot 和小通知。

#### 设计与机制

Mailbox actor 预注册 send/receive buffers；producer 发布携带 slot、shape、dtype、layer、
request row 信息的通知，receiver 发起 NIXL READ 或读取同机 slot，消费后 ACK/release。
slot protocol、cached dlist、msgpack、zero-copy receive 和有限 send/receive leases 避免
覆盖活跃 tensor。跨节点保留 NIXL；同节点后续演进为 local-fast/IPC slot 路径。

#### 关键实现提交

- `998965825` Add PAP NIXL mailbox handoff checkpoint；
- `35a19f14c` Fix PAP mailbox routing across attention peers；
- `c505ed6fd` Optimize PAP QKV mailbox receive path；
- `2facd88ac` Piggyback PAP mailbox receive-slot releases；
- `c77dc7342` Cache PAP push-write xfer handles；
- `014f33b3d` Cut NIXL roundtrip via push-write result/direct QKV send；
- `e826128d1` Remove PAP NCCL offload transport；
- `883a8969d` Remove obsolete PAP NIXL projection receive path。

#### 关键实验与证据

- `PAP-20260526-MAILBOX`：第一版功能 checkpoint 和多种 mailbox A/B；
- `PAP-20260526-3WAY`：3-way runner pipeline 对选定 workload 无明显收益；
- `PAP-20260701-MAILBOX-HOTPATH`：direct slot、ACK piggyback、handle cache 等热路径实验；
- `PAP-20260702-REMOTE-TRACE`：证明主要开销不是单次 raw NIXL READ，而是逐层往返、
  Python/线程和调度交替。

#### 负结果与被替代方案

- async send slots、piggyback ACK、inline poll/publish 等单点微优化在受控 A/B 中多次
  退化，未作为默认；
- Q-first/KV-later、Q-first Projection 和 Attention partial overlap 虽功能可行，但首轮
  CUDA A/B 退化，保留为关闭的研究路径；
- mailbox slot count 2 没有改善串行 decode workload；
- 旧 TCP trigger/NCCL offload 已被删除，不应作为当前实现选项。

#### 当前结论与边界

NIXL mailbox 仍是跨进程/可扩展 transport checkpoint，但同机优化重点已转向固定 slot、
CUDA IPC/event 和减少逐层 CPU 控制。继续调整单个 mailbox sleep/poll 参数不再是首选；
若未来跨节点，应重新验证当前 NIXL backend 和 mailbox contract。

#### 深入阅读与原始证据

- [PAP NIXL Mailbox Handoff](pap-nixl-mailbox-handoff.md)（`tracked`）；
- [NIXL/NVLink 优化 idea book](pap-nixl-nvlink-optimization-idea-book-20260707.md)
  （`tracked`）；
- [3-way microbatch 设计与结果](pap-runner-3way-microbatch.md)（`tracked`）；
- `$PAP_RESULTS/20260525_*`、`$PAP_RESULTS/20260526_*`（`external`）。

### M3. Projection KV-unaware 调度

#### 问题与动机

Projection 需要知道 remote prefix 进度以保持 transformer layer 顺序，却不需要读取
prompt/decode KV。如果仍按普通 vLLM request 分配本地 KV blocks 或启动 KVConnector，
会浪费显存、错误参与 prefix cache，并把 PA ownership 语义重新混回 Projection。

#### 设计与机制

请求通过 `pap_projection_kv_unaware` 和 `pap_remote_prefix_len` 声明 remote progress。
Scheduler 用显式 PAP Projection state 把 remote computed tokens 与 local slot ownership
分开：可以推进 request 的逻辑 computed position，但不为 remote prefix 分配本地 blocks。
Model runner 使用 metadata-only KV placeholders；Attention 从 PA-owned KV 服务请求。

#### 关键实现提交

- `d5464a6b7` Implement PAP KV-unaware projection admission；
- `cd8207e67` Run PAP Projection without KVConnector；
- `45b93fcdf` Validate metadata-only PAP Projection X:Y routing；
- `0fb0c8bc3` Avoid remote prefix block allocation；
- `0232ca462` Offset Projection running local slots；
- `fa5c6e316` Disable Projection request KV slot allocation；
- `a98f4bd1a` Add explicit Projection scheduler state；
- `0262f8d3f` Make Projection startup KV metadata-only；
- `558db3cdd` Skip local cache registration for PAP Projection。

#### 关键实验与证据

- `PAP-20260524-PROJECTION-KVUNAWARE`：metadata-only X:Y routing 和 admission；
- `PAP-20260710-ARBITRARY-XY`：任意 topology 下 Projection remote prefix contract；
- `PAP-20260711-MULTITURN-EXACT`：Projection 可以是任意健康实例，历史 KV 仍在 PA；
- 2026-07-11 exact clean rep1 暴露 zero-local-block cache registration 错误，修复后 rep2
  通过。

#### 负结果与被替代方案

- Projection 通过 KVConnector 获得 prompt KV 的路线被删除；
- 普通 scheduler block allocation 会造成不必要显存占用，已由 explicit PAP state 替代；
- `SingleTypeKVCacheManager` 的 fail-closed 分配检查曾把 Projection 的合法 0 block 误判为
  错误；根因不是检查过严，而是 Projection 不应进入本地 cache registration。

#### 当前结论与边界

Projection 的 zero-local-KV 是架构不变量，不是临时内存优化。任何未来 scheduler 重构
必须同时保留逻辑 remote progress 和物理 local ownership 的分离；普通非 PAP request
仍按 vLLM 原语分配 KV。

#### 深入阅读与原始证据

- [多对多 scheduler 源码审计](pap-many-to-many-cohort-scheduler-20260711.md)
  （`tracked`）；
- [多轮 APC 设计和 Projection 边界](pap-xpayp-multiturn-kv-affinity-20260710.md)
  （`tracked`）；
- `$PAP_RESULTS/20260711_043339691_multiturn_clean_rep1`（`external`，失败证据）；
- `$PAP_RESULTS/20260711_558db3cdd_multiturn_clean_rep2`（`external`，通过）。

### M4. Prefill-owned Shared/Unified KV

#### 问题与动机

如果 Attention 复制 prompt KV 到自己的 local pool，decode KV 会与 Prefill cache 分叉，
既增加显存和 copy，也无法让后续请求通过 Prefill APC 命中 decode history。目标是只有
一套物理 KV：Prefill 分配和拥有，Attention 通过 CUDA IPC 直接写入。

#### 设计与机制

Prefill Scheduler 为 prompt 和计划 decode capacity 正式分配 paged blocks，并把 block
IDs、block size、layout、IPC handles 和 writable range 发布给 Attention。Attention 对当前
token 执行 `reshape_and_cache_flash`，把 K/V 写入同一 block table，再做 paged
FlashAttention。lease 阻止 Attention 活跃时 blocks 被回收。

#### 关键实现提交

- `5543617a0` Design PAP shared KV owner；
- `21789daae` Keep IPC Prefill KV shared in Attention；
- `0b2fc04bb` Add paged KV IPC descriptors；
- `0e2d4ffe3` Use paged descriptors from Qwen3 Prefill；
- `a62895ef7` Write decode KV into resident paged blocks；
- `9f8fe013b` Reserve decode slots through KV owner；
- `214dff673` Unified KV with lease, scheduler preallocation and remote append；
- `960d3ab7d` Batch slot mapping for unified KV append；
- `81c5abadc` Remove Attention-local KV pool, fail-closed unified path；
- `c71ccc9df` Reserve multi-turn decode slots through the Scheduler。

#### 关键实验与证据

- `PAP-20260524-SHARED-KV`：第一版 shared owner、resident coverage 和 IPC path；
- `PAP-20260703-UNIFIED-KV`：统一 KV/lease/single-source paged FA 主路径；
- `PAP-20260703-SLOTMAPPING`：batched slot mapping 相对 per-row TPOT 改善约 29%；
- `PAP-20260711-MULTITURN-EXACT/CHAT`：decode blocks 在请求结束后进入 Prefill APC。

#### 负结果与被替代方案

- Attention-local KV pool、copy-prefix fallback 和 local-paged 分叉已删除；
- per-row slot mapping 产生重复 Python/kernel 开销，被 batched mapping 替代；
- 旧代码在 scheduler manager 外私自追加 decode blocks，只增长 cache/hash 计数却没有让
  model-runner block table 获得 slots；`c71ccc9df` 改为统一 allocation；
- writable capacity 不能只信环境变量，descriptor 必须 clamp 到真实 block capacity。

#### 当前结论与边界

Prefill 是 KV 的唯一物理 owner。完整 hashed blocks 在 lease release 后作为 APC LRU
candidate 保留；partial tail 和最后 sampled token 不保证命中。decode capacity 仍是有界
预留，不是无限 conversation reservation。

#### 深入阅读与原始证据

- [KV residency/lease 计划](../superpowers/plans/2026-07-02-pap-kv-residency-lease.md)
  （`tracked`）；
- [Unified KV 计划](../superpowers/plans/2026-07-03-pap-unified-kv-cache.md)
  （`tracked`）；
- [IPC-only KV + commit 计划](../superpowers/plans/2026-07-06-pap-ipc-only-kv-with-commit.md)
  （`tracked`）；
- `$PAP_REPO_RESULTS/20260703_unified_*`（`repo-untracked`）。

### M5. Decode Commit、Lease 与正确性闭环

#### 问题与动机

远端 GPU 写入 KV 只改变了显存；如果 Prefill Scheduler 不同时获得新 token IDs、
`num_computed_tokens`、block hash 和 cache registration，后续 APC 会把有效 KV 当成 miss，
或更糟糕地把错误 token 与 blocks 关联。异步通知还可能出现丢失、重复、乱序和请求已
释放后的 stale commit。

#### 设计与机制

Attention 在 KV append 成功后发送 `(request_id, new_seq_len, new_token_ids)`。Prefill
EngineCore 校验 delta、已有 handoff sampled token 和 request 生命周期，幂等推进 token
ledger/hash，并把新完整 blocks 注册到 APC。commit client 使用 ACK watermark、有限重试、
bounded queue 和 flush。request lease 在所有 commit flush 后释放，session cleanup 再删除
IPC views。deferred blocks 以 tail-to-prefix 顺序回到 block pool。

#### 关键实现提交

- `1bbdc5531` Add decode commit data structure；
- `1cb9eb685` Advance block hashes in `apply_decode_commit`；
- `dd4ced661` Add Prefill decode-commit FastAPI router；
- `7dde6d4a7` Add Attention-side commit client；
- `ea69e1b63` Wire commit into Attention executor；
- `e9044a88c` Fix overlap decode commit；
- `86a7c1273` Make decode commits reliably acknowledged；
- `fd723d2e2` Fix leased block eviction order；
- `c71ccc9df` Reconcile existing sampled handoff token and fail closed；
- `558db3cdd` Keep Projection out of local cache registration。

#### 关键实验与证据

- `PAP-20260706-DECODE-COMMIT`：commit endpoint、token propagation、lease release smoke；
- `PAP-20260710-ACK-LEASE`：ACK watermark、retry、queue drain 和 session drain；
- `PAP-20260711-MULTITURN-EXACT`：commit 后 hash chain 产生 decode-derived APC hit；
- 最终 exact/Chat clean run 各 109 次 commit、3 次 release 全部 200，active session 0。

#### 负结果与被替代方案

- 早期 empty-token/overlap commit 会错误推进状态，`e9044a88c` 修复真实 delta；
- fire-and-forget HTTP 不能证明状态已落稳，被 ACK watermark 和 bounded retry 替代；
- 原序释放 leased blocks 会让公共 prefix 比 tail 更早进入 eviction，改为反向 allocation
  order；
- Prefill 已包含 handoff sampled token 时重复 append 会破坏 token ledger，现改为比较并只
  追加缺失 delta；
- unknown/stale/mismatch 均 fail closed，不用吞异常维持表面 HTTP 成功。

#### 当前结论与边界

KV 写入、token ledger、hash chain、APC registration 和 lease 生命周期现在是一条可审计
事务链。最后 sampled token 没有 KV 属于模型 forward 语义，不应通过伪 commit 补齐。

#### 深入阅读与原始证据

- [Unified KV consistency closure](../superpowers/plans/2026-07-06-pap-unified-kv-consistency-closure.md)
  （`tracked`）；
- [IPC-only KV + commit](../superpowers/plans/2026-07-06-pap-ipc-only-kv-with-commit.md)
  （`tracked`）；
- `$PAP_REPO_RESULTS/20260710_ack_watermark_e2e*`（`repo-untracked`）；
- `$PAP_REPO_RESULTS/20260710_e904_nixl_rep*`（`repo-untracked`）；
- `$PAP_RESULTS/20260711_*multiturn*`（`external`）。

### M6. Benchmark、Tracing 与审计体系

#### 问题与动机

早期结果来自不同模型、输入输出长度、QPS、prompt 数、MPS 和服务启动方式，甚至可能在
请求完成但内部 session/commit 错误时仍显示 HTTP success。没有统一方法学，无法判断
一个优化是真实收益、排队变化、trace 扰动还是配置差异。

#### 设计与机制

标准 runner 固定模型、dataset、input/output、QPS、prompt count、warmup、MPS 和本地
代理绕过；记录 Git commit、tracked dirty、effective config、topology manifest 和 run
metadata。结束后验证 completed/failed、token correctness、pair routing、decode commit、
lease release、Attention session drain 和服务日志错误。Trace 运行独立标记，不与 normal
baseline 混报。

#### 关键实现提交

- `a5a0781d1` Record PAP experiment results；
- `1787786ec` Document PAP trace critical path；
- `4ee6b7126` Add wait/read trace instrumentation；
- `2e6db3b33` Add Projection timeline profiling；
- `279a8e8f2` Record PAP profiling experiments；
- `72b0c1598` Harden PAP benchmark correctness checks；
- `1ff0acee8` Document clean PAP route-copy baseline；
- `d8bce2e6c` Document clean active-peer PAP baseline；
- `043339691` Add strict multi-turn cache audit；
- `ad95c8c12` Add deferred PAP CUDA critical-path tracing；
- `e8ab4ab23` Add five-turn PD/PAP load testbed；
- `340c11abc` Make multi-turn load token continuous；
- `a646ae032` Parse completion prompt token IDs。

#### 关键实验与证据

- `PAP-20260701-PD-METHODOLOGY`：固定 Qwen3-8B `i128/o32` 的 PD/PAP 比较合同；
- `PAP-20260702-REMOTE-TRACE`：forward/layer/kernel 级关键路径归因；
- `PAP-20260710-QPS4-PD-AB`：降低 QPS 后区分 fixed TPOT 与 queue-driven TTFT；
- `PAP-20260711-ACTIVE-PEER`：clean 多轮 + strict routing/correctness/drain；
- `PAP-20260711-MULTITURN-EXACT/CHAT`：warm/cold cache salt 和 token-ID 等价性；
- `PAP-20260712-DEFERRED-GPU-TRACE`：session drain 后统一回收 CUDA events，trace
  TPOT 扰动约 2%，并对 span/counter 完整性 fail closed。
- `PAP-20260713-PD-PUSH-ROOTCAUSE`：CUDA P2P、UCX protocol selection、NIXL backend
  test 和官方 push connector 形成完整证据链；旧 pull `0.42 GiB/s` 被判定为无效基线；
- `PAP-20260713-FIVE-TURN-C4`：16K/5-turn/C4、每侧三次交错重启，实际 HTTP/decode
  peak concurrency 4，60/60 请求和所有 artifact-backed Gate 通过。

#### 负结果与被替代方案

- 只看 benchmark client 的成功请求数会漏掉 Attention OOM、commit 失败和 session 泄漏；
- trace run 的 TPOT 可能显著高于正常运行，不能作为 baseline；
- QPS16 下 TTFT 的大幅差距主要来自 PAP service rate 不足和排队，不能等同于单请求
  Prefill 固定开销；
- tracked-dirty A/B 可以做单变量归因，但提交后必须 clean 复验才能升级为正式基线；
- 不同拓扑、模型或 MPS 的结果不能只按 TPOT 数字横向排序。

#### 当前结论与边界

任何新的“默认开启”性能结论都应至少提供同代码 A/B、完整 workload、Git/config
provenance 和严格审计；关键 baseline 还需要 clean 多轮。当前索引引用旧历史结果时会降低
证据等级，而不是假设它满足今天的 runner 标准。

#### 深入阅读与原始证据

- [PD/PAP 对比方法](pap-pd-comparison-methodology-20260701.md)（`tracked`）；
- [同机 PD/NIXL 根因与校正](pd-same-node-nixl-transfer-root-cause-20260713.md)
  （`tracked`）；
- [PD Push 与 PAP 五轮 C4 报告](pd-pap-five-turn-load-results-20260713.md)
  （`tracked`）；
- [Remote Attention 优化规格](../superpowers/specs/2026-07-02-pap-remote-attention-optimization-design.md)
  （`tracked`）；
- [Remote Attention 诊断计划](../superpowers/plans/2026-07-02-pap-remote-attention-diagnostics.md)
  （`tracked`）；
- `$PAP_REPO/dev-memory/2026-07-10_PAP本周进展汇报.md`（`repo-untracked`）；
- `$PAP_PROFILES/*`（`repo-untracked`）。

### M7. Same-node Local-fast 与 TPOT 热路径

#### 问题与动机

Qwen3-8B 的 raw QKV/Attention-output P2P 数据量不是数十毫秒差距的充分解释。Profiler
显示每层重复的 Python 调度、临时 tensor/index 构造、slot wait/release、metadata 解析和
Attention KV append 准备累计成为 TPOT 主体。目标是利用 same-node P2P/CUDA IPC，把
控制面移出 36 层热路径。

#### 设计与机制

演进顺序是：direct-slot QKV send → 固定 mailbox/ring slot → typed local-fast transport →
stream-ordered ready generation/binary doorbell → step/layer metadata cache → batched unified-KV
slot mapping → Attention slot-plan cache。每个阶段保留开关做同代码 A/B，并要求无额外
CPU copy、无 per-row temporary tensor、无隐式全局同步。

#### 关键实现提交

- `401ce1425` Optimize offload route planning；
- `61848acf5` Add direct-slot QKV send；
- `49e02e5f5` Enable native KV append in 128 testbed；
- `9efafcda3` Align mailbox slots for 3-way pipeline；
- `e89346dc8` Make paged flash the only decode path；
- `2ebae15fa` Add local-fast transport scaffold and remove dead paths；
- `169066c78` Carry metadata and typed tensors in local-fast；
- `a6010cce6` Add adaptive spin, async doorbell, and metadata cache；
- `960d3ab7d` Batch unified-KV slot mapping；
- `87bb1061f` Optimize same-node decode data path；
- `6bc383dab` Vectorize PAP paged attention metadata；
- `c134bc3d9` Make PAP slot plans generation aware；
- `0727ed946` Use topology tokens for PAP metadata cache。
- `ad95c8c12` Add deferred Attention critical-chain CUDA timing。

#### 关键实验与证据

- `PAP-20260701-MAILBOX-HOTPATH`：分解 notification/read/route/append；
- `PAP-20260702-REMOTE-TRACE`：确认主要差距来自逐层重复控制而非 raw copy；
- `PAP-20260703-SLOTMAPPING`：batched slot mapping 相对 per-row TPOT 降低约 29%；
- `PAP-20260710-SLOTPLAN`：QPS16 三轮 median TPOT `33.21/32.99/34.37 ms`，
  跨轮中位数 `33.21 ms`；cache-off 回到 `42.12 ms`；
- `PAP-20260710-QPS4-PD-AB`：PAP/PD median TPOT `28.06/24.48 ms`；
- `PAP-20260712-METADATA-BULK`：1,024-block metadata miss GPU microbenchmark
  `7.31 -> 0.104 ms`；16K 两轮 clean formal Round2 TPOT
  `39.128 -> 30.585 ms`，PAP/PD `1.215x`；
- `PAP-20260712-TOPOLOGY-GENERATION`：三次 slot-plan 计数均从
  `8925/255/1` 变为 `17850/510/0`；Round1 TPOT `35.593 -> 30.521 ms`，
  Round2 保持在 `30.780 ms`；
- `PAP-20260712-METADATA-FAST-KEY`：三对交替 OFF/ON 的 Round2 TPOT
  paired 变化为 `-1.39%/-1.33%/-1.41%`；完整 block-ID 扫描减少 `36x`，clean
  formal Round1/Round2 TPOT 为 `30.196/30.449 ms`。
- `PAP-20260712-DEFERRED-GPU-TRACE`：相对 clean reference 的 Round1/Round2 TPOT
  扰动仅 `+2.12%/+1.77%`；每层 p50 为 QKV ready `0.567 ms`、KV append
  `0.008 ms`、paged FA `0.191 ms`、output P2P copy `0.007 ms`，四段计数完整且无
  pending/drop/error。

#### 负结果与被替代方案

- offload ubatch、decode barrier 和多个历史 runner microbatch 分支被删除；
- 简单 Q-first/KV-later 会增加消息和等待，没有形成 TPOT 收益；
- 某些 wavefront 在小 batch 把请求切碎，消息数上升并退化；
- 单纯加大 scheduler concurrency 不能解决逐层 handoff；
- 无保护地预提交跨进程 CUDA event/memory-flag wait 存在可见性和循环依赖风险；
- MPS 在本阶段固定 70/30，没有用扫描掩盖代码路径收益。

#### 当前结论与边界

same-node 固定开销已把短上下文 QPS4 TPOT 收敛到 PD 的约 `1.15x`；16K 两轮 Stage C
把 Round1/Round2 TPOT 收敛到约 `1.20x/1.21x`。Stage D 已排除 KV append、paged FA
本体和 output raw P2P copy，剩余最大可见区段是每层约 `0.567 ms` 的 QKV ready wait。
该值仍混合了 Projection 端必要 QKV 计算和 PAP handoff；下一步必须先从 Projection 进程
导出 source compute/copy timing，不能直接把全部 wait 记为通信开销。当前没有实现跨
layer 混合 Attention batch。

#### 深入阅读与原始证据

- [Attention–Projection same-node 数据面设计](pap-tpot-attention-projection-dataplane-20260710.md)
  （`tracked`）；
- [Remote Attention 诊断计划](../superpowers/plans/2026-07-02-pap-remote-attention-diagnostics.md)
  （`tracked`）；
- [NIXL root-cause tracing 计划](../superpowers/plans/2026-07-06-pap-nixl-root-cause-tracing.md)
  （`tracked`）；
- `$PAP_RESULTS/20260710_phase{1,2}_*`（`external`）；
- `$PAP_PROFILES/pap_*`（`repo-untracked`）。

### M8. 任意 x:y 拓扑与路由

#### 问题与动机

生产请求不能假设 PA/P 数量相等或 pair 永久对角绑定。连接层需要让任意 PA 都能服务
任意 P，同时一次请求在其当前 round 内绑定一个 pair。早期 round-robin 的 2PA2P 会只
覆盖 `pa0:p0`、`pa1:p1`，看似成功却没有验证 full Cartesian crossbar。

#### 设计与机制

Launcher/Proxy 根据 `xPAyP` 构造独立 PA groups 和 Projection instances；routing policy
显式选择 PA/P pair，并在 benchmark 中审计 pair coverage。Projection payload 携带目标
Attention endpoint；每个 PA 能接收多个 P source。active membership 以 request cohort
变化而不是永久连接数驱动 expected peer set。

#### 关键实现提交

- `f38b24351` Parameterize PAP benchmark topology；
- `45b93fcdf` Validate metadata-only X:Y routing；
- `7b1e04b1e` Add tensor-parallel launch support；
- `f7052fb84` Stabilize TP2 multi-PA offload；
- `45c302bb3` Support arbitrary PAP x:y topologies；
- `d654f6011` Add full-crossbar central dispatcher；
- `54bd1a59c` Track active PAP projection peers。

#### 关键实验与证据

- 2026-05 的 4PA4P、6PA2P、7PA1P/5PA3P 搜索记录了大拓扑最初的扩展问题；
- `PAP-20260710-ARBITRARY-XY`：1PA2P、2PA1P、2PA2P、3PA2P 短 smoke；
- `PAP-20260711-CENTRAL-DISPATCH`：2PA2P full-crossbar 四 pair contract；
- `PAP-20260711-ACTIVE-PEER`：1PA1P、1PA2P、2PA1P、3PA2P active set 最终清空。

#### 负结果与被替代方案

- 2PA2P 默认 `request_index % 2` 的 diagonal 分配不能证明 full crossbar；
- 5PA3P 30B run 为 `0/2000`，伴随 mailbox ACK/receive-slot timeout，标记 `invalid`；
- 节点数增加不带来单调性能：更多 Projection 可能降低每个 P 的 batch，更多 PA 也会增加
  peer/Attention 侧调度压力；
- x:y 连接成功不等于执行层能够高效 combine，多对多执行问题由 M9 解决。

#### 当前结论与边界

当前代码可以配置任意正整数 x:y，并对多种非对称拓扑完成 strict smoke。2PA2P 有正式
performance baseline；3PA2P 等更高拓扑只有 correctness 证据。多轮 conversation 下一轮
落到同一 PA 的 affinity 仍由未来外部 router 负责。

#### 深入阅读与原始证据

- [4PA4P Qwen3-8B baseline](pap-4pa4p-qwen3-8b-baseline.md)（`tracked`）；
- [6PA2P 大负载记录](pap-6pa2p-large-workload-20260526.md)（`tracked`）；
- [30B 优化 handoff](pap-30b-optimization-handoff-20260527.md)（`tracked`）；
- [多对多 Cohort 设计](pap-many-to-many-cohort-scheduler-20260711.md)（`tracked`）；
- `$PAP_REPO_RESULTS/20260710_xpayp_*`（`repo-untracked`）；
- `$PAP_RESULTS/20260711_active_peer_*`（`external`）。

### M9. 多对多 Cohort、Combine/Scatter 与 Active Peer

#### 问题与动机

一个 PA 接收多个 P 时，旧 mailbox 每个 peer 各自提交小 Attention batch，导致 1PA2P
TPOT 接近翻倍；2PA2P 交织 rows 又触发逐 row QKV packing/output copy。Projection 仍希望
保持大 cohort GEMM，因此不能用任意跨 layer 调度把 batch 打碎。目标是在 P 端保留
same-layer cohort，在 PA 端 combine 多来源并 scatter 回原 source。

#### 设计与机制

Phase 1 把多 ingress 收敛到单 central dispatcher/GPU compute owner；Phase 2 按同 layer、
QKV ABI 和 scale 选择 compatible work items，在 GPU workspace concat，一次 KV append +
paged FlashAttention，再按 row slices scatter。Phase 3 用缓存 index 的 `index_select`/
`index_copy_` 向量化交织 route。Phase 4 用 active-source membership 排除已 idle/finished
peer，避免不可能满足的固定窗口 timeout。

#### 关键实现提交

- `d654f6011` Add full-crossbar central dispatcher；
- `12b689d1b` Combine Attention work across Projection peers；
- `bdb7a7dc7` Vectorize crossbar route copies；
- `1ff0acee8` Document clean route-copy baseline；
- `581387a51` Measure coalescing wait outcomes；
- `8cb6e2022` Document adaptive coalescing experiment；
- `54bd1a59c` Track active Projection peers；
- `d8bce2e6c` Document clean active-peer baseline。

#### 关键实验与证据

- `PAP-20260711-CENTRAL-DISPATCH`：1PA1P legacy/central 三轮 TPOT
  `28.138/28.514 ms`，central 仅 `+1.34%`；
- `PAP-20260711-ATTENTION-COMBINE`：1PA2P 从 `53.67` 改善到 `36.75 ms`；2PA2P
  leader combine 从 `74.29` 改善到 `40.58 ms`；
- `PAP-20260711-ROUTE-COPY`：legacy/batched 三轮 mean TPOT
  `44.735/41.923 ms`，batched 改善 `6.29%`；
- `PAP-20260711-ADAPTIVE-COALESCE`：自适应窗口未胜固定 1 ms；
- `PAP-20260711-ACTIVE-PEER`：正式 clean 2PA2P 三轮 median TPOT
  `40.49 ms`、p99 `50.03 ms`，相对 PD `1.654x/1.942x`。

#### 负结果与被替代方案

- per-peer compute thread 会让 GPU 收到多个碎片化小 batch，被 central dispatcher 替代；
- 简单 ready scan/opportunistic receive 不能稳定形成跨 peer combine；
- 2PA2P 不做 leader/cohort 对齐时 combine coverage 和 TPOT 显著退化；
- 两状态全局自适应窗口 mean/median TPOT、TTFT、吞吐和 coverage 同时劣于固定窗口，已
  回滚；
- 永久以“历史绑定过的 peer 数”作为 expected group size 会等待 idle peer，active-source
  membership 取代该隐式 barrier；
- 跨 layer pointer-indirect FA 仍是研究方向，没有作为当前实现。

#### 当前结论与边界

2PA2P full crossbar 已同时满足四 pair、correctness、commit/release/drain 和 `<2x PD`
mean/median/p99 TPOT。Projection 端仍按 same-layer cohort forward；新请求只能加入最早
未 committed cohort，这主要影响首次 token/TTFT。任意更高 x:y 的性能尚未正式验证。

#### 深入阅读与原始证据

- [多对多 Cohort 与 Attention 聚合设计](pap-many-to-many-cohort-scheduler-20260711.md)
  （`tracked`，包含 Phase 1–4 完整表格）；
- [Central Dispatcher 实施计划](../superpowers/plans/2026-07-11-pap-phase1-central-dispatcher.md)
  （`tracked`）；
- `$PAP_REPO_RESULTS/20260711_phase{0,1,2,3,4}_*`（`repo-untracked`）；
- `$PAP_REPO_RESULTS/20260711_{d654f6011,12b689d1b,bdb7a7dc7}_*`
  （`repo-untracked`）；
- `$PAP_RESULTS/20260711_{54bd1a59c,active_peer,phase4_active_peer}_*`
  （`external`）。

### M10. 多轮对话与原生 Prefix Cache 复用

#### 问题与动机

PAP 的核心多轮优势是第一轮 prompt 和 decode KV 都已位于 PA 的 Prefill-owned blocks。
如果第二轮路由到同一 PA，应直接通过 vLLM APC 命中，而不需要像 PD 那样把 Decode KV
回传 Prefill。需要严格区分“命中原 prompt”与“真正命中第一轮 decode blocks”。

#### 设计与机制

每轮结束时 pair 和 Attention session 解散；commit 把 token/hash/cache 状态推进到
Prefill request，lease release 后完整 hashed blocks 以 refcount 0 留在 APC LRU。第二轮
使用新的 request ID，只要 tokenized prompt 与 committed history 共享完整 block 前缀，
APC longest-prefix lookup 就能重挂这些 blocks。最后 sampled token 和 partial tail 重算。

#### 关键实现提交

- `6a7094c3b` Revise multi-turn cache reuse design；
- `fd723d2e2` Fix leased block eviction order；
- `c71ccc9df` Enable decode KV prefix reuse；
- `043339691` Add strict exact-token multi-turn audit；
- `558db3cdd` Skip Projection local cache registration；
- `1a264c2b1` Plan Chat multi-turn audit；
- `d5ea82ca3` Add Chat multi-turn audit；
- `848f321ab` Preserve Qwen3 chat token continuity；
- `ba4d41c5b` Document multi-turn validation；
- `e8ab4ab23` Add five-turn PD/PAP load testbed；
- `340c11abc` Make multi-turn load token continuous；
- `a646ae032` Parse completion prompt token IDs。

#### 关键实验与证据

- `PAP-20260711-MULTITURN-EXACT`：第一轮 `128+48`，第二轮 prompt `183`，
  expected/actual hit `160/160`，decode-derived `32`，cold `0`；
- `PAP-20260711-MULTITURN-CHAT`：第一轮 `142+48`，materialized/LCP `189`，
  expected/actual hit `176/176`，decode-derived `48`，cold `0`；
- 两组 warm/cold 输出 token IDs 完全一致，strict audit、109 次 commit、3 次 release 和
  session drain 全部通过；
- `PAP-20260713-FIVE-TURN-C4`：4 条会话、5 轮、每轮 o256；16 个轮次转换逐一命中
  256 个 decode-derived tokens，三次 PAP session drain 均为 0 active sessions；校正
  PD 后稳态 PAP TTFT 为 PD `0.811x`，TPOT 为 `1.220x`。

#### 负结果与被替代方案

- ConversationDirectory、stable backend session、persistent Attention、resident snapshot
  和 scheduler detach/attach 方案被 native APC 路线替代；
- exact clean rep1 因 Projection zero-local-block 被错误 cache registration 而 500，修复
  后通过；
- Qwen3 `enable_thinking=false` generation prompt 插入空 reasoning scaffold，但该 token
  scaffold 不随 assistant content 回传，第二轮 decode-derived hit 为 0；thinking 模式保持
  token continuity；
- 不实现 final-token closure 或 partial-block attach，它们不是证明完整 decode block 复用
  的前置条件。

#### 当前结论与边界

1PA1P 的 prompt + decode 原生 APC 复用已经从两轮扩展到 5 轮和 C4，并通过每次
transition 的 decode-derived token 证据闭合。后续轮次能否命中仍取决于 token LCP 和
路由到同一 PA；当前 Proxy 不实现多 PA cache-aware affinity。APC blocks 是 LRU
candidate，高压力 eviction 时允许退化为重算，但不能影响输出正确性。

#### 深入阅读与原始证据

- [多轮原生 Prefix Cache 设计与结果](pap-xpayp-multiturn-kv-affinity-20260710.md)
  （`tracked`）；
- [Chat 多轮实施计划](../superpowers/plans/2026-07-11-pap-chat-multiturn-prefix-cache.md)
  （`tracked`）；
- [PD/PAP 五轮长上下文结果](pd-pap-five-turn-load-results-20260713.md)
  （`tracked`）；
- `$PAP_RESULTS/20260711_{fd723d_native_multiturn_diag1,native_multiturn_diag2,native_multiturn_diag3_audit,native_multiturn_diag4_reserved_slots}`
  （`external`，诊断链）；
- `$PAP_RESULTS/20260711_558db3cdd_multiturn_clean_rep2`（`external`，formal-clean）；
- `$PAP_RESULTS/20260711_848f321ab_chat_multiturn_clean_rep2`
  （`external`，formal-clean）。

## 6. 实验账本

实验账本以“一个可回答问题的 A/B 或验证”为单位，而不是以单个目录为单位。同一行可
聚合多个 repetition，但不会把不同 workload、代码状态或证据等级混在一起。`代码/状态`
中的 `clean` 指 tracked worktree clean；未注明 clean 的旧实验不得据此推断当时工作树状态。

### 6.1 P1–P3：原型、Mailbox 与大模型调度

| 稳定实验 ID | 模块 | 基线 → 处理；工作负载 | 代码/状态 | 最小结果、等级与决策 | 原始证据 |
| --- | --- | --- | --- | --- | --- |
| `PAP-20260522-PROTO-NIXL` | M1/M2 | 单体/早期 trigger → NIXL true-split；早期 1PA1P 功能请求 | `e3901b7a2`、`5d10f3e89`；历史状态 | Projection/Attention 真分离并完成请求；`historical/smoke`；**接受架构方向**，不作为性能基线 | Git commits；`$PAP_RESULTS/20260522_*`、`20260523_*`（`external`） |
| `PAP-20260524-PROJECTION-KVUNAWARE` | M1/M3 | 普通本地 KV admission → metadata-only remote progress；X:Y 路由 smoke | `d5464a6b7`；历史状态 | Projection 无 prompt KV 仍能 admission/forward；`historical/smoke`；**接受** | Git commit；M3；早期 `$PAP_RESULTS/20260523_*`（`external`） |
| `PAP-20260524-SHARED-KV` | M4 | Attention-local/copy → 写 Prefill resident paged blocks；单/短请求 | `a62895ef7`；历史状态 | Prefill-owned decode KV 写入路径成立；`historical/smoke`；**接受 ownership，后被 unified KV 完善** | Git commit；M4；早期 `$PAP_RESULTS/20260525_*`（`external`） |
| `PAP-20260526-MAILBOX` | M1/M2 | old handoff → NIXL mailbox/default poll；0.6B `i512/o8/q1/prompts2` 及 scale smoke | `998965825`；历史 checkpoint | microbench mean TPOT `56.54 → 24.64 ms`，并闭合 peer routing；`historical/controlled`；**接受 Mailbox checkpoint** | [Mailbox handoff](pap-nixl-mailbox-handoff.md)；`$PAP_RESULTS/20260525_*`、`20260526_*` |
| `PAP-20260526-3WAY` | M2/M6/M7 | serial → 3-way runner microbatch；8B 4PA4P `i1024/o16/q80/64 prompts` | `6e912dbbc` 附近；历史 clean 状态未保留 | threshold 12 mean TPOT `69.82 → 63.81 ms`，但小 batch/其他模型可退化；`controlled`；**不作为通用默认** | [3-way 记录](pap-runner-3way-microbatch.md)；`$PAP_RESULTS/20260526_{135807,135517,135644,135339}` |
| `PAP-20260527-WAVEFRONT` | M2/M7 | serial → 2/3-way layer wavefront；30B 1PA1P，`i128/o4`，macro batch 6/24/48 | `a2fb19376`、`7009a1cac`、`a500ba08a` | B24 2-way `143.41 → 105.53 ms`；B48 3-way `234.36 → 241.92 ms`，2-way到 `186.33 ms`；`controlled`；**仅保留 workload-aware 路线** | [6PA2P/30B 记录](pap-6pa2p-large-workload-20260526.md)；`$PAP_RESULTS/20260527_10*` |
| `PAP-20260527-CONCURRENCY` | M6/M7/M8 | `MAX_NUM_SEQS=320..2000`；30B 6PA2P `i1024/o64/q256/2000 prompts` | `821f21cb6`；历史受控 sweep | throughput `491.62..500.65 tok/s`、TPOT 非单调；`controlled`；**拒绝“只加 concurrency 即可解决”** | [30B handoff](pap-30b-optimization-handoff-20260527.md)；`$PAP_RESULTS/20260527_{122512,132011,132539,133154,133717,134611}` |
| `PAP-20260528-TP2` | M2/M8 | OOM 配置 `mem=0.90` → `0.78`；32B 3PA1P TP2 `i128/o64/512 prompts` | `7b1e04b1e`、`f7052fb84` | valid run `512/0`、完整 `32768` output tokens、mean TPOT `2224.55 ms`；`smoke`（OOM run 为 `invalid`）；**接受 TP2 稳定性，不接受性能** | [Mailbox handoff](pap-nixl-mailbox-handoff.md) TP2 snapshot |

### 6.2 P4–P6：方法学、统一 KV 与 same-node TPOT

| 稳定实验 ID | 模块 | 基线 → 处理；工作负载 | 代码/状态 | 最小结果、等级与决策 | 原始证据 |
| --- | --- | --- | --- | --- | --- |
| `PAP-20260701-PD-METHODOLOGY` | M6 | 不同口径历史结果 → 固定 model/input/output/QPS/concurrency/warmup；8B `i128/o32/q16/c64` | `4a7737647` 后的方法学基线 | 同口径旧点 PD/PAP median TPOT `24.9/294.8 ms`，暴露真实差距；`controlled`；**接受比较合同** | [PD/PAP 方法](pap-pd-comparison-methodology-20260701.md)；`$PAP_REPO_RESULTS/pd_pap_methodology_20260701` |
| `PAP-20260701-MAILBOX-HOTPATH` | M2/M7 | cached dlist/direct slot/piggyback/inline 等单变量开关；标准 1PA1P | `401ce1425`、`61848acf5` 周边；多组受控 A/B | 多个局部技巧无稳定收益或退化，raw copy 不是充分瓶颈解释；`controlled/diagnostic`；**保留 direct slot，拒绝退化微优化** | [NIXL/NVLink idea book](pap-nixl-nvlink-optimization-idea-book-20260707.md)；`$PAP_RESULTS/20260701_*` |
| `PAP-20260702-REMOTE-TRACE` | M2/M6/M7 | normal run → critical-path/Projection/Attention trace；1PA1P | `e89346dc8`、相关 trace patch；trace-on | 36 层逐层往返、Python/线程/调度交替累计主导，trace 本身扰动 TPOT；`diagnostic`；**接受瓶颈归因** | [Remote Attention 诊断计划](../superpowers/plans/2026-07-02-pap-remote-attention-diagnostics.md)；`$PAP_PROFILES/*` |
| `PAP-20260703-UNIFIED-KV` | M3/M4/M5 | Attention-local/copy-prefix → Prefill-owned unified KV/lease/remote append；1PA1P staged smoke | `214dff673`；提交后 smoke | 单一 paged-FA KV source 跑通，local pool/copy fallback 可删除；`smoke/controlled`；**接受** | [Unified KV 计划](../superpowers/plans/2026-07-03-pap-unified-kv-cache.md)；`$PAP_REPO_RESULTS/20260703_unified_*` |
| `PAP-20260703-SLOTMAPPING` | M4/M7 | per-row → batched unified-KV slot mapping；相同 1PA1P decode workload | `960d3ab7d`；受控 A/B | TPOT 相对 per-row 降低约 `29%`；`controlled`；**接受 batched mapping** | Git commit；`$PAP_REPO_RESULTS/20260703_unified_batched_*` |
| `PAP-20260706-DECODE-COMMIT` | M4/M5 | GPU 写 KV 但不推进 scheduler → commit token/hash/cache transaction；IPC-only smoke | `24361dd67`；实现/正确性 smoke | token propagation、hash advance、lease release 闭合；`smoke`；**接受，继续 ACK 加固** | [IPC-only KV + commit](../superpowers/plans/2026-07-06-pap-ipc-only-kv-with-commit.md)；`$PAP_REPO_RESULTS/20260706_ipc_only_smoke*` |
| `PAP-20260710-ACK-LEASE` | M5/M6 | fire-and-forget/原序释放 → ACK watermark/retry/drain/tail-first release；1PA1P E2E | `86a7c1273`；提交后 smoke/clean NIXL reps | commit queue 能 drain、session/lease 能释放，错误 fail closed；`controlled/smoke`；**接受** | `$PAP_REPO_RESULTS/20260710_ack_watermark_e2e*`、`20260710_e904_nixl_rep*_clean` |
| `PAP-20260710-SLOTPLAN` | M4/M7 | stream ring `47.21 ms` → binary/descriptorless + all-active + cross-layer slot-plan；8B 1PA1P `i128/o32/q16/128` | `87bb1061f` 前受控实现；提交前 A/B | 三轮 median TPOT 中位数 `33.21 ms`；cache-off `42.12 ms`；`controlled`；**接受并提交** | [same-node 数据面](pap-tpot-attention-projection-dataplane-20260710.md)；`$PAP_RESULTS/20260710_phase{1,2}_*` |
| `PAP-20260710-QPS4-PD-AB` | M6/M7 | PD 1P1D vs PAP 1PA1P；8B `i128/o32/q4/128`，同 GPU 对、无 backlog | 同一 tracked-dirty PAP 状态，双方各三轮 | median TPOT `24.48/28.06 ms`（`1.146x`）；median TTFT `171.18/163.68 ms`；`controlled`；**接受“高 QPS TTFT 主要由排队放大”** | [same-node 数据面](pap-tpot-attention-projection-dataplane-20260710.md)；`$PD_RESULTS/20260710_pd_qps4_rep*_current`、`$PAP_RESULTS/20260710_pap_slot_plan_qps4_rep*_current` |

### 6.3 P7–P10：任意 x:y、多对多、多轮 APC 与校正基线

| 稳定实验 ID | 模块 | 基线 → 处理；工作负载 | 代码/状态 | 最小结果、等级与决策 | 原始证据 |
| --- | --- | --- | --- | --- | --- |
| `PAP-20260710-ARBITRARY-XY` | M1/M3/M8 | 等量/对角假设 → 任意正整数 x:y 和显式 pair routing；1PA2P/2PA1P/2PA2P/3PA2P 短输出 | `45c302bb3`；clean correctness smokes | topology、metadata-only Projection 和 pair audit 跑通；`smoke`；**接受连接/控制面** | `$PAP_REPO_RESULTS/20260710_xpayp_*` |
| `PAP-20260711-CENTRAL-DISPATCH` | M8/M9 | per-peer compute owner → single central FIFO；1PA1P QPS4 交替三轮 | `d654f6011`；clean | legacy/central median TPOT `28.138/28.514 ms`（`+1.34%`）；`controlled`；**接受为 combine 基线** | [多对多设计](pap-many-to-many-cohort-scheduler-20260711.md)；`$PAP_REPO_RESULTS/20260711_d654f6011_*` |
| `PAP-20260711-ATTENTION-COMBINE` | M9 | central FIFO per-source → same-layer combine/scatter；1PA2P/2PA2P QPS4 | `12b689d1b` 的 tracked patch/提交结果 | 1PA2P `53.67 → 36.75 ms`；2PA2P `74.29 → 40.58 ms`；`controlled`；**接受** | [多对多设计](pap-many-to-many-cohort-scheduler-20260711.md)；`$PAP_REPO_RESULTS/20260711_phase2_*`、`20260711_12b689d1b_*` |
| `PAP-20260711-ROUTE-COPY` | M9 | Python per-row gather/scatter → cached `index_select/index_copy_`；2PA2P QPS4 交替三轮 | tracked-dirty A/B；`bdb7a7dc7` 后 clean 三轮 | A/B mean TPOT `44.735 → 41.923 ms`（`-6.29%`）；clean mean `41.778 ms`；`controlled/formal-clean`；**默认开启** | `$PAP_REPO_RESULTS/20260711_phase3_ab_*`、`20260711_bdb7a7dc7_*` |
| `PAP-20260711-ADAPTIVE-COALESCE` | M9 | fixed 1 ms → global two-state 1 ms/500 us；2PA2P QPS4 交替三轮 | `581387a51` 后 tracked-dirty prototype，未提交默认 | mean/median TPOT `+0.86%/+0.66%`，coverage `-5.45 pp`；`controlled`；**拒绝并回滚** | `$PAP_REPO_RESULTS/20260711_phase4_ab_{fixed,adaptive500}_rep*` |
| `PAP-20260711-ACTIVE-PEER` | M8/M9 | historical peer count → request-cohort active membership；2PA2P QPS4 A/B + clean 三轮 + x:y smoke | tracked-dirty A/B；`54bd1a59c` 后 clean | clean median/p99 TPOT `40.490/50.031 ms`，为 PD `1.654x/1.942x`；`formal-clean`；**默认开启于 multi-P combine** | `$PAP_RESULTS/20260711_phase4_active_peer_ab_*`、`20260711_54bd1a59c_*`、`20260711_active_peer_*` |
| `PAP-20260711-MULTITURN-EXACT` | M3/M4/M5/M10 | cold salt vs 同 PA warm；round1 `128+48`、round2 materialized prompt `183` | `558db3cdd`；clean rep2 | expected/actual hit `160/160`，decode-derived `32`，cold `0`；`formal-clean`；**接受 native APC** | [多轮设计](pap-xpayp-multiturn-kv-affinity-20260710.md)；`$PAP_RESULTS/20260711_558db3cdd_multiturn_clean_rep2` |
| `PAP-20260711-MULTITURN-CHAT` | M5/M6/M10 | Qwen3 non-thinking discontinuous history → thinking 模板连续 history；真实两轮 messages warm/cold | `848f321ab`；clean rep2 | expected/actual hit `176/176`，decode-derived `48`，cold `0`；`formal-clean`；**接受 token-continuous Chat lane** | [多轮设计](pap-xpayp-multiturn-kv-affinity-20260710.md)；`$PAP_RESULTS/20260711_848f321ab_chat_multiturn_clean_rep2` |
| `PAP-20260712-MULTITURN-NORTHSTAR` | M6/M7/M10 | 临时多轮 smoke → 固定 16K/2-turn/C1 PD/PAP test bed；NIXL mailbox → same-node local-fast；HTTP EOF 计时 → last-output-token v2 | `7e81e2d10`；clean PD/PAP formal | v2 round2 PD/PAP TTFT `267.27/235.39 ms`、TPOT `25.18/39.13 ms`，PAP 为 PD `0.881x/1.554x`，三轮 exact cache/audit 通过；每轮复现一个 chunk-generation topology false mismatch；**接受 v2 reference，P0 转向 metadata bulk build** | [北极星记录](pap-pd-multiturn-north-star-20260712.md)；`$PAP_RESULTS/20260712_{161402,162130}_*`；legacy `20260712_{031855,032326}_*` |
| `PAP-20260712-METADATA-BULK` | M6/M7/M10 | paged-FA miss 的逐元素 CUDA metadata 写 → bulk tensor build；同一 16K/2-turn/C1 PAP formal | `6bc383dab`；clean 三轮 | Round2 TTFT `235.39 -> 218.26 ms`、TPOT `39.128 -> 30.585 ms`（`-21.83%`），PAP/PD TPOT `1.215x`；三轮 exact cache/output/audit 稳定；**接受并晋升 PAP reference** | [北极星记录](pap-pd-multiturn-north-star-20260712.md)；`$PAP_REPO_RESULTS/20260712_171755_6bc383dab_pap_multiturn_formal` |
| `PAP-20260712-TOPOLOGY-GENERATION` | M6/M7/M10 | request 级永久 topology latch → prefix activation + session epoch + generation/topology ID；同一 16K/2-turn/C1 PAP formal | `c134bc3d9`；clean 三轮 | slot `hits/misses/mismatch 8925/255/1 -> 17850/510/0`；Round1 TPOT `35.593 -> 30.521 ms`（-14.25%），Round2 `30.780 ms`（+0.64%，neutral）；conversation -5.91%；**接受为默认并晋升当前 reference** | [北极星记录](pap-pd-multiturn-north-star-20260712.md)；`$PAP_REPO_RESULTS/20260712_181613_c134bc3d9_pap_multiturn_formal` |
| `PAP-20260712-METADATA-FAST-KEY` | M6/M7/M10 | cache hit 每层扫描完整 block table → process-unique topology token + seq-len key；同代码 OFF/ON 三对与 clean formal | `0727ed946`；controlled 六轮 + clean 三轮 | OFF/ON Round2 TPOT `30.848 -> 30.419 ms`（-1.39%），三对均改善；block IDs scanned `18994176 -> 527616`（`36x`）；clean TPOT R1/R2 `30.196/30.449 ms`，为 PD `1.203x/1.209x`；比较器仍为 neutral；**接受默认并晋升当前 reference，不宣称显著收益** | [北极星记录](pap-pd-multiturn-north-star-20260712.md)；`$PAP_REPO_RESULTS/20260712_stagec_{off,on}{1,2,3}`；`$PAP_REPO_RESULTS/20260712_201947_0727ed946_pap_multiturn_formal` |
| `PAP-20260712-DEFERRED-GPU-TRACE` | M6/M7 | 每层同步 CUDA trace → deferred event record/query + drain 后 flush；16K/2-turn/C1 quick | `ad95c8c12`；dirty diagnostic | trace TPOT 扰动 R1/R2 `+2.12%/+1.77%`；QKV ready p50 `0.567 ms/layer`，FA `0.191`、append `0.008`、output copy `0.007`；四段 count 精确、drop/error 为 0；**接受诊断工具，下一步只拆 Projection→Attention QKV chain** | [北极星记录](pap-pd-multiturn-north-star-20260712.md)；`$PAP_REPO_RESULTS/20260712_stagec_deferred_gpu_trace_v1` |
| `PAP-20260713-PD-PUSH-ROOTCAUSE` | M6 | 旧 V2 pull → V1 cross-layer pull → emulation fail-closed → 官方 push；16K/C1 同机 GPU1/2 | `131e1dfa2` 后恢复上游源码；diagnostic | pull GET 为 TCP emulation，`2254.5 MiB / 5.34 s / 422 MiB/s / 72144 descriptors`；push PUT 为 CUDA IPC，`91.984 ms / 24509.697 MiB/s / 1 descriptor`；**废止旧 pull 公平基线，接受官方 push** | [根因报告](pd-same-node-nixl-transfer-root-cause-20260713.md)；`$PAP_REPO_RESULTS/20260713_pd_*` |
| `PAP-20260713-FIVE-TURN-C4` | M6/M7/M10 | 两轮 C1 → exact-token 16K/5-turn/C1,C2,C4；corrected-push PD vs local-fast PAP | `e8ab4ab23`、`340c11abc`、`a646ae032`；clean C4 三轮 | C4 steady TTFT `306.166/248.321 ms`（PAP/PD `0.811x`），TPOT `42.115/51.375 ms`（`1.220x`）；两侧各 60/60、peak concurrency 4、无 OOM/fatal；**接受为当前多轮并发北极星** | [五轮报告](pd-pap-five-turn-load-results-20260713.md)；`$PAP_REPO_RESULTS/20260713_031215_a646ae032_pd_pap_load_c4_formal` |

## 7. 负结果、回滚与被替代路线

本节只登记有明确 rejecting evidence、失效原因或替代方案的路线。它们不是“无效工作”，
而是避免以后重复走弯路的设计边界。

| 负结果 ID | 路线/触发条件 | Rejecting evidence | 决策与替代方案 | 证据位置 |
| --- | --- | --- | --- | --- |
| `NEG-TRANSPORT-LEGACY` | TCP trigger、NCCL offload、旧 Projection receive path | 多套 transport 增加分叉，未解决逐层控制开销 | **被替代/删除**；跨节点保留 NIXL mailbox，同节点走 local-fast | M1/M2；`e826128d1`、`883a8969d` |
| `NEG-QFIRST-PARTIAL` | Q-first/KV-later、Q-first Projection、Attention partial overlap | 首轮 CUDA A/B 增加消息/等待并退化，依赖链仍在 | **拒绝默认启用**；优先复用整 step metadata/slot plan | M2/M7；[NIXL/NVLink idea book](pap-nixl-nvlink-optimization-idea-book-20260707.md) |
| `NEG-MAILBOX-MICRO` | async send slots、ACK piggyback、inline poll/publish、slot count 2 | 受控 A/B 多次无稳定收益或回退 | **回滚/保留实验开关**；瓶颈升级到逐层 CPU/调度链 | M2；`PAP-20260701-MAILBOX-HOTPATH` |
| `NEG-3WAY-GENERAL` | 固定 3-way runner microbatch 作为通用默认 | 小 batch 拆成 B=1/2 后消息和上下文切换主导；7PA1P 也破坏 Projection 算术密度 | **拒绝通用默认**；只保留 workload-aware 2-way/auto 的历史结论 | `PAP-20260526-3WAY`；[3-way 记录](pap-runner-3way-microbatch.md) |
| `NEG-WAVEFRONT-FIXED` | 固定 3-way layer wavefront | 30B B6 `68.11 → 96.18 ms`；B48 `234.36 → 241.92 ms`，而 2-way 更快 | **被动态粒度替代**；小 batch 禁用、优先 2-way | `PAP-20260527-WAVEFRONT`；[30B 记录](pap-6pa2p-large-workload-20260526.md) |
| `NEG-ATTENTION-LOCAL-KV` | Attention-local paged pool、copy-prefix/local-paged fallback | 双 KV source 破坏 ownership、重复显存与 copy，不能自然进入 Prefill APC | **删除**；统一为 Prefill-owned KV + IPC remote append | M4；`214dff673`、`24361dd67` |
| `NEG-HIGHSCALE-INVALID` | OOM、0 完成或 incomplete 的大拓扑结果 | TP2 mem 0.90 Attention OOM；6PA2P 早期 `0/600`；5PA3P `0/2000` mailbox timeout | **标记 invalid，禁止引用延迟**；降低 mem/修 liveness 后重跑 | M8；[Mailbox handoff](pap-nixl-mailbox-handoff.md)、[6PA2P 记录](pap-6pa2p-large-workload-20260526.md) |
| `NEG-SLOTMAP-PERROW` | unified KV 每 row 构造 slot mapping | 同 workload batched 版本 TPOT 约低 `29%` | **被替代**；默认 batched mapping，partial batch 保守 fallback | `PAP-20260703-SLOTMAPPING`；`960d3ab7d` |
| `NEG-1PA2P-SPLIT-PEER` | 一个 PA 为两个 P 分别提交 Attention 小 batch | clean QPS4 median TPOT `28.19 → 53.67 ms`；总有效 Attention FLOPs 未翻倍 | **被 central combine/scatter 替代** | [多对多设计](pap-many-to-many-cohort-scheduler-20260711.md)；`$PAP_RESULTS/20260711_ab_localfast_q4_*` |
| `NEG-ADAPTIVE-COALESCE` | 全局二态 fixed/adaptive wait | mean/median TPOT、TTFT、吞吐、coverage 同时不胜 fixed；状态频繁抖动 | **回滚**；保留固定窗口，以 active membership 判断值得等待的 peer | `PAP-20260711-ADAPTIVE-COALESCE` |
| `NEG-RESIDENT-MULTITURN` | ConversationDirectory、stable backend session、resident snapshot、detach/attach | 需要长期绑定和新状态机；现有 APC 已能挂回 refcount-0 hashed blocks | **被替代**；每轮 pair 解散，下一轮由 cache-aware router 回同 PA | M10；`6a7094c3b`；[多轮设计](pap-xpayp-multiturn-kv-affinity-20260710.md) |
| `NEG-PROJECTION-ZEROBLOCK` | 通用 cache registration 对 Projection 0 local blocks fail closed | exact clean rep1 返回 500：“0 个本地块却缓存 8 个块” | **修复边界而非放宽检查**；Projection 跳过本地 cache registration | M3/M10；`558db3cdd`；`$PAP_RESULTS/20260711_043339691_multiturn_clean_rep1` |
| `NEG-QWEN3-NONTHINK` | `enable_thinking=false` 的 Chat 多轮 token 历史 | 空 reasoning scaffold 不随 assistant content 回传，clean rep1 decode-derived hit `0` | **拒绝用于连续 token 复用验收**；thinking 模式保留完整 materialized LCP | M10；`848f321ab`；`$PAP_RESULTS/20260711_d5ea82ca3_chat_multiturn_clean_rep1` |

## 8. 关键提交时间线

本节只保留改变架构、协议、证据方法或默认决策的里程碑；代码级完整顺序仍以
`git log --first-parent feature/pap` 和每个 commit diff 为准。

| 阶段/日期 | 关键提交 | 为什么重要 | 结果入口 |
| --- | --- | --- | --- |
| P1 / 05-22..24 | `e3901b7a2` NIXL prototype；`5d10f3e89` true split；`d5464a6b7` KV-unaware admission；`a62895ef7` resident paged write | 确立 PA owns KV、Projection metadata-only、Attention remote service 的三角色边界 | M1/M3/M4；账本 P1 |
| P2 / 05-26 | `998965825` mailbox checkpoint；`6e912dbbc` 6PA2P large workload | 把 transport/routing 固化到可扩展 checkpoint，并首次系统记录 scale-out 失败 | M2/M6；账本 P2 |
| P3 / 05-27..28 | `a2fb19376` MoE wavefront；`7009a1cac` batch sweep；`821f21cb6` fixed PD/PAP sweep；`7b1e04b1e` TP launch；`f7052fb84` TP2 multi-PA fix | 证明 batch 粒度与 Projection arithmetic density 的权衡，并补齐 TP2 correctness | M6/M7/M8；账本 P3 |
| P4 / 06-30..07-01 | `e826128d1` remove NCCL；`a017bc7d4` Attention breakdown；`401ce1425` route plan；`61848acf5` direct slot | 从“换 transport”转向可归因的逐层热路径分析 | M2/M6/M7 |
| P5 / 07-02..06 | `e89346dc8` paged-FA only/trace；`169066c78` typed local-fast；`214dff673` unified KV；`960d3ab7d` batched slot mapping；`24361dd67` consistency closure | 删除 local-KV 分叉，把 KV write、slot、hash 和 lease 连成一条主线 | M4/M5/M7；账本 P5 |
| P6 / 07-10 | `86a7c1273` reliable ACK；`72b0c1598` strict benchmark checks；`87bb1061f` same-node data path | 正确性 fail-closed 后，以 slot-plan 把 1PA1P TPOT 收敛到 PD `1.146x`（QPS4） | M5/M6/M7；账本 P6 |
| P7 / 07-10..11 | `45c302bb3` arbitrary x:y；`d654f6011` central dispatcher；`12b689d1b` combine；`bdb7a7dc7` route copy；`581387a51` wait metrics；`54bd1a59c` active peer；`d8bce2e6c` clean baseline | 从“能连接多 peer”升级到真正 combine/scatter，并用 active cohort 去除 idle-peer barrier | M8/M9；账本 P7 |
| P8 / 07-11 | `6a7094c3b` native APC design；`fd723d2e2` eviction order；`c71ccc9df` decode reuse；`043339691` exact audit；`558db3cdd` Projection boundary；`848f321ab` Chat continuity；`ba4d41c5b` validation note | 证明无需 resident session/KV 回传即可命中第一轮 decode blocks | M10；账本 P8 |
| P9 / 07-12 | `7e81e2d10` v2 timing/gates；`7d0fd13cb` formal references；`6bc383dab` bulk metadata；`c134bc3d9` generation-aware slot-plan；`0727ed946` topology-token fast key；`ad95c8c12` deferred GPU trace | 把 16K 两轮 PD/PAP 固化为可审计 test bed，移除 metadata miss 标量写、chunk topology false mismatch 和 cache-hit 全 block 扫描，并用低扰动 GPU timing 将剩余热点收敛到 QKV ready chain | M6/M7/M10；账本 P9 |
| P10 / 07-13 | `131e1dfa2` revert diagnostic UCX override；`e8ab4ab23` five-turn testbed；`340c11abc` exact-token continuity；`a646ae032` completion token parsing | 证明旧 PD pull 走 TCP emulation，以官方 push CUDA IPC 重建同机基线，并把比较升级到 16K/5-turn/C4/三次交错正式矩阵 | M6/M10；账本 P10；[五轮报告](pd-pap-five-turn-load-results-20260713.md) |

## 9. 未完成问题与外部依赖

- 多 PA 多轮 cache-aware routing 由未来 Dynamo 等外部路由框架负责；
- arbitrary x:y 的连接/控制面已支持，但除 2PA2P 外主要是 correctness smoke；
- cross-layer Attention batch 和同进程双 GPU executor 尚未通过 profiler gate；Attention
  进程 GPU timeline 已闭合，Projection source compute/copy 的跨进程导出尚未完成；
- 同步 chunked Prefill 的 topology generation 已闭环；异步 import 尚缺 unified descriptor
  字段透传、readiness failed 标记和 session-epoch queue guard，当前默认关闭；
- 旧 pull PD 已被 corrected-push 基线替代；历史 PAP 内部 A/B 仍有效，但旧 PAP/PD 比值
  不应继续用于公平结论；
- `P0-CORRECTNESS-DIVERGENCE`：C4 中 PD 有一次相同 R4 prompt 的内部轨迹分叉；
  PD/PAP 则在相同 R2 prompt 下稳定分叉。需要依次补齐 raw token/top-k margin、PD
  batch-invariant A/B、teacher-forced cold/warm R2、QKV transport/KV slot checksum 和
  逐层 Attention/hidden/logits 对比；关闭前不宣称逐 token 数值等价。详细 Gate 见
  [五轮报告的 P0 待办](pd-pap-five-turn-load-results-20260713.md#61-p0-待办输出分叉的诊断与修复)；
- 2PA2P 已达到旧 `2x PD` 目标，但需在 corrected-push PD、五轮负载下重验；更高拓扑、
  不同模型和更高并发的普适性仍需实验；
- raw results、profiles 和 `/tmp` handoff 需要独立归档策略，Git 只追踪本索引和摘要。

## 10. 新增实验记录模板

未来实验复制以下模板。ID 一经引用不复用；重跑用同一 ID 下的 `rep`，改变 workload、
核心变量或问题时新建 ID。不要把 raw 目录移动进 Git；只记录真实存储类别。

```markdown
### PAP-YYYYMMDD-SHORT-NAME — 一句话问题

- 日期/操作者：
- 阶段/模块：P?；M?
- 假设：
- Baseline：
- Treatment（只改变的主变量）：
- Workload contract：model；dataset；input/output；QPS；prompts；warmup；
  max concurrency；topology；TP；MPS；transport；GPU binding；proxy bypass
- 代码与配置：branch；commit；tracked clean/dirty；dirty diff 说明；effective config
- Repetitions/运行顺序：
- 证据等级：formal-clean / controlled / diagnostic / smoke / historical / invalid

#### 最小结果

| 模式/rep | 完成/失败 | TTFT | TPOT mean/median/p99 | req/s | 关键内部指标 |
| --- | --- | --- | --- | --- | --- |
| baseline | | | | | |
| treatment | | | | | |

#### 严格审计

- [ ] 输出 token / warm-cold token IDs 正确
- [ ] PA/P pair routing 与要求的 crossbar coverage 正确
- [ ] decode commit 数量、状态码、ACK watermark 和 queue drain 正确
- [ ] KV lease release 数量、顺序和 block accounting 正确
- [ ] Attention/Projection session drain，active peer set 最终为空
- [ ] 无 OOM、timeout、dispatcher failure、stale update 或隐藏 server error
- [ ] trace/diagnostic 数据没有被当成 normal baseline

#### 决策

- 结论：接受 / 保留为可选实验 / 拒绝 / 回滚 / 被替代 / 无结论
- 决策门槛与实际值：
- 机制解释与反证：
- 当前默认/回退开关：
- 下一步（若有）：

#### Provenance 与原始证据

- [ ] raw 路径已记录并确认存在
- raw 路径：
- 存储类别：tracked / repo-untracked / external / temporary / missing
- benchmark JSON / metadata / audit：
- service logs / trace / profile：
- owning design/spec/plan：
- implementation commit / reverting commit：

#### 索引维护

- [ ] 已向第 6 节实验账本追加或更新稳定 ID
- [ ] 已更新所属模块的关键实验、负结果和当前边界
- [ ] 若改变主决策，已更新第 3 节时间线和第 8 节关键提交
- [ ] 若结果被拒绝/回滚/替代，已登记到第 7 节
```
