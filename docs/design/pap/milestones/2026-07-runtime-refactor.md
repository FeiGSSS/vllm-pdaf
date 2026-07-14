---
pap_doc_schema: 1
status: active
canonical: null
superseded_by: null
related_experiments:
  - PAP-20260714-SEAL-HANDOFF-KV
last_validated_commit: bcbcefa127aabb06de5daf1d6adc50724b2c764b
---

# PAP Runtime Milestone 重构设计

日期：2026-07-14

状态：已逐节评审通过，待书面 spec 复核

## 1. 背景与目标

PAP 已经完成 Prefill–Attention–Projection 主架构、Prefill-owned unified KV、
local-fast、NIXL mailbox、任意 `xPAyP`、可靠 decode commit/lease、原生 APC
多轮复用以及 sealed KV handoff 等阶段性能力。与此同时，两个月的快速实验迭代也
留下了三类工程债务：

1. `vllm/pap` 约 16840 行，`attention_executor.py` 单文件约 5923 行，同时承担
   协议、session、KV、Attention、dispatcher、transport loop、HTTP/TCP 和 CLI；
2. `tests/pap` 约 15324 行、29 个文件、585 个 collected cases，缺少统一的测试
   层级、不变量和保留理由；
3. 历史实验已经有证据分级和人工索引，但 raw run 存在多代目录与 schema，实验
   数据、聚合结果和结论仍可能漂移。

本 milestone 的目标不是继续优化性能，而是把已经成立的 PAP 主路径冻结、收敛并
模块化，为后续开发建立明确的源码、测试和实验规范。

## 2. 决策原则

本设计采用以下兼容矩阵：

- PAP 正确性语义、KV ownership、请求生命周期和正式性能基线严格保持；
- 历史 raw data 原地只读，不移动、不改写；
- 启动脚本、环境变量和 wire protocol 可以整理，但稳定入口需要兼容 façade；
- 已被拒绝、回滚或替代的内部实验路径直接删除，不为历史复现保留当前实现；
- `vllm.pap` 私有类、函数和文件布局可以重构；
- 历史实验通过对应 Git commit、raw artifacts 和实验 registry 复现。

整体策略是：**外部行为保守、内部结构激进、历史证据不可变。**

## 3. 支持范围与证据边界

### 3.1 保留能力

- 任意正整数 `xPAyP` topology；
- 同机 `local_fast` transport；
- 跨机 NIXL transport；
- direct 与 multi-Projection combine execution；
- session、routing、KV ownership、decode commit、lease 和 drain 的共同语义。

### 3.2 本 milestone 的运行验证范围

唯一运行 release gate 是最新 P17 1PA1P 主路径：

```text
model: Qwen3-8B FP16
topology: 1PA1P / TP1
transport: local_fast
MPS: static 64/28
decode token: async
Prefill KV import: async
KV handoff: sealed manifest
KV ownership: Prefill-owned unified KV
workload: 16K initial context, 5 turns, C4, 256 output tokens/turn
```

`xPAyP` 和跨机 NIXL 在重构期间保留源码与接口，但不运行 E2E。里程碑报告必须将
它们标记为：

> 接口与实现兼容性保留；本 milestone 未重新进行 E2E 验证。

这两个范围不得被描述为本 milestone 已验证通过。

## 4. 实施顺序

重构采用“冻结—契约保护—收敛—拆分—重新冻结”的顺序。

### 4.1 Freeze

在任何 PAP 行为修改前，以包含本设计且尚无 runtime/test 改动的 tracked-clean
commit 运行三轮 P17 C4 formal。该 commit 的 PAP runtime 必须与代码锚点
`bcbcefa127aabb06de5daf1d6adc50724b2c764b` 相同，并记录：

- 完整 Git commit、tracked dirty 状态和实现 fingerprint；
- 完整 workload、hardware、runtime 和 PAP profile；
- correctness、routing、decode-token join、commit、lease 和 session-drain audit；
- R1/steady TTFT、R1/steady TPOT 和原始结果路径。

该结果是 pre-refactor baseline。旧 commit 的实验只能作为历史证据，不能替代这次
冻结。

### 4.2 契约保护

先建立测试不变量账本、P17 profile 和实验 schema，再删除任何旧路径。必要时增加
characterization tests，但不得为覆盖率数字制造低价值测试。

### 4.3 主路径收敛

每次只删除一类旧路径。每类删除作为独立 review/commit 边界，并运行 targeted
tests、完整 PAP CPU suite 和 P17 C1 quick。

### 4.4 模块拆分

只迁移已经收敛的主路径。每次拆出一个有清晰接口的职责单元，不在文件移动期间
加入新算法或性能优化。

### 4.5 文档迁移

建立 PAP canonical 文档入口，把现有 `docs/design` 和 `docs/superpowers/specs`
分类为 current、milestone、archived 或 superseded。提取旧 implementation plans
中的实际决策和验证结果，更新全部引用后删除 `docs/superpowers/`。

### 4.6 Milestone freeze

完成全部测试、registry 校验、P17 C1 和三轮 P17 C4 formal，并生成前后对比报告。

## 5. 最终模块边界

目标包结构如下：

```text
vllm/pap/
├── config.py             # 唯一配置入口和支持的部署 profile
├── protocol/             # wire messages、descriptors、错误模型
├── topology/             # xPAyP、route plan、peer membership
├── lifecycle/            # session、decode token、commit、lease、drain
├── kv/                   # unified KV、sealed handoff、paged metadata
├── attention/            # append、FlashAttention、direct/combine execution
├── transport/
│   ├── base.py           # backend contract
│   ├── local_fast/       # 同机 backend
│   └── nixl/             # 跨机 backend
├── observability/        # 不改变执行语义的 audit、metrics 和 tracing
└── service.py            # HTTP/TCP endpoints、依赖装配、服务生命周期
```

依赖方向必须保持单向：

```text
service/config
      ↓
protocol + topology
      ↓
lifecycle + kv
      ↓
attention execution
      ↓
transport backend
```

具体约束：

- 核心模块不得直接读取环境变量；环境变量只在 launcher/config composition root
  解析一次；
- `attention_executor.py` 在本 milestone 保留为薄兼容入口，只做 re-export 和服务
  启动；
- Qwen3、scheduler 和 model runner 只保留 PAP adapter，不各自实现 PAP 状态机；
- transport-specific 对象不得泄漏到 Attention、KV 或 lifecycle 接口；
- service 显式构造和注入依赖，不使用散落的进程级隐式单例；
- Registry 锁只保护状态转换和 snapshot，CUDA copy、event wait 和 kernel launch
  必须在锁外执行。

## 6. 主路径收敛

### 6.1 固化为唯一实现

| 当前分支或开关 | 目标状态 |
| --- | --- |
| sync/async decode token | 删除 sync，固定 async |
| sync/async Prefill KV import | 删除 sync，固定安全异步 import |
| layer descriptor/sealed manifest | 删除 layer descriptor，固定 sealed handoff |
| unified/non-unified KV | 删除 non-unified 路径 |
| batched/per-row route copy | batched 为主路径；仅保留输入决定的正确性 fallback |
| metadata fast-key on/off | 删除 off 分支，固定 fast-key |
| dynamic/static MPS testbed | 删除 dynamic，P17 static 64/28 为正式 profile |
| legacy/FIFO/combine dispatcher | 改为 topology 派生的 direct/combine executor |
| active-peer tracking flag | multi-Projection 自动启用 |
| async TTFT barrier/gate/isolation | 删除 |
| rejected adaptive coalesce | 删除 |

`local_fast`、NIXL 和 `xPAyP` 不是历史实验分支，而是保留的部署能力。

本设计取代
`2026-07-13-pap-static-mps-benchmark-design.md` 中为通用 runner 保留 dynamic
fallback 的当前默认决策；旧文档继续作为当时实验背景，不回写成新结论。

### 6.2 删除判定

删除一条路径必须同时满足：

1. 实验索引明确记录为 rejected、rolled-back 或 superseded；
2. P17 canonical profile 不使用该路径；
3. 其正确性不变量已由 surviving tests 覆盖；
4. 删除后通过 targeted tests、完整 CPU suite 和 P17 C1 quick。

已知 retired 环境变量不得被静默忽略。`PAPRuntimeConfig` 应 fail fast，并说明替代
行为和对应历史 experiment ID。

### 6.3 未验证能力的限制

NIXL backend 在本 milestone 作为兼容性隔离模块处理。允许机械迁移和接入共同
interface，但不进行无法通过跨机 E2E 验证的内部算法清理。`xPAyP` 保留 route、
membership 和 direct/combine contract，但不作新的运行性能或正确性结论。

## 7. 请求生命周期与错误模型

请求状态机为：

```text
CREATED
   ↓
CATALOG_BOUND
   ↓
MANIFEST_READY
   ↓
ACTIVE
   ↓
DRAINING
   ↓
RELEASED

不可恢复错误 → FAILED → 幂等清理
```

P17 主路径按以下顺序执行：

1. Proxy 通过 `TopologyPlan` 选择 PA 和 Projection；
2. Prefill 创建 Prefill-owned unified KV 并注册静态 layer catalog；
3. Prefill 发布 request-scoped sealed manifest 和 CUDA ready event；
4. Projection 逐层计算 QKV；
5. transport 将 QKV 发送到 Attention；
6. Attention 等待完整 prefix readiness，append K/V 并运行 paged FlashAttention；
7. Attention output 返回 Projection；
8. sampled token 通过 async delivery 与 KV-ready join；
9. decode commit 通过 ACK/watermark 更新 Prefill ledger 和 APC；
10. 结束时依次 flush token、flush commit、release lease 和 drain session。

核心抽象职责固定为：

```text
PAPRuntimeConfig
  topology, placement, transport, resource partition,
  timeouts, capacities, protocol version

PAPTransport
  bind(), send_qkv(), receive_qkv(), send_attention_output(), close()

PAPSessionRegistry
  register(), bind_catalog(), publish_manifest(), claim_for_decode(),
  begin_drain(), release()

PAPAttentionExecutor
  execute_direct(), execute_combined()

PAPLifecycleCoordinator
  join_token_and_kv(), commit_and_ack(), release_lease(), drain_session()
```

实施计划可以根据现有类型细化参数和返回值，但不得移动上述 ownership，也不得将
这些职责重新合并进单一 executor。

以下情况统一 fail closed：

- protocol version 不匹配；
- manifest 不完整或 session epoch 过期；
- route/cohort 不一致；
- sampled token 与 KV progress 不一致；
- commit ACK、lease release 或 drain 超时；
- transport buffer、shape、dtype 或 sequence metadata 不一致。

失败不得静默切换旧算法或另一 transport。session epoch、manifest generation 和
request ID 共同防止 stale update/ABA；失败进入 `FAILED`，记录结构化错误并执行
幂等清理。

## 8. 测试治理

### 8.1 判定标准

测试不按作者来源或数量判断价值，而按其是否保护真实 PAP 不变量判断。每个现有
测试归入以下处置之一：

- `keep`：唯一保护正式行为、错误语义或已发生 regression；
- `merge`：多个测试保护同一不变量，可参数化；
- `rewrite`：目标必要，但依赖源码字符串或过度 mock 私有实现；
- `delete`：只保护已删除路径，或被更强测试完全覆盖；
- `move`：测试必要，但处于错误层级。

### 8.2 目标结构

```text
tests/pap/
├── unit/
│   ├── protocol/
│   ├── topology/
│   ├── lifecycle/
│   ├── kv/
│   ├── attention/
│   └── transport/
├── contract/
│   ├── test_vllm_integration.py
│   ├── test_launcher_profile.py
│   └── test_transport_contract.py
├── integration/
│   └── test_attention_service.py
├── gpu/
│   ├── test_local_fast.py
│   └── test_nixl.py
└── fixtures/
```

`tests/pap/invariants.json` 使用版本化 schema 记录 invariant ID、陈述、测试层级、
源码 owner、pytest node IDs、regression commit/experiment ID 和 required 状态。

### 8.3 测试规则

- 源码字符串断言优先改为行为或结构化配置测试；
- shell 只保留语法和端到端装配检查，配置逻辑进入 `PAPRuntimeConfig`；
- 公共 fake、request builder 和 registry setup 收敛到 fixtures；
- CPU unit/contract/integration suite 是每个提交的必跑 gate；
- CUDA tests 单独标记，缺少硬件时明确 skip；
- P17 C1/C4 由 benchmark runner 和严格 audit 负责，不混入 pytest；
- coverage 只用于发现缺口，不设置人为百分比目标；
- 不设置必须删除的测试数量目标。

测试审计必须输出逐项 disposition 和理由。删除测试时，对应不变量必须随旧路径
一起消失，或明确映射到更强的 surviving test。

## 9. 实验治理

### 9.1 四层模型

```text
immutable raw run
      ↓
versioned run manifest
      ↓
tracked experiment registry
      ↓
generated index/report
```

建议结构：

```text
benchmarks/pap/
├── profiles/
│   └── p17_1pa1p.toml
├── schemas/
│   ├── run_manifest.schema.json
│   └── experiment_record.schema.json
├── registry/
├── validate_registry.py
├── import_legacy_run.py
└── generate_experiment_index.py
```

### 9.2 Run manifest

所有新实验必须记录：

- schema version、stable experiment ID 和 run ID；
- branch、完整 commit、tracked dirty 状态和 patch hash；
- workload、model、dtype、TP、topology、placement 和 transport；
- MPS、PAP runtime profile、hardware/runtime fingerprint；
- repetition、顺序和 aggregation method；
- correctness、routing、commit、lease、join 和 drain 状态；
- metrics、logs 和 trace/profile 的相对路径；
- evidence grade、validity 和明确失败原因。

### 9.3 Experiment record

tracked record 记录 hypothesis、baseline/treatment、唯一主变量、run IDs、聚合方法、
最小指标、结论、决策、替代关系和 raw artifact references。

Evidence 枚举统一为：

- `formal-clean`
- `controlled`
- `diagnostic`
- `smoke`
- `historical`
- `invalid`

Decision 枚举统一为：

- `accepted`
- `optional`
- `rejected`
- `rolled-back`
- `superseded`
- `inconclusive`

### 9.4 历史回填范围

只对当前 `pap-experiment-history-index.md` 中以下集合进行 A 级完整迁移：

1. 第 6 节实验账本中的每个稳定 experiment ID；
2. 第 7 节中的每个 negative-result ID；
3. 第 1.2 节当前正式证据摘要引用、但尚未被前两项覆盖的 experiment ID。

模块档案、时间线或背景段落中仅作为路径示例出现的其他 raw run 不在回填范围。
每个纳入记录需要人工复核 metrics、commit、配置、audit、证据等级、结论和替代
关系。

其他未进入当前索引的 raw directories 原地保留，本 milestone 不登记、不移动、
不改写。历史缺失字段必须显式记为 `missing`，不得推测补全。

外部 raw 路径使用 `root_id + relative_path`。legacy importer 只生成新的 tracked
record；新 runner 缺少 required metadata 或 strict audit 时 fail closed。

历史索引第 6、7 节的表格由 registry 生成，并使用生成区域标记防止手工编辑。
人工维护的模块解释和时间线可以保留，但引用实验数字、状态和替代关系时必须来自
canonical record。纳入 A 级回填的旧结果文档增加统一状态头，包含当前状态、证据
等级、canonical experiment ID 和 `superseded_by`。

## 10. 文档治理

### 10.1 目标结构

PAP 的长期文档统一进入：

```text
docs/design/pap/
├── README.md                 # 唯一入口、当前状态和文档地图
├── architecture.md           # 当前唯一有效的 PAP 架构
├── runtime.md                # runtime 模块、状态机和接口
├── benchmark-methodology.md  # workload、证据等级和统计规则
├── experiment-index.md       # experiment registry 生成的索引
├── milestones/               # 已冻结里程碑的设计与最终结果
└── archive/
    ├── designs/              # 被替代但有历史价值的设计
    ├── experiments/          # 历史实验报告
    └── reports/              # 周报、handoff 和阶段报告
```

`README.md` 是读者的唯一首入口。current 文档不得要求读者先阅读 archive 或 Git
history 才能理解当前 PAP。

### 10.2 文档状态

PAP 文档使用机器可读的状态头：

```yaml
---
pap_doc_schema: 1
status: active | current | milestone | archived | superseded
canonical: relative/path/or/null
superseded_by: stable-milestone-or-document-id-or-null
related_experiments: []
last_validated_commit: full-git-commit-or-null
---
```

- `active`：已经批准、仍在实施或验证中的 milestone 设计；
- `current`：当前唯一有效的架构或方法；
- `milestone`：某个已冻结阶段的设计与最终结果；
- `archived`：仍有历史价值，但不再代表当前实现；
- `superseded`：已被明确替代，必须提供 `canonical` 或 `superseded_by`。

archive 保留历史原文；除增加状态头、修复链接和明显的事实标注外，不把旧文档
重写成当前结论。

本设计在实施期间保持 `active`；只有 Final freeze 全部通过后才能改为
`milestone`。

### 10.3 `docs/superpowers` 迁移

本项目后续不再使用 Superpowers workflow，也不再创建
`docs/superpowers/specs` 或 `docs/superpowers/plans`。

- 仍有效的 spec 内容合并进 `architecture.md`、`runtime.md` 或 milestone 文档；
- 已被替代但有历史价值的 spec 移入 `archive/designs`；
- implementation plan 不整体迁入 archive，只提取已经发生的决策、验证命令和
  结果；
- 完成内容提取并更新全部 inbound links 后，删除 `docs/superpowers/`；
- 删除使用独立 commit，删除前必须证明 tracked 文档和源码中不存在
  `docs/superpowers`、`superpowers/specs` 或 `superpowers/plans` 引用。

Git history 是旧 implementation plans 的最终回溯入口。

### 10.4 临时实施计划

本 milestone 的 implementation plan 写入
`dev-memory/2026-07-14_PAP_runtime重构实施计划.md`。它是未跟踪的本地执行清单，
不得成为 canonical documentation，也不作为最终交付物。计划完成后可以删除。

### 10.5 现有 `docs/design` 分类

每个现有 PAP 文档必须进入迁移清单，记录原路径、文档类型、状态、目标路径、
canonical successor 和关联 experiment IDs。处理规则为：

- 当前有效架构合并到少量 current 文档；
- 已冻结的阶段设计和最终结果进入 `milestones`；
- 旧实验报告进入 `archive/experiments`，结论以 registry 为准；
- handoff、idea book、weekly report 和旧 stage report 进入 `archive/reports`；
- superseded 文档必须指向 current 文档或 successor milestone；
- 历史索引不得继续直接引用 implementation plans；
- raw results、service logs 和 profiler artifacts 不进入 `docs`。

### 10.6 文档 Gate

文档迁移完成必须满足：

- `docs/design/pap/README.md` 能导航全部 current、milestone 和 archive 类别；
- 每份 PAP 文档有合法状态头；
- current 事实、指标和决策只有一个 canonical source；
- registry 生成的索引与 tracked records 一致；
- 所有相对 Markdown 链接存在；
- tracked tree 中不存在指向 `docs/superpowers` 的引用；
- `docs/superpowers/` 已删除；
- 历史 raw data 未移动或改写。

## 11. 验收标准

### 11.1 阶段 Gate

| 阶段 | Gate |
| --- | --- |
| Freeze | tracked-clean P17 C4 formal 三轮和全部 strict audit |
| 测试治理 | PAP CPU suite 和 invariant registry validation |
| 每类路径删除 | targeted tests、完整 CPU suite、P17 C1 quick |
| 每个主要模块拆分 | import/contract tests、完整 CPU suite、P17 C1 quick |
| 实验治理 | schema、A 级历史记录和生成索引校验 |
| 文档治理 | 状态 schema、链接、canonical source、零 superpowers 引用 |
| Final freeze | 完整测试、P17 C1、P17 C4 formal 三轮、前后报告 |

### 11.2 正确性

- 所有请求成功，failed 为零；
- output digest 一致；
- cache transition、decode-token join 和 routing 通过；
- commit/ACK watermark 和 lease accounting 正确；
- drain 后 active sessions 为零；
- 无隐藏 OOM、timeout、stale update 或 server error。

### 11.3 性能

使用三轮中位数比较 R1 TTFT、R1 TPOT、steady TTFT 和 steady TPOT。任一指标相对
pre-refactor baseline 回退超过 `5%` 时复跑一次；复跑仍超过即视为 regression。
对应提交必须定位和修复或回退，不得通过恢复旧 runtime flag 掩盖问题。

## 12. 非目标

- 新性能优化；
- 新 topology 或新模型支持；
- xPAyP、TP 或多机 NIXL 的重新跑数；
- 修复重构前已存在的跨机性能问题；
- 移动或改写历史 raw data；
- 为测试数量或 coverage 百分比制造测试；
- 在文件移动中改变数学、协议或调度行为。

## 13. 交付物

1. 当前 HEAD 的 P17 pre-refactor formal baseline；
2. PAP 测试不变量账本和逐项 disposition；
3. 删除历史分支后的唯一 P17 runtime 主路径；
4. 模块化 `vllm.pap` package；
5. local-fast/NIXL backend contract；
6. 关键历史实验的 versioned registry；
7. 自动生成和校验的实验索引；
8. post-refactor formal 结果与前后对比报告；
9. canonical PAP 文档、milestone/archive 分类和迁移清单；
10. 已删除 `docs/superpowers` 且不存在残留引用；
11. 当前能力、未验证能力和后续工作说明。

## 14. 风险与回退

- 每类删除和每个主要模块拆分使用独立 commit，便于精确回退；
- benchmark artifacts 保持 untracked，不与源码提交混合；
- NIXL/xPAyP 不做未经验证的行为清理；
- P17 C1 在每个行为或结构边界后运行，C4 formal 只用于首尾冻结；
- 如果 pre-refactor freeze 本身不稳定或 strict audit 失败，重构不得开始；
- 回退通过 Git commit 完成，不重新引入长期实验开关。
