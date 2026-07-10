# PAP xPAyP 与多轮全量 KV 命中设计

日期：2026-07-10
状态：Phase 1 xPAyP 已实现并验证；多轮 resident KV 尚未实现
基线提交：`87bb1061f6b91b4989cc29330dff0cf133290088`

## 1. 目标与结论

本文设计两个相互依赖的能力：

1. 支持任意 `xPAyP` 拓扑，即 `x` 个 Prefill+Attention（PA）组和
   `y` 个 Projection（Proj）组；
2. 支持多轮对话，使第 2 轮进入 Prefill 时，第 1 轮中属于下一轮历史前缀的
   prompt KV 和 decode KV 全部在原 PA 节点本地命中，不回传 KV、不重算历史
   token，只计算新增的对话 suffix。

推荐路线是：

- 在代理/调度层维护 conversation 到 PA 的亲和关系，而不是把 conversation
  placement 塞入 Prefill、Attention 或 Projection 计算进程；
- 将当前 request 级统一 KV lease 提升为 conversation 级 resident KV owner；
- 通过 scheduler resident-attach 直接把 PA 上仍存活的 block table 交给下一轮，
  不依赖普通 Automatic Prefix Caching（APC）的 full-block hash 命中；
- Prefill 与 Attention 继续共享同一套 Prefill-owned paged KV，Projection 保持
  KV-unaware；
- 为最后一个可见生成 token 增加 cache-only closure forward，否则严格意义上
  无法做到“第 1 轮所有 decode token 都命中”；
- 第一阶段采用稳定的 `PA -> Projection` 静态亲和，先让任意 x/y 数量正确运行；
  后续再把 Attention 的单 peer transport 扩展为 lazy full crossbar，使同一个 PA
  可被任意健康 Projection 服务。

这不是简单地恢复历史上的 conversation stickiness。仅把下一轮路由回同一 PA、
或复用旧 `pap_prefill_kv_handle`，都不足以让新 request 安全接管旧 request 的
partial block、token ledger、lease 和跨进程同步状态。

## 2. 当前基线与已验证范围

### 2.1 2026-07-10 xPAyP 实施进展

Phase 1 已完成以下实现：

- decode commit 与 lease release 按 Attention session 的 Prefill endpoint
  路由，不再把 PA0 环境变量作为多 PA 真值；
- Attention 按 Projection peer metadata lazy 创建独立 mailbox transport；
- Projection mailbox actor id 包含 Projection 实例、TP rank 和 Attention
  endpoint hash，避免多实例冲突；
- launcher 与自包含 benchmark runner 支持任意正整数 `<x>pa<y>p`；
- runner 记录 topology manifest，并逐 PA 审计 routed request、decode commit、
  lease release 和 Attention session drain。

Qwen3-8B、TP1 的实机 correctness 矩阵已通过 `1PA2P`、`2PA1P`、
`2PA2P`、非整除 `3PA2P`，并额外通过固定 70/30 MPS 的 `2PA1P`。
`3PA2P` 中每个 Attention 均成功绑定两个 Projection peer。GPU0 有无关服务，
因此本轮未运行需要 8 张空闲 GPU 的 `6PA2P`。

以下原始设计章节保留实现前审计背景；其中“当前”描述应结合本节实施状态阅读。

最新数据路径基线已经完成清理、测试和提交：

- focused PAP suite：`155 passed, 2 skipped`；
- 标准 `1PA1P`、Qwen3-8B、input 128、output 32、QPS 16：128/128 成功；
- median/p99 TPOT：`35.10 / 40.23 ms`；
- median TTFT：`1546.59 ms`；
- fast path、slot plan、session drain 和 correctness audit 均通过；
- 低负载 QPS 4 时 PAP/PD median TPOT 为 `1.146x`。

这些是 Phase 1 实施前的 `1PA1P` 单轮基线；xPAyP 的新增验证结果见 2.1 节。
多轮负载仍未通过，不能由单轮 xPAyP 结果外推。

## 3. 源码审计得到的现状

### 3.1 路由仍以 request 为单位

`examples/pap/multi_pap_proxy_server.py` 的 `select_instances()` 只使用递增的
`request_number`：

- PA 按 request round-robin；
- Projection 按 `round_robin`、`projection_affinity` 或
  `projection_sticky` 选择；
- `conversation_id` 只被传给 Attention registration；
- 代理没有 conversation directory，测试还明确断言当前不保存 conversation
  placement。

因此，同一 conversation 的第 2 轮没有任何保证返回第 1 轮 PA。

### 3.2 生命周期仍以 request 为单位

当前请求结束时，代理在 `finally` 中删除：

```text
/v1/pap/attention/sessions/{request_id}
```

Attention 的 `release_session()` 随后：

1. 删除 request 级 session 和 unified paged state；
2. flush decode commit；
3. 向 Prefill 释放 request 级 KV lease；
4. 忘记 commit client 的 request watermark。

这条路径适合单轮 drain，却与 conversation 级复用相冲突。多轮模式下，请求结束
应当结束 turn，而不是销毁 conversation session。

### 3.3 unified KV 已经具备正确的数据面基础

当前 unified-KV 路径已经实现了多轮方案最关键的一半：

- Prefill 拥有 vLLM paged KV block；
- Attention 通过 CUDA IPC 打开 Prefill 的 KV tensor；
- decode K/V 由 Attention 直接写入 Prefill-owned block；
- Prefill 收到 decode commit 后更新 token ledger、computed watermark 和 block
  hash；
- request 结束时，active lease 可以阻止 scheduler 把 block 立即还给 block pool。

所以不需要再引入 Attention 私有 KV 副本，也不需要把 decode KV 拉回 Prefill。
缺失的是 conversation 级所有权、下一轮 attach、严格 commit 和路由协议。

### 3.4 普通 APC 不能满足“全部 token 命中”

当前 decode commit 会把已物化的完整 block 放入 prefix-cache hash 表；释放 lease
后，这些 block 也可能暂时留在 LRU 中。因此，同 PA 的下一请求可能偶然得到一部分
APC 命中，但不能把它作为多轮语义：

- APC 只匹配完整 block；partial tail block 不可命中；
- `get_computed_blocks()` 为了得到 logits，最多命中 `prompt_length - 1`，并受
  block-size 对齐限制；
- lease 释放后 block 只是 eviction candidate，不能保证仍驻留；
- APC 没有 conversation version、单写者和精确 token ledger 语义。

严格全量命中必须走显式 resident-attach，而不是依赖 APC。

### 3.5 当前 xPAyP 有两个控制/传输边界

第一，launcher 给所有 Attention 进程注入同一个：

```text
PAP_DECODE_COMMIT_ENDPOINT=http://127.0.0.1:${PREFILL_PORT_BASE}/...
PAP_LEASE_RELEASE_ENDPOINT=http://127.0.0.1:${PREFILL_PORT_BASE}/...
```

也就是默认都指向 PA0 的 Prefill。`PA1...PAx` 的 Attention 不能继续使用进程级
固定 endpoint，必须按 session 的 `prefill_endpoint` 动态路由 commit 和 release。

第二，Projection 已经按 Attention endpoint 缓存独立 transport，能够从一个
Projection fan-out 到多个 PA；但每个 Attention executor 目前只拥有一个 transport，
且 bind API 在首次 peer bind 后忽略后续 peer。也就是说，同一 Attention 当前不能
安全地被多个 Projection 轮流连接。

### 3.6 当前 commit watermark 不是严格的全层提交

Attention 在每层完成 unified KV append 后都会尝试发 decode commit；进程级 client
再按 `new_seq_len` 去重。结果通常是第一个完成该 token 的 layer 触发 session-level
watermark，而不是所有 layer 都完成后再提交。

单轮请求中，最终响应返回前所有层仍会完成，所以暂未暴露为结果错误。但多轮 resident
attach 需要明确的发布屏障：只有所有 layer、所有 TP rank 的目标 token 都已物化，
下一轮才可以观察到新的 committed watermark。

### 3.7 最后一个生成 token 的 KV 边界

自回归模型在 step `n` 中用 token `t[n-1]` 做 forward，得到并采样 `t[n]`。因此：

- `t[n-1]` 的 KV 在该 forward 中被写入；
- 刚采样出的 `t[n]` 只有在下一次 forward 被作为输入时才会产生 KV；
- 如果请求在 `t[n]` 采样后结束，最后一个可见 token 的 KV 尚不存在。

所以仅保留当前 decode block 仍只能保证到倒数第一个输出 token。若验收标准是第 1 轮
所有可见 decode token 在第 2 轮全部命中，必须增加一次不采样、不返回 token 的
cache-only closure forward。

## 4. 设计不变量

实现必须维护以下不变量：

1. `conversation_id` 是路由键，`session_id` 是内部 KV ownership capability，
   `request_id` 只标识一次 turn；三者不能混用。
2. 同一 conversation 同时最多有一个写 turn。
3. committed watermark 之前的每个 token，所有 layer 和 TP rank 的 KV 都已物化。
4. 第 2 轮 attach 前，输入 token 前缀必须与 conversation token ledger 完全一致。
5. 未通过 token、model、layout、epoch 校验时必须 fail closed，回退为完整 Prefill，
   不能继续读取可能错误的 KV。
6. Projection 不拥有持久 KV；切换 Projection 不改变 conversation KV ownership。
7. conversation 处于 IDLE 时，PA resident owner 仍持有 block 引用；普通 request
   finish 不得释放该 owner 引用。
8. 只有显式关闭、TTL/LRU eviction、PA failure 或 capacity eviction 才释放
   conversation lease。
9. 对客户端已承诺的 turn version 必须对应一个完整 committed watermark；失败或
   中断的下一 turn 不得覆盖上一稳定 version。
10. 任何 partial layer write 都只存在于 uncommitted tail，重试可以覆盖它，但不能
    被下一轮读取。

## 5. 目标架构

```text
                            +-------------------------+
Client ------------------->| PAP Router / Proxy      |
 conversation_id           | ConversationDirectory   |
                            +-----+---------------+---+
                                  |               |
                         PA affinity         Projection choice
                                  |               |
                 +----------------v--+       +----v-------------+
                 | PA group k         |<----->| Projection j     |
                 |                    | QKV/O | KV-unaware       |
                 | Prefill vLLM       |       +------------------+
                 | - resident owner   |
                 | - token ledger     |
                 | - block attach     |
                 |                    |
                 | Attention rank(s)  |
                 | - IPC KV view      |
                 | - decode append    |
                 | - step barrier     |
                 +--------------------+
```

PA 是持久 KV placement 单元；Projection 是可替换的计算资源。第一轮决定 PA，后续
turn 优先回到同一 PA。Projection 在逻辑上不需要 conversation stickiness，但受当前
单 peer transport 限制，MVP 先使用稳定 PA-to-Projection 映射。

## 6. 核心数据模型

建议新增独立的 `ConversationDirectory` 接口。原型可在 proxy 进程中使用内存实现，
但接口不能与 FastAPI handler 状态耦合，以便后续替换成 Redis/etcd 或独立 scheduler。

```python
@dataclass
class ConversationRecord:
    conversation_id: str
    session_id: str
    pa_group_id: str
    pa_incarnation: int
    version: int
    state: Literal["active", "closing", "idle", "evicting", "lost"]
    active_request_id: str | None
    projection_id: str | None
    committed_token_count: int
    committed_token_digest: str
    lease_expires_at: float
```

PA 内部 resident owner 维护更详细的状态：

```python
@dataclass
class ResidentConversationKV:
    session_id: str
    conversation_epoch: int
    model_fingerprint: str
    cache_layout_fingerprint: str
    token_ids: tuple[int, ...]
    committed_seq_len: int
    emitted_history_seq_len: int
    block_ids_by_group: tuple[tuple[int, ...], ...]
    materialized_seq_len_by_layer_rank: dict[tuple[int, str], int]
    turn_fences_by_rank: tuple[object, ...]
    owner_state: Literal["idle", "attached", "evicting"]
    active_request_id: str | None
    expires_at: float
```

必须同时保留 `emitted_history_seq_len` 和 `committed_seq_len`。在 closure 完成前，
前者可能比后者大 1；该 conversation 不能进入可复用 IDLE 状态。

`model_fingerprint` 至少包含 model revision、tokenizer/chat-template revision、TP size、
KV dtype、block size、attention layout 和相关 RoPE 配置，避免服务滚动升级后误用旧 KV。

## 7. xPAyP 路由设计

### 7.1 第一轮选择

第一轮 PA 选择使用“负载过滤 + rendezvous hash”两阶段算法：

1. 过滤 unhealthy、KV 水位过高、无法容纳预计 prompt+output 的 PA；
2. 在剩余 PA 中，用 `conversation_id` 做 rendezvous hash，并把 queue depth、可用
   KV blocks、same-node/NVLink locality 作为权重修正。

这样在服务拓扑稳定时 placement 可预测，扩缩容时只迁移较少 conversation，同时
仍避免把新会话压到高负载 PA。

没有 `conversation_id` 的请求继续使用普通无状态负载均衡，并在 turn 结束后释放。

### 7.2 后续 turn 选择

对于已有 record：

1. 校验 PA incarnation 和 session epoch；
2. 若原 PA 健康且 resident owner 命中，强制路由回原 PA；
3. 若原 PA 丢失或 session 已 eviction，增加 epoch/version，在其他 PA 做完整 Prefill；
4. MVP 不做跨 PA live migration；后续可接 NIXL migration backend。

“同 PA 命中”优先于短时 queue-depth 均衡。否则每轮迁移/重算会抵消多轮缓存收益。

### 7.3 Projection 选择

分两阶段实施：

#### 阶段 A：静态 PA-to-Projection affinity

```text
home_projection(pa_id) = stable_hash(pa_id) % y
```

或者用 `projection_affinity` 将连续 PA 均匀分片给 Projection。该方案已经能支持
任意 x/y 数量，且满足当前 Attention 只 bind 一个 peer 的约束。

优点：改动小、连接数 O(x)、容易先验证 correctness。
缺点：Projection 故障和负载倾斜时弹性有限。

#### 阶段 B：lazy full crossbar

Attention 将单一 `offload_exec_transport` 改成：

```text
transport_by_projection[(projection_id, tp_rank)]
```

每个 peer 使用独立 actor id、ring、doorbell 和 mailbox loop；首次使用时 lazy bind，
空闲连接按 LRU 回收。此后 Projection 可以按实时负载独立选择，同一 PA 不再绑定某个
Projection。

连接内存从 O(x) 上升到最坏 O(x*y*tp*slots)，因此不能在启动时无条件预分配完整
crossbar。same-node local-fast 优先；不在同一主机或 P2P 不可用时回退 NIXL。

### 7.4 每 session 动态控制 endpoint

Attention registration 已携带 `prefill_endpoint`，应把它作为 session control route：

- `DecodeCommitClient.commit(..., endpoint=session.prefill_endpoint)`；
- lease/end-turn/release 同样按 session endpoint；
- client 连接池按 Prefill base URL 缓存；
- 不再使用指向 PA0 的进程级 `PAP_DECODE_COMMIT_ENDPOINT` 作为多 PA真值；环境变量
  只保留单 PA fallback。

TP 模式下建议只由 rank 0 发布 control commit，但必须在 commit 前确认所有 rank 的
GPU fence 已记录。也可以先由 PA control coordinator 聚合各 rank ACK，再由 coordinator
向 Prefill 发一次 commit。

## 8. conversation KV 生命周期

建议状态机如下：

```text
ABSENT
  | create + first prefill
  v
ACTIVE(turn N)
  | all-layer/rank commit + final-token closure
  v
CLOSING
  | fence published + owner detached from request
  v
IDLE(version N)
  | begin_turn CAS(version N)
  +--------------------------> ACTIVE(turn N+1)
  |
  | delete / TTL / pressure / PA loss
  v
EVICTING -> ABSENT or LOST
```

同一 conversation 的 `begin_turn(expected_version)` 是 CAS 操作：

- IDLE 且 version 匹配：进入 ACTIVE；
- 已 ACTIVE：排队或返回 409，MVP 推荐 409；
- version 落后：返回 stale-parent 409；
- session/epoch 不匹配：返回 cache-miss，让 proxy 重建 placement。

## 9. block 所有权与 resident-attach

### 9.1 为什么不能只保存 block id

block id 会在 block pool 中复用。必须由 resident owner 持有实际 block reference/refcount
并带 PA incarnation/epoch，才能避免 ABA：旧 session 保存的 block id 在 eviction 后
可能已经装入另一请求的 KV。

### 9.2 推荐的 ownership transfer

每个 conversation 只允许单写者，因此可采用“owner ref 在 IDLE 与 request 之间转移”
的方式，避免每轮额外 touch/free：

1. 第 1 轮 request 分配 block，结束时不 free；scheduler 将 request block table 从
   request bookkeeping 中 pop，转交 resident owner，保留原 refcount；
2. 第 2 轮 begin-turn 时，resident owner 把同一批 block ref 移入新 request 的
   `req_to_blocks`，不增加 refcount；状态从 IDLE 变为 ATTACHED；
3. scheduler 在尾部按需分配新 block；
4. turn 完成后把完整 block table 再转回 resident owner；
5. eviction 时才真正调用 `block_pool.free_blocks()`。

这种 move semantics 比“owner 始终持一份 ref，新 request 再 touch 一份 ref”更简单，
也减少 refcount 错误。前提是同一 conversation 不允许并发 turn。

### 9.3 scheduler attach API

需要增加显式内部 API，例如：

```python
kv_cache_manager.attach_resident_blocks(
    request=request,
    blocks=resident.blocks,
    num_computed_tokens=resident.committed_seq_len,
)
```

该 API 需要：

- 对每个 KV cache group 安装旧 block table；
- 恢复 `num_cached_block` 等 manager bookkeeping；
- 设置 `request.num_computed_tokens = committed_seq_len`；
- 保留 partial tail block，而不是只接 full blocks；
- 让后续 suffix 可以继续写 partial tail；
- 校验完整 block 的 block hash，校验 partial tail 的精确 token ledger；
- 支持 turn 结束时把旧+新增 block 一次性 detach 回 owner。

这条路径只对通过 session capability 和 epoch 校验的 PAP 请求开放，不能变成任意客户端
可指定 block id 的公共接口。

### 9.4 容量管理

当前 `PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS=32` 只适合固定单轮 output 32。多轮模式
需要在每轮边界动态扩容：

- begin-turn 先为 `new_suffix + max_tokens + closure_token` 做 admission；
- decode 期间可按 block chunk 增长，但必须在进入 token step 前完成 block 分配和
  Attention topology update；
- IDLE conversation 进入独立 LRU/TTL 队列；
- 设置 per-tenant/per-conversation token 或 block quota；
- 内存压力下只 eviction IDLE session，绝不回收 ACTIVE session。

## 10. 两轮请求的数据流

### 10.1 第 1 轮

1. Proxy 根据 conversation_id 选择 PA group 和 home Projection。
2. 创建随机、不可猜测的 `session_id`，记录 PA incarnation 和 version 0。
3. 在各 Attention rank 注册 conversation session；注册只执行一次，不随 request 结束删除。
4. Prefill 对 prompt 做正常计算，分配并预留 decode block。
5. Prefill 把 paged KV tensor IPC descriptor、block table 和 session epoch 发布给 Attention。
6. Projection 开始 decode；每层 Q/K/V 发给目标 Attention。
7. Attention 把当前输入 token 的 K/V 写入 Prefill-owned block，再计算 attention。
8. StepCommitTracker 等待该 token 的所有 layer、所有 TP rank 完成。
9. Prefill owner 原子追加 token IDs，并推进 committed watermark。
10. 最后一个可见 token 采样后执行 closure forward，使其 K/V 物化但不再采样输出。
11. 各 rank 记录 turn fence；Prefill owner 接管 block table，session 进入 IDLE version 1。
12. Proxy 返回/完成响应，但不 DELETE Attention session、不 release conversation lease。

### 10.2 第 2 轮

1. Proxy 用 conversation_id 查 directory，并 acquire version 1。
2. 请求仍路由到原 PA；Projection 使用 home Projection 或 crossbar 中的最优实例。
3. Prefill tokenize 新一轮请求，并验证前 `committed_seq_len` 个 token 与 resident token
   ledger 完全一致。
4. Prefill stream wait 上一 turn 的 IPC fence，不做 host-wide/device-wide synchronize。
5. scheduler 把 resident block table attach 给新 request，设置 computed watermark。
6. 只计算新增的 assistant-close/user/assistant-open 等模板 token 和新 user content；历史
   token 的重算数必须为 0。
7. Prefill 对新增 suffix 的 KV 仍写入同一套 block，Attention 只更新 block table 和
   seq_len，无需复制历史 KV 或重新打开整块 KV tensor。
8. Projection/Attention 执行第 2 轮 decode，重复 commit、closure 和 detach。

## 11. 精确 token 验证与 Chat Completions

### 11.1 全历史兼容模式

现有 multi-turn benchmark 会发送完整 `messages`，并可同时发送：

- body `conversation_id`；
- header `X-Session-ID`。

兼容模式下 Prefill 仍按正常 Chat Completions 方式 tokenize，然后比较：

```text
incoming_prompt_token_ids[:committed_seq_len] == resident.token_ids
```

相等才允许 resident-attach；不等则记录 mismatch reason，并完整 Prefill。不能只比较
conversation_id、文本或长度。

### 11.2 严格保证模式

文本 decode 后重新 tokenize 可能因 BPE 边界、空白或 chat template 变化而不能还原
原始 sampled token IDs。要对所有模型严格保证 100% history hit，推荐增加 stateful
delta 协议：

- 客户端发送 `conversation_id + parent_version + new user message`；
- PA 保存前一轮精确 sampled token IDs；
- 服务端基于已保存 token ledger 构造 continuation，只 tokenize 新增模板/suffix；
- 返回新 version 给客户端。

MVP 可先支持 Qwen3 的 full-history verified 模式；严格 API 随后补充。无论哪种模式，
chat template 在模型升级后变化都会使 fingerprint 失效并触发 miss。

### 11.3 “全部命中”的边界定义

验收中的第 1 轮 history 包含：

- 第 1 轮全部 prompt token；
- 会出现在第 2 轮输入前缀中的全部第 1 轮可见 assistant token。

第 2 轮新加入的 assistant end marker、user role marker、新 user content 和下一次
assistant generation marker 属于新 suffix，不计入历史命中分母。EOS 若不会进入下一轮
chat prompt，也不要求作为历史 token；以精确 token ledger 与模板构造结果为准。

## 12. final-token closure 设计

推荐在 Projection vLLM scheduler 中加入 PAP 专用的 cache-only finish 状态：

```text
RUNNING
  -> sample final visible token
  -> PAP_FINALIZING_KV
  -> schedule sampled token once as model input
  -> run all layers / remote Attention append
  -> discard logits, do not sample, do not emit another token
  -> FINISHED
```

closure 需要满足：

- 不改变客户端可见 token、finish_reason、usage 和 sampling RNG 序列；
- 对 stop、EOS、length、abort 分别定义行为；
- 在正常完成的 turn 上，`materialized_history_len == emitted_history_len`；
- 客户端 abort 时默认回滚到上一 committed version，不把未完整交付的输出发布为新历史；
- closure 只增加每 turn 一次 decode step，不增加每 token TPOT。

不推荐通过内部请求 `max_tokens+1` 后丢弃最后一个 token：它会干扰 EOS/stop、usage、
streaming 和采样语义，也很难覆盖工具调用等上层协议。

## 13. 全层/全 rank commit 与 GPU 同步

新增 `StepCommitTracker`，key 至少包含：

```text
(session_id, conversation_epoch, turn_version, target_seq_len)
```

每个 layer append 完成后只更新 bitmap；只有以下条件同时满足才发布 step commit：

```text
all_layers_complete
and all_tp_ranks_complete
and token_ids_verified
```

完成后：

1. 每个 Attention rank 在 decode stream 上 record IPC event；
2. PA coordinator 聚合 rank completion；
3. Prefill control plane 原子更新 token ledger 和 committed watermark；
4. 下一轮 Prefill stream 使用 `cudaStreamWaitEvent` 等待，不让 CPU 做
   `torch.cuda.synchronize()`；
5. commit ACK 带 `session_id/epoch/version/seq_len`，旧 epoch 的迟到 ACK 被丢弃。

当前 `layer_complete` 字段没有形成真正的 barrier，可以复用字段但需要改变语义。

## 14. 故障、并发与回退

### 14.1 PA 失败或重启

PA health incarnation 改变后，directory 中旧 session 立即标记 LOST。下一轮选择新 PA
做完整 Prefill，epoch 加 1。MVP 不尝试从失效 PA 恢复 block id。

### 14.2 Projection 失败

如果失败发生在已提交 step 之后，可从最新 committed watermark 重试；partial 下一
step 的 KV 槽位由重试覆盖。静态 affinity 阶段可以切换到备用 Projection，但需要先
为该 PA 建立新 peer transport；尚未支持前则整 turn abort，保留上一 version。

### 14.3 同一 conversation 并发 turn

MVP 返回 409 `conversation_busy`，不做并发分支。未来若要支持 branch，需要 copy-on-write
block table 和分支 version tree，超出本阶段范围。

### 14.4 token mismatch

记录 mismatch 位置和 reason，但不要记录敏感 token 内容。行为有两个选项：

- 默认：完整 Prefill 并用新 session/version 替换旧 placement；
- debug strict：返回 409，要求客户端纠正 parent version/history。

生产默认推荐安全重算，测试环境开启 strict 以发现协议错误。

### 14.5 TTL 与 eviction

conversation TTL 与当前短 request lease 分离。建议：

- ACTIVE turn 使用短 heartbeat/自动 refresh；
- IDLE conversation 使用较长 TTL + LRU；
- eviction 先 CAS 到 EVICTING，拒绝新 begin-turn，等待在途引用归零后 free；
- proxy directory 和 PA owner 的 TTL 不一致时，以 PA owner 响应为真值，proxy 只缓存
  placement hint。

## 15. 代码改动边界

### 15.1 Proxy 与 topology

`examples/pap/multi_pap_proxy_server.py`

- 增加 `ConversationDirectory` 接口和内存实现；
- first-turn PA 选择、subsequent-turn PA affinity；
- per-conversation lock/version/epoch；
- request cleanup 改成 `end_turn`，conversation delete/TTL 才 release session；
- Projection 选择使用 PA affinity 或 crossbar capability；
- 对 TP rank registration/readiness 使用并发 gather，避免串行 HTTP 开销。

`examples/pap/launch_pap_nixl.sh`

- 为每个 PA 生成独立 control endpoint；
- 输出明确的 PA/Projection topology manifest，而不是依赖端口算术推断；
- 阶段 A 建立静态 PA-to-Projection 映射；
- 后续支持每 peer actor/doorbell 配置和多主机 locality 元数据。

### 15.2 KV owner 与 scheduler

`vllm/pap/kv_lease.py`

- 从 request-only registry 演进为 request lease + conversation resident owner；
- 增加 begin-turn/detach/evict/CAS/version API；
- 保留 request-only 路径兼容无 conversation 的请求。

`vllm/v1/core/kv_cache_manager.py`、
`vllm/v1/core/kv_cache_coordinator.py`、
`vllm/v1/core/single_type_kv_cache_manager.py`

- 增加 trusted resident attach/detach；
- 支持 partial tail block；
- 恢复/转移 manager bookkeeping；
- 精确 refcount 和 rollback。

`vllm/v1/core/sched/scheduler.py`

- 在 admission 前查询 conversation owner；
- attach 后只调度 suffix；
- turn 完成时把 block ownership 转回 owner；
- dynamic capacity admission；
- cache-only finalization 状态。

### 15.3 Attention 与 control plane

`examples/pap/pap_attention_executor.py`

- session 主键改为稳定 `session_id`，per-turn request 建 alias；
- response 结束只 end turn，不删除 resident session；
- per-session Prefill control route；
- StepCommitTracker 和全层/rank fence；
- 支持新增 block table/topology，而不重复打开整套 IPC tensor；
- transport registry 支持多个 Projection peer。

`vllm/pap/decode_commit_client.py`、
`vllm/pap/prefill_control_router.py`、
`vllm/pap/lease_release_client.py`

- payload 加 session/epoch/version；
- endpoint 从全局变为 per-session；
- ACK watermark 改为 conversation/turn 级；
- release 拆分为 end-turn 和 release-conversation。

### 15.4 Projection/model runner

`vllm/model_executor/models/qwen3.py` 与 GPU runner PAP metadata：

- 显式传递稳定 `pap_session_id`，不再把 request id 和 KV handle 混作 session id；
- route group 保持按 Attention endpoint 分组；
- 增加 closure row 标记，使最后一步只写 KV、丢弃 logits/sample；
- full crossbar 阶段按 Projection peer 选择 transport。

第一阶段仍限定 Qwen3；在数据模型和 scheduler API 稳定后再抽象为通用 model hook。

## 16. 推荐实施顺序

### Phase 0：观测与 fail-closed contract

1. 增加 session_id/epoch/version/materialized watermark 指标和日志；
2. 增加当前 final-token gap、首层早提交和多 PA endpoint 的失败测试；
3. 保持默认单轮行为不变。

完成标准：测试能稳定证明当前缺口，而不是先修改行为后失去对照。

### Phase 1：xPAyP correctness

1. commit/release 改为 per-session Prefill endpoint；
2. 使用静态 PA-to-Projection affinity；
3. topology manifest 与路由审计；
4. 跑 `2PA1P`、`1PA2P`、`2PA2P`、`6PA2P` 单轮 correctness。

完成标准：每个 request 的 Prefill、Attention、commit 和 lease 全部落到同一 PA group；
所有 session/lease drain 为 0。

### Phase 2：conversation owner 与 resident-attach

1. ConversationDirectory + per-conversation CAS；
2. request lease 转 conversation owner；
3. scheduler attach/detach partial block；
4. Attention persistent session + turn alias；
5. 第 2 轮只计算 suffix。

完成标准：除最后一个输出 token 的已知 closure gap 外，历史重算为 0；故障时安全 miss。

### Phase 3：严格全量命中

1. StepCommitTracker 全层/全 rank barrier；
2. final-token cache-only closure；
3. exact token ledger 与 full-history verification；
4. 正常完成时 `committed_history == emitted_history`。

完成标准：第 2 轮 `history_kv_hit_tokens == round1_history_tokens`，包含第 1 轮最后一个
可见 assistant token。

### Phase 4：Projection 弹性与跨 PA fallback

1. Attention multi-peer lazy transport registry；
2. Projection load-aware 选择；
3. PA lost 后完整 Prefill fallback；
4. 最后再评估 NIXL cross-PA migration，MVP 不实现。

## 17. 测试与实验计划

### 17.1 单元/contract 测试

- 同 conversation 两个并发 begin-turn 只有一个成功；
- stale version/epoch/incarnation 被拒绝；
- request block ownership 在 IDLE/ACTIVE 间移动时 refcount 不变；
- partial tail block attach 后 suffix 写入正确 slot；
- token mismatch 时不 attach；
- all-layer/rank 未完成时 committed watermark 不前进；
- final visible token closure 后 materialized watermark 加 1；
- conversation end-turn 不 release，delete/TTL 才 release；
- PA1 Attention 的 commit 不会发往 PA0；
- 同一 Attention bind 第二个 Projection 时建立独立 transport，不复用错误 peer。

### 17.2 两轮 E2E correctness

先用 Qwen3-8B、temperature 0：

1. 第 1 轮固定 prompt，生成 32 token；
2. 把完整第 1 轮 assistant 输出加入第 2 轮 messages；
3. 第 2 轮增加短 user suffix，再生成 32 token；
4. 与单体 vLLM 或冷 Prefill PAP 的 token 输出逐 token 对比；
5. strict audit 检查 PA affinity、token digest、block ids、commit watermark、final closure。

建议复用：

```text
benchmarks/multi_turn/benchmark_serving_multi_turn.py
  --send-conversation-id
```

并增加 PAP audit 输出。

### 17.3 xPAyP 矩阵

在 8 GPU 同机环境依次跑：

| 拓扑 | 目的 |
|---|---|
| `1PA1P` | resident-attach 与 closure 最小正确性 |
| `2PA1P` | 多 PA control endpoint 与 conversation affinity |
| `1PA2P` | 单 PA 多 Projection peer/crossbar |
| `2PA2P` | 多 conversation 混合路由与负载均衡 |
| `6PA2P` | 当前目标规模回归 |
| `1PA1P TP2` | 全 TP rank barrier 与 block table 一致性 |

阶段 A 的 `1PA2P` 只验证静态选择其中一个 Projection；full crossbar 完成后再验证请求间
切换 Projection。

### 17.4 性能指标

必须新增：

- `pap_conversation_history_tokens`；
- `pap_conversation_kv_hit_tokens`；
- `pap_conversation_history_recomputed_tokens`；
- `pap_conversation_suffix_prefill_tokens`；
- `pap_conversation_materialized_tokens`；
- `pap_conversation_emitted_history_tokens`；
- `pap_conversation_affinity_hit/miss`；
- miss reason：`not_found`、`evicted`、`token_mismatch`、`epoch_mismatch`、
  `pa_lost`、`capacity`；
- resident block 数、IDLE/ACTIVE conversation 数、TTL/LRU eviction 数；
- begin-turn attach、fence wait、suffix prefill、closure 的耗时。

主要对比：

1. 第 2 轮冷 Prefill TTFT vs resident-hit TTFT；
2. 第 2 轮 history token 数增长时的 TTFT 曲线；
3. closure 对每 turn 尾延迟的固定开销；
4. 单轮 TPOT 是否相对提交 `87bb1061` 回退；
5. 多 conversation 下 Projection 利用率和 PA KV 水位。

## 18. 验收标准

功能验收：

- 同一 conversation 第 2 轮落到第 1 轮 PA；
- `history_kv_hit_tokens == round1_history_tokens`；
- `history_recomputed_tokens == 0`；
- 第 1 轮最后一个可见 assistant token 包含在命中 token ledger 中；
- 第 2 轮只 Prefill 新 suffix；
- 输出 token 与确定性参考基线一致；
- token/version/epoch 不匹配时 fail closed；
- conversation delete/TTL 后 PA/Attention session、block ref 和 lease 全部归零。

xPAyP 验收：

- x、y 不要求整除；每个 PA 都有且只有一个有效 control route；
- 阶段 A 每个 PA 稳定映射到一个 Projection，分布差不超过 1 个 PA；
- full crossbar 阶段任意健康 Projection 可服务任意 PA；
- 不出现 wrong Attention endpoint、unknown request、commit to PA0、peer rebind 或
  stale block ABA。

性能验收：

- 现有 `1PA1P i128/o32/qps16` 单轮 median TPOT 回退不超过 3%；
- 第 2 轮 warm TTFT 随历史长度增长基本保持只与新 suffix 有关；
- closure 只引入每 turn 一个 decode step，不改变 steady-state per-token TPOT；
- 相同多轮负载下，PA 历史 Prefill FLOPs 接近 0，Projection/Attention TPOT 不因
  conversation control plane 增加同步等待。

## 19. 历史实现的经验

git 历史中：

- `7ae51f27b` 曾加入 conversation placement；
- `400eae351` 曾尝试复用 conversation 的旧 KV handle；
- `c12e99edc` 又移除了这套绑定，明确要求 placement 由外部 cache/load-aware
  policy 管理，并指出 scheduler resident attach/skip 尚未实现。

这次设计保留该架构边界：conversation affinity 位于外部 router/scheduler 层，计算
进程不自行决定 placement。同时补齐当时缺失的 resident owner、partial-block attach、
版本化生命周期、全层 commit 和 final-token closure。只有这些条件同时成立后，旧 KV
handle 才能成为稳定 session capability，而不是一个跨 request 的脆弱字符串引用。

## 20. 最终推荐

下一轮编码先做 Phase 0 和 Phase 1，不要直接从 conversation sticky routing 开始。
原因是当前多 PA control endpoint 和 Attention 单 peer transport 尚未闭合；如果先加
conversation affinity，测试可能在 `1PA1P` 看似成功，却在真正 xPAyP 中把 commit 发到
错误 Prefill 或复用错误 peer。

Phase 1 通过后，再实现 conversation owner + scheduler resident-attach；最后以
StepCommitTracker 和 cache-only closure 把“多数 token 命中”提升为严格的“全部历史
token 命中”。
