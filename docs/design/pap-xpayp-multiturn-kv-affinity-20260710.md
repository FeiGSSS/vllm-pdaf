# PAP 多轮原生 Prefix Cache 复用设计

日期：2026-07-11
状态：1PA1P exact-token 与 Chat Completions 已验证
基线提交：d8bce2e6c

## 1. 目标

第一轮 PAP 请求完成后，Prefill 和 decode 已物化的 KV 都留在同一个 PA
的 Prefill-owned paged KV 中。第二轮只要进入相同 PA，就通过 vLLM 原生
Automatic Prefix Caching（APC）复用这些 KV，不做 D 到 P 回传，也不新增
KV Connector 或 KV Transfer。

本阶段先在 1PA1P 上闭合数据面和证据链。多 PA 时的 cache-aware 路由不由
当前 Proxy 实现，后续交给 Dynamo 等外部路由框架。

本阶段明确接受以下边界：

- 第一轮最后一个 sampled token 没有 KV，第二轮需要重新计算；
- APC 只命中完整 block，partial tail 也需要重新计算；
- cache block 是 LRU eviction candidate，不提供永久驻留保证；
- 每一轮结束后 PA 与 Projection pair 解散，Attention session 和 request lease
  正常释放。

## 2. 非目标

本阶段不实现：

- ConversationDirectory 或 Proxy conversation affinity；
- 跨轮稳定 backend request ID；
- 跨轮固定 Projection；
- persistent Attention session；
- conversation resident owner；
- scheduler block detach/attach；
- partial-block attach；
- final-token cache-only closure；
- 跨 PA KV migration。

每一轮继续使用独立 request ID。Prefix cache 通过 token 内容和模型/cache
元数据寻址，而不是通过 conversation ID 或 request ID 寻址。

## 3. 已确认的数据路径

### 3.1 第一轮 Prefill

Prefill vLLM 在 PA 的 Prefill 进程中分配 paged KV blocks，并计算 prompt KV。
这些 block 的物理所有权始终属于 Prefill 进程。

### 3.2 第一轮 Decode

Attention 进程通过 CUDA IPC 打开 Prefill-owned KV tensor。Projection 将当前
decode token 的 Q/K/V 发给 Attention；Attention 直接把 K/V 写入 Prefill
分配的 block，再执行 attention。

因此 decode KV 没有先落到 Projection，也没有从 Attention 复制回 Prefill：
它从一开始就在 Prefill-owned paged KV 中。

### 3.3 Decode commit

Attention 在 decode KV 写入后向对应 Prefill control endpoint 发送 commit。
Prefill Scheduler：

1. 把新 token ID 追加到原 Prefill request；
2. 推进 num_computed_tokens；
3. 更新 block hash 链；
4. 将新形成的完整 block 注册到原生 prefix cache。

Request lease 保证 Attention 仍可能写入时，Scheduler 不会提前把 block 返回
block pool。

### 3.4 第一轮结束

Projection 完成后，Proxy 删除 request-scoped Attention session。Attention：

1. flush 该 request 的所有 decode commit；
2. 释放 Prefill request lease；
3. 删除 CUDA IPC view 和 request-scoped session。

lease 释放后：

- 有 hash 的完整 block 以 refcount 0 留在 prefix-cache LRU；
- 没有 hash 的 partial tail 优先进入可重用队列；
- PA 与 Projection 不再保持 request 绑定。

### 3.5 第二轮

第二轮完整 prompt 进入同一 PA 后，Prefill Scheduler 根据 token IDs 计算
block hash，并调用原生 longest-prefix lookup。命中的 block 被 touch 并重新
挂到新 request；Scheduler 只计算未命中的 suffix。

Projection 可以是任意健康实例。它不拥有历史 KV，也不需要知道第二轮是否
命中。

## 4. PAP 相对 PD 的多轮优势

PD 第一轮结束后：

- Prefill 节点可保留它计算过的原始 prompt KV；
- decode KV 位于 Decode 节点；
- 如果不做 D 到 P 回传，下一轮 Prefill 不能直接命中第一轮 decode KV。

PAP 第一轮结束后：

- 原始 prompt KV 在 PA；
- decode K/V 由 Attention 直接追加到同一套 PA Prefill-owned blocks；
- decode commit 又把这些 token 纳入 Prefill block hash 链；
- 下一轮落到同一 PA 后，可以命中 prompt 和完整 decode blocks。

因此核心验收不是仅观察第二轮有 prefix hit，而是证明 hit 长度超过第一轮
原始 prompt 的可缓存边界，确实覆盖了第一轮 decode 产生的 block。

## 5. 精确命中边界

设：

- P：第一轮原始 prompt token 数；
- C：第一轮结束前已经 commit 且物化 KV 的总 token 数；
- L：第一轮 committed token 序列与第二轮 prompt 的最长公共前缀长度；
- N：第二轮 prompt token 数；
- B：block size，当前默认是 16。

原生 APC 的预期命中长度是：

    expected_hit =
        min(floor(L / B) * B, floor((N - 1) / B) * B)

第二项来自 vLLM 的 logits 约束：即使整个 prompt 都命中，也必须重新计算
最后一个 token；当前 slot allocation 又要求命中长度按 block 对齐。

可归因于第一轮 decode 的命中下界是：

    decode_hit =
        max(0, expected_hit - floor(P / B) * B)

例如 C 等于 100、B 等于 16 时，最多有 96 token 可通过 APC 直接命中。即使前
100 个 token 都有物理 KV，最后 4 个 partial-tail token 仍会重算；再加上最后一个
没有 KV 的 sampled token和第二轮新增 suffix。

## 6. Token 前缀要求

命中只依赖第二轮 tokenized prompt 是否与第一轮 committed token 链共享前缀。
conversation_id、文本长度或 request ID 都不能替代 token 比较。

Chat Completions 需要特别审计：

- assistant 文本 decode 后重新 tokenize 可能改变 BPE 边界；
- stop/EOS token 可能不会出现在第二轮 messages；
- chat template 会添加 assistant-end、user role 和 assistant-open token；
- 模型、tokenizer或 chat template 变化会自然造成 miss。

因此先运行两类实验：

1. exact-token 两轮实验：消除文本重新 tokenize 歧义，验证 KV 数据面；
2. Qwen3 Chat Completions 两轮实验：验证真实 messages 和模板下的 LCP/hit。

## 7. 当前需要修复的缺陷

KV manager 的 pop_blocks_for_free 明确要求调用者按反向 allocation order 释放，
以便同一条 prefix chain 中 tail block 比 prefix block更早被淘汰。

正常 request free 路径满足这一要求，但 PAP lease-release 当前保存 allocation-order
blocks，并把它们原序传给 block_pool.free_blocks。显存压力下会导致公共
prefix 比 decode tail 更早淘汰；保留下来的 tail 又因为缺少前缀链而无法
单独产生长命中。

修复要求：

- lease deferred free 必须以 tail-to-prefix 顺序返回 block pool；
- unhashed partial tail 仍应最先成为重用候选；
- 不改变 refcount、hash map 或 request-only lease 生命周期；
- 增加单测直接验证 free callback 收到的顺序。

## 8. 观测设计

新增仅在 PAP 多轮审计开关启用时输出的逐请求结构化记录：

- request_id；
- prompt_tokens；
- local_prefix_hit_tokens；
- block_size；
- cached_block_count；
- num_computed_tokens；
- decode_commit old/new seq len；
- lease_id；
- lease release block count；
- Attention session drain 状态。

实验工具额外记录：

- 第一轮 prompt token IDs 或安全 digest；
- 第一轮最终 committed seq len；
- 第一轮可缓存完整 block 边界；
- 第二轮 prompt token 数；
- 两轮 token LCP；
- expected/actual hit tokens；
- decode-derived hit tokens；
- warm/cold TTFT、Prefill 时间和 TPOT。

默认日志不输出原始敏感 token 内容。测试模式可把 token ID 写入本地实验
目录。

## 9. 1PA1P 严格 AB

固定环境：

- 模型：/data/ssd1/llm-models/Qwen3-8B；
- 拓扑：1PA1P；
- Prefill/Attention MPS：70/30；
- 不扫描 MPS；
- VLLM_USE_FLASHINFER_SAMPLER=0；
- 本地请求绕过 HTTP/HTTPS proxy；
- 不访问 Hugging Face。

### 9.1 Warm PAP

1. 发第一轮请求并生成足够长的输出，使 decode 至少形成一个完整 block；
2. 等待响应完成、decode commit flush、Attention session 删除和 lease release；
3. 不重启服务，立即发送包含第一轮历史的第二轮；
4. 记录第二轮 actual prefix hit。

### 9.2 Cold PAP

使用和 Warm 第二轮完全相同的 tokenized input，但在无可用 prefix cache
的服务状态下运行。它提供输出正确性和完整 Prefill 时间基线。

### 9.3 可选 PD

在相同 tokenized history 下运行 1P1D。预期同一 Prefill 最多命中第一轮原始
prompt blocks，而不会像 PAP 一样命中 decode-derived blocks；除非另行启用 D 到 P
KV 回传。

## 10. 验收标准

正确性：

- Warm 与 Cold 第二轮输出 token 完全一致；
- 所有 decode commit、lease release 返回成功；
- 请求结束后 active Attention sessions 等于 0；
- 请求结束后 active lease 等于 0；
- 不出现 unknown request、stale commit、越界 slot 或错误 PA endpoint。

缓存：

- actual_hit 等于根据真实 token LCP 和 block size 计算出的 expected_hit；
- actual_hit 大于第一轮原始 prompt 的完整 block 边界；
- decode_hit 至少为一个完整 block；
- 服务日志中没有 KV Connector 或 decode KV 回传；
- cache miss/eviction 时只退化为重算，不影响输出正确性。

性能：

- Warm 第二轮 Prefill 时间和 TTFT显著低于 Cold；
- 单轮和第二轮 TPOT 不因观测或释放顺序修复产生可测回退；
- 后续高压力实验中，修复后的 prefix retention 不低于修复前。

## 11. 实施顺序

1. TDD 修复 PAP lease-release block 顺序；
2. 增加逐请求 prefix-hit 审计；
3. 编写最小 exact-token 两轮 runner；
4. 运行 1PA1P Warm/Cold correctness；
5. 接入真实 Chat Completions 两轮 workload；
6. 根据实验中发现的问题修复 commit、release 或 token ledger；
7. 最后才评估多 PA + Dynamo cache-aware routing。

每个阶段单独测试和提交。任何实验若不能证明 decode-derived block 命中，
都不能宣称多轮 KV 复用已经完成。

## 12. 回滚

本阶段不改变 Proxy routing 或公开 API。观测功能默认关闭；release-order
修复只恢复 KV manager 已声明的释放顺序语义。若出现回归，可关闭审计开关
并单独回退顺序修复，现有单轮 PAP 数据面不依赖多轮实验工具。

## 13. 被替代的方案

此前提出的 ConversationDirectory、稳定 backend session、persistent Attention、
resident snapshot 和 scheduler detach/attach 方案已被本设计替代。提交
e6c499c78 及其未提交 Proxy WIP 已从 feature/pap 分支移除。

当未来需要保证 partial tail 永久驻留、跨轮单写者语义或完全不允许 eviction
时，可以重新评估 resident owner；它不是当前 1PA1P 原生 prefix-cache 复用的
前置条件。

## 14. 实施与实验结果

### 14.1 核心实现

本阶段实现拆分为以下提交：

- `c71ccc9df`：通过 Scheduler 正式预留 unified-KV decode slots，修复 handoff
  token ledger，增加安全 prefix-cache audit 和已分配块不变量；
- `043339691`：增加 exact-token 两轮 warm/cold 审计和 Prefill cached-token
  可观测性；
- `558db3cdd`：Projection KV-unaware 请求跳过本地 cache registration；
- `d5ea82ca3`：增加真实 Chat Completions 两轮审计模式；
- `848f321ab`：使用可保持跨轮 token 连续性的 Qwen3 thinking chat template。

核心缺陷是 Prefill 的一次性请求从 Scheduler waiting 分支完成，旧代码只在
running 分支私自追加 decode blocks。其结果是 request hash 和 cached-block 计数
可以增长，但 model-runner block table 仍只有 prompt blocks。修复后，decode
capacity 作为 `num_lookahead_tokens` 进入统一 slot allocation，block ownership、
Scheduler output 和 model-runner block table 保持一致。

### 14.2 Exact-token clean 实验

最终 clean run：

`/home/fei/research/PD/test/baseline/pap/results/runs/20260711_558db3cdd_multiturn_clean_rep2`

- 第一轮 prompt/output：128/48 token；
- 第二轮 prompt：183 token；已物化历史 175 token；
- expected/actual prefix hit：160/160 token；
- decode-derived hit：32 token；cold hit：0；
- Warm/Cold 第二轮 Prefill：45/58 ms；
- Warm/Cold 第二轮输出 token 完全一致；
- strict correctness、三请求路由和 session drain 全部通过；
- 109 次 decode commit 和 3 次 lease release 全部返回 200；
- 服务日志无错误匹配。

首次 clean run 在 Projection 端触发“0 个本地块却缓存 8 个块”的 fail-closed
检查。Projection KV-unaware 本来就不拥有本地 KV，因此 `558db3cdd` 将其从本地
cache registration 中明确排除，同时保留 Prefill 侧的已分配块不变量。

### 14.3 Chat Completions clean 实验

最终 clean run：

`/home/fei/research/PD/test/baseline/pap/results/runs/20260711_848f321ab_chat_multiturn_clean_rep2`

- Qwen3 chat template：`enable_thinking=true`；
- 第一轮 prompt/output：142/48 token；
- 第二轮 prompt：207 token；已物化历史和真实 LCP 均为 189 token；
- 第一轮 prompt 完整块边界：128 token；
- expected/actual prefix hit：176/176 token；
- decode-derived hit：48 token；cold hit：0；
- Warm/Cold 第二轮 Prefill：46/70 ms；
- Warm/Cold 第二轮输出 token 完全一致；
- 本地 tokenizer 与服务端 chat-template token IDs 完全一致；
- strict correctness、三请求路由和 session drain 全部通过；
- 109 次 decode commit 和 3 次 lease release 全部返回 200；
- 服务日志无错误匹配。

`enable_thinking=false` 是一个重要负对照：Qwen3 模板会在第一轮 generation
prompt 中插入空的 `<think>...</think>` scaffold，但普通 assistant content 不会把
该 scaffold 带回第二轮历史。clean rep1 因而观测到 decode-derived hit 为 0。
这不是 PAP KV 数据面丢失，而是第二轮 token 前缀本身发生变化。thinking 模式
不会插入该不可回传 scaffold，clean rep2 的 LCP 覆盖全部已物化历史并命中 3 个
decode-derived blocks。

### 14.4 回归结果

最终实现工作树上：

- `tests/pap`：383 passed，3 skipped；
- `tests/v1/core/test_prefix_caching.py`：83 passed；
- Runner Bash 语法、Python 编译和 `git diff --check` 均通过；
- 测试和实验均只使用本地模型，未访问 Hugging Face，未运行 pre-commit。
