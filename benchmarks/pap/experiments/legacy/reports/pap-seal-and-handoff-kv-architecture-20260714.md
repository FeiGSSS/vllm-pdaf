# PAP Seal-and-Handoff KV 架构

日期：2026-07-14

状态：第一阶段已实现并通过 C1/C2 与 C4 quick；见
[阶段结果](pap-seal-and-handoff-kv-results-20260714.md)

文档生命周期：`historical/accepted`。本文保留 P17 首次引入双路径时的设计与
A/B 验证语境；Runtime Milestone Task 2.4 已将 sealed catalog + request manifest
收敛为唯一主路径，并移除 `PAP_KV_HANDOFF_MODE` 与逐层 descriptor 实现。当前契约见
[Runtime 重构设计](../../../../../docs/design/pap/milestones/2026-07-runtime-refactor.md)，
下文关于兼容回退的描述仅作为历史证据。

## 1. 目标

将 Prefill 和 Decode 从同一份逐层 mutable KV registry 中解耦，同时保留：

- Prefill-owned paged KV 和原生 APC；
- 多请求共享只读 prefix blocks；
- 每请求独占 writable decode tail；
- 多轮结束后释放执行 session，但保留 Prefill cache blocks；
- 1PA1P 以及未来多 P 来源的 Attention combine/scatter。

本次只改变 Prefill 向 Attention 发布 KV 布局的控制面，不替换 vLLM 的 block allocator、
prefix hash、引用计数或 eviction。

## 2. 当前问题

当前 Qwen3 Prefill 在每个 layer、每个 chunk 中执行：

```text
current stream synchronize
  -> 构造整层 KV cache CUDA IPC handle
  -> 发送 per-request/per-layer TCP descriptor
  -> Attention 更新 mutable layer state
```

16K、4 chunk、36 layer 时，一个请求最多产生约 `144` 次 descriptor publish。Decode 又在
同一 layer state 上推进 `seq_len`，因此需要 registry lock、decode append lock、condition、
state identity 和 seq_len 二次校验。

## 3. 新的数据模型

### 3.1 Static KV Cache Catalog

每个 Prefill worker 的每层 `kv_cache` backing tensor 在进程生命周期内稳定。每层只向
对应 Attention rank 注册一次：

```text
layer_name -> CUDA IPC tensor handle + layout + block_size + num_kv_heads
```

Attention 打开 handle 后长期复用，不再随请求重复传输。

### 3.2 Session Manifest

Qwen3 最后一层在每个 Prefill chunk 完成后发布一个 request-level manifest：

```text
request_id / session_epoch
prefix_len
block_ids
lease_id / capacity / writable range
expected_layer_count
CUDA interprocess ready event
```

后续 chunk 的 manifest 在 Decode 开始前覆盖早期 snapshot。第一条 Decode work 用
`decode_seq_len - 1` 验证最终 prefix，随后布局不可再替换。

### 3.3 Decode Cursor

不可变布局与可变进度分离：

```text
SealedKVLayout: kv_cache, block_ids, prefix_len, writable range, lease
DecodeCursor: current_seq_len
```

Decode cursor 最终只由 Attention dispatcher 推进；阶段一仍保留现有 append 串行化作为
回退安全网，待所有执行模式统一到单 writer 后移除。

## 4. GPU 可见性

禁止在每层 descriptor 前做 CPU `stream.synchronize()`。Prefill 最后一层记录：

```python
torch.cuda.Event(interprocess=True)
```

Attention 打开 event handle，并在第一次使用该 session 前执行 stream wait。这样保证完整
Prefix KV 对 Decode 可见，而 CPU 不等待 GPU。

## 5. 正确性不变量

1. catalog entry 在 Prefill worker 生命周期内不可替换；重启必须使用新 process epoch；
2. manifest 的 layer count 必须与已注册 catalog 完全一致；
3. 第一条 Decode 的 `decode_seq_len - 1` 必须等于 manifest `prefix_len`；
4. shared prefix blocks 只读，Decode 只能写 lease 指定的 writable range；
5. 同一 session 的 Decode `seq_len` 严格单调，重复 step 只允许幂等跳过；
6. session epoch 不匹配的迟到 catalog/manifest/import 必须丢弃；
7. release 必须发生在最后一个 Attention work、decode commit 和 lease release 之后。

## 6. 兼容与回退

新增 `PAP_KV_HANDOFF_MODE`：

- `layer_descriptor`：当前逐层 descriptor 路径；
- `sealed_manifest`：static catalog + request manifest 路径。

通用代码保持旧模式；固定 PAP test bed 已在 C1/C2 正确性通过后将新模式设为默认。
所有结果必须把 mode 写入 effective config、run metadata 和 implementation fingerprint。

## 7. 实施和验收顺序

1. 增加 catalog/manifest wire schema 和纯 CPU contract tests；
2. Attention 安装 catalog，并原子生成全部 layer state；
3. Qwen3 新路径发布 catalog/manifest 和 CUDA IPC event；
4. C1 五轮验证 exact output、APC、join、lease 和 drain；
5. C2 并发验证，并用 contract test 验证共享只读 prefix/private tail 布局；
6. 同一 C4 下与 `bd164d8ff` layer-descriptor 基线做 TTFT/TPOT/吞吐 A/B；
7. 验证后提交并晋升 test-bed 默认；失败则保留旧模式回退。

第 1--7 项已经在 `25c8723de` 和提交后 C1/C2/C4 quick 中完成。共享只读 prefix/private
tail 已有 registry contract；C4 三次 formal、同 salt 的 allocator/APC 端到端共享以及
1PA2P/2PA2P 是下一阶段 Gate。
