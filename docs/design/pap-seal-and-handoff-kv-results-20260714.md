# PAP Seal-and-Handoff KV 实现与阶段结果

日期：2026-07-14

状态：实现已提交；C1/C2 正确性与 C4 quick A/B 已通过；C4 formal 待运行

文档生命周期：`historical/accepted`。本文原样保留 `25c8723de` 阶段的双路径 A/B
结论；Runtime Milestone Task 2.4 后，sealed catalog + request manifest 已成为唯一
主路径，`PAP_KV_HANDOFF_MODE` 与旧逐层 descriptor 路径均已移除。本文出现的旧默认值、
回滚方式和 mode 配置只用于解释历史实验，不再是当前运行说明。

## 1. 实现结论

提交 `25c8723de` 实现了设计文档中的第一阶段 Seal-and-Handoff 数据面：

- Prefill 每层 KV backing 在 worker 生命周期内只注册一次；
- Qwen3 最后一层按请求发布 block table、prefix、lease、writable range 和完整 layer count；
- Prefill 用 interprocess CUDA event 表达 GPU ready，不再在每层 descriptor 前执行 CPU
  `stream.synchronize()`；
- Attention 一次性生成全部 layer 的 unified KV state，并在第一次使用 session 前将 event
  wait 排入当前 CUDA stream；
- Decode claim 之前允许 chunked-prefill manifest 单调更新，claim 后布局不可替换；
- `PAP_KV_HANDOFF_MODE=layer_descriptor` 保留旧路径回滚；固定 PAP test bed 默认使用
  `sealed_manifest`，通用 Qwen3 代码未显式配置时仍默认旧路径。

实现没有替换 Prefill allocator、APC、block refcount、eviction、decode commit 或 lease
release。Prefill 仍拥有 KV，Attention 只打开静态 backing 并消费请求级布局。

## 2. 版本与验证

| 类型 | 证据 |
| --- | --- |
| tactical registry 基线 | `bd164d8ff` |
| 架构设计 | `bef48f04b` |
| Seal-and-Handoff 实现 | `25c8723de` |
| 合并单元/contract 测试 | 228 passed（实现提交前 227；随后增加共享 prefix/private tail contract） |
| 编译与脚本 | `py_compile`、runner `bash -n`、`git diff --check` 全部 exit 0 |
| 提交前说明 | 未运行 pre-commit |

覆盖内容包括 wire schema round-trip、catalog 不可替换、完整 layer count、chunked manifest
单调更新、Decode claim 后冻结、无 CUDA stream synchronize、Attention 原子安装、test-bed
默认值和 implementation fingerprint。

## 3. 固定负载

所有 GPU 实验均使用 Qwen3-8B FP16/TP1、GPU1+GPU2、静态 MPS 64/28、local-fast
OFFLOAD_EXEC、16K 首轮文档、后续每轮新增 120 token、每轮输出 256 token、5 轮、
2 requests/s、max model length 20000、max batched tokens 4096、max seqs 4。

### 3.1 C1 默认模式 clean smoke

目录：

```text
test/baseline/pap/results/runs/
  20260714_25c8723de_default_sealed_c1_quick/
```

- 未显式设置 `PAP_KV_HANDOFF_MODE`，effective config 和 implementation fingerprint 均记录
  `sealed_manifest`；
- 5/5 请求、4/4 cache transition、output digest、join、routing、correctness 和 drain 通过；
- R1 TTFT/TPOT：`5389.897 ms / 30.207 ms`；
- R2--R5 TTFT/TPOT 中位：`156.500 ms / 30.109 ms`。

### 3.2 C2 并发 smoke

目录：

```text
test/baseline/pap/results/runs/20260714_sealed_manifest_c2_quick/
```

- 10/10 请求、8/8 cache transition 和所有外部正确性 Gate 通过；
- 两个 conversation 的 prompt/output/text digest 一致；
- R2--R5 TTFT/TPOT 中位：`171.667 ms / 38.257 ms`；
- implementation fingerprint 显式包含 `kv_handoff_mode=sealed_manifest`。

该结果证明多 session、并发 Decode cursor 和跨轮 APC 生命周期可用。由于固定负载为每个
conversation 使用独立 cache salt，它不单独构成“两个独立请求共享同一物理 prefix block”
的端到端证明；该场景仍需专门的同 salt/COW 测试。

## 4. C4 提交后严格 A/B

两次 run 都绑定提交 `25c8723de`、tracked worktree clean、20/20 请求、16/16 cache
transition，且 output digest、join、routing、correctness、session drain 全部通过。唯一主要
变量为 KV handoff mode。

| Scope | 指标 | layer descriptor | sealed manifest | 变化 |
| --- | --- | ---: | ---: | ---: |
| R1 | TTFT | 10938.780 ms | 10514.883 ms | -3.88% |
| R1 | TPOT | 38.830 ms | 39.340 ms | +1.31% |
| R2--R5 | TTFT | 236.289 ms | 208.346 ms | -11.83% |
| R2--R5 | TPOT | 50.320 ms | 50.296 ms | -0.05% |

原始目录：

```text
test/baseline/pap/results/runs/
  20260714_25c8723de_c4_layer_descriptor_quick/
  20260714_25c8723de_c4_sealed_manifest_quick/
```

新路径日志确认 Qwen3-8B 的 36 层 backing 各注册一次；Attention 接收 35 个请求级
manifest snapshot。TPOT 基本不变符合设计预期：该阶段消除的是 Prefill→Attention
发布和 admission 开销，没有改变每个 Decode token 的 Projection、P2P、FlashAttention
和 output 热路径。

## 5. 与冻结 PD baseline 的位置

冻结的 UCX 1.22 C4 formal PD baseline 为：

| Scope | 指标 | PD-oneway | PD-twoway | 新 PAP quick |
| --- | --- | ---: | ---: | ---: |
| R1 | TTFT | 8112.026 ms | 8128.513 ms | 10514.883 ms |
| R1 | TPOT | 35.516 ms | 35.477 ms | 39.340 ms |
| R2--R5 | TTFT | 280.867 ms | 251.716 ms | 208.346 ms |
| R2--R5 | TPOT | 42.176 ms | 42.155 ms | 50.296 ms |

以双向 PD 为参照，新 PAP 的 R1 TTFT/TPOT 为 `1.294x/1.109x`，steady TTFT/TPOT 为
`0.828x/1.193x`。这里把新的单次 quick 与历史三次 formal baseline 对照，只能用于阶段
定位；正式替换 PAP 北极星仍需三次交错 formal。

## 6. 当前边界和下一步

1. 运行三次 C4 formal，确认 TTFT 收益和 TPOT neutral 在重启后稳定；
2. 增加同 cache salt 的跨请求共享 prefix/COW 端到端验证；
3. 在新 manifest 路径验证 1PA2P 和 2PA2P，确保 catalog epoch 和多 Projection 来源不改变
   Attention 的单 writer 语义；
4. steady TPOT 的下一步仍是 Decode 热路径，优先继续拆解 PAP 比 PD 多出的
   FlashAttention/metadata/调度开销，而不是继续优化 Prefill manifest。
