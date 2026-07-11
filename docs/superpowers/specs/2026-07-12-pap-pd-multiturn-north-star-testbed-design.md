# PAP/PD 多轮北极星 Test Bed 设计

日期：2026-07-12

状态：已批准，进入实施

目标分支：`feature/pap`

## 1. 目标

构建一个可重复、可审计的固定性能基准，在相同两张 GPU 的条件下比较：

- 官方 1P1D PD/NIXL；
- 当前 1PA1P PAP/NIXL-mailbox。

基准使用已经打通的两轮对话路径，核心指标是每轮 TTFT 和 TPOT。日常优化只重跑
PAP，并同时回答两个问题：

1. 候选版本是否优于已确认的 PAP reference；
2. 候选版本距离固定 PD reference 还有多远，是否达到
   `PAP TPOT < 2 * PD TPOT`。

长期目标是把同一结果合同扩展到任意 X:Y PAP 与相同 GPU 预算的 PD，但本阶段只实现
和验收 1:1。

## 2. 边界

### 2.1 本阶段包含

- 一个冻结的 16K、两轮、单会话 profile；
- quick（1 次）和 formal（3 次）两种 PAP 运行模式；
- 一次性、显式触发的官方 PD reference bootstrap；
- Git-tracked 的精简 PD/PAP reference；
- 每次运行的原始结果、配置、日志审计和 Markdown/JSON 比较报告；
- 第二轮缓存复用、固定输出长度、致命日志和 Attention session drain Gate；
- 相对 PD、相对 PAP reference 的 TTFT/TPOT 对比和优化分类。

### 2.2 本阶段不包含

- 修改 PD、NIXL、官方 PD proxy 或 OpenAI API；
- 增加 PD token-accounting instrumentation；
- OOM mission、容量扫描或 MPS 扫描；
- 4K/32K、并发矩阵、think-time、模型矩阵；
- 自动修改 reference；
- X:Y 的调度或性能验收实现；
- 根据一次 quick run 宣称稳定优化。

## 3. 固定实验合同

### 3.1 公共 workload

| 参数 | 固定值 |
| --- | --- |
| Profile ID | `qwen3_8b_chat_16k_2turn_o256_c1_v1` |
| 模型 | `/data/ssd1/llm-models/Qwen3-8B` |
| 语料 | `/home/fei/research/PD/refer_codes/vllm/benchmarks/sonnet_4x.txt` |
| API | `/v1/chat/completions`，streaming，`return_token_ids=true` |
| 对话数 / 并发 | 1 / 1，closed loop |
| 轮数 | 2 |
| 第一轮文档 token | 本地 tokenizer 取语料前 16,000 tokens |
| 第二轮新增正文 | 同一语料后续 120 tokens，渲染后记录真实长度 |
| 每轮输出 | 256 tokens |
| Sampling | `temperature=0`、`seed=0`、`ignore_eos=true` |
| Qwen thinking | 开启并完整保留 assistant 内容，保证跨轮 token continuity |
| Warmup | 0；每个 repetition 重启服务 |
| Tokenizer | `local_files_only=True`、`trust_remote_code=False` |

客户端必须保存每轮服务端 prompt/output token IDs 的 digest、真实 token 数、assistant
文本 digest 和完整 profile fingerprint。第二轮消息由第一轮真实 assistant 文本构造，
并用返回的 token IDs 计算重新渲染后的真实 LCP，禁止只凭文本长度或 conversation ID
推断 Decode 历史命中。客户端必须消费 SSE 直到 HTTP EOF；看到 `[DONE]` 后不能提前关闭
连接，以便 PAP proxy 的 streaming cleanup 和 Attention session DELETE 正常执行。

### 3.2 公共服务参数

- GPU 1/2；启动前必须确认两张卡无其他计算进程；
- TP=1、dtype=`float16`、`--enforce-eager`；
- `max_model_len=20000`；
- `max_num_batched_tokens=4096`；
- `max_num_seqs=2`；
- prefix caching 和 chunked prefill 开启；
- `VLLM_USE_FLASHINFER_SAMPLER=0`；
- `NO_PROXY/no_proxy` 必须包含 `127.0.0.1,localhost`；
- 所有服务和 repetitions 串行运行，禁止并行占用其他 GPU 来制造不可比结果。

PD 使用 1P1D、官方多轮 proxy 和 NIXL 双向复用配置。PAP 使用 1PA1P、当前已验证的
NIXL-mailbox/CUDA-IPC 数据面、统一 KV 和固定 MPS 70/30。两条 lane 均使用相同 GPU
编号和公共参数；lane-specific memory utilization 保持已验证值并写入 artifact。

### 3.3 性能期间的观测边界

性能运行不得开启逐 token 的 `PAP_PREFIX_CACHE_AUDIT`，因为它会引入同步日志开销。
PAP 缓存命中使用现有 Prefill response headers 和只读服务日志确认；PD 只使用官方响应、
日志和 `/metrics`。任何诊断 trace/profile 必须作为独立运行，不能冒充性能结果。

## 4. 组件和职责

### 4.1 确定性两轮客户端

文件：`benchmarks/multi_turn/pap_pd_multiturn_client.py`

职责：

- 纯本地构造固定两轮消息；
- 以同一 payload 驱动 PD 或 PAP；
- 从 SSE 记录 TTFT、完成时间和 token usage；
- 从 streaming `prompt_token_ids` / `token_ids` 记录真实 token 序列；
- 按 `(latency - TTFT) / max(completion_tokens - 1, 1)` 计算 TPOT；
- 记录响应头中的 Prefill token 证据（存在时）；
- 读取到 EOF，并把单次 repetition 写为 `result.json`；
- 对 HTTP、长度、usage 和 finish reason 失败 fail closed。

客户端不启动服务、不比较 reference，也不修改 cache/reference。

### 4.2 单次 PAP 生命周期 runner

文件：`.claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh`

在现有自包含 runner 中增加 `multiturn_north_star` client mode。该 mode 只改变固定
workload 和结果验证，不复制 PAP 启停逻辑。它必须：

- 冻结本设计中的 1PA1P 参数；
- 调用确定性客户端；
- 等待 Attention active sessions 降到 0；
- 运行现有 correctness、routing 和 fast-path audit；
- 记录 Git commit、tracked dirty 状态、effective config 和完整服务日志；
- 任何 Gate 失败时保留 artifact 并返回非零。

### 4.3 Test-bed orchestrator

文件：`.claude/skills/vllm-pap-benchmark/scripts/run_multiturn_north_star.sh`

职责：

- `quick` 串行运行 1 个 PAP repetition；
- `formal` 串行运行 3 个 PAP repetitions；
- 为每次 repetition 分配独立端口和 run directory；
- 调用比较器生成总报告；
- 不启动 PD，不写 reference；
- 运行结束后只清理自己启动的服务。

默认模式是 `quick`。正式优化结论必须来自 `formal`。

### 4.4 比较与 reference 管理器

文件：`benchmarks/multi_turn/compare_pap_pd_multiturn.py`

职责：

- 验证 candidate、PD reference 和 PAP reference 的 profile fingerprint 一致；
- 验证所有 repetitions 的 correctness Gate；
- 对 formal 运行按指标取跨运行中位数，并保留每次原值；
- 输出 `comparison.json` 和 `report.md`；
- 提供显式 `bootstrap-reference` / `promote-pap-reference` 子命令；
- reference 写入必须原子化，且默认 compare 命令永不修改 reference。

### 4.5 一次性 PD bootstrap

文件：`.claude/skills/vllm-pap-benchmark/scripts/bootstrap_pd_multiturn_reference.sh`

该脚本只编排官方、未修改的 PD/NIXL 服务和多轮 proxy，串行运行 3 次同一客户端，完成
Gate 后生成 reference candidate。将 candidate 晋升为 tracked reference 需要显式命令。
日常 test bed 不调用此脚本。

## 5. Artifact 和 reference 合同

Git-tracked reference 目录：

```text
test/baseline/pap/references/qwen3_8b_chat_16k_2turn_o256_c1_v1/
  profile.json
  pd_reference.json
  pap_reference.json
  README.md
```

大体积运行结果保存在：

```text
test/baseline/pap/results/runs/<run-id>/
```

并继续由 Git 忽略。每个 repetition 至少保存：

- `result.json`；
- `effective_config.env`；
- `run_metadata.json`；
- `git_status.txt` 和 `tracked_worktree.patch`；
- `correctness_audit.env`；
- `session_drain.env`；
- `service_logs/`。

reference 只保存 profile、硬件签名、架构/拓扑、源 commit、三个 repetition 的核心原值、
跨运行聚合值、Gate 状态和原始 artifact 相对路径，不保存服务日志。

Profile fingerprint 至少覆盖模型路径及配置摘要、语料摘要、消息模板参数、输入切片、输出
长度、dtype、TP、max model length、max batched tokens、max sequences 和 GPU 型号。
quick/formal repetition 数属于运行模式而不是 workload fingerprint，因此单独记录。
fingerprint 不一致时比较器必须拒绝给出优化结论。

## 6. 有效性 Gate

一个 repetition 只有同时满足以下条件才有效：

1. 两个请求均返回 HTTP 200；
2. 每轮 completion tokens 均为 256，`finish_reason=length`；
3. 每轮都有非空首 token，TTFT、TPOT、latency 均为有限正数；
4. 第二轮 prompt 比第一轮长，并有可验证的 prefix-cache hit；
5. 根据两轮真实 token IDs 计算出的 Decode-derived LCP 至少包含一个完整 16-token block，
   且 PAP 第二轮实际 cached tokens 等于该 LCP 的完整 block 边界；
6. 没有 OOM、Traceback、EngineDeadError、transfer/commit/release consistency error；
7. 所有请求结束后 Attention `active_sessions=0`；
8. formal reference/candidate 的 tracked worktree 为 clean；quick 允许 dirty，但报告必须标红；
9. 三次 formal repetition 的 profile fingerprint 完全一致。

任一 Gate 失败时，状态为 `invalid`，保留原始证据，但不计算“优化/回归”结论。

## 7. 指标和判定

报告按 round 展示：

- TTFT ms；
- TPOT ms；
- turn latency ms；
- prompt/completion tokens；
- PAP/PD 和 candidate/PAP-reference ratio。

同时展示两轮 conversation latency。formal 模式使用三个 repetition 的中位数作为主值。

Primary optimization metric 是第二轮 TPOT：

- `candidate <= 0.97 * PAP reference`：`improved`；
- `candidate >= 1.03 * PAP reference`：`regressed`；
- 其他：`neutral`。

TTFT、第一轮 TPOT 和 conversation latency 是回归告警项，不否决 TPOT 专项优化，但必须
在报告中突出。北极星目标独立计算：

```text
round_2_PAP_TPOT < 2 * round_2_PD_TPOT
```

quick 只输出 `diagnostic`，不能晋升 PAP reference。只有 valid formal candidate 才能通过
显式命令晋升；晋升提交必须包含比较报告摘要。

## 8. 测试策略

Python 单元测试覆盖：

- 固定语料切片和 Chat 渲染；
- SSE 多 chunk、usage-only chunk、`[DONE]` 后继续读 EOF；
- TTFT/TPOT 公式和单 token 边界；
- response header/cache evidence 解析；
- profile fingerprint 和 mismatch fail-closed；
- quick/formal 聚合与 3% 分类边界；
- invalid Gate 不产生性能结论；
- reference 原子写入和禁止隐式晋升。

静态 runner 测试覆盖固定参数、repo venv、无裸 Python/pip、无 `pkill`、PD 源码不修改、
PAP performance audit 关闭以及 session drain 必须通过。最后运行一次 GPU smoke，再建立
三次 PD/PAP formal references。

## 9. 后续 X:Y 扩展

当前 schema 从第一天保留：

- `architecture`；
- `topology`（`pa_count`、`projection_count`、`pd_prefill_count`、
  `pd_decode_count`）；
- `gpu_count` 和硬件签名；
- per-pair route counts；
- per-round/per-conversation metrics。

以后扩展 X:Y 时，客户端、指标和 reference 合同不变，只新增拓扑 profile、等 GPU 预算
映射、cache-aware routing Gate 和并发 workload。1:1 reference 不因扩展而改变。

## 10. 完成标准

- 一条命令可运行 quick PAP test bed 并生成 JSON/Markdown 对比；
- 一条命令可运行 formal 三次并给出稳定分类；
- 初始 PD/PAP references 已建立并由 Git 跟踪；
- 当前 1PA1P formal 结果可重复，缓存命中和 session drain 均通过；
- reference/profile 不匹配、请求错误或生命周期错误均 fail closed；
- 仓库没有 PD/NIXL/官方 proxy 源码改动；
- test bed 结果足以驱动下一阶段 TTFT/TPOT profile 和严格 A/B。
