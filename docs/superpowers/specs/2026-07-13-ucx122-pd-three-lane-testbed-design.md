# UCX 1.22 与 PD 三模式多轮 Test Bed 设计

日期：2026-07-13

状态：方案 B 已实施；C2/C4 quick 通过，formal-clean 待提交后执行

目标分支：`feature/pap`

## 1. 目标

在同一台双 L20 服务器、同一模型和同一五轮长上下文负载下，默认比较：

1. `PD-oneway`：官方 `NixlConnector`，只允许 P→D；
2. `PD-twoway`：官方 `NixlConnector`，允许 P→D 和 D→P；
3. `PAP`：当前 1PA1P same-node local-fast 实现。

同时把已经验证的 UCX 1.22 CUDA IPC 路径做成仓库本地、可检查、可重建的默认
运行环境，并把根因、修复、原始实验与新三模式结果写入 Git 可追踪文档。

## 2. UCX 根因与修复

### 2.1 大白话根因

NIXL 的 `READ` 在语义上是“本地 GPU 从远端 GPU 读取”。UCX 1.21 检测到本机 L20
之间只有 PCIe、没有 NVLink 时，会把 CUDA IPC 直接 GET 的 `auto` 策略判定为关闭。
NIXL 请求本身仍能成功，但 UCX 会使用基于 Active Message/TCP 的软件模拟完成 GET。
结果是 CPU 参与数据搬运，2.2 GiB KV 被软件协议处理，吞吐只有约 500 MiB/s。

这不是 `nvidia_peermem` 缺失导致的。`nvidia_peermem` 服务于 GPUDirect RDMA/verbs；
本项目的同机数据路径是 CUDA IPC，不经过 RDMA 网卡。

### 2.2 选择 UCX 1.22

UCX 1.22 增加 RMA GET/PUT rendezvous 协议。请求方仍发起 GET，但数据拥有方可以用
CUDA IPC zero-copy WRITE 把数据写进请求方 GPU。少量控制消息允许走 TCP，大块 KV
数据必须走 `cuda_ipc/cuda`。

默认环境设置：

```text
UCX_TLS=cuda_ipc,cuda_copy,tcp
UCX_PROTO_EMULATION_ENABLE=n
UCX_MODULE_DIR=<repo-local UCX 1.22>/lib/ucx
NIXL_PLUGIN_DIR=<repo-local NIXL UCX 1.22 plugin>
LD_LIBRARY_PATH=<repo-local UCX 1.22>/lib:<repo-local UCX 1.22>/lib/ucx
```

`UCX_PROTO_EMULATION_ENABLE=n` 是 fail-closed Gate：若 CUDA IPC 数据面不可用，运行
直接失败，禁止退回 TCP 软件模拟。`tcp` 保留在 `UCX_TLS` 中只为握手和控制面服务；
UCX 协议日志必须证明大块 GPU 数据选择 `cuda_ipc/cuda`。

不采用 UCX 1.21 的 `UCX_CUDA_IPC_ENABLE_GET_ZCOPY=y` 作为默认。它在本机单次实验
可以达到约 23.6 GiB/s，但它强制覆盖 UCX 对无 NVLink 拓扑的自动策略。UCX 1.22 的
rendezvous 路径不需要该强制开关，默认行为更适合作为长期 test bed 基线。

## 3. 持久默认环境

### 3.1 目录

构建产物放在仓库内但不进入 Git：

```text
.local/ucx-1.22/
.local/nixl-ucx122/
.local/src/ucx-1.22/
.local/src/nixl-1.3.0/
```

Git 只追踪安装/验证脚本和版本清单，不追踪二进制与源码缓存。这样不会覆盖 `.venv`
自带的 UCX，也不需要写 `/usr` 或 `/opt`，可以通过删除 `.local` 完整回滚。

### 3.2 安装与验证接口

新增一个仓库脚本，完成以下职责：

- 固定 UCX `1.22.0` 与 NIXL `1.3.0`；
- UCX 启用 CUDA、shared library 和 CMA，禁用本实验不使用的 verbs、rdmacm、gdrcopy；
- 只构建 NIXL UCX 插件，并链接到仓库本地 UCX；
- 生成机器可读取的版本清单；
- 支持重复执行，版本和配置一致时不重新构建；
- 提供 `verify` 模式，不修改环境，只检查现有安装。

每次 PD runner 启动前默认执行只读验证：

1. `ucx_info -v` 必须报告 `1.22.0`；
2. `libplugin_UCX.so` 的 `libucp.so.0`、`libuct.so.0` 和 `libucs.so.0` 必须解析到
   `.local/ucx-1.22`；
3. NIXL agent 必须能加载 UCX plugin；
4. `UCX_PROTO_EMULATION_ENABLE` 必须是 `n`。

环境缺失或版本不符时 runner fail closed，并提示先执行安装命令；性能运行不在后台
自动下载或编译依赖。

## 4. 三条架构 Lane

### 4.1 PD-oneway

P 和 D 都使用：

```json
{
  "kv_connector": "NixlConnector",
  "kv_connector_extra_config": {
    "bidirectional_kv_xfer": false,
    "enable_cross_layers_blocks": "True"
  }
}
```

Prefill role 为 `kv_producer`，Decode role 为 `kv_consumer`。D 使用 UCX 1.22 GET 从 P
读取 KV。D 完成一轮后不向 proxy 返回可供下一轮 P 使用的 D→P block handle，因此
五轮 proxy 语义必须是五次 MISS、零次 HIT。后续轮 P 重新计算上一轮 Decode 历史，
再把完整可用前缀交给 D。

### 4.2 PD-twoway

两个节点都使用同一个 `NixlConnector`，仅把额外配置改为：

```json
{
  "bidirectional_kv_xfer": true,
  "kv_recompute_threshold": 0,
  "decoder_kv_blocks_ttl": 480,
  "enable_cross_layers_blocks": "True"
}
```

`kv_recompute_threshold=0` 保证测试负载中的 Decode 历史实际走 D→P，而不是被启发式
重新计算。第一轮必须 MISS；第 2–5 轮必须 HIT，并出现四次“发送 D cached blocks
to P”的代理证据。P 的 NIXL receive 指标证明 D→P，D 的 NIXL receive 指标证明 P→D。

### 4.3 PAP

保持当前五轮北极星配置：1PA1P、GPU1/2、MPS 70/30、local-fast execution、CUDA IPC
KV、unified KV、direct mailbox output 和 metadata fast key。不扫描 MPS，不在本任务中
修改 PAP 性能实现。

## 5. 公共负载与资源合同

三组固定使用：

| 参数 | 值 |
| --- | --- |
| 模型 | `/data/ssd1/llm-models/Qwen3-8B` |
| dtype / TP | FP16 / TP1 |
| GPU | GPU1、GPU2；GPU0 永不触碰 |
| 第一轮文档 | 16,000 tokens |
| 后续新增正文 | 每轮 120 tokens |
| 输出 | 每轮 256 tokens |
| 轮数 | 5 |
| active conversations | quick 默认 C2；formal 默认 C4 |
| 每轮请求速率 | 2 requests/s |
| max model length | 20,000 |
| max batched tokens / seqs | 4,096 / 4 |
| sampling | temperature 0、seed 0、ignore EOS |

每个 conversation 内严格串行推进轮次，不同 conversation 允许并发。三条 lane 使用相同
conversation payload、prompt token shape、cache salt 规则和客户端计时定义。服务必须
串行运行，不能同时占用其他闲置 GPU 来缩短墙钟时间，因为那会引入不可控跨 GPU
干扰并破坏顺序平衡。

## 6. Test Bed 编排

默认入口仍是：

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/run_pd_pap_multiturn_load.sh
```

`quick` 每组运行一次，顺序为：

```text
PD-oneway → PD-twoway → PAP
```

`formal` 每组三次，使用三阶拉丁方平衡启动顺序：

```text
PD-oneway → PD-twoway → PAP
PD-twoway → PAP → PD-oneway
PAP → PD-oneway → PD-twoway
```

输出目录：

```text
<group-root>/
  pd-oneway/run1..3/
  pd-twoway/run1..3/
  pap/run1..3/
  pd-oneway-aggregate.json
  pd-twoway-aggregate.json
  pap-aggregate.json
  comparison.json
  report.md
  testbed.env
```

原 `run_pd_multiturn_load.sh` 参数化为 `oneway|twoway`，两种模式共用服务启动、客户端、
端口、清理、artifact 和 aggregate 逻辑。禁止复制两套容易漂移的 PD runner。

## 7. 审计与失败处理

### 7.1 公共 Gate

- 每个 cell 完成 `rounds * conversations` 个请求；
- 每个请求输出 256 tokens，TTFT/TPOT/latency 有限且为正；
- Prompt token shape 在三条 lane 间一致；
- 服务日志没有 OOM、EngineDeadError、Traceback 或 NIXL failure；
- NIXL failed transfer/notification 为零；
- formal 的 tracked worktree 必须 clean；
- GPU1/2 在每次启动前空闲，结束后无残留服务。

跨 lane 的 prompt digest 必须一致。由于不同算子排布可能产生微小 logits 差异，输出
digest 不一致沿用当前策略记录为 correctness warning 和后续诊断证据，不把性能 cell
自动判 invalid；请求不完整、非有限指标或 token 数错误仍然 fail closed。

### 7.2 PD-oneway Gate

- effective config 明确 `bidirectional_kv_xfer=false`；
- proxy 为五次 MISS、零次 HIT；
- Prefill 端 NIXL receive transfer count 为零；
- Decode 端 P→D transfer count 为五轮请求总数；
- 后续轮 Decode-derived history 由 P local compute 覆盖；
- UCX 1.22、cross-layer 单描述符路径和 emulation disabled 均有证据。

### 7.3 PD-twoway Gate

- effective config 明确 `bidirectional_kv_xfer=true`、threshold 0；
- 每个 conversation 第一轮一次 MISS，后四轮四次 HIT；
- Prefill 端存在 D→P transfer，Decode 端存在 P→D transfer；
- 第 2–5 轮必须记录 Decode-derived remote hit；
- 两端 NIXL bytes/time/descriptor 计数守恒且失败为零；
- UCX 协议日志至少一次证明 GPU rendezvous 数据面选择 `cuda_ipc/cuda`。

任何 Gate 失败都保留原始结果与日志，runner 返回非零，不进入正式比较。

## 8. 报告

三模式报告按 Round 1、steady Round 2–5 展示 TTFT、TPOT 和 latency 的 median、p90、max。
同时展示三组绝对值和以下比值：

```text
PD-twoway / PD-oneway
PAP / PD-oneway
PAP / PD-twoway
```

报告必须单独列出 P→D、D→P 的 transfer count、MiB、时间、吞吐和 descriptors，避免只
看 TTFT 而无法判断双向路径是否真的执行。quick 只用于功能诊断；阶段性性能结论使用
formal 三次聚合。

## 9. 代码与测试边界

本任务允许修改：

- Chat Completions 流式终止 chunk，使其返回 D 端 `kv_transfer_params`；
- PD runner、模式感知 auditor、三模式 orchestrator 和比较器；
- UCX/NIXL repo-local 安装与验证脚本；
- 对应单元测试、静态 shell 合同测试和文档。

本任务不修改 NIXL scheduler/worker 的双向算法，不修改 PAP 热路径，不扫描 MPS，不安装
系统内核模块，不访问 Hugging Face，不运行 pre-commit。

实现采用测试驱动：先让流式字段、oneway/twoway 审计、三模式比较和 runner 合同测试按
缺失行为失败，再写最小实现使其通过。GPU 实验顺序是 UCX/NIXL smoke、C2 quick、C4
quick，最后 C4 formal。

## 10. 文档归档

完成后更新：

1. `docs/design/pd-same-node-nixl-transfer-root-cause-20260713.md`：加入大白话根因、
   UCX 1.22 修复、为何与 `nvidia_peermem` 无关，以及旧 Push-only 决策被新证据修正；
2. 新建三模式实验结果报告，保存 workload、Git/环境版本、每次 repetition、聚合表、
   correctness warning 和结论；
3. `docs/design/pap-experiment-history-index.md`：登记 UCX 1.22 GET A/B、流式 metadata
   修复和三模式 formal 实验，链接原始结果目录。

## 11. 完成标准

- 默认 PD runner 不额外设置环境变量即可使用 repo-local UCX 1.22；
- `PD-oneway` 与 `PD-twoway` 均使用官方 `NixlConnector`；
- 双向第二轮及后续轮发生真实 D→P CUDA IPC transfer；
- 一条默认命令生成三组 aggregate、统一 JSON 和 Markdown；
- C2 quick、C4 quick 和 C4 formal 均通过全部有效性 Gate；
- UCX 根因、修复和新实验结果进入 Git 可追踪文档；
- 本任务所有新增单元测试通过，`git diff --check` 通过；
- 未运行 pre-commit，未提交代码。
