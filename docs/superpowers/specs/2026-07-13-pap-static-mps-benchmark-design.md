# PAP Static MPS Benchmark Design

## 目标

在不修改 PAP 公共启动入口和正式 70:30 北极星默认值的前提下，为
PAP-only 多轮 benchmark test bed 增加可审计的 static MPS 模式。该模式用
GPU SM partition 将同一 PA GPU 的 Prefill 与 Attention 静态隔离，用来判断
async decode-token 改动造成的 TTFT 退化是否与 dynamic MPS 下的跨进程资源
竞争有关。

本改动只提供实验能力，不预设 static MPS 一定改善性能，也不把诊断结果自动
升级成正式基线。

## 范围

修改范围限定为：

- `.claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh`
- `.claude/skills/vllm-pap-benchmark/scripts/run_pap_multiturn_load.sh`
- `tests/benchmarks/test_pap_multiturn_mps_contract.py`
- 本设计文档

`examples/pap/launch_pap_nixl.sh` 保持不变。static MPS 仍是 benchmark 专用的
诊断接口，待实验确认价值后再考虑公共启动入口。

## 配置接口

现有行为保持为默认：

```text
PAP_BENCH_MPS_PROFILE=baseline_70_30
PAP_MPS_MODE=dynamic
PAP_PREFILL_MPS_PERCENT=70
PAP_ATTENTION_MPS_PERCENT=30
```

新增诊断配置：

```text
PAP_BENCH_MPS_PROFILE=diagnostic_static_64_28
PAP_MPS_MODE=static
PAP_STATIC_PREFILL_CHUNKS=16
PAP_STATIC_ATTENTION_CHUNKS=7
```

本机 L20 有 92 个 SM；R595 static MPS 每个 chunk 对应 4 个 SM，因此
16/7 chunks 分别给 Prefill/Attention 暴露 64/28 个 SM。runner 必须校验
profile、mode、chunk 数严格匹配，避免实验标签与实际资源配置不一致。

PAP-only wrapper 接受这两个 profile，并把 mode、百分比或 chunk 数显式传给
底层 runner。三路正式 PD/PAP orchestrator 不做修改，因此继续隐式使用
dynamic 70:30。

## Static MPS 生命周期

每个 PA GPU 使用独立的 pipe/log 目录，并按以下顺序管理：

1. 用物理 GPU 启动 `nvidia-cuda-mps-control -d -S`。
2. 读取该 GPU UUID。
3. 通过 `sm_partition create <chunks> <uuid>` 创建 Prefill 与 Attention
   partition。
4. 从 R595 的 `Partition <full-id> created` 输出中保留完整 partition ID；ID
   自身含 `/`，不能按路径分隔符截断。
5. 各启动一个短 CUDA client，分别验证可见 SM 数恰好为 64 和 28。
6. Attention 与 Prefill client 使用相同 pipe/log 目录、
   `CUDA_VISIBLE_DEVICES=0`，并分别设置
   `CUDA_MPS_SM_PARTITION=<full-id>`。static 模式不得再设置
   `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`。
7. benchmark 结束后先停止所有 client，再删除两个 partition，最后退出 MPS
   daemon。

任一创建、解析、可见 SM 校验或 partition 删除失败，都应使运行 fail closed。
cleanup 在正常退出和异常退出时都执行；只有未完成的 best-effort 清理步骤可以
记录警告，不能掩盖原始 benchmark 退出码。

## 审计产物

`effective_config.env` 记录 mode、profile、chunk 数以及期望可见 SM 数。每个
PA 额外生成一份 static MPS 审计文件，至少包含：

- 物理 GPU index 与 UUID；
- Prefill/Attention 完整 partition ID；
- 两侧 chunk 数；
- 两侧实测可见 SM 数；
- 创建完成后的 `lspart` 输出。

这使得结果目录可以证明实验使用了真正的 static partition，而不是仅改变
dynamic active-thread percentage。

## TDD 与验证

先扩展文本合约测试，使其要求：

- wrapper 支持 `diagnostic_static_64_28`；
- runner 的默认 mode 仍为 `dynamic`；
- static profile 严格绑定 `static + 16/7 chunks`；
- static client 使用 `CUDA_MPS_SM_PARTITION`，dynamic client 使用
  `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`；
- static 生命周期包含 `-S`、`sm_partition create/remove`、可见 SM 校验和
  审计输出。

确认测试先失败后再实现。实现后运行目标 pytest、两个 shell 的 `bash -n`，
最后运行一个 C2 static canary，要求 10/10 请求完成、正确性/routing/session
drain 全部通过，并核对 static MPS 审计文件。

## 后续实验矩阵

canary 通过后，用同一 C4 70:30 北极星请求形状做 2×2 诊断：

| MPS | async decode token | 目的 |
| --- | --- | --- |
| dynamic 70:30 | off | 现有同步控制组 |
| static 64/28 | off | 隔离 static partition 本身的影响 |
| dynamic 70:30 | on | 已观察到 TTFT 退化的实验组 |
| static 64/28 | on | 判断退化是否依赖 dynamic 资源竞争 |

首轮只作为 dirty-worktree 诊断，不替换 formal C4 baseline。只有在代码冻结、
tracked worktree clean 且三次重复后，才允许形成新的正式性能结论。
