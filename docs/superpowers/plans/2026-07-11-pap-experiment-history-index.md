# PAP Experiment History Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and Git-track one progressive-disclosure index that connects
the complete PAP timeline, subsystem motivations, implementation commits,
controlled experiments, negative results, and raw evidence locations.

**Architecture:** Create a single curated Markdown entry point under
`docs/design/`. Its top level answers timeline and module questions; stable
experiment IDs and uniform module dossiers then lead to existing tracked design
documents, commits, raw run directories, audits, and traces. Evidence remains
in place and is classified as tracked, repo-untracked, external, temporary, or
missing.

**Tech Stack:** Git history, Markdown, Bash, `rg`, `find`, `sed`, and existing
PAP benchmark metadata. No runtime code or dependencies are added.

## Global Constraints

- Cover the full PAP history reachable from `feature/pap`, from 2026-05-22
  through the current 2026-07-11 multi-turn validation.
- Include accepted, opt-in, negative, failed, invalid, reverted, removed,
  superseded, and inconclusive work.
- Do not move, rename, delete, or commit raw result directories.
- Do not silently drop missing or machine-local evidence; classify it.
- Do not upgrade dirty, traced, partial, or smoke runs to formal baselines.
- Summarize existing documents instead of duplicating their long tables.
- Use relative Markdown links for tracked documents and symbolic roots for raw
  paths.
- Never use system `python3`, bare `pip`, or pre-commit.

---

### Task 1: Build the evidence inventory and index skeleton

**Files:**
- Create: `docs/design/pap-experiment-history-index.md`
- Read: `docs/superpowers/specs/2026-07-11-pap-experiment-history-index-design.md`
- Read: all tracked `docs/design/pap-*.md` and `docs/superpowers/*/*pap*.md`
- Read: `$PAP_RESULTS`, `$PAP_REPO_RESULTS`, `$PD_RESULTS`, and
  `$PAP_PROFILES`

**Interfaces:**
- Consumes: the approved symbolic roots, phase list, module list, evidence
  grades, and status vocabulary from the specification.
- Produces: stable anchors, symbolic-root definitions, phase IDs, module IDs,
  and experiment IDs used by every later task.

- [ ] **Step 1: Confirm the branch and evidence roots without mutation**

Run:

```bash
git branch --show-current
git status --porcelain --untracked-files=no
git log --reverse --date=short --format='%ad|%h|%s' \
  --since=2026-05-01 --grep='PAP\|pap' -i HEAD
find /home/fei/research/PD/test/baseline/pap/results/runs \
  -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
find test/baseline/pap/results/runs \
  -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
```

Expected: branch is `feature/pap`, tracked status is empty, and both result
roots enumerate existing runs.

- [ ] **Step 2: Create the exact top-level structure**

Create the index with these sections in this order:

```markdown
# PAP 开发与实验历史索引

## 0. 如何使用本索引
## 1. 当前状态与证据边界
## 2. 路径与存储类别
## 3. 全局时间线
## 4. 模块地图
## 5. 模块档案
### M1. PAP 分离架构与控制流
### M2. OFFLOAD_EXEC 与 NIXL Mailbox
### M3. Projection KV-unaware 调度
### M4. Prefill-owned Shared/Unified KV
### M5. Decode Commit、Lease 与正确性闭环
### M6. Benchmark、Tracing 与审计体系
### M7. Same-node Local-fast 与 TPOT 热路径
### M8. 任意 x:y 拓扑与路由
### M9. 多对多 Cohort、Combine/Scatter 与 Active Peer
### M10. 多轮对话与原生 Prefix Cache 复用
## 6. 实验账本
## 7. 负结果、回滚与被替代路线
## 8. 关键提交时间线
## 9. 未完成问题与外部依赖
## 10. 新增实验记录模板
```

- [ ] **Step 3: Add navigation and evidence conventions**

The opening must define the three lookup paths, symbolic roots, storage labels,
evidence grades, decision vocabulary, and a warning that Git does not preserve
repo-untracked/external/temporary artifacts.

- [ ] **Step 4: Validate the skeleton**

Run:

```bash
rg -n '^## |^### M[0-9]+\.' docs/design/pap-experiment-history-index.md
git diff --check
```

Expected: all 11 top-level numbered sections and all 10 module headings are
present; diff check exits zero.

### Task 2: Populate the phase timeline and module dossiers

**Files:**
- Modify: `docs/design/pap-experiment-history-index.md`
- Reference: all tracked PAP design/spec/plan documents

**Interfaces:**
- Consumes: phase and module anchors from Task 1.
- Produces: motivation-to-decision chains referenced by experiment-ledger rows.

- [ ] **Step 1: Add all eight timeline phases**

Use stable IDs and dates:

```text
P1  2026-05-22..25  NIXL/true-split prototype and first shared-KV ownership
P2  2026-05-26      Mailbox, topology, 4PA4P/6PA2P, 3-way pipeline
P3  2026-05-27..28  MoE, 30B/32B, wavefront, concurrency, TP/TP2
P4  2026-06-30..07-01  Upstream sync, PD/PAP methodology, mailbox hot path
P5  2026-07-02..07  Remote-attention tracing, local-fast, unified KV, commit
P6  2026-07-10      ACK/lease hardening, same-node slot-plan TPOT optimization
P7  2026-07-10..11  Arbitrary x:y and many-to-many scheduling
P8  2026-07-11      Exact-token and Chat multi-turn APC reuse
```

Each phase includes its question, key commits, accepted conclusion, negative
branches, evidence grade, and links to owning dossiers.

- [ ] **Step 2: Populate M1-M5**

For each dossier, write the exact schema:

```markdown
#### 问题与动机
#### 设计与机制
#### 关键实现提交
#### 关键实验与证据
#### 负结果与被替代方案
#### 当前结论与边界
#### 深入阅读与原始证据
```

M1-M5 must connect the May prototypes, Projection metadata-only/KV-unaware
path, Prefill-owned paged KV, unified KV append, decode commit, ACK, lease, and
release-order fixes.

- [ ] **Step 3: Populate M6-M10**

M6-M10 must connect the PD/PAP workload contract, trace attribution,
local-fast/doorbell/slot-plan path, arbitrary x:y, central dispatcher,
same-layer combine/scatter, vectorized route copy, adaptive coalescing,
active-peer membership, and exact/Chat multi-turn reuse.

- [ ] **Step 4: Check dossier completeness**

Run:

```bash
for id in M1 M2 M3 M4 M5 M6 M7 M8 M9 M10; do
  rg -q "^### ${id}\." docs/design/pap-experiment-history-index.md
done
rg -n '^#### (问题与动机|设计与机制|关键实现提交|关键实验与证据|负结果与被替代方案|当前结论与边界|深入阅读与原始证据)$' \
  docs/design/pap-experiment-history-index.md
git diff --check
```

Expected: all module anchors exist and each dossier exposes the seven standard
disclosure headings.

### Task 3: Populate the experiment ledger and negative-result registry

**Files:**
- Modify: `docs/design/pap-experiment-history-index.md`
- Read: representative run metadata, benchmark JSON, audit files, and design
  result tables

**Interfaces:**
- Consumes: module IDs and symbolic roots.
- Produces: stable experiment IDs and decision records used for future
  timeline reconstruction.

- [ ] **Step 1: Add the minimum logical experiment set**

The ledger must include at least these stable entries:

```text
PAP-20260522-PROTO-NIXL
PAP-20260524-PROJECTION-KVUNAWARE
PAP-20260524-SHARED-KV
PAP-20260526-MAILBOX
PAP-20260526-3WAY
PAP-20260527-WAVEFRONT
PAP-20260527-CONCURRENCY
PAP-20260528-TP2
PAP-20260701-PD-METHODOLOGY
PAP-20260701-MAILBOX-HOTPATH
PAP-20260702-REMOTE-TRACE
PAP-20260703-UNIFIED-KV
PAP-20260703-SLOTMAPPING
PAP-20260706-DECODE-COMMIT
PAP-20260710-ACK-LEASE
PAP-20260710-SLOTPLAN
PAP-20260710-QPS4-PD-AB
PAP-20260710-ARBITRARY-XY
PAP-20260711-CENTRAL-DISPATCH
PAP-20260711-ATTENTION-COMBINE
PAP-20260711-ROUTE-COPY
PAP-20260711-ADAPTIVE-COALESCE
PAP-20260711-ACTIVE-PEER
PAP-20260711-MULTITURN-EXACT
PAP-20260711-MULTITURN-CHAT
```

Each logical entry supplies module, question, baseline/treatment, workload,
commit/clean state, minimal result, evidence grade, decision, and artifact
paths. Repetitions remain grouped under one ID.

- [ ] **Step 2: Add first-class negative entries**

The negative registry must include, where evidence exists:

```text
prototype TCP/NCCL paths later removed
Q-first/KV-later and Attention partial overlap
mailbox async/piggyback/inline micro-optimizations that regressed
3-way or wavefront shapes that fragmented batches
Attention-local KV and copy-prefix fallbacks
invalid OOM/incomplete high-scale runs
per-row unified-KV slot mapping
1PA2P split-peer small-batch regression
adaptive coalescing that lost to fixed waiting
resident-session multi-turn proposal replaced by native APC
Projection fail-closed false positive from zero local blocks
Qwen3 enable_thinking=false token discontinuity
```

Each entry links the rejecting evidence and the replacement or current state.

- [ ] **Step 3: Add the append-only maintenance template**

The template contains all ledger fields plus explicit checkboxes for strict
correctness, routing, decode commit, lease release, session drain, and raw-path
storage class.

- [ ] **Step 4: Validate experiment and negative coverage**

Run:

```bash
for id in \
  PAP-20260522-PROTO-NIXL \
  PAP-20260710-SLOTPLAN \
  PAP-20260711-ROUTE-COPY \
  PAP-20260711-MULTITURN-EXACT \
  PAP-20260711-MULTITURN-CHAT; do
  rg -q "$id" docs/design/pap-experiment-history-index.md
done
rg -n 'reject|回滚|被替代|invalid|退化|负结果' \
  docs/design/pap-experiment-history-index.md
git diff --check
```

Expected: representative early, middle, many-to-many, and multi-turn IDs exist;
negative decisions are discoverable; diff check exits zero.

### Task 4: Verify provenance, links, and commit the index

**Files:**
- Modify: `docs/design/pap-experiment-history-index.md`
- Modify: `docs/superpowers/plans/2026-07-11-pap-experiment-history-index.md`

**Interfaces:**
- Consumes: the completed index.
- Produces: a tracked, self-consistent historical entry point and a completed
  implementation checklist.

- [ ] **Step 1: Verify commit references**

Extract backticked 9-character hexadecimal hashes and verify each object:

```bash
for commit in $(rg -o '`[0-9a-f]{9}`' \
  docs/design/pap-experiment-history-index.md | tr -d '`' | sort -u); do
  git cat-file -e "${commit}^{commit}"
done
```

Expected: every referenced commit resolves.

- [ ] **Step 2: Verify tracked Markdown links**

List relative Markdown targets under `docs/` and confirm each file exists. Fix
broken links; do not replace them with opaque prose references.

```bash
rg -o '\[[^]]+\]\(([^)]+\.md)\)' \
  docs/design/pap-experiment-history-index.md
git diff --check
```

Expected: every emitted `.md` target resolves relative to the index file.

- [ ] **Step 3: Verify representative artifact roots**

Check at least one existing directory for each of prototype/mailbox, TPOT,
many-to-many, and multi-turn phases. Missing historical references remain in
the index with a `missing` label.

- [ ] **Step 4: Self-review the index**

Check:

```bash
rg -n 'T[B]D|T[O]DO|F[I]XME|unknown without explanation' \
  docs/design/pap-experiment-history-index.md || true
rg -n '^## |^### M[0-9]+\.' docs/design/pap-experiment-history-index.md
git diff --check
```

Expected: no unresolved placeholder, all required sections are present, and
formatting is clean.

- [ ] **Step 5: Mark the plan complete and commit**

Stage only the index and this plan, then commit with hooks disabled:

```bash
git add \
  docs/design/pap-experiment-history-index.md \
  docs/superpowers/plans/2026-07-11-pap-experiment-history-index.md
git commit --no-verify -m "Index PAP development and experiment history"
```

The full commit message must include the existing Codex co-author and human
sign-off trailers.

- [ ] **Step 6: Confirm final tracked state**

```bash
git status --porcelain --untracked-files=no
git log -2 --oneline
```

Expected: tracked worktree is clean and the index commit is HEAD.
