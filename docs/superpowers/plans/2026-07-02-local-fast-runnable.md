# PAP Local Fast Runnable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `PAP_OFFLOAD_EXEC_TRANSPORT=local_fast` carry descriptor metadata and typed tensor layout so PAP decode can run end-to-end and be profiled against NIXL-based PAP and PD.

**Architecture:** Extend the existing `/dev/shm` doorbell mmap from a 32-byte data-only record to a fixed-size per-direction record containing header plus JSON metadata. Sender writes descriptor metadata, dtype, and shape before publishing the sequence number; receiver spin-waits on the sequence number, parses metadata, reconstructs the `PAPOffloadExecBatchDescriptor`, and materializes the recv buffer as a typed tensor view.

**Tech Stack:** PyTorch CUDA tensors and CUDA IPC scaffold, Python stdlib `mmap`/`json`/`struct`, existing PAP data-plane descriptor helpers, existing benchmark scripts.

---

### Task 1: Doorbell metadata record

**Files:**
- Modify: `vllm/pap/local_fast_transport.py`

- [ ] Increase per-direction doorbell record size from 32 bytes to a fixed 64 KiB record.
- [ ] Change header layout from `(seq, nbytes, offset, reserved)` to `(seq, nbytes, offset, metadata_len)`.
- [ ] Write metadata JSON bytes into the record body before writing the header.
- [ ] Read metadata JSON after observing the expected sequence.

### Task 2: Typed materialization

**Files:**
- Modify: `vllm/pap/local_fast_transport.py`

- [ ] Add dtype name/from-name helpers.
- [ ] Send metadata with `shape`, `dtype`, and descriptor metadata for QKV/output.
- [ ] Change `_materialize_recv()` to return `self._recv_buffer.narrow(...).view(dtype).reshape(shape)`.
- [ ] Validate `prod(shape) * element_size == nbytes`.

### Task 3: Descriptor reconstruction for attention loop

**Files:**
- Modify: `vllm/pap/local_fast_transport.py`

- [ ] Make `recv_next_qkv_batch_message()` parse descriptor metadata from the QKV doorbell record.
- [ ] Return `_LocalFastMessage` with the real `attention_task_batch` metadata.
- [ ] Keep `recv_qkv_batch_message()` and `recv_output_batch_message()` consistent.

### Task 4: Verification and benchmark

**Files:**
- No source modifications expected.

- [ ] AST parse: `local_fast_transport.py`, `data_plane.py`, `pap_attention_executor.py`, `qwen3.py`.
- [ ] Import check for `PAPLocalFastTransport`.
- [ ] Run `PAP_OFFLOAD_EXEC_TRANSPORT=local_fast` benchmark for 1PA1P Qwen3-8B i128/o16/q16 if GPUs are available.
- [ ] Parse trace summary and compare TPOT/transit against NIXL-based PAP (`20260702_175525`) and available PD baseline.
