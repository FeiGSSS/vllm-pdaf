# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data movement and wire handling for the PAP local transport."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from math import prod
from typing import Any

import torch

from vllm.pap.cuda_stream_memops import (
    stream_wait_value32,
    stream_write_value32,
)
from vllm.pap.deferred_cuda_trace import (
    begin_deferred_cuda_span,
    end_deferred_cuda_span,
    record_deferred_host_duration,
)
from vllm.pap.protocol import PAPOffloadExecBatchDescriptor
from vllm.pap.protocol.offload_exec import (
    _offload_exec_batch_descriptor_from_metadata,
    _offload_exec_batch_descriptor_to_metadata,
    _offload_exec_batch_plan_id,
    _offload_exec_batch_plan_payload,
)
from vllm.pap.transport.local.protocol import (
    _CODE_TO_DTYPE,
    _DTYPE_TO_CODE,
    DIR_OUTPUT,
    DIR_QKV,
    RECORD_FLAG_FIXED_TENSOR,
    RECORD_FLAG_OUTPUT_DESCRIPTORLESS,
    RECORD_FLAG_PLAN_FULL,
    RECORD_FLAG_PLAN_REF,
    _doorbell_ack,
    _doorbell_read_metadata,
    _doorbell_read_record,
    _doorbell_record_offset,
    _dtype_from_name,
    _dtype_name,
    _layer_index_and_template,
    _layer_name_from_template,
    _signal_index,
    _WireMetadata,
)

logger = logging.getLogger(__name__)

SPIN_TIGHT_ITERS = int(os.environ.get("PAP_LOCAL_FAST_SPIN_ITERS", "2048"))
SPIN_YIELD_ITERS = int(os.environ.get("PAP_LOCAL_FAST_YIELD_ITERS", "64"))
SPIN_SLEEP_US = int(os.environ.get("PAP_LOCAL_FAST_SLEEP_US", "20"))
SPIN_SLEEP_AFTER_US = int(os.environ.get("PAP_LOCAL_FAST_SLEEP_AFTER_US", "50"))


def _sched_yield() -> None:
    try:
        os.sched_yield()
    except AttributeError:
        time.sleep(0)


@dataclass
class _LocalFastMessage:
    """Minimal duck-typed stand-in for ``PAPMailboxMessage``.

    The mailbox transport returns message objects with ``tensor``,
    ``release()``, and ``kind`` attributes; some call sites use
    ``recv_*_batch_message`` and then call ``.release()`` on the result.
    For the stream-ordered transport, ``release()`` publishes the slot's GPU
    release generation after the consumer work already queued on the stream.
    """

    msg_id: str
    kind: str
    tensor: torch.Tensor
    metadata: dict[str, Any]
    release_callback: Any = None
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self.release_callback is not None:
            self.release_callback()


class _PAPLocalFastIOMixin:
    """Own local-fast payload encoding, transfer, and receive APIs."""

    def _descriptor_metadata(
        self,
        direction: int,
        descriptor: PAPOffloadExecBatchDescriptor,
    ) -> dict[str, Any]:
        if direction == DIR_OUTPUT:
            self._output_descriptor_elisions += 1
            return {}
        if direction != DIR_QKV:
            raise ValueError(f"invalid PAP local transport direction: {direction}")

        plan_key = descriptor.batch_id_suffix or descriptor.batch_id
        plan_id = self._sent_step_plans.get(plan_key)
        if plan_id is not None:
            self._sent_step_plans.move_to_end(plan_key)
            self._step_plan_refs += 1
            return {
                "v": 5,
                "l": descriptor.layer_name,
                "p": plan_id,
            }

        plan_payload = _offload_exec_batch_plan_payload(descriptor)
        plan_id = _offload_exec_batch_plan_id(plan_payload)
        self._sent_step_plans[plan_key] = plan_id
        self._sent_step_plans.move_to_end(plan_key)
        if self._step_plan_cache_limit > 0:
            while len(self._sent_step_plans) > self._step_plan_cache_limit:
                self._sent_step_plans.popitem(last=False)
        self._step_plan_builds += 1
        return {
            "v": 4,
            "l": descriptor.layer_name,
            "p": plan_id,
            **plan_payload,
        }

    def _wire_metadata(
        self,
        direction: int,
        descriptor: PAPOffloadExecBatchDescriptor,
        tensor: torch.Tensor,
    ) -> _WireMetadata:
        descriptor_metadata = self._descriptor_metadata(direction, descriptor)
        dtype_code = _DTYPE_TO_CODE.get(tensor.dtype)
        layer_info = _layer_index_and_template(descriptor.layer_name)
        if tensor.ndim != 2 or dtype_code is None or layer_info is None:
            raise RuntimeError(
                "PAP local transport requires rank-2 FP16/BF16/FP32 tensors "
                "and indexed transformer layer names"
            )

        shape = (int(tensor.shape[0]), int(tensor.shape[1]))
        layer_index = layer_info[0]
        if direction == DIR_OUTPUT:
            plan_key = descriptor.batch_id_suffix or descriptor.batch_id
            plan_id_text = self._sent_step_plans.get(plan_key)
            if plan_id_text is None:
                plan_id_text = self._recv_plan_ids_by_key.get(plan_key)
            if plan_id_text is None:
                raise RuntimeError("PAP local output is missing its step plan")
            self._binary_outputs += 1
            return _WireMetadata(
                metadata=None,
                plan_id=int(plan_id_text, 16),
                shape=shape,
                layer_index=layer_index,
                dtype_code=dtype_code,
                flags=RECORD_FLAG_FIXED_TENSOR | RECORD_FLAG_OUTPUT_DESCRIPTORLESS,
            )

        metadata_version = int(descriptor_metadata.get("v", 0))
        if metadata_version in {4, 5}:
            try:
                plan_id = int(str(descriptor_metadata["p"]), 16)
            except (KeyError, ValueError):
                plan_id = 0
            if plan_id > 0:
                if metadata_version == 4:
                    self._json_records += 1
                    return _WireMetadata(
                        metadata={"descriptor": descriptor_metadata},
                        plan_id=plan_id,
                        shape=shape,
                        layer_index=layer_index,
                        dtype_code=dtype_code,
                        flags=RECORD_FLAG_FIXED_TENSOR | RECORD_FLAG_PLAN_FULL,
                    )
                self._binary_qkv_refs += 1
                return _WireMetadata(
                    metadata=None,
                    plan_id=plan_id,
                    shape=shape,
                    layer_index=layer_index,
                    dtype_code=dtype_code,
                    flags=RECORD_FLAG_FIXED_TENSOR | RECORD_FLAG_PLAN_REF,
                )

        raise RuntimeError(
            "PAP local QKV descriptor is missing a valid step plan"
        )

    def _send_to_peer(
        self,
        *,
        direction: int,
        descriptor: PAPOffloadExecBatchDescriptor,
        tensor: torch.Tensor,
    ) -> int:
        """Memcpy ``tensor`` into peer's recv buffer and ring the doorbell.

        Returns the slot offset used in the peer's buffer.
        """

        peer = self._require_peer()
        nbytes = int(tensor.numel() * tensor.element_size())
        if nbytes > peer.slot_bytes:
            raise RuntimeError(
                f"PAP local fast payload {nbytes}B exceeds peer slot {peer.slot_bytes}B"
            )
        if nbytes > peer.peer_tensor.numel() * peer.peer_tensor.element_size():
            raise RuntimeError(
                f"PAP local fast payload {nbytes}B exceeds peer IPC buffer "
                f"{peer.peer_tensor.numel() * peer.peer_tensor.element_size()}B"
            )

        wire = self._wire_metadata(direction, descriptor, tensor)

        with peer.send_lock:
            seq = self._next_seq(peer, direction)
            slot_id = (seq - 1) % peer.slot_count
            offset = slot_id * peer.slot_bytes
            last_by_slot = (
                peer.last_qkv_seq_by_slot
                if direction == DIR_QKV
                else peer.last_output_seq_by_slot
            )
            previous_seq = last_by_slot[slot_id]
            self._wait_control_slot(
                peer=peer,
                direction=direction,
                slot_id=slot_id,
                previous_seq=previous_seq,
            )

            # Copy raw bytes from the source tensor into the peer's uint8 recv
            # buffer.  We re-view both sides as 1-D uint8 of the correct length
            # so dtype/shape mismatches don't matter; the receiver reinterprets
            # the bytes via the carried ``nbytes`` + the descriptor's expected
            # layout.
            src_bytes = tensor.detach().contiguous().view(-1).view(torch.uint8)
            src_bytes = src_bytes.narrow(0, 0, nbytes)
            dst_bytes = peer.peer_tensor.narrow(0, offset, nbytes)

            stream = torch.cuda.current_stream(self.device)
            if previous_seq:
                stream_wait_value32(
                    self._signal_buffer,
                    _signal_index(
                        direction,
                        slot_id,
                        self._slot_count,
                        release=True,
                    ),
                    previous_seq,
                    stream,
                )
            t_memcpy_start = time.perf_counter()
            copy_span_name = None
            if self._deferred_cuda_trace:
                copy_span_name = (
                    "qkv_p2p_copy_gpu_ms"
                    if direction == DIR_QKV
                    else "output_p2p_copy_gpu_ms"
                )
            if copy_span_name is not None:
                copy_trace = begin_deferred_cuda_span(
                    copy_span_name,
                    stream,
                )
                try:
                    dst_bytes.copy_(src_bytes, non_blocking=True)
                finally:
                    end_deferred_cuda_span(copy_trace)
            else:
                dst_bytes.copy_(src_bytes, non_blocking=True)
            t_sync_start = time.perf_counter()

            stream_write_value32(
                peer.peer_signal_tensor,
                _signal_index(
                    direction,
                    slot_id,
                    peer.slot_count,
                    release=False,
                ),
                seq,
                stream,
            )
            t_doorbell_start = time.perf_counter()
            self._write_doorbell_sync(
                peer=peer,
                direction=direction,
                seq=seq,
                nbytes=nbytes,
                offset=offset,
                wire=wire,
            )
            t_done = time.perf_counter()
            enqueue_ms = (t_doorbell_start - t_sync_start) * 1000.0
            doorbell_ms = (t_done - t_doorbell_start) * 1000.0
            peer.source_refs[(direction, slot_id)] = src_bytes
            last_by_slot[slot_id] = seq

        if self._trace:
            kind = "qkv" if direction == DIR_QKV else "output"
            logger.info(
                "PAP local fast transport send trace kind=%s layer=%s "
                "batch=%s memcpy_ms=%.3f enqueue_ms=%.3f "
                "doorbell_ms=%.3f "
                "slot=%d nbytes=%d seq=%d wire_flags=%d has_json=%d",
                kind,
                descriptor.layer_name,
                getattr(descriptor, "batch_id", ""),
                (t_sync_start - t_memcpy_start) * 1000.0,
                enqueue_ms,
                doorbell_ms,
                slot_id,
                nbytes,
                seq,
                wire.flags,
                int(bool(wire.metadata)),
            )
        return offset

    def _recv_from_peer(
        self,
        *,
        direction: int,
    ) -> tuple[int, int, int, int, dict[str, Any]]:
        """Spin until peer has rung the doorbell for ``direction``."""

        peer = self._require_peer()
        # We read the *local* doorbell (peer wrote into it via mmap).
        mm = self._doorbell_mm
        expected = (
            peer.expected_qkv_seq if direction == DIR_QKV else peer.expected_output_seq
        )
        slot_id = (expected - 1) % self._slot_count
        record_offset = _doorbell_record_offset(
            direction,
            slot_id,
            self._slot_count,
        )
        iters = 0
        t_start = time.perf_counter()
        while True:
            record = _doorbell_read_record(mm, record_offset)
            seq = record.seq
            if seq == expected:
                break
            if seq > expected:
                raise RuntimeError(
                    "PAP local fast control ring skipped a message: "
                    f"expected={expected} observed={seq} slot={slot_id}"
                )
            iters += 1
            if iters < SPIN_TIGHT_ITERS:
                continue
            if iters < SPIN_TIGHT_ITERS + SPIN_YIELD_ITERS:
                _sched_yield()
                continue
            waited_us = (time.perf_counter() - t_start) * 1_000_000.0
            if waited_us >= SPIN_SLEEP_AFTER_US:
                time.sleep(max(SPIN_SLEEP_US, 0) / 1_000_000.0)
            else:
                _sched_yield()
        t_doorbell_seen = time.perf_counter()
        if self._deferred_cuda_trace and direction == DIR_OUTPUT:
            record_deferred_host_duration(
                "output_doorbell_wait_wall_ms",
                (t_doorbell_seen - t_start) * 1000.0,
            )
        nbytes = record.nbytes
        offset = record.offset
        metadata = _doorbell_read_metadata(
            mm,
            record_offset,
            record.metadata_len,
        )
        if record.flags & RECORD_FLAG_FIXED_TENSOR:
            try:
                dtype = _CODE_TO_DTYPE[record.dtype_code]
            except KeyError as exc:
                raise RuntimeError(
                    "PAP local fast fixed record has invalid dtype code: "
                    f"{record.dtype_code}"
                ) from exc
            metadata.update(
                {
                    "shape": [record.dim0, record.dim1],
                    "dtype": _dtype_name(dtype),
                    "_fixed_flags": record.flags,
                    "_plan_id": record.plan_id,
                    "_layer_index": record.layer_index,
                }
            )
        _doorbell_ack(mm, record_offset, seq)
        stream = torch.cuda.current_stream(self.device)
        ready_span_name = None
        if self._deferred_cuda_trace:
            ready_span_name = (
                "qkv_ready_wait_gpu_ms"
                if direction == DIR_QKV
                else "output_ready_wait_gpu_ms"
            )
        if ready_span_name is not None:
            ready_trace = begin_deferred_cuda_span(
                ready_span_name,
                stream,
            )
            try:
                stream_wait_value32(
                    self._signal_buffer,
                    _signal_index(
                        direction,
                        slot_id,
                        self._slot_count,
                        release=False,
                    ),
                    seq,
                    stream,
                )
            finally:
                end_deferred_cuda_span(ready_trace)
        else:
            stream_wait_value32(
                self._signal_buffer,
                _signal_index(
                    direction,
                    slot_id,
                    self._slot_count,
                    release=False,
                ),
                seq,
                stream,
            )
        # Bump our expectation for the next round.
        if direction == DIR_QKV:
            peer.expected_qkv_seq = expected + 1
        else:
            peer.expected_output_seq = expected + 1
        if self._trace:
            kind = "qkv" if direction == DIR_QKV else "output"
            logger.info(
                "PAP local fast transport recv trace kind=%s "
                "spin_ms=%.3f slot=%d "
                "nbytes=%d offset=%d seq=%d wire_flags=%d has_json=%d",
                kind,
                (time.perf_counter() - t_start) * 1000.0,
                slot_id,
                nbytes,
                offset,
                seq,
                int(metadata.get("_fixed_flags", 0)),
                int(record.metadata_len > 0),
            )
        return seq, slot_id, nbytes, offset, metadata

    def _release_recv_slot(self, direction: int, slot_id: int, seq: int) -> None:
        peer = self._require_peer()
        stream_write_value32(
            peer.peer_signal_tensor,
            _signal_index(
                direction,
                slot_id,
                peer.slot_count,
                release=True,
            ),
            seq,
            torch.cuda.current_stream(self.device),
        )

    def _materialize_recv(
        self,
        *,
        nbytes: int,
        offset: int,
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        shape = tuple(int(dim) for dim in metadata["shape"])
        dtype = _dtype_from_name(str(metadata["dtype"]))
        expected_nbytes = int(prod(shape) * torch.empty((), dtype=dtype).element_size())
        if expected_nbytes != int(nbytes):
            raise RuntimeError(
                f"PAP local fast payload size mismatch: metadata shape={shape} "
                f"dtype={dtype} expects {expected_nbytes} bytes, got {nbytes}"
            )
        view = self._recv_buffer.narrow(0, int(offset), int(nbytes))
        return view.view(dtype).reshape(shape)

    def _validate_output_record(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        metadata: dict[str, Any],
    ) -> None:
        flags = int(metadata.get("_fixed_flags", 0))
        if not flags & RECORD_FLAG_OUTPUT_DESCRIPTORLESS:
            return
        layer_info = _layer_index_and_template(descriptor.layer_name)
        if layer_info is not None and int(metadata["_layer_index"]) != layer_info[0]:
            raise RuntimeError("PAP local fast output layer index mismatch")
        plan_key = descriptor.batch_id_suffix or descriptor.batch_id
        expected_plan_id = self._sent_step_plans.get(plan_key)
        received_plan_id = int(metadata.get("_plan_id", 0))
        if (
            expected_plan_id is not None
            and received_plan_id > 0
            and int(expected_plan_id, 16) != received_plan_id
        ):
            raise RuntimeError("PAP local fast output step plan id mismatch")

    # ------------------------------------------------------------------
    # Public transport API (mirrors PAPNixlMailboxOffloadExecTransport)
    # ------------------------------------------------------------------

    # --- batched data plane ---

    def send_qkv_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        qkv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        self._send_to_peer(direction=DIR_QKV, descriptor=descriptor, tensor=qkv)

    def send_qkv_batch_direct(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        qkv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        # In the local-fast path there is no separate "direct payload slot"
        # ceremony; the batched path is already direct.
        self.send_qkv_batch(descriptor, qkv, remote_address=remote_address)

    def send_output_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        output: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        self._send_to_peer(direction=DIR_OUTPUT, descriptor=descriptor, tensor=output)

    def recv_output_batch_message(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        remote_address: str,
    ) -> Any:
        seq, slot_id, nbytes, offset, metadata = self._recv_from_peer(
            direction=DIR_OUTPUT
        )
        self._validate_output_record(descriptor, metadata)
        tensor = self._materialize_recv(
            nbytes=nbytes,
            offset=offset,
            metadata=metadata,
        )
        return _LocalFastMessage(
            msg_id=descriptor.output_tensor_id,
            kind="attention_result_batch",
            tensor=tensor,
            metadata=_offload_exec_batch_descriptor_to_metadata(descriptor),
            release_callback=lambda: self._release_recv_slot(
                DIR_OUTPUT,
                slot_id,
                seq,
            ),
        )

    def prepare_output_batch_message(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        shape: tuple[int, int],
        dtype: torch.dtype,
        remote_address: str,
    ) -> Any:
        """Reserve the next output slot and enqueue its GPU-ready wait.

        The sealed step plan supplies output shape and dtype, so Projection
        does not need to wait for or read the per-layer output doorbell.  The
        GPU ready/release generations remain the payload-lifetime authority.
        """
        del remote_address
        if dtype not in _DTYPE_TO_CODE:
            raise RuntimeError("PAP descriptorless output dtype is unsupported")
        if tuple(shape)[0] != descriptor.item_count:
            raise RuntimeError("PAP descriptorless output row count mismatch")

        peer = self._require_peer()
        seq = int(peer.expected_output_seq)
        slot_id = (seq - 1) % self._slot_count
        offset = slot_id * self._slot_bytes
        nbytes = int(prod(shape) * torch.empty((), dtype=dtype).element_size())
        if nbytes > self._slot_bytes:
            raise RuntimeError(
                f"PAP descriptorless output {nbytes}B exceeds slot {self._slot_bytes}B"
            )

        record_offset = _doorbell_record_offset(
            DIR_OUTPUT,
            slot_id,
            self._slot_count,
        )
        _doorbell_ack(self._doorbell_mm, record_offset, seq)
        stream = torch.cuda.current_stream(self.device)
        wait_trace = None
        if self._deferred_cuda_trace:
            wait_trace = begin_deferred_cuda_span(
                "output_ready_wait_gpu_ms",
                stream,
            )
        try:
            stream_wait_value32(
                self._signal_buffer,
                _signal_index(
                    DIR_OUTPUT,
                    slot_id,
                    self._slot_count,
                    release=False,
                ),
                seq,
                stream,
            )
        finally:
            end_deferred_cuda_span(wait_trace)
        peer.expected_output_seq = seq + 1
        self._descriptorless_output_receives += 1

        tensor = self._materialize_recv(
            nbytes=nbytes,
            offset=offset,
            metadata={"shape": list(shape), "dtype": _dtype_name(dtype)},
        )
        return _LocalFastMessage(
            msg_id=descriptor.output_tensor_id,
            kind="attention_result_batch",
            tensor=tensor,
            metadata=_offload_exec_batch_descriptor_to_metadata(descriptor),
            release_callback=lambda: self._release_recv_slot(
                DIR_OUTPUT,
                slot_id,
                seq,
            ),
        )

    def recv_next_qkv_batch_message(
        self,
    ) -> tuple[PAPOffloadExecBatchDescriptor, Any]:
        seq, slot_id, nbytes, offset, metadata = self._recv_from_peer(direction=DIR_QKV)
        fixed_flags = int(metadata.get("_fixed_flags", 0))
        if fixed_flags & RECORD_FLAG_PLAN_REF:
            plan_id = f"{int(metadata['_plan_id']):016x}"
            try:
                layer_template = self._recv_plan_layer_templates[plan_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"PAP local fast received unknown step plan id: {plan_id}"
                ) from exc
            descriptor_metadata = {
                "v": 5,
                "l": _layer_name_from_template(
                    layer_template,
                    int(metadata["_layer_index"]),
                ),
                "p": plan_id,
            }
        else:
            descriptor_metadata = dict(metadata["descriptor"])
            if fixed_flags & RECORD_FLAG_PLAN_FULL:
                plan_id = str(descriptor_metadata["p"])
                layer_info = _layer_index_and_template(str(descriptor_metadata["l"]))
                if layer_info is None:
                    raise RuntimeError(
                        "PAP local fast step plan has an invalid layer name"
                    )
                if int(plan_id, 16) != int(metadata["_plan_id"]):
                    raise RuntimeError("PAP local fast step plan id mismatch")
                self._recv_plan_layer_templates[plan_id] = layer_info[1]
        descriptor = _offload_exec_batch_descriptor_from_metadata(
            descriptor_metadata,
            plan_cache=self._recv_batch_plans,
            template_only=True,
        )
        received_plan_id = int(metadata.get("_plan_id", 0))
        if received_plan_id > 0:
            plan_key = descriptor.batch_id_suffix or descriptor.batch_id
            self._recv_plan_ids_by_key[plan_key] = f"{received_plan_id:016x}"
        tensor = self._materialize_recv(nbytes=nbytes, offset=offset, metadata=metadata)
        message = _LocalFastMessage(
            msg_id=descriptor.qkv_tensor_id,
            kind="attention_task_batch",
            tensor=tensor,
            metadata=descriptor_metadata,
            release_callback=lambda: self._release_recv_slot(
                DIR_QKV,
                slot_id,
                seq,
            ),
        )
        return descriptor, message

    # ------------------------------------------------------------------
    # Cleanup (best-effort; daemon process exit will reclaim resources)
    # ------------------------------------------------------------------
