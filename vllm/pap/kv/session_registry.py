# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session, Prefill catalog, and lease ownership for PAP Attention."""

# mypy: disable-error-code="attr-defined, has-type"

from __future__ import annotations

import logging
import time
from threading import Thread
from typing import Any

import torch

from vllm.pap.kv.control_client import DecodeCommitClient as _DecodeCommitClient
from vllm.pap.kv.control_client import (
    LeaseReleaseClient as _LeaseReleaseClient,
)
from vllm.pap.kv.lease import _kv_lease_profile_enabled
from vllm.pap.kv.models import (
    PAPAttentionSession,
    PAPAttentionStepContext,
    PAPPrefillKVCacheCatalogEntry,
    PAPPrefillLayerReadiness,
    PAPUnifiedPagedKVState,
)
from vllm.pap.protocol import (
    PAPAttentionRegistration,
    PAPPrefillKVCacheCatalogDescriptor,
    PAPPrefillKVSessionManifest,
)

logger = logging.getLogger("pap_attention")

_commit_client: _DecodeCommitClient | None = None
_lease_release_client: _LeaseReleaseClient | None = None

_DECODE_COMMIT_PATH = "/v1/pap/prefill/decode-commit"
_LEASE_RELEASE_PATH = "/v1/pap/prefill/lease-release"
_RELEASED_SESSION_ALIAS_LIMIT = 4096


def _prefill_control_endpoint(prefill_endpoint: str, path: str) -> str:
    return f"{str(prefill_endpoint).rstrip('/')}{path}"


def _get_commit_client() -> _DecodeCommitClient:
    global _commit_client
    if _commit_client is None:
        _commit_client = _DecodeCommitClient()
    return _commit_client


def _get_lease_release_client() -> _LeaseReleaseClient:
    global _lease_release_client
    if _lease_release_client is None:
        _lease_release_client = _LeaseReleaseClient()
    return _lease_release_client


class _PAPSessionRegistryMixin:
    """Own session, Prefill catalog, readiness, and lease transitions."""

    _prefill_kv_catalog_id: str | None
    _last_attention_step_context: PAPAttentionStepContext | None

    def _release_session_locked(
        self, request_id: str
    ) -> tuple[bool, str | None, str | None]:
        session = self._sessions.pop(request_id, None)
        existed = session is not None
        prefill_endpoint = None if session is None else session.prefill_endpoint
        if session is not None:
            released_aliases = {request_id, session.prefill_kv_handle}
            released_aliases.update(
                cached_request_id
                for cached_request_id, cached_session_id in (
                    self._request_id_resolution_cache.items()
                )
                if cached_session_id == request_id
            )
            for alias in released_aliases:
                self._released_session_aliases[alias] = request_id
                self._released_session_aliases.move_to_end(alias)
            while len(self._released_session_aliases) > _RELEASED_SESSION_ALIAS_LIMIT:
                self._released_session_aliases.popitem(last=False)
        self._session_manifest_prefix_lens.pop(request_id, None)
        self._session_manifest_ready_prefix_lens.pop(request_id, None)
        self._session_manifest_events.pop(request_id, None)
        self._session_manifest_claimed.discard(request_id)
        self._prefill_readiness.pop(request_id, None)
        self._unified_paged_kv.pop(request_id, None)
        self._session_epochs.pop(request_id, None)
        self._unified_slot_activations.pop(request_id, None)
        self._drop_offload_exec_session_entry_cache_locked(request_id)
        self._drop_attention_step_contexts_locked(request_id)
        lease_id = self._session_lease_ids.pop(request_id, None)
        leased_blocks = self._session_leased_block_ids.pop(request_id, None)
        self._session_lease_capacity_tokens.pop(request_id, None)
        if lease_id is not None and _kv_lease_profile_enabled():
            logger.info(
                "PAP Attention release session lease request_id=%s "
                "lease_id=%s leased_blocks=%d",
                request_id,
                lease_id,
                len(leased_blocks or ()),
            )
        for cached_request_id, cached_session_id in list(
            self._request_id_resolution_cache.items()
        ):
            if cached_request_id == request_id or cached_session_id == request_id:
                self._request_id_resolution_cache.pop(cached_request_id, None)
        return existed, lease_id, prefill_endpoint

    def _replace_existing_session_locked(
        self, request_id: str
    ) -> tuple[str | None, str | None]:
        if request_id in self._sessions:
            _existed, lease_id, prefill_endpoint = self._release_session_locked(
                request_id
            )
            return lease_id, prefill_endpoint

        self._session_manifest_prefix_lens.pop(request_id, None)
        self._session_manifest_ready_prefix_lens.pop(request_id, None)
        self._session_manifest_events.pop(request_id, None)
        self._session_manifest_claimed.discard(request_id)
        self._prefill_readiness.pop(request_id, None)
        self._unified_paged_kv.pop(request_id, None)
        self._session_epochs.pop(request_id, None)
        self._unified_slot_activations.pop(request_id, None)
        self._drop_offload_exec_session_entry_cache_locked(request_id)
        self._drop_attention_step_contexts_locked(request_id)
        lease_id = self._session_lease_ids.pop(request_id, None)
        self._session_leased_block_ids.pop(request_id, None)
        self._session_lease_capacity_tokens.pop(request_id, None)
        for cached_request_id, cached_session_id in list(
            self._request_id_resolution_cache.items()
        ):
            if cached_request_id == request_id or cached_session_id == request_id:
                self._request_id_resolution_cache.pop(cached_request_id, None)
        self._sessions.pop(request_id, None)
        self._request_id_resolution_cache.pop(request_id, None)
        return lease_id, None

    def release_session(
        self,
        request_id: str,
    ) -> bool:
        request_id = self.resolve_session_request_id(request_id) or str(request_id)
        deferred_flushed = self._decode_token_committer.flush_request(
            request_id,
            timeout_s=self._decode_token_flush_timeout_s,
        )
        if not deferred_flushed:
            logger.warning(
                "PAP decode-token join flush timed out before session release "
                "request_id=%s",
                request_id,
            )
        with self._lock:
            existed, lease_id, prefill_endpoint = self._release_session_locked(
                request_id
            )
        commit_client = _get_commit_client()
        flush_submitted = getattr(
            commit_client,
            "flush_submitted_request",
            commit_client.flush_request,
        )
        if not flush_submitted(request_id):
            logger.warning(
                "PAP decode commit submission timed out before lease release "
                "request_id=%s",
                request_id,
            )
            self._decode_token_committer.forget_request(request_id)
            return existed
        if lease_id is not None:
            release_endpoint = (
                None
                if prefill_endpoint is None
                else _prefill_control_endpoint(
                    prefill_endpoint,
                    _LEASE_RELEASE_PATH,
                )
            )
            release_client = _get_lease_release_client()
            release_client.release(
                request_id=request_id,
                lease_id=lease_id,
                endpoint=release_endpoint,
            )
        commit_client.forget_request(request_id)
        self._decode_token_committer.forget_request(request_id)
        return existed

    def register_prefill_kv_catalog(
        self,
        *,
        descriptor: PAPPrefillKVCacheCatalogDescriptor,
        kv_cache: torch.Tensor,
    ) -> bool:
        """Register one immutable process-lifetime Prefill KV-cache layer."""
        entry = PAPPrefillKVCacheCatalogEntry(
            catalog_id=descriptor.catalog_id,
            layer_name=descriptor.layer_name,
            kv_cache=kv_cache.detach(),
            block_size=descriptor.block_size,
            num_kv_heads=descriptor.num_kv_heads,
            layout=descriptor.layout,
        )
        with self._lock:
            catalog_id = self._prefill_kv_catalog_id
            if catalog_id is None:
                self._prefill_kv_catalog_id = descriptor.catalog_id
            elif catalog_id != descriptor.catalog_id:
                raise RuntimeError(
                    "PAP Prefill KV catalog epoch changed while Attention is "
                    f"running current={catalog_id} incoming={descriptor.catalog_id}"
                )
            existing = self._prefill_kv_catalog.get(descriptor.layer_name)
            if existing is not None:
                if (
                    existing.catalog_id != entry.catalog_id
                    or existing.block_size != entry.block_size
                    or existing.num_kv_heads != entry.num_kv_heads
                    or existing.layout != entry.layout
                    or tuple(existing.kv_cache.shape) != tuple(entry.kv_cache.shape)
                    or existing.kv_cache.dtype != entry.kv_cache.dtype
                ):
                    raise RuntimeError(
                        "PAP Prefill KV catalog layer changed after registration "
                        f"layer={descriptor.layer_name}"
                    )
                return False
            self._prefill_kv_catalog[descriptor.layer_name] = entry
            self._prefill_condition.notify_all()
        logger.info(
            "PAP Prefill KV catalog registered catalog_id=%s layer=%s shape=%s",
            descriptor.catalog_id,
            descriptor.layer_name,
            tuple(kv_cache.shape),
        )
        self._start_paged_decode_warmup(
            kv_cache=entry.kv_cache,
            block_size=entry.block_size,
        )
        return True

    def _start_paged_decode_warmup(
        self,
        *,
        kv_cache: torch.Tensor,
        block_size: int,
    ) -> None:
        if kv_cache.device.type != "cuda":
            return
        with self._lock:
            if self._paged_decode_warmup_started:
                return
            self._paged_decode_warmup_started = True
        num_heads = int(self._offload_exec_shape_defaults[2])
        head_dim = int(self._offload_exec_shape_defaults[4])

        def _warm() -> None:
            try:
                from vllm.pap.attention.triton_backend import (
                    warm_paged_decode_attention,
                )

                warm_paged_decode_attention(
                    kv_cache=kv_cache,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                )
            except Exception:
                with self._lock:
                    self._paged_decode_warmup_failed = True
                logger.exception("PAP paged decode kernel warmup failed")
                return
            with self._lock:
                self._paged_decode_warmup_done = True
            logger.info("PAP paged decode kernel warmup complete")

        Thread(
            target=_warm,
            name="pap-paged-decode-warmup",
            daemon=True,
        ).start()

    def install_prefill_kv_session_manifest(
        self,
        *,
        manifest: PAPPrefillKVSessionManifest,
        ready_event: Any | None,
    ) -> int:
        """Atomically publish one complete request-level KV layout snapshot."""
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(
                manifest.session_handle
            )
            if session_request_id is None:
                raise KeyError(manifest.session_handle)
            request_session_id = self._resolve_session_request_id_locked(
                manifest.request_id
            )
            if request_session_id != session_request_id:
                raise RuntimeError(
                    "PAP Prefill KV manifest session generation mismatch "
                    f"request_id={manifest.request_id} "
                    f"session_handle={manifest.session_handle}"
                )
            if self._prefill_kv_catalog_id != manifest.catalog_id:
                raise RuntimeError(
                    "PAP Prefill KV manifest catalog mismatch "
                    f"current={self._prefill_kv_catalog_id} "
                    f"incoming={manifest.catalog_id}"
                )
            if len(self._prefill_kv_catalog) != manifest.expected_layer_count:
                raise RuntimeError(
                    "PAP Prefill KV manifest layer count mismatch "
                    f"catalog={len(self._prefill_kv_catalog)} "
                    f"expected={manifest.expected_layer_count}"
                )
            previous_prefix_len = self._session_manifest_prefix_lens.get(
                session_request_id
            )
            if session_request_id in self._session_manifest_claimed:
                if previous_prefix_len == manifest.prefix_len:
                    return manifest.prefix_len
                raise RuntimeError(
                    "PAP Prefill KV manifest changed after Decode claimed layout "
                    f"request_id={session_request_id} "
                    f"current={previous_prefix_len} incoming={manifest.prefix_len}"
                )
            if (
                previous_prefix_len is not None
                and manifest.prefix_len < previous_prefix_len
            ):
                raise RuntimeError(
                    "PAP Prefill KV manifest prefix regressed "
                    f"request_id={session_request_id} "
                    f"current={previous_prefix_len} incoming={manifest.prefix_len}"
                )
            existing_lease = self._session_lease_ids.get(session_request_id)
            if existing_lease not in (None, manifest.lease_id):
                raise RuntimeError(
                    "PAP Prefill KV manifest lease changed "
                    f"request_id={session_request_id}"
                )
            self._session_lease_ids[session_request_id] = manifest.lease_id
            self._session_leased_block_ids[session_request_id] = (
                manifest.leased_block_ids
            )
            self._session_lease_capacity_tokens[session_request_id] = (
                manifest.lease_capacity_tokens
            )

            layer_states: dict[str, PAPUnifiedPagedKVState] = {}
            for layer_name, entry in sorted(self._prefill_kv_catalog.items()):
                if entry.block_size != manifest.block_size:
                    raise RuntimeError(
                        "PAP Prefill KV manifest block size differs from catalog "
                        f"layer={layer_name}"
                    )
                state = PAPUnifiedPagedKVState(
                    kv_cache=entry.kv_cache,
                    block_ids=manifest.block_ids,
                    prefix_len=manifest.prefix_len,
                    seq_len=manifest.prefix_len,
                    writable_start_token=manifest.writable_start_token,
                    writable_end_token=manifest.writable_end_token,
                    lease_id=manifest.lease_id,
                    block_size=entry.block_size,
                    num_kv_heads=entry.num_kv_heads,
                    layout=entry.layout,
                )
                self._record_unified_slot_topology_locked(
                    session_request_id=session_request_id,
                    layer_name=layer_name,
                    state=state,
                )
                layer_states[layer_name] = state
            activation = self._unified_slot_activations[session_request_id]
            activation.expected_layers = frozenset(layer_states)
            activation.complete = True
            self._unified_paged_kv[session_request_id] = layer_states

            session = self._sessions[session_request_id]
            session.prefix_len = manifest.prefix_len
            session.seq_len = max(int(session.seq_len), manifest.prefix_len)
            session.block_ids = manifest.block_ids
            readiness_by_layer = self._prefill_readiness.setdefault(
                session_request_id,
                {},
            )
            readiness_by_layer.clear()
            for layer_name in layer_states:
                self._mark_prefill_opened_locked(
                    session_request_id=session_request_id,
                    layer_name=layer_name,
                )
            self._session_manifest_prefix_lens[session_request_id] = manifest.prefix_len
            if ready_event is None:
                for layer_name in layer_states:
                    self._mark_prefill_ready_locked(
                        session_request_id=session_request_id,
                        layer_name=layer_name,
                    )
                self._session_manifest_ready_prefix_lens[session_request_id] = (
                    manifest.prefix_len
                )
                self._session_manifest_events.pop(session_request_id, None)
            else:
                self._session_manifest_events[session_request_id] = ready_event
            self._prefill_condition.notify_all()
        logger.info(
            "PAP Prefill KV manifest installed request_id=%s catalog_id=%s "
            "prefix_len=%d layers=%d blocks=%d",
            session_request_id,
            manifest.catalog_id,
            manifest.prefix_len,
            manifest.expected_layer_count,
            len(manifest.block_ids),
        )
        return manifest.prefix_len

    def register_prefill_kv(
        self, registration: PAPAttentionRegistration
    ) -> PAPAttentionSession:
        session = PAPAttentionSession(
            request_id=registration.request_id,
            conversation_id=registration.conversation_id,
            prefill_endpoint=registration.prefill_endpoint,
            kv_transfer_params=dict(registration.kv_transfer_params),
            prefix_len=registration.prefix_len,
            block_size=registration.block_size,
            max_seq_len=registration.max_seq_len,
            prefill_kv_handle=registration.request_id,
            q_size=registration.q_size,
            kv_size=registration.kv_size,
        )
        if self.resolve_session_request_id(registration.request_id) is not None:
            deferred_flushed = self._decode_token_committer.flush_request(
                registration.request_id,
                timeout_s=self._decode_token_flush_timeout_s,
            )
            if not deferred_flushed:
                logger.warning(
                    "PAP decode-token join flush timed out before replaced "
                    "session request_id=%s",
                    registration.request_id,
                )
        with self._lock:
            replaced_lease_id, replaced_prefill_endpoint = (
                self._replace_existing_session_locked(registration.request_id)
            )
            for alias, released_request_id in list(
                self._released_session_aliases.items()
            ):
                if released_request_id == registration.request_id:
                    self._released_session_aliases.pop(alias, None)
            session_epoch = self._next_session_epoch
            session.prefill_kv_handle = (
                f"{registration.request_id}@pap-session-{session_epoch}"
            )
            self._session_epochs[registration.request_id] = session_epoch
            self._next_session_epoch += 1
            self._sessions[registration.request_id] = session
            self._prefill_readiness.setdefault(registration.request_id, {})
            self._request_id_resolution_cache[registration.request_id] = (
                registration.request_id
            )
            self._request_id_resolution_cache[session.prefill_kv_handle] = (
                registration.request_id
            )
        if replaced_lease_id is not None:
            commit_client = _get_commit_client()
            flush_submitted = getattr(
                commit_client,
                "flush_submitted_request",
                commit_client.flush_request,
            )
            commits_submitted = flush_submitted(registration.request_id)
            if not commits_submitted:
                logger.warning(
                    "PAP decode commit submission timed out before replaced "
                    "lease release request_id=%s",
                    registration.request_id,
                )
            if commits_submitted:
                commit_client.forget_request(registration.request_id)
            if commits_submitted:
                release_endpoint = (
                    None
                    if replaced_prefill_endpoint is None
                    else _prefill_control_endpoint(
                        replaced_prefill_endpoint,
                        _LEASE_RELEASE_PATH,
                    )
                )
                _get_lease_release_client().release(
                    request_id=registration.request_id,
                    lease_id=replaced_lease_id,
                    endpoint=release_endpoint,
                )
            self._decode_token_committer.forget_request(registration.request_id)
        logger.info(
            "registered PAP attention session request_id=%s "
            "conversation_id=%s kv_keys=%s",
            registration.request_id,
            registration.conversation_id,
            sorted(registration.kv_transfer_params.keys()),
        )
        return session.copy()

    def get_session(self, request_id: str) -> PAPAttentionSession | None:
        with self._lock:
            session = self._sessions.get(request_id)
            return None if session is None else session.copy()

    def active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def resolve_session_request_id(self, request_id: str) -> str | None:
        """Map vLLM-wrapped request ids back to the proxy-level PAP id."""
        with self._lock:
            return self._resolve_session_request_id_locked(request_id)

    def _drop_offload_exec_session_entry_cache_locked(
        self, session_request_id: str
    ) -> None:
        for cache_key in list(self._offload_exec_session_entry_cache):
            if cache_key[0] == session_request_id:
                self._offload_exec_session_entry_cache.pop(cache_key, None)

    def _drop_attention_step_contexts_locked(self, session_request_id: str) -> None:
        last_context = self._last_attention_step_context
        if (
            last_context is not None
            and session_request_id in last_context.session_request_ids
        ):
            self._last_attention_step_context = None
        for cache_key, context in list(self._attention_step_context_cache.items()):
            if session_request_id in context.session_request_ids:
                self._attention_step_context_cache.pop(cache_key, None)

    def _resolve_session_request_id_locked(self, request_id: str) -> str | None:
        cached = self._request_id_resolution_cache.get(request_id)
        if cached is not None:
            if cached in self._sessions:
                return cached
            self._request_id_resolution_cache.pop(request_id, None)

        if request_id in self._sessions:
            self._request_id_resolution_cache[request_id] = request_id
            return request_id

        candidates = [request_id]
        for prefix in ("cmpl-", "chatcmpl-"):
            if request_id.startswith(prefix):
                candidates.append(request_id[len(prefix) :])

        for candidate in candidates:
            if candidate in self._sessions:
                self._request_id_resolution_cache[request_id] = candidate
                return candidate
            for session_request_id in self._sessions:
                if candidate.startswith(
                    f"{session_request_id}-"
                ) or candidate.startswith(f"{session_request_id}_"):
                    self._request_id_resolution_cache[request_id] = session_request_id
                    return session_request_id
        return None

    def _prefill_readiness_locked(
        self,
        *,
        session_request_id: str,
        layer_name: str,
    ) -> PAPPrefillLayerReadiness:
        return self._prefill_readiness.setdefault(session_request_id, {}).setdefault(
            layer_name,
            PAPPrefillLayerReadiness(
                request_id=session_request_id,
                layer_name=layer_name,
            ),
        )

    def _mark_prefill_ready_locked(
        self,
        *,
        session_request_id: str,
        layer_name: str,
    ) -> PAPPrefillLayerReadiness:
        readiness = self._mark_prefill_opened_locked(
            session_request_id=session_request_id,
            layer_name=layer_name,
        )
        readiness.ready = True
        readiness.ready_at = time.perf_counter()
        self._prefill_condition.notify_all()
        return readiness

    def _mark_prefill_opened_locked(
        self,
        *,
        session_request_id: str,
        layer_name: str,
    ) -> PAPPrefillLayerReadiness:
        readiness = self._prefill_readiness_locked(
            session_request_id=session_request_id,
            layer_name=layer_name,
        )
        readiness.descriptor_received = True
        readiness.descriptor_opened = True
        readiness.ready = False
        readiness.failed = False
        readiness.error = ""
        if readiness.opened_at == 0.0:
            readiness.opened_at = time.perf_counter()
        readiness.ready_at = 0.0
        readiness.failed_at = 0.0
        self._prefill_condition.notify_all()
        return readiness

    def _mark_prefill_failed_locked(
        self,
        *,
        session_request_id: str,
        layer_name: str,
        error: str,
    ) -> PAPPrefillLayerReadiness:
        readiness = self._prefill_readiness_locked(
            session_request_id=session_request_id,
            layer_name=layer_name,
        )
        readiness.ready = False
        readiness.failed = True
        readiness.error = error
        readiness.ready_at = 0.0
        readiness.failed_at = time.perf_counter()
        self._prefill_condition.notify_all()
        return readiness

    def get_prefill_readiness_snapshot(
        self,
        request_id: str,
        *,
        include_layers: bool = True,
    ) -> dict[str, Any]:
        """Advance and return one session's non-blocking KV-ready state."""
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                return {
                    "session_handle": None,
                    "manifest_prefix_len": None,
                    "ready_prefix_len": None,
                    "ready": False,
                    "failed": False,
                    "error": "",
                    "layers": [],
                }
            event = self._session_manifest_events.get(session_request_id)

        event_ready = False
        event_error = ""
        if event is not None:
            try:
                event_ready = bool(event.query())
            except Exception as exc:
                event_error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                return {
                    "session_handle": None,
                    "manifest_prefix_len": None,
                    "ready_prefix_len": None,
                    "ready": False,
                    "failed": False,
                    "error": "",
                    "layers": [],
                }
            current_event = self._session_manifest_events.get(session_request_id)
            if event is not None and current_event is event:
                readiness_by_layer = self._prefill_readiness.get(
                    session_request_id,
                    {},
                )
                if event_error:
                    for layer_name in readiness_by_layer:
                        self._mark_prefill_failed_locked(
                            session_request_id=session_request_id,
                            layer_name=layer_name,
                            error=event_error,
                        )
                    self._session_manifest_events.pop(session_request_id, None)
                elif event_ready:
                    for layer_name in readiness_by_layer:
                        self._mark_prefill_ready_locked(
                            session_request_id=session_request_id,
                            layer_name=layer_name,
                        )
                    prefix_len = self._session_manifest_prefix_lens.get(
                        session_request_id
                    )
                    if prefix_len is not None:
                        self._session_manifest_ready_prefix_lens[session_request_id] = (
                            prefix_len
                        )
                    self._session_manifest_events.pop(session_request_id, None)

            session = self._sessions[session_request_id]
            readiness_by_layer = self._prefill_readiness.get(session_request_id, {})
            failed_layers = [
                readiness
                for readiness in readiness_by_layer.values()
                if readiness.failed
            ]
            return {
                "session_handle": session.prefill_kv_handle,
                "manifest_prefix_len": self._session_manifest_prefix_lens.get(
                    session_request_id
                ),
                "ready_prefix_len": self._session_manifest_ready_prefix_lens.get(
                    session_request_id
                ),
                "ready": bool(readiness_by_layer)
                and all(readiness.ready for readiness in readiness_by_layer.values()),
                "failed": bool(failed_layers),
                "error": failed_layers[0].error if failed_layers else "",
                "layers": (
                    [
                        readiness.copy()
                        for _layer_name, readiness in sorted(readiness_by_layer.items())
                    ]
                    if include_layers
                    else []
                ),
            }

    def get_prefill_readiness(self, request_id: str) -> list[PAPPrefillLayerReadiness]:
        snapshot = self.get_prefill_readiness_snapshot(request_id)
        return list(snapshot["layers"])

    def size(self) -> int:
        with self._lock:
            return len(self._sessions)
