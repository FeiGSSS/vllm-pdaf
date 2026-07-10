"""PAP KV block lease registry.

Tracks Prefill-side KV blocks that have been exported to remote Attention
consumers and must not be returned to the block pool until the lease is
released. This is a process-local registry; PAP proxy release signals are
translated into release_lease calls via the Prefill HTTP endpoint.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace

logger = logging.getLogger(__name__)


def _kv_lease_profile_enabled() -> bool:
    return os.environ.get("PAP_KV_LEASE_PROFILE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _kv_lease_default_ttl_seconds() -> float:
    raw = os.environ.get("PAP_KV_LEASE_TTL_SECONDS", "300")
    if not raw:
        return 300.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("Invalid PAP_KV_LEASE_TTL_SECONDS=%r; using 300 seconds", raw)
        return 300.0


@dataclass(frozen=True)
class PAPKVLeaseEntry:
    lease_id: str
    request_id: str
    block_ids: tuple[int, ...]
    created_at: float
    expires_at: float | None
    released_at: float | None = None


@dataclass
class PAPKVLeaseRegistry:
    """Process-local PAP KV lease registry.

    Notes:
    - Leases are keyed by `lease_id`; a request may have multiple leases over
      time (e.g., re-export) but only one active lease at a time.
    - Active leases guard block IDs from being returned to the block pool by
      vLLM scheduler free path.
    - Deferred block objects (with optional release callback) can be stashed
      on the registry so scheduler finish can fully bypass block_pool free.
    - Expired leases are freed when the scheduler calls sweep_expired_leases.
    """

    _lock: threading.RLock = field(default_factory=threading.RLock)
    _by_lease: dict[str, PAPKVLeaseEntry] = field(default_factory=dict)
    _active_by_request: dict[str, str] = field(default_factory=dict)
    _ttl_seconds: float = field(default_factory=_kv_lease_default_ttl_seconds)
    _deferred_blocks: dict[str, tuple[object, Callable[[object], None] | None]] = field(
        default_factory=dict
    )

    def pin_blocks(
        self,
        *,
        request_id: str,
        block_ids: Sequence[int],
        lease_id: str | None = None,
        ttl_seconds: float | None = None,
    ) -> str:
        """Pin blocks under a new lease for `request_id`.

        If an active lease exists for the request, it is replaced by the new
        lease (the previous lease's blocks are dropped from this registry
        without being implicitly returned; the caller is responsible for the
        prior blocks).
        """
        lease_id = lease_id or f"pap-lease-{uuid.uuid4().hex[:16]}"
        block_tuple = tuple(int(b) for b in block_ids)
        now = time.time()
        ttl = self._ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        expires_at = (now + ttl) if ttl > 0 else None
        entry = PAPKVLeaseEntry(
            lease_id=lease_id,
            request_id=str(request_id),
            block_ids=block_tuple,
            created_at=now,
            expires_at=expires_at,
        )
        with self._lock:
            self._by_lease[lease_id] = entry
            self._active_by_request[str(request_id)] = lease_id
        if _kv_lease_profile_enabled():
            logger.info(
                "PAP KV lease pin request_id=%s lease_id=%s blocks=%d ttl=%s",
                request_id,
                lease_id,
                len(block_tuple),
                expires_at,
            )
        return lease_id

    def release_lease(self, lease_id: str) -> tuple[int, ...]:
        """Release a lease and return the block IDs that it guarded.

        If deferred block objects were stashed for this lease, their release
        callback is invoked inside the lock to serialize against pin/release.
        """
        with self._lock:
            entry = self._by_lease.pop(lease_id, None)
            if entry is None:
                if _kv_lease_profile_enabled():
                    logger.info("PAP KV lease release miss lease_id=%s", lease_id)
                return ()
            active = self._active_by_request.get(entry.request_id)
            if active == lease_id:
                self._active_by_request.pop(entry.request_id, None)
            deferred = self._deferred_blocks.pop(lease_id, None)
            released_at = time.time()
            entry = PAPKVLeaseEntry(
                lease_id=entry.lease_id,
                request_id=entry.request_id,
                block_ids=entry.block_ids,
                created_at=entry.created_at,
                expires_at=entry.expires_at,
                released_at=released_at,
            )
            self._by_lease[lease_id] = entry
        if deferred is not None:
            blocks, cb = deferred
            if cb is not None:
                try:
                    cb(blocks)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "PAP KV lease deferred free callback failed lease_id=%s",
                        lease_id,
                    )
        if _kv_lease_profile_enabled():
            logger.info(
                "PAP KV lease release lease_id=%s request_id=%s blocks=%d deferred=%s",
                lease_id,
                entry.request_id,
                len(entry.block_ids),
                deferred is not None,
            )
        return entry.block_ids

    def stash_deferred_blocks(
        self,
        *,
        lease_id: str,
        blocks: object,
        free_callback: Callable[[object], None] | None,
    ) -> None:
        """Stash block objects (and optional free callback) for a lease.

        Used by scheduler finish hook so PAP-leased requests can fully bypass
        block_pool.free_blocks until release_lease() invokes free_callback.
        """
        with self._lock:
            if lease_id not in self._by_lease:
                raise KeyError(f"unknown lease_id: {lease_id}")
            self._deferred_blocks[lease_id] = (blocks, free_callback)
        if _kv_lease_profile_enabled():
            logger.info("PAP KV lease stash deferred lease_id=%s", lease_id)

    def pop_deferred_blocks(
        self, lease_id: str
    ) -> tuple[object, Callable[[object], None] | None] | None:
        with self._lock:
            return self._deferred_blocks.pop(lease_id, None)

    def has_active_lease(self, request_id: str) -> bool:
        with self._lock:
            lease_id = self._active_by_request.get(str(request_id))
            if lease_id is None:
                return False
            entry = self._by_lease.get(lease_id)
            if entry is None or entry.released_at is not None:
                return False
            return entry.expires_at is None or time.time() <= entry.expires_at

    def leased_block_ids(self, request_id: str) -> tuple[int, ...]:
        with self._lock:
            lease_id = self._active_by_request.get(str(request_id))
            if lease_id is None:
                return ()
            entry = self._by_lease.get(lease_id)
            if entry is None:
                return ()
            if entry.released_at is not None:
                return ()
            if entry.expires_at is not None and time.time() > entry.expires_at:
                return ()
            return entry.block_ids

    def active_lease_id(self, request_id: str) -> str | None:
        with self._lock:
            lease_id = self._active_by_request.get(str(request_id))
            if lease_id is None:
                return None
            entry = self._by_lease.get(lease_id)
            if entry is None or entry.released_at is not None:
                return None
            if entry.expires_at is not None and time.time() > entry.expires_at:
                return None
            return lease_id

    def refresh_lease(self, request_id: str) -> bool:
        """Extend the active lease after an acknowledged decode commit."""
        with self._lock:
            lease_id = self._active_by_request.get(str(request_id))
            if lease_id is None:
                return False
            entry = self._by_lease.get(lease_id)
            if entry is None or entry.released_at is not None:
                return False
            if self._ttl_seconds <= 0:
                return True
            self._by_lease[lease_id] = replace(
                entry,
                expires_at=time.time() + self._ttl_seconds,
            )
            return True

    def sweep_expired_leases(self) -> list[str]:
        """Drop expired leases from active map and free their deferred blocks.

        Returns released lease IDs. Expiry triggers a full release_lease(),
        which invokes the per-lease free_callback (returning blocks to the
        block pool) if one was registered via stash_deferred_blocks.
        """
        swept: list[str] = []
        now = time.time()
        with self._lock:
            for lease_id, entry in list(self._by_lease.items()):
                if entry.released_at is not None:
                    continue
                if entry.expires_at is not None and now > entry.expires_at:
                    active = self._active_by_request.get(entry.request_id)
                    if active == lease_id:
                        self._active_by_request.pop(entry.request_id, None)
                    swept.append(lease_id)
        for lease_id in swept:
            self.release_lease(lease_id)
        if swept and _kv_lease_profile_enabled():
            logger.info(
                "PAP KV lease sweep_expired count=%d lease_ids=%s",
                len(swept),
                swept,
            )
        return swept


_GLOBAL_REGISTRY: PAPKVLeaseRegistry | None = None
_GLOBAL_LOCK = threading.Lock()


def get_global_kv_lease_registry() -> PAPKVLeaseRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_REGISTRY is None:
                _GLOBAL_REGISTRY = PAPKVLeaseRegistry()
    return _GLOBAL_REGISTRY


def reset_global_kv_lease_registry() -> None:
    """Reset the process-global registry. Intended for tests/in-process resets."""
    global _GLOBAL_REGISTRY
    with _GLOBAL_LOCK:
        _GLOBAL_REGISTRY = None


def pap_pin_blocks(
    request_id: str, block_ids: Iterable[int], lease_id: str | None = None
) -> str:
    return get_global_kv_lease_registry().pin_blocks(
        request_id=request_id, block_ids=tuple(block_ids), lease_id=lease_id
    )


def pap_release_lease(lease_id: str) -> tuple[int, ...]:
    return get_global_kv_lease_registry().release_lease(lease_id)


def pap_has_active_lease(request_id: str) -> bool:
    return get_global_kv_lease_registry().has_active_lease(request_id)


def pap_leased_block_ids(request_id: str) -> tuple[int, ...]:
    return get_global_kv_lease_registry().leased_block_ids(request_id)


def pap_active_lease_id(request_id: str) -> str | None:
    return get_global_kv_lease_registry().active_lease_id(request_id)


def pap_refresh_lease(request_id: str) -> bool:
    return get_global_kv_lease_registry().refresh_lease(request_id)


def pap_stash_deferred_blocks(
    lease_id: str,
    blocks: object,
    free_callback: Callable[[object], None] | None,
) -> None:
    get_global_kv_lease_registry().stash_deferred_blocks(
        lease_id=lease_id, blocks=blocks, free_callback=free_callback
    )


def pap_pop_deferred_blocks(
    lease_id: str,
) -> tuple[object, Callable[[object], None] | None] | None:
    return get_global_kv_lease_registry().pop_deferred_blocks(lease_id)


def pap_sweep_expired_leases() -> list[str]:
    return get_global_kv_lease_registry().sweep_expired_leases()
