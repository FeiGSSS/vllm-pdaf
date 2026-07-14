# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP token, commit, lease, and request-drain lifecycle."""

from vllm.pap.lifecycle.commit import DecodeCommitClient
from vllm.pap.lifecycle.decode_token import (
    DeferredDecodeCommit,
    DeferredDecodeTokenCommitter,
)
from vllm.pap.lifecycle.lease import (
    PAPKVLeaseEntry,
    PAPKVLeaseRegistry,
    get_global_kv_lease_registry,
    pap_active_lease_id,
    pap_has_active_lease,
    pap_leased_block_ids,
    pap_pin_blocks,
    pap_pop_deferred_blocks,
    pap_refresh_lease,
    pap_release_lease,
    pap_stash_deferred_blocks,
    pap_sweep_expired_leases,
    reset_global_kv_lease_registry,
)

__all__ = [
    "DecodeCommitClient",
    "DeferredDecodeCommit",
    "DeferredDecodeTokenCommitter",
    "PAPKVLeaseEntry",
    "PAPKVLeaseRegistry",
    "get_global_kv_lease_registry",
    "pap_active_lease_id",
    "pap_has_active_lease",
    "pap_leased_block_ids",
    "pap_pin_blocks",
    "pap_pop_deferred_blocks",
    "pap_refresh_lease",
    "pap_release_lease",
    "pap_stash_deferred_blocks",
    "pap_sweep_expired_leases",
    "reset_global_kv_lease_registry",
]
