# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP token, commit, lease, and request-drain lifecycle."""

from vllm.pap.lifecycle.commit import DecodeCommitClient
from vllm.pap.lifecycle.decode_token import (
    DeferredDecodeCommit,
    DeferredDecodeTokenCommitter,
)
from vllm.pap.lifecycle.decode_token_client import DecodeTokenClient
from vllm.pap.lifecycle.lease import (
    PAPKVLeaseEntry,
    PAPKVLeaseRegistry,
    get_global_kv_lease_registry,
    pap_active_lease_id,
    pap_has_active_lease,
    pap_kv_seq_len,
    pap_leased_block_ids,
    pap_pin_blocks,
    pap_record_kv_seq_len,
    pap_refresh_lease,
    pap_release_lease,
    pap_stash_deferred_blocks,
    pap_sweep_expired_leases,
    pap_update_kv_seq_len,
    pap_was_recently_released,
    reset_global_kv_lease_registry,
)
from vllm.pap.lifecycle.lease_release import LeaseReleaseClient

__all__ = [
    "DecodeCommitClient",
    "DecodeTokenClient",
    "DeferredDecodeCommit",
    "DeferredDecodeTokenCommitter",
    "LeaseReleaseClient",
    "PAPKVLeaseEntry",
    "PAPKVLeaseRegistry",
    "get_global_kv_lease_registry",
    "pap_active_lease_id",
    "pap_has_active_lease",
    "pap_kv_seq_len",
    "pap_leased_block_ids",
    "pap_pin_blocks",
    "pap_record_kv_seq_len",
    "pap_refresh_lease",
    "pap_release_lease",
    "pap_stash_deferred_blocks",
    "pap_sweep_expired_leases",
    "pap_update_kv_seq_len",
    "pap_was_recently_released",
    "reset_global_kv_lease_registry",
]
