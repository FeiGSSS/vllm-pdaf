# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed runtime configuration for PAP services.

Environment variables are parsed at a service composition root and the
resulting immutable configuration is passed to PAP runtime components. Legacy
modules may continue to read their existing variables during the convergence
phase, but new configuration reads belong here.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar


class PAPConfigError(ValueError):
    """Raised when PAP runtime configuration is invalid."""


class PAPOffloadExecTransport(str, Enum):
    """Projection-to-Attention execution transport."""

    NIXL_MAILBOX = "nixl_mailbox"
    LOCAL_FAST = "local_fast"


class PAPOffloadKVTransport(str, Enum):
    """Prefill-to-Attention KV transport."""

    CUDA_IPC = "cuda_ipc"
    NIXL_MAILBOX = "nixl_mailbox"


class PAPAttentionDispatchMode(str, Enum):
    """Legacy Attention execution selection retained during convergence."""

    LEGACY = "legacy"
    CENTRAL_FIFO = "central_fifo"
    CENTRAL_COMBINE = "central_combine"


class PAPRoutingPolicy(str, Enum):
    """Supported PA and Projection routing policies."""

    ROUND_ROBIN = "round_robin"
    CROSSBAR_ROUND_ROBIN = "crossbar_round_robin"
    PROJECTION_AFFINITY = "projection_affinity"
    PROJECTION_STICKY = "projection_sticky"


class PAPKVHandoffMode(str, Enum):
    """Prefill KV handoff modes retained during convergence."""

    LAYER_DESCRIPTOR = "layer_descriptor"
    SEALED_MANIFEST = "sealed_manifest"


class PAPMPSMode(str, Enum):
    """PAP MPS resource partition modes."""

    DISABLED = "disabled"
    DYNAMIC = "dynamic"
    STATIC = "static"


_TOPOLOGY_PATTERN = re.compile(r"^(?P<pa>[1-9][0-9]*)pa(?P<p>[1-9][0-9]*)p$")
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})
_EnumT = TypeVar("_EnumT", bound=Enum)


@dataclass(frozen=True)
class PAPTopology:
    """Arbitrary positive xPAyP topology and tensor-parallel width."""

    pa_count: int
    projection_count: int
    tensor_parallel_size: int = 1

    def __post_init__(self) -> None:
        if self.pa_count < 1 or self.projection_count < 1:
            raise PAPConfigError("PAP topology counts must be positive")
        if self.tensor_parallel_size < 1:
            raise PAPConfigError("PAP tensor parallel size must be positive")

    @property
    def name(self) -> str:
        """Return the canonical xPAyP topology name."""
        return f"{self.pa_count}pa{self.projection_count}p"

    @property
    def prefill_device_count(self) -> int:
        """Return the number of Prefill devices required by the topology."""
        return self.pa_count * self.tensor_parallel_size

    @property
    def projection_device_count(self) -> int:
        """Return the number of Projection devices required by the topology."""
        return self.projection_count * self.tensor_parallel_size

    @classmethod
    def parse(cls, value: str, *, tensor_parallel_size: int = 1) -> PAPTopology:
        """Parse an xPAyP topology name.

        Args:
            value: Topology name such as ``1pa1p`` or ``6pa2p``.
            tensor_parallel_size: Tensor-parallel width for every group.

        Returns:
            Parsed topology.

        Raises:
            PAPConfigError: If the name or tensor-parallel width is invalid.
        """
        normalized = str(value).strip().lower()
        match = _TOPOLOGY_PATTERN.fullmatch(normalized)
        if match is None:
            raise PAPConfigError(
                "PAP_TOPOLOGY must use positive xPAyP syntax, "
                f"got {value!r}"
            )
        return cls(
            pa_count=int(match.group("pa")),
            projection_count=int(match.group("p")),
            tensor_parallel_size=int(tensor_parallel_size),
        )


@dataclass(frozen=True)
class PAPPlacement:
    """Global device placement for PAP roles."""

    prefill_devices: tuple[int, ...]
    attention_devices: tuple[int, ...]
    projection_devices: tuple[int, ...]


@dataclass(frozen=True)
class PAPMPSConfig:
    """MPS partition requested by a PAP launcher."""

    mode: PAPMPSMode
    profile_id: str
    prefill_requested_percent: int
    attention_requested_percent: int
    prefill_chunks: int
    attention_chunks: int
    prefill_visible_sms: int
    attention_visible_sms: int

    def __post_init__(self) -> None:
        for role, percent in (
            ("Prefill", self.prefill_requested_percent),
            ("Attention", self.attention_requested_percent),
        ):
            if not 1 <= percent <= 100:
                raise PAPConfigError(
                    f"PAP {role} MPS percentage must be between 1 and 100"
                )
        if self.prefill_requested_percent + self.attention_requested_percent > 100:
            raise PAPConfigError("PAP MPS requested percentages must not exceed 100")
        if self.prefill_chunks < 1 or self.attention_chunks < 1:
            raise PAPConfigError("PAP static MPS chunk counts must be positive")
        if self.prefill_visible_sms < 1 or self.attention_visible_sms < 1:
            raise PAPConfigError("PAP expected visible SM counts must be positive")

    @property
    def total_visible_sms(self) -> int:
        """Return the total visible SMs in the static partition."""
        return self.prefill_visible_sms + self.attention_visible_sms


@dataclass(frozen=True)
class PAPDecodeCommitConfig:
    """Decode-commit delivery and acknowledgement policy."""

    endpoint: str
    fail_closed: bool
    timeout_s: float
    queue_size: int
    max_attempts: int
    retry_initial_s: float
    retry_max_s: float
    flush_timeout_s: float


@dataclass(frozen=True)
class PAPDecodeTokenConfig:
    """Projection-to-Attention sampled-token delivery policy."""

    timeout_s: float
    queue_size: int
    max_attempts: int
    retry_initial_s: float
    retry_max_s: float
    flush_timeout_s: float


@dataclass(frozen=True)
class PAPLeaseReleaseConfig:
    """Lease-release delivery policy."""

    endpoint: str
    timeout_s: float
    max_attempts: int
    retry_initial_s: float
    retry_max_s: float
    lease_ttl_s: float


@dataclass(frozen=True)
class PAPAttentionServiceConfig:
    """Attention service composition and retained dispatcher settings."""

    dispatch_mode: PAPAttentionDispatchMode
    dispatch_queue_size: int
    combine_wait_us: float
    active_peer_tracking: bool
    actor_id: str
    local_rank: int
    storage_device: str | None
    prefill_wait_timeout_s: float
    decode_kv_initial_capacity: int


@dataclass(frozen=True)
class PAPRuntimeFeatures:
    """Feature-selectable PAP paths retained until Phase 2 convergence."""

    kv_handoff_mode: PAPKVHandoffMode
    unified_kv: bool
    batched_route_copy: bool
    unified_md_fast_key: bool
    direct_mailbox_output: bool
    local_fast_stream_ordered: bool
    local_fast_slot_count: int
    decode_slot_plan_cache_limit: int
    prefill_ipc_profile: bool
    prefill_torch_profile: bool


@dataclass(frozen=True)
class PAPRetiredFlag:
    """A selectable path scheduled for or already past removal."""

    name: str
    p17_values: frozenset[str]
    replacement: str
    experiment_id: str
    removed: bool = False


@dataclass(frozen=True)
class PAPRetiredFlagSetting:
    """One explicitly configured flag scheduled for retirement."""

    spec: PAPRetiredFlag
    value: str

    @property
    def matches_p17(self) -> bool:
        """Whether this setting selects the frozen P17 behavior."""
        return _normalize_flag_value(self.value) in self.spec.p17_values


PAP_RETIRED_FLAGS = (
    PAPRetiredFlag(
        name="PAP_ASYNC_DECODE_TOKEN",
        p17_values=_TRUE_VALUES,
        replacement="unconditional asynchronous sampled-token delivery",
        experiment_id="PAP-20260713-ASYNC-DECODE-TOKEN-D2H",
        removed=True,
    ),
    PAPRetiredFlag(
        name="PAP_ASYNC_DECODE_TOKEN_SYNC_ONLY_BARRIER",
        p17_values=_FALSE_VALUES,
        replacement="no sampled-token timing barrier",
        experiment_id="PAP-20260714-ASYNC-TTFT-ROOTCAUSE",
        removed=True,
    ),
    PAPRetiredFlag(
        name="PAP_PROJECTION_SYNC_ONLY_BARRIER",
        p17_values=_FALSE_VALUES,
        replacement="no Projection timing barrier",
        experiment_id="PAP-20260714-ASYNC-TTFT-ROOTCAUSE",
        removed=True,
    ),
    PAPRetiredFlag(
        name="PAP_PREFILL_SYNC_ONLY_BARRIER",
        p17_values=_FALSE_VALUES,
        replacement="no Prefill timing barrier",
        experiment_id="PAP-20260714-ASYNC-TTFT-ROOTCAUSE",
        removed=True,
    ),
    PAPRetiredFlag(
        name="PAP_PREFILL_KV_ASYNC",
        p17_values=_TRUE_VALUES,
        replacement="unconditional safe asynchronous Prefill KV import",
        experiment_id="PAP-20260714-REGISTRY-LOCK-SAFE-ASYNC",
        removed=True,
    ),
    PAPRetiredFlag(
        name="PAP_KV_HANDOFF_MODE",
        p17_values=frozenset({"sealed_manifest"}),
        replacement="sealed catalog and request manifest handoff",
        experiment_id="PAP-20260714-SEAL-HANDOFF-KV",
    ),
    PAPRetiredFlag(
        name="PAP_UNIFIED_KV",
        p17_values=_TRUE_VALUES,
        replacement="Prefill-owned unified KV",
        experiment_id="PAP-20260703-UNIFIED-KV",
    ),
    PAPRetiredFlag(
        name="PAP_BATCHED_ROUTE_COPY",
        p17_values=_TRUE_VALUES,
        replacement="batched route copy with input-driven fallback",
        experiment_id="PAP-20260711-ROUTE-COPY",
    ),
    PAPRetiredFlag(
        name="PAP_UNIFIED_MD_FAST_KEY",
        p17_values=_TRUE_VALUES,
        replacement="unified metadata fast-key lookup",
        experiment_id="PAP-20260712-METADATA-FAST-KEY",
    ),
    PAPRetiredFlag(
        name="PAP_ATTENTION_DISPATCH_MODE",
        p17_values=frozenset({"legacy"}),
        replacement="topology-derived direct or combine execution",
        experiment_id="PAP-20260711-ATTENTION-COMBINE",
    ),
    PAPRetiredFlag(
        name="PAP_ATTENTION_ACTIVE_PEER_TRACKING",
        p17_values=_FALSE_VALUES,
        replacement="topology-derived peer membership tracking",
        experiment_id="PAP-20260711-ACTIVE-PEER",
    ),
    PAPRetiredFlag(
        name="PAP_MPS_MODE",
        p17_values=frozenset({"static"}),
        replacement="P17 static 64/28 MPS partition",
        experiment_id="PAP-20260714-ASYNC-STATIC-BASELINE",
    ),
    PAPRetiredFlag(
        name="PAP_PREFILL_IPC_PROFILE",
        p17_values=_FALSE_VALUES,
        replacement="non-blocking observability only",
        experiment_id="PAP-20260714-REGISTRY-LOCK-SAFE-ASYNC",
    ),
    PAPRetiredFlag(
        name="PAP_PREFILL_TORCH_PROFILE",
        p17_values=_FALSE_VALUES,
        replacement="non-blocking observability only",
        experiment_id="PAP-20260714-REGISTRY-LOCK-SAFE-ASYNC",
    ),
    PAPRetiredFlag(
        name="PAP_DIAG_R1_PROJECTION_GATE_COUNT",
        p17_values=frozenset({"0"}),
        replacement="no diagnostic Projection gate",
        experiment_id="PAP-20260714-ASYNC-TTFT-STRICT-ISOLATION",
        removed=True,
    ),
    PAPRetiredFlag(
        name="PAP_DIAG_R1_COMMIT_GATE_COUNT",
        p17_values=frozenset({"0"}),
        replacement="no diagnostic decode-commit gate",
        experiment_id="PAP-20260714-ASYNC-TTFT-STRICT-ISOLATION",
        removed=True,
    ),
    PAPRetiredFlag(
        name="PAP_DIAG_DECODE_COMMIT_GATE_FILE",
        p17_values=_FALSE_VALUES,
        replacement="unconditional decode-commit delivery",
        experiment_id="PAP-20260714-ASYNC-TTFT-STRICT-ISOLATION",
        removed=True,
    ),
    PAPRetiredFlag(
        name="PAP_DIAG_DECODE_COMMIT_GATE_TIMEOUT",
        p17_values=frozenset({"120"}),
        replacement="the normal decode-commit timeout and retry policy",
        experiment_id="PAP-20260714-ASYNC-TTFT-STRICT-ISOLATION",
        removed=True,
    ),
)


@dataclass(frozen=True)
class PAPRuntimeConfig:
    """Immutable process configuration shared by PAP runtime components."""

    topology: PAPTopology
    placement: PAPPlacement
    routing_policy: PAPRoutingPolicy
    offload_exec_transport: PAPOffloadExecTransport
    offload_kv_transport: PAPOffloadKVTransport
    same_host: bool
    mps: PAPMPSConfig
    features: PAPRuntimeFeatures
    attention: PAPAttentionServiceConfig
    decode_commit: PAPDecodeCommitConfig
    decode_token: PAPDecodeTokenConfig
    lease_release: PAPLeaseReleaseConfig
    protocol_version: int

    def __post_init__(self) -> None:
        required_prefill = self.topology.prefill_device_count
        required_projection = self.topology.projection_device_count
        if len(self.placement.prefill_devices) < required_prefill:
            raise PAPConfigError(
                "PAP_PREFILL_GPUS has fewer devices than the topology requires"
            )
        if len(self.placement.attention_devices) < required_prefill:
            raise PAPConfigError(
                "PAP_ATTENTION_GPUS has fewer devices than the topology requires"
            )
        if len(self.placement.projection_devices) < required_projection:
            raise PAPConfigError(
                "PAP_PROJECTION_GPUS has fewer devices than the topology requires"
            )
        if (
            self.offload_exec_transport is PAPOffloadExecTransport.LOCAL_FAST
            and not self.same_host
        ):
            raise PAPConfigError("PAP local_fast transport requires same-host peers")
        if self.protocol_version < 1:
            raise PAPConfigError("PAP protocol version must be positive")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> PAPRuntimeConfig:
        """Parse a PAP configuration from an environment mapping.

        Args:
            environ: Environment mapping. Defaults to ``os.environ``.

        Returns:
            Fully parsed immutable runtime configuration.

        Raises:
            PAPConfigError: If any configured value is invalid.
        """
        env = os.environ if environ is None else environ
        reject_removed_pap_flags(env)
        tp_size = _env_int(env, "PAP_TP_SIZE", 1, minimum=1)
        topology_name = _env_text(
            env,
            "PAP_TOPOLOGY",
            _env_text(env, "TOPOLOGY", "1pa1p"),
        )
        topology = PAPTopology.parse(
            topology_name,
            tensor_parallel_size=tp_size,
        )
        _validate_topology_count(env, "PAP_PA_COUNT", topology.pa_count)
        _validate_topology_count(
            env,
            "PAP_PROJECTION_COUNT",
            topology.projection_count,
        )

        default_prefill = tuple(range(topology.prefill_device_count))
        default_projection = tuple(
            range(
                topology.prefill_device_count,
                topology.prefill_device_count + topology.projection_device_count,
            )
        )
        prefill_devices = _env_devices(
            env,
            "PAP_PREFILL_GPUS",
            default_prefill,
        )
        placement = PAPPlacement(
            prefill_devices=prefill_devices,
            attention_devices=_env_devices(
                env,
                "PAP_ATTENTION_GPUS",
                prefill_devices,
            ),
            projection_devices=_env_devices(
                env,
                "PAP_PROJECTION_GPUS",
                default_projection,
            ),
        )

        exec_transport = _parse_exec_transport(
            _env_text(env, "PAP_OFFLOAD_EXEC_TRANSPORT", "nixl_mailbox")
        )
        kv_transport = _parse_kv_transport(
            _env_text(env, "PAP_OFFLOAD_KV_TRANSPORT", "cuda_ipc")
        )
        same_host = _env_bool(
            env,
            "PAP_SAME_HOST",
            exec_transport is PAPOffloadExecTransport.LOCAL_FAST,
        )

        mps_mode = _parse_enum(
            PAPMPSMode,
            _env_text(env, "PAP_MPS_MODE", "dynamic"),
            "PAP_MPS_MODE",
        )
        if not _env_bool(env, "PAP_ENABLE_MPS", True):
            mps_mode = PAPMPSMode.DISABLED
        mps = PAPMPSConfig(
            mode=mps_mode,
            profile_id=_env_text(
                env,
                "PAP_BENCH_MPS_PROFILE",
                "baseline_70_30",
            ),
            prefill_requested_percent=_env_int(
                env,
                "PAP_PREFILL_MPS_PERCENT",
                70,
            ),
            attention_requested_percent=_env_int(
                env,
                "PAP_ATTENTION_MPS_PERCENT",
                30,
            ),
            prefill_chunks=_env_int(
                env,
                "PAP_STATIC_PREFILL_CHUNKS",
                16,
                minimum=1,
            ),
            attention_chunks=_env_int(
                env,
                "PAP_STATIC_ATTENTION_CHUNKS",
                7,
                minimum=1,
            ),
            prefill_visible_sms=_env_int(
                env,
                "PAP_STATIC_PREFILL_EXPECTED_SMS",
                64,
                minimum=1,
            ),
            attention_visible_sms=_env_int(
                env,
                "PAP_STATIC_ATTENTION_EXPECTED_SMS",
                28,
                minimum=1,
            ),
        )

        features = PAPRuntimeFeatures(
            kv_handoff_mode=_parse_enum(
                PAPKVHandoffMode,
                _env_text(env, "PAP_KV_HANDOFF_MODE", "layer_descriptor"),
                "PAP_KV_HANDOFF_MODE",
            ),
            unified_kv=_env_bool(env, "PAP_UNIFIED_KV", False),
            batched_route_copy=_env_bool(env, "PAP_BATCHED_ROUTE_COPY", True),
            unified_md_fast_key=_env_bool(env, "PAP_UNIFIED_MD_FAST_KEY", True),
            direct_mailbox_output=_env_bool(
                env,
                "PAP_DIRECT_MAILBOX_OUTPUT",
                False,
            ),
            local_fast_stream_ordered=_env_bool(
                env,
                "PAP_LOCAL_FAST_STREAM_ORDERED",
                True,
            ),
            local_fast_slot_count=_env_int(
                env,
                "PAP_LOCAL_FAST_SLOT_COUNT",
                2,
                minimum=1,
            ),
            decode_slot_plan_cache_limit=_env_int(
                env,
                "PAP_DECODE_SLOT_PLAN_CACHE_LIMIT",
                256,
                minimum=0,
            ),
            prefill_ipc_profile=_env_bool(
                env,
                "PAP_PREFILL_IPC_PROFILE",
                False,
            ),
            prefill_torch_profile=_env_bool(
                env,
                "PAP_PREFILL_TORCH_PROFILE",
                False,
            ),
        )

        attention = PAPAttentionServiceConfig(
            dispatch_mode=_parse_enum(
                PAPAttentionDispatchMode,
                _env_text(env, "PAP_ATTENTION_DISPATCH_MODE", "legacy"),
                "PAP_ATTENTION_DISPATCH_MODE",
            ),
            dispatch_queue_size=_env_int(
                env,
                "PAP_ATTENTION_DISPATCH_QUEUE_SIZE",
                0,
                minimum=0,
            ),
            combine_wait_us=_env_float(
                env,
                "PAP_ATTENTION_COMBINE_WAIT_US",
                0.0,
                minimum=0.0,
            ),
            active_peer_tracking=_env_bool(
                env,
                "PAP_ATTENTION_ACTIVE_PEER_TRACKING",
                False,
            ),
            actor_id=_env_text(
                env,
                "PAP_NIXL_MAILBOX_ACTOR_ID",
                "attention",
            ),
            local_rank=_env_int(
                env,
                "PAP_OFFLOAD_EXEC_LOCAL_RANK",
                0,
                minimum=0,
            ),
            storage_device=_env_optional_text(env, "PAP_ATTENTION_STORAGE_DEVICE"),
            prefill_wait_timeout_s=_env_float(
                env,
                "PAP_ATTENTION_PREFILL_WAIT_TIMEOUT",
                5.0,
                minimum=0.0,
            ),
            decode_kv_initial_capacity=_env_int(
                env,
                "PAP_ATTENTION_DECODE_KV_INITIAL_CAPACITY",
                128,
                minimum=1,
            ),
        )

        decode_commit = PAPDecodeCommitConfig(
            endpoint=_env_text(env, "PAP_DECODE_COMMIT_ENDPOINT", ""),
            fail_closed=_env_bool(
                env,
                "PAP_DECODE_COMMIT_FAIL_CLOSED",
                False,
            ),
            timeout_s=_env_float(
                env,
                "PAP_DECODE_COMMIT_TIMEOUT",
                0.2,
                minimum=0.0,
            ),
            queue_size=_env_int(
                env,
                "PAP_DECODE_COMMIT_QUEUE_SIZE",
                1024,
                minimum=1,
            ),
            max_attempts=_env_int(
                env,
                "PAP_DECODE_COMMIT_MAX_ATTEMPTS",
                8,
                minimum=1,
            ),
            retry_initial_s=_env_float(
                env,
                "PAP_DECODE_COMMIT_RETRY_INITIAL_SECONDS",
                0.05,
                minimum=0.0,
            ),
            retry_max_s=_env_float(
                env,
                "PAP_DECODE_COMMIT_RETRY_MAX_SECONDS",
                0.5,
                minimum=0.0,
            ),
            flush_timeout_s=_env_float(
                env,
                "PAP_DECODE_COMMIT_FLUSH_TIMEOUT",
                5.0,
                minimum=0.0,
            ),
        )
        _validate_retry_range(
            "PAP decode commit",
            decode_commit.retry_initial_s,
            decode_commit.retry_max_s,
        )

        decode_token = PAPDecodeTokenConfig(
            timeout_s=_env_float(
                env,
                "PAP_DECODE_TOKEN_TIMEOUT",
                0.2,
                minimum=0.0,
            ),
            queue_size=_env_int(
                env,
                "PAP_DECODE_TOKEN_QUEUE_SIZE",
                1024,
                minimum=1,
            ),
            max_attempts=_env_int(
                env,
                "PAP_DECODE_TOKEN_MAX_ATTEMPTS",
                8,
                minimum=1,
            ),
            retry_initial_s=_env_float(
                env,
                "PAP_DECODE_TOKEN_RETRY_INITIAL_SECONDS",
                0.05,
                minimum=0.0,
            ),
            retry_max_s=_env_float(
                env,
                "PAP_DECODE_TOKEN_RETRY_MAX_SECONDS",
                0.5,
                minimum=0.0,
            ),
            flush_timeout_s=_env_float(
                env,
                "PAP_DECODE_TOKEN_FLUSH_TIMEOUT",
                5.0,
                minimum=0.0,
            ),
        )
        _validate_retry_range(
            "PAP decode token",
            decode_token.retry_initial_s,
            decode_token.retry_max_s,
        )

        lease_release = PAPLeaseReleaseConfig(
            endpoint=_env_text(env, "PAP_LEASE_RELEASE_ENDPOINT", ""),
            timeout_s=_env_float(
                env,
                "PAP_LEASE_RELEASE_TIMEOUT",
                5.0,
                minimum=0.0,
            ),
            max_attempts=_env_int(
                env,
                "PAP_LEASE_RELEASE_MAX_ATTEMPTS",
                5,
                minimum=1,
            ),
            retry_initial_s=_env_float(
                env,
                "PAP_LEASE_RELEASE_RETRY_INITIAL_SECONDS",
                0.05,
                minimum=0.0,
            ),
            retry_max_s=_env_float(
                env,
                "PAP_LEASE_RELEASE_RETRY_MAX_SECONDS",
                0.5,
                minimum=0.0,
            ),
            lease_ttl_s=_env_float(
                env,
                "PAP_KV_LEASE_TTL_SECONDS",
                300.0,
                minimum=0.0,
            ),
        )
        _validate_retry_range(
            "PAP lease release",
            lease_release.retry_initial_s,
            lease_release.retry_max_s,
        )

        return cls(
            topology=topology,
            placement=placement,
            routing_policy=_parse_enum(
                PAPRoutingPolicy,
                _env_text(env, "PAP_ROUTING_POLICY", "round_robin"),
                "PAP_ROUTING_POLICY",
            ),
            offload_exec_transport=exec_transport,
            offload_kv_transport=kv_transport,
            same_host=same_host,
            mps=mps,
            features=features,
            attention=attention,
            decode_commit=decode_commit,
            decode_token=decode_token,
            lease_release=lease_release,
            protocol_version=_env_int(
                env,
                "PAP_PROTOCOL_VERSION",
                1,
                minimum=1,
            ),
        )

    def configured_retired_flags(
        self,
        environ: Mapping[str, str],
    ) -> tuple[PAPRetiredFlagSetting, ...]:
        """Return explicitly configured flags in the retirement registry.

        Removed entries are rejected by :meth:`from_env`; this method remains
        available for inventory and migration reporting.
        """
        del self
        return tuple(
            PAPRetiredFlagSetting(spec=spec, value=str(environ[spec.name]))
            for spec in PAP_RETIRED_FLAGS
            if spec.name in environ
        )

    def p17_profile_contract(self) -> dict[str, object]:
        """Return config-owned sections in the canonical P17 profile."""
        features = self.features
        dispatch_mode = self.attention.dispatch_mode
        if (
            dispatch_mode is PAPAttentionDispatchMode.LEGACY
            and self.topology.projection_count == 1
        ):
            attention_execution = "topology_derived_direct"
        else:
            attention_execution = dispatch_mode.value
        return {
            "topology": {
                "name": self.topology.name,
                "pa_count": self.topology.pa_count,
                "projection_count": self.topology.projection_count,
                "routing_policy": self.routing_policy.value,
            },
            "placement": {
                "prefill_devices": list(self.placement.prefill_devices),
                "attention_devices": list(self.placement.attention_devices),
                "projection_devices": list(self.placement.projection_devices),
            },
            "transport": {
                "offload_exec": self.offload_exec_transport.value,
                "offload_kv": self.offload_kv_transport.value,
                "same_host": self.same_host,
            },
            "mps": {
                "mode": self.mps.mode.value,
                "profile_id": self.mps.profile_id,
                "prefill_requested_percent": self.mps.prefill_requested_percent,
                "attention_requested_percent": (
                    self.mps.attention_requested_percent
                ),
                "prefill_chunks": self.mps.prefill_chunks,
                "attention_chunks": self.mps.attention_chunks,
                "prefill_visible_sms": self.mps.prefill_visible_sms,
                "attention_visible_sms": self.mps.attention_visible_sms,
                "total_visible_sms": self.mps.total_visible_sms,
            },
            "runtime": {
                "decode_token_delivery": "async",
                "prefill_kv_import": "async",
                "kv_handoff": features.kv_handoff_mode.value,
                "kv_ownership": (
                    "prefill_owned_unified" if features.unified_kv else "legacy_split"
                ),
                "route_copy": (
                    "batched_with_input_fallback"
                    if features.batched_route_copy
                    else "per_row"
                ),
                "metadata_lookup": (
                    "fast_key" if features.unified_md_fast_key else "full_scan"
                ),
                "attention_execution": attention_execution,
                "direct_mailbox_output": features.direct_mailbox_output,
                "local_fast_stream_ordered": (
                    features.local_fast_stream_ordered
                ),
                "local_fast_slot_count": features.local_fast_slot_count,
                "decode_slot_plan_cache_limit": (
                    features.decode_slot_plan_cache_limit
                ),
                "phase0_flags": {
                    "pap_async_decode_token": True,
                    "pap_prefill_kv_async": True,
                    "pap_kv_handoff_mode": features.kv_handoff_mode.value,
                    "pap_unified_kv": features.unified_kv,
                    "pap_batched_route_copy": features.batched_route_copy,
                    "pap_unified_md_fast_key": features.unified_md_fast_key,
                    "pap_direct_mailbox_output": features.direct_mailbox_output,
                    "pap_attention_dispatch_mode": dispatch_mode.value,
                    "pap_projection_sync_only_barrier": False,
                    "pap_prefill_ipc_profile": features.prefill_ipc_profile,
                    "pap_prefill_torch_profile": features.prefill_torch_profile,
                    "pap_diagnostic_projection_gate_count": 0,
                    "pap_diagnostic_commit_gate_count": 0,
                },
            },
        }


def _normalize_flag_value(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def reject_removed_pap_flags(environ: Mapping[str, str]) -> None:
    """Reject PAP variables whose runtime control paths no longer exist.

    Args:
        environ: Environment mapping to validate.

    Raises:
        PAPConfigError: If a removed variable is explicitly present.
    """
    for spec in PAP_RETIRED_FLAGS:
        if spec.removed and spec.name in environ:
            raise PAPConfigError(
                f"{spec.name} was removed; use {spec.replacement}. "
                f"Historical evidence: {spec.experiment_id}."
            )


def _env_text(
    environ: Mapping[str, str],
    name: str,
    default: str,
) -> str:
    value = environ.get(name)
    return str(default) if value is None else str(value).strip()


def _env_optional_text(
    environ: Mapping[str, str],
    name: str,
) -> str | None:
    value = environ.get(name)
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _env_bool(
    environ: Mapping[str, str],
    name: str,
    default: bool,
) -> bool:
    value = environ.get(name)
    if value is None:
        return default
    normalized = _normalize_flag_value(value)
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise PAPConfigError(f"{name} must be a boolean, got {value!r}")


def _env_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    raw = environ.get(name)
    try:
        value = int(default if raw is None else raw)
    except (TypeError, ValueError) as exc:
        raise PAPConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise PAPConfigError(f"{name} must be at least {minimum}, got {value}")
    return value


def _env_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float | None = None,
) -> float:
    raw = environ.get(name)
    try:
        value = float(default if raw is None else raw)
    except (TypeError, ValueError) as exc:
        raise PAPConfigError(f"{name} must be a number, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise PAPConfigError(f"{name} must be at least {minimum}, got {value}")
    return value


def _env_devices(
    environ: Mapping[str, str],
    name: str,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    raw = environ.get(name)
    if raw is None:
        return default
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise PAPConfigError(f"{name} must contain at least one device")
    try:
        devices = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise PAPConfigError(f"{name} must be a comma-separated integer list") from exc
    if len(set(devices)) != len(devices):
        raise PAPConfigError(f"{name} must not contain duplicate devices")
    if any(device < 0 for device in devices):
        raise PAPConfigError(f"{name} device indices must be non-negative")
    return devices


def _parse_enum(enum_type: type[_EnumT], value: str, name: str) -> _EnumT:
    normalized = _normalize_flag_value(value)
    try:
        return enum_type(normalized)
    except ValueError as exc:
        choices = ", ".join(str(member.value) for member in enum_type)
        raise PAPConfigError(
            f"{name} must be one of {choices}, got {value!r}"
        ) from exc


def _parse_exec_transport(value: str) -> PAPOffloadExecTransport:
    normalized = _normalize_flag_value(value)
    aliases = {
        "nixl": PAPOffloadExecTransport.NIXL_MAILBOX,
        "nixl_mailbox": PAPOffloadExecTransport.NIXL_MAILBOX,
        "local_fast": PAPOffloadExecTransport.LOCAL_FAST,
        "cuda_ipc_fast": PAPOffloadExecTransport.LOCAL_FAST,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise PAPConfigError(
            "PAP_OFFLOAD_EXEC_TRANSPORT must be nixl_mailbox or local_fast, "
            f"got {value!r}"
        ) from exc


def _parse_kv_transport(value: str) -> PAPOffloadKVTransport:
    normalized = _normalize_flag_value(value)
    if normalized == "nixl":
        normalized = "nixl_mailbox"
    try:
        return PAPOffloadKVTransport(normalized)
    except ValueError as exc:
        raise PAPConfigError(
            "PAP_OFFLOAD_KV_TRANSPORT must be cuda_ipc or nixl_mailbox, "
            f"got {value!r}"
        ) from exc


def _validate_topology_count(
    environ: Mapping[str, str],
    name: str,
    expected: int,
) -> None:
    if name not in environ:
        return
    actual = _env_int(environ, name, expected, minimum=1)
    if actual != expected:
        raise PAPConfigError(
            f"{name}={actual} disagrees with PAP_TOPOLOGY count {expected}"
        )


def _validate_retry_range(name: str, initial_s: float, maximum_s: float) -> None:
    if maximum_s < initial_s:
        raise PAPConfigError(
            f"{name} retry maximum must not be below its initial delay"
        )
