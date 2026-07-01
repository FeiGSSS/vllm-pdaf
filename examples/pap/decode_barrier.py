# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional proxy-side decode barrier for PAP experiments."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

_TRUE_ENV_VALUES = ("1", "true", "yes", "on")


@dataclass(frozen=True)
class DecodeBarrierRelease:
    generation: int
    count: int
    reason: str


class DecodeBarrier:
    """Async cyclic barrier used before PAP proxy sends projection requests."""

    def __init__(self, count: int, timeout_s: float) -> None:
        self.count = max(0, count)
        self.timeout_s = max(0.0, timeout_s)
        self._condition = asyncio.Condition()
        self._generation = 0
        self._arrived = 0
        self._last_release = DecodeBarrierRelease(
            generation=-1,
            count=0,
            reason="disabled",
        )

    @property
    def enabled(self) -> bool:
        return self.count > 1

    @classmethod
    def from_env(cls) -> DecodeBarrier:
        count = _env_int(
            "PAP_PROXY_DECODE_BARRIER_COUNT",
            fallback_names=("PAP_PROXY_PROJECTION_BARRIER_COUNT",),
            default=0,
        )
        timeout_s = _env_float(
            "PAP_PROXY_DECODE_BARRIER_TIMEOUT_SEC",
            fallback_names=("PAP_PROXY_PROJECTION_BARRIER_TIMEOUT_SEC",),
            default=120.0,
        )
        return cls(count=count, timeout_s=timeout_s)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "count": self.count,
            "timeout_s": self.timeout_s,
            "generation": self._generation,
            "arrived": self._arrived,
            "last_release": {
                "generation": self._last_release.generation,
                "count": self._last_release.count,
                "reason": self._last_release.reason,
            },
        }

    async def wait(
        self,
        request_id: str,
        *,
        logger: logging.Logger | None = None,
    ) -> DecodeBarrierRelease | None:
        if not self.enabled:
            return None

        start = time.perf_counter()
        async with self._condition:
            generation = self._generation
            self._arrived += 1
            arrived = self._arrived
            if logger is not None:
                logger.info(
                    "request_id=%s decode_barrier_arrive generation=%d "
                    "arrived=%d/%d",
                    request_id,
                    generation,
                    arrived,
                    self.count,
                )

            if arrived >= self.count:
                release = self._release_locked(generation, reason="count")
            else:
                release = await self._wait_for_release_locked(generation)

        wait_ms = (time.perf_counter() - start) * 1000.0
        if logger is not None:
            logger.info(
                "request_id=%s decode_barrier_release generation=%d "
                "released=%d reason=%s wait_ms=%.3f",
                request_id,
                release.generation,
                release.count,
                release.reason,
                wait_ms,
            )
        return release

    async def _wait_for_release_locked(
        self,
        generation: int,
    ) -> DecodeBarrierRelease:
        if self.timeout_s <= 0.0:
            await self._condition.wait_for(lambda: self._generation != generation)
            return self._last_release

        try:
            await asyncio.wait_for(
                self._condition.wait_for(lambda: self._generation != generation),
                timeout=self.timeout_s,
            )
        except TimeoutError:
            if self._generation == generation:
                return self._release_locked(generation, reason="timeout")
        return self._last_release

    def _release_locked(
        self,
        generation: int,
        *,
        reason: str,
    ) -> DecodeBarrierRelease:
        release = DecodeBarrierRelease(
            generation=generation,
            count=self._arrived,
            reason=reason,
        )
        self._last_release = release
        self._arrived = 0
        self._generation += 1
        self._condition.notify_all()
        return release


def _env_int(
    name: str,
    *,
    fallback_names: tuple[str, ...] = (),
    default: int,
) -> int:
    value = _env_value(name, fallback_names=fallback_names)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(
    name: str,
    *,
    fallback_names: tuple[str, ...] = (),
    default: float,
) -> float:
    value = _env_value(name, fallback_names=fallback_names)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_value(name: str, *, fallback_names: tuple[str, ...]) -> str | None:
    for candidate in (name, *fallback_names):
        value = os.environ.get(candidate)
        if value is not None:
            return value
    return None


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in _TRUE_ENV_VALUES
