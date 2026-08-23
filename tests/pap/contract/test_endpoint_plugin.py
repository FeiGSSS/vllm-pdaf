# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio

from vllm.pap.prefill_control_router import PAPControlDispatcher


class _EngineClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    async def call_utility_async(
        self, method: str, operation: str, payload: dict[str, object]
    ) -> dict[str, object]:
        await asyncio.sleep(0)
        self.calls.append((method, operation, payload))
        return {
            "applied": operation == "decode_commit",
            "released": operation == "lease_release",
        }


def test_dispatcher_preserves_commit_release_order() -> None:
    asyncio.run(_exercise_dispatcher())


async def _exercise_dispatcher() -> None:
    engine = _EngineClient()
    dispatcher = PAPControlDispatcher(engine)
    first = await dispatcher.submit("decode_commit", {"commit_seq": 1}, wait=False)
    second = await dispatcher.submit("decode_commit", {"commit_seq": 2}, wait=False)
    released = await dispatcher.submit(
        "lease_release", {"final_commit_seq": 2}, wait=True
    )
    await dispatcher.close()

    assert first == {"accepted": True}
    assert second == {"accepted": True}
    assert released["released"] is True
    assert [(call[1], call[2]) for call in engine.calls] == [
        ("decode_commit", {"commit_seq": 1}),
        ("decode_commit", {"commit_seq": 2}),
        ("lease_release", {"final_commit_seq": 2}),
    ]
