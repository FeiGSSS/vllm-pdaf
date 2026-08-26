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
            "quiesced": operation == "request_quiesce",
        }

    async def abort(self, request_ids: list[str]) -> None:
        self.calls.append(("abort", "abort", {"request_ids": request_ids}))


def test_dispatcher_preserves_commit_release_order() -> None:
    asyncio.run(_exercise_dispatcher())


def test_dispatcher_acknowledges_projection_abort_barrier() -> None:
    async def run() -> None:
        engine = _EngineClient()
        dispatcher = PAPControlDispatcher(engine)
        result = await dispatcher.abort_and_quiesce(["chatcmpl-request-0"])

        assert result["quiesced"] is True
        assert engine.calls == [
            (
                "abort",
                "abort",
                {"request_ids": ["chatcmpl-request-0"]},
            ),
            (
                "pap_control",
                "request_quiesce",
                {"request_ids": ["chatcmpl-request-0"]},
            ),
        ]

    asyncio.run(run())


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
