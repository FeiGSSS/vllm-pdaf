"""PAP topology and projection-peer activity tests."""

import pytest

from vllm.pap.topology import (
    PAPProjectionPeerActivity,
    active_pap_attention_endpoints,
    pap_attention_endpoint_for_rank,
    sync_pap_projection_peer_activity,
)


def test_projection_peer_activity_notifies_only_membership_changes() -> None:
    notifications: list[dict[str, object]] = []
    tracker = PAPProjectionPeerActivity(
        source_id="projection-0-r0",
        notify=lambda **kwargs: notifications.append(kwargs),
    )

    assert tracker.update(("http://pa0:8300",)) is True
    assert tracker.update(("http://pa0:8300",)) is False
    assert tracker.update(("http://pa1:8301",)) is True

    assert notifications == [
        {
            "attention_endpoint": "http://pa0:8300",
            "source_id": "projection-0-r0",
            "active": True,
            "membership_generation": 1,
        },
        {
            "attention_endpoint": "http://pa0:8300",
            "source_id": "projection-0-r0",
            "active": False,
            "membership_generation": 2,
        },
        {
            "attention_endpoint": "http://pa1:8301",
            "source_id": "projection-0-r0",
            "active": True,
            "membership_generation": 2,
        },
    ]
    assert tracker.active_endpoints == ("http://pa1:8301",)
    assert tracker.known_endpoints == ("http://pa0:8300", "http://pa1:8301")
    assert tracker.membership_generation == 2


def test_projection_peer_activity_retries_after_notification_failure() -> None:
    attempts: list[tuple[str, bool, int]] = []
    fail_once = True

    def notify(**kwargs) -> None:
        nonlocal fail_once
        attempts.append(
            (
                str(kwargs["attention_endpoint"]),
                bool(kwargs["active"]),
                int(kwargs["membership_generation"]),
            )
        )
        if fail_once:
            fail_once = False
            raise RuntimeError("control update failed")

    tracker = PAPProjectionPeerActivity(
        source_id="projection-0-r0",
        notify=notify,
    )

    with pytest.raises(RuntimeError, match="control update failed"):
        tracker.update(("http://pa0:8300",))

    assert tracker.active_endpoints == ()
    assert tracker.update(("http://pa0:8300",)) is True
    assert attempts == [
        ("http://pa0:8300", True, 1),
        ("http://pa0:8300", True, 2),
    ]


def test_active_pap_attention_endpoints_selects_local_tp_rank() -> None:
    endpoints = active_pap_attention_endpoints(
        request_ids=("req-a", "req-b", "req-c"),
        endpoint_by_request={
            "req-a": "http://pa0-r0:8300,http://pa0-r1:8301",
            "req-b": ("http://pa1-r0:8310", "http://pa1-r1:8311"),
            "req-c": "http://pa0-r0:8300,http://pa0-r1:8301",
        },
        tp_rank=1,
    )

    assert endpoints == ("http://pa0-r1:8301", "http://pa1-r1:8311")
    assert pap_attention_endpoint_for_rank("http://pa0:8300", tp_rank=0) == (
        "http://pa0:8300"
    )


def test_pap_attention_endpoint_for_rank_rejects_missing_rank() -> None:
    with pytest.raises(ValueError, match="1 endpoint"):
        pap_attention_endpoint_for_rank("http://pa0:8300", tp_rank=1)


def test_sync_projection_peer_activity_covers_empty_cohort(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PAP_PROJECTION_KV_UNAWARE", "1")
    monkeypatch.setenv("PAP_PROJECTION_COUNT", "2")
    monkeypatch.setenv("PAP_NIXL_MAILBOX_ACTOR_ID", "projection-0")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_LOCAL_RANK", "0")
    notifications: list[dict[str, object]] = []

    tracker = sync_pap_projection_peer_activity(
        tracker=None,
        request_ids=("req-a",),
        endpoint_by_request={"req-a": "http://pa0:8300"},
        notify=lambda **kwargs: notifications.append(kwargs),
    )
    assert tracker is not None
    assert (
        sync_pap_projection_peer_activity(
            tracker=tracker,
            request_ids=("req-a",),
            endpoint_by_request={"req-a": "http://pa0:8300"},
            notify=lambda **kwargs: notifications.append(kwargs),
        )
        is tracker
    )
    assert (
        sync_pap_projection_peer_activity(
            tracker=tracker,
            request_ids=(),
            endpoint_by_request={},
            notify=lambda **kwargs: notifications.append(kwargs),
        )
        is tracker
    )

    outcomes = [
        (item["active"], item["membership_generation"]) for item in notifications
    ]
    assert outcomes == [
        (True, 1),
        (False, 2),
    ]


def test_sync_projection_peer_activity_ignores_non_projection_role(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PAP_PROJECTION_KV_UNAWARE", raising=False)
    monkeypatch.setenv("PAP_PROJECTION_COUNT", "2")

    assert (
        sync_pap_projection_peer_activity(
            tracker=None,
            request_ids=("req-a",),
            endpoint_by_request={"req-a": "http://pa0:8300"},
            notify=lambda **_kwargs: pytest.fail("unexpected notification"),
        )
        is None
    )
