from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)


class LeaseReleaseClient:
    def __init__(
        self,
        endpoint: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.endpoint = endpoint or os.environ.get("PAP_LEASE_RELEASE_ENDPOINT")
        self.timeout_s = (
            float(timeout_s)
            if timeout_s is not None
            else float(os.environ.get("PAP_LEASE_RELEASE_TIMEOUT", "0.2"))
        )

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint)

    def release(self, *, request_id: str, lease_id: str) -> None:
        if not self.enabled:
            return
        try:
            response = httpx.post(
                str(self.endpoint),
                json={"request_id": request_id, "lease_id": lease_id},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except Exception:
            logger.warning(
                "PAP lease release failed request_id=%s lease_id=%s",
                request_id,
                lease_id,
                exc_info=True,
            )
