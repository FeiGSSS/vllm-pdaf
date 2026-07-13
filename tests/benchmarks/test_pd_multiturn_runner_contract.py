from pathlib import Path


ROOT = Path(__file__).parents[2]
RUNNER = (
    ROOT
    / ".claude/skills/vllm-pap-benchmark/scripts/run_pd_multiturn_load.sh"
)


def test_pd_runner_uses_one_official_connector_for_both_modes() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert "NixlConnector" in text
    assert "NixlPushConnector" not in text
    assert "bidirectional_kv_xfer" in text
    assert "oneway" in text
    assert "twoway" in text
    assert "configure_ucx122_runtime" in text
    assert "verify_ucx122_runtime" in text
    assert "disagg_proxy_multiturn.py" in text


def test_pd_runner_records_and_audits_transfer_mode() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert "PD_TRANSFER_MODE=" in text
    assert "BIDIRECTIONAL_KV_XFER=" in text
    assert "--proxy-log" in text
    assert '"nixl-${TRANSFER_MODE}"' in text
