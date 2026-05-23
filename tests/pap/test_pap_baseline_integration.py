from pathlib import Path

BASELINE = Path("/home/fei/research/PD/test/baseline")
PAPF = BASELINE / "papf"


def test_external_baseline_has_papf_mode_contract() -> None:
    config = PAPF / "config.sh"
    launcher = PAPF / "launch_service.sh"

    assert config.exists()
    assert launcher.exists()

    config_text = config.read_text()
    launcher_text = launcher.read_text()

    assert "PAPF_PROXY_PORT" in config_text
    assert "VLLM_BIN" in config_text
    assert "/home/fei/research/PD/vllm-papf/.venv/bin/vllm" in config_text
    assert "PAP_SERVICE_ONLY=1" in launcher_text
    assert "PAP_SKIP_SMOKE_REQUEST=1" in launcher_text
    assert "PAP_STATUS_FILE" in launcher_text
    assert "launch_pap_6pa2p_qwen3_8b_nixl.sh" in launcher_text



def test_external_baseline_papf_launcher_cleans_child_process_group() -> None:
    launcher = PAPF / "launch_service.sh"
    text = launcher.read_text()

    assert "LAUNCH_PID" in text
    assert "setsid bash examples/pap/launch_pap_6pa2p_qwen3_8b_nixl.sh" in text
    assert 'kill -TERM -- "-${LAUNCH_PID}"' in text
    assert 'kill -KILL -- "-${LAUNCH_PID}"' in text
    assert 'wait "${LAUNCH_PID}"' in text
