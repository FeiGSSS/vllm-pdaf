from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".claude/skills/vllm-pap-benchmark/scripts"
SETUP = SCRIPTS / "setup_ucx122_nixl.sh"
RUNTIME = SCRIPTS / "ucx122_runtime_env.sh"


def test_repo_local_ucx122_runtime_is_pinned_and_fail_closed() -> None:
    assert SETUP.is_file()
    assert RUNTIME.is_file()

    setup_text = SETUP.read_text(encoding="utf-8")
    runtime_text = RUNTIME.read_text(encoding="utf-8")
    gitignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert 'UCX_VERSION="1.22.0"' in setup_text
    assert 'NIXL_VERSION="1.3.0"' in setup_text
    for flag in (
        "--enable-shared",
        "--disable-static",
        "--enable-cma",
        "--enable-devel-headers",
        "--enable-mt",
        "--with-cuda=/usr",
        "--without-verbs",
        "--without-rdmacm",
        "--without-gdrcopy",
    ):
        assert flag in setup_text

    assert "configure_ucx122_runtime" in runtime_text
    assert "verify_ucx122_runtime" in runtime_text
    assert "UCX_PROTO_EMULATION_ENABLE=n" in runtime_text
    assert "UCX_TLS=cuda_ipc,cuda_copy,tcp" in runtime_text
    assert "NIXL_PLUGIN_DIR" in runtime_text
    assert "libplugin_UCX.so" in runtime_text
    assert "ucx_info" in runtime_text
    assert "--enable-mt" in runtime_text
    assert "ldd" in runtime_text
    assert ".venv/bin/python" in runtime_text
    assert ".local/" in gitignore_text.splitlines()
