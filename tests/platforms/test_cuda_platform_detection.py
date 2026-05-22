from unittest.mock import MagicMock, patch

from vllm.platforms import cuda_platform_plugin


def test_cuda_platform_falls_back_to_torch_when_nvml_fails() -> None:
    class FakeNVMLError(Exception):
        pass

    fake_pynvml = MagicMock()
    fake_pynvml.nvmlInit.side_effect = FakeNVMLError("driver/library mismatch")
    FakeNVMLError.__name__ = "NVMLError_LibraryMismatch"

    with (
        patch("vllm.utils.import_utils.import_pynvml", return_value=fake_pynvml),
        patch("torch.cuda.is_available", return_value=True),
        patch("vllm.platforms.vllm_version_matches_substr", return_value=False),
    ):
        assert cuda_platform_plugin() == "vllm.platforms.cuda.CudaPlatform"
