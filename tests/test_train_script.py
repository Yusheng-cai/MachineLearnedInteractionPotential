from pathlib import Path


def test_gpu_training_script_installs_torch_before_mace() -> None:
    script = Path("scripts/train_mace_gpu.sh").read_text(encoding="utf-8")

    torch_install = script.index('python -m pip install "$PYTORCH_SPEC"')
    mace_install = script.index('python -m pip install -e ".[dev,mace]"')

    assert "PYTORCH_INDEX_URL" in script
    assert "import torch" in script
    assert "torch.cuda.is_available()" in script
    assert torch_install < mace_install


def test_gpu_training_script_does_not_require_torch_audio_packages() -> None:
    script = Path("scripts/train_mace_gpu.sh").read_text(encoding="utf-8")

    assert "torchaudio" not in script
    assert "torchvision" not in script


def test_gpu_training_script_pins_cuda_12_compatible_torch() -> None:
    script = Path("scripts/train_mace_gpu.sh").read_text(encoding="utf-8")

    assert 'PYTORCH_SPEC="${PYTORCH_SPEC:-torch==2.5.1}"' in script
    assert 'MAX_TORCH_CUDA="${MAX_TORCH_CUDA:-12.2}"' in script
    assert "Torch CUDA runtime" in script
