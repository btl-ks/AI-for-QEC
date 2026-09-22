"""Stable identifiers for the selected v0.1 technology profile."""

from enum import StrEnum


class TechnologyId(StrEnum):
    QISKIT_CIRCUIT = "qiskit-circuit"
    STIM_CPU = "stim-cpu"
    CUDA_Q_GPU = "cuda-q-gpu"
    STIM_NOISE = "stim-noise"
    STIM_SYNDROME_CPU = "stim-syndrome-cpu"
    CUDA_Q_SYNDROME_GPU = "cuda-q-syndrome-gpu"
    PYTORCH_CUDA_TRAINER = "pytorch-cuda-trainer"
    PYMATCHING_CPU_DECODER = "pymatching-cpu-decoder"
    PYTORCH_GPU_DECODER = "pytorch-gpu-decoder"
    PYTORCH_TENSOR = "pytorch-tensor"
    NUMPY_ARRAY = "numpy-array"
    BIT_PACKED_CPU_BUFFER = "bit-packed-cpu-buffer"
    PYTORCH_CUDA_TENSOR = "pytorch-cuda-tensor"
    DLPACK = "dlpack"
    PYTORCH_DATALOADER_H2D = "pytorch-dataloader-h2d"
    CUDA_Q_TO_PYTORCH_CUDA = "cuda-q-to-pytorch-cuda"
    NVIDIA_DALI = "nvidia-dali"
    RAY_DATA = "ray-data"
    CUSTOM_CUDA_EXTENSION = "custom-cuda-extension"


class TechnologyImplementation(StrEnum):
    NONE = "none"
    CONTRACTS_ONLY = "contracts-only"
    PARTIAL = "partial"
    IMPLEMENTED = "implemented"
    VALIDATED = "validated"

