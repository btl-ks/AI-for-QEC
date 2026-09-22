"""CPU-to-GPU and GPU-to-GPU QECBatch transfer contracts."""

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from ai_qec.data.schema.batch import BatchLayout, QECBatch


HostArrayT = TypeVar("HostArrayT")
DeviceArrayT = TypeVar("DeviceArrayT")
SourceDeviceArrayT = TypeVar("SourceDeviceArrayT")
TargetDeviceArrayT = TypeVar("TargetDeviceArrayT")


@dataclass(frozen=True, slots=True)
class CPUToGPUPipelineSpec:
    """PyTorch Dataset/DataLoader host-to-device policy."""

    technology_id: str = "pytorch-dataloader-h2d"
    dataset_api: str = "pytorch-dataset"
    loader_api: str = "pytorch-dataloader"
    pin_memory: bool = True
    non_blocking: bool = True
    prefetch_factor: int = 2
    num_workers: int = 0
    persistent_workers: bool = False


@dataclass(frozen=True, slots=True)
class GPUToGPUPipelineSpec:
    """CUDA-Q to PyTorch CUDA device-to-device policy."""

    technology_id: str = "cuda-q-to-pytorch-cuda"
    source_runtime: str = "cuda-q"
    target_runtime: str = "pytorch-cuda"
    interchange_priority: tuple[str, ...] = ("cuda-tensor", "dlpack")
    zero_copy_preferred: bool = True
    allow_host_staging: bool = False


@dataclass(frozen=True, slots=True)
class TransferEvidence:
    """Observed transfer path; zero-copy is evidence, not a configuration claim."""

    technology_id: str
    source_layout: BatchLayout
    target_layout: BatchLayout
    zero_copy_observed: bool
    host_staging_observed: bool
    stream_synchronized: bool


@runtime_checkable
class CPUToGPUDataPipeline(Protocol[HostArrayT, DeviceArrayT]):
    @property
    def spec(self) -> CPUToGPUPipelineSpec: ...

    def transfer(
        self,
        batch: QECBatch[HostArrayT],
    ) -> tuple[QECBatch[DeviceArrayT], TransferEvidence]: ...


@runtime_checkable
class GPUToGPUDataPipeline(
    Protocol[SourceDeviceArrayT, TargetDeviceArrayT],
):
    @property
    def spec(self) -> GPUToGPUPipelineSpec: ...

    def transfer(
        self,
        batch: QECBatch[SourceDeviceArrayT],
    ) -> tuple[QECBatch[TargetDeviceArrayT], TransferEvidence]: ...
