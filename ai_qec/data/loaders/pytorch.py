"""PyTorch Dataset/DataLoader host-to-device pipeline (``pytorch-dataloader-h2d``)."""

from collections.abc import Iterator, Sequence

from ai_qec.data.pipeline import CPUToGPUPipelineSpec, TransferEvidence
from ai_qec.data.schema.batch import BatchLayout, BatchRepresentation, MemoryResidency, QECBatch
from ai_qec.registries import CPU_TO_GPU_PIPELINES
from ai_qec.technology import TechnologyId

ARRAY_FIELDS = ("detector_events", "observable_truth", "physical_errors")


class _IndexBatchDataset:
    """Map-style dataset whose items are whole minibatches selected by an index list."""

    def __init__(self, tensors: dict) -> None:
        self.tensors = tensors
        self.length = len(next(iter(tensors.values())))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, indices):
        return {name: tensor[indices] for name, tensor in self.tensors.items()}


class TorchHostToDevicePipeline:
    """Move host QECBatch arrays to the execution device and record what happened."""

    technology_id = TechnologyId.PYTORCH_DATALOADER_H2D.value

    def __init__(self, *, spec: CPUToGPUPipelineSpec, device: str) -> None:
        import torch

        self._spec = spec
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        self.pin = bool(spec.pin_memory and self.is_cuda)
        self.non_blocking = bool(spec.non_blocking and self.is_cuda)

    @property
    def spec(self) -> CPUToGPUPipelineSpec:
        return self._spec

    def target_layout(self, shape: Sequence[int]) -> BatchLayout:
        if self.is_cuda:
            return BatchLayout(
                representation=BatchRepresentation.PYTORCH_CUDA_TENSOR,
                residency=MemoryResidency.CUDA_DEVICE,
                device=str(self.device),
                dtype="uint8",
                shape=tuple(shape),
            )
        return BatchLayout(
            representation=BatchRepresentation.PYTORCH_TENSOR,
            residency=MemoryResidency.HOST,
            device="cpu",
            dtype="uint8",
            shape=tuple(shape),
        )

    def _evidence(self, source: BatchLayout, shape: Sequence[int]) -> TransferEvidence:
        return TransferEvidence(
            technology_id=self.technology_id,
            source_layout=source,
            target_layout=self.target_layout(shape),
            zero_copy_observed=not self.is_cuda,
            host_staging_observed=False,
            stream_synchronized=self.is_cuda,
        )

    def _to_device(self, tensor):
        if self.pin:
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=self.non_blocking)

    def transfer(self, batch: QECBatch) -> tuple[QECBatch, TransferEvidence]:
        """Copy every present array field; identities and context are preserved."""

        import dataclasses

        import numpy as np
        import torch

        moved = {}
        for name in ARRAY_FIELDS:
            value = getattr(batch, name)
            if value is not None:
                moved[name] = self._to_device(
                    torch.from_numpy(np.ascontiguousarray(value, dtype=np.uint8))
                )
        if self.is_cuda:
            torch.cuda.current_stream(self.device).synchronize()
        shape = tuple(moved["detector_events"].shape)
        evidence = self._evidence(batch.layout, shape)
        device_batch = dataclasses.replace(batch, layout=evidence.target_layout, **moved)
        return device_batch, evidence

    def loader(
        self,
        batch: QECBatch,
        *,
        batch_size: int,
        generator,
        fields: Sequence[str] = ("physical_errors", "detector_events"),
    ) -> Iterator[dict]:
        """Shuffled minibatches through DataLoader with pinned memory and non-blocking copies."""

        import numpy as np
        import torch
        from torch.utils.data import BatchSampler, DataLoader, RandomSampler

        tensors = {
            name: torch.from_numpy(np.ascontiguousarray(getattr(batch, name), dtype=np.uint8))
            for name in fields
        }
        dataset = _IndexBatchDataset(tensors)
        sampler = BatchSampler(
            RandomSampler(dataset, generator=generator), batch_size, drop_last=False
        )
        workers = self._spec.num_workers
        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=workers,
            pin_memory=self.pin,
            prefetch_factor=self._spec.prefetch_factor if workers > 0 else None,
            persistent_workers=self._spec.persistent_workers and workers > 0,
            generator=generator,
        )
        for item in loader:
            yield {
                name: tensor.to(self.device, non_blocking=self.non_blocking)
                for name, tensor in item.items()
            }

    def loader_evidence(self, batch: QECBatch, batch_size: int) -> TransferEvidence:
        evidence = self._evidence(batch.layout, (batch_size, batch.detector_events.shape[1]))
        return TransferEvidence(
            technology_id=evidence.technology_id,
            source_layout=batch.layout,
            target_layout=evidence.target_layout,
            zero_copy_observed=False,
            host_staging_observed=False,
            stream_synchronized=False,
        )


@CPU_TO_GPU_PIPELINES.register(TechnologyId.PYTORCH_DATALOADER_H2D.value)
def build_torch_h2d_pipeline(
    *, spec: CPUToGPUPipelineSpec, device: str
) -> TorchHostToDevicePipeline:
    return TorchHostToDevicePipeline(spec=spec, device=device)
