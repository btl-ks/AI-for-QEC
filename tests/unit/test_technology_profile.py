from pathlib import Path
import subprocess
import sys
import unittest

import ai_qec.notebook_api as qec


CATALOG_PATH = Path("openspec/technology-catalog.yaml")


class TechnologyProfileTests(unittest.TestCase):
    def test_selected_technology_ids_are_stable_and_cataloged(self) -> None:
        catalog = CATALOG_PATH.read_text(encoding="utf-8")
        selected = (
            qec.TechnologyId.QISKIT_CIRCUIT,
            qec.TechnologyId.STIM_CPU,
            qec.TechnologyId.CUDA_Q_GPU,
            qec.TechnologyId.STIM_NOISE,
            qec.TechnologyId.STIM_SYNDROME_CPU,
            qec.TechnologyId.CUDA_Q_SYNDROME_GPU,
            qec.TechnologyId.PYTORCH_CUDA_TRAINER,
            qec.TechnologyId.PYMATCHING_CPU_DECODER,
            qec.TechnologyId.PYTORCH_GPU_DECODER,
        )
        for technology_id in selected:
            with self.subTest(technology_id=technology_id):
                self.assertIn(f"id: {technology_id.value}", catalog)

    def test_circuit_and_noise_adapter_results_preserve_provenance(self) -> None:
        circuit = qec.CircuitBuildResult(
            circuit=object(),
            technology_id=qec.TechnologyId.QISKIT_CIRCUIT,
            technology_version="unresolved",
            qec_spec_digest="sha256:qec",
            circuit_digest="sha256:circuit",
        )
        noise = qec.NoiseCompilation(
            model=object(),
            technology_id=qec.TechnologyId.STIM_NOISE,
            technology_version="unresolved",
            source_spec_digest="sha256:noise",
            support=qec.NoiseApproximation.EXACT,
        )

        self.assertEqual(circuit.technology_id, "qiskit-circuit")
        self.assertEqual(noise.support, qec.NoiseApproximation.EXACT)
        self.assertIsNone(noise.approximation_id)

    def test_tensor_first_batch_describes_cpu_and_cuda_layouts(self) -> None:
        cpu_layout = qec.BatchLayout(
            representation=qec.BatchRepresentation.PYTORCH_TENSOR,
            residency=qec.MemoryResidency.PINNED_HOST,
            device="cpu",
            dtype="bool",
            shape=(32, 24),
        )
        cuda_layout = qec.BatchLayout(
            representation=qec.BatchRepresentation.PYTORCH_CUDA_TENSOR,
            residency=qec.MemoryResidency.CUDA_DEVICE,
            device="cuda:0",
            dtype="bool",
            shape=(32, 24),
            interchange="dlpack",
            zero_copy=True,
        )
        batch = qec.QECBatch(
            detector_events=((False,),),
            observable_truth=((False,),),
            sample_ids=("sample-1",),
            dataset_artifact_id="dataset-1",
            layout=cuda_layout,
        )

        self.assertEqual(cpu_layout.residency, qec.MemoryResidency.PINNED_HOST)
        self.assertEqual(batch.layout, cuda_layout)
        self.assertTrue(batch.layout.zero_copy)

    def test_transfer_policies_express_v01_defaults(self) -> None:
        cpu_to_gpu = qec.CPUToGPUPipelineSpec(num_workers=4)
        gpu_to_gpu = qec.GPUToGPUPipelineSpec()

        self.assertTrue(cpu_to_gpu.pin_memory)
        self.assertTrue(cpu_to_gpu.non_blocking)
        self.assertEqual(cpu_to_gpu.loader_api, "pytorch-dataloader")
        self.assertTrue(gpu_to_gpu.zero_copy_preferred)
        self.assertFalse(gpu_to_gpu.allow_host_staging)
        self.assertIn("dlpack", gpu_to_gpu.interchange_priority)

    def test_training_and_decoder_runtime_choices_are_explicit(self) -> None:
        execution = qec.ExecutionSpec(
            device="cuda:0",
            cpu_count=8,
            gpu_count=1,
            distributed=False,
            num_workers=4,
            mixed_precision=True,
            compile_model=False,
        )
        runtime = qec.DecoderRuntimeDescriptor(
            technology_id=qec.TechnologyId.PYMATCHING_CPU_DECODER,
            technology_version="unresolved",
            device="cpu",
        )
        result = qec.DecodeResult(
            request_id="request-1",
            decoder_id="baseline",
            status=qec.DecodeStatus.SUCCEEDED,
            predictions=(False,),
            runtime=runtime,
        )

        self.assertEqual(execution.trainer_framework, "pytorch")
        self.assertEqual(result.runtime.technology_id, "pymatching-cpu-decoder")

    def test_contract_import_does_not_import_selected_frameworks(self) -> None:
        script = (
            "import sys, ai_qec.notebook_api; "
            "print([name for name in ('qiskit', 'stim', 'cudaq', 'torch', 'pymatching') if name in sys.modules])"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
