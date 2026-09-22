"""Stim CPU generator for code-capacity phase-flip data on a stabilizer code.

The circuit prepares ``|+>`` on every data qubit, applies the compiled noise
channel, and measures every X-type check and logical operator with ``MPP``.
Stim's Pauli-frame simulator then yields the physical Z error frame together
with detector and observable flips, so each sample carries ``e``, ``S(e)`` and
the logical truth needed by the joint RBM of Torlai & Melko.
"""

from collections.abc import Iterator

from ai_qec.data.datasets.spec import DatasetSpec
from ai_qec.data.schema.batch import BatchLayout, BatchRepresentation, MemoryResidency, QECBatch
from ai_qec.qec.backends.protocol import BackendCompatibility
from ai_qec.qec.noise import NoiseApproximation, NoiseCompilation
from ai_qec.registries import GENERATORS
from ai_qec.technology import TechnologyId
from ai_qec.utils.hashing import derive_seed

from .protocol import GeneratorDevice, SyndromeGeneratorDescriptor

SHARD_SIZE = 65_536
SPLITS = ("train", "validation", "test")
SUPPORTED_SCHEMA = "qec-batch-v1"
SPLIT_POLICY = "independent-streams"


class GeneratorCompatibilityError(RuntimeError):
    """Raised before sampling when the generator cannot honour a DatasetSpec exactly."""


class StimCodeCapacityGenerator:
    generator_id = TechnologyId.STIM_SYNDROME_CPU.value

    def __init__(
        self,
        *,
        code,
        noise: NoiseCompilation,
        dataset_artifact_id: str | None = None,
        shard_size: int = SHARD_SIZE,
    ) -> None:
        self.code = code
        self.noise = noise
        self.dataset_artifact_id = dataset_artifact_id
        self.shard_size = shard_size

    @property
    def descriptor(self) -> SyndromeGeneratorDescriptor:
        import stim

        return SyndromeGeneratorDescriptor(
            technology_id=self.generator_id,
            technology_version=stim.__version__,
            device=GeneratorDevice.CPU,
            representation=BatchRepresentation.NUMPY_ARRAY,
            residency=MemoryResidency.HOST,
        )

    def compatibility(self, spec: DatasetSpec) -> BackendCompatibility:
        import stim

        reasons: list[str] = []
        if spec.generator != self.generator_id:
            reasons.append(f"dataset.generator={spec.generator!r} is not {self.generator_id!r}")
        if spec.generator_version != stim.__version__:
            reasons.append(
                f"dataset.generator_version={spec.generator_version!r} but installed stim=={stim.__version__}"
            )
        if spec.qec.circuit_family != "code-capacity" or spec.qec.rounds != 1:
            reasons.append(
                "only single-round code-capacity sampling is implemented "
                f"(circuit_family={spec.qec.circuit_family!r}, rounds={spec.qec.rounds})"
            )
        if spec.qec.logical_basis != "X":
            reasons.append(
                "phase-flip observables are the X-type logical operators; "
                f"qec.logical_basis must be 'X', got {spec.qec.logical_basis!r}"
            )
        if spec.qec.code_family != self.code.code_family or spec.qec.distance != self.code.distance:
            reasons.append("DatasetSpec code does not match the constructed code")
        if spec.backend_semantics != "exact":
            reasons.append(
                f"dataset.backend_semantics={spec.backend_semantics!r}; only 'exact' is supported"
            )
        if self.noise.support is not NoiseApproximation.EXACT:
            reasons.append(f"noise cannot be compiled exactly: {self.noise.reason}")
        if spec.split_policy != SPLIT_POLICY:
            reasons.append(
                f"dataset.split_policy={spec.split_policy!r}; only {SPLIT_POLICY!r} is supported"
            )
        if spec.preprocessing:
            reasons.append(f"persisted preprocessing {list(spec.preprocessing)} is not supported")
        if spec.schema_version != SUPPORTED_SCHEMA:
            reasons.append(
                f"dataset.schema_version={spec.schema_version!r}; expected {SUPPORTED_SCHEMA!r}"
            )
        if reasons:
            return BackendCompatibility(
                backend_id=self.generator_id,
                support=NoiseApproximation.UNSUPPORTED,
                reason="; ".join(reasons),
            )
        return BackendCompatibility(backend_id=self.generator_id, support=NoiseApproximation.EXACT)

    def circuit(self):
        import numpy as np
        import stim

        qubits = range(self.code.num_data_qubits)
        circuit = stim.Circuit()
        circuit.append("RX", qubits)
        self.noise.model.append_to(circuit, qubits)
        rows = [*self.code.parity_check, *self.code.logical_operators]
        for row in rows:
            support = [stim.target_x(int(qubit)) for qubit in np.flatnonzero(row)]
            circuit.append("MPP", stim.target_combined_paulis(support))
        measured = len(rows)
        for check in range(self.code.num_checks):
            circuit.append("DETECTOR", [stim.target_rec(check - measured)])
        for logical in range(self.code.num_logicals):
            circuit.append(
                "OBSERVABLE_INCLUDE",
                [stim.target_rec(logical - self.code.num_logicals)],
                logical,
            )
        return circuit

    def generate(self, spec: DatasetSpec) -> Iterator[QECBatch]:
        import numpy as np
        import stim

        compatibility = self.compatibility(spec)
        if compatibility.support is not NoiseApproximation.EXACT:
            raise GeneratorCompatibilityError(compatibility.reason)
        if not self.dataset_artifact_id:
            raise GeneratorCompatibilityError("generate() requires the target dataset_artifact_id")
        circuit = self.circuit()
        counts = dict(zip(SPLITS, (spec.train_samples, spec.validation_samples, spec.test_samples)))
        for split, count in counts.items():
            for shard_index, start in enumerate(range(0, count, self.shard_size)):
                shots = min(self.shard_size, count - start)
                stream = f"dataset/{split}/shard-{shard_index:05d}"
                seed = derive_seed(spec.seed, stream)
                simulator = stim.FlipSimulator(
                    batch_size=shots,
                    num_qubits=self.code.num_data_qubits,
                    disable_stabilizer_randomization=True,
                    seed=seed,
                )
                simulator.do(circuit)
                _, errors, _, detectors, observables = simulator.to_numpy(
                    transpose=True,
                    output_zs=True,
                    output_detector_flips=True,
                    output_observable_flips=True,
                )
                detectors = detectors.astype(np.uint8)
                yield QECBatch(
                    detector_events=detectors,
                    observable_truth=observables.astype(np.uint8),
                    physical_errors=errors.astype(np.uint8),
                    sample_ids=tuple(
                        f"{split}-{index:07d}" for index in range(start, start + shots)
                    ),
                    dataset_artifact_id=self.dataset_artifact_id,
                    layout=BatchLayout(
                        representation=BatchRepresentation.NUMPY_ARRAY,
                        residency=MemoryResidency.HOST,
                        device="cpu",
                        dtype="uint8",
                        shape=tuple(detectors.shape),
                    ),
                    context={
                        "split": split,
                        "shard_index": shard_index,
                        "random_stream": stream,
                        "derived_seed": seed,
                        "technology_id": self.generator_id,
                        "technology_version": stim.__version__,
                    },
                )


@GENERATORS.register(TechnologyId.STIM_SYNDROME_CPU.value)
def build_stim_code_capacity_generator(
    *,
    code,
    noise: NoiseCompilation,
    dataset_artifact_id: str | None = None,
) -> StimCodeCapacityGenerator:
    return StimCodeCapacityGenerator(
        code=code, noise=noise, dataset_artifact_id=dataset_artifact_id
    )
