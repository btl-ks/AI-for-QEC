"""File-backed DatasetKey, DatasetRegistry, DatasetResolver and split reader.

Layout under the datasets root::

    registry.json                      DatasetKey -> artifact id + manifest checksum
    <artifact-id>/manifest.json        splits, shard checksums, dataset spec
    <artifact-id>/generation.json      generator/noise/code provenance
    <artifact-id>/<split>/shard-NNNNN.npz   bit-packed physical_errors/detector_events/observable_truth

All URIs are POSIX paths relative to the datasets root. Committed artifacts are
never modified; a failed verification is reported, never repaired in place.
"""

from collections.abc import Callable, Iterable
from datetime import datetime, UTC
import os
from pathlib import Path
import shutil
import uuid

from ai_qec.data.schema.batch import BatchLayout, BatchRepresentation, MemoryResidency, QECBatch
from ai_qec.qec.noise import NoiseApproximation
from ai_qec.utils.hashing import read_json, sha256_file, sha256_json, to_jsonable, write_json_atomic

from .artifact import DatasetArtifact, DatasetSplit
from .identity import DatasetKey
from .instance import DatasetInstance, DatasetRole
from .resolution import DatasetResolution
from .spec import DatasetSpec

MANIFEST_SCHEMA = "local-dataset-manifest-v1"
REGISTRY_SCHEMA = "local-dataset-registry-v1"
KEY_ALGORITHM = "sha256-canonical-json-v1"
FIELDS = ("physical_errors", "detector_events", "observable_truth")
SPLIT_NAMES = ("train", "validation", "test")


class DatasetIntegrityError(RuntimeError):
    """A registered or staged dataset failed checksum or consistency verification."""


class CanonicalDatasetIdentity:
    """DatasetKey = SHA-256 of the canonical JSON of the DatasetSpec."""

    def key_for(self, spec: DatasetSpec) -> DatasetKey:
        return DatasetKey(value=sha256_json(spec), algorithm=KEY_ALGORITHM)


def dataset_artifact_id(key: DatasetKey) -> str:
    return "ds-" + key.value.split(":", 1)[1][:20]


def code_consistency_validator(code) -> Callable[[QECBatch], None]:
    """Check every sample against the vendor-neutral code definition."""

    import numpy as np

    def validate(batch: QECBatch) -> None:
        split = batch.context.get("split", "?")
        if batch.physical_errors is None:
            raise DatasetIntegrityError(
                f"[{split}] physical_errors are required for code-capacity data"
            )
        arrays = {name: np.asarray(getattr(batch, name)) for name in FIELDS}
        widths = {
            "physical_errors": code.num_data_qubits,
            "detector_events": code.num_checks,
            "observable_truth": code.num_logicals,
        }
        rows = len(batch.sample_ids)
        for name, array in arrays.items():
            if array.shape != (rows, widths[name]) or array.dtype != np.uint8:
                raise DatasetIntegrityError(
                    f"[{split}] {name} has shape {array.shape}/{array.dtype}, expected ({rows}, {widths[name]})/uint8"
                )
            if array.size and int(array.max()) > 1:
                raise DatasetIntegrityError(f"[{split}] {name} contains non-binary values")
        bad_syndrome = np.flatnonzero(
            np.any(code.syndrome(arrays["physical_errors"]) != arrays["detector_events"], axis=1)
        )
        if bad_syndrome.size:
            raise DatasetIntegrityError(
                f"[{split}] {bad_syndrome.size} samples have detector_events != S(physical_errors); "
                f"first: {batch.sample_ids[int(bad_syndrome[0])]}"
            )
        bad_logical = np.flatnonzero(
            np.any(
                code.logical_flips(arrays["physical_errors"]) != arrays["observable_truth"], axis=1
            )
        )
        if bad_logical.size:
            raise DatasetIntegrityError(
                f"[{split}] {bad_logical.size} samples have observable_truth inconsistent with physical_errors"
            )

    return validate


class LocalDatasetStore:
    """Immutable dataset artifacts and their key registry (single writer)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- registry ---------------------------------------------------------------------------

    @property
    def registry_path(self) -> Path:
        return self.root / "registry.json"

    def _entries(self) -> dict[str, dict[str, str]]:
        if not self.registry_path.exists():
            return {}
        data = read_json(self.registry_path)
        if not isinstance(data, dict) or data.get("schema_version") != REGISTRY_SCHEMA:
            raise DatasetIntegrityError(f"unrecognized dataset registry at {self.registry_path}")
        return dict(data["entries"])

    def find(self, key: DatasetKey) -> DatasetArtifact | None:
        entry = self._entries().get(key.value)
        if entry is None:
            return None
        manifest_path = self.path(entry["manifest"])
        if not manifest_path.is_file():
            raise DatasetIntegrityError(
                f"registry entry {key.value} points to missing manifest {entry['manifest']}"
            )
        return self._artifact_from_manifest(
            read_json(manifest_path), checksum=entry["manifest_checksum"]
        )

    def register(self, artifact: DatasetArtifact) -> None:
        entries = self._entries()
        existing = entries.get(artifact.key.value)
        record = {
            "artifact_id": artifact.artifact_id,
            "manifest": artifact.manifest_uri,
            "manifest_checksum": artifact.checksum,
        }
        if existing is not None and existing != record:
            raise DatasetIntegrityError(
                f"DatasetKey {artifact.key.value} is already registered to {existing['artifact_id']}"
            )
        entries[artifact.key.value] = record
        write_json_atomic(
            self.registry_path, {"schema_version": REGISTRY_SCHEMA, "entries": entries}
        )

    def verify(self, artifact: DatasetArtifact) -> bool:
        return not self.verification_errors(artifact)

    def verification_errors(self, artifact: DatasetArtifact) -> list[str]:
        errors: list[str] = []
        manifest_path = self.path(artifact.manifest_uri)
        if not manifest_path.is_file():
            return [f"manifest {artifact.manifest_uri} is missing"]
        actual = sha256_file(manifest_path)
        if actual != artifact.checksum:
            errors.append(f"manifest checksum {actual} != registered {artifact.checksum}")
        entry = self._entries().get(artifact.key.value)
        if entry is None or entry["artifact_id"] != artifact.artifact_id:
            errors.append(
                f"{artifact.artifact_id} is not registered under key {artifact.key.value}"
            )
        manifest = read_json(manifest_path)
        if manifest.get("key", {}).get("value") != artifact.key.value:
            errors.append("manifest key does not match the registered DatasetKey")
        generation = manifest.get("generation", {})
        generation_path = self.path(f"{artifact.artifact_id}/{generation.get('path', '')}")
        if not generation_path.is_file() or sha256_file(generation_path) != generation.get(
            "checksum"
        ):
            errors.append("generation provenance is missing or modified")
        for split in manifest.get("splits", []):
            errors.extend(self._split_errors(artifact.artifact_id, split))
        return errors

    def _split_errors(self, artifact_id: str, split: dict) -> list[str]:
        errors = []
        for shard in split["shards"]:
            path = self.path(f"{artifact_id}/{shard['path']}")
            if not path.is_file():
                errors.append(f"{split['name']}: shard {shard['path']} is missing")
            elif sha256_file(path) != shard["checksum"]:
                errors.append(f"{split['name']}: shard {shard['path']} checksum mismatch")
        return errors

    # -- reading ----------------------------------------------------------------------------

    def path(self, uri: str) -> Path:
        return self.root / Path(uri)

    def manifest(self, artifact: DatasetArtifact) -> dict:
        return read_json(self.path(artifact.manifest_uri))

    def load_split(
        self,
        artifact: DatasetArtifact,
        split: str,
        *,
        validator: Callable[[QECBatch], None] | None = None,
    ) -> QECBatch:
        """Verify shard checksums, unpack bits, and return one host QECBatch."""

        import numpy as np

        manifest = self.manifest(artifact)
        entries = {item["name"]: item for item in manifest["splits"]}
        if split not in entries:
            raise KeyError(f"dataset {artifact.artifact_id} has no split {split!r}")
        entry = entries[split]
        errors = self._split_errors(artifact.artifact_id, entry)
        if errors:
            raise DatasetIntegrityError(f"{artifact.artifact_id}: " + "; ".join(errors))
        widths = manifest["fields"]
        parts: dict[str, list] = {name: [] for name in FIELDS}
        for shard in entry["shards"]:
            with np.load(self.path(f"{artifact.artifact_id}/{shard['path']}")) as data:
                for name in FIELDS:
                    parts[name].append(np.unpackbits(data[name], axis=1, count=widths[name]))
        arrays = {name: np.concatenate(chunks, axis=0) for name, chunks in parts.items()}
        count = int(entry["sample_count"])
        batch = QECBatch(
            detector_events=arrays["detector_events"],
            observable_truth=arrays["observable_truth"],
            physical_errors=arrays["physical_errors"],
            sample_ids=tuple(f"{split}-{index:07d}" for index in range(count)),
            dataset_artifact_id=artifact.artifact_id,
            layout=BatchLayout(
                representation=BatchRepresentation.NUMPY_ARRAY,
                residency=MemoryResidency.HOST,
                device="cpu",
                dtype="uint8",
                shape=tuple(arrays["detector_events"].shape),
            ),
            context={
                "split": split,
                "persisted_representation": BatchRepresentation.BIT_PACKED_CPU_BUFFER.value,
                "conversion": "numpy.unpackbits(axis=1) -> uint8",
            },
        )
        if validator is not None:
            validator(batch)
        return batch

    # -- writing ----------------------------------------------------------------------------

    def commit(
        self,
        spec: DatasetSpec,
        key: DatasetKey,
        batches: Iterable[QECBatch],
        *,
        validator: Callable[[QECBatch], None],
        producer_attempt_id: str,
        provenance: dict[str, object],
    ) -> DatasetArtifact:
        """Stage, validate, atomically publish, and register a new artifact."""

        import numpy as np

        artifact_id = dataset_artifact_id(key)
        final_dir = self.root / artifact_id
        if final_dir.exists():
            return self._adopt_existing(key, final_dir)

        staging = self.root / ".staging" / f"{artifact_id}-{uuid.uuid4().hex[:8]}"
        staging.mkdir(parents=True)
        try:
            expected = dict(
                zip(SPLIT_NAMES, (spec.train_samples, spec.validation_samples, spec.test_samples))
            )
            splits: dict[str, dict] = {
                name: {"name": name, "sample_count": 0, "shards": []} for name in SPLIT_NAMES
            }
            widths: dict[str, int] | None = None
            shard_seeds: list[dict[str, object]] = []
            for batch in batches:
                split = str(batch.context["split"])
                if batch.dataset_artifact_id != artifact_id:
                    raise DatasetIntegrityError(
                        f"generator produced dataset id {batch.dataset_artifact_id!r}, expected {artifact_id!r}"
                    )
                entry = splits[split]
                first_id = f"{split}-{entry['sample_count']:07d}"
                if not batch.sample_ids or batch.sample_ids[0] != first_id:
                    raise DatasetIntegrityError(
                        f"[{split}] non-contiguous sample ids at {first_id}"
                    )
                validator(batch)
                arrays = {name: np.asarray(getattr(batch, name), dtype=np.uint8) for name in FIELDS}
                batch_widths = {name: int(array.shape[1]) for name, array in arrays.items()}
                if widths is None:
                    widths = batch_widths
                elif widths != batch_widths:
                    raise DatasetIntegrityError(f"[{split}] field widths changed between shards")
                relative = f"{split}/shard-{len(entry['shards']):05d}.npz"
                shard_path = staging / relative
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    shard_path,
                    **{name: np.packbits(array, axis=1) for name, array in arrays.items()},
                )
                entry["shards"].append(
                    {
                        "path": relative,
                        "sample_count": len(batch.sample_ids),
                        "checksum": sha256_file(shard_path),
                    }
                )
                entry["sample_count"] += len(batch.sample_ids)
                shard_seeds.append(
                    {
                        "shard": relative,
                        **{
                            k: batch.context[k]
                            for k in ("random_stream", "derived_seed")
                            if k in batch.context
                        },
                    }
                )
            for name, count in expected.items():
                if count < 1 or splits[name]["sample_count"] != count:
                    raise DatasetIntegrityError(
                        f"[{name}] generated {splits[name]['sample_count']} samples, DatasetSpec requires {count}"
                    )
            for entry in splits.values():
                entry["checksum"] = sha256_json([shard["checksum"] for shard in entry["shards"]])

            generation_path = staging / "generation.json"
            write_json_atomic(generation_path, {**provenance, "shards": shard_seeds})
            manifest = {
                "schema_version": MANIFEST_SCHEMA,
                "artifact_id": artifact_id,
                "key": to_jsonable(key),
                "dataset_spec": to_jsonable(spec),
                "persisted_representation": BatchRepresentation.BIT_PACKED_CPU_BUFFER.value,
                "fields": widths,
                "splits": [splits[name] for name in SPLIT_NAMES],
                "generation": {"path": "generation.json", "checksum": sha256_file(generation_path)},
                "producer_attempt_id": producer_attempt_id,
                "created_at": datetime.now(UTC).isoformat(),
            }
            write_json_atomic(staging / "manifest.json", manifest)
            os.rename(staging, final_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        artifact = self._artifact_from_manifest(
            read_json(final_dir / "manifest.json"),
            checksum=sha256_file(final_dir / "manifest.json"),
        )
        self.register(artifact)
        errors = self.verification_errors(artifact)
        if errors:
            raise DatasetIntegrityError(
                f"{artifact_id} failed post-commit verification: " + "; ".join(errors)
            )
        return artifact

    def _adopt_existing(self, key: DatasetKey, final_dir: Path) -> DatasetArtifact:
        """Register a published but unregistered artifact only if it verifies completely."""

        manifest_path = final_dir / "manifest.json"
        if not manifest_path.is_file():
            raise DatasetIntegrityError(
                f"{final_dir.name} exists without a manifest; refusing to overwrite it"
            )
        manifest = read_json(manifest_path)
        if manifest.get("key", {}).get("value") != key.value:
            raise DatasetIntegrityError(f"{final_dir.name} exists for a different DatasetKey")
        artifact = self._artifact_from_manifest(manifest, checksum=sha256_file(manifest_path))
        split_errors = [
            error
            for split in manifest["splits"]
            for error in self._split_errors(artifact.artifact_id, split)
        ]
        if split_errors:
            raise DatasetIntegrityError(f"{final_dir.name} is corrupt: " + "; ".join(split_errors))
        self.register(artifact)
        return artifact

    def _artifact_from_manifest(self, manifest: dict, *, checksum: str) -> DatasetArtifact:
        artifact_id = manifest["artifact_id"]
        key = manifest["key"]
        return DatasetArtifact(
            artifact_id=artifact_id,
            key=DatasetKey(
                value=key["value"], algorithm=key["algorithm"], schema_version=key["schema_version"]
            ),
            manifest_uri=f"{artifact_id}/manifest.json",
            checksum=checksum,
            splits=tuple(
                DatasetSplit(
                    name=split["name"],
                    sample_count=int(split["sample_count"]),
                    shard_uris=tuple(f"{artifact_id}/{shard['path']}" for shard in split["shards"]),
                    checksum=split["checksum"],
                )
                for split in manifest["splits"]
            ),
            generator_provenance_uri=f"{artifact_id}/{manifest['generation']['path']}",
        )


class LocalDatasetResolver:
    """Verified cache hit, or generate + validate + commit on a miss."""

    def __init__(
        self,
        store: LocalDatasetStore,
        *,
        generator_factory: Callable[[str], object],
        validator: Callable[[QECBatch], None],
        provenance: dict[str, object],
        identity: CanonicalDatasetIdentity | None = None,
    ) -> None:
        self.store = store
        self.identity = identity or CanonicalDatasetIdentity()
        self.generator_factory = generator_factory
        self.validator = validator
        self.provenance = provenance

    def resolve(self, spec: DatasetSpec, attempt_id: str) -> DatasetResolution:
        key = self.identity.key_for(spec)
        artifact = self.store.find(key)
        cache_hit = artifact is not None
        if artifact is not None:
            errors = self.store.verification_errors(artifact)
            if errors:
                raise DatasetIntegrityError(
                    f"cached dataset {artifact.artifact_id} failed verification and will not be reused: "
                    + "; ".join(errors)
                )
        else:
            generator = self.generator_factory(dataset_artifact_id(key))
            compatibility = generator.compatibility(spec)
            if compatibility.support is not NoiseApproximation.EXACT:
                raise DatasetIntegrityError(
                    f"generator rejected DatasetSpec: {compatibility.reason}"
                )
            descriptor = generator.descriptor
            artifact = self.store.commit(
                spec,
                key,
                generator.generate(spec),
                validator=self.validator,
                producer_attempt_id=attempt_id,
                provenance={**self.provenance, "generator": to_jsonable(descriptor)},
            )
        instance = DatasetInstance(
            instance_id=f"{attempt_id}.dataset",
            attempt_id=attempt_id,
            dataset_artifact_id=artifact.artifact_id,
            roles=(
                DatasetRole.TRAINING,
                DatasetRole.VALIDATION,
                DatasetRole.TEST,
                DatasetRole.SCIENTIFIC_EVALUATION,
            ),
            resolved_from_cache=cache_hit,
        )
        return DatasetResolution(artifact=artifact, instance=instance, cache_hit=cache_hit)
