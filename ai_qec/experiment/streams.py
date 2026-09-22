"""Named random streams derived from one master seed."""

import json

from ai_qec.utils.hashing import derive_seed

from .random_streams import RandomStreamDescriptor

ALGORITHM = "sha256(master_seed:name)[:63 bits]/numpy-pcg64"


class DerivedRandomStreams:
    """Independent numpy streams; consuming one never changes another.

    Framework generators (for example torch) seed themselves from
    ``describe(name).derived_seed`` and persist their own state.
    """

    def __init__(self, master_seed: int) -> None:
        self.master_seed = int(master_seed)
        self._generators: dict[str, object] = {}

    def describe(self, name: str) -> RandomStreamDescriptor:
        return RandomStreamDescriptor(
            name=name,
            master_seed=self.master_seed,
            derived_seed=derive_seed(self.master_seed, name),
            algorithm=ALGORITHM,
        )

    def numpy(self, name: str):
        """Return the lazily created numpy Generator for ``name``."""

        import numpy as np

        if name not in self._generators:
            self._generators[name] = np.random.Generator(
                np.random.PCG64(self.describe(name).derived_seed)
            )
        return self._generators[name]

    def snapshot(self, name: str) -> bytes:
        return json.dumps(self.numpy(name).bit_generator.state, sort_keys=True).encode("utf-8")

    def restore(self, name: str, state: bytes) -> None:
        self.numpy(name).bit_generator.state = json.loads(state.decode("utf-8"))
