"""Kitaev toric code on an L x L periodic square lattice.

Qubits live on edges. Edge ``2 * (r * L + c) + o`` joins vertex ``(r, c)`` to
``(r, c + 1)`` when ``o == 0`` (horizontal) and to ``(r + 1, c)`` when ``o == 1``
(vertical). Vertex stabilizers ``X_v`` detect phase-flip (Z) errors, so the
syndrome of a Z chain ``e`` is its boundary ``H e mod 2``.

The two X-type logical operators are non-contractible dual loops: the horizontal
edges in column 0 and the vertical edges in row 0. The parity of a Z cycle on
each loop gives its homology class, which is what the paper measures with
Wilson loops.
"""

from ai_qec.qec.spec import QECSpec
from ai_qec.registries import CODES


class ToricCode:
    """Parity-check and logical-operator matrices for phase-flip decoding."""

    code_family = "toric"
    num_logicals = 2

    def __init__(self, distance: int) -> None:
        import numpy as np

        if not isinstance(distance, int) or isinstance(distance, bool) or distance < 2:
            raise ValueError(f"toric code distance must be an integer >= 2, got {distance!r}")
        self.distance = distance
        self.num_data_qubits = 2 * distance * distance
        self.num_checks = distance * distance

        parity_check = np.zeros((self.num_checks, self.num_data_qubits), dtype=np.uint8)
        for row in range(distance):
            for column in range(distance):
                vertex = row * distance + column
                for edge in (
                    self.edge_index(row, column, 0),
                    self.edge_index(row, column - 1, 0),
                    self.edge_index(row, column, 1),
                    self.edge_index(row - 1, column, 1),
                ):
                    parity_check[vertex, edge] = 1

        logical_operators = np.zeros((self.num_logicals, self.num_data_qubits), dtype=np.uint8)
        for row in range(distance):
            logical_operators[0, self.edge_index(row, 0, 0)] = 1
        for column in range(distance):
            logical_operators[1, self.edge_index(0, column, 1)] = 1

        parity_check.setflags(write=False)
        logical_operators.setflags(write=False)
        self.parity_check = parity_check
        self.logical_operators = logical_operators

    def edge_index(self, row: int, column: int, orientation: int) -> int:
        size = self.distance
        return 2 * ((row % size) * size + (column % size)) + orientation

    def plaquette_checks(self):
        """Z-type plaquette stabilizers; unused by phase-flip decoding, kept for audits."""

        import numpy as np

        plaquettes = np.zeros((self.num_checks, self.num_data_qubits), dtype=np.uint8)
        for row in range(self.distance):
            for column in range(self.distance):
                for edge in (
                    self.edge_index(row, column, 0),
                    self.edge_index(row + 1, column, 0),
                    self.edge_index(row, column, 1),
                    self.edge_index(row, column + 1, 1),
                ):
                    plaquettes[row * self.distance + column, edge] = 1
        return plaquettes

    def syndrome(self, chains):
        """Vertex syndrome ``H e mod 2`` for each row of ``chains``."""

        return _binary_product(chains, self.parity_check)

    def logical_flips(self, chains):
        """Parity of each chain on the two dual loops (observable flips)."""

        return _binary_product(chains, self.logical_operators)

    def describe(self) -> dict[str, object]:
        return {
            "code_family": self.code_family,
            "distance": self.distance,
            "num_data_qubits": self.num_data_qubits,
            "num_checks": self.num_checks,
            "num_logicals": self.num_logicals,
            "edge_convention": "2*(r*L+c)+o; o=0 horizontal, o=1 vertical",
        }


def _binary_product(chains, matrix):
    import numpy as np

    chains = np.asarray(chains, dtype=np.uint8)
    if chains.ndim != 2 or chains.shape[1] != matrix.shape[1]:
        raise ValueError(f"expected chains with shape (n, {matrix.shape[1]}), got {chains.shape}")
    return ((chains.astype(np.int64) @ matrix.T.astype(np.int64)) % 2).astype(np.uint8)


@CODES.register("toric")
def build_toric_code(*, spec: QECSpec) -> ToricCode:
    if spec.code_family != ToricCode.code_family:
        raise ValueError(f"toric factory cannot build code_family={spec.code_family!r}")
    return ToricCode(spec.distance)
