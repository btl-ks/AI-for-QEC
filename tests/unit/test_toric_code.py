import unittest

from tests.helpers import requires_numpy


@requires_numpy
class ToricCodeTests(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        from ai_qec.qec.codes.toric import ToricCode

        self.np = np
        self.code = ToricCode(4)

    def test_lattice_degrees_and_stabilizer_commutation(self) -> None:
        np = self.np
        code = self.code
        self.assertEqual(code.parity_check.shape, (16, 32))
        self.assertTrue(np.all(code.parity_check.sum(axis=1) == 4))
        self.assertTrue(np.all(code.parity_check.sum(axis=0) == 2))
        plaquettes = code.plaquette_checks()
        self.assertFalse(np.any((code.parity_check.astype(int) @ plaquettes.T) % 2))
        self.assertFalse(np.any((code.logical_operators.astype(int) @ plaquettes.T) % 2))

    def test_homology_of_cycles(self) -> None:
        np = self.np
        code = self.code
        plaquette = code.plaquette_checks()[5][None, :]
        self.assertFalse(code.syndrome(plaquette).any())
        self.assertEqual(code.logical_flips(plaquette).tolist(), [[0, 0]])

        horizontal_loop = np.zeros((1, code.num_data_qubits), dtype=np.uint8)
        horizontal_loop[0, [code.edge_index(0, column, 0) for column in range(4)]] = 1
        vertical_loop = np.zeros_like(horizontal_loop)
        vertical_loop[0, [code.edge_index(row, 0, 1) for row in range(4)]] = 1
        self.assertFalse(code.syndrome(horizontal_loop).any())
        self.assertEqual(code.logical_flips(horizontal_loop).tolist(), [[1, 0]])
        self.assertEqual(code.logical_flips(vertical_loop).tolist(), [[0, 1]])
        self.assertEqual(
            code.logical_flips(horizontal_loop ^ vertical_loop ^ plaquette).tolist(), [[1, 1]]
        )

    def test_single_error_flags_its_two_endpoints(self) -> None:
        np = self.np
        code = self.code
        error = np.zeros((1, code.num_data_qubits), dtype=np.uint8)
        error[0, code.edge_index(3, 3, 0)] = 1
        syndrome = code.syndrome(error)[0]
        self.assertEqual(np.flatnonzero(syndrome).tolist(), [3 * 4 + 0, 3 * 4 + 3])

    def test_rejects_invalid_distance_and_shape(self) -> None:
        from ai_qec.qec.codes.toric import ToricCode

        with self.assertRaises(ValueError):
            ToricCode(1)
        with self.assertRaises(ValueError):
            self.code.syndrome(self.np.zeros((2, 5), dtype=self.np.uint8))


if __name__ == "__main__":
    unittest.main()
