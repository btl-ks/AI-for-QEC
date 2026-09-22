from pathlib import Path
import tempfile
import unittest
from unittest import mock

import ai_qec.notebook_api as qec
from ai_qec.utils import paths


class FindProjectRootTests(unittest.TestCase):
    def test_finds_the_repository_from_a_nested_directory(self) -> None:
        root = Path.cwd().resolve()
        self.assertTrue((root / "openspec" / "config.yaml").is_file())
        self.assertEqual(qec.find_project_root(root / "paper" / "srcs"), root)
        self.assertEqual(qec.find_project_root(), root)

    def test_falls_back_to_the_editable_source_checkout(self) -> None:
        outside = Path(tempfile.mkdtemp())
        self.assertEqual(qec.find_project_root(outside), Path(paths.__file__).resolve().parents[2])

    def test_raises_when_neither_location_is_a_project(self) -> None:
        outside = Path(tempfile.mkdtemp())
        fake_module = outside / "site-packages" / "ai_qec" / "utils" / "paths.py"
        with (
            mock.patch.object(paths, "__file__", str(fake_module)),
            self.assertRaises(FileNotFoundError),
        ):
            paths.find_project_root(outside)


if __name__ == "__main__":
    unittest.main()
