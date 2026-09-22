import hashlib
import importlib
from pathlib import Path
import tomllib
import unittest

import ai_qec
import ai_qec.notebook_api as qec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PACKAGE = PROJECT_ROOT / "ai_qec"
PUBLIC_EXPORTS_SHA256 = "f2516ff66e76ca2ca8efb36ac367a18a797e387492081bea63f0008578a1d9b5"


class PackagingContractTests(unittest.TestCase):
    def test_source_uses_a_single_root_layout_tree(self) -> None:
        self.assertTrue((SOURCE_PACKAGE / "notebook_api.py").is_file())
        self.assertFalse((PROJECT_ROOT / "src" / "ai_qec").exists())

    def test_source_and_tests_have_no_init_files(self) -> None:
        init_files = sorted(
            path.relative_to(PROJECT_ROOT)
            for root in (SOURCE_PACKAGE, PROJECT_ROOT / "tests")
            for path in root.rglob("__init__.py")
        )
        self.assertEqual(init_files, [])

    def test_setuptools_discovers_only_the_ai_qec_namespace(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
            pyproject = tomllib.load(handle)

        self.assertEqual(pyproject["build-system"]["build-backend"], "setuptools.build_meta")
        package_find = pyproject["tool"]["setuptools"]["packages"]["find"]
        self.assertEqual(package_find["where"], ["."])
        self.assertEqual(package_find["include"], ["ai_qec*"])
        self.assertIs(package_find["namespaces"], True)
        self.assertIsNone(ai_qec.__spec__.origin)
        self.assertIsNone(ai_qec.__file__)

    def test_repository_root_import_resolves_to_the_source_tree(self) -> None:
        self.assertEqual(Path(qec.__file__).resolve(), SOURCE_PACKAGE / "notebook_api.py")

    def test_public_facade_preserves_export_snapshot(self) -> None:
        digest = hashlib.sha256("\0".join(sorted(qec.__all__)).encode()).hexdigest()
        self.assertEqual(len(qec.__all__), 130)
        self.assertEqual(digest, PUBLIC_EXPORTS_SHA256)
        for name in qec.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(qec, name))

    def test_nested_leaf_modules_import_without_initializers(self) -> None:
        module_names = (
            "ai_qec.data.datasets.artifact",
            "ai_qec.evaluation.scientific.result",
            "ai_qec.experiment.local",
            "ai_qec.models.decoders.protocol",
            "ai_qec.training.executors.protocol",
        )
        for module_name in module_names:
            with self.subTest(module_name=module_name):
                self.assertEqual(importlib.import_module(module_name).__name__, module_name)


if __name__ == "__main__":
    unittest.main()
