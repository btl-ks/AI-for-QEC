"""Integrity smoke tests that execute the real experiment runner."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "run_experiment.py"


def smoke_config(tmp: Path) -> Path:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "experiment.smoke.yaml").read_text(encoding="utf-8"))
    config["data"].update(train_samples=80, validation_samples=30, test_samples=30, output_dir=str(tmp / "datasets"), dataset_id="integrity")
    config.pop("paper_export")
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


class PipelineSmokeTest(unittest.TestCase):
    def run_runner(self, config: Path, run_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(RUNNER), "--config", str(config), "--project-root", str(PROJECT_ROOT), "--run-dir", str(run_dir), "--allow-dirty", *extra], capture_output=True, text=True)

    def test_real_runner_records_complete_immutable_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            config = smoke_config(tmp)
            run_dir = tmp / "run"
            proc = self.run_runner(config, run_dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "success")
            self.assertTrue(manifest["dataset"]["config_hash_matches_run"])
            self.assertRegex(manifest["git"]["commit"], r"^[0-9a-f]{40}$")
            self.assertIn("numpy", manifest["environment"]["packages"])
            self.assertTrue(all(step["status"] == "success" for step in manifest["steps"]))
            self.assertTrue((run_dir / "checkpoints" / "best.npz").is_file())
            self.assertTrue((run_dir / "resolved_plan.json").is_file())
            self.assertEqual(self.run_runner(config, run_dir).returncode, 2)  # collision never overwrites a run

    def test_zero_step_and_partial_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            config = smoke_config(tmp)
            self.assertEqual(self.run_runner(config, tmp / "zero", "--only", "missing").returncode, 2)
            partial = tmp / "partial"
            proc = self.run_runner(config, partial, "--only", "generate_data")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            manifest = json.loads((partial / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "partial")

    def test_child_failure_is_terminally_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            config_path = smoke_config(tmp)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["flow"] = [{"id": "failure", "script": "scripts/adapt.py", "args": [], "outputs": [{"path": "${RUN_DIR}/never", "artifact_type": "none"}]}]
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            run_dir = tmp / "failed"
            proc = self.run_runner(config_path, run_dir)
            self.assertNotEqual(proc.returncode, 0)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["steps"][0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
