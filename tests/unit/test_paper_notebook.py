from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import ai_qec.notebook_api as qec


NOTEBOOK_PATH = Path("paper/srcs/ai_for_qec_workflow.ipynb")


def execute_notebook_cells() -> dict[str, object]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "paper_contract_test"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        code = compile(source, f"{NOTEBOOK_PATH}#cell-{index}", "exec")
        with redirect_stdout(io.StringIO()):
            exec(code, namespace)
    return namespace


class PaperNotebookTests(unittest.TestCase):
    def test_notebook_is_valid_v4_json(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(notebook["nbformat_minor"], 5)
        self.assertTrue(any(cell["cell_type"] == "code" for cell in notebook["cells"]))

    def test_clean_code_cell_execution_only_defines_contract_workflow(self) -> None:
        namespace = execute_notebook_cells()
        self.assertIn("CONFIG", namespace)
        self.assertIn("run_paper", namespace)
        self.assertIsNone(namespace["RUNTIME"])

    def test_gate_fail_skips_performance_and_finishes(self) -> None:
        namespace = execute_notebook_cells()
        run_paper = namespace["run_paper"]
        events: list[str] = []

        class RecordingRun:
            attempt_id = "attempt-test"

            def resolve_dataset(self):
                events.append("resolve_dataset")
                return object()

            def train(self, dataset):
                events.append("train")
                return object()

            def evaluate_accuracy(self, model, dataset, baselines):
                events.append("evaluate_accuracy")
                self.assert_baseline = tuple(baselines)
                return object()

            def check_accuracy_gate(self, result):
                events.append("check_accuracy_gate")
                return SimpleNamespace(decision=qec.GateDecision.FAIL)

            def evaluate_performance(self, model, dataset):
                raise AssertionError("performance MUST be skipped after Gate FAIL")

            def visualize(self, scientific_result, acceptance, performance_result):
                events.append("visualize")
                self.assert_performance = performance_result
                return ()

            def finish(self):
                events.append("finish")

        run = RecordingRun()

        class RecordingExperiment:
            def start_or_recover(self):
                events.append("start_or_recover")
                return run

        class RecordingPlatform:
            def create_experiment(self, config):
                events.append("create_experiment")
                self.config = config
                return RecordingExperiment()

        platform = RecordingPlatform()
        run_paper(platform)

        self.assertIsInstance(platform, qec.NotebookPlatform)
        self.assertEqual(run.assert_baseline, ("mwpm",))
        self.assertIsNone(run.assert_performance)
        self.assertEqual(
            events,
            [
                "create_experiment",
                "start_or_recover",
                "resolve_dataset",
                "train",
                "evaluate_accuracy",
                "check_accuracy_gate",
                "visualize",
                "finish",
            ],
        )


if __name__ == "__main__":
    unittest.main()
