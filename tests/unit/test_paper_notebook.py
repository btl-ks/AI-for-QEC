from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import ai_qec.notebook_api as qec


NOTEBOOK_PATH = Path("paper/srcs/ai_for_qec_workflow.ipynb")
EXPERIMENT_TAG = "run-experiment"


def notebook_cells() -> list[dict]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))["cells"]


def execute_definition_cells() -> dict[str, object]:
    """Execute every code cell that is not tagged to launch experiments."""

    namespace: dict[str, object] = {"__name__": "paper_contract_test"}
    for index, cell in enumerate(notebook_cells()):
        if cell["cell_type"] != "code" or EXPERIMENT_TAG in cell["metadata"].get("tags", []):
            continue
        code = compile("".join(cell["source"]), f"{NOTEBOOK_PATH}#cell-{index}", "exec")
        with redirect_stdout(io.StringIO()):
            exec(code, namespace)
    return namespace


class RecordingRun:
    attempt_id = "attempt-test"

    def __init__(self, events: list[str], decision: qec.GateDecision) -> None:
        self.events = events
        self.decision = decision

    def resolve_dataset(self):
        self.events.append("resolve_dataset")
        return object()

    def train(self, dataset):
        self.events.append("train")
        return object()

    def evaluate_accuracy(self, model, dataset, baselines):
        self.events.append("evaluate_accuracy")
        self.baselines = tuple(baselines)
        return object()

    def check_accuracy_gate(self, result):
        self.events.append("check_accuracy_gate")
        return SimpleNamespace(decision=self.decision)

    def evaluate_performance(self, model, dataset):
        if self.decision is not qec.GateDecision.PASS:
            raise AssertionError("performance MUST be skipped after Gate FAIL")
        self.events.append("evaluate_performance")
        return object()

    def visualize(self, scientific_result, acceptance, performance_result):
        self.events.append("visualize")
        self.performance = performance_result
        return ()

    def finish(self):
        self.events.append("finish")


class RecordingPlatform:
    def __init__(self, run: RecordingRun) -> None:
        self.run = run

    def create_experiment(self, config):
        self.run.events.append("create_experiment")
        self.config = config
        run = self.run

        class RecordingExperiment:
            def start_or_recover(self):
                run.events.append("start_or_recover")
                return run

        return RecordingExperiment()


class PaperNotebookTests(unittest.TestCase):
    def test_notebook_is_valid_v4_json(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(notebook["nbformat_minor"], 5)
        self.assertTrue(any(cell["cell_type"] == "code" for cell in notebook["cells"]))
        self.assertTrue(
            all(
                cell.get("outputs", []) == []
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
        )

    def test_definition_cells_create_no_runtime(self) -> None:
        namespace = execute_definition_cells()
        for name in ("CONFIG", "run_paper", "GRIDS", "GRID"):
            self.assertIn(name, namespace)
        self.assertNotIn("RUNTIME", namespace)
        self.assertNotIn("RESULTS", namespace)

    def test_only_tagged_cells_construct_the_runtime(self) -> None:
        for index, cell in enumerate(notebook_cells()):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            tagged = EXPERIMENT_TAG in cell["metadata"].get("tags", [])
            launches = "LocalNotebookPlatform(" in source or "run_paper(RUNTIME" in source
            if launches:
                self.assertTrue(
                    tagged, f"cell {index} launches experiments without the {EXPERIMENT_TAG} tag"
                )

    def test_paper_grid_matches_figure_3(self) -> None:
        namespace = execute_definition_cells()
        grid = namespace["GRID"]
        self.assertEqual(namespace["PROFILE"], "paper")
        self.assertIs(grid, namespace["GRIDS"]["paper"])
        self.assertEqual(len(grid), 22)
        self.assertEqual({point.values["L"] for point in grid}, {4, 6})
        rates = sorted({point.values["p"] for point in grid})
        self.assertEqual(rates, [round(0.05 + 0.01 * i, 2) for i in range(11)])
        point = next(item for item in grid if item.values == {"L": 6, "p": 0.13})
        self.assertEqual(point.config["qec"]["distance"], 6)
        self.assertEqual(point.config["noise"]["parameters"]["physical_error_rate"], 0.13)
        self.assertEqual(point.config["model"]["parameters"]["hidden_units"], 128)
        self.assertEqual(point.config["training"]["epochs"], 40)
        self.assertEqual(point.config["experiment"]["name"], "torlai-melko-2017-paper-l6-p0.13")
        self.assertEqual(
            namespace["CONFIG"]["qec"]["distance"], 4, "the grid must not mutate CONFIG"
        )
        self.assertEqual(len({item.config["experiment"]["name"] for item in grid}), len(grid))
        smoke = namespace["GRIDS"]["smoke"]
        self.assertEqual([item.values["p"] for item in smoke], [0.05, 0.10, 0.15])
        self.assertEqual(smoke[0].config["dataset"]["train_samples"], 20_000)
        self.assertEqual(smoke[0].config["training"]["epochs"], 10)

    def test_config_passes_structural_validation(self) -> None:
        from ai_qec.experiment.config import parse_experiment_config

        namespace = execute_definition_cells()
        qec.validate_no_unresolved(namespace["CONFIG"])
        parsed = parse_experiment_config(namespace["CONFIG"])
        self.assertEqual(
            parsed.spec.accuracy_gate.comparison_rule, "paired-relative-non-inferiority"
        )
        self.assertEqual(parsed.spec.accuracy_gate.tolerance, 0.15)
        self.assertEqual(parsed.spec.dataset.qec.code_family, "toric")

    def test_gate_fail_skips_performance_and_finishes(self) -> None:
        run_paper = execute_definition_cells()["run_paper"]
        events: list[str] = []
        run = RecordingRun(events, qec.GateDecision.FAIL)
        platform = RecordingPlatform(run)
        run_paper(platform)

        self.assertIsInstance(platform, qec.NotebookPlatform)
        self.assertEqual(run.baselines, ("pymatching-cpu-decoder",))
        self.assertIsNone(run.performance)
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

    def test_gate_pass_runs_performance_before_visualization(self) -> None:
        namespace = execute_definition_cells()
        events: list[str] = []
        run = RecordingRun(events, qec.GateDecision.PASS)
        platform = RecordingPlatform(run)
        point = next(item for item in namespace["GRID"] if item.values == {"L": 6, "p": 0.05})
        namespace["run_paper"](platform, point.config)
        self.assertEqual(platform.config["qec"]["distance"], 6)
        self.assertEqual(
            events[-4:], ["check_accuracy_gate", "evaluate_performance", "visualize", "finish"]
        )
        self.assertIsNotNone(run.performance)


if __name__ == "__main__":
    unittest.main()
