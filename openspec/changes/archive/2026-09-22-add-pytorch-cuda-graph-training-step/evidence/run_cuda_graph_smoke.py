"""Run the notebook's declared smoke grid and persist auditable change evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _definitions(notebook: Path) -> dict[str, object]:
    document = json.loads(notebook.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "cuda_graph_smoke"}
    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] != "code" or "run-experiment" in cell["metadata"].get("tags", []):
            continue
        code = compile("".join(cell["source"]), f"{notebook}#cell-{index}", "exec")
        exec(code, namespace)  # noqa: S102 - execute version-controlled Notebook definitions
    return namespace


def main() -> None:
    import ai_qec.notebook_api as qec
    from ai_qec.utils.hashing import read_json, write_json_atomic

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project = qec.find_project_root()
    notebook = project / "paper" / "srcs" / "ai_for_qec_workflow.ipynb"
    namespace = _definitions(notebook)
    runtime = qec.LocalNotebookPlatform(project)
    records = []
    for point in namespace["GRIDS"]["smoke"]:
        config = point.config
        experiment = runtime.create_experiment(config)
        run = experiment.start_or_recover()
        dataset = run.resolve_dataset()
        model = run.train(dataset)
        scientific = run.evaluate_accuracy(
            model=model,
            dataset=dataset,
            baselines=tuple(config["scientific_evaluation"]["baseline_decoders"]),
        )
        acceptance = run.check_accuracy_gate(scientific)
        performance = None
        if acceptance.decision is qec.GateDecision.PASS:
            performance = run.evaluate_performance(model=model, dataset=dataset)
        figures = run.visualize(
            scientific_result=scientific,
            acceptance=acceptance,
            performance_result=performance,
        )
        run.finish()
        training = read_json(run.directory / "stages" / "training.meta.json")
        records.append(
            {
                "grid": dict(point.values),
                "experiment_id": run.experiment_id,
                "attempt_id": run.attempt_id,
                "attempt_status": run.status.value,
                "dataset_artifact_id": dataset.artifact_id,
                "dataset_checksum": dataset.checksum,
                "model_artifact_id": model.artifact_id,
                "model_checksum": model.checksum,
                "scientific_result_artifact_id": scientific.result_artifact_id,
                "gate": acceptance.decision.value,
                "performance_artifact_id": None if performance is None else performance.artifact_id,
                "figure_artifact_ids": [item.artifact_id for item in figures],
                "step_executor": training["step_executor"],
                "transfer_evidence": training["transfer_evidence"],
            }
        )
    write_json_atomic(
        args.output,
        {
            "profile": "smoke",
            "notebook": str(notebook.relative_to(project)),
            "points": records,
        },
    )
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
