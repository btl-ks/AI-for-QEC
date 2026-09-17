# AGENTS.md

This is the project entry file for coding agents.

## Read First

1. `docs/RESEARCH_TOPICS.md`
2. `docs/PROJECT_DESIGN.md`
3. `docs/TECHNOLOGY_STACK.md` — external tools, formats, backends, and model choices
4. `docs/OPTIMIZATION_PLAN.md` — current phase, task IDs, and exit criteria
5. `configs/experiment.smoke.yaml`
6. `scripts/run_experiment.py`
7. `ai_qec/`

## Hard Constraints

1. Definitive academic claims must be cited from papers.
   - If a document, report, README, experiment note, or final analysis states a settled scientific conclusion, cite the source paper directly.
   - Prefer peer-reviewed papers or well-identified preprints with title, authors, venue or arXiv identifier, year, and link/DOI when available.
   - If there is no paper citation, phrase the statement as a hypothesis, design assumption, preliminary observation, or project-local result.

2. Do not fake or stub generated data.
   - Do not create empty files, zero-sample arrays, placeholder `.npz` files, or synthetic manifests just to make a pipeline appear successful.
   - Generated datasets must contain real samples, schema-compatible arrays, feature names, and a manifest with sample counts and provenance.
   - If data generation cannot produce valid non-empty data, fail loudly and leave a clear error instead of writing fake artifacts.

3. Keep experiment configuration, run outputs, checkpoints, and paper exports traceable.
   - Every completed run should have a config snapshot, logs, metrics, and a `run_manifest.json`.
   - Paper exports must use explicit allowlists; never copy the whole private project tree by default.

4. Register every new public API in `ai_qec/notebook_api.py`.
   - Import the new function or class there, add its name to `__all__`, and verify it can be imported through `ai_qec.notebook_api`.

5. Keep implementation cohesive and loosely coupled.
   - Give each function and module one clear responsibility. Keep experiment parameters independent of run metadata and artifact writing; combine them only at an explicit run boundary.
   - Pass required values explicitly, avoid hidden shared state and unrelated configuration arguments, and test each boundary through its public interface.

## Git Workflow

- Work on a branch named `phase<N>/<short-name>`; merge into `main` when the task is done.
- Start the commit subject with the plan task ID, e.g. `P0.4: forbid unknown config keys`, and tick the task in `docs/OPTIMIZATION_PLAN.md` in the same commit.
- Never commit generated data, runs, checkpoints, or paper releases; `.gitignore` excludes them.
- Formal (non-smoke) experiments should run from a clean working tree so `run_manifest.json` records an exact commit.

## Smoke Commands

```bash
python3 scripts/run_experiment.py --config configs/experiment.smoke.yaml --project-root .
python3 -m unittest discover -s tests -t .
```
