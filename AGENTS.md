# AGENTS.md

This is the project entry file for coding agents.

## Read First

1. `docs/RESEARCH_TOPICS.md`
2. `docs/PROJECT_DESIGN.md`
3. `docs/TECHNOLOGY_STACK.md` — external tools, formats, backends, and model choices
4. `docs/OPTIMIZATION_PLAN.md` — current phase, task IDs, and exit criteria
5. Relevant `openspec/specs/` and active `openspec/changes/` — target behavior and acceptance scenarios
6. `configs/experiment.smoke.yaml`
7. `scripts/run_experiment.py`
8. `ai_qec/`

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

6. Keep target requirements and current capability state distinct.
   - OpenSpec requirements describe intended behavior and acceptance scenarios; their presence does not mean the capability is implemented.
   - Update `docs/PROJECT_DESIGN.md`, the capability registry, and `docs/OPTIMIZATION_PLAN.md` only when implementation and required evidence agree.

## OpenSpec Workflow

- For behavior changes, inspect the relevant main spec and active changes before editing code. Create or update a proposal/spec/design/tasks set when no reviewed change covers the work.
- Interpret the user's requested work scope using the intent rules below. Do not require the literal word apply when the user has already authorized implementation.
- Map implementation to the scenario-level acceptance tests and the parent task ID in `docs/OPTIMIZATION_PLAN.md`.
- Archive a change only after implementation, tests, capability status, and current-architecture documentation agree.

### Work Scope by User Intent

Treat the user's explicit action verbs as authorization boundaries:

- **Record / document / update notes**: write the requested existing information into requirements, decisions, or project documentation. Do not expand the work into design or implementation.
- **Analyze / review / investigate**: inspect the current state and report findings. Do not modify project files unless the user also asks to record or fix the findings.
- **Design / plan / propose / specify**: create or update the bounded child change, proposal, specs, design, tasks, and trace; run strict validation. Do not implement code.
- **Implement / develop / fix / refactor / complete**: authorize the full requirement-to-evidence lifecycle for the requested behavior. If no suitable child change exists, create or update it, complete and strictly validate its planning artifacts, then continue directly through implementation, automated tests, OpenSpec verify, and verification evidence. Do not ask the user to repeat the request with the word apply.
- **Continue / resume**: continue the named or unambiguous previously authorized change from its first incomplete task without requesting authorization again.

When a request contains multiple action verbs, perform the union of the explicitly requested scopes. For example, design and implement authorizes the full lifecycle, while analyze and record authorizes analysis plus documentation only. Infer scope from the requested object as well as the verb; fixing documentation does not authorize unrelated code changes.

A planning-only request remains planning-only even when the resulting artifacts are ready to apply. An implementation-authorizing request remains valid across the planning steps, context compaction, and later turns until the requested change is complete, paused, cancelled, or materially rescoped.

### Requirement-to-Evidence Chain

Every implementable OpenSpec change MUST preserve this complete chain:

1. **Requirement**: specs under the change state observable SHALL/MUST behavior and named acceptance scenarios.
2. **Functional design**: design.md maps those requirements to component boundaries, APIs, data/state models, failure semantics, compatibility or migration decisions, and mature-library/adaptor choices.
3. **Implementation tasks**: tasks.md decomposes the design into bounded implementation work; every task references its requirement/scenario, design section, and parent OPTIMIZATION_PLAN task ID when one exists.
4. **Test evidence**: every task names the automated test, verification command, or inspectable artifact that proves the scenario. A task is complete only after that evidence passes.

An umbrella change may establish a cross-project requirement baseline, but it MUST be split into bounded child changes before implementation. Capability specs stored inside one umbrella change are not separate changes. Each child change must carry its own spec delta, detailed design, implementation tasks, and test evidence so the trace remains requirement -> functional design -> implementation task -> test evidence.

### Change Delivery Gates

Use this order for every implementable child change:

1. Freeze the bounded scope in proposal.md and write observable requirements and named scenarios in the change specs.
2. Complete design.md and tasks.md with the requirement-to-evidence trace defined above.
3. Run strict OpenSpec validation before implementation. Resolve every structural or semantic validation error.
4. Start implementation only when the user intent authorizes implementation. If the original request said implement, develop, fix, refactor, complete, or an equivalent action, continue automatically after strict validation; no second apply message is required.
5. Run the named automated tests and verification commands. Check a task only after its evidence passes.
6. Run the OpenSpec verify workflow against completeness, correctness, and coherence. Resolve every CRITICAL issue before archive.
7. Create verification.md in the change directory from openspec/templates/verification.md. Record the tested commit, environment, requirement/scenario/test matrix, exact commands and results, evidence paths and hashes, limitations, and final verdict.
8. Sync the accepted delta specs, update current-state documentation, and archive only when all tasks are complete, verification.md is present, and the verification verdict permits archive.

Strict validation and implementation verification are separate gates. Strict validation checks whether proposal, specs, design, and tasks are structurally valid before implementation. Verification checks whether the resulting code and tests actually satisfy those artifacts before archive.

The local OpenSpec workflow profile must include Verify change. After changing the profile, run openspec update in the repository so the official verify skill is available to Codex. Generated local agent tool state remains excluded from project source.

### Archive Blockers

Do not archive a change when any task is unchecked, any required scenario lacks passing evidence, verification.md is missing, the verify report contains a CRITICAL issue, or current-state documentation disagrees with the implementation. Warnings and accepted limitations must be explicit in verification.md together with their effect on the archive verdict.

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
