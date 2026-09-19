"""Strict experiment configuration loading and resolution.

The project deliberately treats a YAML file as an executable contract. A
configuration is rejected when it names an unavailable capability, contains
unknown fields, or describes an ambiguous flow. This prevents a successful
run from silently meaning something other than the configuration requested.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = 1
_VARIABLE_RE = re.compile(r"\$\{([^}]+)\}")
_ALLOWED_VARIABLES = {"PROJECT_ROOT", "CONFIG", "RUN_DIR", "RUN_ID", "PYTHON", "DATASET_DIR"}
_CONDA_ENV_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ConfigValidationError(ValueError):
    """Raised for invalid configurations and unavailable capabilities."""


@dataclass(frozen=True)
class ResolvedOutputContract:
    """An artifact declared by one flow step."""

    path: str
    artifact_type: str
    required: bool
    non_empty: bool
    schema_version: int | None
    hash_policy: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "path": self.path,
            "artifact_type": self.artifact_type,
            "required": self.required,
            "non_empty": self.non_empty,
            "schema_version": self.schema_version,
            "hash_policy": self.hash_policy,
        }


@dataclass(frozen=True)
class ResolvedFlowStep:
    """A validated, configuration-owned subprocess invocation."""

    id: str
    enabled: bool
    script: str | None
    module: str | None
    args: tuple[str, ...]
    outputs: tuple[ResolvedOutputContract, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        data: dict[str, Any] = {
            "id": self.id,
            "enabled": self.enabled,
            "args": list(self.args),
            "outputs": [contract.to_dict() for contract in self.outputs],
        }
        # Follow this branch when self.script is not None.
        if self.script is not None:
            data["script"] = self.script
        # Follow this branch when self.module is not None.
        if self.module is not None:
            data["module"] = self.module
        return data


@dataclass(frozen=True)
class ResolvedExperimentSpec:
    """The single validated representation used by the experiment runner."""

    config: dict[str, Any]
    flow: tuple[ResolvedFlowStep, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical configuration with a parsed flow."""
        data = deepcopy(self.config)
        data["flow"] = [step.to_dict() for step in self.flow]
        return data


def _fail(location: str, message: str) -> None:
    raise ConfigValidationError(f"{location}: {message}")


def _mapping(value: Any, location: str) -> dict[str, Any]:
    # Follow this branch when not isinstance(value, dict).
    if not isinstance(value, dict):
        _fail(location, "must be a mapping")
    return value


def _list(value: Any, location: str) -> list[Any]:
    # Follow this branch when not isinstance(value, list).
    if not isinstance(value, list):
        _fail(location, "must be a list")
    return value


def _string(value: Any, location: str, *, non_empty: bool = True) -> str:
    # Follow this branch when not isinstance(value, str).
    if not isinstance(value, str):
        _fail(location, "must be a string")
    # Follow this branch when non_empty and (not value.strip()).
    if non_empty and not value.strip():
        _fail(location, "must not be empty")
    return value


def _boolean(value: Any, location: str) -> bool:
    # Follow this branch when not isinstance(value, bool).
    if not isinstance(value, bool):
        _fail(location, "must be true or false")
    return value


def _integer(value: Any, location: str, *, minimum: int | None = None) -> int:
    # Follow this branch when isinstance(value, bool) or not isinstance(value, int).
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(location, "must be an integer")
    # Follow this branch when minimum is not None and value < minimum.
    if minimum is not None and value < minimum:
        _fail(location, f"must be >= {minimum}")
    return value


def _number(value: Any, location: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    # Follow this branch when isinstance(value, bool) or not isinstance(value, (int, float)).
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(location, "must be a finite number")
    number = float(value)
    # Follow this branch when not math.isfinite(number).
    if not math.isfinite(number):
        _fail(location, "must be finite")
    # Follow this branch when minimum is not None and number < minimum.
    if minimum is not None and number < minimum:
        _fail(location, f"must be >= {minimum}")
    # Follow this branch when maximum is not None and number > maximum.
    if maximum is not None and number > maximum:
        _fail(location, f"must be <= {maximum}")
    return number


def _keys(mapping: Mapping[str, Any], location: str, *, allowed: set[str], required: set[str] = frozenset()) -> None:
    unknown = sorted(set(mapping) - allowed)
    # Follow this branch when unknown.
    if unknown:
        _fail(location, f"unknown field(s): {', '.join(unknown)}")
    missing = sorted(required - set(mapping))
    # Follow this branch when missing.
    if missing:
        _fail(location, f"missing required field(s): {', '.join(missing)}")


def _validate_template(value: str, location: str) -> None:
    for variable in _VARIABLE_RE.findall(value):
        # Follow this branch when variable not in _ALLOWED_VARIABLES.
        if variable not in _ALLOWED_VARIABLES:
            _fail(location, f"uses unsupported variable ${{{variable}}}")


def _validate_experiment(value: Any) -> None:
    data = _mapping(value, "experiment")
    _keys(data, "experiment", allowed={"name", "title", "description", "owner", "tags"}, required={"name"})
    _string(data["name"], "experiment.name")
    for name in ("title", "description", "owner"):
        # Follow this branch when name in data.
        if name in data:
            _string(data[name], f"experiment.{name}")
    # Follow this branch when 'tags' in data.
    if "tags" in data:
        for index, tag in enumerate(_list(data["tags"], "experiment.tags")):
            _string(tag, f"experiment.tags[{index}]")


def _validate_topic(value: Any) -> None:
    data = _mapping(value, "topic")
    _keys(data, "topic", allowed={"direction", "subtopic", "name"})
    for name, item in data.items():
        _string(item, f"topic.{name}")


def _validate_reproducibility(value: Any) -> None:
    data = _mapping(value, "reproducibility")
    _keys(
        data,
        "reproducibility",
        allowed={"seeds", "deterministic", "save_environment", "save_git_commit", "require_clean_worktree"},
    )
    seeds = _list(data["seeds"], "reproducibility.seeds")
    # Follow this branch when not seeds.
    if not seeds:
        _fail("reproducibility.seeds", "must contain at least one seed")
    for index, seed in enumerate(seeds):
        _integer(seed, f"reproducibility.seeds[{index}]")
    # Follow this branch when len(seeds) != 1.
    if len(seeds) != 1:
        _fail("reproducibility.seeds", "multiple seeds are not implemented; configure exactly one seed")
    for name in ("deterministic", "save_environment", "save_git_commit", "require_clean_worktree"):
        _boolean(data[name], f"reproducibility.{name}")
    # Follow this branch when not data['deterministic'].
    if not data["deterministic"]:
        _fail("reproducibility.deterministic", "false is not supported by the current deterministic toy backend")


def _validate_execution(value: Any) -> None:
    """Validate the optional interpreter selection for all flow subprocesses."""
    data = _mapping(value, "execution")
    _keys(data, "execution", allowed={"conda_env"}, required={"conda_env"})
    conda_env = _string(data["conda_env"], "execution.conda_env")
    # Follow this branch when not _CONDA_ENV_NAME_RE.fullmatch(conda_env).
    if not _CONDA_ENV_NAME_RE.fullmatch(conda_env):
        _fail(
            "execution.conda_env",
            "must be a Conda environment name containing only letters, digits, '.', '_' or '-'",
        )


CODE_CAPACITY_CODES = {"toric_code", "surface_code", "repetition_code"}


def _validate_qec(value: Any) -> None:
    data = _mapping(value, "qec")
    _keys(data, "qec", allowed={"code", "distance", "rounds", "task"}, required={"code", "distance", "rounds", "task"})
    code = _string(data["code"], "qec.code")
    # Follow this branch when code not in {'surface_code', 'repetition_code', 'toric_code'}.
    if code not in {"surface_code", "repetition_code", "toric_code"}:
        _fail("qec.code", f"unsupported code '{code}'")
    distance = _integer(data["distance"], "qec.distance", minimum=1)
    rounds = _integer(data["rounds"], "qec.rounds", minimum=1)
    task = _string(data["task"], "qec.task")
    # Follow this branch when code == 'toric_code'.
    if code == "toric_code":
        # Follow this branch when distance < 2.
        if distance < 2:
            _fail("qec.distance", "toric lattice size must be at least 2")
        # Follow this branch when rounds != 1.
        if rounds != 1:
            _fail("qec.rounds", "toric code-capacity experiments require exactly one ideal round")
        # Follow this branch when task != 'code_capacity'.
        if task != "code_capacity":
            _fail("qec.task", "toric_code requires 'code_capacity'")
        return
    # Follow this branch when task == 'code_capacity'.
    if task == "code_capacity":
        # surface and repetition carry a real parity-check matrix, so they decode too
        if rounds != 1:
            _fail("qec.rounds", "code-capacity experiments require exactly one ideal round")
        # Follow this branch when code == 'surface_code' and (distance < 3 or distance % 2 == 0).
        if code == "surface_code" and (distance < 3 or distance % 2 == 0):
            _fail("qec.distance", "rotated surface-code distance must be an odd integer of at least 3")
        # Follow this branch when code == 'repetition_code' and distance < 2.
        if code == "repetition_code" and distance < 2:
            _fail("qec.distance", "repetition-code distance must be at least 2")
        return
    # Follow this branch when distance % 2 == 0.
    if distance % 2 == 0:
        _fail("qec.distance", "must be odd")
    # Follow this branch when task != 'memory'.
    if task != "memory":
        _fail("qec.task", "only 'memory' is implemented for the toy backends")


def _validate_rate_range(value: Any, location: str) -> None:
    data = _mapping(value, location)
    _keys(data, location, allowed={"distribution", "min", "max"}, required={"distribution", "min", "max"})
    # Follow this branch when the compound condition is satisfied.
    if _string(data["distribution"], f"{location}.distribution") != "uniform":
        _fail(f"{location}.distribution", "only 'uniform' is implemented")
    low = _number(data["min"], f"{location}.min", minimum=0.0, maximum=1.0)
    high = _number(data["max"], f"{location}.max", minimum=0.0, maximum=1.0)
    # Follow this branch when low > high.
    if low > high:
        _fail(location, "min must be <= max")


def _validate_noise(value: Any) -> None:
    data = _mapping(value, "noise")
    # Follow this branch when 'model' in data.
    if "model" in data:
        _keys(data, "noise", allowed={"model", "p_error"}, required={"model", "p_error"})
        # Follow this branch when _string(data['model'], 'noise.model') != 'phase_flip'.
        if _string(data["model"], "noise.model") != "phase_flip":
            _fail("noise.model", "only 'phase_flip' is implemented for toric code capacity")
        _number(data["p_error"], "noise.p_error", minimum=0.0, maximum=1.0)
        return
    _keys(data, "noise", allowed={"target", "nuisance", "controls"}, required={"target", "nuisance", "controls"})
    target = _mapping(data["target"], "noise.target")
    _keys(target, "noise.target", allowed={"name", "values"}, required={"name", "values"})
    # Follow this branch when _string(target['name'], 'noise.target.name') != 'crosstalk'.
    if _string(target["name"], "noise.target.name") != "crosstalk":
        _fail("noise.target.name", "only 'crosstalk' is implemented")
    values = _list(target["values"], "noise.target.values")
    # Follow this branch when not values.
    if not values:
        _fail("noise.target.values", "must contain at least one value")
    for index, item in enumerate(values):
        _number(item, f"noise.target.values[{index}]", minimum=0.0, maximum=1.0)

    nuisance = _mapping(data["nuisance"], "noise.nuisance")
    _keys(nuisance, "noise.nuisance", allowed={"depolarizing", "measurement"}, required={"depolarizing", "measurement"})
    _validate_rate_range(nuisance["depolarizing"], "noise.nuisance.depolarizing")
    _validate_rate_range(nuisance["measurement"], "noise.nuisance.measurement")

    controls = _mapping(data["controls"], "noise.controls")
    _keys(controls, "noise.controls", allowed={"match_total_syndrome_density"}, required={"match_total_syndrome_density"})
    _boolean(controls["match_total_syndrome_density"], "noise.controls.match_total_syndrome_density")


def _validate_data(value: Any) -> None:
    data = _mapping(value, "data")
    _keys(
        data,
        "data",
        allowed={"generator", "train_samples", "validation_samples", "test_samples", "dataset_id", "output_dir", "batch_size", "synthetic_crosstalk_gain", "preprocessing"},
        required={"generator", "train_samples", "validation_samples", "test_samples", "dataset_id", "output_dir", "preprocessing"},
    )
    generator = _string(data["generator"], "data.generator").lower()
    # Follow this branch when generator not in {'toy_synthetic', 'toric_code_capacity'}.
    if generator not in {"toy_synthetic", "toric_code_capacity"}:
        # Follow this branch when generator == 'stim'.
        if generator == "stim":
            _fail("data.generator", "'stim' is not implemented; select an implemented generator")
        _fail("data.generator", f"unsupported backend '{generator}'")
    for name in ("train_samples", "validation_samples", "test_samples"):
        _integer(data[name], f"data.{name}", minimum=1)
    _string(data["dataset_id"], "data.dataset_id")
    _validate_template(_string(data["output_dir"], "data.output_dir"), "data.output_dir")
    _integer(data.get("batch_size", 8192), "data.batch_size", minimum=1)
    # Follow this branch when generator == 'toy_synthetic' and 'synthetic_crosstalk_gain' in data.
    if generator == "toy_synthetic" and "synthetic_crosstalk_gain" in data:
        _number(data["synthetic_crosstalk_gain"], "data.synthetic_crosstalk_gain", minimum=0.0)
    # Follow this branch when generator != 'toy_synthetic' and 'synthetic_crosstalk_gain' in data.
    if generator != "toy_synthetic" and "synthetic_crosstalk_gain" in data:
        _fail("data.synthetic_crosstalk_gain", "is only valid for toy_synthetic")
    preprocessing = _mapping(data["preprocessing"], "data.preprocessing")
    _keys(preprocessing, "data.preprocessing", allowed={"representation"}, required={"representation"})
    representation = _string(preprocessing["representation"], "data.preprocessing.representation")
    # Choose the first expression when generator == 'toric_code_capacity'; otherwise use the fallback.
    expected = "error_syndrome" if generator == "toric_code_capacity" else "detector_summary"
    # Follow this branch when representation != expected.
    if representation != expected:
        _fail("data.preprocessing.representation", f"{generator} requires '{expected}'")


def _validate_model(value: Any) -> None:
    data = _mapping(value, "model")
    _keys(data, "model", allowed={"implementation", "hidden_units", "init_width"}, required={"implementation"})
    implementation = _string(data["implementation"], "model.implementation")
    # Follow this branch when implementation == 'joint_error_syndrome_rbm'.
    if implementation == "joint_error_syndrome_rbm":
        _keys(data, "model", allowed={"implementation", "hidden_units", "init_width"}, required={"implementation", "hidden_units", "init_width"})
        _integer(data["hidden_units"], "model.hidden_units", minimum=1)
        _number(data["init_width"], "model.init_width", minimum=1e-12)
        return
    # Follow this branch when implementation == 'linear_detector_summary_baseline'.
    if implementation == "linear_detector_summary_baseline":
        # Follow this branch when set(data) != {'implementation'}.
        if set(data) != {"implementation"}:
            _fail("model", "linear_detector_summary_baseline accepts only 'implementation'")
        return
    # Follow this branch when implementation == 'transformer_decoder'.
    if implementation == "transformer_decoder":
        _fail("model.implementation", "'transformer_decoder' is not implemented")
    _fail("model.implementation", f"unregistered implementation '{implementation}'")


def _validate_training(value: Any) -> None:
    data = _mapping(value, "training")
    trainer = _string(data.get("trainer"), "training.trainer")
    # Follow this branch when trainer == 'rbm_cd'.
    if trainer == "rbm_cd":
        _keys(
            data,
            "training",
            allowed={"trainer", "epochs", "batch_size", "learning_rate", "cd_steps", "weight_decay", "decoder", "device"},
            required={"trainer", "epochs", "batch_size", "learning_rate", "cd_steps", "weight_decay", "decoder"},
        )
        _integer(data["epochs"], "training.epochs", minimum=1)
        _integer(data["batch_size"], "training.batch_size", minimum=1)
        _number(data["learning_rate"], "training.learning_rate", minimum=1e-12)
        _integer(data["cd_steps"], "training.cd_steps", minimum=1)
        _number(data["weight_decay"], "training.weight_decay", minimum=0.0)
        # Follow this branch when 'device' in data and data['device'] not in ('cpu', 'cuda').
        if "device" in data and data["device"] not in ("cpu", "cuda"):
            _fail("training.device", "must be 'cpu' or 'cuda'")
        decoder = _mapping(data["decoder"], "training.decoder")
        _keys(decoder, "training.decoder", allowed={"burn_in", "max_steps", "parallel_chains"}, required={"burn_in", "max_steps"})
        burn_in = _integer(decoder["burn_in"], "training.decoder.burn_in", minimum=0)
        max_steps = _integer(decoder["max_steps"], "training.decoder.max_steps", minimum=1)
        # Follow this branch when 'parallel_chains' in decoder.
        if "parallel_chains" in decoder:
            _integer(decoder["parallel_chains"], "training.decoder.parallel_chains", minimum=1)
        # Follow this branch when burn_in >= max_steps.
        if burn_in >= max_steps:
            _fail("training.decoder", "burn_in must be smaller than max_steps")
        return
    _keys(data, "training", allowed={"trainer", "optimizer"}, required={"trainer", "optimizer"})
    # Follow this branch when trainer != 'multitask_ridge'.
    if trainer != "multitask_ridge":
        _fail("training.trainer", "only 'multitask_ridge' and 'rbm_cd' are implemented")
    optimizer = _mapping(data["optimizer"], "training.optimizer")
    _keys(optimizer, "training.optimizer", allowed={"name", "weight_decay"}, required={"name", "weight_decay"})
    # Follow this branch when _string(optimizer['name'], 'training.optimizer.name') != 'ridge'.
    if _string(optimizer["name"], "training.optimizer.name") != "ridge":
        _fail("training.optimizer.name", "only 'ridge' is implemented")
    _number(optimizer["weight_decay"], "training.optimizer.weight_decay", minimum=0.0)


def _validate_outputs(value: Any) -> None:
    data = _mapping(value, "outputs")
    _keys(data, "outputs", allowed={"runs_root"}, required={"runs_root"})
    _validate_template(_string(data["runs_root"], "outputs.runs_root"), "outputs.runs_root")


def _parse_outputs(value: Any, location: str) -> tuple[ResolvedOutputContract, ...]:
    contracts: list[ResolvedOutputContract] = []
    seen_paths: set[str] = set()
    for index, raw_contract in enumerate(_list(value, location)):
        contract_location = f"{location}[{index}]"
        data = _mapping(raw_contract, contract_location)
        _keys(data, contract_location, allowed={"path", "artifact_type", "required", "non_empty", "schema_version", "hash_policy"}, required={"path", "artifact_type"})
        path = _string(data["path"], f"{contract_location}.path")
        _validate_template(path, f"{contract_location}.path")
        # Follow this branch when path in seen_paths.
        if path in seen_paths:
            _fail(contract_location, "duplicates an output path in the same step")
        seen_paths.add(path)
        artifact_type = _string(data["artifact_type"], f"{contract_location}.artifact_type")
        required = _boolean(data.get("required", True), f"{contract_location}.required")
        non_empty = _boolean(data.get("non_empty", True), f"{contract_location}.non_empty")
        schema_version = data.get("schema_version")
        # Follow this branch when schema_version is not None.
        if schema_version is not None:
            schema_version = _integer(schema_version, f"{contract_location}.schema_version", minimum=1)
        hash_policy = _string(data.get("hash_policy", "sha256"), f"{contract_location}.hash_policy")
        # Follow this branch when hash_policy not in {'none', 'sha256'}.
        if hash_policy not in {"none", "sha256"}:
            _fail(f"{contract_location}.hash_policy", "must be 'none' or 'sha256'")
        contracts.append(ResolvedOutputContract(path, artifact_type, required, non_empty, schema_version, hash_policy))
    # Follow this branch when not contracts.
    if not contracts:
        _fail(location, "must declare at least one output contract")
    return tuple(contracts)


def _validate_script_path(script: str, project_root: Path | None, location: str) -> None:
    _validate_template(script, location)
    # Return early when project_root is None or '${' in script.
    if project_root is None or "${" in script:
        return
    path = Path(script)
    # Follow this branch when not path.is_absolute().
    if not path.is_absolute():
        path = project_root / path
    try:
        path.resolve().relative_to(project_root.resolve())
    except ValueError:
        _fail(location, "must resolve under project_root")
    # Follow this branch when not path.is_file().
    if not path.is_file():
        _fail(location, f"does not exist: {path}")


def _parse_flow(value: Any, project_root: Path | None) -> tuple[ResolvedFlowStep, ...]:
    raw_steps = _list(value, "flow")
    # Follow this branch when not raw_steps.
    if not raw_steps:
        _fail("flow", "must contain at least one step")
    steps: list[ResolvedFlowStep] = []
    ids: set[str] = set()
    for index, raw_step in enumerate(raw_steps):
        location = f"flow[{index}]"
        step = _mapping(raw_step, location)
        _keys(step, location, allowed={"id", "enabled", "script", "module", "args", "outputs"}, required={"id", "outputs"})
        step_id = _string(step["id"], f"{location}.id")
        # Follow this branch when step_id in ids.
        if step_id in ids:
            _fail(f"{location}.id", f"duplicates '{step_id}'")
        ids.add(step_id)
        enabled = _boolean(step.get("enabled", True), f"{location}.enabled")
        script = step.get("script")
        module = step.get("module")
        # Follow this branch when (script is None) == (module is None).
        if (script is None) == (module is None):
            _fail(location, "requires exactly one of 'script' or 'module'")
        # Follow this branch when script is not None.
        if script is not None:
            script = _string(script, f"{location}.script")
            _validate_script_path(script, project_root, f"{location}.script")
        # Follow this branch when module is not None.
        if module is not None:
            module = _string(module, f"{location}.module")
            _validate_template(module, f"{location}.module")
        raw_args = _list(step.get("args", []), f"{location}.args")
        args = tuple(_string(item, f"{location}.args[{arg_index}]", non_empty=False) for arg_index, item in enumerate(raw_args))
        outputs = _parse_outputs(step["outputs"], f"{location}.outputs")
        steps.append(ResolvedFlowStep(step_id, enabled, script, module, args, outputs))
    return tuple(steps)


def _validate_paper_export(value: Any) -> None:
    data = _mapping(value, "paper_export")
    _keys(data, "paper_export", allowed={"enabled", "package_name", "scope", "public_config_keys", "include", "reproduce_command"}, required={"enabled", "package_name", "scope", "include", "reproduce_command"})
    _boolean(data["enabled"], "paper_export.enabled")
    _string(data["package_name"], "paper_export.package_name")
    # Follow this branch when 'scope' in data.
    if "scope" in data:
        _string(data["scope"], "paper_export.scope")
    # Follow this branch when 'reproduce_command' in data.
    if "reproduce_command" in data:
        _validate_template(_string(data["reproduce_command"], "paper_export.reproduce_command"), "paper_export.reproduce_command")
    # Follow this branch when 'public_config_keys' in data.
    if "public_config_keys" in data:
        for index, key in enumerate(_list(data["public_config_keys"], "paper_export.public_config_keys")):
            _string(key, f"paper_export.public_config_keys[{index}]")
    for index, raw_item in enumerate(_list(data["include"], "paper_export.include")):
        location = f"paper_export.include[{index}]"
        item = _mapping(raw_item, location)
        _keys(item, location, allowed={"from", "to"}, required={"from", "to"})
        _validate_template(_string(item["from"], f"{location}.from"), f"{location}.from")
        _validate_template(_string(item["to"], f"{location}.to"), f"{location}.to")


def _validate_reserved(value: Any) -> None:
    data = _mapping(value, "reserved")
    _keys(data, "reserved", allowed={"reason", "fields"}, required={"reason", "fields"})
    _string(data["reason"], "reserved.reason")
    _mapping(data["fields"], "reserved.fields")


def _validate_cross_fields(config: dict[str, Any], flow: tuple[ResolvedFlowStep, ...]) -> None:
    # Keep only values that satisfy step.enabled and step.script is not None.
    scripts = {Path(step.script).name for step in flow if step.enabled and step.script is not None}
    # Follow this branch when the compound condition is satisfied.
    if {"generate_data.py", "train.py", "evaluate.py", "benchmark.py"} & scripts:
        required = {"qec", "noise", "data", "model", "training"}
        missing = sorted(required - set(config))
        # Follow this branch when missing.
        if missing:
            _fail("config", f"data pipeline requires section(s): {', '.join(missing)}")
        generator = config["data"]["generator"]
        implementation = config["model"]["implementation"]
        trainer = config["training"]["trainer"]
        # Follow this branch when generator == 'toric_code_capacity'.
        if generator == "toric_code_capacity":
            # the name is kept so existing content-addressed datasets stay valid; any
            # code that supplies a parity-check matrix can use this generator
            if config["qec"]["code"] not in CODE_CAPACITY_CODES or config["qec"]["task"] != "code_capacity":
                _fail("qec", f"toric_code_capacity requires task='code_capacity' and one of {sorted(CODE_CAPACITY_CODES)}")
            # Follow this branch when config['noise'].get('model') != 'phase_flip'.
            if config["noise"].get("model") != "phase_flip":
                _fail("noise", "toric_code_capacity requires phase_flip noise")
            # Follow this branch when the compound condition is satisfied.
            if config["data"]["preprocessing"]["representation"] != "error_syndrome":
                _fail("data.preprocessing.representation", "toric_code_capacity requires raw error_syndrome data")
            # Follow this branch when implementation != 'joint_error_syndrome_rbm' or trainer != 'rbm_cd'.
            if implementation != "joint_error_syndrome_rbm" or trainer != "rbm_cd":
                _fail("model", "toric_code_capacity requires joint_error_syndrome_rbm with rbm_cd")
        # Use this alternative branch when generator == 'toy_synthetic'.
        elif generator == "toy_synthetic":
            # Follow this branch when the compound condition is satisfied.
            if config["data"]["preprocessing"]["representation"] != "detector_summary":
                _fail("data.preprocessing.representation", "does not match the linear detector-summary baseline")
            # Follow this branch when the compound condition is satisfied.
            if implementation != "linear_detector_summary_baseline" or trainer != "multitask_ridge":
                _fail("model", "toy_synthetic requires linear_detector_summary_baseline with multitask_ridge")


def _normalise_defaults(config: dict[str, Any]) -> None:
    repro = config.setdefault("reproducibility", {})
    # Return early when not isinstance(repro, dict).
    if not isinstance(repro, dict):
        return
    repro.setdefault("seeds", [0])
    repro.setdefault("deterministic", True)
    repro.setdefault("save_environment", True)
    repro.setdefault("save_git_commit", True)
    repro.setdefault("require_clean_worktree", True)


def resolve_experiment_spec(
    config: Mapping[str, Any], project_root: str | Path | None = None, *, allow_empty_flow: bool = False,
) -> ResolvedExperimentSpec:
    """Validate a raw config; notebook-owned stages may opt out of a script flow."""
    # Follow this branch when not isinstance(config, Mapping).
    if not isinstance(config, Mapping):
        _fail("config", "must be a mapping")
    resolved = deepcopy(dict(config))
    _normalise_defaults(resolved)
    # Choose the first expression when allow_empty_flow; otherwise use the fallback.
    _keys(
        resolved,
        "config",
        allowed={"schema_version", "experiment", "topic", "reproducibility", "execution", "qec", "noise", "data", "model", "training", "outputs", "flow", "paper_export", "reserved"},
        required={"schema_version", "experiment", "data"} if allow_empty_flow else {"schema_version", "experiment", "data", "flow"},
    )
    # Follow this branch when the compound condition is satisfied.
    if _integer(resolved["schema_version"], "schema_version", minimum=1) != SCHEMA_VERSION:
        _fail("schema_version", f"unsupported version; expected {SCHEMA_VERSION}")
    _validate_experiment(resolved["experiment"])
    _validate_reproducibility(resolved["reproducibility"])
    # Follow this branch when 'execution' in resolved.
    if "execution" in resolved:
        _validate_execution(resolved["execution"])
    # Follow this branch when 'topic' in resolved.
    if "topic" in resolved:
        _validate_topic(resolved["topic"])
    # Follow this branch when 'qec' in resolved.
    if "qec" in resolved:
        _validate_qec(resolved["qec"])
    # Follow this branch when 'noise' in resolved.
    if "noise" in resolved:
        _validate_noise(resolved["noise"])
    _validate_data(resolved["data"])
    # Follow this branch when 'model' in resolved.
    if "model" in resolved:
        _validate_model(resolved["model"])
    # Follow this branch when 'training' in resolved.
    if "training" in resolved:
        _validate_training(resolved["training"])
    # Follow this branch when 'outputs' in resolved.
    if "outputs" in resolved:
        _validate_outputs(resolved["outputs"])
    # Follow this branch when 'paper_export' in resolved.
    if "paper_export" in resolved:
        _validate_paper_export(resolved["paper_export"])
    # Follow this branch when 'reserved' in resolved.
    if "reserved" in resolved:
        _validate_reserved(resolved["reserved"])
    # Choose the first expression when project_root is not None; otherwise use the fallback.
    root = Path(project_root).resolve() if project_root is not None else None
    raw_flow = resolved.get("flow", [])
    # Choose the first expression when allow_empty_flow and raw_flow == []; otherwise use the fallback.
    flow = () if allow_empty_flow and raw_flow == [] else _parse_flow(raw_flow, root)
    _validate_cross_fields(resolved, flow)
    resolved["flow"] = [step.to_dict() for step in flow]
    return ResolvedExperimentSpec(config=resolved, flow=flow)


def require_experiment_kind(
    experiment_config: Mapping[str, Any], *, generator: str, model: str,
) -> None:
    """Require the experiment parameters to select the expected workflow."""
    data = experiment_config.get("data")
    model_config = experiment_config.get("model")
    # Reject this state when not isinstance(data, Mapping) or data.get('generator') != generator.
    if not isinstance(data, Mapping) or data.get("generator") != generator:
        raise ValueError(f"实验要求 data.generator={generator!r}")
    # Reject this state when the invalid compound condition is detected.
    if not isinstance(model_config, Mapping) or model_config.get("implementation") != model:
        raise ValueError(f"实验要求 model.implementation={model!r}")


def resolve_notebook_config(
    run_config: Mapping[str, Any], experiment_config: Mapping[str, Any], *, project_root: str | Path,
) -> dict[str, Any]:
    """Assemble and validate the complete config at the notebook run boundary."""
    repeated = set(run_config) & set(experiment_config)
    # Reject this state when repeated.
    if repeated:
        raise ValueError(f"配置键不能同时出现在两组中：{', '.join(sorted(repeated))}")
    return resolve_experiment_spec(
        {**run_config, **experiment_config}, project_root=project_root, allow_empty_flow=True,
    ).config


def load_experiment_spec(path: str | Path, project_root: str | Path | None = None) -> ResolvedExperimentSpec:
    """Load YAML and resolve it before callers write experiment artifacts."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    # Reject this state when not isinstance(data, dict).
    if not isinstance(data, dict):
        raise ConfigValidationError(f"Config must be a mapping: {config_path}")
    return resolve_experiment_spec(data, project_root=project_root)


def load_config(path: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    """Load and strictly validate a config, returning its canonical dictionary."""
    return load_experiment_spec(path, project_root=project_root).config


def resolve_project_path(project_root: str | Path, value: str | Path) -> Path:
    """Resolve relative paths against the project root."""
    path = Path(value)
    # Return early when path.is_absolute().
    if path.is_absolute():
        return path
    return Path(project_root).resolve() / path


def data_output_dir(config: dict[str, Any], project_root: str | Path) -> Path:
    """Return the immutable data directory derived from the generation spec."""
    return resolve_project_path(project_root, config["data"]["output_dir"]) / dataset_directory_name(config)


def generation_spec(config: dict[str, Any]) -> dict[str, Any]:
    """Return every effective input that establishes toy dataset identity."""
    data = config["data"]
    # Choose the first expression when data['generator'] == 'toric_code_capacity'; otherwise use the fallback.
    return {
        "schema_version": 1,
        "config_hash": config_hash(config),
        "seed": first_seed(config),
        "batch_size": int(data.get("batch_size", 8192)),
        "sample_counts": split_sample_counts(config),
        "requested_generator": data["generator"],
        "generator_version": "toric_code_capacity_v1" if data["generator"] == "toric_code_capacity" else "toy_synthetic_v1",
    }


def generation_hash(config: dict[str, Any]) -> str:
    """Hash the effective generation spec, including its temporary batch identity."""
    return hashlib.sha256(json.dumps(generation_spec(config), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def dataset_directory_name(config: dict[str, Any]) -> str:
    """Create a stable human-readable, content-addressed dataset directory name."""
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(config["data"]["dataset_id"])).strip("._") or "dataset"
    return f"{safe_id}-{generation_hash(config)[:16]}"


def split_sample_counts(config: dict[str, Any]) -> dict[str, int]:
    """Extract train/validation/test sample counts from a validated config."""
    data = config["data"]
    return {"train": int(data["train_samples"]), "validation": int(data["validation_samples"]), "test": int(data["test_samples"])}


def first_seed(config: dict[str, Any], default: int = 0) -> int:
    """Return the first configured seed, falling back to a default."""
    seeds = config.get("reproducibility", {}).get("seeds", [])
    # Return early when seeds.
    if seeds:
        return int(seeds[0])
    return default


def config_hash(config: dict[str, Any]) -> str:
    """Hash a normalized, validated config so artifacts record provenance."""
    payload = yaml.safe_dump(config, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write indented UTF-8 JSON, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
