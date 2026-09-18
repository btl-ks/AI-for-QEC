"""Convenience imports for the Torlai–Melko experiment notebook."""

from ai_qec.benchmarks.decoding.exact_posterior import (
    MaximumLikelihoodTieRule,
    closed_cycle_basis,
    exact_posterior_reference,
    homology_class_posteriors,
)
from ai_qec.benchmarks.decoding.toric import (
    build_toric_benchmark_report,
    logical_class_counts,
    logical_class_labels,
    mwpm_reference_recoveries,
    mwpm_tie_sensitivity,
    write_benchmark_outputs,
)
from ai_qec.data.datasets.dataset_resolution import resolve_dataset
from ai_qec.data.generators.toric_generator import sample_toric_splits
from ai_qec.data.datasets.toric_dataset import (
    SPLIT_SEED_OFFSETS,
    build_toric_dataset_manifest,
    code_from_manifest,
    load_toric_split,
    staged_dataset_dir,
    validate_toric_dataset,
    write_toric_split,
)
from ai_qec.models.decoders.classical.mwpm import ExactToricMWPMDecoder
from ai_qec.models.decoders.generative.rbm_decoder import RBMGibbsDecoder, first_compatible_chain, torch_generator_from
from ai_qec.models.registry import build_model, load_model
from ai_qec.qec.circuits.registry import build_circuit
from ai_qec.qec.codes.registry import build_code
from ai_qec.qec.codes.stabilizer import StabilizerCode, gf2_matvec
from ai_qec.qec.noise.registry import build_noise_model
from ai_qec.training.evaluation.toric_rbm import DecodingRecord, decoding_rng, rbm_decoding_metrics, save_toric_predictions
from ai_qec.training.trainers.rbm import (
    BEST_CHECKPOINT_SELECTION,
    RBMTrainingResult,
    finish_training,
    rbm_checkpoint_metadata,
    rbm_training_summary,
    reuse_training_outputs,
    run_rbm_training,
    save_rbm_checkpoint,
)
from ai_qec.utils.config import (
    config_hash,
    data_output_dir,
    first_seed,
    load_config,
    require_experiment_kind,
    resolve_experiment_spec,
    resolve_notebook_config,
    resolve_project_path,
    split_sample_counts,
    write_json,
)
from ai_qec.utils.experiment_setup import (
    ExperimentSetup,
    build_experiment,
    find_project_root,
    prepare_experiment,
    prepare_run_environment,
)
from ai_qec.utils.run_record import RunRecord, start_notebook_run
from ai_qec.utils.source_run import (
    verify_source_consistency,
    SourceRun,
    compare_with_source_predictions,
    copy_source_checkpoint,
    link_source_dataset,
    load_source_run,
    source_training_summary,
    validate_source_dataset,
)

__all__ = [
    "BEST_CHECKPOINT_SELECTION",
    "DecodingRecord",
    "ExactToricMWPMDecoder",
    "ExperimentSetup",
    "MaximumLikelihoodTieRule",
    "RBMGibbsDecoder",
    "RBMTrainingResult",
    "RunRecord",
    "SourceRun",
    "StabilizerCode",
    "SPLIT_SEED_OFFSETS",
    "build_circuit",
    "build_experiment",
    "build_code",
    "build_model",
    "build_noise_model",
    "build_toric_benchmark_report",
    "build_toric_dataset_manifest",
    "compare_with_source_predictions",
    "closed_cycle_basis",
    "code_from_manifest",
    "config_hash",
    "copy_source_checkpoint",
    "data_output_dir",
    "decoding_rng",
    "exact_posterior_reference",
    "finish_training",
    "first_compatible_chain",
    "gf2_matvec",
    "find_project_root",
    "first_seed",
    "link_source_dataset",
    "homology_class_posteriors",
    "load_config",
    "load_model",
    "load_source_run",
    "load_toric_split",
    "logical_class_counts",
    "logical_class_labels",
    "mwpm_reference_recoveries",
    "mwpm_tie_sensitivity",
    "prepare_experiment",
    "prepare_run_environment",
    "require_experiment_kind",
    "rbm_checkpoint_metadata",
    "rbm_decoding_metrics",
    "rbm_training_summary",
    "resolve_experiment_spec",
    "resolve_dataset",
    "resolve_notebook_config",
    "resolve_project_path",
    "reuse_training_outputs",
    "run_rbm_training",
    "sample_toric_splits",
    "save_rbm_checkpoint",
    "save_toric_predictions",
    "source_training_summary",
    "split_sample_counts",
    "staged_dataset_dir",
    "start_notebook_run",
    "torch_generator_from",
    "verify_source_consistency",
    "validate_source_dataset",
    "validate_toric_dataset",
    "write_benchmark_outputs",
    "write_json",
    "write_toric_split",
]
