# Verification

## Scope and boundaries

Implemented only the Registry-selected PyTorch training-step executor requested by this change:

- `pytorch-eager` is the reference implementation.
- `pytorch-cuda-graph` captures loss, backward, and optimizer update per bounded full batch
  signature and fails without fallback on incompatible capture/replay.
- The existing `pytorch-dataloader-h2d` pipeline still transfers each minibatch. No full training
  split is copied to, cached on, or retained by the GPU executor.
- ModelCheckpoint remains inference-only. TrainingRecoveryCheckpoint payloads record executor ID,
  implementation version, and options digest, but never graph, pool, or static-input objects.

## Environment

Real CUDA verification used:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| OS | Linux 6.6.87.2-microsoft-standard-WSL2 |
| Python | 3.12.14 |
| PyTorch | 2.14.0+cu130 |
| CUDA runtime | 13.0 |
| NumPy | 2.5.3 |
| Stim | 1.16.0 |
| PyMatching | 2.4.0 |

## Tests

Base environment, without the runtime extra:

```bash
python -m unittest discover -s tests -v
```

Result: 108 tests run, 40 expected runtime/CUDA skips, all remaining tests passed.

Complete `quantum` environment with host GPU access:

```bash
/home/zephy/miniconda3/envs/quantum/bin/python -m unittest discover -s tests -v
```

Result on the final code state: 108 tests run, 0 skipped, all passed in 8.061 s. This includes:

- exact warmup/capture/replay update counts and new static-input contents;
- alternating signatures, final smaller batches, and pre-update `max_graphs` failure;
- capture failure with failed Training Stage, no ModelCheckpoint, and no eager fallback;
- graph-safe generator advancement across replay;
- CUDA Graph recovery with a fresh capture and bitwise equality to uninterrupted training;
- one executor contract exercised by convolutional, recurrent, attention, and graph-aggregation
  PyTorch fixtures;
- existing CPU eager interruption/recovery and local runtime regressions.

After the final immutable-plan adjustment, the directly affected GPU subset was rerun:

```bash
/home/zephy/miniconda3/envs/quantum/bin/python -m unittest \
  tests.unit.test_training_step_executors \
  tests.unit.test_cuda_graph_executor \
  tests.integration.test_local_runtime.LocalRuntimeTests.test_cuda_graph_training_recovers_with_new_capture_and_same_result
```

Result: 12 tests passed in 2.645 s.

## CUDA performance evidence

Command:

```bash
/home/zephy/miniconda3/envs/quantum/bin/python \
  openspec/changes/archive/2026-09-22-add-pytorch-cuda-graph-training-step/evidence/benchmark_cuda_graph_training_step.py \
  --steps 1000 --output /tmp/ai-qec-cuda-graph-benchmark.json
```

- Archived raw result: `evidence/benchmark-results.json`
- Result checksum: `sha256:338ffeaef2cd62ec76ecb236e2e10e3a41720d428df328fc1a5405fd45d5ded6`

The L=6, batch-100, 128-hidden-unit, CD-10 microbenchmark cycled over 64 distinct minibatches:

| Measurement | Result |
|---|---:|
| eager step | 2.265 ms |
| batch-sized static-input D2D copy | 0.0237 ms |
| capture plus the capture minibatch's first launch | 7.494 ms |
| steady copy + replay | 0.350 ms |
| steady-state speedup | 6.47x |

The end-to-end Stage benchmark retained the normal per-batch H2D path and trained 100,000 samples
for three epochs (3,000 updates):

| Executor | Experiment | Training Stage | Gate |
|---|---|---:|---|
| eager | `cuda-graph-benchmark-pytorch-eager-574e818b95f2` | 7.618 s | PASS |
| CUDA Graph | `cuda-graph-benchmark-pytorch-cuda-graph-07974c5fde63` | 1.529 s | PASS |

Both Experiments resolved the same immutable DatasetArtifact
`ds-8a7f272efa07384a7711`, while training and scientific evaluation were not reused. Both produced
the independently verified model checksum
`sha256:aaca87a883ae78239275b2da6fe02c016476372aaa13119bc1218234a55f0e6c`.
CUDA Graph evidence recorded one warmup, one capture, 2,998 replays, and
`fallback_observed=false`. End-to-end Training Stage speedup was 4.98x. Detailed analysis is in
`paper/srcs/performance.md`.

## Detached notebook smoke validation

The Notebook's complete predeclared three-point `smoke` profile was started according to the
long-running execution rule. The runner was subsequently archived with the change evidence; the
equivalent replay command is:

```bash
setsid nohup bash -c \
  'echo $$ > /tmp/ai-qec-cuda-graph-smoke.pid; exec /home/zephy/miniconda3/envs/quantum/bin/python openspec/changes/archive/2026-09-22-add-pytorch-cuda-graph-training-step/evidence/run_cuda_graph_smoke.py --output /tmp/ai-qec-cuda-graph-smoke-results.json' \
  > /tmp/ai-qec-cuda-graph-smoke.log 2>&1 < /dev/null &
```

- PID file: `/tmp/ai-qec-cuda-graph-smoke.pid` (PID 22284)
- Log: `/tmp/ai-qec-cuda-graph-smoke.log`
- Archived result: `evidence/smoke-results.json`
- Result checksum: `sha256:0ac24c8bc2ccbaec6fb2bbb8ba53262359d0f542fe22c94a5a3ad1eb25812f8d`

All three Attempts reached `completed`. Dataset, training, scientific evaluation, Accuracy Gate,
and visualization Stages completed; performance was correctly not run after the smoke profile's
final Gate verdict was FAIL. No training Stage was reused. Every point recorded
`requested_executor=observed_executor=pytorch-cuda-graph`, one capture, 1,998 replays,
`fallback_observed=false`, host source residency, and CUDA-device target residency.

| Point | Experiment / Attempt | Dataset checksum | Model checksum | Gate |
|---|---|---|---|---|
| L=4, p=0.05 | `torlai-melko-2017-smoke-l4-p0.05-0cf34c3b4e49/attempt-0001` | `sha256:cc9a348f5be037b84814df462068a1d5d5fd8a2ce98a2b9947aac0f1a4d5452a` | `sha256:8ee2d8bb66f25071b442ae71b5c7cc510cb487e8c5ec313e8fa69075717c295a` | FAIL |
| L=4, p=0.10 | `torlai-melko-2017-smoke-l4-p0.10-6f783a94a265/attempt-0001` | `sha256:ddd326e88a7f89bb909ec5aaf534438b1ff5f803a26fe41720baccef86261d4a` | `sha256:dfd6c3039615c1c85b1181ee4ced3c7272e4864cd39ae2b28aff1d5f75585935` | FAIL |
| L=4, p=0.15 | `torlai-melko-2017-smoke-l4-p0.15-f17a8ce2f6fc/attempt-0001` | `sha256:239bdd0bf12f72ae1093e722e49ca69f016135d5ea3d0bd01b329d7d1ee18ce7` | `sha256:4db4e554dceeeb8e142fa61f4beae04db908735228c22b631b9cdd4efc202c68` | FAIL |

The smoke Gate failures are scientific outcomes of the deliberately reduced 10-epoch/20,000-shot
profile, not execution failures. The separate 3,000-step benchmark validation above passed its
Accuracy Gate for both eager and CUDA Graph.

## Governance validation

```bash
python -m compileall -q ai_qec tests \
  openspec/changes/archive/2026-09-22-add-pytorch-cuda-graph-training-step/evidence/benchmark_cuda_graph_training_step.py \
  openspec/changes/archive/2026-09-22-add-pytorch-cuda-graph-training-step/evidence/run_cuda_graph_smoke.py
git diff --check
openspec validate add-pytorch-cuda-graph-training-step --strict --no-interactive
openspec validate --all --strict --no-interactive
```

Results: compile and diff checks passed; the change passed strict validation; all 10 active specs
and changes passed global strict validation. `openspec/capabilities.yaml` and
`openspec/technology-catalog.yaml` also passed their JSON Schemas under jsonschema 4.26.0.
