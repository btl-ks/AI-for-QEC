# Verification

## Scope

本验证证明：`paired-relative-non-inferiority` 规则按 `LER_AI − (1+δ)·LER_baseline` 的 MOVER 配对区间判定，δ = 0 时与原 Newcombe 区间逐位一致，在边界上的 PASS 频率接近单侧名义水平；训练、科学评估与性能评估按 Stage 复用键跨 Experiment 复用，Gate 与可视化始终重新计算；论文 Notebook 改为相对 15% 后，22 个新 Experiment 不训练任何 epoch 即完成。它不证明相对 15% 是科学上最优的容差（该值是在看到绝对 0.02 的结果后选定的），也不证明复用键覆盖代码版本或运行环境的变化。

环境：WSL2 Linux，NVIDIA GeForce RTX 4060 Laptop GPU，conda 环境 `quantum`（Python 3.12.14）；无运行时依赖的基础环境为 Python 3.13.12。

## Evidence

### 单元与集成测试

命令：

```bash
/home/zephy/miniconda3/envs/quantum/bin/python -m unittest discover -s tests
python3 -m unittest discover -s tests
python3 -m compileall -q ai_qec tests
```

结果：`quantum` 环境 91 个测试全部通过（7.2 s）；基础环境 91 个测试通过，其中 31 个需要运行时依赖的测试被跳过；compileall 通过。新增的关键行为证据：

- `test_statistics.test_unit_scale_reproduces_persisted_newcombe_intervals_exactly`：L=6 网格 p=0.05 与 p=0.15 的配对表，在 `scale=1` 下得到的区间与旧实现持久化在 `acceptance.json` 中的浮点值逐位相等。
- `test_statistics.test_scaled_contrast_follows_the_relative_margin`：scale=1.15 时 p=0.05 的上界 > 0，p=0.15 的上界 < 0；scale ≤ 0 报错。
- `test_statistics.test_pass_rate_on_the_relative_margin_matches_the_one_sided_level`：在 `p_AI = 1.15·p_base` 的配对多项分布下，10 000 shots、各 4000 次重复，PASS 比例为 2.65%（低 p：0.030/0.0275/0.020/0.9225）和 2.43%（高 p：0.450/0.125/0.050/0.375），单侧名义水平为 2.5%。
- `test_pipeline_and_gate.test_relative_rule_scales_the_margin_with_the_baseline`：同一张 p=0.05 配对表在绝对 0.02 下 PASS、在相对 15% 下 FAIL；p=0.15 的表相反。证据文件包含 `rule_parameters = {relative_tolerance: 0.15, scale: 1.15}`、`ratio_estimate = 431/384` 与 `method = mover-wilson-linear-contrast`。
- `test_pipeline_and_gate.test_negative_relative_tolerance_produces_no_decision`，以及 `test_local_runtime.test_preflight_failures_create_nothing` 的 `negative relative tolerance` 用例：负 δ 在 Gate 中抛出 `AccuracyGateError` 且不写证据；在预检中报告 `accuracy_gate.tolerance`，不创建 `runs/` 或 `datasets/`。
- `test_stage_reuse.test_only_the_declared_fields_are_ignored`：对三个 Stage 各改动 8 个字段，只有删除表中的字段不改变复用键；`training.epochs`、`experiment.master_seed`、`execution.device` 改变所有键；`model.decoding` 与 `scientific_evaluation` 只不影响训练键；`performance` 只影响性能键。输入引用或 Stage 名不同时键不同；dataset、Gate 与 visualization 没有复用键。
- `test_local_runtime.test_gate_only_change_reuses_stages_across_experiments`：只改 Gate 时，新 Experiment 的 `attempt-0001` 在训练、科学评估与性能评估三个 Stage 上都记录了 `reused_from_experiment`、`reused_from` 与 `reuse_key`。该 Attempt 没有 `checkpoints/` 与 `evaluation/` 目录，模型与科学结果对象和来源相等；Gate 在本 Experiment 登记新证据，并引用来源的评估 Artifact。再次运行得到 `attempt-0002`，它在同 Experiment 内复用，只记录 `reused_from`，没有继承 `reused_from_experiment`。
- `test_local_runtime.test_cross_experiment_reuse_follows_the_reuse_key`：只改实验名时三个 Stage 都复用；改 `training.epochs` 为 4 时不复用，训练 4 个 epoch。
- `test_local_runtime.test_modified_source_model_is_not_reused_across_experiments`：来源 `final.pt` 被篡改后不复用，重新训练 3 个 epoch。
- 原有测试不变并通过：同 Experiment 恢复复用、`KeyboardInterrupt` 后逐位续训、篡改后从恢复 checkpoint 继续。

### 论文 Notebook（paper profile，相对 15%）

命令（`quantum` 环境，cwd 为 `paper/srcs`，nbclient 执行全部 cells）：

```python
NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "paper/srcs"}}).execute()
```

结果：92 s，8 个 cells 均无错误输出。`runs/` 从 42 个 Experiment 增加到 64 个，即 22 个新 Experiment，各自只有 `attempt-0001`。

- dataset：22 次 `reused verified`。
- training：22 次 `reused verified model from <来源 Experiment>/<Attempt>`，没有任何 `trained N epoch(s)` 日志行。
- scientific_evaluation：22 次跨 Experiment 复用。
- accuracy_gate：22 次重新计算，21 PASS、1 FAIL，与实现前用持久化配对表离线计算的结果一致。
- performance：21 个 PASS 点中，12 个复用了绝对 0.02 下已有的性能 Stage；其余 9 个在 0.02 下为 FAIL、从未测过性能，本次新测。L=6、p=0.05 在 0.02 下 PASS、在 15% 下 FAIL，本次不运行性能。

样例：`torlai-melko-2017-paper-l6-p0.12-549c0583a627/attempt-0001` 的训练与科学评估复用自 `…-l6-p0.12-c8047f2dc98a/attempt-0002`，即硬中断后恢复的那个模型；它与一次性训练的旧格式模型逐位相同，见 `add-config-grid` 的 verification。

```text
 L     p |               RBM P_fail timeouts |              MWPM P_fail | gate
 4  0.05 |  0.0754 [0.0704, 0.0807]        0 |  0.0763 [0.0713, 0.0817] | PASS
 4  0.06 |  0.1167 [0.1106, 0.1231]        1 |  0.1168 [0.1107, 0.1232] | PASS
 4  0.07 |  0.1597 [0.1527, 0.1670]        0 |  0.1561 [0.1491, 0.1633] | PASS
 4  0.08 |  0.2044 [0.1966, 0.2124]        0 |  0.1960 [0.1883, 0.2039] | PASS
 4  0.09 |  0.2450 [0.2367, 0.2535]        0 |  0.2366 [0.2284, 0.2450] | PASS
 4  0.10 |  0.2953 [0.2864, 0.3043]        0 |  0.2820 [0.2733, 0.2909] | PASS
 4  0.11 |  0.3414 [0.3322, 0.3508]        0 |  0.3314 [0.3222, 0.3407] | PASS
 4  0.12 |  0.4005 [0.3909, 0.4101]        0 |  0.3773 [0.3678, 0.3868] | PASS
 4  0.13 |  0.4406 [0.4309, 0.4504]        0 |  0.4121 [0.4025, 0.4218] | PASS
 4  0.14 |  0.4833 [0.4735, 0.4931]        0 |  0.4584 [0.4487, 0.4682] | PASS
 4  0.15 |  0.5266 [0.5168, 0.5364]        0 |  0.5022 [0.4924, 0.5120] | PASS
 6  0.05 |  0.0431 [0.0393, 0.0473]       88 |  0.0384 [0.0348, 0.0423] | FAIL
 6  0.06 |  0.0712 [0.0663, 0.0764]       84 |  0.0715 [0.0666, 0.0767] | PASS
 6  0.07 |  0.1077 [0.1018, 0.1139]       76 |  0.1076 [0.1017, 0.1138] | PASS
 6  0.08 |  0.1611 [0.1540, 0.1684]       66 |  0.1611 [0.1540, 0.1684] | PASS
 6  0.09 |  0.2151 [0.2072, 0.2233]       59 |  0.2146 [0.2067, 0.2228] | PASS
 6  0.10 |  0.2759 [0.2672, 0.2847]       75 |  0.2731 [0.2645, 0.2819] | PASS
 6  0.11 |  0.3401 [0.3309, 0.3494]       81 |  0.3363 [0.3271, 0.3456] | PASS
 6  0.12 |  0.4048 [0.3952, 0.4145]      135 |  0.3891 [0.3796, 0.3987] | PASS
 6  0.13 |  0.4708 [0.4610, 0.4806]      299 |  0.4437 [0.4340, 0.4535] | PASS
 6  0.14 |  0.5346 [0.5248, 0.5444]      407 |  0.5057 [0.4959, 0.5155] | PASS
 6  0.15 |  0.5760 [0.5663, 0.5857]      876 |  0.5435 [0.5337, 0.5532] | PASS
```

P_fail 与绝对 0.02 时逐行相同，因为科学评估是复用的；只有 gate 列变化。L=6、p=0.05 的 rationale：比值 1.122，`LER_RBM − 1.15·LER_MWPM = −0.0011`，95% MOVER 区间 [−0.0049, +0.0027]，上界 > 0，判定 FAIL。

绝对 0.02 下的 22 个 Experiment 及其 `acceptance.json` 保留未改动。

### OpenSpec

```bash
openspec validate --all --strict --no-interactive
```

结果：全部通过。

## Gaps

- 复用键不含代码版本与运行环境（与 Experiment 身份的现有语义一致）。修复训练代码后，需要修改配置字段（例如 `model.architecture_version`）才能强制重训。
- 相对 15% 是事后重新登记的容差；Notebook 的结论部分如实说明了这一点。
- 跨 Experiment 查找逐个扫描 `runs/`，没有索引；本地 64 个 Experiment 时不可察觉，规模化时需要索引。
