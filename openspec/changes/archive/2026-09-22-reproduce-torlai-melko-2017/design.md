# Design

## Context

动机见 proposal.md。当前仓库只有 vendor-neutral dataclass/Protocol、空的全局 Registry 和配置预检；`test_contract_import_does_not_import_selected_frameworks` 要求导入 `ai_qec.notebook_api` 时不加载 torch、stim、pymatching、qiskit、cudaq。旧实现（被 `paper/srcs/torlai_melko_2017.ipynb` 调用的 `build_experiment`、`resolve_dataset`、`RBMGibbsDecoder` 等）不在本仓库中，只能作为算法参考。

本机 `quantum` 环境提供 Python 3.12、torch 2.14（CUDA 可用）、stim 1.16.0、pymatching 2.4.0、matplotlib。

## Goals / Non-Goals

**Goals:**

- 在不改变 Notebook 编排形状的前提下，让 `run_paper(runtime, config)` 真实执行论文实验。
- 每个可配置选择都经 `REGISTRIES_BY_PATH` 构造；Registry 只登记本 change 真正实现的工厂。
- 所有不支持的组合在创建 Experiment 目录之前失败。
- 训练恢复在 CPU 上逐位可重现，作为恢复语义的可测试证据。

**Non-Goals:**

- 不实现跨进程并发写入保护（registry 与 attempt 目录假定单写者）。
- 不实现 mid-epoch 恢复、跨 Experiment 的模型复用（旧 Notebook 的 `REUSE_CHECKPOINT_FROM_RUN`）、Gate B 与正式性能研究。
- 不为 RBM 实现 AIS 等似然估计；验证集只做一步重建 BCE 监控，不用于模型选择。

## Decisions

### 1. 显式加载实现，facade 保持轻量

`ai_qec/implementations.py` 的 `load_builtin_implementations()` 通过 `importlib` 导入各实现模块，模块顶层用 Registry 装饰器登记工厂；Python 模块缓存保证只登记一次。`LocalNotebookPlatform` 构造时调用它。第三方框架只在实现模块或函数内部导入，因此 `import ai_qec.notebook_api` 仍不加载任何选定框架，契约测试在子进程中验证这一点。

替代方案：在 `notebook_api` 导入时注册全部实现。否决原因：会让无依赖的 contract 安装失败，并违反既有导入约束。

登记的 key：`qec.code_family=toric`、`noise.family=independent-phase-flip`、`noise.adapter=stim-noise`、`dataset.generator=stim-syndrome-cpu`、`model.family=joint-error-syndrome-rbm`、`training.optimizer=sgd`、`training.scheduler=constant`、`training.loss=contrastive-divergence`、`execution.trainer_framework=pytorch`、`scientific_evaluation.baseline_decoders=pymatching-cpu-decoder`、`data_pipeline.cpu_to_gpu.technology=pytorch-dataloader-h2d`。`qiskit-circuit`、CUDA-Q 与 GPU→GPU pipeline 仍不登记。

### 2. 码定义与后端分离，并互相校验

`ToricCode` 只用 numpy 表达：边 `2·(r·L+c)+o`（o=0 水平、o=1 竖直），顶点校验矩阵 `H`（L²×2L²，对应 X_v，检测 Z 错误），两个对偶环逻辑算符（第 0 列的水平边、第 0 行的竖直边）。残余同调类由 `(e ⊕ r)` 与两个对偶环的奇偶性给出，等价于论文中的 Wilson loop 测量。

Stim adapter 用 `H` 与逻辑算符构造 code-capacity 电路（`RX`、`Z_ERROR(p)`、每个校验与逻辑的 `MPP`、`DETECTOR`、`OBSERVABLE_INCLUDE`），用 `FlipSimulator(disable_stabilizer_randomization=True)` 采样并通过 `to_numpy(output_zs=True, ...)` 同时取回物理 Z 错误帧、detector 与 observable。提交前用 `ToricCode` 重新计算 syndrome 与 observable 并逐样本比对，使 Stim 输出与 vendor-neutral 定义互相校验。

替代方案：用 DEM sampler 的 `return_errors`。否决原因：DEM 错误机制的顺序与合并规则需要额外映射，而帧模拟器直接给出每个量子比特的错误。

`qec.logical_basis` 必须为 `X`：相位翻转只影响 X 基逻辑信息，可观测量为 X 型逻辑算符；其他取值显式拒绝。

### 3. 生成版本与随机流

`dataset.generator_version` 必须等于已安装的 `stim.__version__`，在创建 Experiment 前检查。Split 与分片种子由 `sha256(f"{seed}:dataset/{split}/shard-{i:05d}")` 派生，分片固定 65 536 shots。Stim 只保证同版本同 SIMD 宽度下种子可重现，所以复用依赖已提交 Artifact 的 checksum，而不是重新生成比对。

训练与解码随机性来自 `experiment.master_seed` 的具名流：`model_init`、`training_shuffle`、`training_gibbs`、`decoding_gibbs`。`DerivedRandomStreams` 实现 `RandomStreams` Protocol，派生算法同上。

### 4. 本地 DatasetArtifact 存储

- DatasetKey = `sha256:` + 规范化 JSON（排序键、tuple→list）的 DatasetSpec 摘要；artifact id = `ds-` + 摘要前 20 位。模型、训练与执行配置不进入 DatasetSpec，因此不影响 key。
- 布局：`datasets/<id>/manifest.json`、`generation.json`、`<split>/shard-NNNNN.npz`。每个分片以 `np.packbits` 保存 `physical_errors`、`detector_events`、`observable_truth`（持久化表示 `bit-packed-cpu-buffer`）。
- manifest 记录每个分片的 sha256、每个 split 的内容摘要和 manifest 自身摘要；`datasets/registry.json` 记录 key → artifact id 与 manifest 摘要。
- 未命中：写入 `datasets/.staging/<id>-<uuid>/`，完成语义校验后 `os.rename` 为最终目录，再以临时文件 + `os.replace` 原子更新 registry。若最终目录已存在但未登记（上次在 rename 与登记之间崩溃），完整校验并确认 key 一致后才登记，否则失败。
- 命中：校验 manifest 摘要与全部分片 checksum，失败抛出 `DatasetIntegrityError`，不替换、不重新生成。
- 读取 split 时再次校验分片 checksum，并解包为 `QECBatch[np.ndarray]`（`numpy-array`、host），context 记录持久化表示与转换。

URI 一律相对于各自存储根（`datasets/` 或 `runs/`），使仓库可移动；runtime 通过 `artifact_path()` 解析。

### 5. 模型族同时拥有模型与解码算法

`MODELS["joint-error-syndrome-rbm"]` 构造 `JointErrorSyndromeRBMFamily`，负责创建/加载模型（可见层 `[e | S]`，参数 `weight`（n_h×(n_e+n_s)，W 与 U 为其列块）、`visible_bias`（b 与 d）、`hidden_bias`（c）），以及构造对应的 `RBMGibbsDecoder`。AI decoder 的 `decoder_id` 等于模型族 id。这样 Gibbs 解码参数（`model.decoding.burn_in/max_steps`）跟随模型族，而不是冒充 baseline 登记进 `scientific_evaluation.baseline_decoders`。

`model.architecture_version` 当前只接受 `"1"`。

### 6. 训练器与恢复 checkpoint

`TRAINERS["pytorch"]` 是通用 epoch/minibatch 循环；目标函数来自 `LOSSES["contrastive-divergence"]`（`loss = F(v_data) − F(v_model)` 的均值，负相为从数据出发的 k 步块 Gibbs，梯度即 CD-k），optimizer 来自 `OPTIMIZERS["sgd"]`，调度器来自 `SCHEDULERS["constant"]`。`TrainingSpec` 增加 `optimizer_parameters` 与 `loss_parameters` 映射（默认空）。

数据经 `CPU_TO_GPU_PIPELINES["pytorch-dataloader-h2d"]`：索引批 `Dataset` + `BatchSampler(RandomSampler(generator=training_shuffle))` + `pin_memory`（仅 CUDA）+ `non_blocking` 拷贝；每个 epoch 记录一次 `TransferEvidence`。`execution.num_workers` 传给 DataLoader，`cpu_count` 在训练期间设置 torch 线程数并在结束后恢复。

每个 epoch 结束写 `checkpoints/recovery/epoch-NNNN.pt`：模型、optimizer、scheduler、epoch、global step、shuffle 与 Gibbs 生成器状态、监控历史。训练结束写 `checkpoints/model/final.pt`：只有模型状态、模型身份、输入宽度与 dataset artifact id；`selected_metric="final-epoch"`（论文按固定 epoch 训练，未做基于验证集的选择）。

替代方案：按验证重建 BCE 选择最佳 epoch（旧 Notebook 的约定）。否决原因：重建误差与解码成功率不单调相关，且不是论文方法。

### 7. 解码结果表示

两个 decoder 都接收同一个 host `QECBatch[np.ndarray]`。RBM decoder 内部经 H2D pipeline 把 syndrome 送到执行设备，并在所有样本上并行运行算法 1（每个 test 样本一条独立链，分块以限制显存），每 100 步检查一次是否全部完成以提前结束。预测为 `int8` 的逻辑可观测量翻转；超时行填 `-1`，同时列入 `failed_sample_ids` 并使状态为 `timed-out`，因此无效行不可能被当作 0/1 预测。PyMatching decoder 用 `Matching.from_check_matrix(H, faults_matrix=logicals)`，均匀权重。

### 8. 科学评估与 Gate

- 评估器对每个 decoder 统计失败（预测与真值任一位不同，或按策略计入的无效样本）、Wilson 区间、超时数、残余逻辑类计数（位串 `00/10/01/11`，第 i 位对应第 i 个逻辑可观测量；绘图时映射为 h0..h3），并写出预测 `.npz`、每个 decoder 的 metrics JSON 与包含配对列联表的评估 JSON。`DecoderEvaluation` 增加 `logical_class_counts` 字段（默认空映射）。
- Gate 读取评估 JSON 中候选 decoder 与 baseline 的配对列联表 (a, b, c, d)，用 Newcombe (1998) 方法 10 计算配对差异区间，PASS 当且仅当上界 ≤ `tolerance`。Notebook 预注册 `tolerance = 0.02`（绝对 LER，项目约定，非论文内容）。
- Performance（仅 Gate PASS）：对 test split 的前 `performance.shots` 个样本，每个 decoder 预热后重复解码 `repetitions` 次，CUDA 上以 `synchronize` 为边界计时，报告批量吞吐与每 shot 平均时间及软硬件版本；报告明确标注为批量软件测量，不是实时延迟研究。

### 9. 文件型生命周期

```text
runs/<experiment_id>/                  experiment_id = <experiment.name>-<config 摘要前 12 位>
  experiment.json  config.json
  artifacts/<artifact_id>.json         ArtifactManifest，写一次
  attempts/attempt-NNNN/
    attempt.json                       status、owner(pid, host)、recovery_from、resume checkpoint
    execution_plan.json  dataset_instance.json
    stages/<stage>.json + <stage>.meta.json
    checkpoints/{recovery,model}/  evaluation/  gate/  performance/  figures/
```

- 终态（completed/failed/interrupted/partial）写入后不可再改；试图改动抛出 `AttemptStateError`。
- Stage 方法内的异常：`KeyboardInterrupt` → Stage 与 Attempt `interrupted`；其他异常 → `failed`；异常继续向上抛出。
- `start_or_recover()`：无 Attempt → 新建；最新为 running → 所属进程不存在或就是当前进程时标记 interrupted，否则拒绝；随后以最新终态 Attempt 为源构造 `RecoveryPlan`。
- 复用：dataset 每次重新解析（resolver 自身完成校验复用）；training、scientific_evaluation、performance 在源 Stage completed 且输出全部校验通过、上游输入身份一致时复用；gate 与 visualization 每次重新计算。训练未完成时，从源 Attempt 自身及其 resume 链中 epoch 最大且 checksum 通过的恢复 checkpoint 继续。
- `create_experiment(config)` 顺序：配置预检（无 unresolved、已登记选择、未知键）→ 构造全部 spec → 执行计划解析（CUDA、DDP、AMP、compile 检查）→ adapter 预检（码、噪声精确编译、generator 版本与兼容性、模型族版本、baseline 可构造）→ 才创建 Experiment 目录。

### 10. Notebook 结构

- 定义 cells：路径发现并导入 `ai_qec.notebook_api`；`CONFIG`（单点基线）、`PAPER_GRID`、按 L 固定的超参数与 `paper_config(L, p)`；`run_paper(runtime, config=CONFIG)`（保持既有顺序，baseline 取自配置）。
- `run-experiment` 标签的 cells：构造 `LocalNotebookPlatform`、逐点执行、展示单点图、图 3、图 4 与 Gate 汇总表。契约测试只执行未标记 cells；runtime 的端到端行为由独立测试以极小配置在 CPU 上验证。
- 超参数来自本 change 的原型扫描：L=4 使用 64 个隐藏单元、30 epoch；L=6 使用 128 个隐藏单元、40 epoch；均为 SGD lr=0.1、CD-10、batch 100、1e5 训练样本、burn-in 100、`max_steps` 20 000。动量 0.9 的原型先快速收敛后明显退化，因此保持论文的朴素 SGD。

## Risks / Trade-offs

- [论文网格约 40 分钟 GPU 时间] → 默认 `paper` profile，第二次 Run All 复用已完成 Stage；提供 `smoke` profile。
- [超参数未按每个 p 网格搜索] → Notebook 明确标注差异；高 p 下 RBM 劣于 MWPM 与论文结论同向，但数值不承诺一致。
- [GPU 上训练非逐位确定] → 恢复可重现性只在 CPU 上作为测试证据；GPU 恢复只保证语义（从 epoch 边界继续）。
- [Stim 种子跨版本/机器不可重现] → 生成版本精确匹配；复用基于 checksum。
- [单写者假设] → 文档说明；并发 Notebook 写同一 `datasets/` 可能竞争 registry，本 change 不处理。
- [RBM decoder 在 L=6、高 p 可能超时] → 超时按策略计为失败并单独报告，可在配置中提高 `max_steps`。

## Migration Plan

1. 新增实现模块与 Registry 登记；扩展 `DecoderEvaluation`、`TrainingSpec` 默认字段。
2. 扩展 `notebook_api` 导出；把导入无副作用、Registry 空状态的测试移到子进程。
3. 重写 Notebook 与 README；更新 technology catalog 与 capability registry。
4. 在 `quantum` 环境运行全部测试与 Notebook 论文网格，记录 verification 证据后归档。

回滚：删除新增模块与 Notebook 改动即可恢复 contracts-only 状态；运行产物位于被忽略的 `datasets/`、`runs/`。
