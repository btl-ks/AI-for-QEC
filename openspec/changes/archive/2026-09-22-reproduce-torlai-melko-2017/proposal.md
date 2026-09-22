# Proposal

## Why

`paper/srcs/ai_for_qec_workflow.ipynb` 只固定了标准研究顺序，仓库没有任何满足 `NotebookPlatform` 的 runtime，因此无法用这条工作流复现任何论文。Torlai & Melko (PRL 119, 030501, 2017) 的 RBM 神经解码器是项目第一篇目标论文：它同时需要数据集复用、训练、AI 与经典解码器的公平比较和可恢复的 Attempt，正好覆盖平台优先交付的 Dataset reuse、Attempt recovery、Scientific Accuracy Gate 与 Training/Execution 分离。

## What Changes

- 新增一个本地单进程 runtime，实现 `NotebookPlatform`/`NotebookExperiment`/`NotebookRun`，让 `run_paper(runtime, config)` 端到端执行 Configure → Start/Recover → Resolve Dataset → Train → Scientific Evaluation → Accuracy Gate → 条件式 Performance → Visualize → Finish。
- 实现 vendor-neutral 的 toric code（L×L 环面、顶点 syndrome、两个逻辑类）与独立相位翻转噪声定义，并在 Registry 中登记。
- 实现 Stim CPU code-capacity adapter：Stim noise compiler（相位翻转精确编译，其他噪声显式拒绝）与 `stim-syndrome-cpu` 数据生成器，输出物理错误、syndrome 与逻辑可观测量。
- 实现本地 DatasetArtifact 存储：内容寻址 DatasetKey、bit-packed 分片、原子提交、命中前完整性校验、每个 Attempt 独立 DatasetInstance。
- 实现 PyTorch Dataset/DataLoader CPU→GPU pipeline（pinned memory、non-blocking，并记录 TransferEvidence）。
- 实现论文模型与算法：联合 `[e | S]` RBM、CD-k/SGD 训练器（epoch 边界 TrainingRecoveryCheckpoint 与独立 ModelCheckpoint）、syndrome 钳制的 Gibbs 解码器（论文算法 1）。
- 实现 PyMatching CPU MWPM baseline 与 PyTorch RBM decoder 的统一 DecodeRequest/DecodeResult。
- 实现科学评估（Wilson 区间 LER、超时计入失败、残余逻辑类分布）与配对非劣效 Accuracy Gate，以及 Gate PASS 后的解码吞吐量测量和图表。
- 实现文件型 Experiment/Attempt/Stage/Artifact 生命周期：终态 Attempt 不可变，`start_or_recover()` 创建新 Attempt，只复用校验通过的已完成 Stage，训练从最近一个校验通过的恢复 checkpoint 继续。
- 重写 `ai_for_qec_workflow.ipynb` 为 Torlai–Melko 复现：保持 `run_paper` 标准顺序，按网格（L ∈ {4, 6}，p ∈ {0.05, …, 0.15}）逐点执行，复现论文图 3（P_fail–p）与图 4（同调类直方图）。
- 为 `DecoderEvaluation` 增加残余逻辑类计数字段，为 `TrainingSpec` 增加 optimizer/loss 参数映射（均带默认值，向后兼容）。
- 更新 technology catalog 与 capability registry，使已实现的 adapter 和能力状态与证据一致。

非目标：不实现 CUDA-Q GPU 生成、Qiskit circuit、circuit-level/多轮噪声、surface code、DDP/AMP/torch.compile、Gate B、正式性能研究、FPGA/ASIC/QPU 或跨 Experiment 的模型复用；不承诺与论文逐点数值一致（论文为每个 p 单独网格搜索超参数，本复现按 L 固定超参数）。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `qec/dataset-pipeline`: 增加 toric code-capacity 相位翻转数据生成、生成版本精确匹配和提交前码语义一致性校验。
- `qec/decoder-workbench`: 增加 syndrome 钳制 Gibbs 解码（算法 1）与均匀权重 MWPM baseline 的可观测行为。
- `research/training-contracts`: 增加 CD-k 联合 RBM 训练、epoch 边界恢复可重现性和不支持执行选项的具体拒绝行为。
- `research/scientific-evaluation`: 增加 Wilson LER、超时计数策略、残余逻辑类分布与配对非劣效 Gate 规则。
- `research/experiment-lifecycle`: 增加本地 `start_or_recover` 语义、陈旧 RUNNING Attempt 处理和 Stage 复用条件。
- `research/paper-reproduction`: 将 contracts-only 执行要求改为“定义 cell 无副作用、实验 cell 显式标记”，并增加 Torlai–Melko 2017 复现协议。

## Impact

- 新增模块：`ai_qec/qec/codes/`、`ai_qec/qec/noise_models.py`、`ai_qec/qec/backends/stim_noise.py`、`ai_qec/data/generators/stim_code_capacity.py`、`ai_qec/data/datasets/local.py`、`ai_qec/data/loaders/`、`ai_qec/models/generative/`、`ai_qec/models/decoders/{classical,generative}/`、`ai_qec/training/{trainers,objectives,optimizers.py,execution_planner.py}`、`ai_qec/evaluation/{scientific/statistics.py,scientific/local.py,performance/}`、`ai_qec/experiment/{local.py,config.py,streams.py}`、`ai_qec/reporting/`、`ai_qec/paper/local_runtime.py`、`ai_qec/utils/hashing.py`、`ai_qec/implementations.py`。
- `ai_qec.notebook_api` 新增 `LocalNotebookPlatform`、绘图与统计函数和错误类型；导入 facade 仍不加载 torch、stim、pymatching、qiskit、cudaq。
- 运行时依赖（可选组 `runtime`）：numpy、torch、stim、pymatching、matplotlib；contract 安装仍无依赖。
- 运行产物写入已被 git 忽略的 `datasets/` 与 `runs/`。
- 测试：新增 code/noise/generator、数据集存储、训练恢复、解码器、统计与 Gate、生命周期和端到端 runtime 测试；运行时测试在缺少依赖的环境中跳过，verification 在 `quantum` conda 环境执行。
