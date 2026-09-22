# Proposal

## Why

当前 RBM 训练在小 batch 下由 Python 逐个提交大量 CUDA kernel，GPU 计算能力没有被充分利用；实测捕获完整训练步后单步可从约 2.03 ms 降至约 0.313 ms。平台还需要支持 CNN、RNN、GNN、Transformer 等不同模型结构，因此 CUDA Graph 必须作为模型无关、显式选择且可审计的训练步执行实现，而不能写死在 RBM trainer 中。

## What Changes

- 新增配置字段 `execution.step_executor`，由 Registry 选择 `pytorch-eager` 或 `pytorch-cuda-graph`；选择未知、未决或与执行资源不兼容的实现时，在创建 Run 前失败。
- 从 PyTorch trainer 中抽出模型无关的训练步执行 contract；现有 eager 路径成为可执行基线，保持 minibatch、optimizer、RNG、checkpoint 与恢复语义。
- 新增 PyTorch CUDA Graph 训练步 executor：每个 minibatch 仍由现有 DataLoader 逐批传输到 CUDA，只把当前 batch 复制到固定地址的 batch 级输入缓冲，再 replay 捕获的 loss、backward 与 optimizer step。完整训练集不常驻 GPU。
- 按输入字段、shape、dtype 与 device 形成 batch signature；相同 signature 复用 graph，新的受支持 signature 可捕获独立 graph，超过声明上限或遇到不支持的操作时明确失败。
- CUDA Graph warmup、capture 与恢复不得增加 optimizer update、global step、minibatch 消费或训练随机流消费；每次 replay 必须推进 RNG，恢复 Attempt 重新 capture 而不持久化 graph 对象。
- Training Stage 记录请求和实际 executor、signature、capture/replay 次数与耗时以及 fallback 证据；CUDA Graph 失败不得静默退回 eager。
- CUDA Graph 与 eager 使用不同的完整配置和训练 Stage 复用键；切换 executor 不改变 DatasetKey，但必须重新训练并重新通过 Accuracy Gate。

非目标：不实现完整训练集常驻 GPU，不修改 Dataset Pipeline contract、DatasetArtifact 或 DatasetKey，不优化 DataLoader/H2D，不自动 padding 或 bucketing，不承诺所有动态 shape 模型都能 capture，不引入 `torch.compile`、AMP、DDP、多 GPU、训练队列、解码 CUDA Graph、FPGA、ASIC 或实时 QPU feedback，也不增加任何静默 fallback。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `research/training-contracts`: 增加 Registry 驱动的训练步 executor contract，以及单 GPU float32 PyTorch CUDA Graph 训练步的 capture、shape、RNG、恢复、证据和 fail-fast 要求。

## Impact

- 配置与选择：`ai_qec/training/execution.py`、`ai_qec/training/execution_planner.py`、`ai_qec/experiment/config.py`、`ai_qec/registries.py`、`ai_qec/implementations.py`。
- 训练实现：新增 `ai_qec/training/executors/`，并修改 `ai_qec/training/trainers/pytorch_trainer.py` 通过 executor 执行每个 minibatch。
- 生命周期与证据：`ai_qec/paper/local_runtime.py` 和 TrainingRecoveryCheckpoint payload 记录 executor/RNG 恢复所需信息；ModelCheckpoint 格式不变，CUDA Graph 对象不持久化。
- Notebook：`paper/srcs/ai_for_qec_workflow.ipynb` 显式选择 executor；Dataset Pipeline、数据生成、解码和科学评估实现不变。
- 测试：覆盖 eager 回归、Registry/preflight、实际 CUDA capture/replay、多个 batch signature、禁止 fallback、RNG 推进、Attempt 恢复、Stage 复用隔离、Accuracy Gate 与性能证据。
