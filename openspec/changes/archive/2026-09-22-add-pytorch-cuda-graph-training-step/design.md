# Design

## Context

动机见 proposal.md。当前 `PyTorchTrainer` 在 minibatch 循环中直接执行 objective、`zero_grad`、backward 与 optimizer step；模型家族负责创建网络和把字段组合成可见 Tensor，objective 和 optimizer 已通过 Registry 构造。`ExecutionSpec` 已包含 `compile_model`，但本地 planner 明确拒绝它；本 change 使用直接 PyTorch CUDA Graph，不改变 `compile_model=True` 仍不受支持的事实。

现有 CPU→GPU pipeline 每个 epoch 用 DataLoader 打乱训练集，并把每个 minibatch 复制到执行设备。完整 L=6 训练集约 10.8 MB，虽然可以常驻 GPU，但用户已明确排除该功能；CUDA Graph 必须在保留现有逐 batch H2D 的条件下工作。TrainingRecoveryCheckpoint 当前保存模型、optimizer、scheduler、shuffle generator、训练 Gibbs generator、epoch、global step 与历史，ModelCheckpoint 只保存推理 payload。

CUDA Graph 要求 replay 使用固定的 Tensor 地址，且 capture 内不得出现不安全的 CPU 操作、动态分配或数据依赖控制流。平台未来会承载多种模型结构，因此兼容性由统一训练步 contract 和实际 batch signature 决定，而不是模型家族名称。

## Goals / Non-Goals

**Goals:**

- 在现有 PyTorch trainer 下建立可注册、可替换的训练步 executor，并让 eager 成为行为基线。
- 用直接 PyTorch CUDA Graph 捕获完整的 loss、backward 与 optimizer update，消除当前小 batch 训练的大量逐 kernel 提交开销。
- 保留现有 DataLoader、逐 batch H2D、epoch、checkpoint、恢复和 Accuracy Gate 语义。
- 对固定或少量离散 shape 的任意 ModelFamily 提供同一执行路径，并对不兼容行为显式失败。
- 让配置身份、Stage evidence 和验证 Artifact 足以证明实际使用了 CUDA Graph 而非 fallback。

**Non-Goals:**

- 不让完整训练集常驻 GPU，不新增 dataset cache、GPU DatasetArtifact 或 data-pipeline technology。
- 不优化 DataLoader、pinned memory、prefetch 或 H2D；这些仍可能在 graph 优化后成为下一瓶颈。
- 不自动 padding、bucketing、改变 sampler 或丢弃最后一个 batch。
- 不实现 `torch.compile`、AMP、DDP、多 GPU、训练队列或解码 CUDA Graph。
- 不新增生产 CNN、RNN、GNN 或 Transformer ModelFamily；测试模型只证明 executor 不依赖模型名称。

## Decisions

### 1. 一个 vendor-neutral executor contract，具体实现经 Registry 构造

新增 `TrainingStepExecutor` Protocol 与 `TrainingStepResult`/evidence 值对象，公共 contract 从 `ai_qec.notebook_api` 导出。新增 Registry 路径 `execution.step_executor`，首批登记：

```text
pytorch-eager
pytorch-cuda-graph
```

配置同时增加 `execution.step_executor_options` mapping；`pytorch-eager` 只接受空 mapping，`pytorch-cuda-graph` 首期只接受正整数 `max_graphs`。字段属于 `ExecutionSpec`，因此不进入 DatasetKey；现有 Stage 复用键的“新增字段默认保留”规则会自动隔离 eager 与 graph 产物。

executor context 持有 network、ModelFamily 的 Tensor 组装 callable、objective、optimizer 与训练随机 generator。`PyTorchTrainer` 继续拥有 epoch、DataLoader、scheduler、监控、checkpoint 与恢复，不识别具体 executor 名称。

替代方案：复用 `compile_model: bool`。否决原因：该字段表示模型编译且目前被明确拒绝，不能表达 eager、直接 CUDA Graph、未来 torch.compile 等互斥实现及其选项，也无法经 Registry 构造。

### 2. eager 先抽成等价实现，再加入 CUDA Graph

`pytorch-eager` 把当前五个动作原样封装：构造 visible、计算 objective、清梯度、backward、optimizer step。先用现有 CPU 恢复测试和固定 seed 训练测试证明抽取没有改变结果，再让 trainer 只调用 Protocol。

这样 CUDA Graph 是并列实现，而不是 `if cuda_graph` 散落在 trainer 内；未来模型只需满足 ModelFamily/objective contract。

### 3. 保留现有 H2D，executor 只分配 batch 级静态输入

DataLoader 仍返回当前 device batch。CUDA Graph executor 为每个已接受 signature 分配 `empty_like` 静态输入，并在每次训练步开始时用当前 stream 把 device batch `copy_` 到静态输入；copy 不属于被捕获训练图。graph 读取固定地址静态输入，参数、gradient 与 optimizer state 同样保持固定地址。

这会比让 DataLoader 直接填充静态 buffer 多一次 device-to-device copy，但可以完全保持 Dataset Pipeline contract，避免把 graph 生命周期泄漏到数据层。性能验证分别记录 batch copy 与 replay 时间，后续如它成为瓶颈再提出独立 change。

替代方案：把完整训练 split 传到 GPU 后用 device index 取 batch。否决原因：这是用户明确排除的数据驻留功能，且对大图、长序列和未来模型不具备通用内存上界。

### 4. warmup 与 capture 使用真实 minibatch，但都计作正常训练步骤

每个 signature 维护 `unseen → warming → captured` 状态。为避免 warmup 或 optimizer 懒状态初始化产生隐藏更新：

- warming 阶段的调用用 eager 执行当前真实 minibatch，每次都正常增加一个 global step；
- optimizer state 和 gradient storage 稳定后，下一个该 signature 的真实 minibatch用于 capture；capture 本身执行的更新就是该 minibatch 唯一的训练更新；
- 后续 minibatch 才 replay；
- 不复制、回滚或重放 warmup/capture minibatch。

warmup 次数是 executor 的固定、版本化实现参数，不由 Notebook 调节；Stage evidence 记录实际 warmup、capture 与 replay 步数。对只出现一次的 signature，训练可能只完成 warming 而没有对应 graph；整个成功 Stage 仍必须至少有一个 signature 实际 capture 并 replay，否则拒绝宣称 CUDA Graph 成功。

替代方案：用真实模型执行额外 warmup 后恢复快照。否决原因：恢复 optimizer state 可能替换被 graph 引用的 Tensor 地址，且容易错误消费 RNG。用克隆模型 warmup也不能初始化真实 optimizer 的懒状态。

### 5. graph cache 以完整 batch signature 为键并有硬上限

signature 是字段名排序后每个 Tensor 的 shape、dtype、device 与 stride。每个 signature 拥有静态输入、graph、静态 loss 输出和生命周期计数；graph 共享同一 network 参数和 optimizer state，capture 使用同一个 CUDA graph memory pool，并在创建和 replay 边界显式同步所需 stream。

`max_graphs` 防止可变长度序列或动态图造成无界 capture 与显存增长。新 signature 首次出现时先检查上限；超限在任何 optimizer update 前失败。数据依赖 shape、CPU op 或 capture unsafe op 由 capture 异常显式暴露，Stage 失败且不 fallback。

自动 padding/bucketing 被排除，因为它会改变模型输入与潜在科学语义；未来 RNN、GNN 或 Transformer ModelFamily 可在自己的 preprocessing/spec 中产生有限 signature，再使用本 executor。

### 6. capture 内使用稳定 gradient 和 optimizer state

CUDA Graph executor 在 warming 阶段完成 optimizer 的懒状态初始化，并确保所有参数 gradient storage 已建立。captured step 使用不会把 `.grad` 重新置为 `None` 的清零方式，避免 replay 时重新分配或更换地址。loss 累计保持在 device 上，epoch 结束后才允许现有监控路径执行主机读取；训练热路径不得每步 `.item()` 或同步。

如果 optimizer 无法在 warming 后提供稳定、可 capture 的状态，executor 在 capture 时失败。首期不按 optimizer 名称维护白名单；真实可执行性由 capture 测试和 evidence 证明。

### 7. 自定义 CUDA generator 注册到 graph，checkpoint 不保存 graph

当前 objective 显式接收具名训练 generator。CUDA executor 在 capture 前把该 generator 注册为 graph-safe RNG；capture 和每次 replay按 PyTorch 的 graph-safe seed/offset 机制推进同一 generator。checkpoint 继续保存 generator 的可恢复状态，并增加 executor ID、版本和 options digest，用于拒绝不匹配的恢复状态。

恢复顺序为：验证 checkpoint → 恢复 network/optimizer/scheduler/shuffle/RNG → 清除进程内 graph cache → 从下一 epoch 的真实 minibatch 重新 warming/capture。CUDA Graph 对象、静态输入和 graph memory pool 都是 Attempt 进程内状态，绝不序列化。ModelCheckpoint schema 不变。

### 8. requested、observed 与失败证据分开记录

resolved execution plan 记录 requested executor；Training Stage metadata 记录 observed executor/version、device、options digest、每个 signature 的 warming/capture/replay 计数、capture/copy/replay 时间和 `fallback_observed=false`。失败发生在 capture/replay 时，Stage 记录原异常类型和消息，Attempt 走现有失败或中断路径，不提交成功 ModelCheckpoint。

Notebook 初次接入时显式选择 `pytorch-cuda-graph`，由完整配置产生新 Experiment。Dataset 按 DatasetKey 复用，训练、科学评估和性能评估因 executor 与新模型引用而重新执行；Accuracy Gate 必须在任何性能结论前通过。

## Risks / Trade-offs

- [不同 optimizer 的懒状态或原地行为不满足 capture] → 用真实 warming 初始化并以 capture 作为可执行性判据；失败明确终止，不维护名称白名单或 fallback。
- [多 signature graph 共享参数和 optimizer state 时出现内存池或 stream 顺序错误] → 使用共享 graph pool、稳定 Tensor 地址和显式事件/同步；测试最后一个小 batch及交替 signature。
- [graph replay 重复 capture 时的随机数] → 注册自定义 generator，验证连续 replay 的参数更新和 generator state 均不同，并验证 epoch 恢复轨迹。
- [warming/capture 悄悄多训练] → 每次调用只接受一个真实 minibatch并返回一个 StepResult；global step 只由 trainer 对成功调用递增，测试总更新数等于 DataLoader minibatch 数。
- [静态 batch 的额外 device copy 抵消部分收益] → 明确记录 copy 与 replay 时间；本 change 优先保证边界正确，后续再决定是否优化 pipeline。
- [动态图或可变序列产生过多 signature] → `max_graphs` 是硬上限，超限 fail fast；padding/bucketing 留给独立科学配置。
- [CUDA Graph 与 eager 浮点/RNG 路径不逐位相同] → 二者拥有不同 Experiment/Stage 身份，重新运行 Scientific Evaluation 与 Accuracy Gate，不把 eager 模型作为等价证据。

## Migration Plan

1. 先引入 executor contract、Registry 与 `pytorch-eager`，保持现有 Notebook 显式选择 eager，运行全量回归与 CPU 恢复测试。
2. 实现 `pytorch-cuda-graph` 和 GPU 专用测试；未通过真实 capture/replay、RNG 与恢复验证前，不在 Notebook 选择它，也不登记为可执行 technology。
3. 在 CUDA 环境执行 smoke profile 和预注册的 Accuracy Gate；按长任务规则用脱离会话的入口执行需要数分钟以上的网格验证。
4. 验证通过后更新 Notebook 配置、technology catalog、capability evidence 和性能分析，再归档 change。

回滚时把 Notebook 的 `execution.step_executor` 改回 `pytorch-eager`；完整配置摘要会选择 eager Experiment，已完成的 CUDA Graph Experiment 与 Artifact 保持不可变历史，不删除也不改写。
