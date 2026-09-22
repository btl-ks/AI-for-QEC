# Tasks

## 1. Executor contract、配置与 Registry

- [x] 1.1 新增 vendor-neutral `TrainingStepExecutor`、step result 与 evidence 值对象，并从 `ai_qec.notebook_api` 导出公共 contract；以 `python -m unittest tests.unit.test_public_contracts` 验证导入不加载 torch 且值对象字段稳定
- [x] 1.2 新增 `TRAINING_STEP_EXECUTORS`、`execution.step_executor` 的 `REGISTRIES_BY_PATH` 映射及 builtin 加载，登记真实可执行的 `pytorch-eager` 与 `pytorch-cuda-graph`；以 `python -m unittest tests.unit.test_registry tests.unit.test_training_step_executors` 验证路径唯一、重复注册失败且不存在占位工厂
- [x] 1.3 扩展配置解析、`ExecutionSpec` 与 resolved plan，支持 `step_executor` 和严格的 `step_executor_options`（eager 为空、CUDA Graph 只接受正整数 `max_graphs`）；以 `python -m unittest tests.unit.test_training_step_executors tests.unit.test_decoders` 验证未知键、额外 option、`unresolved`、非 CUDA graph 请求和不可用 CUDA 都在创建 `runs/`/`datasets/` 前失败
- [x] 1.4 保证 executor 字段进入 Experiment 摘要和训练 Stage 复用键但不进入 DatasetKey；以 `python -m unittest tests.unit.test_stage_reuse tests.unit.test_dataset_store` 验证 eager/graph 不复用训练 Stage而解析到同一数据集

## 2. Eager 基线重构

- [x] 2.1 实现 `pytorch-eager` executor，把 visible 构造、objective、清梯度、backward 与 optimizer step 从 trainer 抽出，并让 `PyTorchTrainer` 只依赖 executor contract；以 `python -m unittest tests.unit.test_rbm_training tests.unit.test_training_step_executors` 验证固定 seed 下抽取前的 objective、更新次数和最终参数基线不变
- [x] 2.2 保持 epoch、scheduler、监控、ModelCheckpoint 与每 epoch TrainingRecoveryCheckpoint 由 trainer 管理，并把 requested/observed eager evidence 写入 Training Stage；以 `python -m unittest tests.integration.test_local_runtime` 验证端到端 Stage 与 Artifact 行为不回归
- [x] 2.3 保持 eager 中断恢复的模型、optimizer、scheduler、shuffle 与训练 RNG 轨迹；以 `python -m unittest tests.integration.test_local_runtime.LocalRuntimeTests.test_interrupted_training_resumes_and_matches_uninterrupted_training` 验证现有 CPU 逐位一致性

## 3. CUDA Graph 训练步

- [x] 3.1 实现包含字段、shape、dtype、device 与 stride 的 batch signature，以及受 `max_graphs` 限制的 per-signature 生命周期和静态 batch 输入；以 CUDA 测试 `python -m unittest tests.unit.test_cuda_graph_executor.CUDAGraphExecutorTests.test_static_input_uses_each_new_batch` 验证连续相同 signature 使用新内容且只保留 batch 级输入
- [x] 3.2 实现 `unseen → warming → captured` 流程，让真实 warmup 与 capture minibatch 各自只产生一次正常训练更新，并稳定初始化 gradient 与 optimizer state；以 CUDA 测试 `python -m unittest tests.unit.test_cuda_graph_executor.CUDAGraphExecutorTests.test_warmup_capture_and_replay_count_exact_updates` 验证 optimizer update/global step 等于消费的 minibatch 数
- [x] 3.3 捕获 loss、backward 与 optimizer step并实现 replay，保持热路径无逐步 `.item()`/主机同步；以 CUDA profiler 测试和 executor evidence 验证至少一次真实 capture、非零 replay、参数持续更新且 replay 不重新 capture
- [x] 3.4 支持有限个离散 signature 共用模型与 optimizer state；以 CUDA 测试覆盖交替 signature 和最后一个小 batch，并验证超过 `max_graphs` 时在该 minibatch 更新前失败、样本未被 padding/裁剪/丢弃且没有 eager fallback
- [x] 3.5 捕获不兼容的 CPU op、数据依赖 shape 或 replay 错误时保留原异常并使 training Stage 失败；以 `python -m unittest tests.unit.test_cuda_graph_executor.CUDAGraphExecutorTests.test_capture_failure_has_no_fallback` 验证没有成功 ModelCheckpoint且 `fallback_observed` 不被伪报为 false 成功

## 4. CUDA RNG、checkpoint 与 Attempt 恢复

- [x] 4.1 把具名 CUDA 训练 generator 以 graph-safe 方式注册并让 capture/replay 顺序推进同一随机流；以 CUDA 测试 `python -m unittest tests.unit.test_cuda_graph_executor.CUDAGraphExecutorTests.test_replay_advances_registered_generator` 验证连续 replay 不重复随机样本或参数更新
- [x] 4.2 在 TrainingRecoveryCheckpoint payload 中记录 executor ID、实现版本与 options digest并校验恢复兼容性，同时保持 ModelCheckpoint payload 不含 optimizer、RNG 或 graph 对象；以 `python -m unittest tests.unit.test_training_step_executors tests.unit.test_public_contracts` 检查两类 checkpoint 的边界
- [x] 4.3 恢复顺序实现为校验 checkpoint、恢复全部训练状态、清空进程内 graph cache、从下一 epoch 的真实 minibatch 重新 warmup/capture；以真实 CUDA 集成测试比较中断恢复与不中断的同 executor 最终模型、global step、history 与 generator state
- [x] 4.4 验证 graph、静态输入和 memory pool 不被序列化或跨 Attempt 复用；以 checkpoint 内容断言和恢复 Stage evidence 中的新 capture 计数作为可观察证据

## 5. 模型无关性与 runtime evidence

- [x] 5.1 用纯 PyTorch 测试夹具覆盖固定 shape 的卷积、循环、attention 与图聚合训练步，并全部经同一个 CUDA Graph executor contract 执行；以 `python -m unittest tests.unit.test_cuda_graph_model_shapes` 验证 trainer/executor 不读取模型家族名称，且测试夹具不被登记为生产 ModelFamily
- [x] 5.2 在 `LocalNotebookRun` 的 Training Stage metadata 中记录 requested/observed executor、版本、device、options digest、每个 signature 的 warmup/capture/replay 次数、copy/capture/replay 时间和 fallback 状态；以 `python -m unittest tests.integration.test_local_runtime` 校验成功与失败证据
- [x] 5.3 对动态 shape 或 capture-unsafe 测试模型验证显式失败，同时保持 eager executor 可正常训练；以 `python -m unittest tests.unit.test_cuda_graph_model_shapes` 证明平台支持不同模型 contract但不伪称所有模型都可 CUDA capture

## 6. Notebook、科学验证与性能证据

- [x] 6.1 在论文 Notebook 配置中显式加入 `execution.step_executor` 与 options，先以 eager 保持回归，再切换 CUDA Graph；以 `python -m unittest tests.unit.test_paper_notebook` 验证 Notebook 仍只使用公共 API、定义 cells 无副作用且没有训练数据驻留配置
- [x] 6.2 在真实 CUDA 环境执行 smoke profile，验证 Dataset 仍由现有 DataLoader 逐 batch H2D、Training Stage 有真实 capture/replay evidence、全部 Attempt/Stage 正确终态；把命令、环境、Experiment/Attempt 和 Artifact checksum 写入 `verification.md`
- [x] 6.3 按长任务规则用 `setsid nohup` 执行 CUDA Graph 论文验证网格或预先声明的完整科学验证集合，确认新模型重新运行 Scientific Evaluation 与 Accuracy Gate且未复用 eager 训练/评估；以日志、PID、最终 Gate 判定和 Stage reuse metadata 作为 `verification.md` 证据
- [x] 6.4 在相同硬件与配置上测量 eager、batch copy、capture 和 steady-state replay，更新 `paper/srcs/performance.md` 并报告实际端到端训练收益而非只报告重复同一 batch 的微基准

## 7. 治理、全量验证与归档准备

- [x] 7.1 在真实实现和测试完成后更新 technology catalog、README 与 `openspec/capabilities.yaml`，只登记实际可执行的 executor并引用实现、测试和 `verification.md`；以 catalog/schema 校验和 capability evidence 文件存在性验证
- [x] 7.2 在基础环境运行不需要 CUDA 的全部测试，在 `quantum`/CUDA 环境运行完整测试与 GPU 专用测试，记录精确命令、通过/跳过数和失败处置到 `verification.md`
- [x] 7.3 运行 `openspec validate add-pytorch-cuda-graph-training-step --strict --no-interactive` 与 `openspec validate --all --strict --no-interactive`，确认 tasks 全部完成、科学和性能证据可复核后再归档 change
