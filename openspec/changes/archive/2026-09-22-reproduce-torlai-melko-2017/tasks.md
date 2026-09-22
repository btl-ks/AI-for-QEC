# Tasks

## 1. 基础设施与 contract 扩展

- [x] 1.1 新增 `ai_qec/utils/hashing.py`（规范化 JSON、sha256 文件/对象摘要）与 `ai_qec/experiment/streams.py`（DerivedRandomStreams），并以 `tests/unit/test_lifecycle.py` 验证派生种子稳定且各流互不影响
- [x] 1.2 为 `DecoderEvaluation` 增加 `logical_class_counts`、为 `TrainingSpec` 增加 `optimizer_parameters`/`loss_parameters`（带默认值），并以 `tests/unit/test_public_contracts.py` 验证旧构造方式仍然有效
- [x] 1.3 新增 `ai_qec/implementations.py` 显式加载实现，并以子进程测试验证导入 facade 不加载选定框架、全局 Registry 为空，加载后只出现本 change 实现的 key

## 2. QEC 码、噪声与数据集

- [x] 2.1 实现 `ToricCode`（校验矩阵、逻辑算符、syndrome、残余逻辑类）与独立相位翻转噪声模型，并以 `tests/unit/test_toric_code.py` 验证 L=4 的度数、逻辑算符对易性和同调类判定
- [x] 2.2 实现 Stim noise compiler 与 `stim-syndrome-cpu` 生成器（版本精确匹配、不支持语义显式拒绝），并以 `tests/unit/test_stim_generator.py` 验证样本与码定义一致、同种子可重现、版本不符与非 X 基被拒绝
- [x] 2.3 实现本地 DatasetKey、registry、resolver 与 split 读取（原子提交、命中校验、损坏拒绝、提交前语义校验），并以 `tests/unit/test_dataset_store.py` 验证 GPU 数量不影响 key、缓存命中复用、分片被改后命中与读取均失败
- [x] 2.4 实现 `pytorch-dataloader-h2d` pipeline，并以 `tests/unit/test_pipeline_and_gate.py` 在 CPU 与 CUDA 上验证 transfer 保留 sample/dataset 身份且 TransferEvidence 与实际 device 一致

## 3. 模型、训练与解码

- [x] 3.1 实现联合 RBM 模型族、CD-k 目标、SGD 与 constant 调度器，并以 `tests/unit/test_rbm_training.py` 验证自由能与条件概率解析式一致
- [x] 3.2 实现 PyTorch trainer（epoch 恢复 checkpoint、独立 ModelCheckpoint、恢复继续训练），并验证 CPU 上“中断+恢复”与一次性训练的最终权重逐位相同
- [x] 3.3 实现 RBM Gibbs decoder（算法 1、超时显式标记）与 PyMatching baseline，并以 `tests/unit/test_decoders.py` 验证接受的恢复链满足 syndrome、超时行为 `-1` 且状态为 `timed-out`、宽度不符返回 `unsupported`
- [x] 3.4 实现本地执行计划解析，并验证 distributed、mixed precision、compile、多 GPU 与不可用 CUDA 在创建 Run 目录前失败

## 4. 科学评估与 Gate

- [x] 4.1 实现 Wilson 区间与 Newcombe 配对差异区间，并以 `tests/unit/test_statistics.py` 验证 Wilson 与公开数值一致、Newcombe 区间满足反对称、包含点估计和随样本量收窄
- [x] 4.2 实现本地科学评估器（无效样本策略、残余逻辑类、配对列联表、Artifact）与配对非劣效 Gate，并以 `tests/unit/test_pipeline_and_gate.py` 与集成测试验证 PASS/FAIL 判定、未知规则或 baseline 不产生判定、证据持久化
- [x] 4.3 实现 Gate PASS 后的批量吞吐测量与单点图表，并验证只在 PASS 时被编排调用

## 5. 生命周期与 runtime

- [x] 5.1 实现文件型 Experiment/Attempt/Stage/ArtifactRepository，并以 `tests/unit/test_lifecycle.py` 验证终态不可变、陈旧 RUNNING 被标记 interrupted、Artifact 被修改后校验失败
- [x] 5.2 实现 `LocalNotebookPlatform` 编排与恢复计划，并以 `tests/integration/test_local_runtime.py` 在 CPU 小配置上验证完整顺序、第二次执行复用已完成 Stage、训练中断后新 Attempt 从恢复 checkpoint 继续
- [x] 5.3 从 `ai_qec.notebook_api` 导出 runtime、绘图与统计函数及错误类型，并以公共导入测试验证

## 6. Notebook 与文档

- [x] 6.1 重写 `paper/srcs/ai_for_qec_workflow.ipynb` 为 Torlai–Melko 复现（定义 cells 无副作用、实验 cells 标记 `run-experiment`），并以 `tests/unit/test_paper_notebook.py` 验证定义 cells 执行、标签和编排顺序
- [x] 6.2 更新 `README.md`、`paper/README.md`、`pyproject.toml` 可选依赖、technology catalog 与 capability registry，并以 schema 校验验证
- [x] 6.3 在 `quantum` 环境用 nbclient 执行 `paper` profile 的 Notebook，记录每个 (L, p) 点的 LER、区间与 Gate 结果

## 7. 验证与归档

- [x] 7.1 在 `quantum` 环境运行全部测试与 compileall，在基础环境运行契约测试（运行时测试跳过），运行 `openspec validate --all --strict --no-interactive`，并把输出写入 `verification.md`
- [x] 7.2 依据证据更新 capability 状态后归档 change，并再次运行全量 strict validation
