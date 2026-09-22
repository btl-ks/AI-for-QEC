# Proposal

## Why

论文网格在绝对容差 0.02 下，Gate 的严格程度随 baseline 错误率大幅变化：p=0.05 时 0.02 相当于允许 RBM 比 MWPM 差约 50%，p=0.15 时只允许约 4%，所以阈值以上的点几乎必然 FAIL，阈值以下又过于宽松。随 baseline 按比例缩放的相对容差可以让各点的严格程度一致。另外，Gate 字段属于完整配置摘要，改容差会让 22 个 Experiment 的身份全部变化；现有复用只在同一 Experiment 内生效，所以一次 Gate 调整要重训全部模型（约 40 分钟），尽管训练结果与 Gate 无关。

## What Changes

- 新增 Gate 规则 `paired-relative-non-inferiority`：`tolerance` 表示相对容差 δ，Gate 在同一 shot 集合上估计 `LER_AI − (1+δ)·LER_baseline` 的置信区间，只有区间上界 ≤ 0 时 PASS。现有 `paired-non-inferiority`（绝对容差）的语义与结果保持不变。
- 新增跨 Experiment 的 Stage 复用：训练、科学评估与性能评估 Stage 按“Stage 复用键”（输入 Artifact 引用加上该 Stage 及其上游实际读取的配置字段）查找任意 Experiment 中已完成且 checksum 校验通过的同名 Stage，并复用其输出。实验名、Gate 等下游字段不进入对应的复用键。复用来源记录在新 Stage 的元数据中。
- Accuracy Gate 与可视化始终在当前 Attempt 中重新计算，不跨 Experiment 复用。
- 论文 Notebook 的 Gate 改为 `paired-relative-non-inferiority`、`tolerance = 0.15`，并更新说明。这一改动是在看到 0.02 的判定结果之后作出的重新登记，Notebook 如实记录这一点；0.02 下的 22 个 Experiment 保留在 `runs/`，不删除也不改写。
- Experiment 身份仍然是完整配置的摘要。改 Gate 会产生新的 Experiment，旧 Experiment 的历史保持不变。

非目标：不引入模型注册表服务、数据库或分布式缓存；不做跨 Experiment 的训练中途恢复（只复用已完成的 Stage）；不把代码版本纳入复用键；不改变数据集的 DatasetKey 复用机制；不调整 Gate B 或性能研究的判定。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `research/scientific-evaluation`: 增加配对相对非劣效 Gate 规则。
- `research/experiment-lifecycle`: 增加跨 Experiment 按 Stage 复用键复用已完成 Stage 的要求。

## Impact

- `ai_qec/evaluation/scientific/statistics.py`：配对差异区间支持对 baseline 比例乘以系数。
- `ai_qec/evaluation/scientific/local.py`、`ai_qec/experiment/config.py`：新增并校验 Gate 规则。
- `ai_qec/experiment/`：新增 Stage 复用键计算。`ai_qec/paper/local_runtime.py`：跨 Experiment 查找与复用，按引用而不是按本 Experiment 清单读取复用来的 Artifact。
- `paper/srcs/ai_for_qec_workflow.ipynb`：Gate 配置与说明。
- 测试：`tests/unit/test_statistics.py`、`tests/unit/test_pipeline_and_gate.py`、`tests/integration/test_local_runtime.py`、`tests/unit/test_paper_notebook.py`。
