# AI for QEC Research Platform

面向 AI for Quantum Error Correction 的可复现研究平台。

> 当前状态：**contracts + 本地单进程 runtime（partial）**。本地 runtime 能端到端执行 toric code、相位翻转 code-capacity 场景下的 Torlai & Melko (2017) 神经解码器复现：Stim 数据生成与复用、Registry 驱动的 PyTorch eager/CUDA Graph CD-k 训练步与 epoch 边界恢复、按 Stage 复用键跨 Experiment 复用已完成的训练与评估、RBM Gibbs 解码、PyMatching baseline、Wilson LER、配对非劣效 Accuracy Gate（绝对或相对容差）与 Gate 后的吞吐量测量。CUDA-Q、Qiskit、circuit-level 噪声、多 GPU、AMP、Gate B 与正式性能研究仍未实现。

## 研究主路径

```text
Configure
  → Dataset Resolve / Reuse
  → AI Training
  → Scientific Evaluation + Classical Baseline
  → Accuracy Gate
  → Performance Research
  → Accuracy Revalidation
  → Trade-off Report
```

项目坚持 accuracy first、performance second。正式性能研究必须在科学 Accuracy Gate 通过之后进行。

## v0.1 技术基线

| 组件 | Primary | 状态 |
| --- | --- | --- |
| QEC Circuit | Qiskit | contracts-only |
| QEC Code | Stim（CPU）、CUDA-Q（GPU） | Stim：partial（code-capacity）；CUDA-Q：contracts-only |
| QEC Noise | Stim noise adapter | partial（独立相位翻转精确编译，其他显式拒绝） |
| Syndrome Generator | Stim（CPU）、CUDA-Q（GPU） | Stim：partial（单轮 code-capacity）；CUDA-Q：contracts-only |
| Trainer | PyTorch + CUDA | partial（单进程、单设备、float32；eager 与直接 CUDA Graph 训练步） |
| Decoder | PyMatching（CPU）、PyTorch（GPU） | PyMatching：implemented；PyTorch：partial（RBM Gibbs） |
| QECBatch runtime | PyTorch Tensor | partial；持久化为 bit-packed CPU buffer |
| CPU→GPU | Dataset/DataLoader + pinned memory + non-blocking | implemented |
| GPU→GPU | CUDA-Q → PyTorch CUDA，zero-copy preferred | contracts-only |

完整 primary/alternative/phase 状态以 `openspec/technology-catalog.yaml` 为准。DALI、Ray Data 和自定义 CUDA extension 不属于 v0.1 必需实现。

配置中的待定选择统一使用字符串 `unresolved`。任何构造开始前必须调用配置预检；只有当前阶段明确不消费的完整字段路径才能通过 allow set 暂时放行。Registry 只登记真实可用的实现，且只在 `LocalNotebookPlatform` 构造时通过 `ai_qec.registry.bootstrap.load_builtin_implementations()` 显式加载；导入 `ai_qec.notebook_api` 不会加载 torch、stim、pymatching、qiskit 或 cudaq。选中尚不可用的技术（例如 `qiskit-circuit`）会在创建任何目录前明确失败。

训练步由 `execution.step_executor` 选择 `pytorch-eager` 或 `pytorch-cuda-graph`。CUDA Graph
实现按完整 batch signature 建立有界 graph cache，只保存 batch 级静态输入；现有 DataLoader 仍逐
batch 执行 H2D，完整训练集不会常驻 GPU。未知选项、非 CUDA 设备、不可 capture 的模型或
objective 都会明确失败，不会静默回退 eager。

## 仓库结构

- `openspec/`：正式需求、active changes、能力状态与验收依据。
- `ai_qec/`：vendor-neutral 公共 contract，以及 adapter/实现层（`qec/codes`、`qec/backends/stim_noise.py`、`data/generators/stim_code_capacity.py`、`data/datasets/local.py`、`data/loaders/`、`models/`、`training/`、`evaluation/`、`experiment/local.py`、`paper/local_runtime.py`）。该源码树由 setuptools 按 PEP 420 隐式 namespace package 发现，不使用 `__init__.py`。
- `configs/`：仅用于说明配置边界的示例；可执行配置见论文 Notebook。
- `paper/srcs/`：论文 Notebook 源码；`ai_for_qec_workflow.ipynb` 复现 Torlai & Melko (2017)。
- `tests/unit/`、`tests/integration/`：contract、实现与端到端 runtime 测试；缺少运行时依赖时相关测试自动跳过。
- `datasets/`、`runs/`：运行时生成的不可变数据集与 Experiment/Attempt 记录（git 忽略）。
- `docs/AI_for_QEC_*.md`：架构背景，不能替代 OpenSpec。

Notebook 和外部调用方的稳定导入入口是：

```python
import ai_qec.notebook_api as qec
```

中间目录不提供包级便捷导出；请使用上述 facade，或从定义符号的叶子模块显式导入。源码位于仓库根目录，因此在仓库根目录运行 Python 时可直接导入；正式分发仍应通过构建 wheel 并在仓库外验证。

本地 runtime 通过依赖注入使用：

```python
runtime = qec.LocalNotebookPlatform(project_root)
experiment = runtime.create_experiment(config)   # 全部预检通过后才创建 runs/<experiment-id>/
run = experiment.start_or_recover()               # 新 Attempt；终态 Attempt 永不复活
dataset = run.resolve_dataset()                   # 已校验复用或生成+校验+原子提交
model = run.train(dataset)                        # 复用已校验模型，或从最新 epoch checkpoint 继续
```

运行时依赖通过可选组安装（Stim 版本必须与 `dataset.generator_version` 一致）：

```bash
pip install -e ".[runtime]"
```

## 验证

```bash
python -m unittest discover -s tests/unit -v
python -m unittest discover -s tests/integration -v  # 在安装 runtime 依赖的环境（如 conda env quantum）运行集成测试
python -m compileall -q ai_qec tests
uv build --wheel --no-build-isolation
openspec validate --all --strict --no-interactive
```

## 开发规则

先阅读 `AGENTS.md` 和相关 OpenSpec。接口存在不等于 capability 可用；以 `openspec/capabilities.yaml` 的状态和 evidence 为准。
