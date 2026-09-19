# Proposal

## Why

项目已有研究主题、架构、技术栈和优化计划，但需求分散在多份文档中，缺少统一、可验证、可由实现任务持续追踪的行为规格。现在需要把研究平台从“论文复现脚本集合”收敛为一组明确的能力契约，使训练数据生成、解码研究、性能部署和 AI 自动验证能够按同一套验收标准演进。

## Change Type and Required Children

本 change 是 umbrella requirement baseline，只建立跨项目目标行为、依赖顺序和拆分边界，不直接授权实现。任何业务或工程实现都必须先进入以下可独立验收的 child change：

| Child change | 主要范围 | 独立验收边界 |
| --- | --- | --- |
| `implement-experiment-lifecycle-integrity` | 实验身份、run 终态与产物不可变 | 可证明 dataset 身份复用、明确终态与已登记产物的哈希一致性 |
| `implement-stim-dataset-pipeline` | Stim、raw schema、shard、split、batch 与预取 | 可生成并审计真实非空数据，检测泄漏且稳定供给训练 |
| `integrate-pymatching-unified-decoder` | PyMatching、Sinter adapter、decoder contract 与 LER | 同一批 shots 上得到可追溯预测、LER 和置信区间 |
| `build-neural-training-workbench` | PyTorch trainer、模型 registry、checkpoint、神经与混合 decoder、数据高效训练 | 配置真实生效，并在冻结协议下完成公平对照 |
| `establish-engineering-delivery` | wheel/CLI、CI、Docker、benchmark 与按证据触发的性能后端 | 干净环境可安装、运行 smoke、导出最小复现包并报告性能边界 |
| `establish-research-validation-protocol` | claim、protocol、统计有效性、证据图与自动验证 | 从来源到结论全链路可追溯，预算或证据不足时产生明确非成功结论 |

每个 child change 必须单独提供 proposal、spec delta、design、tasks、自动化测试与 verification.md。若某个 child 仍无法在一次评审和验收中完成，应继续按 gate 或纵向能力拆分，不得直接从本 umbrella tasks 开始实现。

## What Changes

- 建立实验生命周期契约，统一配置解析、不可变数据与 run、产物身份、失败语义和论文导出。
- 建立 QEC 数据管线契约，以 Stim CPU 为首个可信后端，允许经过能力检查和等价性验证的 GPU 后端，并明确 split、batch、预取、困难样本与数据 provenance。
- 建立统一解码工作台契约，使 PyMatching、RBM、混合解码器和直接神经解码器在相同样本、真值、协议和统计口径下训练与评估。
- 建立数据高效训练契约，支持固定样本预算下的困难样本挖掘、课程学习、采样偏差控制和跨噪声/码距泛化评估。
- 建立性能与部署契约，分别测量模型、decoder 和端到端时延，并为 C++/CUDA、TensorRT、FPGA/ASIC 接入保留稳定边界。
- 建立 AI 辅助论文和想法验证契约，把论文主张或研究假设转化为带来源、基线、协议、停止规则和证据等级的可复现实验。
- 明确分阶段交付顺序：先完成实验诚信与真实电路级纵切面，再完成神经训练和数据高效研究，最后开展自适应、实时部署和自动研究闭环。
- 不在本变更中声明尚未实现的能力已经可用；配置选择未实现能力时仍须 fail loudly。

## Capabilities

### New Capabilities

- `research/experiment-lifecycle`: 配置、dataset、run、checkpoint、产物和论文导出的可追溯生命周期。
- `qec/dataset-pipeline`: 电路级数据生成、schema、split、batch、预取、CPU/GPU sampler 和采样策略。
- `qec/decoder-workbench`: 经典、RBM、混合与神经解码器的统一接口、训练、评估和公平对照。
- `research/data-efficient-training`: 稀有逻辑错误下的困难样本、课程学习、采样偏差控制和样本效率评估。
- `deployment/performance-runtime`: 解码性能测量、异构设备边界和后期原生实时运行时。
- `research/automated-validation`: AI 辅助的论文复现与研究想法验证流程及证据约束。

### Modified Capabilities

无。当前 OpenSpec 主规格为空，本变更建立首个需求基线。

## Impact

- 主要影响 `openspec/`、`docs/` 中的需求职责，以及后续对 `ai_qec/`、`configs/`、`scripts/`、`tests/` 和论文 Notebook 的实现安排。
- 保持 Python 作为研究控制层，复用 Stim、PyMatching、Sinter 和 PyTorch；只有 profiling 证明必要时才增加 C++/CUDA 数据层。
- 后续实现必须继续遵守现有学术引用、真实非空数据、fail-loudly、公开 API 注册和 Git/任务 ID 规则。
- 本变更跨度较大，实施时按 capability 和 gate 分批交付；任何后期能力不得绕过其前置验收。
