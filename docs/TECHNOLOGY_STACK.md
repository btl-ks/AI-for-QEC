# AI-QEC 技术栈决策

> 决策日期：2026-09-15
> 本文定义目标技术栈、外部工具边界和实施顺序。当前已经可执行的能力仍以 [PROJECT_DESIGN.md](PROJECT_DESIGN.md) 为准，具体任务以 [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md) 为准。

## 1. 已确认的技术决策

| 层 | 选型 | 在项目中的角色 |
|---|---|---|
| 主语言 | Python | 研究 API、数据处理、训练与实验编排 |
| 项目稳定契约 | 自有 `QECProblem`、dataset schema、sampler/decoder protocol | 隔离外部 SDK 类型与版本变化 |
| 稳定子 QEC 电路 | Stim circuit | Phase 1 电路级 memory 实验的首选实现 |
| QEC 交换产物 | `.stim`、Stim DEM、项目 metadata | 保存 detector、observable、噪声和身份信息 |
| 通用电路互操作 | OpenQASM 3；Qiskit adapter | 接入通用电路与 IBM 生态，不进入核心数据模型 |
| CPU 数据生成 | Stim | 默认且必须首先完成的真实 QEC sampler |
| GPU 数据生成 | cuStabilizer DEM sampler | 可选后端；通过能力检查和基准后启用 |
| benchmark 兼容 | Sinter adapter | 与外部 DEM decoder 生态互操作 |
| Surface-code 基线 | PyMatching | 第一项可扩展 MWPM 基线 |
| qLDPC 基线 | `ldpc` / `stimbposd` | BP、BP+OSD、BP+LSD 的后续扩展 |
| AI 训练 | PyTorch | 首选训练框架；GPU 不可用时允许显式 CPU 执行 |
| 第一条神经纵切面 | AI reweighting/predecoder + PyMatching | 保留结构化全局解码，并学习相关噪声或残余 syndrome |
| 直接神经 decoder | detector-graph GNN、recurrent Transformer | 在统一数据与协议下作为后续比较模型 |
| NVIDIA 部署 | CUDA-Q QEC / ONNX / TensorRT adapter | 仅在明确采用 NVIDIA 部署环境后接入 |

这些是项目边界和实施优先级，不表示表中组件已经安装或已经接入。任何配置只能选择 registry 中真实存在的能力；缺少依赖、GPU 或 adapter 时必须在写入数据前失败。

## 2. 核心边界

### 2.1 项目核心不依赖 Qiskit 类型

OpenQASM 项目当前公开版本为 3.1，它适合描述通用量子程序。Qiskit 提供 OpenQASM 3 导入与导出，但官方文档仍将部分导入能力描述为早期或实验性功能。Qiskit Aer 0.17.1 的 GPU 支持表明确列出 statevector、density matrix、unitary 和 tensor network 等方法，而 stabilizer 与 extended stabilizer 不支持 GPU。[OpenQASM 官方仓库](https://github.com/openqasm/openqasm)、[Qiskit OpenQASM 3 API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/2.2/qasm3)、[Qiskit Aer GPU 支持表](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html)

因此，Qiskit 用于 IBM hardware、`QuantumCircuit` 和 OpenQASM 互操作 adapter。项目内部 code、detector、observable、dataset 和 decoder 不能暴露 `qiskit.*` 或 `qiskit_qec.*` 类型。`qiskit-qec` 官方仓库仍警告其处于早期阶段并可能发生破坏性 API 变更，所以它只作为候选 adapter 依赖。[qiskit-qec 官方仓库](https://github.com/qiskit-community/qiskit-qec)

### 2.2 Stim/DEM 是稳定子电路工作流的交换边界

Stim 针对稳定子电路和 QEC 批量采样实现 reference-frame sampling；其 detector sampler 可以同时产生 detection events 与 logical observable flips。DEM 描述错误机制如何翻转 detector 和 logical observable，PyMatching、Sinter、stimbposd 与 CUDA-Q QEC 均可消费或转换这一表示。[Stim 官方仓库](https://github.com/quantumlib/Stim)、[Stim DEM 格式](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md)、[PyMatching Stim 示例](https://github.com/oscarhiggott/PyMatching)

项目不会把 Stim Python 对象作为持久领域模型。稳定的内部契约至少保存：

```text
QECProblem
├── code identity / distance / rounds
├── circuit identity / format / hash
├── detector and observable semantics
├── noise model and effective parameters
├── DEM text or hash（适用时）
└── backend capability and provenance
```

Stim circuit 与 DEM 是 Phase 1 的规范交换产物。Toric code-capacity、一般 CSS/qLDPC parity-check matrix、实验硬件软读出或非 Clifford 精确验证可以使用其他表示，但必须转换到同一 dataset/decoder 契约，不能伪造 DEM。

### 2.3 CPU 与 GPU sampler 使用同一数据契约

Stim CPU 是默认实现。NVIDIA cuStabilizer 26.09 文档提供 Pauli-frame、leakage 和 GPU DEM sampling，并说明其 circuit 格式与 Stim 相近、DEM sampler 通过稀疏 GF(2) 运算生成 detection events。[cuStabilizer overview](https://docs.nvidia.com/cuda/cuquantum/26.09.0/custabilizer/overview.html)、[cuStabilizer 文档入口](https://docs.nvidia.com/cuda/cuquantum/26.09.0/custabilizer/index.html)

GPU 后端必须遵守以下规则：

1. 配置显式选择 `stim_cpu` 或 `custabilizer_gpu`，不得静默切换。
2. 启动前检查包、CUDA、可见 GPU 和目标功能；检查失败不创建 dataset。
3. CPU/GPU 对相同 circuit、DEM、seed 语义和输出 schema 保持一致；统计等价性由测试验证，不要求逐 bit 相同。
4. 以 `distance × rounds × shots × noise model` 进行吞吐、内存与总耗时基准，小任务不预设 GPU 更快。
5. dataset manifest 记录 backend、版本、设备、batch、seed 和 circuit/DEM hash。

CUDA-Q QEC 已提供 DEM sampling、PyMatching、GPU qLDPC、TensorRT decoder 等接口，证明这些 adapter 可以共享 DEM/PCM 边界；它仍属于可选 NVIDIA 集成，不成为项目核心依赖。[CUDA-Q QEC decoder 文档](https://nvidia.github.io/cudaqx/components/qec/decoders.html)

### 2.4 项目 decoder protocol 是 Sinter 的超集

Sinter 的 `Decoder.compile_decoder_for_dem` 接收 `stim.DetectorErrorModel`；编译后的 decoder 接收 bit-packed detection events 并返回 predicted observable flips。这很适合公平比较 DEM decoder。[Sinter API](https://github.com/quantumlib/Stim/blob/main/doc/sinter_api.md)

项目自己的 decoder protocol 还需要支持 code-capacity recovery chain、soft readout、leakage、calibration/domain context 和混合 decoder 中间结果，因此保留自己的稳定接口：

```text
prepare(problem, decoder_config) -> CompiledDecoder
decode_batch(detector_events, side_information) -> DecoderOutput

DecoderOutput
├── predicted_observable_flips
├── optional recovery / residual syndrome
├── convergence / timeout
└── decoder-specific diagnostics
```

当输入是 DEM 与 hard detector bits 时提供双向 Sinter adapter。所有 decoder 必须在同一 dataset identity、shot 集、observable truth 和 LER 定义下比较。

### 2.5 神经 decoder 的实现顺序

第一条电路级神经纵切面采用 AI edge-weight estimator 或局部 predecoder，再调用 PyMatching。这个选择便于单独验证 AI 输出、residual syndrome 和全局 decoder，也能与纯 PyMatching 共用相同输入。

直接神经 decoder 保留两条研究路线：

- detector-graph GNN：M. Lange et al. 在电路级 surface-code 噪声上研究了 graph-classification decoder，并报告其推理随 code 的时空体积近似线性扩展。M. Lange et al., “Data-driven decoding of quantum error correcting codes using graph neural networks,” *Physical Review Research* 7, 023181 (2025), [DOI](https://doi.org/10.1103/PhysRevResearch.7.023181)。
- recurrent Transformer：AlphaQubit 使用模拟数据预训练和实验数据微调，并使用 soft readout 与 leakage 信息。J. Bausch et al., “Learning high-accuracy error decoding for quantum processors,” *Nature* 635, 834–840 (2024), [DOI](https://doi.org/10.1038/s41586-024-08148-8)。

NVIDIA 在 2026 年发布了 AI predecoder 与 PyMatching 组合的厂商研究结果，可作为部署候选的工程证据；其延迟和 LER 数字只适用于文中硬件、模型、code 和测试条件，不能直接写成本项目性能目标。[NVIDIA Research: Fast AI-Based Pre-Decoders for Surface Codes](https://research.nvidia.com/publication/2026-04_fast-ai-based-pre-decoders-surface-codes)

## 3. Dataset schema 必须保留的信息

真实电路级数据不能只保存 `(syndrome, label)` 或摘要特征。schema v2 至少包含：

| 类别 | 必需字段 |
|---|---|
| 样本 | bit-packed `detector_events`、bit-packed `observable_flips` |
| 形状 | shots、detector 数、observable 数、bit order、round layout |
| QEC | code family、distance、rounds、basis、protocol ID |
| 物理 | noise model ID、有效参数、采样单位 |
| 身份 | circuit hash、DEM hash（适用时）、schema/generator version |
| 生成 | backend、设备、seed、batch/shard、时间与环境引用 |
| 划分 | split、session/domain/pair ID 与泄漏审计信息 |

以下字段按后端能力可选，但 schema 从第一版开始保留扩展位置：

- soft readout probability 或原始 I/Q 引用；
- leakage/herald flags；
- detector、qubit、edge coordinates/features；
- calibration ID、device ID、timestamp、domain ID；
- 可重建的 physical error/recovery chain（仅模拟或特定论文需要）。

摘要、tensor、token 和 graph 均是从 raw schema 重建的 preprocessing view。任何不可重建的转换都必须生成带输入 hash 的新 dataset identity。

## 4. 与当前仓库的关系

| 能力 | 当前状态 | 下一落点 |
|---|---|---|
| Toy synthetic flow | 可执行 | 保留为 plumbing 测试 |
| Toric code-capacity RBM 论文路径 | PyTorch RBM + 单链 Gibbs 的可执行 smoke；直接相位翻转采样与内置精确 MWPM | 独立扩展论文网格与正式 protocol |
| Toric RBM 并行采样路径 | 可执行 CPU smoke；CUDA 需显式配置并通过设备检查 | 作为平台加速实验与单链论文路径分别报告 |
| Stim circuit / DEM / raw events | 未接入 | P1.1–P1.4 |
| PyMatching scalable MWPM | Toric code-capacity adapter 已接入；Stim DEM 版尚未接入 | P1.5 |
| Sinter compatibility | 环境与项目均未接入 | P1.9 |
| PyTorch circuit-level model | 项目未接入；论文 RBM 已使用 PyTorch | Phase 3 |
| Qiskit/OpenQASM adapter | Qiskit 环境可用，项目未接入 | P5.5 |
| cuStabilizer/CUDA-Q GPU sampler | 当前环境未安装且当前会话无可用 GPU | P5.4 |

上表中的环境状态来自 2026-09-15 对 Conda `quantum` 环境的 project-local 探针，不是仓库可移植能力。正式能力以 optional dependency、registry、测试和 run manifest 为准。

## 5. 实施顺序

1. 实现 Stim CPU circuit-level sampler、schema v2 和 PyMatching LER，完成 G1。
2. 固定项目 sampler/decoder protocol，并增加 Sinter compatibility adapter。
3. 在同一 detector dataset 上实现 PyTorch hybrid decoder，随后比较 GNN 与 recurrent Transformer。
4. 在有真实需求时增加 OpenQASM/Qiskit circuit adapter；adapter 必须显式提供 detector/observable 语义。
5. 有可用 NVIDIA GPU 后实现 cuStabilizer backend，并与 Stim CPU 做等价性和吞吐 crossover 基准。
6. 只有实时部署目标明确后才增加 CUDA-Q QEC、ONNX/TensorRT 和 C++/CUDA data plane。

## 6. 明确不采纳的做法

- 不以 Qiskit `QuantumCircuit` 或 `qiskit-qec` 类作为项目核心 domain model。
- 不用 Aer statevector GPU 批量生成稳定子 surface-code 训练数据。
- 不把 Sinter 当作唯一 decoder 接口，也不重复实现它已经稳定提供的兼容协议。
- 不因检测到 GPU 就自动切换 sampler。
- 不把厂商 benchmark 数字写成项目性能结论。
- 不丢弃 observable truth、circuit/DEM identity、soft information 或 domain provenance 后再生成摘要数据。

## 7. 项目特有逻辑与成熟库的分工（2026-09-19）

继续采用模块化单体；`ai_qec/` 定义稳定的领域契约，Notebook 和 CLI 只负责配置与编排。外部库经 adapter 接入，不让其对象成为数据集或 checkpoint 的持久身份。当前能力与目标能力仍以第 4 节区分。

| 工作 | 采用成熟实现 | 项目需要保留的实现与验收 |
|---|---|---|
| 电路级采样与经典解码 | Stim 生成电路、DEM 和 detector 样本；PyMatching 批量 MWPM；Sinter 负责兼容基准接口 | `QECProblem`、sampler/decoder protocol、observable 语义、统一 LER 与 provenance；同一批样本交叉校验。见 P1.2–P1.9。[Stim](https://github.com/quantumlib/Stim)、[PyMatching](https://github.com/oscarhiggott/PyMatching)、[Sinter API](https://github.com/quantumlib/Stim/blob/main/doc/sinter_api.md) |
| Toric code-capacity 论文路径 | PyTorch tensor、`Dataset`/`DataLoader`、优化器和模块 | 论文特有的错误链、同调判定、RBM 能量、CD-k、Gibbs 更新及其正确性测试；不要把电路级 DEM 假装成该论文的数据模型。现有批次收拢见 P4.12。 |
| 数据与训练 | PyTorch 管理模型、批次和优化步骤；大数据按 shard 加载 | schema、按生成规格标识的数据集、无泄漏 split、checkpoint 身份与随机状态、恢复一致性。见 P0.14/P0.16、P1.3/P1.4、P3.5。 |
| 实验产物 | 标准库负责路径、哈希和原子写入 | `dataset_manifest.json` 与 `run_manifest.json` 的项目契约、完整终态和阶段产物不可变；续跑关系见 P0.15/P0.17–P0.19。 |

## 8. Python 包管理决策（2026-09-19）

目标使用标准 `pyproject.toml` 声明包与依赖，使用 **uv** 管理项目环境、依赖锁、同步、运行和构建；现有 setuptools 继续充当 wheel 构建后端。[Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)、[uv 项目与锁文件](https://docs.astral.sh/uv/concepts/projects/sync/)、[uv 命令概览](https://docs.astral.sh/uv/getting-started/features/)

当前 WSL 仓库状态：`pyproject.toml` 是依赖声明唯一来源，已声明 `[sim]`、`[torch]`、`[plot]`、`[notebook]`、`[matching]` extras 和 `dependency-groups.dev`；已提交 `uv.lock`，并已用 `uv lock --check`、全组合 dry-run 和 `uv build --wheel` 验证。脚本仍直接注入 `sys.path`，属于 P4.1 CLI 待办；现有 Conda `quantum` 环境继续保留。

落地规则：

1. `pyproject.toml` 是直接依赖的唯一真相源：`[sim]` 放 Stim/PyMatching/Sinter，`[torch]` 放训练运行依赖；开发工具放 `dependency-groups.dev`，Notebook 的运行依赖也要显式归组。清理或自动生成 `requirements.txt`，不能手工维护两套版本。
2. 提交并审查 `uv.lock`。WSL fresh environment 用 `uv sync --locked` 安装所选 extras/groups；`uv build` 生成 wheel，P4.1 继续负责从 wheel 安装后的 `ai-qec` CLI 验证。锁文件解决 Python 依赖重建，dataset/run/checkpoint 的身份仍由各自 manifest 记录。
3. 先固定支持矩阵（Python 版本、WSL/Linux CPU、实际要支持的 CUDA 构建）并验证 PyTorch wheel 来源；uv 不管理 GPU 驱动。已有 `execution.conda_env: quantum` 仍是独立执行方式，不能把 uv 的 `.venv` 当作该 Conda 环境；迁移前分别做 CPU/CUDA smoke。[uv 的 PyTorch 指南](https://docs.astral.sh/uv/guides/integration/pytorch/)
4. 正式 run 与论文导出记录并校验 lock hash、解释器和关键包/设备版本，连同代码 commit、配置、seed 和 dataset identity 保持可追溯。此项与 P0.10、P4.10 对齐。

实施顺序及可验收任务见 [优化计划的近期工作总览](OPTIMIZATION_PLAN.md#近期工作总览2026-09-19)。当前单包仓库无需 uv workspace；当出现独立发布的多个 Python 包时再评估。

## 9. 规格驱动工作流（2026-09-19）

项目使用 repo-local [OpenSpec](../openspec/) 管理目标行为与变更历史。活动需求基线位于 [`establish-qec-research-platform-requirements`](../openspec/changes/establish-qec-research-platform-requirements/proposal.md)，当前实现状态仍以 [PROJECT_DESIGN.md](PROJECT_DESIGN.md)、capability registry 和通过的测试为准。

OpenSpec 属于开发工作流工具，不是 `ai_qec` 的运行依赖：

1. OpenSpec CLI 所需的 Node 环境不进入 `pyproject.toml`、Python wheel、正式 run 或论文复现包。
2. `openspec/specs/` 保存归档后的稳定行为；`openspec/changes/` 保存 proposal、delta specs、design 和 tasks。
3. 外部库、格式和 backend 选型仍由本文维护；OpenSpec 只规定用户和下游系统可验证的行为。
4. Python、C++/CUDA、FPGA/ASIC 的内部实现可以演进，但必须满足相同 dataset、decoder、experiment 和 evidence 合同。
5. 新 capability 的目录或 spec 存在不代表已经实现；只有对应 Gate、测试和状态清单通过后才能标为 validated。
