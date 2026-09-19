# Design

## Context

参见 [proposal.md](proposal.md) 的动机。当前仓库已经具有模块化 Python 包、配置驱动 runner、不可变 dataset/run 的部分基础、Toric RBM smoke、uv 锁文件和多份职责明确的项目文档；真实 Stim 电路级纵切面、可恢复正式训练、统一神经对照、实时部署与自动研究代理仍处于计划状态。

本设计必须遵守以下已有约束：

- 学术定论需要论文引用，项目结果必须标明适用协议和证据等级。
- 不生成空数据、占位数据或静默成功结果。
- 外部库通过项目稳定契约接入，持久化产物不依赖外部 SDK 对象。
- 当前形态保持模块化单体；训练和研究控制以 Python 为主。
- 正式实验的配置、数据、run、checkpoint、统计协议和导出材料必须可追溯。

六份 capability spec 是目标行为合同；`PROJECT_DESIGN.md` 继续描述当前实现，`TECHNOLOGY_STACK.md` 记录外部选型，`OPTIMIZATION_PLAN.md` 记录执行状态，`RESEARCH_TOPICS.md` 维护研究主题。

## Goals / Non-Goals

**Goals:**

- 用 OpenSpec 建立唯一的目标行为需求源，并把现有 P0–P5 任务映射到可测试场景。
- 固定跨 sampler、decoder、trainer、benchmark 和部署后端的稳定数据语义。
- 建立从可信电路级基线到数据高效算法研究，再到实时部署与 AI 自动验证的依赖顺序。
- 允许 CPU、GPU 和未来原生/硬件实现替换内部实现，而不改变数据和解码语义。
- 让每项科研结论都可从协议和原始产物重算。

**Non-Goals:**

- 本次规划不实现 Stim、神经 decoder、GPU sampler、C++ runtime 或 FPGA/ASIC 设计。
- 不完整复现 AlphaQubit，也不预设任何学习方法必须优于 PyMatching。
- 不把单一厂商 benchmark 转化为本项目性能承诺。
- 不在真实部署需求出现前引入微服务、分布式调度或独立 C++ 产品树。
- 不用 OpenSpec 取代论文、架构说明或运行记录；OpenSpec 只承担行为合同和变更历史。

## Baseline Decomposition and Trace

本 umbrella 只保存跨 capability 的目标基线。实现追踪在 proposal 指定的 child change 中闭环，映射如下：

| Capability baseline | Child change | 本文件任务库存 | 必需的验收证据 |
| --- | --- | --- | --- |
| `research/experiment-lifecycle` | `implement-experiment-lifecycle-recovery` | tasks 2 | 状态机、故障注入、恢复等价性与产物哈希 |
| `qec/dataset-pipeline` | `implement-stim-dataset-pipeline` | tasks 3 | 真实数据 manifest、schema/shard/split 审计与供数 smoke |
| `qec/decoder-workbench` | `integrate-pymatching-unified-decoder` | tasks 4.1–4.2 | 同 shots 预测、LER、置信区间与 protocol ID |
| `research/data-efficient-training` 及神经训练部分 | `build-neural-training-workbench` | tasks 4.3–5.7 | 可恢复训练、统一对照、固定预算与多 seed 统计证据 |
| `deployment/performance-runtime` 及工程交付 | `establish-engineering-delivery` | tasks 6、8.2–8.4 | fresh install、CLI/CI/Docker smoke、分层 benchmark 与等价性检查 |
| `research/automated-validation` 及统计协议 | `establish-research-validation-protocol` | tasks 7、8.1 | claim→protocol→run→evidence 追踪与 evidence-insufficient 路径 |

child change 的 design 必须把稳定 requirement/scenario 名称映射到组件、API、状态、失败语义和测试；tasks 必须继续映射到实现位置和实际证据。上表只负责分解路由，不能替代 child change 内的详细 trace matrix。

## Decisions

### 1. OpenSpec 作为目标行为真相源

归档后的 `openspec/specs/` 维护系统 SHALL/MUST 做什么。现有文档通过链接说明原因、现状、选型和进度，不再复制完整需求正文。

**理由：** 当前重复信息跨文档漂移，场景化规格可以直接映射为测试和验收。

**替代方案：** 继续只维护 `OPTIMIZATION_PLAN.md`。该文件适合任务顺序和状态，但不适合长期保存稳定行为合同。

恢复执行采用新 run，并通过 `resumed_from` 指向原 interrupted run；原 run 和原 checkpoint 保持不可变。这样保留每次进程执行的独立身份，也能重建完整中断与续跑链。

### 2. 按 gate 纵向交付，而不是按目录横向铺开

实施顺序固定为：

1. G0：实验生命周期、身份、恢复和产物不可变。
2. G1：Stim CPU、raw detector/observable schema、PyMatching 和 LER。
3. G2/G3：科研 split、统计协议、统一训练与 decoder 对照。
4. Thesis slice：数据高效训练与稀有困难样本。
5. G5：持续适应、性能部署和可选 GPU/native backend。
6. Research automation：在前述合同稳定后自动生成和执行验证协议。

每个 gate 都必须形成一条可运行纵切面，后续 gate 不能用占位实现绕过前置条件。

**理由：** 数据、真值和评估尺子错误会使上层模型结果失去意义。

**替代方案：** 同时开发所有模块。它会扩大接口返工并增加错误结果被误读的风险。

### 3. 使用四个稳定边界隔离外部库

核心合同保持为：

- `QECProblem`：code、circuit、detector、observable、noise 和身份。
- Dataset contract：raw samples、side information、split 和 provenance。
- Decoder contract：prepare/decode request、observable prediction 与 diagnostics。
- Experiment contract：resolved plan、run timeline、artifact graph 和 protocol ID。

Stim、PyMatching、Sinter、PyTorch、CUDA-Q QEC 或其他实现通过 adapter 消费这些合同。外部对象可以在进程内使用，但不得直接成为持久化身份。

**理由：** 研究平台需要替换后端和比较算法，同时保证数据和指标含义不变。

### 4. CPU 生成与 GPU 训练采用异步流水线

第一条可信数据路径使用 CPU sampler，通过多 worker、分片、有限预取队列、pinned memory 和异步复制向 GPU 训练供数。困难样本评分在模型所在设备执行；参考 decoder 可以异步评估。

只有 profiling 显示队列持续枯竭且 CPU 扩展无效后，才增加 GPU sampler。GPU sampler 必须显式选择并完成统计等价性验证。

**理由：** 单 GPU 同时生成和训练可能争用计算资源，成熟 CPU sampler 也已经提供经验证的物理语义。

**替代方案：** 立即手写 GPU 仿真器。该方案增加正确性和维护成本，且缺少已测量收益。

### 5. 数据高效训练采用“固定预算 + 原分布测试”协议

训练侧允许候选池、困难样本缓存、RBM 能量、模型损失/置信度和跨码距课程。所有策略必须：

- 保留均匀原分布样本；
- 记录选择概率和实际分布；
- 在固定独立测试集上评估；
- 使用相同模型和预算比较；
- 报告候选生成成本与训练成本；
- 通过消融区分评分器、缓冲区和权重修正的贡献。

RBM 是可替换评分器和现有研究基线，不作为预设成功条件。

**理由：** 该设计形成可证伪的算法问题，并复用现有代码而不把论文结论绑定到单一模型。

### 6. Python 控制层与原生数据层分离

Python 继续负责配置、训练、实验、统计和 Notebook。性能热点依次尝试成熟库、向量化、`torch.compile`/图优化和部署引擎。仍不能满足目标时，使用 CMake 管理的 C++/CUDA 模块，经 pybind11 或 PyTorch extension 接入；实时部署可形成独立 C++ 进程。

原生层优先承载：

- syndrome 位打包与转换；
- 固定内存池和环形队列；
- 融合 CUDA kernel；
- TensorRT 调用；
- FPGA/ASIC 主机通信和 Pauli-frame 更新。

**理由：** 当前 Python API 有利于研究迭代，底层库已经在原生代码中执行主要计算。稳定协议和 profiler 应先于自定义原生实现。

### 7. 准确率、性能和端到端系统分别验收

评估分为：

- 科研正确性：固定测试集上的 LER、置信区间和泛化。
- 组件性能：预生成输入上的 model/decoder latency 和 throughput。
- 系统性能：包括生成或 readout、传输、预处理、解码、后处理和反馈的 T_E2E。
- 硬件证据：实测、综合、仿真和投影分别标记。

任何性能优化必须同时运行输出等价或准确率退化检查。

### 8. 自动研究代理采用“来源—协议—执行—证据”状态机

AI 可以抽取论文主张、生成候选协议、调用已有 capability 和撰写报告。正式执行前必须得到完整且版本化的协议；预算耗尽或证据不足产生明确的非成功终态。

代理不得把技术博客当同行评审论文、把 toy 结果外推到硬件，也不得在查看结果后无版本地修改主要判据。

**理由：** 自动化的价值来自加快受约束验证，而不是替代证据和科研决策记录。

### 9. 测试与需求场景一一对应

每个 scenario 至少映射到以下一种验证：

- unit：schema、身份、统计和纯函数；
- integration：真实 backend、adapter、恢复链；
- regression：固定 fixture 与数值容差；
- smoke：正式 runner 的最小纵切面；
- benchmark validation：多 seed、CI、尾延迟和资源报告。

测试元数据记录对应的 capability、requirement 和 scenario 名称，便于检查需求覆盖。

## Risks / Trade-offs

- [范围过大导致长期无法归档] → 按 gate 和 capability 拆分实施提交；每次只应用一条可验收纵切面，未完成能力保持 fail-loudly。
- [OpenSpec 与既有文档再次重复] → 主规格保存行为合同，四份现有文档仅保存现状、选型、研究地图和任务状态，并通过链接引用。
- [目标规格被误读为当前能力] → capability registry、README 和 run manifest 分别标记 planned、available、validated，不以目录或 spec 存在判断已实现。
- [困难样本方法引入选择偏差] → 混入原分布、记录选择概率、保留独立测试集并报告未修正对照。
- [极低 LER 需要不可承受的样本量] → 预登记 failure budget 和上限；达不到时报告证据不足，后续再研究可靠的稀有事件估计方法。
- [CPU/GPU 流水线测量互相污染] → 分离生成、传输、decoder 和 E2E benchmark，并冻结计时边界。
- [没有真实量子硬件或 FPGA] → 允许使用公开数据、软件仿真和综合结果，但严格标注证据类型。
- [原生代码增加双语言维护成本] → 仅迁移 profiler 证明的热点，保持 Python 参考实现和跨后端一致性测试。
- [近期论文使算法新颖性变化] → 正式选题前更新文献矩阵；方法与协议版本化，不把工程功能自动视为论文创新。

## Migration Plan

1. 将本变更的六项 delta spec 评审为首个需求基线，并运行严格验证。
2. 在现有文档中增加 OpenSpec 索引，明确需求、现状、选型、研究主题和任务进度的唯一职责。
3. 先应用 `research/experiment-lifecycle` 中尚缺的 P0.14–P0.19 行为，完成恢复和产物不可变。
4. 应用 `qec/dataset-pipeline` 与 `qec/decoder-workbench` 的 G1 纵切面，完成 Stim→dataset→PyMatching→LER。
5. 在 G1/G2/G3 后实现 `research/data-efficient-training` 的最小论文实验，并冻结协议后执行正式多 seed 运行。
6. 根据 profiler 决定是否应用 GPU sampler、C++/CUDA runtime 或 FPGA adapter；每个后端单独建立 OpenSpec change。
7. 当前置能力稳定后实现 `research/automated-validation`，先自动化已有协议的重复执行，再扩展到论文主张抽取和想法验证。
8. 每个子变更完成后同步主规格、更新现状文档并归档变更；回滚只撤销对应实现和状态，不删除历史规格或实验产物。

## Open Questions

- 首个真实 FPGA 目标板卡、工具链和时钟约束在进入硬件 change 时确定。
- C++ 依赖管理使用 vcpkg 还是 Conan，在首个原生第三方依赖出现时决定；CMake 与 Python wheel 边界保持不变。
- 大数据远程保存采用 DVC、对象存储还是本地内容寻址，在本地 shard 规模达到迁移阈值前决定。
