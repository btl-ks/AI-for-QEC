# 论文复现的目录与调用设计

## 目标边界

`paper/` 是论文材料与实验 Notebook 的入口。每篇论文的 Notebook 是该复现实验的程序入口，负责最上层控制流、产物核对和结果展示。可复用的 QEC、数据、模型、训练、评估能力由主项目 `ai_qec/` 提供；Notebook 不定义这些能力的第二份实现。

```text
paper/
├── docs/
│   └── <论文标题>.pdf              # 仅存论文原文及补充材料 PDF
├── srcs/
│   └── <paper_id>.ipynb           # 每篇论文一个实验 Notebook；不放算法 .py
├── templates/                    # 既有论文导出模板
└── releases/                     # 既有不可变导出包，属于生成产物

ai_qec/                        # 共用、可测试的功能实现
configs/                           # 脚本 runner 的配置；Torlai–Melko Notebook 参数直接写在单元格中
datasets/                          # 可复用、内容寻址且经 manifest 验证的数据
runs/<run_id>/                     # 正式运行的快照、日志、指标、checkpoint、预测和图表
docs/                              # 项目设计文档；paper/docs 不放项目说明
```

其中 `paper/templates/` 和 `paper/releases/` 保留当前导出机制。`paper/docs/` 只存论文文件；Notebook 的使用说明放在主目录 `docs/` 下。

## Notebook 入口与实现边界

每篇论文的 Notebook 应：

1. 引用原论文，声明独立复现的范围、参数来源和不确定项；在配置单元格中定义或加载参数并严格校验，以实际运行配置作为参数、实验网格和随机种子的真相源。
2. 创建独立 run，按配置执行数据生成、训练、解码、benchmark 和报告阶段。Notebook 可以直接展开 split、batch、epoch、minibatch、采样步及测试样本层级的循环，使论文算法的步骤和执行顺序可见。
3. 在循环体中调用 `ai_qec/` 的库函数完成物理采样与 syndrome、模型单步更新/采样、恢复链兼容性判定、数据写入/校验、checkpoint 元数据和指标构建。Notebook 只决定何时调用，不重写这些操作的语义。
4. 核查 dataset manifest、checkpoint、metrics、predictions 和 run 状态，并在同一测试数据身份下展示结果；保存的图表也应记录在对应 run 中。

`scripts/run_experiment.py` 是通用配置和批量实验的另一入口。它与论文 Notebook 可以有不同的上层编排，但必须复用同一套库函数、配置校验、数据/schema 契约、checkpoint 身份检查及 benchmark 指标定义；两条入口不能各自维护一套算法或指标公式。论文特有的控制流可以留在 Notebook，不要求藏进通用 runner。若某段控制流需要被多个入口复用，再将其提取为 `ai_qec/` 中有明确调用方的函数。

Notebook 不应在单元格中定义 `ToricCode`、RBM 结构、CD-k 单步算法、Gibbs 条件采样、syndrome/同调/逻辑失败率公式、MWPM 算法或 checkpoint/schema 序列化规则，关键参数应集中放在明确标注的配置单元格中，并保存到每次 run 的配置快照。验收时从清空的 kernel 按顺序执行全部单元格，检查最终 `run_manifest.json`、非空数据及其身份、checkpoint 和 benchmark 产物；乱序执行或残留变量不能成为成功的前提。阶段失败或中断必须在 run 状态中显式记录，不能留下看似成功的结果。

## Torlai–Melko (2017) 的功能映射

论文：G. Torlai and R. G. Melko, “Neural Decoder for Topological Codes,” *Physical Review Letters* **119**, 030501 (2017), https://doi.org/10.1103/PhysRevLett.119.030501。该项目拟实现的是论文方法的独立复现，不声称拥有作者的官方源码或最终参数配置。

| 功能 | 主项目归属 | Notebook 调用或展示 |
|---|---|---|
| 周期方格 Toric code、边编号、syndrome、同调 | `ai_qec/qec/codes/toric_code.py`，必要时 `qec/detectors/` | 仅调用一致性检查并展示几何示例 |
| 独立 `Z` 相位翻转噪声 | `ai_qec/qec/noise/phase_flip.py` | 从配置选择错误率网格 |
| code-capacity 物理采样 | `ai_qec/qec/simulators/toric_backend.py` | 启动生成步骤并核对真实样本数 |
| 原始错误链与 syndrome schema、分片、manifest | `ai_qec/data/datasets/toric_dataset.py`、`data/generators/toric_generator.py` | 读取数据身份与形状，不手写 NPZ 格式 |
| `(e,S)` 二进制输入构造 | `ai_qec/data/datasets/toric_dataset.py` | 显示少量样本/统计 |
| RBM 联合模型、条件概率、参数保存 | `ai_qec/models/decoders/generative/rbm.py`，`models/registry.py` | 从配置选择模型并载入 checkpoint |
| CD-k 单步更新、checkpoint 元数据与选择规则 | `ai_qec/training/trainers/rbm.py` 及 `training/` 相关模块 | 组织 epoch/minibatch 循环并展示训练记录 |
| 固定 syndrome 的 Gibbs 单步采样、兼容性判定与 timeout 契约 | `ai_qec/models/decoders/generative/rbm_decoder.py` | 组织采样步/样本循环并显示 timeout 统计 |
| 周期边界 MWPM 对照 | `ai_qec/models/decoders/classical/mwpm.py` | 在同一测试集上调用对照 |
| `e XOR r` 同调失败率、区间、Fig. 3/4 数据 | `ai_qec/benchmarks/decoding/` | 绘制曲线与同调直方图 |
| 论文配置、网格、运行参数和导出 allowlist | Notebook 内联配置、`ai_qec/utils/config.py`、`scripts/export_paper.py` | 校验内联配置并读取 `run_manifest.json`；脚本 runner 的独立配置仍在 `configs/` |

模型选择（论文补充材料中的超参数 grid search）属于训练/实验配置与评估协议，应在主项目建立可追溯的候选 run 和选择规则；Notebook 只调用该规则并展示各候选结果。不要把论文超参数搜索与 `design/decoder_search` 的 QEC decoder 设计搜索混为同一个功能。

## 正式实验的数据流

```text
paper/srcs/torlai_melko_2017.ipynb（程序入口与最上层逻辑；每个 L × p_error × seed 一个可追溯 run）
    ├── RAW_CONFIG → resolve_experiment_spec → RunRecord.create：runs/<run_id>/、config.yaml 与 run_manifest.json
    ├── 数据生成循环：noise.sample_errors → code.syndrome → write_toric_split → validate_toric_dataset（Algorithm 1 第 1–2 行）
    ├── 训练循环：epoch/minibatch → RBM.contrastive_divergence_step → checkpoint
    ├── 解码循环：RBM.sample_hidden / sample_error → first_compatible_chain（Algorithm 1 第 3–8 行）
    ├── benchmark 循环：mwpm_reference_recoveries（精确 MWPM，超限时 PyMatching）→ build_toric_benchmark_report
    └── 结果表、Fig. 3/4 风格图、异常与局限说明
```

每次执行 Notebook 都分配唯一 run ID，配置快照、环境、指标、checkpoint 与各阶段状态和产物 sha256 归属该 run，不覆盖旧结果。生成数据不得用空文件、零样本数组或占位 manifest 代替。

## 已实现的代码归属

论文代码按以下归属实现；Notebook 是入口并承载最上层逻辑，其下细节全部在 `ai_qec/`，`paper/srcs/` 不保留算法 `.py`：

| 论文复现所需内容 | 目标位置或处理 |
|---|---|
| 程序入口、最上层逻辑与展示 | `paper/srcs/torlai_melko_2017.ipynb`；阶段顺序与数据生成/训练/Gibbs 解码/benchmark 循环，调用下列库函数 |
| run 记录 | `ai_qec/utils/run_record.py` |
| Toric code | `ai_qec/qec/codes/toric_code.py`，已接入 code registry |
| 数据 schema、生成与加载 | `ai_qec/data/datasets/toric_dataset.py`、`data/generators/toric_generator.py` |
| RBM 模型与 Gibbs 解码 | `ai_qec/models/decoders/generative/` 提供模型与单步采样；`training/trainers/rbm.py` 提供 CD-k 单步更新和脚本入口复用的训练器，论文 Notebook 展开自己的上层循环 |
| MWPM | `ai_qec/models/decoders/classical/mwpm.py` 的精确实现有 defect 上限；`ai_qec/benchmarks/decoding/toric.py` 的 `mwpm_reference_recoveries` 对超过上限的 syndrome 改用 PyMatching（同为最小权重完美匹配），报告记录 `pymatching_fallback_shots` |
| 复用已训练模型重新解码 | `ai_qec/utils/source_run.py`：按源 run 配置校验并引用其数据集，核对 `best.pt` 记录的哈希与训练数据集，并在共同步数预算内逐条核对解码结果；Notebook 通过 `REUSE_CHECKPOINT_FROM_RUN` 跳过训练 |
| 逻辑失败率与论文 benchmark | `ai_qec/benchmarks/decoding/toric.py` |
| 论文实验说明 | 本文档；`paper/docs/` 只保留 PDF |

实现已对齐主项目的配置、schema、registry、训练、benchmark 与 run manifest 契约；持续验证包括物理一致性测试和从零执行的 smoke pipeline。

当前论文 Notebook 的内联配置使用 PyTorch 联合 RBM、CD-k、单链 syndrome-clamped Gibbs 与内置精确 Toric MWPM；模型创建和 checkpoint 加载都通过项目 registry。若在配置单元格中将 `training.decoder.parallel_chains` 改为 64，可作为单独的并行链平台实验运行。并行链改变候选恢复链的选择过程，其指标必须与单链论文 smoke 分别标注；CUDA 仅在显式选择且可用时执行。

论文路径与平台扩展共享 `ai_qec/` 中的代码、数据身份和 `DecodeRequest` / `DecodeResult` 契约，但实验配置和 benchmark 解释各自独立：

| 路径 | 当前选择 | 后续扩展 |
|---|---|---|
| 论文单链复现 | 自研 Toric code、独立 `Z` 翻转、PyTorch 联合 RBM、单链 Gibbs、精确 Toric MWPM | 预注册 `L × p_error × seed` 网格和论文规模的非空数据；保持每个错误率单独训练 |
| 平台加速验证 | 同一 Toric 物理问题、显式 64 链 PyTorch Gibbs、同一内部 decoder 契约 | 有可用 GPU 时单独评估 CUDA 吞吐、端到端延迟和恢复链分布 |
| 电路级平台 | 已有可选 Toric code-capacity PyMatching adapter；尚未接入 Stim 电路 | Stim/DEM、DEM 版 PyMatching、Sinter adapter 和多轮噪声按优化计划分别实现 |

现阶段没有论文规模的 `P_fail(p_error)` 曲线，也没有 Stim/DEM adapter 或正式规模的 PyMatching 对照 run；项目不将 smoke 指标当作论文性能结论。

## 与现有优化计划的关系

此论文的 toric code/code-capacity 路径不同于 Phase 1 计划的 rotated-surface-code memory circuit。两者可共享数据 schema、checkpoint 和 benchmark 基础设施，但不能把现有 toy baseline 或未来 Stim memory 结果当作这篇论文的物理实验。P4.0 只验收论文 smoke 路径与代码归属；正式论文网格、结果比较及 Phase 1–3 的验收仍需独立任务和正式 run，不能由 smoke 的成功状态推定。
