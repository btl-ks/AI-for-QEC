# 论文复现的目录与调用设计

## 目标边界

`paper/` 是论文材料与实验 Notebook 的入口。可复用的 QEC、数据、模型、训练、评估能力由主项目 `ai_qec/` 提供。Notebook 负责组织实验、核对产物和展示结果，不定义这些能力的第二份实现。

```text
paper/
├── docs/
│   └── <论文标题>.pdf              # 仅存论文原文及补充材料 PDF
├── srcs/
│   └── <paper_id>.ipynb           # 每篇论文一个实验 Notebook；不放算法 .py
├── templates/                    # 既有论文导出模板
└── releases/                     # 既有不可变导出包，属于生成产物

ai_qec/                        # 共用、可测试的功能实现
configs/                           # 每篇论文的实验配置与扫描协议
datasets/                          # 可复用、内容寻址且经 manifest 验证的数据
runs/<run_id>/                     # 正式运行的快照、日志、指标、checkpoint、预测和图表
docs/                              # 项目设计文档；paper/docs 不放项目说明
```

其中 `paper/templates/` 和 `paper/releases/` 保留当前导出机制。`paper/docs/` 只存论文文件；Notebook 的使用说明放在主目录 `docs/` 下。

## Notebook 的职责

每篇论文的 Notebook 可以：

1. 引用论文、声明独立复现范围和已知的不确定参数。
2. 选择并加载项目配置，列出实验网格与随机种子。
3. 通过主项目的公开入口触发数据生成、训练、模型选择、解码和 benchmark；正式运行由 `scripts/run_experiment.py` 建立完整 run 记录。
4. 读取已生成的 dataset manifest、checkpoint 元数据、metrics 和 predictions，核查样本数、身份与状态。
5. 用结果表和图展示论文实验；图表导出由 run flow 中声明的报告步骤完成，Notebook 只读取与展示。

Notebook 不应定义 `ToricCode`、数据生成器、RBM、训练循环、Gibbs 解码器、MWPM、逻辑失败率公式或 checkpoint/schema 写入逻辑。Notebook 也不应把关键参数藏在单元格中；实验参数以主项目配置为真相源。清空 kernel 后应能从头执行，且重新使用或创建真实、非空、可追溯的运行产物。

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
| CD-k 训练、优化器、checkpoint 选择 | `ai_qec/training/trainers/rbm.py` 及 `training/` 相关模块 | 启动训练并展示训练记录 |
| 固定 syndrome 的 Gibbs 恢复链采样与 timeout | `ai_qec/models/decoders/generative/rbm_decoder.py` | 显示采样步数和 timeout 统计 |
| 周期边界 MWPM 对照 | `ai_qec/models/decoders/classical/mwpm.py` | 在同一测试集上调用对照 |
| `e XOR r` 同调失败率、区间、Fig. 3/4 数据 | `ai_qec/benchmarks/decoding/` | 绘制曲线与同调直方图 |
| 论文配置、网格、运行参数和导出 allowlist | `configs/`、`scripts/run_experiment.py`、`scripts/export_paper.py` | 选择配置并读取 `run_manifest.json` |

模型选择（论文补充材料中的超参数 grid search）属于训练/实验配置与评估协议，应在主项目建立可追溯的候选 run 和选择规则；Notebook 只调用该规则并展示各候选结果。不要把论文超参数搜索与 `design/decoder_search` 的 QEC decoder 设计搜索混为同一个功能。

## 正式实验的数据流

```text
paper/srcs/torlai_melko_2017.ipynb
    └── 选择 configs/experiment.torlai_melko_2017.smoke.yaml
        └── scripts/run_experiment.py（每个 L × p_error × seed 一个可追溯 run）
            ├── qec code + noise + backend → data schema/dataset manifest
            ├── model registry → RBM trainer → checkpoint
            ├── RBM Gibbs decoder + MWPM → 同一 test 数据
            └── benchmarks → metrics/predictions/run_manifest.json
    └── 读取 run 产物 → 结果表、Fig. 3/4 风格图、异常与局限说明
```

为防止同一 Notebook 在原地覆盖旧结果，正式 run 由 runner 分配唯一 run ID，配置、日志、指标与 checkpoint 归属该 run。图表导出需要作为 flow 中的报告步骤，记录输入 run ID 与配置/指标 hash 并保存到同一 run 的 `figures/`；Notebook 只读取与显示。生成数据不得用空文件、零样本数组或占位 manifest 代替。

## 已实现的代码归属

论文代码按以下归属实现；`paper/srcs/` 不保留算法 `.py`，避免形成第二套实现：

| 论文复现所需内容 | 目标位置或处理 |
|---|---|
| 实验组织与展示 | `paper/srcs/torlai_melko_2017.ipynb`；只调用主项目入口和读取 run 产物 |
| Toric code | `ai_qec/qec/codes/toric_code.py`，已接入 code registry |
| 数据 schema、生成与加载 | `ai_qec/data/datasets/toric_dataset.py`、`data/generators/toric_generator.py` |
| RBM 模型与 Gibbs 解码 | `ai_qec/models/decoders/generative/`；训练循环位于 `training/trainers/rbm.py` |
| MWPM | `ai_qec/models/decoders/classical/mwpm.py`；当前无外部依赖的精确实现有 defect 上限，正式规模需要经过验证的 scalable matching 后端 |
| 逻辑失败率与论文 benchmark | `ai_qec/benchmarks/decoding/toric.py` |
| 论文实验说明 | 本文档；`paper/docs/` 只保留 PDF |

实现已对齐主项目的配置、schema、registry、训练、benchmark 与 run manifest 契约；持续验证包括物理一致性测试和从零执行的 smoke pipeline。

当前论文 smoke 使用 PyTorch 联合 RBM、CD-k、单链 syndrome-clamped Gibbs 与内置精确 Toric MWPM；模型创建和 checkpoint 加载都通过项目 registry。`configs/experiment.torlai_melko_2017.parallel_smoke.yaml` 另设 64 条并行 Gibbs 链，用于检验平台加速路径。并行链改变候选恢复链的选择过程，其指标必须与单链论文 smoke 分别标注；CUDA 仅在显式选择且可用时执行。

论文路径与平台扩展共享 `ai_qec/` 中的代码、数据身份和 `DecodeRequest` / `DecodeResult` 契约，但实验配置和 benchmark 解释各自独立：

| 路径 | 当前选择 | 后续扩展 |
|---|---|---|
| 论文单链复现 | 自研 Toric code、独立 `Z` 翻转、PyTorch 联合 RBM、单链 Gibbs、精确 Toric MWPM | 预注册 `L × p_error × seed` 网格和论文规模的非空数据；保持每个错误率单独训练 |
| 平台加速验证 | 同一 Toric 物理问题、显式 64 链 PyTorch Gibbs、同一内部 decoder 契约 | 有可用 GPU 时单独评估 CUDA 吞吐、端到端延迟和恢复链分布 |
| 电路级平台 | 已有可选 Toric code-capacity PyMatching adapter；尚未接入 Stim 电路 | Stim/DEM、DEM 版 PyMatching、Sinter adapter 和多轮噪声按优化计划分别实现 |

现阶段没有论文规模的 `P_fail(p_error)` 曲线，也没有 Stim/DEM adapter 或正式规模的 PyMatching 对照 run；项目不将 smoke 指标当作论文性能结论。

## 与现有优化计划的关系

此论文的 toric code/code-capacity 路径不同于 Phase 1 计划的 rotated-surface-code memory circuit。两者可共享数据 schema、checkpoint 和 benchmark 基础设施，但不能把现有 toy baseline 或未来 Stim memory 结果当作这篇论文的物理实验。P4.0 只验收论文 smoke 路径与代码归属；正式论文网格、结果比较及 Phase 1–3 的验收仍需独立任务和正式 run，不能由 smoke 的成功状态推定。
