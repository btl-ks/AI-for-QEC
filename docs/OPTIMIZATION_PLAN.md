# AI-QEC 项目优化计划

> 本文是任务路线图，不是当前目录树或能力说明。当前代码分层与文件放置规则见 [PROJECT_DESIGN.md](PROJECT_DESIGN.md)，外部工具边界见 [TECHNOLOGY_STACK.md](TECHNOLOGY_STACK.md)。任务完成状态必须以代码、测试和对应 Exit Criteria 为准。

> 版本：v3.5（2026-09-19；建立 OpenSpec 需求基线并补齐研究平台任务映射）
> 读者：项目维护者、协作者、coding agent
> 依据：2026-09-14 架构与代码复核，以及 2026-09-15 技术栈核验（摘要见 §1 与 [TECHNOLOGY_STACK.md](TECHNOLOGY_STACK.md)）
> 规则：每个 Phase 有退出标准（Exit Criteria）；未满足时，不开始依赖它的后续工作。

---

## 0. 执行约定

- **分支**：每项任务一个分支 `phase<N>/<short-name>`（例：`phase0/config-schema`），完成后合并到 `main`。
- **提交**：commit message 首行引用任务 ID，例如 `P0.4: forbid unknown config keys`。
- **进度**：任务完成后在本文勾选 `[x]`，同一提交内更新。
- **里程碑 tag**：P0–P4 退出时打 tag（见 §2）；P5 的独立扩展任务各自决定发布版本，不以未启动的 P5 任务阻塞已完成项。基线为 `v0.1.0-scaffold`。
- **学术约束**：继续遵守 `AGENTS.md`，定论性结论必须引用论文；项目内探针结果必须标注为 project-local。
- **能力状态**：目录、接口或配置字段存在，不等于能力已经实现；未实现能力必须 fail loudly，不能返回成功状态或静默退化。
- **架构形态**：当前阶段保持“模块化单体 + 不可变实验产物”，不引入微服务；先做真实契约与纵切面，再考虑分布式执行。
- **成功语义**：零步骤、部分执行、复用 stale 数据、缺失声明产物都不得标记为完整 `success`。
- **Phase 完成定义**：所有非显式 deferred 的任务均已勾选，Exit Criteria 的自动测试全部通过，正式验收 run 来自 clean worktree，且 artifact/manifest schema 可独立验证；待决策项必须有 owner、默认方案和最晚决策点。

---

## 1. 历史基线（2026-09-14，P0 修复前）

本节保留修复前的独立复核记录，用于解释 P0 任务的来源；当前已实现能力以 [PROJECT_DESIGN.md](PROJECT_DESIGN.md) 为准。下表中的失败行为不得作为当前仓库的能力状态引用。

| 能力 | 状态 | 说明 |
|---|---|---|
| CLI 管线可执行 | ⚠️ | generate/train/evaluate/benchmark 子步骤可运行；现有 smoke 手工串联子脚本，尚未覆盖正式 runner 的完整 flow/manifest 生命周期 |
| 论文导出 | ⚠️ | allowlist + sha256；现有包可在仓库外复现 target 回归指标，但 scope 与脚本不完全一致，关键复现文件可被 optional 跳过 |
| 导出安全 | ❌ | `--output` 可指向任意既有目录并被递归删除；缺少安全根、不可变 release、staging 与原子 `latest` 指针 |
| 物理层 | ❌ | synthetic 手写伯努利模型；`generator: stim` 被静默路由到 synthetic，manifest 记录失实 |
| 码几何 | ❌ | `SurfaceCode` 数据比特按 rotated（d²），探测器按 unrotated（2d(d−1)）；探测器邻接为一维直线 |
| 模型 | ⚠️ | `transformer/` 路径下实为 numpy ridge baseline |
| 配置 | ❌ | `model.*`、`objective.*`、`training.epochs/batch_size/scheduler`、`adaptation`、`benchmarks` 等字段不被任何代码读取 |
| 数据 | ⚠️ | 只保存 8 维摘要特征，丢弃原始 detector events；`logical_label` 由特征与 target 构造，存在循环 |
| Shortcut 控制 | ❌ | 密度匹配未生效；基准未检测密度捷径；配置中两项检验未实现 |
| 可追溯性 | ⚠️ | P0.2 已记录 git/environment/config/dataset reference；数据内容未强绑定、目录可覆盖，多 seed 只用第一个 |
| Runner 生命周期 | ⚠️ | 顺序 flow 可运行；run ID 可碰撞，零步骤/部分步骤可报成功，异常可能留下无终态 manifest 的半成品 |
| 测试与 CI | ⚠️ | 6 个本地 unittest 通过，但 smoke 绕过 runner；无物理/配置/导出安全测试，无 CI、lint、coverage 门禁 |
| 扩展平台 | ❌ | adaptation、runtime、deployment、design 大多是 placeholder/no-op，部分未实现入口仍返回成功 |

**Project-local 探针结果**（smoke 物理参数，4 万训练 / 2 万测试样本，非文献结论）：
target 从 0 升至 0.005 时平均探测密度上升约 54%；仅用密度预测 target 的 R² = 0.156，全部 8 个特征为 0.564。

**复核补充**（仓库现有 1200/300/300 smoke 数据，非文献结论）：density-only train→test R² ≈ 0.194；正式配置请求的 `stim` 与 `transformer_decoder` 实际分别解析为 `SyntheticQECBackend` 与 `LinearDetectorSummaryDecoder`。现有测试通过只能证明 plumbing 可执行，不能证明物理或科研有效性。

---

## 2. Phase 总览

| Phase | 目标 | 预估 | 依赖 | 退出 tag |
|---|---|---|---|---|
| **P0** 安全、诚信与可追溯基线 | 现有产物安全、诚实、可追溯；不新增科研能力 | 2–5 天 | — | `v0.1.1-integrity` |
| **P1** 真实 QEC 物理层 | Stim → Surface Code → Dataset → MWPM → LER | 1–2 周 | P0 | `v0.2.0-stim` |
| **P2** D2.2 科研级数据与基准 | 可辨识的弱 crosstalk 问题 + 能发现捷径的基准 | 2–3 周 | P1 | `v0.3.0-d22-benchmark` |
| **P3** 神经解码器与训练栈 | 配置驱动的 PyTorch 模型与训练器 | 2–4 周 | P1 + P2；P3.1–P3.5 可在 P1 后与 P2 并行，P3.6–P3.7 与 Phase exit 依赖 P2 | `v0.4.0-neural` |
| **P4** 工程化与论文复现 | 安装、测试分层、CI、导出重构 | 滚动推进；集中约 1–2 周 | P0；各任务依赖见 Phase 4 | `v0.5.0-eng` |
| **P5** 平台扩展 | 迁移适应、延迟、GPU sampler 与 circuit adapter | 按需 | 各任务依赖见 Phase 5 | — |

**顺序理由**：先建"尺子"（P2 基准），再调模型（P3）。否则会在有泄漏的基准上调参，得到看似成立、实则不成立的结果。
P4 中的安装、lint、测试分层（P4.1–P4.4）可从 P1 起并行推进。

### 近期工作总览（2026-09-19）

这是现有任务的执行摘要，不另建一套路线图。当前已完成 P4.12：Notebook 与脚本共用 Toric split、PyTorch `DataLoader` 批次和 RBM 训练器；其余能力按下面的任务 ID 验收。

| 顺序 | 要做的事 | 对应任务及完成证据 |
|---|---|---|
| 1 | 收紧数据与 run 生命周期，再做长时间论文训练 | P0.14 按生成规格复用数据；P0.15/18 收敛中断并明确 `resumed_from`；P0.16/17 恢复训练与解码；P0.19 防止后续阶段改写已登记产物。用中断/续跑与 hash 回归测试验收。 |
| 2 | 固定 Python 依赖和安装入口，可与第 1 项并行 | **P4.10 已完成**：`pyproject.toml` 分组、`uv.lock`、WSL 锁检查/全组合 dry-run/wheel 构建已通过；P4.1 仍待提供 wheel 可安装的 `ai-qec` CLI 并移除脚本的 `sys.path` 注入。P1.1/P3.1 仍需在真实 Stim/神经能力落地时验收。 |
| 3 | 接入真实 QEC 库，建立一条电路级纵切面 | P1.2–P1.9：Stim 采样和 DEM、schema v2/分片、PyMatching MWPM、Sinter 兼容与 LER；用相同 shots、observable truth 和协议验证。 |
| 4 | 建立可恢复的神经训练与公平对照 | P3.2–P3.5 将注册表、输入表示、模型、训练器放入共享包；P3.6/3.7 在 P2 冻结的 split/benchmark 上做校准与对照。论文 RBM 专属 CD-k/Gibbs 保持独立测试。 |
| 5 | 固定工程门禁与交付 | P4.2–P4.5、P4.9：lint/type/测试、wheel/CLI fresh install、CI 与文档；P0.10/P4.10 让论文包引用经验证的依赖锁和运行身份。 |

库与项目代码的分工、uv/Conda 边界见 [技术栈决策 §7–8](TECHNOLOGY_STACK.md#7-项目特有逻辑与成熟库的分工2026-09-19)。第 2 项可并行推进；P1、P3 的 Phase gate 仍按本计划执行。以上条目未因写入本表而变成已完成。

### OpenSpec 需求映射（2026-09-19）

[活动 OpenSpec change](../openspec/changes/establish-qec-research-platform-requirements/proposal.md) 将目标行为整理为 6 个 capability、37 条 Requirement、49 个 Scenario 和 50 项可验收任务。它描述目标，不改变当前能力状态；实现进度仍由本计划的任务 ID、Phase Gate、代码和测试共同判定。

| OpenSpec capability | 本计划 owner | 当前状态 |
|---|---|---|
| `research/experiment-lifecycle` | P0.14–P0.19、P0.13 | 未完成恢复与产物不可变收口 |
| `qec/dataset-pipeline` | P1.1–P1.9、P5.4 | 真实电路级路径未实现 |
| `qec/decoder-workbench` | P1.5/P1.9、P3.1–P3.7 | Toric/RBM smoke 可执行，电路级统一工作台未完成 |
| `research/data-efficient-training` | P3.8–P3.9 | 新增研究任务，未实现 |
| `deployment/performance-runtime` | P5.3–P5.5 | roadmap；无实时部署结论 |
| `research/automated-validation` | P5.6 | roadmap；先复用冻结协议，后扩展主张抽取 |

OpenSpec 的 `tasks.md` 是场景级实施分解；本计划保留可用于分支、提交和里程碑的父任务 ID，避免维护第二套互相竞争的进度状态。

### 发布语义门槛

| Gate | 必须满足 | 允许的最强表述 |
|---|---|---|
| **G0 / P0** | 无静默 fallback；toy、配置、数据、checkpoint 与导出身份真实且安全 | “project-local toy synthetic plumbing baseline” |
| **G1 / P1** | 真实 Stim circuit、observable、DEM、MWPM 与 LER 协议通过电路级仿真测试 | “Stim surface-code memory benchmark / circuit-level simulated LER” |
| **G2 / P2** | 可辨识性、负对照、OOD、配对、多 seed 与置信区间完整 | 限定噪声模型下的 D2.2 科研结论 |
| **G3 / P3** | 神经模型、训练恢复、校准及统一对照通过 | 神经解码器/噪声估计模型比较结论 |
| **G4 / P4** | wheel 安装、CI、文档与仓库外 paper-package 闭包测试通过 | 可安装、受 CI 门禁且可复现的研究软件版本 |
| **G5.1 / P5.1** | drift 协议、静态对照、恢复与遗忘指标完整 | 限定 drift 场景下的持续适应结论 |
| **G5.2 / P5.2** | source/target 身份、from-scratch 对照、多 seed 与置信区间完整 | 限定 code/noise 域的迁移结论 |
| **G5.3 / P5.3** | 端到端边界、硬件环境、尾延迟、正确性与 deadline miss 完整 | 限定部署环境下的实时性结论 |

---

## Phase 0 — 诚信与可追溯基线

**目标**：让现有 smoke 结果准确地呈现为“玩具合成基线”，消除破坏性文件操作与静默能力降级，并让每次运行、数据和导出包可精确追溯。

- [x] **P0.0 论文导出安全热修复**：`--output` 默认只允许位于 `paper/releases/`；拒绝项目根、其祖先、run/data 根、符号链接逃逸及任何既有目标。每次发布创建新的不可变 release ID，在同文件系统 sibling staging 完整构建并校验后原子 rename，再原子更新 `latest` 指针；失败时旧 release 与旧指针保持不变，绝不先 `rmtree`。增加路径边界、符号链接、项目根误传、重复 ID 和中断测试。
- [x] **P0.1 Git 管理**：`git init -b main`；原样基线提交并打 tag `v0.1.0-scaffold`；新增 `.gitattributes`（LF 规范化、二进制产物）；`.gitignore` 补充本地 agent / notebook 状态目录。
- [x] **P0.2 Run manifest 增强**：manifest 记录 `git`（commit、分支、`dirty`、未提交路径、未提交改动 patch 的 sha256；dirty 时另存 `git_diff.patch`，含未跟踪文件，可在记录的 commit 上 `git apply` 还原工作区，>10 MiB 的未跟踪文件跳过并记录）、`environment`（Python、平台、关键包版本；完整包列表存 `environment.json`）、`config_hash`，以及 `dataset`（manifest 路径、其 `config_hash` 是否与本次 run 一致）。新增 `reproducibility.require_clean_worktree`（缺省 true，smoke 配置为 false）：工作区不干净时拒绝运行（退出码 3），除非 `--allow-dirty`。
- [x] **P0.3 能力诚实性 / Fail loudly**：`generator: stim` 在真实 backend 实现前直接报错；`model.implementation` 未注册即报错；paired generator、adaptation、model export、FTQC runtime、GPU/FPGA、realtime、design search 等未实现入口必须非零退出或 `raise NotImplementedError`，不得 no-op 成功。synthetic 后端重命名为 `toy_synthetic`，manifest 写入 `requested_generator`、`resolved_backend` 与 `physics_fidelity: toy`。
- [x] **P0.4 类型化配置与 ResolvedExperimentSpec**：用 pydantic 或 dataclass 建立唯一配置入口，禁止未知字段；声明但尚未被消费的字段必须报错，或显式放入 `reserved:`。在任何写盘前解析默认值和 CLI override，并做 cross-field/capability 校验：distance、rounds、batch/sample count、概率范围、`min <= max`、flow ID 唯一、step 存在、model/backend/trainer 兼容。为每个 `flow[]` step 定义输出契约（path、artifact type、required/non-empty、schema/version 与 hash policy），并写入 resolved plan 供 runner 独占消费。
- [x] **P0.5 基准诚实化**：移除或显式标记未实现的 `matched_nuisance_test`、`nuisance_permutation_test`；将 `target_permutation_mae` 改名为随机配对基线，避免被误读为统计置换检验。配置声明的每一项 metric/test 必须实际执行，否则配置验证失败。
- [x] **P0.6 修正并拆分码几何语义**：rotated surface code 使用 d² 个数据比特、d²−1 个稳定子 [1,5]；分别建模 `num_data_qubits`、`num_stabilizers_per_round`、measurement/ancilla qubits 与跨 rounds 的 `circuit.num_detectors`，不得把稳定子数、辅助比特数和 detector-event 总数混为一个字段。真实 detector 数在 P1 中由编译后的 circuit 导出；为 d ∈ {3, 5, 7} × 多 rounds 增加一致性测试。
- [x] **P0.7 不可变数据与 run 强绑定**：数据目录只保留一个真相源（去掉 flow 参数中重复的 `--output`）；以 generation-spec/content hash 命名且默认拒绝覆盖。split 写入 staging，全部校验成功后再原子提交 manifest；记录每个文件的 sha256、shape、dtype、样本数。train/evaluate 前强制验证 schema、config、内容 hash 和 dataset ID；stale/mismatch 必须失败，不能只记录布尔值。
- [x] **P0.8 多 seed**：`reproducibility.seeds` 真正生效（每个 seed 独立子 run），或在实现前只允许单 seed 并报错。
- [x] **P0.9 Checkpoint 身份与命名**：numpy checkpoint 扩展名改为 `.npz`，把 `.pt` 留给 P3 的 PyTorch；记录 best 的选择规则。checkpoint 写入并在加载时校验 model implementation、feature schema/order、dataset/config hash 与代码版本；checkpoint sha256 写入 run artifact manifest 或 sidecar，不写回 checkpoint 本体形成自引用。当前 best/last 相同时明确标为单次闭式 baseline，不能伪装 validation selection。
- [x] **P0.10 论文包对齐与闭包验证**：导出包 `scope` 与 `reproduce.py` 实际复现范围一致；记录当前环境的精确依赖快照，并在 P4.10 项目级 lock 存在后强制引用它。复现命令需要的 checkpoint、数据、代码和 metrics 不得标为 optional；导出前校验 run/config/data/checkpoint hash 相互匹配。`reproduce.py` 重算 scope 声明的全部指标并与保存结果按容差核对；内容清单覆盖 payload 与 metadata，清单自身的 hash 放在 release 外部 sidecar/索引中，避免自引用，并提供验证入口。
- [x] **P0.11 有效生成身份与数据契约**：manifest 记录解析后的 seed、batch size、sample cap、所有 CLI override、生成器/模拟器版本及有效噪声参数。backend batch 必须精确匹配 required keys、shape、dtype、finite/binary/range 约束，禁止漏字段后写出未初始化数组。同 config+seed 的内容应与 batch 分块无关；若暂时做不到，batch 参数必须进入数据身份并明确测试。
- [x] **P0.12 Runner 生命周期与状态机**：runner 只执行 P0.4 产出的 resolved plan；run ID 使用高精度时间/UUID + config hash，目标已存在时默认拒绝。写盘前验证完整 flow、`--only/--skip` ID 与依赖；零步骤为错误，部分执行标为 `partial`。run 创建时即写 manifest，并在每次状态变更时原子更新；捕获异常/中断为 `failed`/`interrupted`；按 resolved step output contract 校验 path/type/non-empty/schema/hash，记录 skipped/disabled 原因。
- [x] **P0.13 诚信回归测试**：新增真正调用 `run_experiment.py` 的端到端 smoke；覆盖 unsafe export、unknown/unconsumed config、未实现 capability、zero-step、run collision、stale dataset、multi-seed、缺失 backend 字段、损坏 NPZ、异常终态和 paper package 闭包。测试不得只检查文件存在，还要检查身份、schema 与关键语义。

> 2026-09-17 追加（P0.14–P0.19）：Torlai–Melko 论文规模复现暴露的中断恢复与身份问题。P0.14 补齐 P0.7 的“按生成规格命名”，P0.15 补齐 P0.12 在 Notebook/进程被杀路径上的中断收敛；在它们完成前 Phase 0 不视为退出。

- [ ] **P0.14 数据集身份只取生成规格**：当前 `generation_hash` 包含整份 run 配置的 `config_hash`，实验名称、训练或解码参数（如 `training.decoder.max_steps`）一变就生成新的数据集目录，导致只改解码设置的 run 找不到已有数据。身份改为只覆盖生成规格（code、noise、data 的样本数与生成器版本、seed、batch 策略及有效噪声参数）；训练、解码与实验元数据不得进入数据身份。给出既有数据集目录的兼容或迁移方案，旧 run 的引用必须仍可校验；测试“仅改训练/解码/实验名”时复用同一数据集，改任一生成字段时身份不同。
- [ ] **P0.15 run 存活检测与中断收敛**：run 创建时在 manifest 记录进程号、主机名与启动标识；Notebook（`RunRecord`）与 runner 在启动新 run 或显式清理时，把进程已不存在的 `running` run 标为 `interrupted`，记录中断时所在阶段，不得长期停留在 `running`。`Ctrl+C` 已记为 `failed`，本任务覆盖 SIGKILL、断电、关机与休眠后进程丢失等路径；用被强制结束的子进程做回归测试。
- [ ] **P0.16 训练断点续跑**：每个 epoch 结束时原子写入续跑状态（模型、优化器、数据洗牌与 CD 采样两个随机数生成器的状态、训练历史、当前 best 及其 epoch），并在 manifest 记录其 hash。续跑前校验配置、数据集身份与代码版本一致，从下一个 epoch 继续。测试：任意 epoch 处中断再续跑，得到的参数、训练历史与 `best.pt` 必须与不中断时逐位一致。
- [ ] **P0.17 解码断点续跑**：解码阶段每 N 条样本原子保存部分预测（恢复链、有效标记、步数、延迟与失败标记）；续跑从第一个未完成的样本继续。每条样本使用独立随机流，续跑结果必须与不中断时逐位一致（延迟字段除外），并有测试。
- [ ] **P0.18 续跑 run 的语义**（待决策 DEC-7）：明确续跑是在原 run 内继续，还是新建 run 并以 `resumed_from` 指向被中断的 run；两种方式都不得重新生成数据集（依赖 P0.14），且 manifest 必须保留完整的中断与续跑时间线。若选择在原 run 内继续，只允许状态为 `interrupted`、配置 hash 与代码版本均一致的 run，并同步修订“每次执行分配唯一 run ID”的约定。
- [ ] **P0.19 阶段产物不可变**：当前 benchmark 阶段把指标合并写回 evaluate 阶段已记录 hash 的 `metrics.json`，使 evaluate 记录的 hash 必然失效。任何阶段不得修改前序阶段已登记的产物；需要汇总时写入新文件或由 `finish` 生成汇总。run 结束时校验所有阶段登记的产物 hash 与磁盘一致，不一致即判定失败，并有测试。

**内部依赖**：P0.0 为首要且可独立交付的安全热修复；P0.3 + P0.4 → P0.11 → P0.7 → P0.9 → P0.10；P0.4 → P0.5/P0.12；P0.7 + P0.12 → P0.8；P0.6 在 P1 前完成；P0.14 → P0.16/P0.17；P0.15 + DEC-7 → P0.18 → P0.16/P0.17；P0.19 可独立推进；P0.13 在其覆盖的任务（含 P0.14–P0.19）完成后统一收口。

**Exit Criteria**
- unsafe `--output` 不可能删除项目、run/data 根或未知目录；导出失败不损坏既有 release。
- 真正通过 `run_experiment.py` 的 smoke 测试通过；P0.3、P0.4、P0.7、P0.11–P0.13 的成功与失败路径有自动测试。
- 使用 `generator: stim`、`transformer_decoder` 或其他未实现 capability 的配置会在写盘前报错并给出明确说明。
- 配置中的每个非 `reserved` 字段都被消费并可由行为测试证明；未知字段、无效范围和不存在的 flow step 均失败。
- `run_manifest.json` 中 `git.commit` 非空，并包含 `git.dirty`、包版本、解析后的有效配置、数据/模型内容哈希和明确终态。
- 同 config+seed 在不同生成 batch size 下内容一致；若作为临时例外，则身份 hash 必须不同且 manifest 明示原因。
- 多 seed 产生独立 child runs 与父级汇总，或在未实现时拒绝多 seed 配置。
- README、配置注释、manifest、metric 名和报告中的 smoke 结果统一标注为 `project-local toy synthetic baseline`。
- 只改训练、解码或实验元数据的 run 复用同一数据集；进程被强制结束后，run 会被收敛为 `interrupted` 而非停留在 `running`。
- 训练与解码在任意中断点续跑后，结果与不中断时逐位一致（延迟字段除外），且续跑不重新生成数据；续跑的 run 关系按 DEC-7 记录在 manifest 中。
- run 结束时，所有阶段登记的产物 hash 与磁盘文件一致。

---

## Phase 1 — 真实 QEC 物理层

**目标**：交付真实 Stim surface-code memory 的电路、数据、MWPM 与 LER 纵切面；当前架构状态见 `docs/PROJECT_DESIGN.md`。

**内部依赖**：P1.1 → P1.2；P1.2 → P1.3/P1.5；P1.3 → P1.4；P1.2 + P1.5 → P1.6；P1.3 + P1.5 → P1.9；P1.8 的 label provenance 约束前置于相应 P1.7 测试。

- [ ] **P1.1 依赖分组**：`pyproject.toml` 增加 `[sim]` optional extra = stim、pymatching、sinter；`[torch]` 留给 P3，开发工具放 `dependency-groups.dev` 并由 P4.10 锁定。依赖存在不等于 capability 已实现，registry 与 smoke 必须同时通过。
- [ ] **P1.2 StimQECBackend**：以 `stim.Circuit.generated("surface_code:rotated_memory_z", distance=..., rounds=..., after_clifford_depolarization=..., before_round_data_depolarization=..., before_measure_flip_probability=..., after_reset_flip_probability=...)` 生成电路 [2]；用 `compile_detector_sampler().sample(..., separate_observables=True, bit_packed=True)` 采样。code geometry、detector count/coordinates、observable count 和 DEM 必须来自编译后的 circuit，不能再用手写近似计数作为数据真相源。
- [ ] **P1.3 数据 schema v2**：bit-packed detector events、observable flips、每样本噪声参数、电路与 DEM 的 sha256、`schema_version`、shape/dtype 及 shard 索引；为 soft readout、leakage、coordinates/features、calibration/device/time/domain ID 保留可选字段。摘要特征改为可重建的 preprocessing 派生视图，不能替代 raw events。由 schema 模块统一定义 writer/loader/evaluator 的键和校验；`QECSample` 要么成为唯一契约，要么删除，避免双 schema。
- [ ] **P1.4 分片写盘**：大样本量（如 1M）按 shard 写入，生成与加载均不要求全量驻留内存。
- [ ] **P1.5 MWPM 基线**：`pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))` [3]。
- [ ] **P1.6 LER benchmark**：按 distance × 物理错误率扫描；LER 必须由 decoder 预测的 observable 与 sampled observable flip 的 residual 是否非零来定义，输出失败数、shots、LER 及二项置信区间。正式 run 前冻结 `protocol_id`，其中给出精确 p-grid、rounds、每 cell 的最小/最大 shots 与最小 failure budget、Wilson 或 Clopper–Pearson 方法、停止规则及 distance 趋势判据；toy label 分类错误率不得沿用 `logical_error_rate` 名称。
- [ ] **P1.7 物理与 schema 测试**：噪声为 0 时 detection events 全 0、observable 不翻转；数据中 detector/observable 数与 circuit 一致；固定 seed 结果可复现；bit-pack 往返无损；损坏 shard、错误 DEM/circuit hash、错 feature order 会被拒绝。对 d ∈ {3, 5, 7} 检查编译后的几何，不再断言手写公式。增加人工可验算的小型 circuit/DEM fixture，验证 MWPM correction 与 residual-observable LER；在 fresh environment 安装 `.[sim]` 后执行真实 backend smoke。
- [ ] **P1.8 Observable label provenance**：真实路径的 `logical_label` 来自 sampled observable flip，并记录 decoder correction/prediction 的关系；验证 P0.3 建立的 `toy_synthetic` 测试夹具与真实 dataset/model registry 隔离，不能被正式配置、物理 benchmark 或论文导出误引用。
- [ ] **P1.9 Sampler/decoder protocol 与 Sinter 兼容**：建立不泄漏 Stim/Qiskit/CUDA-Q 类型的 project-local `QECProblem`、`SamplerBackend`、`CompiledDecoder` 和 `DecoderOutput` 契约。DEM + hard detector bits 路径提供 Sinter adapter；code-capacity recovery、soft readout、leakage 和 hybrid diagnostics 由项目接口保留。用同一批 bit-packed shots 验证 project runner 与 Sinter adapter 的 observable prediction/LER 一致。
  - **解码器目录决策**：owner 为 P1.9 实施者，最晚在统一 decoder protocol 冻结前确定。默认保留现有 `ai_qec/models/decoders/`，因为它符合当前架构；对 RBM Gibbs、经典 MWPM、PyMatching adapter 和计划中的混合 decoder 一起评估依赖方向。若决定迁到顶层 `ai_qec/decoders/`，应整体迁移协议和实现，同步更新 registry、`notebook_api`、调用方、测试与架构文档，并验证公开导入的兼容过渡；不单独迁移 `rbm_decoder.py`。

**Exit Criteria**
- 一条命令生成 d ∈ {3, 5, 7}、多个物理错误率下的 MWPM LER 曲线。
- 配置、resolved plan、dataset manifest 与 run manifest 对 backend 的记录完全一致，不存在 `requested=stim/resolved=synthetic`。
- raw detector events 与 observable flips 可从 shard 无损读取；摘要特征可由 raw 数据重建并校验。
- 预注册 `protocol_id` 明确 p-grid、rounds、采样/failure budget、CI 与 distance 趋势判据；至少一个预先指定的 sub-threshold 区间通过该判据，与 [4] 描述的表面码行为一致。不通过时不得发布 G1，先排查实现或协议假设。
- fresh environment 中安装 `.[sim]` 后真实 Stim→MWPM smoke 与人工可验算 fixture 均通过，且 residual-observable LER 定义有回归测试。
- 物理单元测试全部通过；本机（12 核 / 15 GB）生成 1M 样本不内存溢出。

---

## Phase 2 — D2.2 科研级数据与基准

**目标**：在电路级噪声上定义可辨识的弱 crosstalk 学习问题，并建立能发现捷径的基准。本阶段与模型无关，用 ridge-summary 与 MWPM 特征基线跑通。

**内部依赖**：P2.0 与 DEC-6 必须先完成；P2.6 的统计协议必须在正式执行 P2.3/P2.4/P2.7/P2.8 前冻结；P2.1/P2.2 通过生成校验后才进入正式 benchmark。

- [ ] **P2.0 明确 estimand 与采样单位**：在实验规范中决定 target 是单 shot、连续 QEC window、calibration session 还是 device/domain 级参数；给 session/domain/pair 写稳定 ID。若噪声率在窗口内应固定，则不得逐 shot 独立重抽。split 必须按可泄漏的最高层级 group 隔离，并说明该选择是研究设计假设。
- [ ] **P2.1 Crosstalk 注入**：在两比特门层之后，对物理相邻的 spectator 比特对施加相关双比特 Pauli 通道（Stim `CORRELATED_ERROR` / `PAULI_CHANNEL_2`），强度即 target。
  *此建模方式是设计假设*；实验说明中须引用具体硬件 crosstalk 文献，并限定结论适用范围。
- [ ] **P2.2 解析密度匹配**：由 DEM 计算各 detector 的期望翻转率，按 target 调整背景噪声率，使期望平均检测率与 target 无关。生成时校验匹配误差，超出容差即失败，不做静默近似。
- [ ] **P2.3 Shortcut benchmark v2**（正式默认 K ≥ 999、α = 0.05；如修改须在 run 前预先登记）
  - 密度探针：仅用 detection density 预测 target 的 R²；
  - 条件评估：按密度分箱报告 target MAE；
  - 置换检验：K 次置换构造零分布，预先定义统计量，报告 p 值、效应量与 Monte Carlo 不确定性；
  - matched counterfactual pairs：相同 nuisance、不同 target 的配对样本（真正实现 `paired_generator`）。
- [ ] **P2.4 Benchmark controls**：构造故意泄漏密度的数据集作为报警能力的正对照；构造 target-null 数据作为假阳性率负对照。两者使用 P2.6 冻结的统计协议和机器可读判定。
- [ ] **P2.5 Group/OOD split**：训练与测试的 session/domain/pair ID 不交叉；OOD nuisance 区间不重叠。增加 split audit，报告各 target/nuisance 桶和 group 数，防止重复电路、共享随机流或 counterfactual pair 跨 split 泄漏。
- [ ] **P2.6 统计规范**：≥3 个独立 seed，每 seed 独立生成/训练/评估；报告逐 seed 值、均值与 95% 置信区间，`metrics.json` 附 shots、失败数、group 数和有效样本量。正式运行前固定阈值、主要指标、检验统计量、样本量/power 目标和排除规则；变更必须产生新 protocol ID。
- [ ] **P2.7 非学习参照**：基于 DEM 统计量的 target 估计器（如矩估计或似然估计），作为可辨识性参照。
- [ ] **P2.8 分层与校准报告**：除 aggregate MAE/RMSE/R² 外，报告常数基线、density-only 基线、各 target 强度误差/排序能力、预测越界率和不确定性校准；logical probability 若保留，报告 log loss、Brier score、可靠性图和类别条件结果。

**Exit Criteria**
- 匹配数据上的 density-only R² 的 95% 置信区间上界低于 P2.6 `protocol_id` 冻结的阈值，而不是只看单次点估计；协议须明确 CI 方法与重复单位。
- 同时满足同一协议冻结的可辨识性门槛：指定的 matched-pair 或 DEM 非学习估计器相对 null/常数基线达到预设最小效应量，并通过预设 α/power 判据；不得以抹掉全部 target 信号的方式通过 density gate。
- 故意泄漏密度的正对照按预设阈值报警；target-null 负对照不超过协议规定的假阳性率。
- 配置中声明的每一项检验都有实现、正/负测试和机器可读结果；未实现项无法启动正式 run。
- session/group/pair split audit 无交叉，至少 3 个 seed 的逐次结果和 95% 置信区间可由父 run manifest 重建。
- 论文结论明确限定到所选电路、crosstalk 通道、estimand、target/nuisance 范围与采样单位，不外推为通用硬件结论。

---

## Phase 3 — 神经解码器与训练栈

**目标**：配置真正驱动模型与训练；神经模型与经典基线在同一基准下对比。

**内部依赖**：P3.1 + P3.2 + P3.3 → P3.4/P3.5 → P3.6/P3.7；P3.7 还依赖已退出的 P2 基准与冻结 seed/split/protocol。P3.8 依赖 P1、P2.5/P2.6 与 P3.5，P3.8 → P3.9。

- [ ] **P3.1 依赖**：`[torch]` extra；需要图模型时增加独立 `[graph]` extra；明确 CPU/CUDA wheel 来源、设备能力检查与显式 CPU 执行，纳入 P4.10 的锁定支持矩阵。
- [ ] **P3.2 注册表与能力解析**：装饰器注册（`@register_model("transformer_decoder")`），由 `ResolvedExperimentSpec` 构建；trainer/evaluator/exporter 都必须通过同一 registry，禁止直接实例化具体模型。`family`、`implementation`、input representation、outputs 与 checkpoint 不匹配即报错。
- [ ] **P3.3 输入表示**：detector events 时空张量和 detector graph + 探测器坐标嵌入（Stim `get_detector_coordinates()`）；side-information schema 能携带 soft readout、leakage 与 calibration/domain context，缺失时使用显式 mask。
- [ ] **P3.4 第一条神经纵切面**：优先实现 AI edge-weight estimator 或局部 predecoder + PyMatching，输出 edge weights/residual syndrome 与 diagnostics；从 d = 3 起步。随后在同一协议下增加 detector-graph GNN 与 recurrent Transformer 直接 decoder 作为比较模型，研究依据见 [5,6]，predecoder 工程路线参考厂商公开结果 [7]。
- [ ] **P3.5 训练器与 checkpoint 状态**：真实读取 `epochs`、`batch_size`、`optimizer`、`scheduler`、`early_stopping`、`checkpoint.monitor`；混合精度；梯度累积；断点恢复；best 按 monitor 选择。checkpoint 保存模型/优化器/scheduler/scaler/RNG 状态及 resolved config、dataset/schema hash；恢复或评估前强校验兼容性。恢复一致性测试须固定 seed、断点 step、总 step 和逐项数值容差。
- [ ] **P3.6 多任务、对抗头与校准**：target 回归 + logical decoding；可选梯度反转 nuisance 头。使用与声明一致的 proper loss，分别验证每个 head 的梯度与权重；概率输出增加校准评估，target 输出范围策略必须显式而非只裁下界。增加从冻结模型表示预测 nuisance 的 latent probe，使用 P2 split/protocol 报告 R²；该模型相关诊断不得反向阻塞 P2 退出。
- [ ] **P3.7 统一对照表**：常数、density-only、非学习估计器、ridge-summary、MWPM、AI+MWPM hybrid、直接神经模型在同一 dataset identity、split、benchmark 与 seed 集上报告；禁止不同数据版本之间直接排名。
- [ ] **P3.8 数据高效训练协议**：冻结码距、噪声、轮数、候选生成量、实际训练量、计算预算、主要指标、failure budget、停止规则和多 seed 方案；均匀采样、固定课程与其他简单重加权方法必须作为同预算基线。
- [ ] **P3.9 困难样本与课程调度**：实现可审计候选池、困难样本 buffer、损失/置信度/RBM 能量评分器、原分布混合与可选 importance weighting；在固定独立测试集和未见噪声域报告 LER、样本效率、生成成本、GPU 时间、消融与置信区间。

**硬件预算**：RTX 4060 Laptop 8 GB。d = 5、20 轮时每样本约数百个 detector token；hidden 256、6 层的模型需依赖混合精度并控制 batch。d ≥ 7 视显存评估梯度检查点或缩小模型。

**Exit Criteria**
- d = 3、5 memory 实验上神经模型与 MWPM 的 LER 同表报告（不预设优劣）。
- D2.2 target 回归在 P2 基准下的检验结果完整报告（通过或失败均如实记录）。
- 训练可从 checkpoint 恢复，且按 P3.5 预先固定的 step/seed/数值容差与不中断训练一致。
- 改变任何已声明的模型/训练配置会改变 resolved plan 或被明确拒绝；不存在无效但被接受的参数。
- evaluator 对错误 dataset、feature order、model implementation 或 config hash 的 checkpoint 必须 fail loudly。
- P3.8/P3.9 的所有方法使用相同模型族、预算和独立测试集；未达到 failure budget 时标记 evidence-insufficient，不发布不稳定成功结论。

---

## Phase 4 — 工程化与论文复现

**执行方式**：滚动工程轨。P4.1、P4.2、P4.4 可在 P1/P2 期间并行；P4.3 依赖稳定的 P1/P2 benchmark；P4.6 依赖 P0.0/P0.10；P4.9 的 dataset/model 文档分别依赖 P1/P3。Phase exit 与 `v0.5.0-eng` 仍需 P3 完成。

- [x] **P4.0 工作区归位与论文 smoke 纵切面**：将可复用实现归位到 `ai_qec/`，数据写入 `datasets/`，论文资料与调用型 Notebook 放入 `paper/`；接入 Torlai–Melko toric code-capacity RBM 单链和独立并行链 smoke，以及小规模精确 MWPM 对照；记录当前架构、技术栈、论文路径和一致性核对。本项只验收这些 smoke 路径，不表示 P1–P3 或 Phase 4 已退出。
- [x] **P4.11 文档去重**：将工作区 Git/写入/产物放置规则归入 `PROJECT_DESIGN.md`，将文档索引归入根 `README.md`；修正所有引用，并标识优化计划的历史基线及论文 smoke 与 P4.0 的关系。保留研究主题、技术栈、任务计划和论文复现设计的独立职责。
- [x] **P4.12 RBM 数据与训练路径收拢**：Notebook 与脚本共用 Toric split 生成器和 RBM 训练器；训练通过 PyTorch `DataLoader` 组织小批次，一次构造 float32 输入、减少逐批设备同步，并按样本数汇总不等长 batch 指标；Notebook 调用共享 Gibbs 解码器。验证独立 split 随机流、训练确定性、非整除 batch，以及 CPU/CUDA Notebook smoke。本项不改变 run 生命周期与产物格式。
- [x] **P4.13 架构与包管理执行摘要**：在技术栈文档明确领域代码与 Stim/PyMatching/Sinter/PyTorch 的分工、`pyproject.toml` + uv 的目标方案及 Conda/CUDA 边界；在本计划索引依赖关系和验收任务。此项仅为文档决策，不表示 uv 已安装、依赖已锁定或后续能力已实现。
- [x] **P4.14 论文总结目录迁移**：将 Torlai–Melko 报告 Markdown 归入同名子目录，修复报告内资源/源码相对链接及项目文档入口；嵌套目录的生成版 PDF/HTML 保持 Git 忽略。以仓库内相对链接校验与 `git check-ignore` 验收。
- [x] **P4.15 解码器目录归属复核**：对照当前架构与未来 decoder protocol，确认现阶段保留 `models/decoders/`，不单独移动 Gibbs 解码器；将整体目录取舍、owner、默认方案和最晚决策点写入 P1.9。本项仅完成架构计划，不表示目录已迁移或 P1.9 已交付。
- [x] **P4.16 OpenSpec 需求基线与文档职责同步**：初始化 repo-local OpenSpec，建立实验生命周期、数据管线、解码工作台、数据高效训练、性能运行时和自动验证 6 个 capability 的 proposal/spec/design/tasks；严格校验通过。README、AGENTS 与 docs 只链接目标需求，不把未实现 capability 写成当前能力。本项只完成规划与文档整理，不表示 50 项实施任务已完成。
- [ ] **P4.1** 提供安装后的统一 `ai-qec` CLI（run/generate/train/evaluate/benchmark/export 子命令）；开发环境通过 P4.10 的 `uv sync --locked` 安装所选 extras/groups，另在仓库外安装 wheel 验证 CLI。移除 scripts 中的 `sys.path` 注入；package data 不得引用 wheel 外的默认配置，各 extra 的支持矩阵必须测试。
- [ ] **P4.2** ruff + pytest + typing + coverage；测试按 `unit/`、`integration/`、`regression/`、`smoke/` 分层，并设置最低覆盖门槛。smoke 必须经过正式 runner，而非手工串联子脚本。
- [ ] **P4.3** 回归测试：固定 seed 小数据集的 golden metrics，按容差比较；golden fixture 使用明确 allowlist，不得因全局 `*.npz` ignore 而静默缺失。
- [ ] **P4.4** pre-commit：ruff、大文件拦截；默认禁止提交 `*.npz` / `*.pt`，仅允许 P4.3 中经审计、尺寸受限且列入显式 allowlist 的 golden fixture 例外。
- [ ] **P4.5** 远程仓库与 CI（待决策 DEC-1）：至少覆盖 Python 3.10 与当前 3.13，执行 lint/type/unit/integration/smoke、构建 wheel、安装后 CLI 和论文包闭包测试。
- [ ] **P4.6** 论文导出重构：从逐文件复制改为导出最小公开子包或打包脚本，消除 `mae`、`r2`、standardizer 等 3–4 处重复实现。
- [ ] **P4.7** 大数据集与 checkpoint 版本管理（待决策 DEC-2）；无论采用本地 hash、DVC 或对象存储，run 中只引用不可变 content ID。P0.7 默认先使用本地 immutable + sha256，不等待此决策。
- [ ] **P4.8** 占位模块收缩：P0.3 已保证 fail-loud；本任务删除无调用方的空壳文件，或移到 roadmap/实验命名空间。保留的公共模块必须有 owner、状态、调用方和测试（待决策 DEC-4）。
- [ ] **P4.9** 更新 README，补齐 `docs/datasets/`、`docs/models/`；README 的能力表由 registry/capability manifest 生成或测试，避免再次与实现漂移。
- [x] **P4.10** 可重建依赖：使用 `pyproject.toml` + uv 管理 `[sim]`、`[torch]`、`[plot]`、`[notebook]` extras 和 `dependency-groups.dev`，提交并审查 `uv.lock`；移除重复的 `requirements.txt`。已验证 WSL fresh environment 的锁文件解析、全组合 dry-run、wheel 构建和当前 Conda `quantum` 环境中的关键包；CUDA wheel 仍需在目标机器上做实际安装和 smoke。P0.10 消费锁定结果；正式 run 与 paper package 记录并验证 lock/hash、解释器与关键包版本。

**Exit Criteria**
- 在 clean checkout / fresh environment 中用已提交的 `uv.lock` 分别同步 `[sim]` 与 `[sim,torch]`，通过 `ai-qec` CLI 完成真实 Stim/MWPM 和神经 smoke；开发工具从 `dependency-groups.dev` 安装。再从构建的 wheel 安装验证 CLI，源码与测试中不再依赖 `sys.path` 注入。
- lint、typing、unit、integration、regression、runner smoke、构建与 paper-package 测试均由 CI 强制执行并通过。
- paper package 在仓库外从锁定依赖安装，验证全部 hash，不含私有绝对路径，并重算 scope 声明的全部结果。
- 所有公开 capability 均可执行并有测试，或明确 fail loudly；不存在 placeholder/no-op success。
- README、dataset/model 文档、capability registry 和 CLI help 与当前实现一致。

---

## Phase 5 — 平台扩展

**Gate**：P5.1/P5.2 要求 P2 与 P3 的 Exit Criteria 均已满足；P5.3 还要求 P4.1、P4.2、P4.10 完成；P5.4 要求 P1 与 P4.10；P5.5 要求 P1.2/P1.3/P1.9；P5.6 要求 G1–G4 的身份、协议、runner 和导出闭环均已验证。完整 Phase 5 验收要求 Phase 4 已退出。

- [ ] **P5.1（研究方向 D2.3）**：时变噪声流数据生成 + continual adaptation benchmark；预先定义 drift、适应延迟、遗忘与静态 baseline 指标。
- [ ] **P5.2（研究方向 D2.4）**：迁移 d = 3 → 5 → 7、noise A → B；实现 source-only、from-scratch、fine-tune 与 adapter 的统一 transfer benchmark。
- [ ] **P5.3（研究方向 D4.1）**：分别测量 T_model、T_decoder 与 T_E2E；端到端边界、数据移动、预处理和后处理必须明确定义并纳入 T_E2E。报告 warm-up、batch=1、p50/p95/p99、throughput、内存、硬件/软件环境和 deadline miss rate；只有 component latency 时不得通过 G5.3。
- [ ] **P5.4 可选 GPU sampler**：实现 `custabilizer_gpu` adapter，不改变 schema 与 sampler protocol；启动前检查 cuStabilizer/CUDA/设备能力，禁止静默回退。对相同 circuit/DEM 做统计等价性测试，并按 distance × rounds × shots × noise model 报告与 Stim CPU 的吞吐 crossover、总耗时、显存和环境。能力依据与限制见 [TECHNOLOGY_STACK.md](TECHNOLOGY_STACK.md)。
- [ ] **P5.5 Circuit interoperability adapters**：增加 OpenQASM 3/Qiskit 输入 adapter，通用电路与 IBM hardware 只通过该边界接入。adapter 必须显式接收或生成经过验证的 detector/observable mapping；无法保留 QEC 语义时拒绝转换。QIR/CUDA-Q adapter 仅在部署需求明确后另建任务。
- [ ] **P5.6 AI 辅助论文与想法验证**：建立带来源状态的 claim/hypothesis record、冻结 protocol、时间与算力预算、证据图和证据等级。第一步只自动重跑已冻结协议；论文主张抽取与新实验生成必须在缺少 estimand、基线、预算或成功判据时停止并等待决策。
- D3（FTQC runtime）与 D5（QEC design）保持 roadmap 状态，暂不投入实现。

**Exit Criteria**
- P5.1：至少一个预先定义的 drift 场景可复现，adaptation 与无适应 baseline 同表报告，包含恢复时间、稳态误差和遗忘指标。
- P5.2：每个 transfer 结论均有 source/target 数据身份、独立 seed、from-scratch 对照和置信区间；不把输入尺寸变化误当作迁移收益。
- P5.3：延迟测量区分模型、decoder 与端到端边界，在固定硬件/软件环境重复运行，并同时报告正确性；仅测 Python helper 不得称为实时部署。
- P5.6：每项自动生成结论均可追溯到来源、冻结 protocol、代码、数据、checkpoint、预测和统计结果；预算耗尽、证据不足和实现失败不得显示为成功复现。
- 任一任务可独立形成里程碑；未完成的 P5 任务保持 roadmap，不阻塞其他已验收任务的发布。

---

## 3. 风险

| 风险 | 应对 |
|---|---|
| 导出路径误传导致递归删除项目或数据 | P0.0 使用安全根、不可变 release、sibling staging 与原子 `latest` 指针；破坏性边界测试先行 |
| 配置声明与解析能力不一致，产生“成功但语义失实”的 run | P0.3/P0.4 建 capability registry 与 ResolvedExperimentSpec；未知、未消费、未实现均 fail loudly |
| 数据/run/checkpoint 可变或交叉拼装 | P0.7/P0.9/P0.10 以 content ID 和相互 hash 绑定；stale/tampered artifact 直接拒绝 |
| runner 碰撞、中断或部分执行仍显示成功 | P0.12 使用 exclusive create、原子状态机、明确 partial/interrupted 状态和输出契约 |
| backend 漏字段后写出未初始化数组 | P0.11 在写盘前验证 exact keys、shape、dtype、finite/binary/range 和填充完整性 |
| 电路级 crosstalk 模型与真实器件偏差 | 作为设计假设明确标注；引用硬件文献；限定论文结论范围 |
| estimand/采样单位不清导致 session 或 counterfactual pair 泄漏 | P2.0 先定义 shot/window/session/domain 层级；P2.5 按最高层 group split 并审计 |
| 密度匹配无法对所有 target 精确实现 | 设容差，超出即失败；报告实际残余相关性 |
| 8 GB 显存限制大 distance 训练 | 从 d = 3、5 开始；梯度检查点、小模型、减小 rounds |
| 大数据集 IO（WSL2） | 数据保留在 Linux 侧文件系统（当前 `/home` 下）；分片存储 |
| paper code、checkpoint、data 来自不同 run/commit | P0.10 导出前验证闭包和相互 hash；在仓库外重算全部 scope 指标 |
| 平台广度继续先于深度扩张 | P5 前不新增顶层方向目录；P0.3 令占位能力 fail-loud，P4.8 收缩无调用方空壳 |

## 4. 待决策

> `DEC-*` 是项目决策编号，避免与研究方向 D1–D5 混淆。每项决策到期前按“默认方案”继续推进，不得成为无限期模糊 blocker。

| ID | Owner | 问题 | 默认方案 | 最晚决策点 / 影响 |
|---|---|---|---|---|
| DEC-1 | 项目负责人 | 远程托管平台与可见性 | 私有 GitHub；保护 `main` | P4.5 CI 启动前 |
| DEC-2 | 平台负责人 | 大数据 / checkpoint 管理：本地、DVC 或对象存储 | P0 使用本地 immutable + sha256；规模化再评估 DVC/对象存储 | P4.7；不得阻塞 P0.7 |
| DEC-3 | ML 负责人 | 深度学习框架是否需要 JAX | PyTorch；只有明确实验需求才增加 JAX | P3.1 前 |
| DEC-4 | 代码维护者 | 占位模块保留还是移出 `ai_qec/` | 活动路径先 fail loudly；无近期调用方者移到 roadmap | P4.8 前；P0.3 不等待此决策 |
| DEC-5 | 科研负责人 | D2.2 target 的采样/预测单位 | 优先评估固定参数的多-shot window/session；单-shot 作为对照假设 | P2.0 完成前 |
| DEC-6 | 科研负责人 | crosstalk 对应的目标硬件、门和耦合场景 | 选择一个明确平台与门级场景，不先宣称跨硬件通用 | P2.1 开始前；决定引用与适用范围 |
| DEC-7 | 平台负责人 | 被中断的 run 续跑时：在原 run 内继续，还是新建 run 并记录 `resumed_from` | 新建 run 并记录 `resumed_from`，保持“每次执行一个 run”的约定 | P0.16/P0.17 实现前 |

---

## 参考文献

[1] C. Horsman, A. G. Fowler, S. Devitt, R. Van Meter, "Surface code quantum computing by lattice surgery," *New J. Phys.* 14, 123011 (2012). https://doi.org/10.1088/1367-2630/14/12/123011
[2] C. Gidney, "Stim: a fast stabilizer circuit simulator," *Quantum* 5, 497 (2021). https://doi.org/10.22331/q-2021-07-06-497
[3] O. Higgott, C. Gidney, "Sparse Blossom: correcting a million errors per core second with minimum-weight matching," *Quantum* 9, 1600 (2025). https://doi.org/10.22331/q-2025-01-20-1600
[4] A. G. Fowler, M. Mariantoni, J. M. Martinis, A. N. Cleland, "Surface codes: Towards practical large-scale quantum computation," *Phys. Rev. A* 86, 032324 (2012). https://doi.org/10.1103/PhysRevA.86.032324
[5] J. Bausch et al., "Learning high-accuracy error decoding for quantum processors," *Nature* 635, 834–840 (2024). https://doi.org/10.1038/s41586-024-08148-8

[6] M. Lange et al., "Data-driven decoding of quantum error correcting codes using graph neural networks," *Physical Review Research* 7, 023181 (2025). https://doi.org/10.1103/PhysRevResearch.7.023181

[7] C. Chamberland, J. Olle, M. Li, S. Thornton, I. Baratta, "Fast AI-Based Pre-Decoders for Surface Codes," NVIDIA Research (2026), vendor research publication. https://research.nvidia.com/publication/2026-04_fast-ai-based-pre-decoders-surface-codes
