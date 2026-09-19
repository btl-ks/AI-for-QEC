# Tasks

> 本文件是 umbrella requirement baseline 的任务库存，用于保存总体范围、依赖和拆分路由，不是可直接 apply 的实现清单。开始任何实现前，必须先创建 proposal.md 指定的 child change，把相应条目分解为带 requirement/scenario、design 章节和测试证据引用的 tasks；完成状态只在 child change 中维护。

> 本 umbrella 保持全部条目未勾选。子 change 归档后，通过主规格、当前能力文档和优化计划反映交付状态，不回填本文件来伪装 umbrella 已实现。

## 1. 需求基线与追踪

- [ ] 1.1 在 README 和文档职责表中加入 OpenSpec 入口，明确主规格、当前架构、技术选型、研究地图和任务计划的边界，并用仓库内链接检查验证无断链
- [ ] 1.2 建立 capability/requirement/scenario 到现有 P0–P5 任务、拟新增任务和测试的追踪矩阵，验证六份 spec 的每个 scenario 都有 owner、gate 和验收方式
- [ ] 1.3 为新增的数据高效训练与自动验证工作分配优化计划任务 ID 和依赖，验证后续提交能够遵守“提交引用任务 ID”的仓库规则
- [ ] 1.4 增加机器可读 capability 状态清单，区分 planned、available 和 validated，并用测试验证 README 或 CLI 不会把 planned 能力显示为已实现

## 2. 实验生命周期与恢复

- [ ] 2.1 完成 P0.14 的 generation-spec dataset 身份，验证仅修改训练、解码或实验名时复用数据而修改生成字段时身份变化
- [ ] 2.2 完成 P0.15 的进程存活检测和中断收敛，使用被强制结束的子进程验证 running run 转为 interrupted
- [ ] 2.3 按 design 冻结“新 run + resumed_from”恢复语义并更新 manifest timeline，验证原 interrupted run 和产物保持不可变
- [ ] 2.4 完成 P0.16 训练断点恢复，验证固定 seed 在预登记断点恢复后满足与连续训练相同的历史和数值容差
- [ ] 2.5 完成 P0.17 解码断点恢复，验证每样本独立随机流和部分预测使恢复结果除延迟字段外保持一致
- [ ] 2.6 完成 P0.19 阶段产物不可变和结束时哈希复核，验证后续阶段改写已登记文件会使 run 失败
- [ ] 2.7 运行正式 runner smoke、损坏产物、身份不匹配和恢复失败集成测试，验证所有路径产生明确终态且零步骤或部分执行不为 success

## 3. 可信 QEC 数据管线

- [ ] 3.1 完成 P1.2 的 surface-code memory circuit 与 QECProblem 身份，验证距离、轮数、detector、observable 和 DEM 来自编译电路
- [ ] 3.2 完成 P1.3 schema v2 的 raw detector events、observable flips、side-information 扩展位和 provenance 校验，验证空、缺键、错 shape/dtype 和损坏哈希均被拒绝
- [ ] 3.3 完成 P1.4 分片 writer/loader，使用超过单 shard 上限的数据验证生成和加载不要求全量驻留内存且总样本数准确
- [ ] 3.4 实现 group/session/domain/pair split 与 audit，构造跨 split 泄漏 fixture 验证正式训练会被拒绝
- [ ] 3.5 完成 P1.9 sampler protocol 和显式 backend 能力检查，验证请求不可用 GPU backend 时不创建数据且不回退
- [ ] 3.6 建立 CPU 多 worker、有界预取、pinned-memory 与异步复制路径，记录 queue wait、生成时间和训练时间并验证开关不改变样本真值
- [ ] 3.7 在具备目标 NVIDIA 环境后实现 P5.4 GPU sampler adapter，使用预登记统计等价性和 crossover benchmark 验证其适用范围；无目标设备时保持 planned

## 4. 统一解码与训练工作台

- [ ] 4.1 完成项目 decoder protocol 和 Sinter adapter，使用同一批 bit-packed shots 验证 observable prediction 与 LER 语义一致
- [ ] 4.2 完成 P1.5 PyMatching DEM 基线和 P1.6 LER benchmark，验证输出包含 failures、shots、LER、置信区间和 protocol ID
- [ ] 4.3 完成 P3.2 模型、trainer、evaluator 和 exporter 的统一 registry，验证未注册实现、输入能力或 checkpoint 身份不匹配均 fail loudly
- [ ] 4.4 完成 P3.3 可重建时空张量、图和 side-information mask，验证所有派生视图绑定 raw dataset hash
- [ ] 4.5 完成 P3.5 配置驱动训练器、best/last checkpoint 和完整状态保存，验证 epochs、batch、optimizer、scheduler、AMP 和 early stopping 配置真实生效
- [ ] 4.6 实现并测试 AI reweighting 或局部 predecoder + PyMatching 的第一条电路级神经纵切面，验证 residual syndrome、diagnostics 和纯 PyMatching 对照使用相同测试 shots
- [ ] 4.7 分别实现 detector-graph GNN 与 recurrent Transformer 比较模型的最小可运行版本，用 d=3 smoke 和输入能力测试验证接口一致
- [ ] 4.8 完成 P3.7 统一对照表，验证常数、density-only、非学习估计器、MWPM、RBM、hybrid 和实际可用神经模型不会跨 dataset 或 protocol 排名

## 5. 数据高效训练研究

- [ ] 5.1 冻结固定预算的数据高效训练 protocol，明确码距、噪声、轮数、候选量、训练量、主要指标、failure budget、停止规则和多 seed 方案，并验证 protocol 变更产生新 ID
- [ ] 5.2 实现在线候选池和可恢复困难样本 buffer，验证每个入选样本保存评分器、分数、选择概率、来源桶和入选原因
- [ ] 5.3 实现损失、置信度和 RBM 能量三个可替换评分器，使用单元测试验证排序确定性并通过消融分别报告贡献
- [ ] 5.4 实现困难样本与原分布的混合采样及可选 importance weighting，验证实际采样分布、权重和未修正对照写入 run
- [ ] 5.5 实现跨码距、物理错误率和轮数的课程状态机，验证 checkpoint 恢复后调度决策和各域消费样本数连续
- [ ] 5.6 在固定独立测试集和未见噪声域上运行均匀、focal/reweight、固定课程和自适应方法的多 seed 比较，验证报告同时包含 LER、样本效率、生成成本、GPU 时间和置信区间
- [ ] 5.7 当任一评估单元未达到 failure budget 时产生 evidence-insufficient 结论，使用低失败数 fixture 验证系统不会发布不稳定成功结论

## 6. 性能与原生运行时

- [ ] 6.1 实现 T_model、T_decoder、T_E2E 三套独立 benchmark 边界，验证报告包含硬件软件环境、输入、warm-up、同步、batch、p50/p95/p99、吞吐、内存和 deadline miss
- [ ] 6.2 对现有 RBM 数据生成、传输、Gibbs 解码和结果验证分别 profiling，使用时间线或 profiler 报告确认 CPU/GPU 等待发生在哪个阶段
- [ ] 6.3 优先通过批量张量、掩码停止、减少 host 同步、预取和成熟运行库优化已确认热点，并用回归测试验证 LER/输出语义保持不变
- [ ] 6.4 在明确 NVIDIA 部署目标后增加 ONNX/TensorRT adapter，验证导出模型与 PyTorch 参考输出在预登记容差内一致
- [ ] 6.5 只有 profiler 和部署目标共同证明需要原生层时才建立 CMake + Python binding 骨架，验证 wheel 安装、参考实现一致性和无 Python 内循环的目标热点
- [ ] 6.6 在选择 FPGA/ASIC 流程后记录板卡或工艺、时钟、资源与功耗来源，验证报告明确区分实测、综合、仿真和投影

## 7. AI 辅助论文与想法验证

- [ ] 7.1 定义 claim record schema，记录论文或博客来源、版本、发表状态、定位和项目解释，并用技术博客 fixture 验证不会被标记为同行评审结论
- [ ] 7.2 实现从 claim/hypothesis 到 estimand、数据域、基线、指标、预算、停止和成功判据的 protocol 草案生成器，验证缺失关键定义时正式执行被阻止
- [ ] 7.3 实现不可变 protocol 版本与事后分析标记，验证查看结果后修改主要指标会生成新版本而不覆盖原协议
- [ ] 7.4 实现受 CPU/GPU、时间、数据和存储预算约束的执行器，使用超时 fixture 验证预算耗尽产生 partial 或 evidence-insufficient
- [ ] 7.5 建立 protocol→run→dataset→checkpoint→prediction→metric→report 证据图，验证报告中的每个主要结论都能追溯并重算
- [ ] 7.6 实现证据等级和限制模板，验证 toy、模拟、公开硬件数据、真实部署和厂商工程来源使用不同且正确的结论措辞
- [ ] 7.7 导出自动验证的最小复现包，在仓库外干净环境校验哈希并重算 scope 声明的结果

## 8. 工程门禁与规格验收

- [ ] 8.1 将 OpenSpec scenario ID 接入 unit、integration、regression、smoke 和 benchmark 测试元数据，验证追踪矩阵能够检测无测试覆盖的强制场景
- [ ] 8.2 完成 P4.1–P4.5 的 wheel/CLI、lint、typing、测试分层、coverage、pre-commit 和 CI，验证 fresh environment 执行真实后端 smoke
- [ ] 8.3 为每个 gate 执行 capability 状态更新、文档同步和严格 OpenSpec 验证，验证状态只在对应测试和 exit criteria 通过后从 available 升为 validated
- [ ] 8.4 运行完整 smoke、恢复、数据完整性、统一 LER、样本效率和性能报告验收，并验证所有未实施后期能力仍显式失败
