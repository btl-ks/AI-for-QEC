# Torlai & Melko (2017) 本地复现：性能瓶颈分析

- 日期：2026-09-22
- 代码版本：`4a1d2ff`（分支 `feat/torlai-melko-reproduction`）
- 范围：数据集生成、RBM 训练、RBM Gibbs 解码（GPU 推理）、PyMatching（CPU 传统算法回测）
- 性质：分析记录；capability 的验收证据见对应 OpenSpec change。第 3 节已补充
  `add-pytorch-cuda-graph-training-step` 的真实实现结果，其余优化建议仍未实现。

## 结论摘要

1. 训练占每个新网格点耗时的 79%（L=6）到 95%（L=4）。数据集生成不足总耗时的 0.3%。
2. 所有 GPU 路径的瓶颈都是 Python 逐个发起 CUDA kernel 的开销，而不是 GPU 算力。训练步的 batch 从 100 增加到 1000，每步耗时不变；单线程 CPU 只比 GPU 慢 1.4 倍。
3. RBM Gibbs 解码的耗时由收敛最慢的链决定。L=6 的每次评估都有超时链，所以循环总是跑满 `max_steps=20000`，与 p 和样本数基本无关。
4. 在当前规模（L≤6）下，PyMatching 不是瓶颈。
5. 性能 Stage 报告的"RBM 比 PyMatching 慢 180–7096 倍"很大程度上是测量批量太小（1000 shots）造成的，不能直接当作算法之间的吞吐比较。
6. 最大的单次墙钟损失（约 7.6 小时）来自主机睡眠，而不是计算。
7. 训练步 CUDA Graph 已按平台 contract 实现。在相同 L=6、batch 100、CD-10 配置下，
   steady step 为 0.350 ms（eager 2.265 ms，快 6.47 倍）；保留逐 batch H2D 的完整
   3 epoch Training Stage 为 1.529 s（eager 7.618 s，端到端快 4.98 倍）。

## 1. 数据与方法

**运行记录。** 读取 `runs/` 下 66 个 Experiment、151 个 Attempt 的 `stages/*.json`，由 `started_at` / `finished_at` 得到各 Stage 的墙钟时间；由 `training/history.json` 得到每个 epoch 的 `seconds`（`time.perf_counter`）。`runs/` 和 `datasets/` 被 git 忽略，文中的 Experiment ID 指本机记录。

**微基准。** 使用与运行记录相同的环境，调用仓库中的 `JointErrorSyndromeRBM`、`ContrastiveDivergence`、`ToricCode`、`LocalDatasetStore` 和 `TorchHostToDevicePipeline`；每项取预热后的多次均值或最小值。基准脚本没有纳入仓库。

**环境。**

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU（8 GB） |
| CPU | Intel i7-14700HX，WSL2 可见 12 个逻辑核 |
| 平台 | Linux 6.6.87.2-microsoft-standard-WSL2 |
| 软件 | Python 3.12.14、torch 2.14.0+cu130、stim 1.16.0、pymatching 2.4.0、numpy 2.5.3 |

**论文配置**（两个码距共用：`train_samples=100000`、`batch_size=100`、CD-10、SGD lr=0.1、`burn_in=100`、`max_steps=20000`、性能测量 1000 shots × 3 次重复 + 1 次预热、`execution.cpu_count=1`）：

| | hidden units | epochs | 优化步数 |
|---|---|---|---|
| L=4 | 64 | 30 | 30,000 |
| L=6 | 128 | 40 | 40,000 |

## 2. 每个网格点的耗时分布

只统计新训练的 Attempt（不含 Stage 复用）：

| | 数据集 | 训练 | 科学评估 | 性能测量 | 可视化 | 总计 |
|---|---|---|---|---|---|---|
| L=4 | 0.1 s | 67–87 s | 0.6–4.8 s | 0.8–14 s | 0.2 s | 72–92 s |
| L=6 | 0.3 s | 87–108 s | 7.0–8.0 s | 17.6–22.2 s | 0.2 s | 113–138 s |

- L=6 的代表点 `l6-p0.10`：训练 91.0 s（78%）、性能测量 18.0 s（15%）、科学评估 7.0 s（6%）。
- 首次顺序执行 22 点网格，Attempt 墙钟合计约 35 分钟。
- Stage 复用正常：重跑的 Attempt 耗时 0.2–0.5 s。
- 数据集复用正常：命中缓存时解析和校验只需约 2.4 ms。

## 3. 训练：kernel 启动开销是主瓶颈

### 3.1 每步耗时

L=6（72 + 36 个可见单元、128 个隐藏单元），CD-10：

| 设备 | batch | 每步耗时 | 样本/s | 每个 epoch（10 万样本） |
|---|---|---|---|---|
| GPU（eager，旧原型） | 100 | 2.00 ms | 50,105 | 2.00 s |
| GPU（eager） | 1,000 | 1.98 ms | 504,072 | 0.20 s |
| GPU（eager） | 10,000 | 3.30 ms | 3,034,185 | 0.03 s |
| CPU，1 线程 | 100 | 2.81 ms | 35,568 | 2.81 s |
| CPU，1 线程 | 1,000 | 25.6 ms | 39,038 | 2.56 s |
| GPU（eager，当前平台） | 100 | 2.265 ms | 44,154 | 2.265 s |
| GPU（CUDA Graph，当前平台） | 100 | 0.350 ms | 285,883 | 0.350 s |

- 每个训练步约发起 130 个 CUDA kernel（`torch.profiler` 统计），对应的循环在 [pytorch_trainer.py:166-172](../../ai_qec/training/trainers/pytorch_trainer.py#L166-L172)。batch 放大 10 倍，每步耗时不变，说明 GPU 在等待 kernel 启动。
- 模型大小同样不影响每步耗时：L=4 的模型更小，但每个 epoch 中位耗时为 2.7 s，与 L=6 的 2.2 s 在同一量级。
- 当前平台实现用 CUDA Graph 捕获整个 loss、backward 和 optimizer update；使用 64 个内容不同、
  shape 相同的 device minibatch 测得每步从 2.265 ms 降到 0.350 ms，快 6.47 倍。

### 3.2 CUDA Graph 平台实现与端到端结果（2026-09-22）

可复现命令：

```bash
/home/zephy/miniconda3/envs/quantum/bin/python \
  openspec/changes/archive/2026-09-22-add-pytorch-cuda-graph-training-step/evidence/benchmark_cuda_graph_training_step.py \
  --steps 1000 --output /tmp/ai-qec-cuda-graph-benchmark.json
```

环境仍为本文第 1 节的 RTX 4060 Laptop GPU、PyTorch 2.14.0+cu130。微基准使用
L=6 对应的 72 个 error unit、36 个 syndrome unit、128 个 hidden unit、batch 100 和
CD-10；不是重复同一 batch，而是在 64 个内容不同的 device batch 间循环。

| 项 | 实测 |
|---|---:|
| eager 完整训练步 | 2.265 ms/step |
| executor 静态 batch D2D copy | 0.0237 ms/batch |
| 首次 capture + capture minibatch 的首次 launch | 7.494 ms |
| steady copy + replay | 0.350 ms/step |
| steady replay 的 host submit（不含 GPU 完成等待） | 0.236 ms/step |
| steady step 加速 | 6.47× |

完整 Training Stage 使用正常 `pytorch-dataloader-h2d`，每个 minibatch 仍从 CPU 逐批传到
CUDA；训练集没有常驻 GPU。两条路径均训练 100,000 个样本、3 epoch、共 3,000 个 step：

| executor | Experiment | Training Stage | capture / replay | Accuracy Gate |
|---|---|---:|---:|---|
| `pytorch-eager` | `cuda-graph-benchmark-pytorch-eager-574e818b95f2` | 7.618 s | 0 / 0 | PASS |
| `pytorch-cuda-graph` | `cuda-graph-benchmark-pytorch-cuda-graph-07974c5fde63` | 1.529 s | 1 / 2,998 | PASS |

端到端 Training Stage 加速为 4.98×，低于纯训练步 6.47×，差额来自 DataLoader/H2D、
epoch 监控和 checkpoint 等未捕获开销。两个 Experiment 复用了同一个不可变 DatasetArtifact
`ds-8a7f272efa07384a7711`，但没有复用训练或科学评估；二者分别执行了 Scientific Evaluation
与 Accuracy Gate。本次确定性配置的 ModelCheckpoint checksum 恰好相同：
`sha256:aaca87a883ae78239275b2da6fe02c016476372aaa13119bc1218234a55f0e6c`。这是一条本机验证证据，
不构成对所有模型或 CUDA/PyTorch 版本逐位一致的承诺。

### 3.3 每个 epoch 的其他开销

| 项 | 耗时 | 占 epoch（约 2.2 s）的比例 |
|---|---|---|
| DataLoader（`num_workers=0`、pinned memory、逐 batch H2D） | 169 µs/batch，每个 epoch 0.17 s | 约 8% |
| TrainingRecoveryCheckpoint 保存 + sha256 | 1.1 ms（66 KB） | 小于 0.1% |
| 重建误差监控（1 万训练样本 + 1 万验证样本） | 毫秒级 | 可忽略 |

启用 CUDA Graph 后，DataLoader/H2D 成为更显著的剩余开销。但平台实现明确继续使用现有
逐 batch H2D contract；完整训练集常驻 GPU 不属于本 change，也不作为通用平台方案。

### 3.4 并发扩展（为 `add-local-experiment-queue` task 4.1 提供数据）

同时运行 N 个独立进程，每个进程执行相同的 L=6、batch 100 训练步：

| 进程数 | GPU：单进程每步 | GPU：总吞吐 | CPU（每进程 1 线程）：单进程每步 | CPU：总吞吐 |
|---|---|---|---|---|
| 1 | 2.03 ms | 492 步/s（1.00×） | 2.95 ms | 339 步/s |
| 2 | 2.62 ms | 765 步/s（1.55×） | 2.98 ms | 670 步/s |
| 4 | 6.63 ms | 603 步/s（1.23×） | 3.33 ms | 1,200 步/s |
| 8 | 未测 | 未测 | 3.98 ms | 2,011 步/s |

- 在本机 WSL 上，同一块 GPU 的多进程扩展性很差：4 个进程的总吞吐反而低于 2 个进程，因为多个进程的 GPU 上下文只能轮流占用 GPU。
- CPU 进程接近线性扩展：8 个进程的总吞吐是单个 GPU 进程的 4.1 倍。
- 这些数字只适用于本机和当前模型规模；更换硬件或模型后需要重新测量。

## 4. RBM Gibbs 解码（GPU 推理）：尾部链决定耗时

### 4.1 终止条件

解码循环每 100 步检查一次 `found.all()`；只要还有一条链没有找到与 syndrome 相容的错误链，就一直跑到 `max_steps`（[rbm_gibbs.py:142-155](../../ai_qec/models/decoders/generative/rbm_gibbs.py#L142-L155)）。

| 点 | 超时链（每 1 万样本） | 接受步数 中位数 / p95 / 最大值 | 科学评估耗时 |
|---|---|---|---|
| L=6，p=0.05 | 88 | 103 / 1,049 / 19,106 | 7.0 s |
| L=6，p=0.10 | 75 | 288 / 4,129 / 19,935 | 7.0 s |
| L=6，p=0.15 | 876 | 1,463 / 12,831 / 19,973 | 7.0 s |
| L=4，p=0.11 | 0 | 111 / 265 / 2,016 | 0.6 s |
| L=4，p=0.06 | 1 | 101 / 209 / 13,395 | 4.8 s |

- L=6 的每个点都有超时链，所以科学评估恒定在约 7 s，与 p 无关。
- L=4 的评估耗时由最慢的那条链决定：一条超时链就足以把耗时从 0.6 s 拉到 4.8 s。

### 4.2 每个 Gibbs 步的耗时

| 设备 | 行数 | eager | CUDA Graph（每个 graph 含 100 步） | × 20000 步 |
|---|---|---|---|---|
| GPU | 1,000 | 279 µs | 53 µs | 5.6 s → 1.1 s |
| GPU | 10,000 | 313 µs | 313 µs | 6.3 s（不变） |
| CPU，1 线程 | 1,000 | 2,861 µs | 未测 | 57 s |
| CPU，1 线程 | 10,000 | 31,476 µs | 未测 | 630 s |

- **1000 行时**卡在 kernel 启动延迟上；用 CUDA Graph 后快 5.3 倍。
- **10000 行时** GPU 在做实际计算，每步约 274 µs 的 GPU 时间：3 个细长 sgemm 约占 43%，两次 Bernoulli 采样约占 24%，其余是逐元素 kernel。
- **算力大多浪费在已收敛的链上。** 以 L=6、p=0.10 为例，中位数在 288 步时收敛，但所有 1 万行都要一直计算到第 20000 步。如果每隔 N 步把已收敛的链剔除，GPU 工作量约可按"平均接受步数 / max_steps"缩小（该点约为 956 / 20000）。
- **解码必须留在 GPU。** CPU 上 1 万行、20000 步需要约 630 s。

### 4.3 性能 Stage 的读数

性能 Stage 用 1000 shots 测量，每次重复约 4.5 s，加上 1 次预热共约 18 s，比科学评估还长。

由于解码耗时约等于 `max_steps × 每步延迟`，报告中的 RBM shots/s 基本与 shots 成正比：L=6 时 1000 shots 测得 158–228 shots/s，1 万 shots 按同样耗时推算约为 1400 shots/s。因此，报告中的"RBM 比 PyMatching 慢 180–7096 倍"（L=6 为 1329–5096 倍）主要反映测量批量和尾部链，而不是两个算法在吞吐上的固有差距。例如 L=4、p=0.06 的测量子集里只要出现一条长链，比值就从约 800 倍跳到 7096 倍。

## 5. PyMatching（CPU 传统算法回测）：当前规模下不是瓶颈

`decode_batch`，均匀边权，单线程：

| L | p=0.05：µs/shot | p=0.10：µs/shot | p=0.10 时 10 万 shots 耗时 |
|---|---|---|---|
| 4 | 0.35 | 0.87 | 88 ms |
| 6 | 0.99 | 2.86 | 286 ms |
| 10 | 3.09 | 9.8 | 0.98 s |
| 16 | 7.4 | 33.8 | 3.4 s |
| 24 | 17.9 | 90.1 | 9.0 s |

- 论文网格（L≤6、1 万测试 shots）的基线解码只需 3–30 ms。
- **1000 shots 太短，不适合正式性能研究。** 性能 Stage 中 1000 shots 只耗时 0.3–3 ms，同一个点在两轮扫描之间最多相差约 23%（L=4、p=0.08）；正式研究应使用更多 shots。
- **码距增大后会变成瓶颈。** L=24、p=0.1 时单线程每个 shot 需要 90 µs；要用 10⁶ 量级的 shots 估计低 LER 时，应把 shots 分片交给多个进程。

## 6. 数据集生成与加载

每 10 万个样本，p=0.1：

| L | Stim 采样 | 校验（int64 稠密矩阵乘法） | 校验（float32 BLAS，仅供对照） | packbits + savez | sha256 | unpackbits |
|---|---|---|---|---|---|---|
| 4 | 14 ms | 38 ms | 60 ms | 7 ms | 0.3 ms | 3.5 ms |
| 6 | 38 ms | 160 ms | 138 ms | 7 ms | 0.8 ms | 3.1 ms |
| 10 | 146 ms | 877 ms | 308 ms | 7 ms | 1.9 ms | 4.5 ms |
| 16 | 485 ms | 5,039 ms | 715 ms | 10 ms | 4.8 ms | 11 ms |

- **Stim 和存储都不是瓶颈。** 数据集 Stage 实测只需 0.1 s（L=4）和 0.3 s（L=6）。
- **校验器是隐藏成本。** `code_consistency_validator` 通过 [toric.py:98-104](../../ai_qec/qec/codes/toric.py#L98-L104) 用 int64 稠密矩阵乘法计算校验子，这条路径不走 BLAS。L=6 时耗时是 Stim 的 4 倍，L=16 时是 10 倍，并按约 L⁴ 增长。而 toric code 的每个 check 只涉及 4 个 qubit，用稀疏 XOR 计算只需 O(n)。
- **校验被重复执行。** 除了提交时（[local.py:290](../../ai_qec/data/datasets/local.py#L290)），每次 `load_split` 也会重新校验（[local.py:243](../../ai_qec/data/datasets/local.py#L243)）：训练时加载 train 和 validation，科学评估和性能测量各加载一次 test。一个新 Attempt 合计约校验 25 万个样本；L=6 时约 0.4 s，目前可以忽略，码距增大后不能忽略。

## 7. 非计算墙钟损失

**主机睡眠（`l6-p0.09-347a0d4cdd09`）。** 训练 Stage 墙钟为 27,539 s，但 40 个 epoch 的 `seconds` 合计只有 102 s。按 recovery checkpoint 的时间戳，空档正好落在 epoch 37（04:39:00 JST）与 epoch 38（12:16:13 JST）之间，而 epoch 38 自身只记了约 2 s。`perf_counter` 使用 CLOCK_MONOTONIC，不计系统挂起时间，所以这段空档是笔记本整夜睡眠（WSL VM 被暂停）。

- 所有 Attempt 的 owner PID 都是 924104，说明整个网格都在 Notebook kernel 内执行，没有按 `AGENTS.md` 用 `setsid nohup` 脱离会话。
- 脱离会话可以防止编辑器重载杀掉进程，但不能防止主机睡眠。长时间运行时还需要在 Windows 电源设置里关闭睡眠。

**中断（`l6-p0.12-c8047f2dc98a`）。** attempt-0001 在 epoch 14（12:21:19 JST）后停止推进，16.6 分钟后被同一进程标记为 `interrupted`。attempt-0002 从 `recovery-epoch-0014` 继续训练了剩余 26 个 epoch，损失不到 1 个 epoch 的计算，恢复机制工作正常。

## 8. 优化建议

除第一项已通过 `add-pytorch-cuda-graph-training-step` 实现外，以下其余项均未实现。凡是会改变
运行行为的项，都必须通过 OpenSpec change 引入；凡是会改变 RNG 消耗或计算路径的项，都必须
重新通过 Accuracy Gate，并让 Stage 复用键能区分新旧实现。

| 优先级 | 措施 | 预期收益（本机，L=6） | 对科学结果的影响 |
|---|---|---|---|
| 已实现 | 训练步用 CUDA Graph；保留逐 batch H2D | 当前 3 epoch Stage 7.618 s → 1.529 s（4.98×） | 独立 Experiment、训练与科学评估；重新通过 Accuracy Gate。没有训练集常驻 GPU |
| 2 | 队列并发（`add-local-experiment-queue`） | GPU：最多 2 个 worker，1.55 倍；CPU 训练 8 个进程：约 4 倍 | GPU worker 不改变结果。CPU 训练会改变 `execution.device`，即改变 Experiment 身份和 RNG 流；解码仍必须在 GPU 上 |
| 3 | Gibbs 解码每隔 N 步剔除已收敛的链，按行数分档使用 CUDA Graph | 科学评估约 7 s → 估计 1–1.5 s（未实测） | 改变 RNG 消耗，需要重新过 Gate |
| 4 | 性能 Stage 报告"耗时–shots"曲线，或增加 shots，并单独报告尾部链（`max_steps`）的影响 | 消除误导性的吞吐比值 | 无 |
| 5 | 校验器改用稀疏 XOR；每次加载是否都要重新做语义校验，需要对照 dataset-pipeline spec 再定 | L=16 时每 10 万样本省约 4 s | 无 |
| 6 | 长时间运行前关闭主机睡眠，并用 `setsid nohup` 启动 | 避免数小时的墙钟空档 | 无 |

**不建议**把 batch 从 100 增加到 1000。在 eager 模式下它能让训练快 10 倍，但它是论文的超参数，会改变科学结果。

## 9. 局限

- 所有数字来自一台 WSL2 笔记本。WSL 的 GPU 半虚拟化会放大 kernel 启动延迟和多进程上下文切换的成本，原生 Linux 或服务器 GPU 上的比例可能不同。
- 训练和解码的微基准使用随机输入与随机初始化的 RBM，以及固定的 L=6 形状；解码的每步耗时与模型参数无关，但接受步数的分布来自真实评估记录（第 4.1 节）。
- CUDA Graph 当前数字同时覆盖微基准和真实 Training Stage，并在该确定性配置下验证了恢复、
  ModelCheckpoint checksum、Scientific Evaluation 与 Accuracy Gate；其他模型与版本仍需各自验证。
- 第 4.3 节中 1 万 shots 的 RBM 吞吐是按"耗时与 shots 基本无关"推算的，没有在性能 Stage 中实测。
