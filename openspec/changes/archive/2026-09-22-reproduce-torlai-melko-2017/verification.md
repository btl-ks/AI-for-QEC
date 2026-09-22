# Verification

## Scope

本验证证明：本地单进程 runtime 满足 `NotebookPlatform`/`NotebookExperiment`/`NotebookRun`，`paper/srcs/ai_for_qec_workflow.ipynb` 能以标准编排端到端复现 Torlai & Melko (2017) 的 toric code 相位翻转实验，且数据集复用、Attempt 恢复、Accuracy Gate 与 Training/Execution 分离具有可测试的行为证据。它不证明 CUDA-Q、Qiskit、circuit-level 噪声、多 GPU、AMP、Gate B 或正式性能研究已经实现，也不证明与论文逐点数值一致。

环境：WSL2 Linux，NVIDIA GeForce RTX 4060 Laptop GPU（CUDA 13.0），conda 环境 `quantum`（Python 3.12.14、torch 2.14.0+cu130、stim 1.16.0、pymatching 2.4.0、numpy 2.5.3、matplotlib 3.11.2）；无运行时依赖的基础环境为 Python 3.13.12。

## Evidence

### 单元与集成测试

命令：

```bash
# 运行时环境
/home/zephy/miniconda3/envs/quantum/bin/python -m unittest discover -s tests -v
# 无依赖的基础环境（README 命令）
python3 -m unittest discover -s tests -v
python3 -m compileall -q ai_qec tests
```

结果：`quantum` 环境 73 个测试全部通过（4.8 s）；基础环境 73 个测试通过，其中 28 个需要运行时依赖的测试被跳过；compileall 通过。关键行为证据：

- `test_toric_code`：顶点校验度数 4、量子比特度数 2；校验与逻辑算符都与 plaquette 对易；可收缩环为 h0，两个非收缩环分别翻转两个逻辑位。
- `test_stim_generator`：Stim 样本逐个满足 `S = H e`、逻辑位 = `L e`；平均错误率 ≈ p；同种子可重现、split 独立；版本不符、非 X 基、多轮和非 exact 语义在采样前被拒绝；非相位翻转噪声编译为 `unsupported`，p > 0.5 报配置错误。
- `test_dataset_store`：执行/训练配置变化不改变 DatasetKey；未命中提交后命中复用同一 Artifact；分片被修改后缓存命中与读取均以 `DatasetIntegrityError` 失败；语义校验失败时 registry 与最终目录都不产生。
- `test_rbm_training`：自由能与对隐藏层穷举求和一致（5 位小数），条件概率与联合分布一致，模型 payload 往返逐位一致，CD-k 更新降低数据自由能。
- `test_decoders`：PyMatching 纠正全部单比特错误；RBM 接受的恢复链复现 syndrome；预算耗尽的样本状态为 `timed-out`、预测为 `-1` 并列入 `failed_sample_ids`；宽度不符返回 `unsupported` 且无预测；执行计划一次报告 distributed、mixed precision、compile、GPU 数与 CUDA 不可用。
- `test_pipeline_and_gate`：CPU 与 CUDA 上 H2D transfer 保留 sample id、dataset id 与 context，layout/TransferEvidence 记录实际 device 与 residency，不报告 host staging；Gate 在区间上界 ≤ 容差时 PASS、否则 FAIL 并写出证据文件；未知规则或未评估的 baseline 抛出 `AccuracyGateError` 且不写任何证据。
- `test_lifecycle`：终态 Attempt 不可再迁移；恢复 Attempt 引用终态源；已结束进程的 owner 判定为非存活；Artifact 修改后校验失败。
- `test_registry`：子进程中导入 facade 后所有 Registry 为空；显式加载后只出现本 change 实现的 11 个 key，`qiskit-circuit` 与 GPU→GPU pipeline 未登记。
- `test_technology_profile`：子进程中导入 facade 不加载 qiskit、stim、cudaq、torch、pymatching。
- `tests/integration/test_local_runtime.py`（CPU、d=3 小配置）：
  - 完整顺序成功；两个 decoder 的 sample-id 摘要相同；逻辑类计数加超时数等于 shots。
  - 第二次执行创建 `attempt-0002`，`recovery_from = attempt-0001 (completed)`，数据集缓存命中，训练与科学评估复用源 Attempt，结果对象相等。
  - 在第 2 个 epoch 后注入 `KeyboardInterrupt`：训练 Stage 与 Attempt 为 `interrupted`；恢复 Attempt 从 epoch 2 继续、只训练 2 个 epoch，最终权重与一次性训练 4 个 epoch **逐位相同**。
  - 篡改 `final.pt` 后不复用训练输出，改从校验通过的 epoch-3 恢复 checkpoint 继续并写出新的 ModelCheckpoint。
  - `unresolved`、未登记的 `qiskit-circuit`、错误 Stim 版本、不支持的噪声、distributed、未知配置键、Gate baseline 不在评估列表、CUDA 不可用：全部在创建 `runs/` 与 `datasets/` 之前失败。
  - 同一进程重复 `start_or_recover()` 把前一个 RUNNING Attempt 标记为 `interrupted`；owner 为其他存活进程时抛出 `AttemptInProgressError`。
  - Gate FAIL 时不执行性能评估；超时样本按 `count-as-failure` 计入失败；`invalid_sample_policy = fail` 时评估报错且 Attempt 为 `failed`，之后不可继续调用。

### 论文 Notebook（paper profile）

命令（`quantum` 环境，cwd 为 `paper/srcs`，nbclient 执行全部 cells）：

```python
NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "paper/srcs"}}).execute()
```

首次执行 22 个网格点共 2111 s，无错误。每个点是独立 Experiment；test split 为 10 000 shots，区间为 Wilson 95%，timeouts 为 RBM 在 20 000 步内未找到兼容链的 shots（计为失败）：

```text
 L     p |               RBM P_fail timeouts |              MWPM P_fail | gate
 4  0.05 |  0.0754 [0.0704, 0.0807]        0 |  0.0763 [0.0713, 0.0817] | PASS
 4  0.06 |  0.1167 [0.1106, 0.1231]        1 |  0.1168 [0.1107, 0.1232] | PASS
 4  0.07 |  0.1597 [0.1527, 0.1670]        0 |  0.1561 [0.1491, 0.1633] | PASS
 4  0.08 |  0.2044 [0.1966, 0.2124]        0 |  0.1960 [0.1883, 0.2039] | PASS
 4  0.09 |  0.2450 [0.2367, 0.2535]        0 |  0.2366 [0.2284, 0.2450] | PASS
 4  0.10 |  0.2953 [0.2864, 0.3043]        0 |  0.2820 [0.2733, 0.2909] | FAIL
 4  0.11 |  0.3414 [0.3322, 0.3508]        0 |  0.3314 [0.3222, 0.3407] | PASS
 4  0.12 |  0.4005 [0.3909, 0.4101]        0 |  0.3773 [0.3678, 0.3868] | FAIL
 4  0.13 |  0.4406 [0.4309, 0.4504]        0 |  0.4121 [0.4025, 0.4218] | FAIL
 4  0.14 |  0.4833 [0.4735, 0.4931]        0 |  0.4584 [0.4487, 0.4682] | FAIL
 4  0.15 |  0.5266 [0.5168, 0.5364]        0 |  0.5022 [0.4924, 0.5120] | FAIL
 6  0.05 |  0.0431 [0.0393, 0.0473]       88 |  0.0384 [0.0348, 0.0423] | PASS
 6  0.06 |  0.0712 [0.0663, 0.0764]       84 |  0.0715 [0.0666, 0.0767] | PASS
 6  0.07 |  0.1077 [0.1018, 0.1139]       76 |  0.1076 [0.1017, 0.1138] | PASS
 6  0.08 |  0.1611 [0.1540, 0.1684]       66 |  0.1611 [0.1540, 0.1684] | PASS
 6  0.09 |  0.2151 [0.2072, 0.2233]       59 |  0.2146 [0.2067, 0.2228] | PASS
 6  0.10 |  0.2759 [0.2672, 0.2847]       75 |  0.2731 [0.2645, 0.2819] | PASS
 6  0.11 |  0.3401 [0.3309, 0.3494]       81 |  0.3363 [0.3271, 0.3456] | PASS
 6  0.12 |  0.4048 [0.3952, 0.4145]      135 |  0.3891 [0.3796, 0.3987] | FAIL
 6  0.13 |  0.4708 [0.4610, 0.4806]      299 |  0.4437 [0.4340, 0.4535] | FAIL
 6  0.14 |  0.5346 [0.5248, 0.5444]      407 |  0.5057 [0.4959, 0.5155] | FAIL
 6  0.15 |  0.5760 [0.5663, 0.5857]      876 |  0.5435 [0.5337, 0.5532] | FAIL
```

与论文的对照：

- 图 3：阈值（p ≈ 0.109）以下 RBM 与 MWPM 的失败率几乎重合（L=6、p=0.08 两者均为 0.1611），阈值以上 RBM 系统性更差；L=4 与 L=6 的 MWPM 曲线在 p ≈ 0.11 附近相交。与论文结论同向。
- 图 4（L=4，RBM 接受的恢复）：平凡类 h0 占比在 p = 0.05、0.08、0.12、0.15 时分别为 92%、80%、60%、47%，非平凡类随 p 上升且 h1、h2 大致对称，与论文直方图一致。
- Gate（容差 0.02，配对 Newcombe 95%）：13 个点 PASS、9 个点 FAIL；阈值以上全部 FAIL。L=4 在 p=0.10 FAIL、p=0.11 PASS，说明阈值附近的判定受统计分辨率影响（Notebook 结论节已说明）。
- 参考点 L=4、p=0.08 的 Gate 证据：差异 +0.0084，区间 [+0.0010, +0.0158]，PASS；吞吐量报告写入该 Attempt 的 `performance/throughput.json`。
- L=6 在 p ≤ 0.11 时仍有约 0.6–0.9% 的 shots 超时，p ≥ 0.12 时超时迅速增加（0.15 时 8.8%），对应论文所述“在合理截止时间内找到兼容链”的瓶颈。

复跑：随后再执行两次 Run All，分别用时 12 s 与 9 s。22 个点均创建新 Attempt（`attempt-0002`、`attempt-0003`），日志显示 22 次训练复用、22 次评估复用、0 个新数据集；`attempt.json` 记录 `recovery_from` 指向上一个 completed Attempt。最终一次执行的 Notebook 含图 3、图 4 与 3 张单点图，无错误输出。

### Package 与依赖

命令：

```bash
uv build --out-dir <scratch>/dist
uv lock && uv lock --check
```

结果：sdist 与 wheel 构建成功；wheel 含 82 个 Python 文件与 `runtime` extra（numpy、torch、stim==1.16.0、pymatching、matplotlib）。在仓库外无依赖的隔离 venv 中安装 wheel：`ai_qec.notebook_api` 的 117 个公共名称均可导入，未加载任何选定框架；构造 `LocalNotebookPlatform` 以 `ModuleNotFoundError: torch` 明确失败。`uv.lock` 解析 48 个包并通过 `--check`。

### 治理文件

`openspec/capabilities.yaml` 与 `openspec/technology-catalog.yaml` 分别通过各自 JSON Schema 校验（jsonschema 4.26.0）。

### OpenSpec strict validation

命令：

```bash
openspec validate reproduce-torlai-melko-2017 --strict --no-interactive
openspec validate --all --strict --no-interactive
```

结果（归档前）：`reproduce-torlai-melko-2017` 有效；全量验证 10 项（1 个 active change 与 9 个正式 specs）全部通过。

## Capability Decision

- `qec/dataset-pipeline`、`qec/decoder-workbench`、`research/training-contracts`、`research/scientific-evaluation`、`research/experiment-lifecycle`、`research/paper-reproduction`：由 `planned/contracts-only` 升级为 `available/partial`。实现、测试与本文件证据一致；仍为 partial，因为只覆盖 toric code-capacity、单设备与单篇论文，CUDA-Q、GPU→GPU、Qiskit、Gate B 与正式性能研究未实现。
- Technology catalog：`pymatching-cpu-decoder`、`pytorch-dataloader-h2d`、`bit-packed-cpu-buffer` 为 `implemented`；`stim-cpu`、`stim-noise`、`stim-syndrome-cpu`、`pytorch-cuda-trainer`、`pytorch-gpu-decoder`、`pytorch-tensor`、`numpy-array`、`pytorch-cuda-tensor` 为 `partial`；Qiskit、CUDA-Q 与 DLPack 保持 `contracts-only`。
