# AI for QEC 研究主题地图

> 目的：给人类研究者、AI 助手和自动化脚本一个统一的研究主题索引。  
> 当前分类采用我们讨论后的 **5 个大方向**：核心、平台、运行时、实时性、设计辅助。

---

## 总览

| ID | 大方向 | 定位 | 核心问题 |
|---|---|---|---|
| D1 | AI Decoding | **核心层** | syndrome / detector events 如何变成 correction / logical decision？ |
| D2 | Adaptive & Transferable QEC | **平台层** | noise、distance、device、hardware 变化后，已有 AI-QEC 如何适应和复用？ |
| D3 | Circuit / FTQC-Level Decoding | **运行时应用层** | 真正执行 logical gates / FTQC circuit 时如何持续正确解码？ |
| D4 | Real-Time Deployment & Feedback | **实时系统层** | decoding、通信与控制反馈是否满足实时 QEC 时延约束？ |
| D5 | AI for QEC Design | **设计辅助层** | AI 能否帮助传统 QEC 方法设计 code、encoder、decoder、protocol？ |

---

# D1. AI Decoding — 核心层

## 目标

给定 QEC 测量数据：

\[
S \rightarrow C
\]

其中：

- \(S\)：syndrome / detector events / measurement record
- \(C\)：correction、Pauli-frame update 或 logical error decision

这是整个 AI for QEC 的核心问题。

## 子课题

### D1.1 Neural / Discriminative Decoding

直接学习：

\[
f_\theta(S)\rightarrow \hat C
\]

典型模型：

- CNN
- RNN / LSTM
- GNN
- Transformer

主要研究：

- logical error rate
- threshold
- decoding accuracy
- scaling with code distance
- scaling with QEC rounds

### D1.2 Generative / Probabilistic Decoding

不只输出一个 correction，而是学习：

\[
P(E\mid S)
\]

或其他 posterior / generative representation。

关键词：

- autoregressive decoder
- diffusion decoder
- posterior inference
- uncertainty / confidence

### D1.3 Hybrid AI + Classical Decoder

AI 不一定完全替代传统 decoder。

例如：

\[
S \rightarrow AI\ predecoder \rightarrow S_{residual}
\rightarrow MWPM / UnionFind
\]

研究：

- AI pre-decoder
- learned graph weights
- neural-assisted matching
- local AI + global classical decoder

---

# D2. Adaptive & Transferable QEC — 平台层

## 目标

现实环境不是固定的：

\[
P_{\mathrm{train}} \neq P_{\mathrm{target}}
\]

甚至：

\[
P_{\mathrm{noise}} = P_{\mathrm{noise}}(t)
\]

因此核心问题是：

> 如何让 AI-QEC 从“单一实验模型”变成可适应、可迁移、可复用的平台能力？

这是当前项目最适合作为平台核心的方向。

## 子课题

### D2.1 Noise Identification / Noise Estimation

从 syndrome statistics 中估计噪声：

\[
S \rightarrow \hat\theta_{\mathrm{noise}}
\]

例如：

- \(p_X,p_Y,p_Z\)
- measurement error
- gate error
- crosstalk
- leakage
- spatial correlation
- temporal correlation

也可以不要求显式物理参数：

\[
S\rightarrow z_{\mathrm{noise}}
\]

其中 \(z_{\mathrm{noise}}\) 是 learned noise representation。

### D2.2 Weak / Specific Error Learning

目标：

> 在强背景误差下，识别一个较弱但重要的特定错误因素。

例如：

\[
p_{\mathrm{cross}} \ll p_{\mathrm{background}}
\]

仍希望：

\[
S\rightarrow p_{\mathrm{cross}}
\]

关键问题：

- shortcut learning
- nuisance factors
- identifiability
- weak signal detection
- rare-event sampling
- matched / balanced data
- residual learning
- adversarial nuisance removal
- hard example mining
- importance sampling

### D2.3 Adaptive / Continual Decoding

当噪声随时间改变：

\[
D_t \rightarrow D_{t+1}
\]

研究：

- online adaptation
- continual learning
- concept drift
- catastrophic forgetting
- fast fine-tuning
- calibration-aware decoder updates

### D2.4 Domain Adaptation / Transfer Learning

研究已有知识如何迁移。

典型层次：

- noise A → noise B
- \(d=5\rightarrow d=7\rightarrow d=9\)
- device A → device B
- simulator → real hardware
- code A → code B
- photonic → superconducting

### D2.5 Universal / Foundation Decoder

最终目标：

\[
D(S,z_{\mathrm{noise}},z_{\mathrm{code}},z_{\mathrm{device}})
\rightarrow C
\]

希望一个模型覆盖：

- 多 noise
- 多 distance
- 多 device
- 多 code
- 甚至多 hardware platform

关键词：

- foundation decoder
- domain generalization
- universal decoder
- conditional decoder
- adapters
- hardware-invariant representation

---

# D3. Circuit / FTQC-Level Decoding — 运行时应用层

## 目标

普通 memory-decoding benchmark 中，logical qubit 往往只是重复做 QEC round。

真正 FTQC 则会执行：

- logical H / S / CNOT / T
- logical measurement
- lattice surgery
- magic-state related operations
- multi-logical-qubit circuits

因此错误会随电路传播和形成相关结构。

关键不是“runtime 临时告诉 AI 上一步是什么门”，因为静态电路可以预编译；关键是：

> decoder 必须正确处理 **compiled circuit structure + runtime stochastic measurement/frame state**。

## 子课题

### D3.1 Circuit-Aware Decoding

针对已知 circuit：

\[
D_C(S)\rightarrow C
\]

其中 circuit 结构可在 compile time 注入 decoder / detector graph。

### D3.2 Multi-Logical-Qubit Decoding

研究：

- entangling gates
- correlated logical errors
- error propagation between logical qubits

### D3.3 Dynamic / Adaptive Circuit Decoding

针对 runtime 才确定的信息：

- mid-circuit measurement
- adaptive branch
- Pauli frame
- measurement-dependent control

### D3.4 Algorithm-Level Generalization

训练于部分 circuit workload，测试：

- unseen circuits
- unseen algorithm structure
- longer logical depth
- different logical-qubit count

---

# D4. Real-Time Deployment & Feedback — 实时系统层

## 目标

这里的重点不是“怎么设计更聪明的纠错算法”，而是：

> 纠错结果是否来得及产生、传输和反馈？

建议区分三种时延：

\[
T_{\mathrm{model}}
\]

仅模型 forward。

\[
T_{\mathrm{decoder}}
=
T_{\mathrm{pre}}
+
T_{\mathrm{model}}
+
T_{\mathrm{post}}
\]

decoder 侧完整时延。

\[
T_{\mathrm{E2E}}
=
T_{\mathrm{readout}}
+
T_{\mathrm{transfer}}
+
T_{\mathrm{decoder}}
+
T_{\mathrm{feedback}}
\]

系统端到端时延。

## 子课题

### D4.1 Low-Latency Decoder

研究：

- pruning
- quantization
- sparse inference
- model compression
- low-overhead preprocessing
- tail latency（p95/p99）

### D4.2 Hardware Acceleration / Co-Design

目标平台：

- GPU
- FPGA
- ASIC
- realtime controller
- future cryogenic electronics

评价：

- latency
- throughput
- memory
- power
- model size

### D4.3 End-to-End QEC Latency

关注：

- readout
- transport
- decode
- Pauli-frame / correction update
- feedback

### D4.4 Feedback / Self-Calibration

进一步从：

\[
S\rightarrow correction
\]

扩展到：

\[
S\rightarrow hardware\ parameter\ update
\]

这部分已超出“纯 decoder”，更接近 AI-assisted quantum control。

---

# D5. AI for QEC Design — 传统方法设计辅助层

## 目标

前四个方向通常假设某个 QEC 方案已经给定。

D5 反过来研究：

> 能否让 AI 帮助搜索或优化 QEC 方案本身？

## 子课题

### D5.1 QEC Code Discovery

输入：

- hardware topology
- noise model
- physical constraints

输出：

- candidate code
- stabilizer structure
- code parameters

### D5.2 Encoder Search

搜索：

- encoding gate sequence
- low-depth encoder
- hardware-aware encoder
- robust encoder circuit

### D5.3 Syndrome Extraction / Decoder Co-Design

联合优化：

- ancilla placement
- stabilizer measurement order
- CNOT schedule
- hook-error suppression
- decoder compatibility

### D5.4 FT Protocol / Architecture Search

搜索：

- lattice surgery
- logical gate implementation
- magic-state protocol
- routing
- QEC scheduling
- decoder workload aware architecture

常见优化目标：

\[
J=
\alpha P_L
+
\beta N_{\mathrm{qubits}}
+
\gamma T_{\mathrm{runtime}}
+
\delta T_{\mathrm{decode}}
\]

---

# 五个方向之间的关系

推荐理解为：

```text
                   ┌─────────────────────────────┐
                   │ D5 AI for QEC Design        │
                   │ 设计 code/encoder/protocol  │
                   └─────────────┬───────────────┘
                                 │
                                 ▼
┌──────────────────┐    ┌────────────────────────────┐
│ D2 Platform      │───▶│ D1 AI Decoding            │
│ Adapt/Transfer   │    │ syndrome → correction      │
└──────────────────┘    └──────────────┬─────────────┘
                                      │
                                      ▼
                         ┌────────────────────────────┐
                         │ D3 Runtime / FTQC          │
                         │ 真正 circuit 中运行        │
                         └──────────────┬─────────────┘
                                      │
                                      ▼
                         ┌────────────────────────────┐
                         │ D4 Real-Time Deployment    │
                         │ 时延 / feedback / hardware │
                         └────────────────────────────┘
```

其中：

- **D1 是核心能力**
- **D2 是平台性与通用化能力**
- **D3 是真实 FTQC 运行时应用**
- **D4 是实际部署与实时性约束**
- **D5 是 AI 对传统 QEC 设计流程的反向辅助**

---

# 建议的项目标签

实验配置建议使用统一 `topic_id`：

- `D1`
- `D2`
- `D3`
- `D4`
- `D5`

子课题使用：

- `D2.2`
- `D4.1`
- `D5.3`

例如：

```yaml
topic:
  direction: D2
  subtopic: D2.2
  name: weak_crosstalk_learning
```

这样论文、实验、run、checkpoint 都可以被自动索引。
