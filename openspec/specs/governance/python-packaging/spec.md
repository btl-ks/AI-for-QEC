# governance/python-packaging Specification

## Purpose
建立可验证的 Python 分发边界，使开发检出与构建产物通过一致的 `ai_qec` 导入名称暴露公共 Notebook API，并确保 namespace package 发现范围、wheel 内容和导入副作用均可审计。

## Requirements

### Requirement: 包目录采用隐式 namespace
项目 SHALL 使用隐式 namespace package 发现根目录 `ai_qec` 的所有子包，并且 `ai_qec` 与测试树中 MUST NOT 存在 `__init__.py`；模块发现 MUST NOT 依赖包初始化文件或其副作用。

#### Scenario: 发现嵌套模块
- **WHEN** wheel 在仓库外的干净环境中安装
- **THEN** 顶层与嵌套叶子模块 SHALL 可按其完整模块路径导入，且构建产物中 SHALL 不包含 `__init__.py`

#### Scenario: 检查源码树
- **WHEN** 维护者扫描 `ai_qec` 与 `tests` 目录
- **THEN** 扫描结果 SHALL 为零个 `__init__.py` 文件

### Requirement: 公共 Notebook API 保持兼容
已安装分发包 SHALL 继续支持 `import ai_qec.notebook_api as qec`，并 SHALL 保持变更前由该模块声明的公共导出集合；导入该 facade MUST NOT 因 namespace package 迁移而提前加载可选 runtime 框架。

#### Scenario: 从仓库外导入公共 facade
- **WHEN** 调用方在仓库外的干净进程中仅安装构建出的 wheel 并导入 `ai_qec.notebook_api`
- **THEN** 导入 SHALL 成功、既有公共名称 SHALL 可访问，且未显式构造 runtime 时 SHALL NOT 加载 torch、stim、pymatching、qiskit 或 cudaq

### Requirement: 内部依赖使用明确模块路径
公共 facade 与实现模块 SHALL 从定义目标符号的明确叶子模块导入，不得依赖已删除的 `__init__.py` 聚合导出。对已移除的非公共包级便捷导入，项目 MUST NOT 通过运行时 hook 或静默兼容层伪造旧行为。

#### Scenario: 导入实现模块
- **WHEN** 测试导入训练、数据、评估和实验生命周期实现模块
- **THEN** 所有依赖 SHALL 解析到明确叶子模块，且导入过程 SHALL NOT 要求执行任何 `__init__.py`

### Requirement: 可安装源码使用受限根目录布局
项目 SHALL 将 `ai_qec` 的可安装 Python 源码置于仓库根目录，并 SHALL 通过项目构建元数据只发现 `ai_qec` namespace 及其子包；构建产物 MUST NOT 收录 `tests`、`docs`、`openspec` 或其他非分发目录作为 Python package。

#### Scenario: 从根目录源码构建分发包
- **WHEN** 维护者使用声明的构建后端从干净检出构建 wheel
- **THEN** wheel SHALL 包含根目录 `ai_qec` 下声明的模块，并且 SHALL NOT 包含非 `ai_qec` namespace 的仓库目录

#### Scenario: 未安装时从仓库根目录导入
- **WHEN** 调用方以仓库根目录为当前目录且尚未安装该项目
- **THEN** Python SHALL 能从根目录源码树导入 `ai_qec.notebook_api`
