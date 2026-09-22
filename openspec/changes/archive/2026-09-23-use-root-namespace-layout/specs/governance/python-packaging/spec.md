# Spec Delta

## ADDED Requirements

### Requirement: 可安装源码使用受限根目录布局
项目 SHALL 将 `ai_qec` 的可安装 Python 源码置于仓库根目录，并 SHALL 通过项目构建元数据只发现 `ai_qec` namespace 及其子包；构建产物 MUST NOT 收录 `tests`、`docs`、`openspec` 或其他非分发目录作为 Python package。

#### Scenario: 从根目录源码构建分发包
- **WHEN** 维护者使用声明的构建后端从干净检出构建 wheel
- **THEN** wheel SHALL 包含根目录 `ai_qec` 下声明的模块，并且 SHALL NOT 包含非 `ai_qec` namespace 的仓库目录

#### Scenario: 未安装时从仓库根目录导入
- **WHEN** 调用方以仓库根目录为当前目录且尚未安装该项目
- **THEN** Python SHALL 能从根目录源码树导入 `ai_qec.notebook_api`

## MODIFIED Requirements

### Requirement: 包目录采用隐式 namespace
项目 SHALL 使用隐式 namespace package 发现根目录 `ai_qec` 的所有子包，并且 `ai_qec` 与测试树中 MUST NOT 存在 `__init__.py`；模块发现 MUST NOT 依赖包初始化文件或其副作用。

#### Scenario: 发现嵌套模块
- **WHEN** wheel 在仓库外的干净环境中安装
- **THEN** 顶层与嵌套叶子模块 SHALL 可按其完整模块路径导入，且构建产物中 SHALL 不包含 `__init__.py`

#### Scenario: 检查源码树
- **WHEN** 维护者扫描 `ai_qec` 与 `tests` 目录
- **THEN** 扫描结果 SHALL 为零个 `__init__.py` 文件

## REMOVED Requirements

### Requirement: 可安装源码使用 src 布局

**Reason**: 项目选择开发便利优先的仓库根目录 namespace layout，不再要求未安装源码与当前工作目录隔离。

**Migration**: 将 `src/ai_qec` 移至根目录 `ai_qec`，并将 package discovery 限定为 `ai_qec*`，以避免根目录 namespace discovery 收录非分发目录。
