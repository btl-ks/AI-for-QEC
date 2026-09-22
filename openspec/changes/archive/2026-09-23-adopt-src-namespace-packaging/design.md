# Design

## Context

见 `proposal.md`。当前 Hatchling 配置直接打包仓库根目录的 `ai_qec/`；28 个 `__init__.py` 中有多个承担聚合导出，`notebook_api.py` 和少量实现模块依赖这些聚合路径。正式公共入口是 `ai_qec.notebook_api`，而不是中间包目录的便捷导出。

## Goals / Non-Goals

**Goals:**

- 让源码、editable 安装和 wheel 安装都由同一 `src/ai_qec` 树提供。
- 让所有包目录由 PEP 420 隐式 namespace 语义发现，项目内 `__init__.py` 数量为零。
- 保持 `ai_qec.notebook_api` 的名称与导出不变，并以仓库外 wheel 导入作为兼容性证据。
- 使测试显式依赖安装后的项目，而不是仓库根目录碰巧位于 `sys.path`。

**Non-Goals:**

- 不为内部包级便捷导入新增兼容模块或导入 hook。
- 不改变领域 contract、Registry 加载时机、可选 runtime 依赖或 capability 科学状态。
- 不借此迁移重命名模块或重排领域边界。

## Decisions

### 使用 setuptools 的 src-layout namespace discovery

`pyproject.toml` 使用 `setuptools.build_meta`，将 `package-dir` 指向 `src`，并在 `tool.setuptools.packages.find` 中设置 `where = ["src"]` 与 `namespaces = true`。这是用户指定的构建模型，也直接表达 PEP 420 发现；继续使用 Hatchling 虽可支持类似布局，但不满足此次明确的 setuptools 选择。

### 删除全部初始化文件并把聚合责任集中到 notebook_api

删除 `src/ai_qec/**/__init__.py` 与 `tests/**/__init__.py`。`notebook_api.py` 改为从符号定义所在叶子模块导入，少量依赖 `ai_qec.training.executors` 聚合导出的实现也改用 `ai_qec.training.executors.protocol`。不创建替代聚合模块，因为这会保留用户希望消除的隐式包初始化接口。

顶层 `__version__` 不再由 `ai_qec.__init__` 提供；分发版本的单一来源是项目元数据，需要查询时使用 `importlib.metadata.version("ai-for-qec")`。当前公开 facade 未导出 `__version__`，因此不新增兼容 API。

### 物理路径与逻辑导入路径分离

源码物理路径统一更新为 `src/ai_qec/...`，Python 导入和 Notebook 内容仍使用 `ai_qec...`。`openspec/capabilities.yaml` 的实现证据、README 结构说明和验证命令更新为物理路径；正式 specs 中描述逻辑 API 的 `ai_qec.notebook_api` 保持不变。

### 同时验证源码树、测试套件和 wheel

静态测试检查源码/测试树不存在 `__init__.py`，并检查根目录没有重复源码树。常规单元与集成测试验证叶子导入和行为未变；构建验证生成 wheel，并在仓库外临时目录安装后导入公共 facade、检查导出和可选依赖未加载。wheel 内容检查确保没有初始化文件。

## Risks / Trade-offs

- [依赖旧包级便捷导入的外部代码会失败] → proposal 明确标记 breaking；正式入口保持不变，并在迁移说明中要求改用 `ai_qec.notebook_api` 或叶子模块。
- [开发环境仍引用旧 editable 安装] → 迁移后重新执行 editable 安装，并以模块 `__path__`、仓库外 wheel 导入和完整测试确认来源。
- [setuptools 发现遗漏无初始化文件的深层目录] → 使用 `namespaces = true`，并从 wheel 清单与深层模块导入双重验证。
- [源码移动导致文档或 capability evidence 悬空] → 全仓搜索旧物理路径，只更新当前文档与机器可读证据；归档历史证据保持不可变。
- [零初始化文件失去包级 docstring 和导出] → 文档归属移至实际模块；唯一正式聚合层保持为 `notebook_api.py`。

## Migration Plan

1. 更新构建元数据并移动完整源码树到 `src/ai_qec`，保持 Git 文件历史可追踪。
2. 将初始化文件中的必要聚合导入改写到 `notebook_api.py` 的叶子模块导入，并删除全部初始化文件。
3. 更新当前测试、README、capability evidence 与验证命令；同步锁文件。
4. 重新 editable 安装，执行 compile、测试、wheel 仓库外导入和 OpenSpec strict validation。
5. 若任一 gate 失败，回滚构建元数据与源码移动，不发布部分迁移的分发包。
